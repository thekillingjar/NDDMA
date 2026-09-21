/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "faker/davinci_model_faker.h"

#include <memory>
#include <stdexcept>

#include "graph/node.h"
#include "graph/utils/graph_utils.h"

namespace ge {
namespace {
std::vector<NodePtr> FindDirectNodesByType(const ComputeGraphPtr &graph, const std::string &node_type) {
  std::vector<NodePtr> nodes;
  for (const auto &node : graph->GetDirectNode()) {
    if (node->GetType() == node_type) {
      nodes.emplace_back(node);
    }
  }
  return nodes;
}
void SortDataNodesByIndex(std::vector<NodePtr> &data_nodes) {
  std::sort(data_nodes.begin(), data_nodes.end(), [](const NodePtr &n1, const NodePtr &n2) -> bool {
    int64_t n1_index, n2_index;
    if (!AttrUtils::GetInt(n1->GetOpDesc(), "index", n1_index) ||
        !AttrUtils::GetInt(n2->GetOpDesc(), "index", n2_index)) {
      throw std::invalid_argument("Data node does not have attr index");
    }
    return n1_index < n2_index;
  });
}
OpIndexAndIo AnchorToOpIo(const NodeIndexIO &anchor) {
  return {anchor.io_type_, static_cast<int32_t>(anchor.index_), anchor.node_ptr_->GetOpDesc()->GetId()};
}
}  // namespace
std::unique_ptr<DavinciModel> DavinciModelFaker::Build() {
  auto dm = std::make_unique<DavinciModel>(0, nullptr);

  dm->Assign(model_);
  dm->SetFeatureBaseRefreshable(is_feature_map_refreshable_);

  for (const auto &node : model_->GetGraph()->GetDirectNode()) {
    const auto &op_desc = node->GetOpDesc();
    dm->op_list_[op_desc->GetId()] = op_desc;
  }

  FakeAllocationTable(dm.get());
  SetSymbolToStubIfNeed();

  uint64_t fm_offset = 0UL;
  SetAdviseSymbolAddr(*dm, fm_offset);
  SetAdviseWsAddr(fm_offset);

  return dm;
}
void DavinciModelFaker::FakeAllocationTable(DavinciModel *dm) {
  if (input_lengths_.empty() && output_lengths_.empty()) {
    if (model_ == nullptr || model_->GetGraph() == nullptr) {
      return;
    }
    // 如果没设置过input output的lengths，那么从model中解析一个
    auto data_nodes = FindDirectNodesByType(model_->GetGraph(), "Data");
    SortDataNodesByIndex(data_nodes);
    for (const auto &data_node : data_nodes) {
      int64_t size = 0;
      TensorUtils::GetTensorSizeInBytes(data_node->GetOpDesc()->GetOutputDesc(0), size);
      input_lengths_.emplace_back(size);
    }
    auto netoutput_node = model_->GetGraph()->FindFirstNodeMatchType("NetOutput");
    if (netoutput_node == nullptr) {
      throw std::invalid_argument("NetOutput node does not exists on the graph");
    }
    for (const auto &td : netoutput_node->GetOpDesc()->GetAllInputsDesc()) {
      int64_t size = 0;
      TensorUtils::GetTensorSizeInBytes(td, size);
      output_lengths_.emplace_back(size);
    }
  }

  uint64_t fusion_offsets = 0;
  for (uint32_t i = 0UL; i < static_cast<uint32_t>(fusion_lengths_.size()); ++i) {
    dm->logical_mem_allocations_.emplace_back(
        MemAllocation{static_cast<uint32_t>(dm->logical_mem_allocations_.size()), kFmDeviceAddrBase + fusion_offsets,
                      static_cast<uint64_t>(fusion_lengths_[i]), ge::MemAllocation::Type::OUTPUT, i, kFmMemType});
    fusion_offsets += static_cast<int64_t>(fusion_lengths_[i]);
    ++dm->fm_mem_allocations_start_id_;
  }

  if (fms_lengths_.empty()) {
    fms_lengths_.emplace_back(kFmLength);
  }
  uint64_t fm_offsets = 0;
  for (uint32_t i = 0UL; i < static_cast<uint32_t>(fms_lengths_.size()); ++i) {
    dm->logical_mem_allocations_.emplace_back(
        MemAllocation{static_cast<uint32_t>(dm->logical_mem_allocations_.size()), kFmDeviceAddrBase + fm_offsets,
                      static_cast<uint64_t>(fms_lengths_[i]), ge::MemAllocation::Type::FEATURE_MAP, i, kFmMemType});
    fm_offsets += static_cast<int64_t>(fms_lengths_[i]);
    ++dm->logical_fm_mem_allocations_size_;
  }

  uint64_t model_io_offsets = 0;
  for (uint32_t i = 0UL; i < static_cast<uint32_t>(input_lengths_.size()); ++i) {
    dm->logical_mem_allocations_.emplace_back(MemAllocation{
        static_cast<uint32_t>(dm->logical_mem_allocations_.size()), kModelIoDevAddrBase + model_io_offsets,
        static_cast<uint64_t>(input_lengths_.at(i)), ge::MemAllocation::Type::INPUT, i, RT_MEMORY_HBM});
    model_io_offsets += static_cast<int64_t>(input_lengths_[i]);
  }
  for (uint32_t i = 0UL; i < static_cast<uint32_t>(output_lengths_.size()); ++i) {
    dm->logical_mem_allocations_.emplace_back(MemAllocation{
        static_cast<uint32_t>(dm->logical_mem_allocations_.size()), kModelIoDevAddrBase + model_io_offsets,
        static_cast<uint64_t>(output_lengths_.at(i)), ge::MemAllocation::Type::OUTPUT, i, RT_MEMORY_HBM});
    model_io_offsets += static_cast<int64_t>(output_lengths_[i]);
  }

  dm->logical_mem_allocations_.emplace_back(MemAllocation{static_cast<uint32_t>(dm->logical_mem_allocations_.size()),
                                                          0U, UINT64_MAX, ge::MemAllocation::Type::ABSOLUTE, 0U,
                                                          kAbsoluteMemType});
}
void DavinciModelFaker::SetSymbolToStubIfNeed() {
  if (task_info_registry_stub_ == nullptr) {
    return;
  }
  if (model_ == nullptr || model_->GetGraph() == nullptr) {
    throw std::invalid_argument("Got null graph when generate symbol");
  }

  SymbolToAnchors symbols_to_anchors;
  AnchorToSymbol anchors_to_symbol;
  auto ret = GraphUtils::GetRefMapping(model_->GetGraph(), symbols_to_anchors, anchors_to_symbol);
  if (ret != GRAPH_SUCCESS) {
    throw std::domain_error("Failed to GetRefMapping for graph");
  }
  std::unordered_map<std::string, std::vector<OpIndexAndIo>> symbols_to_opios;
  std::unordered_map<OpIndexAndIo, std::string> opios_to_symbol;
  for (const auto &symbol_to_anchors : symbols_to_anchors) {
    for (const auto &anchor : symbol_to_anchors.second) {
      auto opio = AnchorToOpIo(anchor);
      symbols_to_opios[symbol_to_anchors.first].emplace_back(opio);
      opios_to_symbol[opio] = symbol_to_anchors.first;
    }
  }
  task_info_registry_stub_->SetSymbol(std::move(symbols_to_opios), std::move(opios_to_symbol));
}
void DavinciModelFaker::SetAdviseSymbolAddr(const DavinciModel &dm, uint64_t &fm_offset) {
  if (task_info_registry_stub_ == nullptr) {
    return;
  }
  if (model_ == nullptr || model_->GetGraph() == nullptr) {
    throw std::invalid_argument("Got null graph when SetAdviseSymbolAddr");
  }

  auto &ios_to_symbol = task_info_registry_stub_->GetIOsToSymbol();
  std::unordered_map<std::string, int64_t> model_input_symbols_to_input_topo_order_index;
  auto data_nodes = FindDirectNodesByType(model_->GetGraph(), "Data");
  for (const auto &node : data_nodes) {
    OpIndexAndIo opio{kOut, 0, node->GetOpDesc()->GetId()};
    auto topo_order_index = static_cast<int64_t>(model_input_symbols_to_input_topo_order_index.size());
    model_input_symbols_to_input_topo_order_index[ios_to_symbol.at(opio)] = topo_order_index;
  }

  std::unordered_map<std::string, int32_t> model_output_symbols_to_output_index;
  auto netoutput_node = model_->GetGraph()->FindFirstNodeMatchType("NetOutput");
  for (size_t i = 0UL; i < netoutput_node->GetOpDesc()->GetInputsSize(); ++i) {
    OpIndexAndIo opio{kIn, static_cast<int32_t>(i), netoutput_node->GetOpDesc()->GetId()};
    model_output_symbols_to_output_index[ios_to_symbol.at(opio)] = opio.io_index;
  }

  auto &symbols_to_ios = task_info_registry_stub_->GetSymbolsToIOs();
  std::unordered_map<std::string, uint64_t> symbols_to_logical_addr;
  for (const auto &symbol_to_ios : symbols_to_ios) {
    auto in_iter = model_input_symbols_to_input_topo_order_index.find(symbol_to_ios.first);
    if (in_iter != model_input_symbols_to_input_topo_order_index.end()) {
      // fake的allocation表中，按顺序fusion, fm, 所有inputs、outputs
      symbols_to_logical_addr[symbol_to_ios.first] =
        dm.logical_mem_allocations_
          .at(in_iter->second + dm.fm_mem_allocations_start_id_ + dm.logical_fm_mem_allocations_size_).logical_addr;
      continue;
    }
    auto out_iter = model_output_symbols_to_output_index.find(symbol_to_ios.first);
    if (out_iter != model_output_symbols_to_output_index.end()) {
      symbols_to_logical_addr[symbol_to_ios.first] =
          dm.logical_mem_allocations_
            .at(dm.fm_mem_allocations_start_id_ + dm.logical_fm_mem_allocations_size_  + out_iter->second +
            model_input_symbols_to_input_topo_order_index.size()).logical_addr;
      continue;
    }

    symbols_to_logical_addr[symbol_to_ios.first] =
      dm.logical_mem_allocations_.at(dm.fm_mem_allocations_start_id_).logical_addr + fm_offset;
    fm_offset += 0x100;
  }
  task_info_registry_stub_->SetSymbolsToLogicalAddr(std::move(symbols_to_logical_addr));
}
void DavinciModelFaker::SetAdviseWsAddr(uint64_t &fm_offset) {
  if (model_ == nullptr || model_->GetGraph() == nullptr) {
    throw std::invalid_argument("Got null graph when SetAdviseSymbolAddr");
  }

  std::unordered_map<int64_t, std::vector<uint64_t>> node_indexes_to_ws_addrs;
  for (const auto &node : model_->GetGraph()->GetAllNodes()) {
    auto ws_sizes = node->GetOpDesc()->GetWorkspaceBytes();
    if (ws_sizes.empty()) {
      continue;
    }
    auto node_id = node->GetOpDesc()->GetId();
    for (auto ws_size : ws_sizes) {
      node_indexes_to_ws_addrs[node_id].push_back(kFmDeviceAddrBase + fm_offset);
      fm_offset += ws_size;
    }
  }

  if (node_indexes_to_ws_addrs.empty()) {
    return;
  }
  if (task_info_registry_stub_ == nullptr) {
    throw std::invalid_argument("Got null TaskInfoFactory when SetAdviseWsAddr");
  }

  task_info_registry_stub_->SetWsAddrs(std::move(node_indexes_to_ws_addrs));
}
}  // namespace ge

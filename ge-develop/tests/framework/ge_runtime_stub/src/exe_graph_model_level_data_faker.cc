/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "faker/exe_graph_model_level_data_faker.h"
#include "exe_graph/runtime/context_extend.h"
#include "faker/node_faker.h"
#include "exe_graph/lowering/bg_kernel_context_extend.h"
#include "exe_graph/lowering/exe_graph_attrs.h"
#include "core/builder/graph_executor_builder.h"
#include "core/executor/sequential/execution_data/sequential_execution_data_builder.h"

namespace gert {
ExeGraphModelLevelData ExeGraphModelLevelDataFaker::Build() {
  ExeGraphModelLevelData mld;
  mld.eq_data_anchors = FakeEquivalentDataAnchors();
  FakeGraphAsyncValue(mld.gav);
  // mld.symbols_to_value = FakeSymbolsToValue();
  mld.kernel_extend_info = GenerateKernelExtendInfo(mld.buffer);
  mld.compute_node_info = GenerateComputeNodeInfo(mld.buffer);
  return mld;
}

EquivalentDataEdges ExeGraphModelLevelDataFaker::FakeEquivalentDataAnchors() const {
  EquivalentDataEdges eq_data_anchors;
  eq_data_anchors.UpdateEquivalentEdges(exe_graph_.get());
  return eq_data_anchors;
}

void ExeGraphModelLevelDataFaker::FakeGraphAsyncValue(GraphAsyncValue &gav) const {
  SymbolsToValue symbols_to_value;
  EquivalentDataEdges eq_data_anchors;
  eq_data_anchors.UpdateEquivalentEdges(exe_graph_.get());
  auto executor_builder = GraphExecutorBuilder({nullptr, nullptr}, exe_graph_, &symbols_to_value);
  auto base_ed_builder = std::make_unique<SequentialExecutionDataBuilder>(executor_builder);
  base_ed_builder->AllocGraphAsyncValues(exe_graph_->GetAllNodes(), gav);
  //resource_guard_.ResetAnyValue(std::move(graph_async_value.values_guarder), graph_async_value.total_num);
}

std::unique_ptr<uint8_t[]> ExeGraphModelLevelDataFaker::GenerateKernelExtendInfo(std::list<std::string> &buffer) const {
  bg::BufferPool pool;
  for (const auto &node : exe_graph_->GetAllNodes()) {
    auto kernel_extend_info_holder = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(KernelExtendInfo)]);
    auto kernel_extend_info = reinterpret_cast<KernelExtendInfo *>(kernel_extend_info_holder.get());
    kernel_extend_info->SetKernelType(buffer.emplace_back(node->GetType()).c_str());
    kernel_extend_info->SetKernelName(buffer.emplace_back(node->GetName()).c_str());
    auto id = pool.AddBuf(kernel_extend_info_holder.get(), sizeof(KernelExtendInfo));
    ge::AttrUtils::SetInt(node->GetOpDescPtr(), kKernelExtendIndex, static_cast<int64_t>(id));
  }
  size_t size;
  return pool.Serialize(size);
}
std::unique_ptr<uint8_t[]> ExeGraphModelLevelDataFaker::GenerateComputeNodeInfo(std::list<std::string> &buffer) const {
  bg::BufferPool buffer_pool, compute_node_info_pool;
  auto compute_node =
      ComputeNodeFaker().NameAndType("FakeName", "FakeType").IoNum(2, 2).Attr<int64_t>("attr1", 1024).Build();
  size_t size;
  auto compute_node_info_holder = bg::CreateComputeNodeInfo(compute_node, buffer_pool, {}, size);

  auto compute_node_info = reinterpret_cast<ComputeNodeInfo *>(compute_node_info_holder.get());
  auto name = buffer_pool.GetBufById(reinterpret_cast<bg::BufferPool::BufId>(compute_node_info->GetNodeName()));
  compute_node_info->SetNodeName(buffer.emplace_back(name).c_str());
  auto node_type = buffer_pool.GetBufById(reinterpret_cast<bg::BufferPool::BufId>(compute_node_info->GetNodeType()));
  compute_node_info->SetNodeType(buffer.emplace_back(node_type).c_str());

  auto id = compute_node_info_pool.AddBuf(compute_node_info_holder.get(), size);
  for (const auto &node : exe_graph_->GetAllNodes()) {
    ge::AttrUtils::SetInt(node->GetOpDescPtr(), kComputeNodeIndex, static_cast<int64_t>(id));
  }
  return compute_node_info_pool.Serialize();
}
}  // namespace gert

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bg_memory.h"
#include "bg_variable.h"
#include "common/checker.h"
#include "common/debug/ge_log.h"
#include "exe_graph/lowering/exe_graph_attrs.h"
#include "exe_graph/lowering/frame_selector.h"
#include "graph/debug/ge_attr_define.h"
#include "graph_builder/bg_checker.h"
#include "runtime/rt.h"
#include "storage_format.h"
#include "value_holder_generator.h"
#include "exe_graph/lowering/value_holder_utils.h"

namespace gert {
namespace bg {
MemoryKernelsName kMemoryTypeToMemKernelName[kTensorPlacementEnd] = {
    {"AllocMemHbm", "FreeMemHbm", "AllocBatchHbm", "FreeBatchHbm"},
    {"AllocMemHost", "FreeMemHost", "AllocBatchHost", "FreeBatchHost"},
    {"UnSupportAlloc", "UnSupportFree", "UnSupportAlloc", "UnSupportFree"},
    {"AllocMemHbm", "FreeMemHbm", "AllocBatchHbm", "FreeBatchHbm"}};
static_assert(sizeof(kMemoryTypeToMemKernelName) / sizeof(MemoryKernelsName) == kTensorPlacementEnd,
              "The element count in kMemoryTypeToMemKernelName is missmatch with MemoryType definition");

ValueHolderPtr CalcTensorSizeFromStorage(ge::DataType dt, const ValueHolderPtr &storage_shape) {
  auto data_type = ValueHolder::CreateConst(&dt, sizeof(dt));
  return ValueHolder::CreateSingleDataOutput("CalcTensorSizeFromStorage", {data_type, storage_shape});
}
ValueHolderPtr CalcTensorSizeFromShape(ge::DataType dt, const ValueHolderPtr &shape) {
  auto data_type = ValueHolder::CreateConst(&dt, sizeof(dt));
  return ValueHolder::CreateSingleDataOutput("CalcTensorSizeFromShape", {data_type, shape});
}

ValueHolderPtr CalcOutTensorSize(const ge::NodePtr &node, int32_t out_index, const ValueHolderPtr &shape) {
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  const auto &td = op_desc->GetOutputDescPtr(out_index);
  if (td == nullptr) {
    return nullptr;
  }
  if (DiffStorageFormat(node, out_index)) {
    return CalcTensorSizeFromStorage(td->GetDataType(), shape);
  } else {
    return CalcTensorSizeFromShape(td->GetDataType(), shape);
  }
}

std::vector<ValueHolderPtr> CalcTensorSize(const ge::NodePtr &node, const std::vector<ValueHolderPtr> &output_shapes) {
  std::vector<ValueHolderPtr> output_sizes;
  for (size_t i = 0U; i < output_shapes.size(); ++i) {
    auto out_size = CalcOutTensorSize(node, static_cast<int32_t>(i), output_shapes[i]);
    CHECK_HOLDER_OK(out_size);
    output_sizes.emplace_back(std::move(out_size));
  }
  return output_sizes;
}

ge::Status CheckOutputsNumAndPlacement(TensorPlacement placement, const ge::NodePtr &node,
                                       const std::vector<ValueHolderPtr> &output_sizes) {
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  if (op_desc->GetAllOutputsDescSize() != output_sizes.size()) {
    GELOGE(ge::FAILED, "Failed to create AllocMem nodes, the output shapes num %zu not equal to the num of "
                       "node output %u, node name %s",
           output_sizes.size(), node->GetOpDesc()->GetAllOutputsDescSize(),
           node->GetName().c_str());
    return ge::FAILED;
  }
  if (placement >= kTensorPlacementEnd) {
    GELOGE(ge::FAILED, "Failed to create AllocMem node for node %s, unknown memory type %d",
           node->GetName().c_str(), placement);
    return ge::FAILED;
  }
  return ge::SUCCESS;
}

namespace {
std::string GetSharedMemoryVariable(const ge::GeTensorDesc &desc) {
  std::string var_shared_memory;
  if (!ge::AttrUtils::GetStr(desc, ge::ASSIGN_VAR_NAME, var_shared_memory)) {
    (void)ge::AttrUtils::GetStr(desc, ge::REF_VAR_SRC_VAR_NAME, var_shared_memory);
  }
  return var_shared_memory;
}

bool IsNodeHasUnfedOptionalInput(const ge::NodePtr &node) {
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  return (op_desc->GetAllInputsSize() != op_desc->GetInputsSize());
}

std::map<std::string, uint32_t> GetNodeInputName2InputIndex(const ge::NodePtr &node) {
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  if (!IsNodeHasUnfedOptionalInput(node)) {
    return op_desc->GetAllInputName();
  }

  std::map<size_t, uint32_t> index_2_fed_index;
  uint32_t fed_index = 0U;
  for (size_t i = 0U; i < op_desc->GetAllInputsSize(); ++i) {
    const auto &input_desc = op_desc->MutableInputDesc(static_cast<uint32_t>(i));
    if (input_desc == nullptr) {
      GELOGD("Node %s has unfed optional input %zu", node->GetName().c_str(), i);
    } else {
      index_2_fed_index[i] = fed_index++;
    }
  }

  auto name_2_input_index = op_desc->GetAllInputName();
  for (auto iter = name_2_input_index.begin(); iter != name_2_input_index.end();) {
    auto index_and_fed_index = index_2_fed_index.find(iter->second);
    if (index_and_fed_index == index_2_fed_index.end()) {
      (void)name_2_input_index.erase(iter++);
    } else {
      GELOGD("Node %s meta input '%s' %u update fed index %u", node->GetNamePtr(), iter->first.c_str(),
             iter->second, index_and_fed_index->second);
      (iter++)->second = index_and_fed_index->second;
    }
  }
  return name_2_input_index;
}
}  // namespace
ge::Status GetNodeRefMap(const ge::NodePtr &node, std::map<size_t, size_t> &ref_map) {
  GE_ASSERT_NOTNULL(node);
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);

  bool has_ref_outputs = false;
  (void)ge::AttrUtils::GetBool(op_desc, ge::ATTR_NAME_REFERENCE, has_ref_outputs);
  if (!has_ref_outputs) {
    return ge::SUCCESS;
  }

  auto input_name_2_fed_index = GetNodeInputName2InputIndex(node);
  const auto &input_name = op_desc->GetAllInputName();
  for (const auto &output : op_desc->GetAllOutputName()) {
    for (const auto &input : input_name) {
      if (input.first == output.first) {
        auto input_name_and_fed_index = input_name_2_fed_index.find(input.first);
        GE_ASSERT(input_name_and_fed_index != input_name_2_fed_index.end(),
                  "Node %s output %u '%s' ref from unfed input", node->GetNamePtr(), output.second,
                  output.first.c_str());
        GELOGD("Node %s output %u '%s' reuse memory with fed input %u", node->GetNamePtr(), output.second,
               output.first.c_str(), input_name_and_fed_index->second);
        ref_map[output.second] = input_name_and_fed_index->second;
      }
    }
  }

  return ge::SUCCESS;
}

std::vector<DevMemValueHolderPtr> AllocOutputMemory(TensorPlacement placement, const ge::NodePtr &node,
                                                    const std::vector<ValueHolderPtr> &output_sizes,
                                                    const std::vector<DevMemValueHolderPtr> &input_addrs,
                                                    LoweringGlobalData &global_data) {
  GE_ASSERT_NOTNULL(node);
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  const auto ret = CheckOutputsNumAndPlacement(placement, node, output_sizes);
  if (ret != ge::SUCCESS) {
    return CreateDevMemErrorValueHolders(op_desc->GetStreamId(), output_sizes.size(),
                                         "Check outputs num and placement failed, node name %s", node->GetNamePtr());
  }
  std::map<size_t, size_t> ref_map;
  if (GetNodeRefMap(node, ref_map) != ge::SUCCESS) {
    return CreateDevMemErrorValueHolders(op_desc->GetStreamId(), output_sizes.size(), "Node %s get ref map failed",
                                         node->GetNamePtr());
  }

  std::vector<bg::DevMemValueHolderPtr> memories;
  for (size_t i = 0U; i < output_sizes.size(); i++) {
    auto desc = op_desc->GetOutputDescPtr(i);
    GE_ASSERT_NOTNULL(desc);
    auto var_shared_memory = GetSharedMemoryVariable(*desc);
    auto iter = ref_map.find(i);
    // 如果一个输出既ref了输入，又ref了变量，这俩地址应该一样，此时优先取输入地址
    // 在单流场景下这俩地址在图上的表达是等价的
    // 在多流场景下，输入地址可能已经跨流访问，从图上表达不等价，此时优先取输入地址
    if (iter != ref_map.end()) {
      auto ref_input_index = iter->second;
      if (ref_input_index >= input_addrs.size()) {
        GELOGW(
            "Node %s output %zu ref from input %zu exceed input addrs num %zu, If are you call bg::AllocOutputMemory "
            "for ref node, replace it with version with input addrs and pass the input addrs properly",
            node->GetNamePtr(), i, ref_input_index, input_addrs.size());
        // todo: Remove this and return errors after engines bugfix
        memories.push_back(AllocMemories(placement, {output_sizes[i]}, global_data, op_desc->GetStreamId())[0]);
      } else {
        memories.push_back(input_addrs[ref_input_index]);
      }
    } else if (!var_shared_memory.empty()) {
      if ((!TensorPlacementUtils::IsOnDeviceHbm(placement))
          && (!TensorPlacementUtils::IsOnHostNotFollowing(placement))) {
        GELOGE(ge::FAILED, "Node %s output %zu ref from variable %s but requested placed on %d",
               node->GetName().c_str(), i, var_shared_memory.c_str(), placement);
        return {};
      }
      memories.push_back(bg::GetVariableAddr(var_shared_memory, global_data, op_desc->GetStreamId()));
    } else {
      memories.push_back(AllocMemories(placement, {output_sizes[i]}, global_data, op_desc->GetStreamId())[0]);
    }
  }

  return memories;
}

std::vector<DevMemValueHolderPtr> AllocOutputMemory(TensorPlacement placement, const ge::NodePtr &node,
                                                    const std::vector<ValueHolderPtr> &output_sizes,
                                                    LoweringGlobalData &global_data) {
  return AllocOutputMemory(placement, node, output_sizes, {}, global_data);
}

// todo 使用统一的guarder机制，需要同步修改每个node converter，去掉释放workspace内存的流程
DevMemValueHolderPtr AllocWorkspaceMem(TensorPlacement placement, const ValueHolderPtr &workspaces_size,
                                       LoweringGlobalData &global_data) {
  GE_ASSERT_TRUE(placement < kTensorPlacementEnd, "Failed to create AllocMem for workspace, unknown memory type %d",
                 placement);
  auto allocator_holder = global_data.GetOrCreateAllocator({placement, AllocatorUsage::kAllocNodeWorkspace});
  auto mem = DevMemValueHolder::CreateSingleDataOutput(kMemoryTypeToMemKernelName[placement].alloc_batch,
                                                       {allocator_holder, workspaces_size}, kMainStream);
  GE_ASSERT_NOTNULL(mem);
  mem->SetPlacement(placement);
  return mem;
}
DevMemValueHolderPtr FreeWorkspaceMem(TensorPlacement placement, const DevMemValueHolderPtr &addr) {
  GE_ASSERT_TRUE(placement < kTensorPlacementEnd, "Failed to create AllocMem for workspace, unknown memory type %d",
                 placement);
  auto free_mem = ValueHolder::CreateVoid<bg::DevMemValueHolder>(kMemoryTypeToMemKernelName[placement].free_batch,
                                                                 {addr}, bg::kMainStream);
  GE_ASSERT_NOTNULL(free_mem);
  GE_ASSERT_TRUE(ge::AttrUtils::SetInt(ValueHolderUtils::GetNodeOpDescBarePtr(free_mem), kReleaseResourceIndex, 0));
  return free_mem;
}

DevMemValueHolderPtr AllocMem(TensorPlacement placement, const ValueHolderPtr &size_holder,
                              LoweringGlobalData &global_data, const int64_t stream_id) {
  auto allocator_holder = global_data.GetOrCreateAllocator({placement, AllocatorUsage::kAllocNodeOutput});
  auto out_addr = DevMemValueHolder::CreateSingleDataOutput(kMemoryTypeToMemKernelName[placement].alloc_name,
                                                            {allocator_holder, size_holder}, stream_id);
  GE_ASSERT_NOTNULL(out_addr);
  out_addr->SetPlacement(placement);
  ValueHolder::CreateVoidGuarder("FreeMemory", out_addr, {});
  return out_addr;
}

ValueHolderPtr AllocFixedFeatureMemory(const ValueHolderPtr &session_id_holder,
                                       const TensorPlacement placement,
                                       const ValueHolderPtr &size_holder,
                                       LoweringGlobalData &global_data) {
  auto addr_vec = FrameSelector::OnInitRoot(
      [&session_id_holder, &size_holder, &placement, &global_data]() -> std::vector<ValueHolderPtr> {
        const auto placement_holder = bg::ValueHolder::CreateConst(&placement, sizeof(placement));
        auto allocator_holder = bg::ValueHolder::CreateSingleDataOutput("GetUserAllocatorOrFixedBaseAllocator",
            {placement_holder, session_id_holder, global_data.GetStreamById(kDefaultMainStreamId)});
        GE_ASSERT_NOTNULL(allocator_holder);
        AllocatorDesc allocator_desc = {placement, AllocatorUsage::kAllocNodeOutput};
        auto init_l2_allocator = global_data.GetOrCreateAllocator(allocator_desc);
        auto out_addr = ValueHolder::CreateSingleDataOutput("AllocFixedFeatureMemory",
                                                            {allocator_holder, size_holder, init_l2_allocator});
        return {out_addr};
      });
  GE_ASSERT_TRUE(!addr_vec.empty());
  auto out_addr = addr_vec.front();
  GE_ASSERT_NOTNULL(out_addr);
  out_addr->SetPlacement(placement);
  FrameSelector::OnDeInitRoot([&out_addr]() -> std::vector<ValueHolderPtr> {
    return {ValueHolder::CreateVoidGuarder("FreeFixedFeatureMemory", out_addr, {})};
  });
  return out_addr;
}

DevMemValueHolderPtr AllocMemWithoutGuarder(TensorPlacement placement, const ValueHolderPtr &size_holder,
                                            LoweringGlobalData &global_data, const int64_t stream_id) {
  auto allocator_holder = global_data.GetOrCreateAllocator({placement, AllocatorUsage::kAllocNodeOutput});
  auto out_addr = DevMemValueHolder::CreateSingleDataOutput(kMemoryTypeToMemKernelName[placement].alloc_name,
                                                            {allocator_holder, size_holder}, stream_id);
  GE_ASSERT_NOTNULL(out_addr);
  out_addr->SetPlacement(placement);
  return out_addr;
}

std::vector<DevMemValueHolderPtr> AllocMemories(TensorPlacement placement, const std::vector<ValueHolderPtr> &sizes,
                                                LoweringGlobalData &global_data, const int64_t stream_id) {
  auto allocator_holder = global_data.GetOrCreateAllocator({placement, AllocatorUsage::kAllocNodeOutput});
  std::vector<DevMemValueHolderPtr> output_addrs;
  for (size_t i = 0U; i < sizes.size(); ++i) {
    CHECK_DEVMEMHOLDER_OK(sizes[i]);
    auto out_addr = DevMemValueHolder::CreateSingleDataOutput(
        TensorPlacementUtils::IsOnDevice(placement) ? "AllocMemHbm" : "AllocMemHost",
        {allocator_holder, sizes[i]}, stream_id);
    CHECK_DEVMEMHOLDER_OK(out_addr);
    out_addr->SetPlacement(placement);
    CHECK_DEVMEMHOLDER_OK(ValueHolder::CreateVoidGuarder("FreeMemory", out_addr, {}));
    output_addrs.emplace_back(std::move(out_addr));
  }
  return output_addrs;
}

std::vector<DevMemValueHolderPtr> AllocMemoriesWithoutGuarder(TensorPlacement placement,
                                                              const std::vector<ValueHolderPtr> &sizes,
                                                              LoweringGlobalData &global_data,
                                                              const int64_t stream_id) {
  auto allocator_holder = global_data.GetOrCreateAllocator({placement, AllocatorUsage::kAllocNodeOutput});
  std::vector<DevMemValueHolderPtr> output_addrs;
  for (size_t i = 0U; i < sizes.size(); ++i) {
    CHECK_DEVMEMHOLDER_OK(sizes[i]);
    auto out_addr = DevMemValueHolder::CreateSingleDataOutput(kMemoryTypeToMemKernelName[placement].alloc_name,
                                                              {allocator_holder, sizes[i]}, stream_id);
    CHECK_DEVMEMHOLDER_OK(out_addr);
    out_addr->SetPlacement(placement);
    output_addrs.emplace_back(std::move(out_addr));
  }
  return output_addrs;
}

DevMemValueHolderPtr AllocMemOnInit(TensorPlacement placement, size_t size, LoweringGlobalData &global_data) {
  auto addr_vec = FrameSelector::OnInitRoot(
      [&size, &placement, &global_data]() -> std::vector<ValueHolderPtr> {
    auto args_size = bg::ValueHolder::CreateConst(&size, sizeof(size));
    return {AllocMem(placement, args_size, global_data, kMainStream)};
  });
  GELOGD("Alloc memory on init size:%zu.", addr_vec.size());
  if (addr_vec.size() != 1) {
    return nullptr;
  }
  return std::dynamic_pointer_cast<DevMemValueHolder>(addr_vec[0]);
}

ValueHolderPtr CreateFftsMemAllocator(const ValueHolderPtr &window_size, LoweringGlobalData &global_data) {
  auto device_allocator =
      global_data.GetOrCreateAllocator({TensorPlacement::kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
  GE_ASSERT_NOTNULL(device_allocator, "allocator_holder must not be nullptr");
  return ValueHolder::CreateSingleDataOutput("CreateFftsMemAllocator", {device_allocator, window_size});
}

ValueHolderPtr RecycleFftsMems(const ValueHolderPtr &ffts_mem_allocator) {
  return ValueHolder::CreateVoid<bg::ValueHolder>("RecycleFftsMems", {ffts_mem_allocator});
}

std::vector<DevMemValueHolderPtr> AllocateFftsMems(const ValueHolderPtr &ffts_mem_allocator, const int64_t stream_id,
                                                   const std::vector<ValueHolderPtr> &block_sizes) {
  std::vector<DevMemValueHolderPtr> output_addrs(block_sizes.size());
  for (size_t i = 0UL; i < block_sizes.size(); ++i) {
    CHECK_DEVMEMHOLDER_OK(block_sizes[i]);
    auto out_addr =
        DevMemValueHolder::CreateSingleDataOutput("AllocateFftsMem", {ffts_mem_allocator, block_sizes[i]}, stream_id);
    CHECK_DEVMEMHOLDER_OK(out_addr);
    out_addr->SetPlacement(TensorPlacement::kOnDeviceHbm);
    CHECK_DEVMEMHOLDER_OK(ValueHolder::CreateVoidGuarder("FreeFftsMem", out_addr, {}));
    output_addrs[i] = std::move(out_addr);
  }
  return output_addrs;
}

DevMemValueHolderPtr AllocateBatchFftsMems(const ValueHolderPtr &ffts_mem_allocator, const ValueHolderPtr &batch_sizes,
                                           const int64_t stream_id) {
  auto batch_addrs =
      DevMemValueHolder::CreateSingleDataOutput("AllocateBatchFftsMems", {ffts_mem_allocator, batch_sizes}, stream_id);
  GE_ASSERT_NOTNULL(batch_addrs, "allocator_holder must not be nullptr");
  batch_addrs->SetPlacement(TensorPlacement::kOnDeviceHbm);
  GE_ASSERT_NOTNULL(ValueHolder::CreateVoidGuarder("FreeBatchFftsMems", batch_addrs, {}));
  return batch_addrs;
}

const MemoryKernelsName &GetMemoryKernelsName(const TensorPlacement placement) {
  return kMemoryTypeToMemKernelName[placement];
}
}  // namespace bg
}  // namespace gert

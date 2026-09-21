/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_builder/assign_device_mem.h"
#include "graph_builder/converter_checker.h"
#include "graph/utils/tensor_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "exe_graph/lowering/frame_selector.h"
#include "common/plugin/ge_make_unique_util.h"
#include "lowering/placement/placed_lowering_register.h"
#include "runtime/model_v2_executor.h"
#include "kernel/memory/sink_weight_assigner.h"
#include "kernel/common_kernel_impl/memory_copy.h"

namespace gert {
namespace {
constexpr uint32_t kFlattenOffsetKey = 0U;
constexpr uint32_t kFlattenSizeKey = 1U;
void PrepareForWeightSink(LoweringGlobalData &global_data) {
  auto require_weight_size = global_data.GetModelWeightSize();
  GELOGD("RequireSize[%zu]", require_weight_size);
  auto required_mem = [&require_weight_size]() -> std::vector<bg::ValueHolderPtr> {
    return bg::FrameSelector::OnInitRoot([&require_weight_size]() -> std::vector<bg::ValueHolderPtr> {
      auto required_size_holder = bg::ValueHolder::CreateConst(
          &require_weight_size, sizeof(require_weight_size));
      return {required_size_holder};
    });
  };
  (void)global_data.GetOrCreateUniqueValueHolder("RequireSize", required_mem);
}
} // namespace

bg::ValueHolderPtr AssignDeviceMem::GetBaseWeightAddr(LoweringGlobalData &global_data) {
  PrepareForWeightSink(global_data);
  auto weight_mem = [&global_data]() -> std::vector<bg::ValueHolderPtr> {
    return bg::FrameSelector::OnInitRoot([&global_data]() -> std::vector<bg::ValueHolderPtr> {
      auto required_size_holder = bg::HolderOnInit(global_data.GetUniqueValueHolder("RequireSize"));
      auto outer_weight_holder = bg::HolderOnInit(global_data.GetUniqueValueHolder("OuterWeightMem"));
      auto allocator_holder = global_data.GetOrCreateAllocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeWorkspace});
      return {bg::ValueHolder::CreateSingleDataOutput("GetOrCreateWeightMem",
                                                      {required_size_holder, outer_weight_holder, allocator_holder})};
    });
  };
  return global_data.GetOrCreateUniqueValueHolder("BaseWeightAddr", weight_mem)[0];
}

bg::ValueHolderPtr AssignDeviceMem::GetOrCreateMemAssigner(LoweringGlobalData &global_data) {
  auto mem_assinger_holder_creator = [&global_data]() -> std::vector<bg::ValueHolderPtr> {
    return bg::FrameSelector::OnInitRoot([&global_data]() -> std::vector<bg::ValueHolderPtr> {
        auto device_mem_holder = bg::HolderOnInit(GetBaseWeightAddr(global_data));
        return {bg::ValueHolder::CreateSingleDataOutput("CreateMemAssigner", {device_mem_holder})};
    });
  };
  return global_data.GetOrCreateUniqueValueHolder("CreateMemAssigner", mem_assinger_holder_creator)[0];
}

/*

     ┌───────────┐        ┌──────────────┐
     │RequireSize│        │OuterWeightMem│
     └───────────┴───┬────┴──────────────┘
                     │
            ┌────────▼───────────┐
            │GetOrCreateWeightMem│
            └────────┬───────────┘
                     │
 ┌─────┐      ┌──────▼──────────┐       ┌─────┐
 │Const│      │CreateMemAssigner│       │Const│
 └─────┴───┬──┴─────────────────┘       └──┬──┘
           │                               │
    ┌──────▼───────────┐             ┌─────▼──────┐
    │AssignWeightMemory│             │SpliteTensor│
    └──────────────────┴─────┬───────┴────────────┘
                             │
                             │
                     ┌───────▼──────┐
                     │SinkWeightData│
                     └──────────────┘
*/
ge::graphStatus SinkWeightData(const ge::Node* const_node, LoweringGlobalData &global_data, ge::DataType data_type,
                               const OutputLowerResult &src, int64_t dst_logic_stream_id, OutputLowerResult &dst) {
  (void)data_type;
  (void)dst_logic_stream_id;
  const auto op_desc = const_node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  // 此处将需要下沉的数据存放到已经预申请好的device内存上
  ge::ConstGeTensorPtr tensor;
  if (!ge::AttrUtils::GetTensor(op_desc, "value", tensor)) {
    GELOGE(ge::PARAM_INVALID, "Failed to get tensor value from const %s", const_node->GetName().c_str());
    return ge::PARAM_INVALID;
  }
  std::vector<int64_t> flatten_weight = GetFlattenOffsetInfo(&(tensor->GetTensorDesc()));
  GE_ASSERT_TRUE(!flatten_weight.empty());

  TensorData weight_offset;
  weight_offset.SetAddr(ge::ValueToPtr(flatten_weight[kFlattenOffsetKey]), nullptr);
  weight_offset.SetSize(static_cast<size_t>(flatten_weight[kFlattenSizeKey]));
  GELOGI("get const[%s], offset[%ld],size[%ld]", const_node->GetNamePtr(),
         flatten_weight[kFlattenOffsetKey], flatten_weight[kFlattenSizeKey]);

  auto weight_offset_holder = bg::ValueHolder::CreateConst(&weight_offset, sizeof(TensorData));
  auto get_or_create = AssignDeviceMem::GetOrCreateMemAssigner(global_data);
  const int64_t logic_stream_id = op_desc->GetStreamId();
  auto stream_id_holder = bg::ValueHolder::CreateConst(&logic_stream_id, sizeof(logic_stream_id));
  GE_ASSERT_TRUE(IsValidHolder(stream_id_holder));
  auto assign_mem_holder = bg::ValueHolder::CreateSingleDataOutput(
      kernel::kAssignWeightMemory, {weight_offset_holder, bg::HolderOnInit(get_or_create), stream_id_holder});
  auto sink_weight_data = bg::DevMemValueHolder::CreateSingleDataOutput(
      "SinkWeightData", {src.address, assign_mem_holder, stream_id_holder,
                         global_data.GetStreamById(logic_stream_id)}, logic_stream_id);
  GE_ASSERT_NOTNULL(sink_weight_data);
  for (const auto &src_ordered_holder : src.order_holders) {
    bg::ValueHolder::AddDependency(src_ordered_holder, sink_weight_data);
  }
  dst.shape = src.shape;
  dst.address = sink_weight_data;
  dst.address->SetPlacement(kOnDeviceHbm);
  dst.has_init = true;
  return ge::GRAPH_SUCCESS;
}
REGISTER_OUTPUT_LOWER(Const, SinkWeightData);
}  // namesapce gert

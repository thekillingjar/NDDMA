/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "split_to_sequence.h"

#include "common/hyper_status.h"
#include "engine/aicpu/graph_builder/bg_launch.h"
#include "framework/common/ge_types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/node_utils.h"
#include "graph_builder/bg_infer_shape.h"
#include "graph_builder/bg_memory.h"
#include "graph_builder/bg_rt_session.h"
#include "graph_builder/bg_tensor.h"
#include "graph_builder/converter_checker.h"

namespace {
  constexpr size_t kOutputNum = 1U;
  constexpr size_t kInputNum = 1U;
}
namespace gert {
bg::ValueHolderPtr SplitToSequenceCompute(bg::ValueHolderPtr &axis_holder,
                                          bg::ValueHolderPtr &split_shapes,
                                          bg::ValueHolderPtr &inner_split_addrs,
                                          const LowerInput &lower_input) {
  return bg::ValueHolder::CreateSingleDataOutput("SplitToSequenceDoCompute",
      {axis_holder, lower_input.input_shapes[0], lower_input.input_addrs[0], split_shapes, inner_split_addrs});
}

bg::ValueHolderPtr InferSplitOutputShape(uint32_t input_num, bg::ValueHolderPtr &axis_holder,
                                         bg::ValueHolderPtr &input_num_holder, const LowerInput &lower_input) {
  if (input_num == 1) {
    return bg::ValueHolder::CreateSingleDataOutput("GetSplitVecWithInput",
        {axis_holder, input_num_holder, lower_input.input_shapes[0]});
  }
  return bg::ValueHolder::CreateSingleDataOutput("GetSplitVecWithInput",
      {axis_holder, input_num_holder, lower_input.input_shapes[0],
      lower_input.input_shapes[1], lower_input.input_addrs[1]});
}

bg::ValueHolderPtr CalcSpecifiedShapeSize(const ge::NodePtr &node, bg::ValueHolderPtr &split_shapes) {
  auto td = node->GetOpDescBarePtr()->GetInputDescPtr(0);
  if (td == nullptr) {
    return nullptr;
  }
  ge::DataType dt = td->GetDataType();
  auto data_type = bg::ValueHolder::CreateConst(&dt, sizeof(dt));
  return bg::ValueHolder::CreateSingleDataOutput("CalcTensorSizeFromSpecifiedShape", {data_type, split_shapes});
}

bg::ValueHolderPtr AllocSpecifedShapeMemory(TensorPlacement placement,
                                            const bg::ValueHolderPtr &specified_sizes,
                                            LoweringGlobalData &global_data) {
  auto allocator_holder = global_data.GetOrCreateAllocator({placement, AllocatorUsage::kAllocNodeOutput});
  auto mem = bg::ValueHolder::CreateSingleDataOutput("AllocSpecifiedMem", {allocator_holder, specified_sizes});
  bg::ValueHolder::CreateVoidGuarder("FreeSpecifiedMem", mem, {});
  return mem;
}

bg::ValueHolderPtr StoreSplitTensorToSequence(bg::StoreSequenceArg storeArg) {
  std::vector<bg::ValueHolderPtr> inputs;

  inputs.emplace_back(storeArg.session_id);
  inputs.emplace_back(storeArg.contanier_id);
  inputs.emplace_back(storeArg.inner_tensor_addrs);
  inputs.emplace_back(storeArg.inner_tensor_shapes);
  inputs.emplace_back(storeArg.input_num);
  inputs.emplace_back(storeArg.keep_dims_holder);
  inputs.emplace_back(storeArg.axis_holder);
  inputs.emplace_back(storeArg.input_x_shape);
  inputs.emplace_back(storeArg.output_tensor);

  return bg::ValueHolder::CreateSingleDataOutput("StoreSplitTensorToSequence", inputs);
}

bg::ValueHolderPtr GetHandleShape() {
  return bg::ValueHolder::CreateSingleDataOutput("GetSequenceHandleShape", {});
}

LowerResult LoweringSplitToSeuqnece(const ge::NodePtr &node, const LowerInput &lower_input) {
  if ((node == nullptr) || (node->GetOpDescBarePtr() == nullptr)) {
    GELOGE(ge::PARAM_INVALID, "[Check][Op]Can not find op.");
    REPORT_INNER_ERR_MSG("E19999", "Can not find op.");
    return {HyperStatus::ErrorStatus(static_cast<const char*>("Can not find op")), {}, {}, {}};
  }
  auto ret = CheckLowerInput(lower_input);
  if (!ret.IsSuccess()) {
    GELOGE(ge::PARAM_INVALID, "[Check][LowerInput]Op %s type %s lower_input is invalid.", node->GetName().c_str(),
           ge::NodeUtils::GetNodeType(node).c_str());
    REPORT_INNER_ERR_MSG("E19999", "Op %s type %s lower_input is invalid.", node->GetName().c_str(),
                       ge::NodeUtils::GetNodeType(node).c_str());
    return {ret, {}, {}, {}};
  }
  auto input_num = lower_input.input_addrs.size();
  if (input_num < 1) {
    GELOGE(ge::PARAM_INVALID, "[Check][Op]Input num err, it is at least 1.");
    REPORT_INNER_ERR_MSG("E19999", "Input num err, it is at least 1.");
    return {HyperStatus::ErrorStatus(static_cast<const char*>("Input num err, it is at least 1")), {}, {}, {}};
  }
  auto input_num_holder = bg::ValueHolder::CreateConst(&input_num, sizeof(input_num));

  auto output_shape = GetHandleShape();
  std::vector<bg::ValueHolderPtr> output_shapes;
  output_shapes.push_back(output_shape);
  auto output_sizes = bg::CalcTensorSize(node, output_shapes);
  auto output_addrs = bg::AllocOutputMemory(kOnHost, node, output_sizes, *(lower_input.global_data));
  CONVERTER_CHECK_HOLDERS_ALL_OK(output_addrs, kOutputNum);
  LOWER_REQUIRE(lower_input.input_shapes.size() >= kInputNum);
  LOWER_REQUIRE(lower_input.input_addrs.size() >= kInputNum);
  int32_t axis = 0;
  (void) ge::AttrUtils::GetInt(node->GetOpDescBarePtr(), "axis", axis);
  auto axis_holder = bg::ValueHolder::CreateConst(&axis, sizeof(axis));
  auto split_shapes = InferSplitOutputShape(input_num, axis_holder, input_num_holder, lower_input);
  auto specified_size = CalcSpecifiedShapeSize(node, split_shapes);
  auto inner_split_addrs = AllocSpecifedShapeMemory(kOnDeviceHbm, specified_size, *(lower_input.global_data));
  auto compute_holder = SplitToSequenceCompute(axis_holder, split_shapes, inner_split_addrs, lower_input);

  int32_t keep_dims = 1;
  (void) ge::AttrUtils::GetInt(node->GetOpDescBarePtr(), "keepdims", keep_dims);
  auto keep_dims_holder = bg::ValueHolder::CreateConst(&keep_dims, sizeof(keep_dims));
  auto output_tensor = bg::BuildTensor(node, 0, kOnHost, output_shapes[0], output_addrs[0]);
  auto session_id = bg::GetSessionId(*lower_input.global_data);
  auto contain_id_holder = bg::GetContainerIdHolder(lower_input);
  auto store_holder = StoreSplitTensorToSequence({session_id, contain_id_holder, inner_split_addrs, split_shapes,
        input_num_holder, keep_dims_holder, axis_holder, lower_input.input_shapes[0], output_tensor});

  bg::ValueHolder::AddDependency(compute_holder, store_holder);
  return {HyperStatus::Success(), {compute_holder, store_holder}, output_shapes, output_addrs};
}

REGISTER_NODE_CONVERTER_PLACEMENT("SplitToSequence", kOnDeviceHbm, LoweringSplitToSeuqnece);
}  // namespace gert

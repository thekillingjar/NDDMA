/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_builder/bg_infer_shape.h"
#include "graph_builder/bg_memory.h"
#include "engine/aicpu/graph_builder/bg_launch.h"
#include "framework/common/ge_types.h"
#include "common/hyper_status.h"

#include "graph/debug/ge_attr_define.h"
#include "engine/aicpu/graph_builder/bg_aicpu_arg.h"
#include "graph/utils/node_utils.h"
#include "graph_builder/converter_checker.h"
#include "graph_builder/bg_tensor.h"
#include "graph_builder/bg_rt_session.h"

namespace {
  constexpr size_t kHandleIndex = 0U;
  constexpr size_t kDataIndex = 1U;
  constexpr size_t kAtIndexNum = 2U;
  constexpr size_t kLengthIndexNum = 1U;
  constexpr size_t kOutputNum = 1U;
}
namespace gert {
LowerResult LoweringSequenceAt(const ge::NodePtr &node, const LowerInput &lower_input) {
  if ((node == nullptr) || (node->GetOpDescBarePtr() == nullptr)) {
    GELOGE(ge::PARAM_INVALID, "[Check][Op]Can not find op.");
    REPORT_INNER_ERR_MSG("E39999", "Can not find op.");
    return {HyperStatus::ErrorStatus(static_cast<const char*>("Can not find op")), {}, {}, {}};
  }
  auto ret = CheckLowerInput(lower_input);
  if (!ret.IsSuccess() || (lower_input.input_addrs.size() < kAtIndexNum)) {
    GELOGE(ge::PARAM_INVALID, "[Check][LowerInput]Op %s type %s lower_input is invalid.", node->GetName().c_str(),
           ge::NodeUtils::GetNodeType(node).c_str());
    REPORT_INNER_ERR_MSG("E39999", "Op %s type %s lower_input is invalid.", node->GetName().c_str(),
                       ge::NodeUtils::GetNodeType(node).c_str());
    return {ret, {}, {}, {}};
  }
  auto session_id = bg::GetSessionId(*lower_input.global_data);
  auto container_id_holder = bg::GetContainerIdHolder(lower_input);

  std::vector<bg::ValueHolderPtr> inputs;
  inputs.emplace_back(session_id);
  inputs.emplace_back(container_id_holder);
  inputs.emplace_back(lower_input.input_addrs[kHandleIndex]);
  inputs.emplace_back(lower_input.input_addrs[kDataIndex]);

  // launch tensor
  auto output_tensors = bg::ValueHolder::CreateSingleDataOutput("SequenceAtCompute", inputs);
  LOWER_REQUIRE_VALID_HOLDER(output_tensors, "Failed create SequenceAtCompute lower run kernel func node");

  // launch shape
  auto output_shapes = bg::ValueHolder::CreateSingleDataOutput("BuildInferShapeFromTensorCompute", {output_tensors});
  LOWER_REQUIRE_VALID_HOLDER(output_shapes, "Failed create BuildInferShapeFromTensorCompute node");

  // launch addr
  auto allocator_holder = lower_input.global_data->GetOrCreateAllocator(
      {static_cast<TensorPlacement>(output_tensors->GetPlacement()), AllocatorUsage::kAllocNodeWorkspace});
  auto output_addrs = bg::DevMemValueHolder::CreateSingleDataOutput(
      "BuildAddrFromTensorCompute", {output_tensors, allocator_holder}, node->GetOpDescBarePtr()->GetStreamId());
  LOWER_REQUIRE_VALID_HOLDER(output_addrs, "Failed create BuildAddrFromTensorCompute lower run kernel func node");

  return {HyperStatus::Success(),
          {output_tensors, output_shapes, output_addrs},
          {output_shapes},
          {output_addrs}};
}

LowerResult LoweringSequenceEmpty(const ge::NodePtr &node, const LowerInput &lower_input) {
  if ((node == nullptr) || (node->GetOpDescBarePtr() == nullptr)) {
    GELOGE(ge::PARAM_INVALID, "[Check][Op]Can not find op.");
    REPORT_INNER_ERR_MSG("E39999", "Can not find op.");
    return {HyperStatus::ErrorStatus(static_cast<const char*>("Can not find op")), {}, {}, {}};
  }
  auto ret = CheckLowerInput(lower_input);
  if (!ret.IsSuccess()) {
    GELOGE(ge::PARAM_INVALID, "[Check][LowerInput]Op %s type %s lower_input is invalid.", node->GetName().c_str(),
           ge::NodeUtils::GetNodeType(node).c_str());
    REPORT_INNER_ERR_MSG("E39999", "Op %s type %s lower_input is invalid.", node->GetName().c_str(),
                       ge::NodeUtils::GetNodeType(node).c_str());
    return {ret, {}, {}, {}};
  }

  auto session_id = bg::GetSessionId(*lower_input.global_data);
  auto container_id_holder = bg::GetContainerIdHolder(lower_input);

  int32_t dtype = 0;
  (void)ge::AttrUtils::GetInt(node->GetOpDescBarePtr(), "dtype", dtype);
  GELOGD("LoweringSequenceEmpty dtype = %u", dtype);
  auto dtype_holder = bg::ValueHolder::CreateConst(&dtype, sizeof(dtype));

  // infershape
  auto output_shapes = bg::InferStorageShape(node, lower_input.input_shapes, *(lower_input.global_data));
  auto output_sizes = bg::CalcTensorSize(node, output_shapes);
  auto output_addrs = bg::AllocOutputMemory(kOnHost, node, output_sizes, *(lower_input.global_data));
  CONVERTER_CHECK_HOLDERS_ALL_OK(output_addrs, kOutputNum);
  CONVERTER_CHECK_HOLDERS_ALL_OK(output_shapes, kOutputNum);
  auto output_tensor = bg::BuildTensor(node, 0, static_cast<TensorPlacement>(output_addrs[0]->GetPlacement()),
                                       output_shapes[0], output_addrs[0]);

  // inputs
  std::vector<bg::ValueHolderPtr> inputs;
  inputs.emplace_back(session_id);
  inputs.emplace_back(container_id_holder);
  inputs.emplace_back(dtype_holder);
  inputs.emplace_back(output_tensor);

  // launch
  auto compute_holder = bg::ValueHolder::CreateSingleDataOutput("SequenceEmptyCompute", inputs);
  LOWER_REQUIRE_VALID_HOLDER(compute_holder, "Failed create SequenceEmptyCompute lower run kernel func node");

  return {HyperStatus::Success(),
          {compute_holder},
          {output_shapes},
          {output_addrs}};
}

LowerResult LoweringSequenceLength(const ge::NodePtr &node, const LowerInput &lower_input) {
  if ((node == nullptr) || (node->GetOpDescBarePtr() == nullptr)) {
    GELOGE(ge::PARAM_INVALID, "[Check][Op]Can not find op.");
    REPORT_INNER_ERR_MSG("E39999", "Can not find op.");
    return {HyperStatus::ErrorStatus(static_cast<const char*>("Can not find op")), {}, {}, {}};
  }
  auto ret = CheckLowerInput(lower_input);
  if (!ret.IsSuccess() || (lower_input.input_addrs.size() < kLengthIndexNum)) {
    GELOGE(ge::PARAM_INVALID, "[Check][LowerInput]Op %s type %s lower_input is invalid.", node->GetName().c_str(),
           ge::NodeUtils::GetNodeType(node).c_str());
    REPORT_INNER_ERR_MSG("E39999", "Op %s type %s lower_input is invalid.", node->GetName().c_str(),
                       ge::NodeUtils::GetNodeType(node).c_str());
    return {ret, {}, {}, {}};
  }

  auto session_id = bg::GetSessionId(*lower_input.global_data);
  auto container_id_holder = bg::GetContainerIdHolder(lower_input);

  // infershape
  auto output_shapes = bg::InferStorageShape(node, lower_input.input_shapes, *(lower_input.global_data));
  auto output_sizes = bg::CalcTensorSize(node, output_shapes);
  auto output_addrs = bg::AllocOutputMemory(kOnHost, node, output_sizes, *(lower_input.global_data));

  CONVERTER_CHECK_HOLDERS_ALL_OK(output_addrs, kOutputNum);
  CONVERTER_CHECK_HOLDERS_ALL_OK(output_shapes, kOutputNum);
  auto output_tensor = bg::BuildTensor(node, 0, static_cast<TensorPlacement>(output_addrs[0]->GetPlacement()),
                                       output_shapes[0], output_addrs[0]);

  // inputs
  std::vector<bg::ValueHolderPtr> inputs;
  inputs.emplace_back(session_id);
  inputs.emplace_back(container_id_holder);
  inputs.emplace_back(lower_input.input_addrs[kHandleIndex]);
  inputs.emplace_back(output_tensor);

  // launch
  auto compute_holder = bg::ValueHolder::CreateSingleDataOutput("SequenceLengthCompute", inputs);
  LOWER_REQUIRE_VALID_HOLDER(compute_holder, "Failed create SequenceLengthCompute lower run kernel func node");

  return {HyperStatus::Success(),
          {compute_holder},
          {output_shapes},
          {output_addrs}};
}

REGISTER_NODE_CONVERTER_PLACEMENT("SequenceAt", kOnHost, LoweringSequenceAt);
REGISTER_NODE_CONVERTER_PLACEMENT("SequenceEmpty", kOnHost, LoweringSequenceEmpty);
REGISTER_NODE_CONVERTER_PLACEMENT("SequenceLength", kOnHost, LoweringSequenceLength);
}  // namespace gert

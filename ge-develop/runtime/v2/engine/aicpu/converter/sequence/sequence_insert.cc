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
#include "sequence_ops.h"
#include "graph_builder/bg_rt_session.h"

namespace {
  constexpr size_t kOutputNum = 1U;
}
namespace gert {
bg::ValueHolderPtr SequenceInsertCompute(bg::ValueHolderPtr session_id, bg::ValueHolderPtr container_id,
                                         bg::ValueHolderPtr input_num_holder, const LowerInput &lower_input,
                                         bg::ValueHolderPtr output_tensor) {
  std::vector<bg::ValueHolderPtr> inputs;
  inputs.emplace_back(session_id);
  inputs.emplace_back(container_id);
  inputs.emplace_back(input_num_holder);
  // For handle and insert index we don't neet to transfer it's shape
  inputs.emplace_back(lower_input.input_addrs[0]);
  inputs.emplace_back(lower_input.input_addrs[1]);
  inputs.emplace_back(lower_input.input_shapes[1]);
  constexpr size_t input_index_idx = 2;
  if (lower_input.input_addrs.size() == (input_index_idx + 1)) {
    inputs.emplace_back(lower_input.input_addrs[input_index_idx]);
  }

  inputs.emplace_back(output_tensor);
  auto compute_holder = bg::ValueHolder::CreateSingleDataOutput("SequenceInsertCompute", inputs);
  return compute_holder;
}

LowerResult LoweringSequenceInsert(const ge::NodePtr &node, const LowerInput &lower_input) {
  if ((node == nullptr) || (node->GetOpDescBarePtr() == nullptr)) {
    GELOGE(ge::PARAM_INVALID, "[Check][Op]Can not find op.");
    REPORT_INNER_ERR_MSG("E39999", "Can not find op.");
    return {HyperStatus::ErrorStatus(static_cast<const char *>("Can not find op")), {}, {}, {}};
  }
  auto ret = CheckLowerInput(lower_input);
  if (!ret.IsSuccess()) {
    GELOGE(ge::PARAM_INVALID, "[Check][LowerInput]Op %s type %s lower_input is invalid.", node->GetName().c_str(),
           ge::NodeUtils::GetNodeType(node).c_str());
    REPORT_INNER_ERR_MSG("E39999", "Op %s type %s lower_input is invalid.", node->GetName().c_str(),
                       ge::NodeUtils::GetNodeType(node).c_str());
    return {ret, {}, {}, {}};
  }

  auto input_num = lower_input.input_addrs.size();
  constexpr size_t least_input_number = 2;
  if (input_num < least_input_number) {
    GELOGE(ge::PARAM_INVALID, "[Check][Op]Input num err, it is at least 2.");
    REPORT_INNER_ERR_MSG("E39999", "Input num err, it is at least 2.");
    return {HyperStatus::ErrorStatus(static_cast<const char *>("Input num err, it is at least 2.")), {}, {}, {}};
  }
  auto input_num_holder = bg::ValueHolder::CreateConst(&input_num, sizeof(input_num));
  auto output_shape = bg::ValueHolder::CreateSingleDataOutput("GetSequenceHandleShape", {});
  std::vector<bg::ValueHolderPtr> output_shapes;
  output_shapes.push_back(output_shape);
  auto output_sizes = bg::CalcTensorSize(node, output_shapes);
  auto output_addrs = bg::AllocOutputMemory(kOnHost, node, output_sizes, *(lower_input.global_data));

  CONVERTER_CHECK_HOLDERS_ALL_OK(output_addrs, kOutputNum);
  auto output_tensor = bg::BuildTensor(node, 0, kOnHost, output_shapes[0], output_addrs[0]);
  auto session_id = bg::GetSessionId(*lower_input.global_data);
  bg::ValueHolderPtr container_id_holder = bg::GetContainerIdHolder(lower_input);
  auto compute_holder =
      SequenceInsertCompute(session_id, container_id_holder, input_num_holder, lower_input, output_tensor);

  return {HyperStatus::Success(), {compute_holder}, output_shapes, output_addrs};
}
REGISTER_NODE_CONVERTER_PLACEMENT("SequenceInsert", kOnHost, LoweringSequenceInsert);
}  // namespace gert

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include "graph_builder/bg_infer_shape.h"
#include "graph_builder/bg_memory.h"
#include "framework/common/ge_types.h"
#include "common/hyper_status.h"
#include "graph/utils/node_utils.h"
#include "graph_builder/converter_checker.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "framework/common/types.h"
#include "engine/node_converter_utils.h"

namespace gert {
namespace {
constexpr int64_t kNpuFloatStatusShape = 8;
constexpr int64_t kNpuFloatDebugStatusShape = 8;
}  // namespace

LowerResult LoweringNpuGetFloatStatus(const ge::NodePtr &node, const LowerInput lower_input) {
  GELOGI("Start lowering %s(%s)", node->GetName().c_str(), node->GetType().c_str());
  StorageShape shape({kNpuFloatStatusShape}, {kNpuFloatStatusShape});
  const auto op_desc = node->GetOpDescBarePtr();
  LOWER_REQUIRE_NOTNULL(op_desc);
  bg::ValueHolderPtr output_shape_tensor = NodeConverterUtils::CreateOutputShape(op_desc->GetOutputDescPtr(0U), shape);

  auto output_size = bg::CalcOutTensorSize(node, 0UL, output_shape_tensor);
  auto output_addrs = bg::AllocOutputMemory(kOnDeviceHbm, node, {output_size}, *(lower_input.global_data));
  // host_args_holder
  std::vector<bg::ValueHolderPtr> get_float_status_args(output_addrs.cbegin(), output_addrs.cend());
  auto host_args_holder = bg::ValueHolder::CreateSingleDataOutput("NpuGetFloatStatusArgs", get_float_status_args);

  const size_t arg_size = sizeof(uint8_t *);
  auto arg_size_holder = bg::ValueHolder::CreateConst(&arg_size, sizeof(arg_size));
  auto arg_dev = bg::AllocMem(kOnDeviceHbm, arg_size_holder, *(lower_input.global_data),
                              node->GetOpDesc()->GetStreamId());

  std::vector<bg::ValueHolderPtr> inputs{arg_dev, host_args_holder, lower_input.global_data->GetStream()};
  auto launch_holder = bg::ValueHolder::CreateSingleDataOutput("NpuGetFloatStatus", inputs);

  return {HyperStatus::Success(), {launch_holder}, {output_shape_tensor}, output_addrs};
}

LowerResult LoweringNpuClearFloatStatus(const ge::NodePtr &node, const LowerInput lower_input) {
  GELOGI("Start lowering %s(%s)", node->GetName().c_str(), node->GetType().c_str());
  StorageShape shape({kNpuFloatDebugStatusShape}, {kNpuFloatDebugStatusShape});
  const auto op_desc = node->GetOpDescBarePtr();
  LOWER_REQUIRE_NOTNULL(op_desc);
  bg::ValueHolderPtr output_shape_tensor = NodeConverterUtils::CreateOutputShape(op_desc->GetOutputDescPtr(0U), shape);
  auto output_size = bg::CalcOutTensorSize(node, 0UL, output_shape_tensor);
  auto output_addrs = bg::AllocOutputMemory(kOnDeviceHbm, node, {output_size}, *(lower_input.global_data));

  std::vector<bg::ValueHolderPtr> inputs{lower_input.global_data->GetStream()};
  auto launch_holder = bg::ValueHolder::CreateSingleDataOutput("NpuClearFloatStatus", inputs);

  return {HyperStatus::Success(), {launch_holder}, {output_shape_tensor}, output_addrs};
}

LowerResult LoweringNpuGetFloatDebugStatus(const ge::NodePtr &node, const LowerInput lower_input) {
  GELOGI("Start lowering %s(%s)", node->GetName().c_str(), node->GetType().c_str());
  StorageShape shape({kNpuFloatDebugStatusShape}, {kNpuFloatDebugStatusShape});
  const auto op_desc = node->GetOpDescBarePtr();
  LOWER_REQUIRE_NOTNULL(op_desc);
  bg::ValueHolderPtr output_shape_tensor = NodeConverterUtils::CreateOutputShape(op_desc->GetOutputDescPtr(0U), shape);
  auto output_size = bg::CalcOutTensorSize(node, 0UL, output_shape_tensor);
  auto output_addrs = bg::AllocOutputMemory(kOnDeviceHbm, node, {output_size}, *(lower_input.global_data));
  // host_args_holder
  // use NpuGetFloatStatusArgs
  std::vector<bg::ValueHolderPtr> get_float_status_args(output_addrs.cbegin(), output_addrs.cend());
  auto host_args_holder = bg::ValueHolder::CreateSingleDataOutput("NpuGetFloatStatusArgs", get_float_status_args);

  const size_t arg_size = sizeof(uint8_t *);
  auto arg_size_holder = bg::ValueHolder::CreateConst(&arg_size, sizeof(arg_size));
  auto arg_dev = bg::AllocMem(kOnDeviceHbm, arg_size_holder, *(lower_input.global_data),
                              node->GetOpDesc()->GetStreamId());

  std::vector<bg::ValueHolderPtr> inputs{arg_dev, host_args_holder, lower_input.global_data->GetStream()};
  auto launch_holder = bg::ValueHolder::CreateSingleDataOutput("NpuGetFloatDebugStatus", inputs);

  return {HyperStatus::Success(), {launch_holder}, {output_shape_tensor}, output_addrs};
}

LowerResult LoweringNpuClearFloatDebugStatus(const ge::NodePtr &node, const LowerInput lower_input) {
  GELOGI("Start lowering %s(%s)", node->GetName().c_str(), node->GetType().c_str());
  StorageShape shape({kNpuFloatDebugStatusShape}, {kNpuFloatDebugStatusShape});
  const auto op_desc = node->GetOpDescBarePtr();
  LOWER_REQUIRE_NOTNULL(op_desc);
  bg::ValueHolderPtr output_shape_tensor = NodeConverterUtils::CreateOutputShape(op_desc->GetOutputDescPtr(0U), shape);
  auto output_size = bg::CalcOutTensorSize(node, 0UL, output_shape_tensor);
  auto output_addrs = bg::AllocOutputMemory(kOnDeviceHbm, node, {output_size}, *(lower_input.global_data));

  std::vector<bg::ValueHolderPtr> inputs{lower_input.global_data->GetStream()};
  auto launch_holder = bg::ValueHolder::CreateSingleDataOutput("NpuClearFloatDebugStatus", inputs);

  return {HyperStatus::Success(), {launch_holder}, {output_shape_tensor}, output_addrs};
}

LowerResult LoweringNpuClearFloatStatusV2(const ge::NodePtr &node, const LowerInput lower_input) {
  GELOGI("Start lowering %s(%s)", node->GetName().c_str(), node->GetType().c_str());

  std::vector<bg::ValueHolderPtr> inputs{lower_input.global_data->GetStream()};
  auto launch_holder = bg::ValueHolder::CreateSingleDataOutput("NpuClearFloatStatus", inputs);

  return {HyperStatus::Success(), {launch_holder}, {}, {}};
}

LowerResult LoweringRtsNode(const ge::NodePtr &node, const LowerInput &lower_input) {
  static std::map<std::string, std::function<LowerResult(const ge::NodePtr, const LowerInput &lowerInput)>>
      type_to_lower_func = {{ge::NPUGETFLOATSTATUS, LoweringNpuGetFloatStatus},
                            {ge::NPUGETFLOATSTATUSV2, LoweringNpuGetFloatStatus},
                            {ge::NPUCLEARFLOATSTATUS, LoweringNpuClearFloatStatus},
                            {ge::NPUCLEARFLOATSTATUSV2, LoweringNpuClearFloatStatusV2},
                            {ge::NPUGETFLOATDEBUGSTATUS, LoweringNpuGetFloatDebugStatus},
                            {ge::NPUCLEARFLOATDEBUGSTATUS, LoweringNpuClearFloatDebugStatus}};
  auto iter = type_to_lower_func.find(node->GetType());
  if (iter != type_to_lower_func.end()) {
    return iter->second(node, lower_input);
  }

  return {
      HyperStatus::ErrorStatus("Cannot find lowering func for node type:[%s].", node->GetType().c_str()), {}, {}, {}};
}

REGISTER_NODE_CONVERTER_PLACEMENT("DNN_VM_RTS_OP_STORE", kOnDeviceHbm, LoweringRtsNode);
REGISTER_NODE_CONVERTER_PLACEMENT("DNN_VM_RTS_FFTS_PLUS_OP_STORE", kOnDeviceHbm, LoweringRtsNode);
}  // namespace gert

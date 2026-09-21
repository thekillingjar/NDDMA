/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dvpp_node_converter.h"
#include "framework/common/ge_types.h"
#include "engine/node_converter_utils.h"
#include "graph_builder/bg_infer_shape.h"
#include "graph_builder/bg_memory.h"
#include "common/hyper_status.h"
#include "graph/utils/node_utils.h"
#include "graph_builder/converter_checker.h"
#include "dvpp_graph_builder/bg_generate_sqe_and_launch_task.h"
#include "dvpp_graph_builder/bg_calc_dvpp_workspace_size.h"
#include "exe_graph/lowering/value_holder.h"
namespace gert {
LowerResult LoweringDvppNode(const ge::NodePtr &node,
                             const LowerInput &lower_input) {
  auto compile_result = lower_input.global_data->FindCompiledResult(node);
  if (compile_result == nullptr) {
    GELOGE(ge::PARAM_INVALID, "Cannot find compile result for node %s type %s",
           node->GetName().c_str(), ge::NodeUtils::GetNodeType(node).c_str());
    return {HyperStatus::ErrorStatus(
        static_cast<const char *>("Can not find compile result")), {}, {}, {}};
  }
  GELOGD("LoweringDvppNode node %s type %s", node->GetName().c_str(),
      ge::NodeUtils::GetNodeType(node).c_str());
  // 1. infershape
  auto output_shapes = bg::InferStorageShape(node, lower_input.input_shapes, *(lower_input.global_data));

  // 2. alloc output mem
  auto output_sizes = bg::CalcTensorSize(node, output_shapes);
  auto output_addrs = bg::AllocOutputMemory(
      kOnDeviceHbm, node, output_sizes, *(lower_input.global_data));

  // 3. alloc workspace mem
  auto workspace_size = bg::CalcDvppWorkSpaceSize(lower_input.input_shapes,
      output_shapes);
  auto workspace_addr = bg::AllocWorkspaceMem(
      kOnDeviceHbm, workspace_size, *(lower_input.global_data));
  // 4. generate dvpp sqe and launch task
  std::vector<bg::ValueHolderPtr> addrs_info;
  addrs_info.insert(addrs_info.end(), lower_input.input_addrs.cbegin(),
      lower_input.input_addrs.cend());
  addrs_info.insert(addrs_info.end(), output_addrs.cbegin(), output_addrs.cend());
  addrs_info.emplace_back(workspace_addr);
  addrs_info.insert(addrs_info.end(), output_sizes.cbegin(), output_sizes.cend());
  addrs_info.emplace_back(workspace_size);

  auto launch_holder = bg::GenerateSqeAndLaunchTask(lower_input.input_shapes,
      output_shapes, addrs_info, lower_input.global_data->GetStream());

  // 5. free workspace mem
  auto free_holder = bg::FreeWorkspaceMem(kOnDeviceHbm, workspace_addr);
  bg::ValueHolder::AddDependency(launch_holder, free_holder);

  return {HyperStatus::Success(),
      {launch_holder}, output_shapes, output_addrs};
}

REGISTER_NODE_CONVERTER_PLACEMENT(ge::kEngineNameDvpp.c_str(),
                                  kOnDeviceHbm, LoweringDvppNode);
} // namespace gert

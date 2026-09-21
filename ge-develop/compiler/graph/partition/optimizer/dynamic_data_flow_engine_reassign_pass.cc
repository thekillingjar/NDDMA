/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/partition/optimizer/dynamic_data_flow_engine_reassign_pass.h"
#include "framework/common/util.h"
#include "framework/common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "mmpa/mmpa_api.h"

namespace ge {
namespace {
constexpr int32_t kIntBase = 10;
}
Status DynamicDataFlowEngineReassignPass::Run(const ComputeGraphPtr &graph,
                                              NodeEngineMap &node_atomic_engine_map,
                                              NodeEngineMap &node_composite_engine_map) {
  const char_t *runtime2_env = nullptr;
  MM_SYS_GET_ENV(MM_ENV_ENABLE_RUNTIME_V2, runtime2_env);
  if ((runtime2_env == nullptr) || (runtime2_env[0] != '0')) {
    GELOGD("ENABLE_RUNTIME_V2 is set, skip DynamicDataFlowEngineReassignPass");
    return SUCCESS;
  }
  GE_CHECK_NOTNULL(graph);
  (void)node_composite_engine_map;
  static const char_t *const kGeLocalEngineName = "DNN_VM_GE_LOCAL";
  static const char_t *const kGeLocalOpKernelLibName = "DNN_VM_GE_LOCAL_OP_STORE";
  static const std::unordered_set<std::string> kDataFlowOps = {STACK, STACKPUSH, STACKPOP, STACKCLOSE};
  for (const NodePtr &node : graph->GetAllNodes()) {
    const auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    if (kDataFlowOps.count(node->GetType()) == 0UL) {
      continue;
    }
    bool is_unknown = false;
    const bool ret = AttrUtils::GetBool(op_desc, ATTR_NAME_IS_UNKNOWN_SHAPE, is_unknown);
    if (ret && is_unknown) {
      op_desc->SetOpEngineName(kGeLocalEngineName);
      op_desc->SetOpKernelLibName(kGeLocalOpKernelLibName);
      node_atomic_engine_map[node] = kGeLocalEngineName;
      GELOGD("ReAssigning DNNEngine %s to node %s, op type %s", kGeLocalEngineName, node->GetName().c_str(),
             node->GetType().c_str());
    }
  }
  return SUCCESS;
}
}  // namespace ge

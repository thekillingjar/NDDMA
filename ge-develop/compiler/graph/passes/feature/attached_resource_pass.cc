/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/feature/attached_resource_pass.h"
#include "common/checker.h"
namespace ge {
const static std::unordered_set<std::string> kUnsupportedV1CtrlOps{
    "StreamActive", "StreamSwitch", "StreamMerge", "Exit",          "Switch",
    "Merge",        "Enter",        "RefEnter",    "NextIteration", "RefNextIteration"};
constexpr char_t const *kDisableAttachedResource = "_disable_attached_resource";

static bool HasUnsupportedOp(const ComputeGraphPtr &graph) {
  for (const auto &node : graph->GetDirectNode()) {
    if (kUnsupportedV1CtrlOps.count(node->GetType()) > 0UL) {
      GELOGI("Cannot allocate attached resource to graph [%s] with unsupported node [%s].", graph->GetName().c_str(),
             node->GetTypePtr());
      return true;
    }
  }
  return false;
}

Status AttachedResourcePass::Run(ComputeGraphPtr graph) {
  GELOGD("AttachedResourcePass begin");
  bool disable_attached_resource = HasUnsupportedOp(graph);
  if (!disable_attached_resource) {
    return SUCCESS;
  }

  for (const auto &node : graph->GetDirectNode()) {
    const auto &op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    GE_ASSERT_TRUE(AttrUtils::SetBool(op_desc, kDisableAttachedResource, true));
  }
  GELOGD("AttachedResourcePass end");
  return SUCCESS;
}

REG_PASS_OPTION("AttachedResourcePass").LEVELS(OoLevel::kO0);
}  // namespace ge

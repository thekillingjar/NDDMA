/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/standard_optimize/constant_folding/replace_with_empty_const_pass.h"
#include <string>
#include "common/plugin/ge_make_unique_util.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_type_utils.h"

namespace {
const std::string kPassName = "ReplaceWithEmptyConstPass";
const std::unordered_set<std::string> kControlFlowOps = {
    ge::SWITCH,
    ge::REFSWITCH,
    ge::MERGE,
    ge::REFMERGE,
    ge::ENTER,
    ge::REFENTER,
    ge::NEXTITERATION,
    ge::REFNEXTITERATION,
    ge::EXIT,
    ge::REFEXIT,
    ge::LOOPCOND
};
// tmp solution, when IR support is_stateful interface, here change to inferface
// resource op always show up on pair, once only one resouce op has empty output tensor,
// we should consider remove them together.
// Currently, we ignore resource op, let them run with empty tensor.
const std::unordered_set<std::string> kResourceOps = {
    ge::STACK,
    ge::STACKPOP,
    ge::STACKPUSH
};

bool IsControlFlowOp(const std::string &node_type) {
  return kControlFlowOps.count(node_type) != 0;
};

bool IsResourceOp(const std::string &node_type) {
  return kResourceOps.count(node_type) != 0;
};

bool IsHcomOp(const std::string &node_type) {
  return node_type.find("Hcom") != std::string::npos;
}
} // namespace
namespace ge {
bool ReplaceWithEmptyConstPass::NeedIgnorePass(const NodePtr &node) {
  const std::set<std::string> constant_like_task_ops = {CONSTANT, CONSTANTOP, FILECONSTANT};
  auto node_type = NodeUtils::GetNodeType(node);
  if ((constant_like_task_ops.count(node_type) > 0) || OpTypeUtils::IsDataNode(node_type)) {
    GELOGI("Node %s is const. Ignore current pass.", node->GetName().c_str());
    return true;
  }
  if (node_type == NETOUTPUT) {
    GELOGD("Node %s is netoutput, should be saved in graph.", node->GetName().c_str());
    return true;
  }
  if (IsControlFlowOp(node_type) || IsResourceOp(node_type) || IsHcomOp(node_type)) {
    GELOGI("Node %s type %s is stateful op. Ignore current pass.", node->GetName().c_str(), node_type.c_str());
    return true;
  }
  // Node like no op, it has no output
  if (node->GetOpDesc()->GetAllOutputsDescPtr().empty()) {
    GELOGI("Node %s has no output desc. Ignore current pass.", node->GetName().c_str());
    return true;
  }
  // if node is inserted by ge, like empty identity inserted by folding pass, need ignore current pass
  // to avoid optimize again and again
  bool is_inserted_by_ge = false;
  (void)AttrUtils::GetBool(node->GetOpDesc(), ATTR_NAME_IS_INSERTED_BY_GE, is_inserted_by_ge);
  if (is_inserted_by_ge) {
    return true;
  }
  return false;
}

bool ReplaceWithEmptyConstPass::NeedFold() const {
  return need_fold_;
}

Status ReplaceWithEmptyConstPass::ComputePotentialWeight(NodePtr &node, std::vector<GeTensorPtr> &outputs) {
  auto op_desc = node->GetOpDesc();
  // If any of outputs of current node is not empty, ignore pass
  if (!AreAllOutputsEmptyShape(op_desc)) {
    GELOGD("Node %s outputs are not all empty tensor, not change.", node->GetName().c_str());
    return NOT_CHANGED;
  }

  GELOGI("Node %s has empty tensor output. It will be replaced by empty const.", node->GetName().c_str());
  return GetOutputsOfCurrNode(node, outputs);
}

Status ReplaceWithEmptyConstPass::GetOutputsOfCurrNode(const NodePtr &node_to_replace,
                                                       std::vector<GeTensorPtr> &outputs) const {
  for (const auto &out_anchor : node_to_replace->GetAllOutDataAnchors()) {
    GE_CHECK_NOTNULL(node_to_replace->GetOpDesc());
    auto out_desc = node_to_replace->GetOpDesc()->GetOutputDesc(out_anchor->GetIdx());
    GeTensorPtr empty_tensor = MakeShared<ge::GeTensor>(out_desc);
    GE_CHECK_NOTNULL(empty_tensor);
    outputs.emplace_back(empty_tensor);
  }
  return SUCCESS;
}

string ReplaceWithEmptyConstPass::GetPassName() const {
  return kPassName;
}

REG_PASS_OPTION("ReplaceWithEmptyConstPass").LEVELS(OoLevel::kO1).SWITCH_OPT(ge::OO_CONSTANT_FOLDING);
}  // namespace ge

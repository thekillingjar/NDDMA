/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/feature/hccl_group_pass.h"
#include <deque>
#include "framework/common/debug/ge_log.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/common/util.h"
#include "common/checker.h"

namespace ge {
Status HcclGroupPass::Run(NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  OpDescPtr op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  bool is_fused_node = false;
  if (!AttrUtils::GetBool(op_desc, ATTR_NAME_HCCL_FUSED_FLAG, is_fused_node)) {
    GELOGW("Get attr ATTR_NAME_GRADIENT_FUSED_GROUP unsuccessful.");
    return SUCCESS;
  }

  if (!is_fused_node) {
    return SUCCESS;
  }

  if (op_desc->HasAttr(ATTR_NAME_HCCL_FUSED_GROUP)) {
    GELOGD("Current node %s already marked group id, ignore it.", node->GetName().c_str());
    return SUCCESS;
  }

  GELOGI("Recognized fused node %s", node->GetName().c_str());
  Status ret = MarkGroupForFusedNode(node);
  if (ret != SUCCESS) {
    GELOGW("Mark group for fused node %s unsuccessful. It might cause performance problem.", node->GetName().c_str());
  }

  GE_ASSERT_SUCCESS(AddCtrlFromLastARToOutput(node));
  return SUCCESS;
}

Status HcclGroupPass::AddCtrlFromLastARToOutput(const NodePtr &node) const {
  const auto switch_node = node->GetOwnerComputeGraph()->FindFirstNodeMatchType(SWITCH);
  if (switch_node == nullptr) {
    return SUCCESS;
  }

  const auto &out_ctrl_nodes = node->GetOutControlNodes();
  if (std::any_of(out_ctrl_nodes.begin(), out_ctrl_nodes.end(),
                  [](const NodePtr &n) { return (n->GetType() == HCOMALLREDUCE); })) {
    return SUCCESS;
  }

  const auto &in_ctrl_nodes = node->GetInControlNodes();
  if (std::none_of(in_ctrl_nodes.begin(), in_ctrl_nodes.end(),
                   [](const NodePtr &n) { return (n->GetType() == HCOMALLREDUCE); })) {
    return SUCCESS;
  }

  const auto output = node->GetOwnerComputeGraph()->FindFirstNodeMatchType(NETOUTPUT);
  GE_CHECK_NOTNULL(output);
  GE_CHECK_NOTNULL(node->GetOutControlAnchor());
  GE_CHECK_NOTNULL(output->GetInControlAnchor());
  if (!node->GetOutControlAnchor()->IsLinkedWith(output->GetInControlAnchor())) {
    GE_ASSERT_SUCCESS(node->GetOutControlAnchor()->LinkTo(output->GetInControlAnchor()));
    GELOGI("Add control edge from last allreduce: %s to %s.", node->GetName().c_str(), output->GetName().c_str());
  }

  return SUCCESS;
}

Status HcclGroupPass::MarkGroupForFusedNode(const NodePtr &fused_node) const {
  std::deque<NodePtr> queue;
  queue.push_back(fused_node);
  std::string group_id = fused_node->GetName();

  while (!queue.empty()) {
    NodePtr node = queue.front();
    queue.pop_front();
    for (auto out_data_node : node->GetOutDataNodes()) {
      if (out_data_node->GetType() == fused_node->GetType()) {
        // if meet fused node, it is the end of current group
        break;
      }
      if (!AttrUtils::SetStr(out_data_node->GetOpDesc(), ATTR_NAME_HCCL_FUSED_GROUP, group_id)) {
        GELOGW("Set attr ATTR_NAME_GRADIENT_FUSED_GROUP unsuccessful.");
        return FAILED;
      }
      GELOGI("Set hccl fused group_id %s for node %s", group_id.c_str(), out_data_node->GetName().c_str());
      queue.emplace_back(out_data_node);
    }
  }
  return SUCCESS;
}

REG_PASS_OPTION("HcclGroupPass").LEVELS(OoLevel::kO3);
}  // namespace ge

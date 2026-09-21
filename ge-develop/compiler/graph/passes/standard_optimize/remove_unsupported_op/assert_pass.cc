/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/standard_optimize/remove_unsupported_op/assert_pass.h"

#include <map>
#include <queue>
#include <string>
#include <vector>

#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/util.h"
#include "graph/utils/op_type_utils.h"
#include "base/err_msg.h"

namespace ge {
// aicpu not support std::string type, so current implemention is Upward traversal
Status AssertPass::Run(NodePtr &node) {
  if (node == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Param node is nullptr, check invalid");
    GELOGE(PARAM_INVALID, "param [node] must not be null.");
    return PARAM_INVALID;
  }
  if (node->GetOpDesc() == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Param op_desc of node is nullptr, check invalid");
    GELOGE(PARAM_INVALID, "[Get][OpDesc] param [node] [opDesc] must not be null.");
    return PARAM_INVALID;
  }

  if (node->GetType() == ASSERT) {
    std::vector<NodePtr> nodes_unused;
    // collect assert and other unused ops
    CollectUnusedNode(node, nodes_unused);
    // remove unused node
    Status status = RemoveUnusedNode(nodes_unused);
    if (status != SUCCESS) {
      GELOGE(status, "[Remove][UnusedNode] failed, ret:%d.", status);
      return status;
    }
  }
  return SUCCESS;
}

void AssertPass::CollectUnusedNode(const NodePtr &assert_node, std::vector<NodePtr> &nodes_unused) const {
  std::map<Node *, uint32_t> invalid_outdata_info;
  std::queue<NodePtr> node_queue;
  node_queue.push(assert_node);

  while (!node_queue.empty()) {
    NodePtr cur_node = node_queue.front();
    if (cur_node == nullptr) {
      continue;
    }
    node_queue.pop();
    nodes_unused.push_back(cur_node);

    for (const auto &src_node : cur_node->GetInDataNodes()) {
      if (src_node != nullptr && src_node->GetOpDesc() != nullptr) {
        auto size = ++invalid_outdata_info[src_node.get()];
        // src_node need to be deleted
        if ((src_node->GetOutDataNodesSize() == size) && (src_node->GetOpDesc()->GetType() != ENTER) &&
          (!OpTypeUtils::IsDataNode(src_node->GetOpDesc()->GetType()))) {
          node_queue.push(src_node);
        }
      }
    }
  }
}

Status AssertPass::RemoveUnusedNode(std::vector<NodePtr> &nodes_unused) {
  for (NodePtr &node : nodes_unused) {
    if (node == nullptr) {
      continue;
    }
    std::vector<int32_t> assert_io_map;
    size_t out_nums = node->GetAllOutDataAnchorsSize();
    while (out_nums > 0) {
      assert_io_map.push_back(-1);
      out_nums--;
    }

    if (IsolateAndDeleteNode(node, assert_io_map) != SUCCESS) {
      REPORT_INNER_ERR_MSG("E19999", "Isolate and delete node:%s(%s) failed",
                        node->GetName().c_str(), node->GetType().c_str());
      GELOGE(FAILED, "[Call][IsolateAndDeleteNode] for node:%s(%s) failed",
             node->GetName().c_str(), node->GetType().c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

REG_PASS_OPTION("AssertPass").LEVELS(OoLevel::kO0);
}  // namespace ge

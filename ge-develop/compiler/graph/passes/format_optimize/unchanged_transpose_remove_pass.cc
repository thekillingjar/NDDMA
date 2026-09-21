/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/format_optimize/unchanged_transpose_remove_pass.h"

#include <memory>
#include <string>
#include <vector>
#include "graph/utils/node_utils.h"
#include "host_kernels/kernel.h"
#include "host_kernels/kernel_factory.h"

namespace ge {
namespace {
constexpr uint32_t kTransposeInputX = 0U;
constexpr uint32_t kTransposeInputPerm = 1U;
}  // namespace

Status UnchangedTransposeRemovePass::Run(NodePtr &node) {
  GE_CHECK_NOTNULL(node);

  if (node->GetType() != TRANSPOSE) {
    return SUCCESS;
  }

  const auto &op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);

  if (!IsUnchangedTranspose(node, op_desc)) {
    return SUCCESS;
  }

  auto perm_node = NodeUtils::GetInDataNodeByIndex(*node, kTransposeInputPerm);
  GE_CHECK_NOTNULL(perm_node);
  GE_CHK_STATUS_RET(DeleteUselessConstAxisNode(perm_node), "Failed to remove const input node[%s] of node[%s]",
                    perm_node->GetName().c_str(), node->GetName().c_str());

  std::vector<int32_t> data_relink_io_map = {kTransposeInputX};
  GE_CHK_STATUS_RET(IsolateAndDeleteNode(node, data_relink_io_map),
                    "Failed to delete node:%s", node->GetName().c_str());
  GELOGI("The output of Node[%s][%s] is unchanged, success to remove unchanged transpose node",
         node->GetName().c_str(), node->GetType().c_str());
  return SUCCESS;
}

bool UnchangedTransposeRemovePass::IsUnchangedTranspose(const NodePtr &node, const OpDescPtr &op_desc) {
  std::vector<int64_t> perm;
  if (!PassUtils::GetPerm(node, kTransposeInputPerm, perm)) {
    return false;
  }

  const auto &input_x_desc = op_desc->MutableInputDesc(kTransposeInputX);
  if (input_x_desc == nullptr) {
    GELOGW("The input desc of node:%s is empty, no need to remove", node->GetName().c_str());
    return false;
  }
  const auto &data_shape = input_x_desc->GetShape();
  if (perm.size() != data_shape.GetDimNum()) {
    // Number of elements in perm should be same as dim_size. Skip if not.
    return false;
  }

  // Unchanged transpose即无效转置算子，需要满足以下两个条件之一：
  // 1.输入x参与转置的维度所对应的shape都为1，即转置前后shape不发生改变
  // 2.输入perm没有维度的变化，即不发生转置，如perm的值为[0,1,2,3]
  bool unchanged = true;
  for (int64_t i = 0; unchanged && i < static_cast<int64_t>(perm.size()); ++i) {
    unchanged = ((data_shape.GetDim(i) == 1) || (i == perm[i]));
  }
  return unchanged;
}

REG_PASS_OPTION("UnchangedTransposeRemovePass").LEVELS(OoLevel::kO3);
}  // namespace ge

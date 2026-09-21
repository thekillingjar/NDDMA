/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/optimize/mem_layout_conflict_optimize/mem_layout_conflict_util.h"
#include "graph/optimize/mem_layout_conflict_optimize/checker/check_register.h"
#include "graph/optimize/mem_layout_conflict_optimize/checker/checker_log.h"
#include "graph/utils/op_type_utils.h"

namespace ge {
Status UserMemoryOutputAndImmutableOutputChecker(CheckFuncContext &context) {
  /*
   * 只需要处理纯静态图的场景
   */
  if (!context.graph_info.is_root_graph_static) {
    return SUCCESS;
  }

  const auto &node_index_out = context.node_a.io_type_ == IOType::kOut ?
                               context.node_a : context.node_b;
  if (OpTypeUtils::IsVarLikeNode(node_index_out.node_ptr_->GetType())) {
    GELOGI("node %s is var like node, skip check.", node_index_out.ToString().c_str());
    return SUCCESS;
  }
  const auto &node_index_in =
    context.node_a.io_type_ == IOType::kIn ?
    context.node_a : context.node_b;
  context.result.insert(node_index_in.node_->GetInDataAnchor(static_cast<int32_t>(node_index_in.index_)));
  GE_MEM_LAYOUT_CONFLICT_LOGI(context, node_index_in);
  return SUCCESS;
}
REGISTER_FUNC(ANCHOR_ATTR_USER_MEMORY_OUTPUT, ANCHOR_ATTR_IMMUTABLE_ADDRESS_OUTPUT,
              UserMemoryOutputAndImmutableOutputChecker);
}  // namespace ge

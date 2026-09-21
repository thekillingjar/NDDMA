/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/shape_optimize/reshape_recovery_pass.h"
#include "common/plugin/ge_make_unique_util.h"
#include "graph/common/trans_op_creator.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/node_utils.h"

namespace ge {
namespace {
NodePtr CreateReshape(const ConstGeTensorDescPtr &src, const ConstGeTensorDescPtr &dst, const ComputeGraphPtr &graph,
                      std::unordered_map<GeShape, NodePtr, GeShapeHasher> &reshape_target_shape_2_const_nodes) {
  static std::atomic_long reshape_num(0);
  auto next_num = reshape_num.fetch_add(1);
  auto reshape_node_name = "Reshape_ReshapeRecoveryPass_" + std::to_string(next_num);
  auto reshape = TransOpCreator::CreateReshapeNodeToGraph(graph, reshape_node_name, *src, *dst,
                                                          reshape_target_shape_2_const_nodes);
  if (reshape == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "Create reshape node failed");
    GELOGE(FAILED, "[New][Node] failed");
    return nullptr;
  }
  return reshape;
}

Status InsertReshapeIfNeed(const NodePtr &node,
                           std::unordered_map<GeShape, NodePtr, GeShapeHasher> &reshape_target_shape_2_const_nodes) {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(node->GetOpDesc());
  for (auto src_anchor : node->GetAllOutDataAnchors()) {
    auto src_tensor = node->GetOpDesc()->GetOutputDescPtr(src_anchor->GetIdx());
    GE_CHECK_NOTNULL(src_tensor);
    for (auto dst_anchor : src_anchor->GetPeerInDataAnchors()) {
      auto dst_node = dst_anchor->GetOwnerNode();
      GELOGD("Try insert reshape between %s[%d] and %s[%d] to keep the shape continues",
             node->GetName().c_str(), src_anchor->GetIdx(), dst_node->GetName().c_str(), dst_anchor->GetIdx());
      GE_CHECK_NOTNULL(dst_node->GetOpDesc());
      auto dst_tensor = dst_node->GetOpDesc()->MutableInputDesc(dst_anchor->GetIdx());
      GE_CHECK_NOTNULL(dst_tensor);
      bool is_dynamic = false;
      const auto &src_tensor_dims = src_tensor->GetShape().GetDims();
      const auto &dst_tensor_dims = dst_tensor->GetShape().GetDims();
      if ((std::any_of(src_tensor_dims.begin(), src_tensor_dims.end(), [](int64_t val) { return val < 0 ; }))
          || (std::any_of(dst_tensor_dims.begin(), dst_tensor_dims.end(), [](int64_t val) { return val < 0; }))) {
        GELOGD("No need to insert reshape node between %s nad %s.", node->GetName().c_str(),
               dst_node->GetName().c_str());
        is_dynamic = true;
      }
      if (dst_node->GetType() == NETOUTPUT && is_dynamic) {
        // NetOutput shape must be continuous when dynamic shape.
        // Otherwise, there may be an error waiting for the shape refresh to time out during execution.
        dst_tensor->SetShape(src_tensor->GetShape());
        continue;
      }
      bool is_need_insert_reshape = src_tensor_dims != dst_tensor_dims && !is_dynamic;
      if (is_need_insert_reshape) {
        auto reshape =
            CreateReshape(src_tensor, dst_tensor, node->GetOwnerComputeGraph(), reshape_target_shape_2_const_nodes);
        GE_CHECK_NOTNULL(reshape);
        auto ret = GraphUtils::InsertNodeBetweenDataAnchors(src_anchor, dst_anchor, reshape);
        if (ret != GRAPH_SUCCESS) {
          REPORT_INNER_ERR_MSG("E19999",
                            "Insert node:%s(%s) between node:%s(%s)(out_index:%d) and node:%s(%s)(out_index:%d) failed",
                            reshape->GetName().c_str(), reshape->GetType().c_str(),
                            node->GetName().c_str(), node->GetType().c_str(), src_anchor->GetIdx(),
                            dst_node->GetName().c_str(), dst_node->GetType().c_str(), dst_anchor->GetIdx());
          GELOGE(INTERNAL_ERROR,
                 "[Insert][Node] %s(%s) between node:%s(%s)(out_index:%d) and node:%s(%s)(out_index:%d) failed",
                 reshape->GetName().c_str(), reshape->GetType().c_str(),
                 node->GetName().c_str(), node->GetType().c_str(), src_anchor->GetIdx(),
                 dst_node->GetName().c_str(), dst_node->GetType().c_str(), dst_anchor->GetIdx());
          return INTERNAL_ERROR;
        }
        GELOGI("Insert reshape[%s] between %s:%d and %s:%d to keep the shape continues", reshape->GetName().c_str(),
               node->GetName().c_str(), src_anchor->GetIdx(), dst_node->GetName().c_str(), dst_anchor->GetIdx());
      }
    }
  }
  return SUCCESS;
}
}  // namespace

Status ReshapeRecoveryPass::Run(ComputeGraphPtr graph) {
  std::unordered_map<GeShape, NodePtr, GeShapeHasher> reshape_target_shape_2_const_nodes;
  for (const auto &node : graph->GetDirectNode()) {
    auto ret = InsertReshapeIfNeed(node, reshape_target_shape_2_const_nodes);
    if (ret != SUCCESS) {
      return ret;
    }
  }
  return SUCCESS;
}

REG_PASS_OPTION("ReshapeRecoveryPass").LEVELS(OoLevel::kO1);
}  // namespace ge

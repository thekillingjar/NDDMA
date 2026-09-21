/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "redundant_slice_remove_pass.h"
#include "graph/debug/ge_op_types.h"
#include "common/checker.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/utils/graph_utils_ex.h"
#include "utils/auto_fuse_config.h"
#include "utils/autofuse_utils.h"
#include "lowering/asc_lowerer/loop_common.h"
#include "utils/graph_utils.h"

namespace {
const std::unordered_set<std::string> kSliceOpTypes = {"Slice", "SliceD"};
}  // namespace

namespace ge {
graphStatus RedundantSliceRemovePass::Run(const ComputeGraphPtr &graph, bool &changed) const {
  GE_CHECK_NOTNULL(graph);

  auto direct_nodes = graph->GetDirectNode();
  for (auto it = direct_nodes.rbegin(); it != direct_nodes.rend(); ++it) {
    ge::NodePtr &current_node = *it;
    GE_ASSERT_NOTNULL(current_node);
    if (IsSliceNoOp(current_node)) {
      GELOGD("Slice node [%s] is redundant, will be deleted.", current_node->GetNamePtr());
      GE_ASSERT_SUCCESS(GraphUtils::IsolateNode(current_node, {0}), "Failed to delete single useless Slice node: %s",
                        current_node->GetNamePtr());
      GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveJustNode(graph, current_node),
                              "[Remove][JustNode] failed, graph:%s, node:%s.", graph->GetName().c_str(),
                              current_node->GetNamePtr());
      changed = true;
    }
  }
  return SUCCESS;
}

bool RedundantSliceRemovePass::IsSliceNoOp(const ge::NodePtr &node) {
  if (kSliceOpTypes.count(node->GetType()) == 0UL) {
    return false;
  }
  const auto &input_shape = node->GetOpDescBarePtr()->GetInputDesc(0).GetOriginShape();
  vector<int64_t> begins = {};
  vector<int64_t> sizes = {};
  graphStatus offset_res = ge::AutofuseUtils::GetListIntByInputOrAttr(node, begins, "offsets", "offsets");
  graphStatus size_res = ge::AutofuseUtils::GetListIntByInputOrAttr(node, sizes, "size", "size");
  if (offset_res != SUCCESS || size_res != SUCCESS) {
    GELOGD("Cannot get begin and end from slice node [%s], may be dynamic.", node->GetNamePtr());
    return false;
  }
  GE_WARN_ASSERT(begins.size() == input_shape.GetDimNum());
  GE_WARN_ASSERT(begins.size() == sizes.size());
  for (size_t i = 0UL; i < begins.size(); ++i) {
    // size=-1 表示从offset到该维度末尾，等价于取全部
    bool is_full_slice = (sizes[i] == -1) || (sizes[i] == input_shape.GetDim(i));
    if (begins[i] != 0 || !is_full_slice) {
      return false;
    }
  }
  return true;
}
}  // namespace ge

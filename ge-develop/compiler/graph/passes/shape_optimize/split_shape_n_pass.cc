/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/shape_optimize/split_shape_n_pass.h"

#include <vector>
#include "common/plugin/ge_make_unique_util.h"
#include "graph/utils/graph_utils.h"
#include "framework/common/debug/ge_log.h"
#include "graph/debug/ge_attr_define.h"
#include "graph_metadef/common/ge_common/util.h"

namespace {
const std::string kDefaultUnknownSuffix = "_unknown";
const std::string kDefaultKnownSuffix = "_known";
}  // namespace

namespace ge {
Status SplitShapeNPass::Run(NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  if (node->GetType() != SHAPEN) {
    return SUCCESS;
  }
  OpDescPtr op_desc_ptr = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc_ptr);
  Clear();
  for (size_t i = 0U; i < op_desc_ptr->GetAllInputsSize(); i++) {
    const GeShape &input_shape = op_desc_ptr->GetInputDesc(i).GetShape();
    if (input_shape.IsUnknownShape()) {
      unknown_index_.emplace_back(static_cast<int32_t>(i));
      GE_CHECK_NOTNULL(op_desc_ptr->GetInputDescPtr(i));
      GE_CHECK_NOTNULL(op_desc_ptr->GetOutputDescPtr(i));
      unknown_input_desc_.emplace_back(op_desc_ptr->GetInputDescPtr(i));
      unknown_output_desc_.emplace_back(op_desc_ptr->GetOutputDescPtr(i));
    } else {
      known_index_.emplace_back(static_cast<int32_t>(i));
      GE_CHECK_NOTNULL(op_desc_ptr->GetInputDescPtr(i));
      GE_CHECK_NOTNULL(op_desc_ptr->GetOutputDescPtr(i));
      known_input_desc_.emplace_back(op_desc_ptr->GetInputDescPtr(i));
      known_output_desc_.emplace_back(op_desc_ptr->GetOutputDescPtr(i));
    }
  }
  if ((unknown_output_desc_.empty()) || (known_output_desc_.empty())) {
    return SUCCESS;
  }
  auto ret = SplitShapeN(node);
  if (ret != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "SplitShapeN node:%s(%s) failed",
                      node->GetName().c_str(), node->GetType().c_str());
    GELOGE(FAILED, "[Split][ShapeN] node:%s(%s) failed",
           node->GetName().c_str(), node->GetType().c_str());
    return FAILED;
  }
  return SUCCESS;
}
Status SplitShapeNPass::SplitShapeN(NodePtr &node) {
  ComputeGraphPtr graph = node->GetOwnerComputeGraph();
  GE_CHECK_NOTNULL(graph);
  auto ret_known = RelinkAnchors(graph, node, kDefaultKnownSuffix);
  auto ret_unknown = RelinkAnchors(graph, node, kDefaultUnknownSuffix);
  if ((ret_known != SUCCESS) || (ret_unknown != SUCCESS)) {
    return FAILED;
  }
  GE_CHK_STATUS_RET(IsolateAndDeleteNode(node, {}));

  return SUCCESS;
}

Status SplitShapeNPass::RelinkAnchors(const ComputeGraphPtr &graph, const NodePtr &node, const std::string &desc_name) {
  bool default_name = (desc_name == kDefaultKnownSuffix);
  const auto &output_desc = default_name ? known_output_desc_ : unknown_output_desc_;
  const auto &input_desc = default_name ? known_input_desc_ : unknown_input_desc_;
  const auto &index = default_name ? known_index_ : unknown_index_;
  OpDescPtr op_desc_ptr = MakeShared<OpDesc>(node->GetName() + desc_name, SHAPEN);
  GE_CHECK_NOTNULL(op_desc_ptr);
  for (const auto &out_opdesc : output_desc) {
    op_desc_ptr->AddOutputDesc(*out_opdesc);
  }
  for (size_t i = 0U; i < input_desc.size(); ++i) {
    op_desc_ptr->AddInputDesc("x" + std::to_string(i), *input_desc[i]);
  }
  NodePtr new_node = graph->InsertNode(node, op_desc_ptr);
  GE_CHECK_NOTNULL(new_node);
  (void)AttrUtils::SetStr(new_node->GetOpDesc(), ATTR_NAME_SPLIT_SHAPEN_ORIGIN_NAME, node->GetName());
  GELOGI("Replace node:%s(%s) by node:%s(%s)", node->GetName().c_str(), node->GetType().c_str(),
         new_node->GetName().c_str(), new_node->GetType().c_str());
  GE_CHECK_NOTNULL(new_node);
  if (GraphUtils::ReplaceNodeAnchors(new_node, node, index, index) != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Replace node:%s(%s) by node:%s(%s) failed",
                      node->GetName().c_str(), node->GetType().c_str(),
                      new_node->GetName().c_str(), new_node->GetType().c_str());
    GELOGE(FAILED, "[Replace][Node] %s(%s) by node:%s(%s) failed",
           node->GetName().c_str(), node->GetType().c_str(),
           new_node->GetName().c_str(), new_node->GetType().c_str());
    return FAILED;
  }
  AddRePassNodesWithInOut(new_node);
  return SUCCESS;
}

void SplitShapeNPass::Clear() {
  known_index_.clear();
  unknown_index_.clear();
  known_input_desc_.clear();
  known_output_desc_.clear();
  unknown_input_desc_.clear();
  unknown_output_desc_.clear();
}

REG_PASS_OPTION("SplitShapeNPass").LEVELS(OoLevel::kO1);
}  // namespace ge

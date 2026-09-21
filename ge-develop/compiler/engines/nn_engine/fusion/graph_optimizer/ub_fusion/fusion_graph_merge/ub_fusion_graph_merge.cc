/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/ub_fusion/fusion_graph_merge/ub_fusion_graph_merge.h"
#include "common/fusion_statistic/fusion_statistic_writer.h"
#include "graph/utils/op_desc_utils_ex.h"

namespace fe {
UBFusionGraphMerge::UBFusionGraphMerge(const std::string &scope_attr, const GraphCommPtr &graph_comm_ptr)
    : FusionGraphMerge(scope_attr, graph_comm_ptr) {}

UBFusionGraphMerge::~UBFusionGraphMerge() {}

void UBFusionGraphMerge::PostUbFusion(const ge::NodePtr &fused_node,
    const std::map<string, BufferFusionPassRegistry::CreateFn> &create_fns) const {
  auto op = fused_node->GetOpDesc();
  string pass_name;
  if (ge::AttrUtils::GetStr(op, kPassNameUbAttr, pass_name) &&
      !pass_name.empty()) {
    auto iter = create_fns.find(pass_name);
    if (iter != create_fns.end()) {
      std::unique_ptr<BufferFusionPassBase> pass = std::unique_ptr<BufferFusionPassBase>(iter->second());
      (void)pass->PostFusion(fused_node);
    }
  }
}

Status UBFusionGraphMerge::AfterMergeFusionGraph(ge::ComputeGraph &graph) {
  // iterate all nodes in the graph to find fused BNTrainingUpdate nodes
  BufferFusionPassRegistry &instance = BufferFusionPassRegistry::GetInstance();
  std::map<string, BufferFusionPassRegistry::CreateFn> create_fns;
  if (graph_comm_ptr_->GetEngineName() == AI_CORE_NAME) {
    create_fns = instance.GetCreateFnByType(BUILT_IN_AI_CORE_BUFFER_FUSION_PASS);
  } else if (graph_comm_ptr_->GetEngineName() == VECTOR_CORE_NAME) {
    create_fns = instance.GetCreateFnByType(BUILT_IN_VECTOR_CORE_BUFFER_FUSION_PASS);
  }

  for (const auto &node : graph.GetDirectNode()) {
    PostUbFusion(node, create_fns);
    bool flag = (node->GetType() != AIPP) || !node->GetOpDesc()->HasAttr(GetScopeAttr());
    if (flag) {
      continue;
    }
    int64_t scope_id = -1;
    (void)ge::AttrUtils::GetInt(node->GetOpDesc(), GetScopeAttr(), scope_id);
    // fused BNTrainingUpdate node's scope_id should be positive
    if (scope_id > 0) {
      FE_LOGD("Get fused node:[%s] scope_id:[%ld].", node->GetOpDesc()->GetName().c_str(), scope_id);
      if (node->GetType() == AIPP) {
        auto op_desc = node->GetOpDesc();
        ge::OpDescUtilsEx::SetType(op_desc, CONV2D);
        continue;
      }
    }
  }
  return SUCCESS;
}
}

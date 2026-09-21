/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <set>
#include "common/aicore_util_types.h"
#include "common/graph_tuner_errorcode.h"
#include "graph/compute_graph.h"
#include "graph/utils/attr_utils.h"

namespace tune {
extern "C" Status L1FusionOptimize(ge::ComputeGraph &graph, const fe::AOEOption options) {
  Status ret = NO_FUSION_STRATEGY;
  ge::AttrUtils::GetInt(graph, "_l1_fusion_ret", ret);
  for (const ge::NodePtr &node : graph.GetDirectNode()) {
    if (node == nullptr) {
      continue;
    }
    int64_t l1_scope_id = 0;
    if (ge::AttrUtils::GetInt(node->GetOpDesc(), "_llt_l1_scope", l1_scope_id)) {
      ge::AttrUtils::SetInt(node->GetOpDesc(), "_l1_fusion_scope", l1_scope_id);
      ge::AttrUtils::SetBool(node->GetOpDesc(), "need_re_precompile", true);
      ret = HIT_FUSION_STRATEGY;
    }
  }
  return ret;
}
extern "C" Status L1Recovery(ge::ComputeGraph &graph, const std::vector<ge::NodePtr> &nodesUbFailed,
                             std::vector<ge::NodePtr> *nodesRecover, std::vector<ge::NodePtr> *nodesNeedToDelete) {
  return SUCCESS;
}
extern "C" Status L2FusionPreOptimize(ge::ComputeGraph &graph, const fe::AOEOption options,
                                      const std::vector<std::string> &disableUbPass,
                                      std::vector<fe::PassChangeInfo> &ubPassChangeInfo) {
  bool is_pass_change = false;
  if (ge::AttrUtils::GetBool(graph, "_is_pass_changed", is_pass_change) && is_pass_change) {
    fe::PassChangeInfo pass1{"TbeConvStridedwriteFusionPass", 10, 5};
    fe::PassChangeInfo pass3{"MatmulConfusiontransposeUbFusion", 3, 10};
    ubPassChangeInfo.push_back(pass1);
    ubPassChangeInfo.push_back(pass3);
    return HIT_FUSION_STRATEGY;
  }
  return SUCCESS;
}
extern "C" Status L2FusionOptimize(ge::ComputeGraph &graph, const fe::AOEOption options) {
  Status ret = NO_FUSION_STRATEGY;
  ge::AttrUtils::GetInt(graph, "_l2_fusion_ret", ret);
  std::set<int64_t> scope_id_vec;
  for (const ge::NodePtr &node : graph.GetDirectNode()) {
    if (node == nullptr) {
      continue;
    }
    int64_t scope_id = 0;
    if (ge::AttrUtils::GetInt(node->GetOpDesc(), "_llt_fusion_scope", scope_id)) {
      ge::AttrUtils::SetInt(node->GetOpDesc(), "fusion_scope", scope_id);
      scope_id_vec.emplace(scope_id);
      ret = HIT_FUSION_STRATEGY;
    }
  }
  for (const ge::NodePtr &node : graph.GetDirectNode()) {
    if (node == nullptr) {
      continue;
    }
    int64_t scope_id = 0;
    if (ge::AttrUtils::GetInt(node->GetOpDesc(), "fusion_scope", scope_id)) {
      if (scope_id_vec.count(scope_id)) {
        ge::AttrUtils::SetBool(node->GetOpDesc(), "need_re_precompile", true);
      }
    }
  }
  return ret;
}
extern "C" Status L2Recovery(ge::ComputeGraph &graph, const std::vector<ge::NodePtr> &nodesUbFailed,
                             std::vector<ge::NodePtr> *nodesRecover, std::vector<ge::NodePtr> *nodesNeedToDelete) {
  for (const ge::NodePtr &node : nodesUbFailed) {
    if (node == nullptr) {
      continue;
    }
    nodesRecover->push_back(node);
    nodesNeedToDelete->push_back(node);
    bool ub_compile_fail = false;
    if (ge::AttrUtils::GetBool(node->GetOpDesc(), "_ub_compile_fail", ub_compile_fail) && ub_compile_fail) {
      node->GetOpDesc()->DelAttr("_ub_compile_fail");
      node->GetOpDesc()->DelAttr("fusion_scope");
    }
  }
  return SUCCESS;
}
extern "C" Status LxFusionFinalize(ge::ComputeGraph &graph) {
  bool is_fail = false;
  if (ge::AttrUtils::GetBool(graph, "_is_lx_fusion_finalize_failed", is_fail) && is_fail) {
    return FAILED;
  }
  return SUCCESS;
}

}

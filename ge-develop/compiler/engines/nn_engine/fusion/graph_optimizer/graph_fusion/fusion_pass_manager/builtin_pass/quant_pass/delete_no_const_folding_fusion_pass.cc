/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/quant_pass/delete_no_const_folding_fusion_pass.h"
#include <string>
#include "common/fe_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

namespace fe {
static const string OP_WEIGHTQUANT = "AscendWeightQuant";
static const string PATN_WEIGHT_QUANT = "weight_quant";

vector<FusionPattern *> DeleteNoConstFolding::DefinePatterns() {
  vector<FusionPattern *> patterns;
  /*
   * ================================pattern0===================================
   *
   *  --> AscendWeightQuant-->
   *
   * ===========================================================================
   */
  FusionPattern *pattern0 = new (std::nothrow) FusionPattern("deleteNoConstFoldingFusion0");
  FE_CHECK(pattern0 == nullptr,
      REPORT_FE_ERROR("[GraphOpt][TfTagNoCstFold][DefPtn] Failed to create new FusionPattern object!"), return patterns);
  /* above defines ops that we need */
  pattern0->AddOpDesc(PATN_WEIGHT_QUANT, {OP_WEIGHTQUANT}).SetOutput(PATN_WEIGHT_QUANT);
  patterns.push_back(pattern0);

  return patterns;
}

Status DeleteNoConstFolding::Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &fusion_nodes) {
  (void)graph;
  (void)fusion_nodes;
  for (Mapping::iterator itr = mapping.begin(); itr != mapping.end(); ++itr) {
    FE_CHECK(itr->second.size() == 0, FE_LOGE("no node in mapping!"), return FAILED);
    auto node = itr->second[0];
    FE_CHECK_NOTNULL(node);
    if (node->GetOpDesc()->HasAttr(ge::ATTR_NO_NEED_CONSTANT_FOLDING)) {
      if (node->GetOpDesc()->DelAttr(ge::ATTR_NO_NEED_CONSTANT_FOLDING) != ge::GRAPH_SUCCESS) {
        return FAILED;
      }
      FE_LOGD("delete attr[%s] of node[%s] successfully.", ge::ATTR_NO_NEED_CONSTANT_FOLDING.c_str(),
              node->GetName().c_str());
    }
  }

  return SUCCESS;
}
REG_PASS("DeleteNoConstFolding", BUILT_IN_DELETE_NO_CONST_FOLDING_GRAPH_PASS,
         DeleteNoConstFolding, SINGLE_SCENE_OPEN | FE_PASS | FORBIDDEN_CLOSE);
}  // namespace fe

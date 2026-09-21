/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_GRAPH_FUSION_FUSION_PASS_MANAGER_BUILTIN_PASS_NODE_OPTIMIZE_SPLIT_CONV_CONCAT_FUSION_PASS_H_
#define FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_GRAPH_FUSION_FUSION_PASS_MANAGER_BUILTIN_PASS_NODE_OPTIMIZE_SPLIT_CONV_CONCAT_FUSION_PASS_H_

#include <vector>
#include "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/node_optimize/checker/concat_optimize_checker.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/node_optimize/checker/split_optimize_checker.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/node_optimize/node_optimize_pass_base.h"
#include "common/math_util.h"

namespace fe {
struct QuantCmpAttr {
  float scale = 1;
  float offset = 0;
  bool sqrt_mode = false;
  ge::DataType data_type;
  bool operator== (const QuantCmpAttr& quant) const {
    if (!FloatEqual(quant.scale, scale) || !FloatEqual(quant.offset, offset) ||
        (quant.sqrt_mode != sqrt_mode) ||
        (quant.data_type !=  data_type)) {
      return false;
    }
    return true;
  }
};
enum class FusionType {
  Reserved = 0,
  OnlyConv,
  ConvAndDequant,
  HasQuant,
};

struct SubGraphStructure {
  bool is_quant;
  bool is_only_conv;
  bool is_conv_and_dequant;
  FusionType fusion_type;
  SubGraphStructure() : is_quant(false), is_only_conv(false),
                        is_conv_and_dequant(false), fusion_type(FusionType::Reserved) {}
};

class SplitConvConcatFusionPass : public NodeOptimizePassBase {
 public:
  Status DoFusion(ge::ComputeGraph &graph, ge::NodePtr &concat, vector<ge::NodePtr> &fusion_nodes) override;
  vector<string> GetNodeTypes() override;
  string GetPatternName() override;

 private:
  Status PatternConcatSplit(ge::NodePtr &concat, ge::NodePtr &split_node,
                            vector<ge::NodePtr> &vector_quant, vector<ge::NodePtr> &vector_dequant);
  Status CheckOutputSingleRef(ge::NodePtr &concat_node);
  Status SetStridedReadAttr(const ge::NodePtr &split_node) const;
  ge::InDataAnchorPtr PatternPrevConv2dWithQuant(ge::OutDataAnchorPtr out_anchor, ge::NodePtr &quant,
                                                 ge::NodePtr &dequant);
  Status FusionSplit(vector<ge::NodePtr> &vector_quant, ge::ComputeGraph &graph, ge::NodePtr &split_node);
  Status FusionConcat(vector<ge::NodePtr> &vector_dequant, ge::ComputeGraph &graph, ge::NodePtr &concat);
  bool IsSameQuant(const vector<ge::NodePtr> &vector_quant) const;
  ConcatOptimizeChecker concat_optimize_checker;
  SplitOptimizeChecker split_optimize_checker;
};

}  // namespace fe
#endif  // FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_GRAPH_FUSION_FUSION_PASS_MANAGER_BUILTIN_PASS_NODE_OPTIMIZE_SPLIT_CONV_CONCAT_FUSION_PASS_H_

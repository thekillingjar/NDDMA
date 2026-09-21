/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_WEIGHT_COMPRESS_JUDGE_H_
#define FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_WEIGHT_COMPRESS_JUDGE_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "common/fe_inner_error_codes.h"
#include "common/fe_utils.h"
#include "common/graph/fe_graph_utils.h"
#include "common/optimizer/optimize_utility.h"
#include "common/util/op_info_util.h"
#include "graph/compute_graph.h"

namespace fe {
class WeightCompressJudge {
 public:
  static Status CompressTypeJudge(ge::OptimizeUtility *const optimize_utility, ge::ComputeGraph &graph);

 private:
  static Status PreConstFolding(ge::OptimizeUtility *const optimize_utility, const ge::NodePtr &host_node);

  static bool GetConstFoldingNodes(const ge::NodePtr &node, std::vector<ge::NodePtr> &const_folding_nodes);

  static ge::NodePtr GetSpecificNode(const ge::NodePtr &node, const int32_t &idx, const std::string &op_type);

  static bool IsSupportTwoCompressTypes(const std::string &op_type);

  static WeightCompressType CompareCompressType(const ge::NodePtr &weight_node,
                                                const bool &is_support_two_compress_types);

  static float DoCompressWeights(char* input, const size_t &input_size, const WeightCompressType &compress_type);

  static size_t ComputeFractalSize(const size_t &weight_size);

  static WeightCompressType GetFinalCompressType(const float &low_sparse_compress_ratio,
                                                 const float &high_sparse_compress_ratio);

  static bool IsMeetCompressRatioThreshold(const float &compress_ratio);
};
}  // namespace fe
#endif

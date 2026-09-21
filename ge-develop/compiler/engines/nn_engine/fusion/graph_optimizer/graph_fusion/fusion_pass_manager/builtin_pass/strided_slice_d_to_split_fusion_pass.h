/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_FUSION_GRAPH_OPTIMIZER_GRAPH_FUSION_FUSION_PASS_MANAGER_BUILTIN_PASS_NODE_OPTIMIZE_STRIDED_SLICE_D_TO_SPLIT_FUSION_PASS_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_FUSION_GRAPH_OPTIMIZER_GRAPH_FUSION_FUSION_PASS_MANAGER_BUILTIN_PASS_NODE_OPTIMIZE_STRIDED_SLICE_D_TO_SPLIT_FUSION_PASS_H_

#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
#include <vector>
#include "common/fe_log.h"

namespace fe {
struct SplitInfo {
  int64_t split_dim;
  int64_t num_split;
  int64_t split_out_anchor;
};

class StridedSliceDToSplitFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &new_nodes) override;

 private:
  static std::vector<size_t> GetMaskAxis(const int64_t &mask_value, const size_t &dims_size);
  static Status ModifyBeginAndEnd(const ge::OpDescPtr &strided_slice_d_desc, const std::vector<int64_t> &ori_shape,
                                  std::vector<int64_t> &begins, std::vector<int64_t> &ends);
  static bool CheckEvenlySlice(const ge::OpDescPtr &strided_slice_d_desc, const std::vector<int64_t> &ori_shape,
                               std::vector<int64_t> &begins, std::vector<int64_t> &ends, SplitInfo &split_info);
  static bool CheckCommonCondition(const ge::NodePtr &strided_slice_d_node, const ge::OpDescPtr &strided_slice_d_desc,
                                   SplitInfo &split_info);
  bool CheckSplitDSupported(const ge::OpDescPtr &strided_slice_d_desc, const SplitInfo &split_info) const;
  Status UpdateGraph(ge::ComputeGraph &graph, ge::NodePtr &strided_slice_d_node, ge::NodePtr &split_d_node,
                     const SplitInfo &split_info) const;
};
}  // namespace fe
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_FUSION_GRAPH_OPTIMIZER_GRAPH_FUSION_FUSION_PASS_MANAGER_BUILTIN_PASS_NODE_OPTIMIZE_STRIDED_SLICE_D_TO_SPLIT_FUSION_PASS_H_

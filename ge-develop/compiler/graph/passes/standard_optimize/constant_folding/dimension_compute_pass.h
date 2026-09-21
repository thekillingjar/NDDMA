/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_DIMENSION_COMPUTE_PASS_H_
#define GE_GRAPH_PASSES_DIMENSION_COMPUTE_PASS_H_

#include "graph/passes/standard_optimize/constant_folding/potential_folding_pass.h"

namespace ge {
class DimensionComputePass : public PotentialFoldingPass {
 public:
  explicit DimensionComputePass(bool need_fold = true) : need_fold_(need_fold) {}
  ~DimensionComputePass() override = default;
  bool NeedIgnorePass(const NodePtr &node) override;
  bool NeedFold() const override;
  Status ComputePotentialWeight(NodePtr &node, std::vector<GeTensorPtr> &outputs) override;
  std::string GetPassName() const override;
 private:
  // indicate pass provide folding func or not, default : need_fold
  bool need_fold_ = true;
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_DIMENSION_COMPUTE_PASS_H_

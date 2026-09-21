/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_CONSTANT_FOLDING_PASS_H_
#define GE_GRAPH_PASSES_CONSTANT_FOLDING_PASS_H_

#include <map>
#include <vector>

#include "graph/passes/standard_optimize/constant_folding/potential_folding_pass.h"
#include "external/ge_common/ge_api_types.h"

namespace ge {
class ConstantFoldingPass : public PotentialFoldingPass {
 public:
  bool NeedIgnorePass(const NodePtr &node) override;
  bool NeedFold() const override;
  Status ComputePotentialWeight(NodePtr &node, std::vector<GeTensorPtr> &outputs) override;
  std::string GetPassName() const override;

  const std::map<std::string, std::pair<std::uint64_t, uint64_t>> &GetGeConstantFoldingPerfStatistic() const;
  const std::map<std::string, std::pair<std::uint64_t, uint64_t>> &GetOpConstantFoldingPerfStatistic() const;

  static Status RunOpKernel(const NodePtr &node, const std::vector<ConstGeTensorPtr> &inputs,
                            std::vector<GeTensorPtr> &outputs);
  static Status ComputeWithHostCpuKernel(const NodePtr &node, const std::vector<ConstGeTensorPtr> &inputs,
                                         std::vector<GeTensorPtr> &outputs);
 private:
  Status ComputeWithBuiltInKernel(NodePtr &node, const std::vector<ConstGeTensorPtr> &inputs,
                                  std::vector<GeTensorPtr> &outputs);
  void CollectCostTimeOfGeConstantFolding(const NodePtr &node, uint64_t start_time);
  void CollectCostTimeOfOpConstantFolding(const NodePtr &node, uint64_t start_time);
  std::map<std::string, std::pair<std::uint64_t, uint64_t>> statistic_of_op_constant_folding_;
  std::map<std::string, std::pair<std::uint64_t, uint64_t>> statistic_of_ge_constant_folding_;
  bool need_fold_ = true;
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_CONSTANT_FOLDING_PASS_H_

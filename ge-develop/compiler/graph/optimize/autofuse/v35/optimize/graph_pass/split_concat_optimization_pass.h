/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_SPLIT_CONCAT_OPTIMIZATION_PASS_H
#define AUTOFUSE_SPLIT_CONCAT_OPTIMIZATION_PASS_H

#include "optimize/graph_pass/base_graph_pass.h"
namespace optimize {
class SplitConcatOptimizationPass final : public BaseGraphPass {
public:
  SplitConcatOptimizationPass() = default;
  ~SplitConcatOptimizationPass() override = default;
  Status RunPass(ge::AscGraph &graph) override;

 private:
  static void FindSplitAndConcatNodes(const ascir::HintGraph &owner_graph, std::vector<ge::AscNodePtr> &split_nodes,
                                      std::vector<ge::AscNodePtr> &concat_nodes);
  static Status OptimizeOutSplit(ascir::HintGraph &owner_graph);
  static Status OptimizeOutConcat(ascir::HintGraph &owner_graph);
};
}  // namespace optimize

#endif  // AUTOFUSE_SPLIT_CONCAT_OPTIMIZATION_PASS_H

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_POTENTIAL_CONST_TAKEN_EFFECT_PASS_H_
#define GE_POTENTIAL_CONST_TAKEN_EFFECT_PASS_H_

#include "graph/passes/standard_optimize/constant_folding/folding_pass.h"

namespace ge {
class PotentialConstTakenEffectPass : public FoldingPass {
 public:
  Status Run(ge::NodePtr &node) override;
  Status OnFinishGraph(ComputeGraphPtr &root_graph, std::vector<NodePtr> &node_to_be_repass) override;
};
}  // namespace ge

#endif  // GE_POTENTIAL_CONST_TAKEN_EFFECT_PASS_H_

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTIMIZE_PLATFORM_V2_GRAPH_PASS_CONTINUES_BROADCAST_OPTIMIZATION_H
#define OPTIMIZE_PLATFORM_V2_GRAPH_PASS_CONTINUES_BROADCAST_OPTIMIZATION_H
#include "optimize/graph_pass/base_graph_pass.h"
namespace optimize {
class ContinuesBroadcastOptimizationPass final : public BaseGraphPass {
 public:
  ContinuesBroadcastOptimizationPass() = default;
  Status RunPass(ge::AscGraph &graph) override;
  ~ContinuesBroadcastOptimizationPass() override = default;
};
}  // namespace optimize

#endif  // OPTIMIZE_PLATFORM_V2_GRAPH_PASS_CONTINUES_BROADCAST_OPTIMIZATION_H

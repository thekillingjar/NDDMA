/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_LOWERING_PASS_PASS_H_
#define AIR_CXX_RUNTIME_V2_LOWERING_PASS_PASS_H_
#include "graph/compute_graph.h"
#include "graph/fast_graph/execute_graph.h"
#include "runtime/gert_api.h"
namespace gert {
namespace bg {
class Pass {
 public:
  virtual ~Pass() = default;
  virtual ge::graphStatus Run(ge::ExecuteGraph *const graph, bool &changed) = 0;

  const LoweringOption &GetLoweringOption() const {
    return optimize_option_;
  }
  void SetLoweringOption(const LoweringOption &optimize_option) {
    optimize_option_ = optimize_option;
  }

 private:
  LoweringOption optimize_option_;
};
}  // namespace bg
}  // namespace gert

#endif  // AIR_CXX_RUNTIME_V2_LOWERING_PASS_PASS_H_

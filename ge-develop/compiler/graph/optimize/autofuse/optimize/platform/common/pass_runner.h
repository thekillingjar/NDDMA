/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPTIMIZE_PLATFORM_COMMON_PASS_RUNNER_H
#define OPTIMIZE_PLATFORM_COMMON_PASS_RUNNER_H

#include <vector>
#include <memory>
#include "optimize/graph_pass/base_graph_pass.h"

namespace optimize {
class BasePassRunner {
 protected:
  std::vector<std::unique_ptr<BaseGraphPass>> passes_;

  template <class PassT>
  void RegisterPass() {
    passes_.emplace_back(std::make_unique<PassT>());
  }

 public:
  Status RunPasses(ge::AscGraph &graph) const {
    for (const auto &pass : passes_) {
      pass->RunPass(graph);
    }
    return ge::SUCCESS;
  }
};
}  // namespace optimize

#endif  // OPTIMIZE_PLATFORM_COMMON_PASS_RUNNER_H

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_LOWERING_PASS_BASE_OPTIMIZER_H_
#define AIR_CXX_RUNTIME_V2_LOWERING_PASS_BASE_OPTIMIZER_H_
#include <memory>
#include <vector>
#include "pass.h"
#include "graph/fast_graph/execute_graph.h"

namespace gert {
namespace bg {
class BaseOptimizer {
 public:
  BaseOptimizer &AddPass(std::string name, std::unique_ptr<Pass> pass);
  BaseOptimizer &AddOncePass(std::string name, std::unique_ptr<Pass> pass);
  ge::graphStatus Run(ge::ExecuteGraph *const graph, bool &changed);

 private:
  std::vector<std::pair<std::string, std::unique_ptr<Pass>>> passes_;
  std::vector<std::pair<std::string, std::unique_ptr<Pass>>> once_passes_;
};
}  // namespace bg
}  // namespace gert

#endif  // AIR_CXX_RUNTIME_V2_LOWERING_PASS_BASE_OPTIMIZER_H_

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_GRAPH_OPTIMIZER_GRAPH_PASS_H
#define INC_GRAPH_OPTIMIZER_GRAPH_PASS_H

#include "pass.h"
#include "graph/compute_graph.h"

namespace aicpu {
class GraphPass : public Pass<ge::ComputeGraph> {
 public:
  virtual Status Run(ge::ComputeGraph &graph) override = 0;
};
}  // namespace aicpu

#endif  // INC_GRAPH_OPTIMIZER_GRAPH_PASS_H
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_COMMON_OPTIMIZER_OPTIMIZE_UTILITY_H_
#define INC_COMMON_OPTIMIZER_OPTIMIZE_UTILITY_H_

#include "common/ge_common/ge_inner_error_codes.h"
#include "graph/compute_graph.h"

namespace ge {
class OptimizeUtility {
 public:
  virtual ~OptimizeUtility() = default;

  // Deprecated: will delete later. Graph infershape util
  virtual Status InferShape(ComputeGraph &compute_graph) {
    (void)compute_graph;
    return SUCCESS;
  }

  // Graph infershape util
  virtual Status InferShape(const ComputeGraphPtr &compute_graph) = 0;

  // Mlti Dims and pre/post process
  virtual Status MultiDimsProcess(const ComputeGraphPtr &compute_graph) {
    (void)compute_graph;
    return SUCCESS;
  }

  // Constant folding
  virtual Status ConstantFolding(NodePtr &node) {
    (void)node;
    return SUCCESS;
  }
};
}  // namespace ge
#endif  // INC_COMMON_OPTIMIZER_OPTIMIZE_UTILITY_H_

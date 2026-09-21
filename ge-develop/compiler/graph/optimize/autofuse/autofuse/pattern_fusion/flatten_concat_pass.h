/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_FLATTEN_CONCAT_PASS_H_
#define AUTOFUSE_FLATTEN_CONCAT_PASS_H_

#include <functional>
#include <memory>
#include <vector>

#include "graph/node.h"

namespace ge {
class FlattenConcatPass {
 public:
  graphStatus Run(const ComputeGraphPtr &graph) const;
  static graphStatus CanFlatten(const NodePtr &node, size_t concat_dim, size_t num_inputs);
  static graphStatus ResolveConcatDim(const NodePtr &concat_node, size_t &concat_dim);
};
}  // namespace ge

#endif  // AUTOFUSE_FLATTEN_CONCAT_PASS_H_

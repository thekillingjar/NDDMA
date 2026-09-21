/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_NODE_SHAPE_PROPAGATOR_H
#define AIR_CXX_NODE_SHAPE_PROPAGATOR_H

#include "shape_propagator.h"

namespace ge {
class OpDesc;
namespace hybrid {
struct NodeShapePropagator {
  NodeShapePropagator() = default;
  void CreatePropagator(const TensorDescHolder &holder, const TensorDescObserver &observer);
  Status DoPropagate() const;

private:
  std::vector<ShapePropagator> shape_propagators_;
};

} // namespace hybrid
} // namespace ge
#endif

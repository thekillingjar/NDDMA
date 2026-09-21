/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hybrid/model/infer/node_shape_propagator.h"
#include "hybrid/model/infer/shape_propagator.h"
#include "common/debug/ge_log.h"
#include "common/debug/log.h"

namespace ge {
namespace hybrid {

void NodeShapePropagator::CreatePropagator(const TensorDescHolder &holder, const TensorDescObserver &observer) {
  const ShapePropagator new_propagator(holder);
  for (auto& propagator: shape_propagators_) {
    if (propagator == new_propagator) {
      return propagator.PropagateTo(observer);
    }
  }
  shape_propagators_.emplace_back(holder);
  shape_propagators_.back().PropagateTo(observer);
}

Status NodeShapePropagator::DoPropagate() const {
  for (auto &cp_tensor : shape_propagators_) {
    GE_CHK_STATUS_RET_NOLOG(cp_tensor.DoPropagate());
  }
  return SUCCESS;
}
} // namespace hybrid
} // namespace ge

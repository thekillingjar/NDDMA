/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hybrid/model/infer/shape_propagator.h"

#include "common/debug/log.h"

namespace ge {
namespace hybrid {

Status ShapePropagator::DoPropagate() const {
  for (auto observer : observers_) {
    GE_CHK_STATUS_RET_NOLOG(observer.OnChanged(tensor_desc_holder_.Output()));
  }
  return SUCCESS;
}

bool ShapePropagator::operator==(const ShapePropagator &left) const {
  return this->tensor_desc_holder_== left.tensor_desc_holder_;
}

void ShapePropagator::PropagateTo(const TensorDescObserver &observer) {
  observers_.push_back(observer);
}
} // namespace hybrid
} // namespace ge

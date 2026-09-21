/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_HYBRID_MODEL_INFER_SHAPE_PROPAGATOR_H
#define GE_HYBRID_MODEL_INFER_SHAPE_PROPAGATOR_H

#include "hybrid/model/infer/tensor_desc_observer.h"

namespace ge {
namespace hybrid {
struct ShapePropagator {
  explicit ShapePropagator(const TensorDescHolder &holder) : tensor_desc_holder_(holder) {}
  Status DoPropagate() const;
  void PropagateTo(const TensorDescObserver &observer);

public:
  bool operator==(const ShapePropagator &left) const;

private:
  const TensorDescHolder tensor_desc_holder_;
  std::vector<TensorDescObserver> observers_;
};
} // namespace hybrid
} // namespace ge

#endif // GE_HYBRID_MODEL_INFER_SHAPE_PROPAGATOR_H

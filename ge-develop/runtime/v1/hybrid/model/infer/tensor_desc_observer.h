/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_HYBRID_MODEL_INFER_TENSOR_DESC_OBSERVER_H
#define GE_HYBRID_MODEL_INFER_TENSOR_DESC_OBSERVER_H

#include "ge/ge_api_error_codes.h"
#include "hybrid/model/infer/tensor_desc_holder.h"

namespace ge {
namespace hybrid {
struct ChangeObserver {
  ChangeObserver() = default;
  virtual ~ChangeObserver() = default;
  virtual void Changed() = 0;

protected:
  ChangeObserver(const ChangeObserver &change) = delete;
  ChangeObserver &operator=(const ChangeObserver &change) = delete;
};

struct TensorDescObserver {
  explicit TensorDescObserver(const TensorDescHolder &holder);
  TensorDescObserver(GeTensorDesc &tensor_desc, ChangeObserver &change_server)
      : tensor_desc_holder_(tensor_desc), change_server_(change_server) {}
  Status OnChanged(const GeTensorDesc &source);

private:
  TensorDescHolder tensor_desc_holder_;
  ChangeObserver &change_server_;
};
} // namespace hybrid
} // namespace ge
#endif // GE_HYBRID_MODEL_INFER_TENSOR_DESC_OBSERVER_H

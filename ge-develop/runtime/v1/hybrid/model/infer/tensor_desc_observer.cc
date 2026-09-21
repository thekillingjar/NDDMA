/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hybrid/model/infer/tensor_desc_observer.h"
#include "graph/ge_tensor.h"
#include "hybrid/model/infer/shape_utils.h"
#include "common/debug/log.h"

namespace ge {
namespace hybrid {

struct DefaultChangeObserver : ChangeObserver {
  void Changed() override {};
} default_observer;

TensorDescObserver::TensorDescObserver(const TensorDescHolder &holder)
    : tensor_desc_holder_(holder), change_server_(default_observer) {}

Status TensorDescObserver::OnChanged(const GeTensorDesc &source) {
  GE_CHK_STATUS_RET_NOLOG(ShapeUtils::CopyShapeAndTensorSize(source, tensor_desc_holder_.Input()));
  change_server_.Changed();
  return SUCCESS;
}
} // namespace hybrid
} // namespace ge

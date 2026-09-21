/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hybrid/model/infer/tensor_desc_holder.h"
#include "hybrid/model/node_item.h"

namespace ge {
namespace hybrid {

bool TensorDescHolder::operator==(const TensorDescHolder &left) const {
  return (this->index_ == left.index_) && (this->node_item_ == left.node_item_);
}

GeTensorDesc& TensorDescHolder::Input() const {
  if ((index_ == INT32_MAX) || (node_item_->MutableInputDesc(index_) == nullptr)) {
    return *tensor_desc_;
  }
  return *(node_item_->MutableInputDesc(index_).get());
}

GeTensorDesc& TensorDescHolder::Output() const {
  if ((index_ == INT32_MAX) || (node_item_->MutableOutputDesc(index_) == nullptr)) {
    return *tensor_desc_;
  }
  return *(node_item_->MutableOutputDesc(index_).get());
}
} // namespace hybrid
} // namespace ge

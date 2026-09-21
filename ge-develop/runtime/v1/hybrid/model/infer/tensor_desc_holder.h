/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TENSOR_DESC_HOLDER_H
#define AIR_CXX_TENSOR_DESC_HOLDER_H

#include <cstdint>
#include "graph/ge_tensor.h"

namespace ge {
namespace hybrid {
struct NodeItem;

struct TensorDescHolder {
  TensorDescHolder(NodeItem &node_item, const int32_t index)
      : node_item_(&node_item), index_(index) {}

  explicit TensorDescHolder(GeTensorDesc &tensor_desc)
      : tensor_desc_(&tensor_desc), index_(INT32_MAX) {}
public:
  bool operator==(const TensorDescHolder &left) const;
  GeTensorDesc& Input() const;
  GeTensorDesc& Output() const;

private:
  union {
    GeTensorDesc *tensor_desc_;
    NodeItem *node_item_;
  };
  int32_t index_;
};
} // namespace hybrid
} // namespace ge
#endif

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/ascendc_ir/utils/asc_tensor_utils.h"

namespace ge {
namespace ascir {
bool AscTensorUtils::IsConstTensor(const AscTensor &t) {
  const auto node = t.anchor.GetOwnerNodeBarePtr();
  GE_ASSERT_NOTNULL(node);
  return node->GetType() == "Constant" || node->GetType() == "IndexExpr" || node->GetType() == "Scalar";
}
Node *AscTensorUtils::GetOwner(const AscTensor &t) {
  return t.anchor.GetOwnerNodeBarePtr();
}
int32_t AscTensorUtils::Index(const AscTensor &t) {
  return t.anchor.GetIdx();
}
}  // namespace ascir
}  // namespace ge

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "storage_format.h"
namespace gert {
namespace bg {
inline bool DiffStorageFormat(const ge::ConstGeTensorDescPtr &td) {
  return td->GetFormat() != td->GetOriginFormat() || td->GetShape().GetDims() != td->GetOriginShape().GetDims();
}
bool DiffStorageFormat(const ge::NodePtr &node, int32_t out_index) {
  const auto &td = node->GetOpDescBarePtr()->GetOutputDescPtr(static_cast<uint32_t>(out_index));
  if (td == nullptr) {
    return true;
  }
  return DiffStorageFormat(td);
}
bool AnyDiffStorageFormat(const ge::NodePtr &node) {
  const auto op_desc = node->GetOpDescBarePtr();
  for (const auto &td : op_desc->GetAllOutputsDescPtr()) {
    if (DiffStorageFormat(td)) {
      return true;
    }
  }
  for (const auto &td : op_desc->GetAllInputsDescPtr()) {
    if (DiffStorageFormat(td)) {
      return true;
    }
  }
  return false;
}
}  // namespace bg
}  // namespace gert

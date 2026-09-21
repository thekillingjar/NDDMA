/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_ASC_TENSOR_UTILS_H
#define METADEF_CXX_ASC_TENSOR_UTILS_H

#include "ascendc_ir/ascendc_ir_core/ascendc_ir.h"

namespace ge {
namespace ascir {
class AscTensorUtils {
 public:
  static bool IsConstTensor(const AscTensor &t);
  static Node *GetOwner(const AscTensor &t);
  static int32_t Index(const AscTensor &t);
};
}
}  // namespace ge

#endif  // METADEF_CXX_ASC_TENSOR_UTILS_H

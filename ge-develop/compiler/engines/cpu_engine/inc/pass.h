/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/** @defgroup Aicpu Pass Interface */

#ifndef INC_AICPU_PASS_H
#define INC_AICPU_PASS_H

#include "framework/common/fmk_error_codes.h"

namespace aicpu {
using Status = uint32_t;
template <typename T>
class Pass {
 public:
  virtual ~Pass() {}
  virtual Status Run(T &graph) = 0;
};
}  // namespace aicpu

#endif  // INC_AICPU_PASS_H

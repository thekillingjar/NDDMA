/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_KERNEL_TRACING_UTILS_H_
#define GE_COMMON_KERNEL_TRACING_UTILS_H_

#include <sstream>
#include <iomanip>

namespace gert {
template <typename T, bool IsPointer = std::is_pointer<T>::value>
void PrintHex(const T *p, size_t num, std::stringstream &ss) {
  for (size_t i = 0; i < num; ++i) {
    if (!IsPointer) {
      // 通过std::setw设置输出位宽为2倍的sizeof(T)
      ss << "0x" << std::setfill('0') << std::setw(static_cast<int32_t>(sizeof(T)) * 2) << std::hex << +p[i] << ' ';
    } else {
      ss << p[i] << ' ';
    }
  }
}
} // namespace gert

#endif // GE_COMMON_KERNEL_TRACING_UTILS_H_

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_CACHE_UTILITY_H
#define AIR_CXX_RUNTIME_V2_CACHE_UTILITY_H

#include <new>

namespace gert {
// cache_utility.h is an internal header file used only within `libgert`, so ABI compatibility is not a concern.
// For more details on `-Winterference-size`, refer to https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html
// This warning was introduced in GCC 12, see https://gcc.gnu.org/gcc-12/changes.html
#if defined(__GNUC__) && (__GNUC__ >= 12)
#define DISABLE_INTERFERENCE_WARNING \
  _Pragma("GCC diagnostic push") _Pragma("GCC diagnostic ignored \"-Winterference-size\"")
#define ENABLE_INTERFERENCE_WARNING _Pragma("GCC diagnostic pop")
#else
#define DISABLE_INTERFERENCE_WARNING
#define ENABLE_INTERFERENCE_WARNING
#endif

DISABLE_INTERFERENCE_WARNING
#ifdef __cpp_lib_hardware_interference_size
constexpr std::size_t kHardwareConstructiveInterferenceSize = std::hardware_constructive_interference_size;
constexpr std::size_t kHardwareDestructiveInterferenceSize = std::hardware_destructive_interference_size;
#else
constexpr std::size_t kHardwareConstructiveInterferenceSize = 64;
constexpr std::size_t kHardwareDestructiveInterferenceSize = 64;
#endif
ENABLE_INTERFERENCE_WARNING
}  // namespace gert

#endif  // AIR_CXX_RUNTIME_V2_CACHE_UTILITY_H

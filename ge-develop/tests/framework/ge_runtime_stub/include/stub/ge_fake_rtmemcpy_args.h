/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_STUB_GE_FAKE_RTMEMCPY_ARGS_H_
#define AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_STUB_GE_FAKE_RTMEMCPY_ARGS_H_
#include <cstdlib>
namespace ge {
enum class MemoryCopyType {
  kRtMemcpy,
  kRtMemcpyAsync
};
struct GeFakeRtMemcpyArgs {
  static GeFakeRtMemcpyArgs RtMemcpy(void *dst, size_t d_max, const void *src, size_t len) {
    return {MemoryCopyType::kRtMemcpy, dst, d_max, src, len, nullptr};
  }
  static GeFakeRtMemcpyArgs RtMemcpyAsync(void *dst, size_t d_max, const void *src, size_t len, void *s) {
    return {MemoryCopyType::kRtMemcpyAsync, dst, d_max, src, len, s};
  }
  MemoryCopyType copy_type;

  void *dst_address;
  size_t dst_max;
  const void *src_address;
  size_t copy_len;

  void *stream;
};
}
#endif  // AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_STUB_GE_FAKE_RTMEMCPY_ARGS_H_

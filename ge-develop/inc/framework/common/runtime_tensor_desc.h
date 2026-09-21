/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_FRAMEWORK_COMMON_RUNTIME_TENSOR_DESC_H_
#define INC_FRAMEWORK_COMMON_RUNTIME_TENSOR_DESC_H_

#include <cstdint>

namespace ge {
constexpr int64_t kMaxDimSize = 32;

#pragma pack(push, 1)
struct RuntimeTensorDesc {
  uint64_t data_addr;
  int64_t data_offset_size;
  int64_t dtype;
  int64_t shape[kMaxDimSize + 1]; // shape:Dim_Num|DIM0|DIM1|...|DIM31
  int64_t original_shape[kMaxDimSize + 1]; // original_shape:Dim_Num|DIM0|DIM1|...|DIM31
  int64_t format;
  int64_t sub_format;
  uint64_t data_size;
  uint8_t reserved[448]; // padding to 1024 bytes
};
#pragma pack(pop)
}  // namespace ge

#endif  // INC_FRAMEWORK_COMMON_RUNTIME_TENSOR_DESC_H_

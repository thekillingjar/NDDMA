/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "utils/data_buffers.h"
#include "framework/common/ge_types.h"

USING_FAKE_NS;

DataBuffers::DataBuffers(int size, bool is_huge_buffer) {
  if (is_huge_buffer) {
    for (int i = 0; i < size; i++) {
        buffers_.emplace_back(DataBuffer(huge_buffer_addr + i * 202400, 202400, false));
    }
  } else {
    for (int i = 0; i < size; i++) {
      buffers_.emplace_back(DataBuffer(buffer_addr + i * 4, 202400, false));
    }
  }
}

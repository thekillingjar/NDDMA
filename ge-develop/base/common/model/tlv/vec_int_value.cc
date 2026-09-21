/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "vec_int_value.h"
#include "common/checker.h"

namespace ge {
size_t vecIntValue::Size() {
  const size_t size = vec_size * sizeof(int64_t) + sizeof(uint32_t);
  return size;
}
bool vecIntValue::Serilize(uint8_t ** const addr, size_t &left_size) {
  GE_ASSERT_NOTNULL(addr, "input param is invalid, addr  is null.");
  GE_ASSERT_NOTNULL(*addr, "addr ptr is null.");

  GE_ASSERT_EOK(memcpy_s(*addr, left_size, static_cast<void *>(&vec_size), sizeof(uint32_t)));
  *addr = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(*addr) + sizeof(uint32_t)));
  left_size -= sizeof(uint32_t);

  if ((value.data() != nullptr) && (vec_size != 0U)) {
    GE_ASSERT_EOK(memcpy_s(*addr, left_size, static_cast<void *>(value.data()), sizeof(uint64_t) * vec_size));
    *addr = PtrToPtr<void, uint8_t>(ValueToPtr(PtrToValue(*addr)
                                    + static_cast<uint64_t>(sizeof(uint64_t) * vec_size)));
    left_size -= sizeof(uint64_t) * vec_size;
  }

  return true;
}
bool vecIntValue::NeedSave() {
  return vec_size > 0U;
}
}

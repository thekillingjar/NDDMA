/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_OP_SO_STORE_UTILS_H
#define GE_COMMON_OP_SO_STORE_UTILS_H

#include "graph/op_so_bin.h"

namespace ge {
class OpSoStoreUtils {
 public:
  static bool IsSoBinType(const uint16_t so_flag, const SoBinType so_bin_type) {
    return (so_flag & EnableSoBinTypeBit(so_bin_type)) != 0;
  }

  static void SetSoBinType(const SoBinType so_bin_type, uint16_t &so_flag) {
    so_flag |= EnableSoBinTypeBit(so_bin_type);
  }

 private:
  // 每个bit位表示一种SoBinType类型, 按照从高位到地位的顺序
  // 1000 0000 中的1表示SpaceRegistry
  // 0100 0000 中的1表示OpMasterDevice
  // 0010 0000 中的1表示Autofuse
  static uint16_t EnableSoBinTypeBit(SoBinType so_bin_type) {
    const uint16_t shift = 15U - static_cast<uint16_t>(so_bin_type);
    constexpr uint16_t base = 1U;
    return base << shift;
  }
};
}  // namespace ge

#endif  // GE_COMMON_OP_SO_STORE_UTILS_H

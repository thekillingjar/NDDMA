/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BASE_TLV_BLOCK_H_
#define BASE_TLV_BLOCK_H_

#include "graph/def_types.h"

namespace ge {
class BaseTlvBlock {
public:
  virtual size_t Size() = 0;
  virtual bool Serilize(uint8_t ** const addr, size_t &left_size) = 0;
  virtual bool NeedSave() = 0;

  BaseTlvBlock() = default;
  virtual ~BaseTlvBlock() = default;

  BaseTlvBlock &operator=(const BaseTlvBlock &) & = delete;
  BaseTlvBlock(const BaseTlvBlock &) = delete;
};
}
#endif

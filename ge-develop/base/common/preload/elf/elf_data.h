/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_ELF_DATA_H_
#define GE_COMMON_ELF_DATA_H_

#include "ge/ge_api_types.h"

namespace ge {
class Elf {
 public:
  static Status GetElfOffset(uint8_t *const bin_data, const uint32_t bin_len, uint32_t &bin_data_offset);
};
}
#endif // GE_COMMON_ELF_DATA_H_
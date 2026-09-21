/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_FRAMEWORK_COMMON_TLV_TLV_H_
#define INC_FRAMEWORK_COMMON_TLV_TLV_H_

#if defined(__cplusplus)
extern "C" {
#endif

#pragma pack(1)  // single-byte alignment

// tlv struct
struct TlvHead {
  uint32_t type;
  uint32_t len;
  uint8_t data[0];
};

#pragma pack()

#if defined(__cplusplus)
}
#endif

#endif  // INC_FRAMEWORK_COMMON_TLV_TLV_H_

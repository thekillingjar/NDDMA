/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_FRAMEWORK_COMMON_TLV_NANO_COMMON_DESC_H_
#define INC_FRAMEWORK_COMMON_TLV_NANO_COMMON_DESC_H_

#include "framework/common/tlv/tlv.h"

#if defined(__cplusplus)
extern "C" {
#endif

#pragma pack(1)  // single-byte alignment

/*********************** definition of pre model desc extend ********************************/
/*
 * ----------------------------------------------------------------------------
 * |   ModelExtendHead        |   tlv data(tlv[len])                          |
 * ----------------------------------------------------------------------------
 * | magic(0x444F4D48U) | len |  TlvHead   |      ModelGlobalDataInfo         |
 * ----------------------------------------------------------------------------
*/
#define MODEL_EXTEND_DESC_MAGIC_NUM 0x444F4D48U
struct ModelExtendHead {
  uint32_t magic;
  uint64_t len;
  uint8_t tlv[0];
};

/*
 * ------------------------------------
 * |      ModelGlobalDataInfo         |
 * ------------------------------------
 * | total_size | num | mem_size[num] |
 * ------------------------------------
*/
#define MODEL_EXTEND_L1_TLV_TYPE_MODEL_GLOBAL_DATA_INFO 0U
struct ModelGlobalDataInfo {
  uint64_t total_size;
  uint32_t num;
  uint64_t mem_size[0];
};
#pragma pack()  // Cancels single-byte alignment

#if defined(__cplusplus)
}
#endif

#endif  // INC_FRAMEWORK_COMMON_TLV_NANO_COMMON_DESC_H_

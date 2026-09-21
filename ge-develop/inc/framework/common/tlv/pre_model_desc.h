/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_FRAMEWORK_COMMON_TLV_pre_model_desc_H_
#define INC_FRAMEWORK_COMMON_TLV_pre_model_desc_H_

namespace ge {
#pragma pack(1)  // single-byte alignment

enum KERNEL_ARG_UPADTE_TYPE {
  KERNEL_ARG_UPDATE_TYPE_ADDR,
  KERNEL_ARG_UPDATE_TYPE_TS,
  KERNEL_ARG_UPDATE_TYPE_P2P,
  KERNEL_ARG_UPDATE_TYPE_CPU_KERNEL_ARGS,
  KERNEL_ARG_UPDATE_TYPE_SESSIONID,
  KERNEL_ARG_UPDATE_TYPE_KERNELID,
  KERNEL_ARG_UPDATE_TYPE_EVENTID,
  KERNEL_ARG_UPDATE_TYPE_BUFF
};
enum KERNEL_ARG_UPADTE_ADDR_TYPE {
  KERNEL_ARG_UPADTE_ADDR_TYPE_ARGS,
  KERNEL_ARG_UPADTE_ADDR_TYPE_WORKSPACE,
  KERNEL_ARG_UPADTE_ADDR_TYPE_WEIGHT,
  KERNEL_ARG_UPADTE_ADDR_TYPE_L1,
  KERNEL_ARG_UPADTE_ADDR_TYPE_TS,
  KERNEL_ARG_UPADTE_ADDR_TYPE_P2P,
  KERNEL_ARG_UPADTE_ADDR_TYPE_VAR,
  KERNEL_ARG_UPADTE_ADDR_TYPE_KERNEL_BIN,
  KERNEL_ARG_UPADTE_ADDR_TYPE_BUFF
};

/********************************************************************************************/
#pragma pack()  // Cancels single-byte alignment
}  // namespace ge
#endif  // INC_FRAMEWORK_COMMON_TLV_pre_model_desc_H_

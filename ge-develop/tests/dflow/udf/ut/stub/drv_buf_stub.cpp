/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ascend_hal.h"

#ifdef __cplusplus
extern "C" {
#endif

int halBuffInit(BuffCfg *cfg) {
  return 0;
}

int halMbufAllocEx(uint64_t size, unsigned int align, unsigned long flag, int grp_id, Mbuf **mbuf) {
  return DRV_ERROR_NONE;
}

int halMbufFree(Mbuf *mbuf) {
  return DRV_ERROR_NONE;
}

int halMbufGetBuffAddr(Mbuf *mbuf, void **buf) {
  return DRV_ERROR_NONE;
}

int halMbufGetBuffSize(Mbuf *mbuf, uint64_t *totalSize) {
  return DRV_ERROR_NONE;
}

int halMbufSetDataLen(Mbuf *mbuf, uint64_t len) {
  return DRV_ERROR_NONE;
}

int halMbufCopyRef(Mbuf *mbuf, Mbuf **newMbuf) {
  return DRV_ERROR_NONE;
}

int halMbufGetPrivInfo(Mbuf *mbuf, void **priv, unsigned int *size) {
  return DRV_ERROR_NONE;
}

int halMbufGetDataLen(Mbuf *mbuf, uint64_t *len) {
  return DRV_ERROR_NONE;
}

#ifdef __cplusplus
}
#endif
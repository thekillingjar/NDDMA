/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "securec.h"
#include "framework/executor_c/ge_log.h"
#include "ge/ge_error_codes.h"
#include "framework/executor_c/ge_executor.h"
#include "tlv_parse.h"
char *StringDepthCpy(const uint32_t len, const void *srcAddr) {
  if (len == 0U) {
    return NULL;
  }
  char *name = (char *)mmMalloc(len + 1);
  if (name == NULL) {
    return NULL;
  }
  errno_t ret = memcpy_s(name, len + 1, srcAddr, len);
    if (ret != 0) {
      (void)mmFree(name);
      name = NULL;
      return NULL;
  }
  name[len] = '\0';
  GELOGI("StringDepthCpy name[%s].", name);
  return name;
}

static int32_t FindTlvIndex(uint32_t parseTlvNum, uint32_t parseSubTlvTypeList[], uint32_t type) {
  for (size_t index = 0UL; index < parseTlvNum; ++index) {
    if (parseSubTlvTypeList[index] == type) {
      return (int32_t)index;
    }
  }
  return INVALID_INDEX_VALUE;
}

static TlvProcPair *FindTlvProc(uint32_t parseTlvNum, TlvProcPair TlvProcList[], uint32_t type) {
  for (size_t index = 0UL; index < parseTlvNum; ++index) {
    if (TlvProcList[index].tlvType == type) {
      return &TlvProcList[index];
    }
  }
  return NULL;
}

bool CheckTlvLenValid(const size_t total, const size_t offset, const size_t nexLen) {
  return !(nexLen > total - offset);
}

uint32_t ParseSubTlvListU16(uint8_t *subTlvList, uint32_t subTlvlistLen, uint32_t parseTlvNum,
                            uint32_t parseSubTlvTypeList[], struct TlvHead *parseSubTlvList[]) {
  size_t offset = 0UL;
  uint32_t parseNum = 0U;
  while ((subTlvlistLen - offset) >= sizeof(struct TlvHead) && (parseNum <= parseTlvNum)) {
    struct TlvHead *tlv = (struct TlvHead *)(subTlvList + offset);
    offset += sizeof(struct TlvHead);
    GELOGI("total len[%u] tlv len[%u] type[%u] offset[%zu].", subTlvlistLen, tlv->len, tlv->type, offset);
    if (!CheckTlvLenValid(subTlvlistLen, offset, tlv->len)) {
      GELOGE(ACL_ERROR_GE_PARAM_INVALID, "tlv parse failed.");
      break;
    }
    int32_t index = FindTlvIndex(parseTlvNum, parseSubTlvTypeList, tlv->type);
    if ((index < (int32_t)parseTlvNum) && (index != INVALID_INDEX_VALUE)) {
      parseSubTlvList[index] = tlv;
      parseNum++;
    }
    offset += tlv->len;
  }
  return parseNum;
}

uint32_t ParseAndProcSubTlvListU32(uint8_t *subTlvList, uint32_t subTlvlistLen, uint32_t parseTlvNum,
                                   TlvProcPair TlvProcList[], void *appInfo) {
  Status ret = SUCCESS;
  size_t offset = 0UL;
  uint32_t parseNum = 0U;
  while ((subTlvlistLen - offset) >= sizeof(struct TlvHead) && (parseNum <= parseTlvNum)) {
    struct TlvHead *tlv = (struct TlvHead *)(subTlvList + offset);
    offset += sizeof(struct TlvHead);
    GELOGI("total len [%u] tlv len[%u] type[%u] offset[%zu].", subTlvlistLen, tlv->len, tlv->type, offset);
    if (!CheckTlvLenValid(subTlvlistLen, offset, tlv->len)) {
       GELOGE(ACL_ERROR_GE_PARAM_INVALID, "check tlv len failed.");
       ret = ACL_ERROR_GE_INTERNAL_ERROR;
       break;
    }
    TlvProcPair *tlvProc = FindTlvProc(parseTlvNum, TlvProcList, tlv->type);
    if (tlvProc != NULL) {
      GELOGI("pfnTlvProc tlv type[%u] len[%u].", tlv->type, tlv->len);
      ret = tlvProc->pfnTlvProc(tlv->len, tlv->data, appInfo);
      if (ret == SUCCESS) {
        parseNum++;
      } else {
        break;
      }
    }
    offset += tlv->len;
  }
  ((ModelDbgHandle *)(appInfo))->offset += offset;
  return ret;
}
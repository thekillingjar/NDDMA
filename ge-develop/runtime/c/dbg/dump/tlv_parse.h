/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_EXECUTOR_C_DBG_TLV_PARSE_FILE_H_
#define GE_EXECUTOR_C_DBG_TLV_PARSE_FILE_H_
#include "framework/executor_c/ge_executor_types.h"
#include "framework/common/tlv/nano_dbg_desc.h"
#include "framework/common/tlv/tlv.h"
#ifdef __cplusplus
extern "C" {
#endif
#define INVALID_INDEX_VALUE (-1)

/******************************** tlv oper ************************************************/
#pragma pack(1)  // single-byte alignment
#define DBG_L1_TLV_TYPE_MODEL_DESC        2U
#define DBG_L1_TLV_TYPE_DUMP_PATH         3U
struct DbgModelDescTlv1 {
  uint32_t flag; // 0x1 load , 0x0 unload //加载时设置
  uint32_t model_id;
  uint64_t *step_id_addr;
  uint64_t iterations_per_loop_addr; // 训练特有，推理填0
  uint64_t loop_cond_addr;
  uint32_t dump_mode;
  uint64_t dump_data;
  uint32_t l2_tlv_list_len;
  uint8_t l2_tlv[0];
};

/****************** definition of dbg L1 Tlv string model name tlv's value *****************/
struct DbgModelNameTlv1 {
  uint8_t name[0];
};

/****************** definition of dbg L1 Tlv string dump path tlv's value *****************/
struct DbgDumpPathTlv1 {
  uint8_t dump_path[0];
};
/********************************************************************************************/
#pragma pack() // Cancels single-byte alignment

typedef struct {
  uint32_t modelId; // 实际加载之后驱动生成的模型id
  size_t aicpuDumpInfoLen;
  uint8_t *aicpuDumpInfo;
  size_t dumpFileLen;
  uint8_t *dumpFileContent;
  size_t offset;
  void *taskDescBaseAddr;
  size_t taskDescSize;
  uint64_t *stepIdAddr;
  uint32_t cfgMatchedCount;
  uint32_t totalTaskNum;
  bool isSendFlag;
  char *modelName; // dbg中解析的model name
} ModelDbgHandle;

uint32_t ParseSubTlvListU16(uint8_t *subTlvList, uint32_t subTlvlistLen, uint32_t parseTlvNum,
                            uint32_t parseSubTlvTypeList[], struct TlvHead *parseSubTlvList[]);
uint32_t ParseAndProcSubTlvListU32(uint8_t *subTlvList, uint32_t subTlvlistLen, uint32_t parseTlvNum,
                                   TlvProcPair TlvProcList[], void *appInfo);

bool CheckTlvLenValid(const size_t total, const size_t offset, const size_t nexLen);

char *StringDepthCpy(const uint32_t len, const void *srcAddr);
#ifdef __cplusplus
}
#endif
#endif  // GE_EXECUTOR_C_DBG_TLV_PARSE_FILE_H_

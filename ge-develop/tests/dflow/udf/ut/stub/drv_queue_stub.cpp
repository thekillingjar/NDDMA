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
drvError_t halQueueDeQueue(unsigned int devId, unsigned int qid, void **mbug) {
  return DRV_ERROR_QUEUE_EMPTY;
}

drvError_t halQueueEnQueue(unsigned int devId, unsigned int qid, void *mbuf) {
  return DRV_ERROR_NONE;
}
drvError_t halQueueSet(unsigned int devId, QueueSetCmdType cmd, QueueSetInputPara *inPut) {
  return DRV_ERROR_NONE;
}
int halGrpAttach(const char *name, int timeout) {
  return 0;
}

int halGrpQuery(GroupQueryCmdType cmd, void *inBuff, unsigned int inLen, void *outBuff, unsigned int *outLen) {
  return 0;
}

drvError_t halQueueGetStatus(unsigned int devId, unsigned int qid, QUEUE_QUERY_ITEM queryItem, unsigned int len,
                             void *data) {
  if (queryItem == QUERY_QUEUE_STATUS) {
    *static_cast<int32_t *>(data) = static_cast<int32_t>(QUEUE_NORMAL);
  }
  return DRV_ERROR_NONE;
}

drvError_t halQueueAttach(unsigned int devId, unsigned int qid, int timeOut) {
  return DRV_ERROR_NONE;
}

drvError_t halQueueInit(unsigned int devid) {
  return DRV_ERROR_NONE;
}

drvError_t halQueueSubF2NFEvent(unsigned int devid, unsigned int qid, unsigned int groupid) {
  return DRV_ERROR_NONE;
}

drvError_t halQueueUnsubF2NFEvent(unsigned int devId, unsigned int qid) {
  return DRV_ERROR_NONE;
}

drvError_t halQueueSubscribe(unsigned int devid, unsigned int qid, unsigned int groupId, int type) {
  return DRV_ERROR_NONE;
}

drvError_t halQueueUnsubscribe(unsigned int devId, unsigned int qid) {
  return DRV_ERROR_NONE;
}
drvError_t halQueueSubEvent(struct QueueSubPara *subPara) {
  return DRV_ERROR_NONE;
}
drvError_t halQueueUnsubEvent(struct QueueUnsubPara *unsubPara) {
  return DRV_ERROR_NONE;
}

drvError_t halQueuePeek(unsigned int devId, unsigned int qid, uint64_t *buf_len, int timeout) {
  return DRV_ERROR_NONE;
}

int halMbufAlloc(uint64_t size, Mbuf **mbuf) {
  return DRV_ERROR_NONE;
}

drvError_t halQueueDeQueueBuff(unsigned int devId, unsigned int qid, struct buff_iovec *vector, int timeout) {
  return DRV_ERROR_NONE;
}

drvError_t halQueueEnQueueBuff(unsigned int devId, unsigned int qid, struct buff_iovec *vector, int timeout) {
  return DRV_ERROR_NONE;
}
drvError_t drvGetProcessSign(struct process_sign *sign) {
  return DRV_ERROR_NONE;
}
#ifdef __cplusplus
}
#endif
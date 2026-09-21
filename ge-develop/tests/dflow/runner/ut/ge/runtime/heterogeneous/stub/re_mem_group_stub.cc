/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "depends/runtime/src/runtime_stub.h"
#include <string.h>
#include <string>
RTS_API rtError_t rtMemGrpCreate(const char *name, const rtMemGrpConfig_t *cfg) {
  if (strncmp("DM_QS_GROUP", name, 11) != 0) {
    return 1;
  }
  return 0;
}

RTS_API rtError_t rtMemGrpAddProc(const char *name, int32_t pid, const rtMemGrpShareAttr_t *attr) {
  if (pid < 0) {
    return 1;
  }
  return 0;
}

RTS_API rtError_t rtMemGrpAttach(const char *name, int32_t timeout) {
  return 0;
}

RTS_API rtError_t rtMemQueueInit(int32_t devId) {
  if (devId > 10000) {
    return 1;
  }
  return 0;
}

RTS_API rtError_t rtQueryDevPid(rtBindHostpidInfo_t *info, int32_t *devPid) {
  if ((info != nullptr) && (info->chipId == 0xff)) {
    return 1;
  }
  *devPid = 100;
  return 0;
}

RTS_API rtError_t rtMemQueueInitQS(int32_t devId, const char* grpName) {
  const std::string group_name(grpName);
  const std::string qs_name("TEST_ERROR");
  if (group_name.find(qs_name) != group_name.npos) {
    return 1;
  }
  return 0;
}

RTS_API rtError_t rtMemQueueGrant(int32_t devId, uint32_t qid, int32_t pid, rtMemQueueShareAttr_t *attr) {
  if (pid < 0) {
    return 1;
  }
  return 0;
}

RTS_API rtError_t rtEschedAttachDevice(int32_t device) {
  return 0;
}

RTS_API rtError_t rtEschedCreateGrp(int32_t devId, uint32_t grpId, rtGroupType_t type) {
  return 0;
}

RTS_API rtError_t rtEschedSubmitEvent(int32_t devId, rtEschedEventSummary_t *event) {
  return 0;
}

RTS_API rtError_t rtEschedSubscribeEvent(int32_t devId,
                                         uint32_t grpId,
                                         uint32_t threadId,
                                         uint64_t eventBitmap) {
  return 0;
}

rtError_t rtEschedQueryInfo(const uint32_t devId,
                            const rtEschedQueryType type,
                            rtEschedInputInfo *input,
                            rtEschedOutputInfo *output) {
  return 0;
}

RTS_API rtError_t rtMemQueueInitFlowGw(int32_t devId, const rtInitFlowGwInfo_t * const initInfo) {
  const std::string group_name(initInfo->groupName);
  const std::string qs_name("TEST_ERROR");
  if (group_name.find(qs_name) != group_name.npos) {
    return 1;
  }
  return 0;
}

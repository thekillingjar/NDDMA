/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_COMMON_UTILS_RTS_API_UTILS_H_
#define AIR_RUNTIME_COMMON_UTILS_RTS_API_UTILS_H_
#include "framework/common/debug/log.h"
#include "runtime/rt.h"
#include "acl/acl.h"
#include "common/df_chk.h"

namespace ge {
class RtsApiUtils {
 public:
  static inline Status SetDevice(int32_t device_id) {
    DF_CHK_ACL_RET(aclrtSetDevice(device_id));
    return SUCCESS;
  }

  static inline Status MemGrpAttach(const std::string &name, int32_t timeout) {
    GE_CHK_RT_RET(rtMemGrpAttach(name.c_str(), timeout));
    return SUCCESS;
  }

  static inline Status MbufInit() {
    rtMemBuffCfg_t buff_cfg{};
    auto ret = rtMbufInit(&buff_cfg);
    GE_CHK_BOOL_RET_STATUS(ret == SUCCESS || ret == ACL_ERROR_RT_REPEATED_INIT,
                           RT_FAILED, "Invoke rtMbufInit failed, ret = 0x%X", static_cast<uint32_t>(ret));
    return SUCCESS;
  }

  static inline Status MbufGetBufferAddr(void *mbuf, void **data_addr) {
    GE_CHK_RT_RET(rtMbufGetBuffAddr(mbuf, data_addr));
    return SUCCESS;
  }

  static inline Status MbufGetBufferSize(void *mbuf, uint64_t &size) {
    GE_CHK_RT_RET(rtMbufGetBuffSize(mbuf, &size));
    return SUCCESS;
  }

  static inline Status MbufGetPrivData(void *mbuf, void **priv, uint64_t *size) {
    GE_CHK_RT_RET(rtMbufGetPrivInfo(mbuf, priv, size));
    return SUCCESS;
  }

  static inline Status MemQueueInit(int32_t device_id) {
    auto ret = rtMemQueueInit(device_id);
    GE_CHK_BOOL_RET_STATUS(ret == SUCCESS || ret == ACL_ERROR_RT_REPEATED_INIT,
                           RT_FAILED, "Invoke rtMemQueueInit failed, device_id = %d, ret = 0x%X", device_id, static_cast<uint32_t>(ret));
    return SUCCESS;
  }

  static inline Status MemQueryGetQidByName(int32_t device_id, const std::string &queue_name, uint32_t &queue_id) {
    GE_CHK_RT_RET(rtMemQueueGetQidByName(device_id, queue_name.c_str(), &queue_id));
    return SUCCESS;
  }

  static inline Status MemQueueAttach(int32_t device_id, uint32_t queue_id, int32_t timeout) {
    GE_CHK_RT_RET(rtMemQueueAttach(device_id, queue_id, timeout));
    return SUCCESS;
  }

  static inline Status EschedAttachDevice(int32_t device_id) {
    GE_CHK_RT_RET(rtEschedAttachDevice(device_id));
    return SUCCESS;
  }

  static inline Status EschedCreateGroup(int32_t device_id, uint32_t group_id, rtGroupType_t type) {
    GE_CHK_RT_RET(rtEschedCreateGrp(device_id, group_id, type));
    return SUCCESS;
  }

  static inline Status EschedSubscribeEvent(int32_t device_id, uint32_t group_id, uint32_t thread_id, uint64_t mask) {
    GE_CHK_RT_RET(rtEschedSubscribeEvent(device_id, group_id, thread_id, mask));
    return SUCCESS;
  }

  static inline Status EschedSubmitEvent(int32_t device_id, rtEschedEventSummary_t &event_summary) {
    GE_CHK_RT_RET(rtEschedSubmitEvent(device_id, &event_summary));
    return SUCCESS;
  }

  static inline Status GetAicpuSchedulePid(int32_t device_id, int32_t host_pid, int32_t &aicpu_pid) {
    rtBindHostpidInfo_t info{};
    info.cpType = RT_DEV_PROCESS_CP1;
    info.hostPid = host_pid;
    info.chipId = static_cast<uint32_t>(device_id);
    GE_CHK_RT_RET(rtQueryDevPid(&info, &aicpu_pid));
    return SUCCESS;
  }
};
}  // namespace ge
#endif  // AIR_RUNTIME_COMMON_UTILS_RTS_API_UTILS_H_

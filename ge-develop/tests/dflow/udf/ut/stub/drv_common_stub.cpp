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
#include <unistd.h>
#include <condition_variable>
#include <list>
#include <mutex>

#ifdef __cplusplus
extern "C" {
#endif

pid_t drvDeviceGetBareTgid(void) {
  return 0;
}

drvError_t halEschedAttachDevice(unsigned int devId) {
  return DRV_ERROR_NONE;
}

drvError_t halEschedDettachDevice(unsigned int devId) {
  return DRV_ERROR_NONE;
}
drvError_t halEschedCreateGrp(unsigned int devId, unsigned int grpId, GROUP_TYPE type) {
  return DRV_ERROR_NONE;
}

drvError_t halEschedCreateGrpEx(uint32_t devId, struct esched_grp_para *grpPara, unsigned int *grpId) {
  return DRV_ERROR_NONE;
}

std::condition_variable condition;
std::mutex event_mutex;
std::list<std::pair<EVENT_ID, uint32_t>> event_list;
drvError_t halEschedSubmitEvent(unsigned int devId, struct event_summary *event) {
  std::unique_lock<std::mutex> lk(event_mutex);
  event_list.emplace_back(event->event_id, event->subevent_id);
  condition.notify_one();
  return DRV_ERROR_NONE;
}

drvError_t halEschedSubmitEventSync(unsigned int devId, struct event_summary *event, int timeout,
                                    struct event_reply *reply) {
  return DRV_ERROR_NONE;
}

drvError_t halEschedSubmitEventBatch(unsigned int devId, SUBMIT_FLAG flag, struct event_summary *events,
                                     unsigned int event_num, unsigned int *succ_event_num) {
  return DRV_ERROR_NONE;
}

drvError_t halEschedSubmitEventToThread(unsigned int devId, struct event_summary *event) {
  return DRV_ERROR_NONE;
}

drvError_t halEschedWaitEvent(unsigned int devId, unsigned int grpId, unsigned int threadId, int timeout,
                              struct event_info *event) {
  if (timeout == 0) {
    return DRV_ERROR_NO_EVENT;
  }
  std::unique_lock<std::mutex> lk(event_mutex);
  // 用例降低10倍等待，防止用例耗时太长
  condition.wait_for(lk, std::chrono::milliseconds(timeout / 10), [] { return !event_list.empty(); });
  if (event_list.empty()) {
    return DRV_ERROR_SCHED_WAIT_TIMEOUT;
  }
  event->comm.event_id = event_list.front().first;
  event->comm.subevent_id = event_list.front().second;
  event_list.pop_front();
  return DRV_ERROR_NONE;
}

drvError_t halEschedGetEvent(unsigned int devId, unsigned int grpId, unsigned int threadId, EVENT_ID eventId,
                             struct event_info *event) {
  return DRV_ERROR_NO_EVENT;
}

drvError_t halEschedThreadSwapout(unsigned int devId, unsigned int grpId, unsigned int threadId) {
  return DRV_ERROR_NONE;
}

drvError_t halEschedSubscribeEvent(unsigned int devId, unsigned int grpId, unsigned int threadId,
                                   unsigned long long eventBitmap) {
  return DRV_ERROR_NONE;
}

drvError_t halEschedSetPidPriority(unsigned int devId, SCHEDULE_PRIORITY priority) {
  return DRV_ERROR_NONE;
}

drvError_t halEschedSetEventPriority(unsigned int devId, EVENT_ID eventId, SCHEDULE_PRIORITY priority) {
  return DRV_ERROR_NONE;
}

drvError_t halGetDeviceInfo(uint32_t devId, int32_t moduleType, int32_t infoType, int64_t *value) {
  switch (moduleType) {
    case MODULE_TYPE_AICPU:
      if (infoType == INFO_TYPE_OS_SCHED) {
        *value = 1L;
      } else if (infoType == INFO_TYPE_CORE_NUM) {
        *value = 6L;
      } else if (infoType == INFO_TYPE_OCCUPY) {
        *value = 0b11111100L;
      } else {
        return DRV_ERROR_NOT_SUPPORT;
      }
      break;
    case MODULE_TYPE_CCPU:
    case MODULE_TYPE_DCPU:
      if (infoType == INFO_TYPE_OS_SCHED) {
        *value = 1L;
      } else if (infoType == INFO_TYPE_CORE_NUM) {
        *value = 1L;
      } else {
        return DRV_ERROR_NOT_SUPPORT;
      }
      break;
    default:
      return DRV_ERROR_NOT_SUPPORT;
  }
  return DRV_ERROR_NONE;
}

drvError_t drvQueryProcessHostPid(int pid, unsigned int *chip_id, unsigned int *vfid, unsigned int *host_pid,
                                  unsigned int *cp_type) {
  *host_pid = 999999;
  return DRV_ERROR_NONE;
}

drvError_t halQueueQueryInfo(unsigned int devId, unsigned int qid, QueueInfo *queInfo) {
  queInfo->size = qid % 2;
  return DRV_ERROR_NONE;
}

drvError_t halQueryDevpid(struct halQueryDevpidInfo info, pid_t *dev_pid) {
  dev_pid = 12345;
  return DRV_ERROR_NONE;
}
#ifdef __cplusplus
}
#endif
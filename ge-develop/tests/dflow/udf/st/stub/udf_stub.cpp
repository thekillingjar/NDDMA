/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "udf_stub.h"
#include <mutex>
#include <list>
#include <map>
#include <chrono>
#include <condition_variable>
#include <atomic>
#include "securec.h"
#include "ascend_hal.h"
#include "common/udf_log.h"

namespace {
std::condition_variable condition;
std::mutex event_mutex;
std::list<event_info> event_list;

std::mutex queue_mutex;
std::map<uint32_t, std::list<Mbuf *>> queue_list;
std::map<uint32_t, uint32_t> queue_depth;

std::map<uint32_t, uint32_t> g_link_queues;

std::atomic_int32_t g_next_queue_id(0);
}  // namespace

namespace FlowFunc {
void ClearStubEschedEvents() {
  std::unique_lock<std::mutex> lk(event_mutex);
  if (event_list.empty()) {
    return;
  }
  UDF_LOG_INFO("clear events size=%zu", event_list.size());
  event_list.clear();
}

uint32_t CreateQueue(uint32_t depth) {
  std::unique_lock<std::mutex> lock(queue_mutex);
  uint32_t queue_id = g_next_queue_id++;
  auto iter = queue_list.find(queue_id);
  if (iter == queue_list.end()) {
    queue_list[queue_id] = {};
    queue_depth[queue_id] = depth;
    return queue_id;
  }
  auto &queue_data = iter->second;
  for (auto mbuf : queue_data) {
    halMbufFree(mbuf);
  }
  queue_data.clear();
  return queue_id;
}

void DestroyQueue(uint32_t queue_id) {
  std::unique_lock<std::mutex> lock(queue_mutex);
  auto iter = queue_list.find(queue_id);
  if (iter == queue_list.end()) {
    return;
  }
  auto &queue_data = iter->second;
  for (auto mbuf : queue_data) {
    halMbufFree(mbuf);
  }
  queue_list.erase(queue_id);
}

int32_t LinkQueueInTest(uint32_t queue_id, uint32_t linked_queue_id) {
  std::unique_lock<std::mutex> lock(queue_mutex);
  auto iter = queue_list.find(queue_id);
  if (iter == queue_list.end()) {
    return DRV_ERROR_QUEUE_NOT_CREATED;
  }
  iter = queue_list.find(linked_queue_id);
  if (iter == queue_list.end()) {
    return DRV_ERROR_QUEUE_NOT_CREATED;
  }
  if (g_link_queues.count(queue_id) != 0) {
    return DRV_ERROR_QUEUE_INNER_ERROR;
  }
  g_link_queues[queue_id] = linked_queue_id;
  return DRV_ERROR_NONE;
}

void UnlinkQueueInTest(uint32_t queue_id) {
  std::unique_lock<std::mutex> lock(queue_mutex);
  g_link_queues.erase(queue_id);
}
}  // namespace FlowFunc

using namespace FlowFunc;

drvError_t halEschedWaitEvent(unsigned int devId, unsigned int grpId, unsigned int threadId, int timeout,
                              struct event_info *event) {
  if (timeout == 0) {
    return DRV_ERROR_NO_EVENT;
  }
  UDF_LOG_INFO("thread %u wait, timeout=%d", threadId, timeout);
  std::unique_lock<std::mutex> lk(event_mutex);
  // 用例降低10倍等待，防止用例耗时太长
  condition.wait_for(lk, std::chrono::milliseconds(timeout / 10), [] { return !event_list.empty(); });
  if (event_list.empty()) {
    UDF_LOG_INFO("thread %u wait timeout", threadId);
    return DRV_ERROR_SCHED_WAIT_TIMEOUT;
  }
  *event = event_list.front();
  event_list.pop_front();
  UDF_LOG_INFO("thread %u wait exit, event_id=%d, subevent_id=%d", threadId, event->comm.event_id,
               event->comm.subevent_id);
  return DRV_ERROR_NONE;
}

drvError_t halEschedSubmitEvent(unsigned int devId, struct event_summary *event) {
  std::unique_lock<std::mutex> lk(event_mutex);
  event_info event_info;
  event_info.comm.event_id = event->event_id;
  event_info.comm.subevent_id = event->subevent_id;
  event_info.priv.msg_len = event->msg_len;
  memcpy_s(event_info.priv.msg, 128, event->msg, event->msg_len);
  event_list.emplace_back(event_info);
  condition.notify_one();
  UDF_LOG_INFO("submit event, event_id=%d, subevent_id=%d", event->event_id, event->subevent_id);
  return DRV_ERROR_NONE;
}

drvError_t halEschedSubmitEventToThread(unsigned int devId, struct event_summary *event) {
  std::unique_lock<std::mutex> lk(event_mutex);
  event_info event_info;
  event_info.comm.event_id = event->event_id;
  event_info.comm.subevent_id = event->subevent_id;
  event_info.priv.msg_len = event->msg_len;
  memcpy_s(event_info.priv.msg, 128, event->msg, event->msg_len);
  event_list.emplace_back(event_info);
  condition.notify_one();
  return DRV_ERROR_NONE;
}

drvError_t halEschedSubmitEventBatch(unsigned int devId, SUBMIT_FLAG flag, struct event_summary *events,
                                     unsigned int event_num, unsigned int *succ_event_num) {
  for (unsigned int index = 0U; index < event_num; event_num++) {
    halEschedSubmitEvent(devId, events);
  }
  return DRV_ERROR_NONE;
}

drvError_t halEschedSubmitEventSync(unsigned int devId, struct event_summary *event, int timeout,
                                    struct event_reply *reply) {
  std::unique_lock<std::mutex> lk(event_mutex);
  return DRV_ERROR_NONE;
}

int halMbufAllocEx(uint64_t size, unsigned int align, unsigned long flag, int grp_id, Mbuf **mbuf) {
  MbufStImpl *mbuf_impl = new (std::nothrow) MbufStImpl();
  memset_s(mbuf_impl, sizeof(MbufStImpl), 0, sizeof(MbufStImpl));
  mbuf_impl->mbuf_size = size;
  *mbuf = reinterpret_cast<Mbuf *>(mbuf_impl);

  UDF_LOG_INFO("--------alloc mbuf=%p, size=%lu", *mbuf, size);
  return DRV_ERROR_NONE;
}

int halMbufCopyRef(Mbuf *mbuf, Mbuf **newMbuf) {
  MbufStImpl *src_mbuf_impl = reinterpret_cast<MbufStImpl *>(mbuf);
  MbufStImpl *new_mbuf_impl = new (std::nothrow) MbufStImpl();
  memcpy_s(new_mbuf_impl, sizeof(MbufStImpl), src_mbuf_impl, sizeof(MbufStImpl));
  *newMbuf = reinterpret_cast<Mbuf *>(new_mbuf_impl);
  UDF_LOG_INFO("--------copy alloc mbuf=%p, size=%lu", new_mbuf_impl, new_mbuf_impl->mbuf_size);
  return DRV_ERROR_NONE;
}

int halMbufFree(Mbuf *mbuf) {
  UDF_LOG_INFO("--------free alloc mbuf=%p", mbuf);

  MbufStImpl *mbuf_impl = reinterpret_cast<MbufStImpl *>(mbuf);
  delete mbuf_impl;
  return DRV_ERROR_NONE;
}

int halMbufGetBuffAddr(Mbuf *mbuf, void **buf) {
  MbufStImpl *mbuf_impl = reinterpret_cast<MbufStImpl *>(mbuf);
  *buf = &(mbuf_impl->tensor_desc);
  return DRV_ERROR_NONE;
}

int halMbufGetBuffSize(Mbuf *mbuf, uint64_t *totalSize) {
  MbufStImpl *mbuf_impl = reinterpret_cast<MbufStImpl *>(mbuf);
  *totalSize = mbuf_impl->mbuf_size;
  return DRV_ERROR_NONE;
}

int halMbufSetDataLen(Mbuf *mbuf, uint64_t len) {
  MbufStImpl *mbuf_impl = reinterpret_cast<MbufStImpl *>(mbuf);
  mbuf_impl->mbuf_size = len;
  return DRV_ERROR_NONE;
}

int halMbufGetPrivInfo(Mbuf *mbuf, void **priv, unsigned int *size) {
  MbufStImpl *mbuf_impl = reinterpret_cast<MbufStImpl *>(mbuf);
  // start from first byte.
  *priv = &mbuf_impl->reserve_head;
  *size = 256;
  return DRV_ERROR_NONE;
}

int halMbufGetDataLen(Mbuf *mbuf, uint64_t *len) {
  MbufStImpl *mbuf_impl = reinterpret_cast<MbufStImpl *>(mbuf);
  *len = mbuf_impl->mbuf_size;
  return DRV_ERROR_NONE;
}

drvError_t halQueueDeQueue(unsigned int devId, unsigned int qid, void **mbuf) {
  std::unique_lock<std::mutex> lock(queue_mutex);
  auto queue_iter = queue_list.find(qid);
  if (queue_iter == queue_list.end()) {
    return DRV_ERROR_QUEUE_NOT_CREATED;
  }
  auto &queue_data = queue_iter->second;
  if (queue_data.empty()) {
    auto iter = g_link_queues.find(qid);
    if (iter != g_link_queues.end()) {
      uint32_t linked_qid = iter->second;
      void *linked_mbuf = nullptr;
      lock.unlock();
      halQueueDeQueue(devId, linked_qid, &linked_mbuf);
      if (linked_mbuf != nullptr) {
        halQueueEnQueue(devId, qid, linked_mbuf);
        return halQueueDeQueue(devId, qid, mbuf);
      }
    }
    return DRV_ERROR_QUEUE_EMPTY;
  }
  bool queue_full = (queue_data.size() >= queue_depth[qid]);
  *mbuf = queue_data.front();
  queue_data.pop_front();
  if (queue_full) {
    struct event_summary event = {};
    event.event_id = EVENT_QUEUE_FULL_TO_NOT_FULL;
    event.subevent_id = qid;
    halEschedSubmitEvent(0, &event);
  }
  return DRV_ERROR_NONE;
}

drvError_t halQueueEnQueue(unsigned int devId, unsigned int qid, void *mbuf) {
  std::unique_lock<std::mutex> lock(queue_mutex);
  auto iter = queue_list.find(qid);
  if (iter == queue_list.end()) {
    return DRV_ERROR_QUEUE_NOT_CREATED;
  }
  auto &queue_data = iter->second;
  if (queue_data.size() >= queue_depth[qid]) {
    return DRV_ERROR_QUEUE_FULL;
  }
  bool empty_to_not_empty = queue_data.empty();
  queue_data.emplace_back((Mbuf *)mbuf);
  if (empty_to_not_empty) {
    struct event_summary event = {};
    event.event_id = EVENT_QUEUE_EMPTY_TO_NOT_EMPTY;
    event.subevent_id = qid;
    halEschedSubmitEvent(0, &event);
  }
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
  } else if (queryItem == QUERY_QUEUE_DEPTH) {
    *static_cast<int32_t *>(data) = UDF_ST_QUEUE_MAX_DEPTH;
  }
  return DRV_ERROR_NONE;
}

drvError_t halQueueAttach(unsigned int devId, unsigned int qid, int timeOut) {
  std::unique_lock<std::mutex> lock(queue_mutex);
  auto iter = queue_list.find(qid);
  if (iter == queue_list.end()) {
    return DRV_ERROR_QUEUE_NOT_CREATED;
  }
  return DRV_ERROR_NONE;
}

drvError_t drvQueryProcessHostPid(int pid, unsigned int *chip_id, unsigned int *vfid, unsigned int *host_pid,
                                  unsigned int *cp_type) {
  *host_pid = 999999;
  return DRV_ERROR_NONE;
}

drvError_t halQueueInit(unsigned int devid) {
  return DRV_ERROR_NONE;
}

drvError_t halQueueSubF2NFEvent(unsigned int devid, unsigned int qid, unsigned int groupid) {
  std::unique_lock<std::mutex> lock(queue_mutex);
  auto iter = queue_list.find(qid);
  if (iter == queue_list.end()) {
    return DRV_ERROR_QUEUE_NOT_CREATED;
  }
  return DRV_ERROR_NONE;
}

drvError_t halQueueSubscribe(unsigned int devid, unsigned int qid, unsigned int groupId, int type) {
  std::unique_lock<std::mutex> lock(queue_mutex);
  auto iter = queue_list.find(qid);
  if (iter == queue_list.end()) {
    return DRV_ERROR_QUEUE_NOT_CREATED;
  }
  return DRV_ERROR_NONE;
}

drvError_t halQueueUnsubF2NFEvent(unsigned int devId, unsigned int qid) {
  std::unique_lock<std::mutex> lock(queue_mutex);
  auto iter = queue_list.find(qid);
  if (iter == queue_list.end()) {
    return DRV_ERROR_QUEUE_NOT_CREATED;
  }
  return DRV_ERROR_NONE;
}

drvError_t halQueueUnsubscribe(unsigned int devId, unsigned int qid) {
  std::unique_lock<std::mutex> lock(queue_mutex);
  auto iter = queue_list.find(qid);
  if (iter == queue_list.end()) {
    return DRV_ERROR_QUEUE_NOT_CREATED;
  }
  return DRV_ERROR_NONE;
}

drvError_t halQueueSubEvent(struct QueueSubPara *subPara) {
  return halQueueSubscribe(subPara->devId, subPara->qid, subPara->groupId, subPara->queType);
}

drvError_t halQueueUnsubEvent(struct QueueUnsubPara *unsubPara) {
  return halQueueUnsubscribe(unsubPara->devId, unsubPara->qid);
}

pid_t drvDeviceGetBareTgid(void) {
  return 0;
}

drvError_t halEschedAttachDevice(unsigned int devId) {
  return DRV_ERROR_NONE;
}

drvError_t halEschedDettachDevice(unsigned int devId) {
  return DRV_ERROR_NONE;
}

int halBuffInit(BuffCfg *cfg) {
  return 0;
}

drvError_t halEschedCreateGrp(unsigned int devId, unsigned int grpId, GROUP_TYPE type) {
  return DRV_ERROR_NONE;
}

drvError_t halEschedCreateGrpEx(uint32_t devId, struct esched_grp_para *grpPara, unsigned int *grpId) {
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

drvError_t halEschedThreadSwapout(unsigned int devId, unsigned int grpId, unsigned int threadId) {
  return DRV_ERROR_NONE;
}

drvError_t halEschedGetEvent(unsigned int devId, unsigned int grpId, unsigned int threadId, EVENT_ID eventId,
                             struct event_info *event) {
  return DRV_ERROR_NO_EVENT;
}

drvError_t halQueuePeek(unsigned int devId, unsigned int qid, uint64_t *buf_len, int timeout) {
  std::unique_lock<std::mutex> lock(queue_mutex);
  auto queue_iter = queue_list.find(qid);
  if (queue_iter == queue_list.end()) {
    return DRV_ERROR_QUEUE_NOT_CREATED;
  }
  auto &queue_data = queue_iter->second;
  if (queue_data.empty()) {
    auto iter = g_link_queues.find(qid);
    if (iter != g_link_queues.end()) {
      uint32_t linked_qid = iter->second;
      lock.unlock();
      return halQueuePeek(devId, linked_qid, buf_len, timeout);
    }
    return DRV_ERROR_QUEUE_EMPTY;
  }
  MbufStImpl *buf_impl = reinterpret_cast<MbufStImpl *>(queue_data.front());
  *buf_len = buf_impl->mbuf_size;
  return DRV_ERROR_NONE;
}

int halMbufAlloc(uint64_t size, Mbuf **mbuf) {
  return halMbufAllocEx(size, 64, 0, 0, mbuf);
}

drvError_t halQueueDeQueueBuff(unsigned int devId, unsigned int qid, struct buff_iovec *vector, int timeout) {
  if (vector == nullptr || vector->count != 1) {
    return DRV_ERROR_PARA_ERROR;
  }
  std::unique_lock<std::mutex> lock(queue_mutex);
  auto queue_iter = queue_list.find(qid);
  if (queue_iter == queue_list.end()) {
    return DRV_ERROR_QUEUE_NOT_CREATED;
  }
  auto &queue_data = queue_iter->second;
  if (queue_data.empty()) {
    auto iter = g_link_queues.find(qid);
    if (iter != g_link_queues.end()) {
      uint32_t linked_qid = iter->second;
      lock.unlock();
      return halQueueDeQueueBuff(devId, linked_qid, vector, timeout);
    }
    return DRV_ERROR_QUEUE_EMPTY;
  }

  Mbuf *buf = queue_data.front();
  MbufStImpl *buf_impl = reinterpret_cast<MbufStImpl *>(buf);
  if (buf_impl->mbuf_size != vector->ptr[0].len) {
    return DRV_ERROR_PARA_ERROR;
  }
  memcpy_s(vector->context_base, vector->context_len, &buf_impl->reserve_head, 256);
  memcpy_s(vector[0].ptr->iovec_base, vector[0].ptr->len, &buf_impl->tensor_desc, buf_impl->mbuf_size);
  queue_data.pop_front();
  halMbufFree(buf);
  return DRV_ERROR_NONE;
}

drvError_t halQueueEnQueueBuff(unsigned int devId, unsigned int qid, struct buff_iovec *vector, int timeout) {
  std::unique_lock<std::mutex> lock(queue_mutex);
  auto iter = queue_list.find(qid);
  if (iter == queue_list.end()) {
    return DRV_ERROR_QUEUE_NOT_CREATED;
  }
  auto &queue_data = iter->second;
  if (queue_data.size() >= queue_depth[qid]) {
    return DRV_ERROR_QUEUE_FULL;
  }
  Mbuf *mbuf = nullptr;
  halMbufAlloc(vector[0].ptr->len, &mbuf);
  MbufStImpl *buf_impl = reinterpret_cast<MbufStImpl *>(mbuf);
  memcpy_s(&buf_impl->reserve_head, 256, vector->context_base, vector->context_len);
  memcpy_s(&buf_impl->tensor_desc, buf_impl->mbuf_size, vector[0].ptr->iovec_base, vector[0].ptr->len);
  queue_data.emplace_back((Mbuf *)mbuf);
  return DRV_ERROR_NONE;
}

drvError_t drvGetProcessSign(struct process_sign *sign) {
  return DRV_ERROR_NONE;
}

drvError_t halQueueQueryInfo(unsigned int devId, unsigned int qid, QueueInfo *queInfo) {
  std::unique_lock<std::mutex> lock(queue_mutex);
  auto queue_iter = queue_list.find(qid);
  if (queue_iter == queue_list.end()) {
    return DRV_ERROR_QUEUE_NOT_CREATED;
  }
  auto &queue_data = queue_iter->second;
  if (queue_data.empty()) {
    auto iter = g_link_queues.find(qid);
    if (iter != g_link_queues.end()) {
      uint32_t linked_qid = iter->second;
      lock.unlock();
      return halQueueQueryInfo(devId, linked_qid, queInfo);
    }
    queInfo->size = queue_data.size();
    return DRV_ERROR_NONE;
  }
  queInfo->size = queue_data.size();
  return DRV_ERROR_NONE;
}

drvError_t halQueryDevpid(struct halQueryDevpidInfo info, pid_t *dev_pid) {
  *dev_pid = 12345;
  return DRV_ERROR_NONE;
}
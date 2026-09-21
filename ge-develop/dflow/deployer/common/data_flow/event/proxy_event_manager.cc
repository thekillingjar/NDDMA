/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/data_flow/event/proxy_event_manager.h"
#include "framework/common/debug/log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "mmpa/mmpa_api.h"

namespace ge {
namespace {
const uint32_t kProxyEvent = 56;
const uint32_t kProxySubEventCreateGroup = 0U;
const uint32_t kProxySubEventAllocMbuf = 1U;
const uint32_t kProxySubEventFreeMbuf = 2U;
const uint32_t kProxySubEventCopyQMbuf = 3U;
const uint32_t kProxySubEventAddGroup = 4U;
const uint32_t kProxySubEventAllocCache = 5U;

struct ProxyMsgRsp {
  uint64_t mbufAddr;
  uint64_t dataAddr;
  int32_t retCode;
  char_t rsv[12];
};

struct ProxyMsgCreateGroup {
  uint64_t head;
  uint64_t size;
  char_t groupName[16];
  int64_t allocSize; // alloc size when create group, 0: alloc by size, -1: not alloc, >0: alloc by allocSize
};

struct ProxyMsgAllocCache {
  uint64_t head;
  uint64_t memSize;
  uint32_t allocMaxSize;
  char_t rsv[20];
};

struct ProxyMsgAddGroup {
  uint64_t head;
  uint32_t admin : 1;
  uint32_t read : 1;
  uint32_t write : 1;
  uint32_t alloc : 1;
  int32_t pid;
  char_t groupName[16];
  char_t rsv[8];
};

struct ProxyMsgAllocMbuf {
  uint64_t head;
  uint64_t size;
  char_t rsv[24];
};

struct ProxyMsgFreeMbuf {
  uint64_t head;
  uint64_t mbufAddr;
  char_t rsv[24];
};

struct ProxyMsgCopyQMbuf {
  uint64_t head;
  uint64_t destAddr;
  uint32_t destLen;
  uint32_t queueId;
  char_t rsv[16];
};
}

Status ProxyEventManager::GetProxyPid(int32_t device_id, int32_t &pid) {
  GELOGI("GetProxyPid begin, device_id = %d.", device_id);
  rtBindHostpidInfo_t info{};
  info.cpType = RT_DEV_PROCESS_CP1;
  info.hostPid = mmGetPid();
  info.chipId = device_id;
  GE_CHK_RT_RET(rtQueryDevPid(&info, &pid));
  GELOGI("GetProxyPid success, device_id = %d, pid = %d.", device_id, pid);
  return SUCCESS;
}

Status ProxyEventManager::SubmitEventSync(int32_t device_id,
                                          uint32_t sub_event_id,
                                          char_t *msg,
                                          size_t msg_len,
                                          rtEschedEventReply_t *ack) {
  int32_t proxy_pid = -1;
  GE_CHK_STATUS_RET(GetProxyPid(device_id, proxy_pid), "Failed to get proxy pid, device_id = %d.", device_id);
  uint32_t group_id = 0U;
  // need aicpu support GetRemoteGroupId
  rtEschedEventSummary_t event_info{};
  event_info.eventId = kProxyEvent;
  event_info.pid = proxy_pid;
  event_info.grpId = group_id;
  event_info.subeventId = sub_event_id;
  event_info.msg = msg;
  event_info.msgLen = msg_len;
  GE_CHK_RT_RET(rtEschedSubmitEventSync(device_id, &event_info, ack));
  GELOGD("[Send][Event] succeeded, device_id = %d, sub_event_id = %u", device_id, sub_event_id);
  return SUCCESS;
}

Status ProxyEventManager::CreateGroup(int32_t device_id, const std::string &group_name, uint64_t mem_size,
                                      bool alloc_when_create) {
  GELOGI("CreateGroup begin, device_id = %d, group_name = %s, mem_size = %lu kb.",
         device_id, group_name.c_str(), mem_size);
  ProxyMsgCreateGroup group_msg = {};
  auto ret = memcpy_s(group_msg.groupName, sizeof(group_msg.groupName), group_name.c_str(), group_name.length());
  GE_CHK_BOOL_RET_STATUS(ret == EOK, FAILED, "Failed to copy group name.");
  group_msg.size = mem_size;
  // 0 means alloc by size when create, -1 means no need alloc, other size means alloc by alloc size.
  group_msg.allocSize = alloc_when_create ? 0 : -1;
  ProxyMsgRsp rsp = {};
  rtEschedEventReply_t ack = {};
  ack.buf = reinterpret_cast<char_t *>(&rsp);
  ack.bufLen = sizeof(rsp);
  GE_CHK_STATUS_RET(SubmitEventSync(device_id,
                                    kProxySubEventCreateGroup,
                                    reinterpret_cast<char_t *>(&group_msg),
                                    sizeof(group_msg),
                                    &ack),
                    "Failed to submit create group event.");
  GE_CHK_STATUS_RET(static_cast<uint32_t>(rsp.retCode),
                    "Failed to process create group event, ret = %u.", rsp.retCode);
  GEEVENT("CreateGroup success, device_id = %d, group_name = %s, mem_size = %lu kb, alloc_when_create = %d.",
         device_id, group_name.c_str(), mem_size, static_cast<int32_t>(alloc_when_create));
  return SUCCESS;
}

Status ProxyEventManager::CreateGroup(int32_t device_id, const std::string &group_name, uint64_t mem_size,
                                      const std::vector<std::pair<uint64_t, uint32_t>> &pool_list) {
  // just one pool and limit is 0, alloc cache pool when create group.
  if (pool_list.empty() || (pool_list.size() == 1 && pool_list.front().second == 0)) {
    GE_CHK_STATUS_RET(CreateGroup(device_id, group_name, mem_size, true),
                      "Failed to create group and alloc cache pool, device_id = %d, "
                      "group_name = %s, mem_size = %lu kb",
                      device_id, group_name.c_str(), mem_size);
    return SUCCESS;
  }
  GE_CHK_STATUS_RET(CreateGroup(device_id, group_name, mem_size, false),
                    "Failed to create group, device_id = %d, group_name = %s, mem_size = %lu kb",
                    device_id, group_name.c_str(), mem_size);
  size_t idx = 0;
  for (const auto &pool : pool_list) {
    ProxyMsgAllocCache alloc_msg = {};
    alloc_msg.memSize = pool.first;
    alloc_msg.allocMaxSize = pool.second;
    ProxyMsgRsp rsp = {};
    rtEschedEventReply_t ack = {};
    ack.buf = reinterpret_cast<char_t *>(&rsp);
    ack.bufLen = sizeof(rsp);
    GE_CHK_STATUS_RET(SubmitEventSync(device_id, kProxySubEventAllocCache, reinterpret_cast<char_t *>(&alloc_msg),
                                      sizeof(alloc_msg), &ack),
                      "Failed to submit alloc cache event, device_id = %d, sub_event_id=%u.", device_id,
                      kProxySubEventAllocCache);
    GE_CHK_STATUS_RET(
        static_cast<uint32_t>(rsp.retCode),
        "Failed to process alloc cache event, ret = %u, device_id = %d, mem_size = %lu kb, alloc_max_size = %u kb.",
        rsp.retCode, device_id, pool.first, pool.second);
    GEEVENT("alloc cache pool[%zu] success, device_id = %d, mem_size = %lu kb, "
            "alloc_max_size = %u kb.", idx, device_id, pool.first, pool.second);
    ++idx;
  }
  GEEVENT("CreateGroup with cache pool success, device_id = %d, group_name = %s, "
          "mem_size = %lu kb, cache pool num = %zu.",
          device_id, group_name.c_str(), mem_size, pool_list.size());
  return SUCCESS;
}

Status ProxyEventManager::AddGroup(int32_t device_id,
                                   const std::string &group_name,
                                   pid_t pid,
                                   bool is_admin,
                                   bool is_alloc) {
  GELOGI("AddGroup begin, device_id = %d, group_name = %s, pid = %d.",
         device_id, group_name.c_str(), pid);
  ProxyMsgAddGroup group_msg = {};
  auto ret = memcpy_s(group_msg.groupName, sizeof(group_msg.groupName), group_name.c_str(), group_name.length());
  GE_CHK_BOOL_RET_STATUS(ret == EOK, FAILED, "Failed to copy group name.");
  group_msg.admin = is_admin ? 1 : 0;
  group_msg.alloc = is_alloc ? 1 : 0;
  group_msg.read = 1;
  group_msg.write = 1;
  group_msg.pid = pid;
  ProxyMsgRsp rsp = {};
  rtEschedEventReply_t ack = {};
  ack.buf = reinterpret_cast<char_t *>(&rsp);
  ack.bufLen = sizeof(rsp);
  GE_CHK_STATUS_RET(SubmitEventSync(device_id,
                                    kProxySubEventAddGroup,
                                    reinterpret_cast<char_t *>(&group_msg),
                                    sizeof(group_msg),
                                    &ack),
                    "Failed to submit add group event.");
  GE_CHK_STATUS_RET(static_cast<uint32_t>(rsp.retCode),
                    "Failed to process add group event, ret = %u.", rsp.retCode);
  GELOGI("AddGroup success, device_id = %d, group_name = %s, pid = %d.",
         device_id, group_name.c_str(), pid);
  return SUCCESS;
}

Status ProxyEventManager::AllocMbuf(int32_t device_id, uint64_t size, rtMbufPtr_t *mbuf, void **data_ptr) {
  GELOGI("AllocMbuf begin, device_id = %d, size = %lu.", device_id, size);
  ProxyMsgAllocMbuf mbuf_msg = {};
  mbuf_msg.size = size;
  ProxyMsgRsp rsp = {};
  rtEschedEventReply_t ack = {};
  ack.buf = reinterpret_cast<char_t *>(&rsp);
  ack.bufLen = sizeof(rsp);
  GE_CHK_STATUS_RET(SubmitEventSync(device_id,
                                    kProxySubEventAllocMbuf,
                                    reinterpret_cast<char_t *>(&mbuf_msg),
                                    sizeof(mbuf_msg),
                                    &ack),
                    "Failed to submit alloc mbuf event.");
  GE_CHK_STATUS_RET(static_cast<uint32_t>(rsp.retCode),
                    "Failed to process alloc mbuf event, ret = %d.", rsp.retCode);
  *mbuf = reinterpret_cast<rtMbufPtr_t>(reinterpret_cast<uintptr_t>(rsp.mbufAddr));
  *data_ptr = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(rsp.dataAddr));
  GELOGI("AllocMbuf success, device_id = %d, size = %lu.", device_id, size);
  return SUCCESS;
}

Status ProxyEventManager::FreeMbuf(int32_t device_id, rtMbufPtr_t mbuf) {
  GELOGI("FreeMbuf begin, device_id = %d.", device_id);
  ProxyMsgFreeMbuf mbuf_msg = {};
  mbuf_msg.mbufAddr = reinterpret_cast<uint64_t>(reinterpret_cast<uintptr_t>(mbuf));
  ProxyMsgRsp rsp = {};
  rtEschedEventReply_t ack = {};
  ack.buf = reinterpret_cast<char_t *>(&rsp);
  ack.bufLen = sizeof(rsp);
  GE_CHK_STATUS_RET(SubmitEventSync(device_id,
                                    kProxySubEventFreeMbuf,
                                    reinterpret_cast<char_t *>(&mbuf_msg),
                                    sizeof(mbuf_msg),
                                    &ack),
                    "Failed to submit alloc mbuf event.");
  GE_CHK_STATUS_RET(static_cast<uint32_t>(rsp.retCode),
                    "Failed to process alloc mbuf event, ret = %d.", rsp.retCode);
  GELOGI("FreeMbuf success, device_id = %d.", device_id);
  return SUCCESS;
}

Status ProxyEventManager::CopyQMbuf(int32_t device_id,
                                    uint64_t dest_addr,
                                    uint32_t dest_len,
                                    uint32_t queue_id) {
  GELOGI("CopyQMbuf begin, device_id = %d.", device_id);
  ProxyMsgCopyQMbuf mbuf_msg = {};
  mbuf_msg.destAddr = dest_addr;
  mbuf_msg.destLen = dest_len;
  mbuf_msg.queueId = queue_id;
  ProxyMsgRsp rsp = {};
  rtEschedEventReply_t ack = {};
  ack.buf = reinterpret_cast<char_t *>(&rsp);
  ack.bufLen = sizeof(rsp);
  GE_CHK_STATUS_RET(SubmitEventSync(device_id,
                                    kProxySubEventCopyQMbuf,
                                    reinterpret_cast<char_t *>(&mbuf_msg),
                                    sizeof(mbuf_msg),
                                    &ack),
                    "Failed to submit alloc mbuf event.");
  GE_CHK_STATUS_RET(static_cast<uint32_t>(rsp.retCode),
                    "Failed to process alloc mbuf event, ret = %d.", rsp.retCode);
  GELOGI("CopyQMbuf success, device_id = %d.", device_id);
  return SUCCESS;
}
}  // namespace ge

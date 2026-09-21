/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_HETEROGENEOUS_COMMON_DATA_FLOW_EVENT_PROXY_EVENT_MANAGER_H_
#define AIR_RUNTIME_HETEROGENEOUS_COMMON_DATA_FLOW_EVENT_PROXY_EVENT_MANAGER_H_

#include <string>
#include <vector>
#include "runtime/rt.h"
#include "ge/ge_api_error_codes.h"

namespace ge {
class ProxyEventManager {
 public:
  ProxyEventManager() = delete;
  ~ProxyEventManager() = delete;

  static Status GetProxyPid(int32_t device_id, int32_t &pid);
  static Status CreateGroup(int32_t device_id, const std::string &group_name, uint64_t mem_size,
                            bool alloc_when_create = true);
  static Status CreateGroup(int32_t device_id, const std::string &group_name, uint64_t mem_size,
                            const std::vector<std::pair<uint64_t, uint32_t>> &pool_list);
  static Status AddGroup(int32_t device_id, const std::string &group_name,
                         pid_t pid,
                         bool is_admin,
                         bool is_alloc);
  static Status AllocMbuf(int32_t device_id, uint64_t size, rtMbufPtr_t *mbuf, void **data_ptr);
  static Status FreeMbuf(int32_t device_id, rtMbufPtr_t mbuf);
  static Status CopyQMbuf(int32_t device_id,
                          uint64_t dest_addr,
                          uint32_t dest_len,
                          uint32_t queue_id);

 private:
  static Status SubmitEventSync(int32_t device_id,
                                uint32_t sub_event_id,
                                char_t *msg,
                                size_t msg_len,
                                rtEschedEventReply_t *ack);
};
}  // namespace ge

#endif  // AIR_RUNTIME_HETEROGENEOUS_COMMON_DATA_FLOW_EVENT_PROXY_EVENT_MANAGER_H_

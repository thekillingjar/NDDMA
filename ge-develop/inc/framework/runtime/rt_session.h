/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_SESSION_H
#define AIR_RUNTIME_SESSION_H
#include "rt_var_manager.h"
#include "stream_allocator.h"
#include "event_allocator.h"
#include "notify_allocator.h"

namespace gert {
class RtSession {
 public:
  explicit RtSession(uint64_t session_id = 0UL) : session_id_(session_id) {
    stream_allocator_ = std::unique_ptr<StreamAllocator>(new (std::nothrow) StreamAllocator());
    event_allocator_ = std::unique_ptr<EventAllocator>(new (std::nothrow) EventAllocator());
    notify_allocator_ = std::unique_ptr<NotifyAllocator>(new (std::nothrow) NotifyAllocator());
  }
  uint64_t GetSessionId() const {
    return session_id_;
  }
  void SetSessionId(uint64_t session_id) {
    session_id_ = session_id;
  }
  const RtVarManager *GetVarManager() const {
    return var_manager_;
  }
  void SetVarManager(RtVarManager *var_manager) {
    var_manager_ = var_manager;
  }

  StreamAllocator *GetStreamAllocator() const {
    return stream_allocator_.get();
  }

  EventAllocator *GetEventAllocator() const {
    return event_allocator_.get();
  }

  NotifyAllocator *GetNotifyAllocator() const {
    return notify_allocator_.get();
  }

  void SetExternalVar(void *external_var_addr, uint64_t external_var_size) {
    external_var_addr_ = external_var_addr;
    external_var_size_ = external_var_size;
  }

  void GetExternalVar(void *&external_var_addr, uint64_t &external_var_size) const {
    external_var_addr = external_var_addr_;
    external_var_size = external_var_size_;
  }

  void DestroyResources() const;

 private:
  uint64_t session_id_;
  RtVarManager *var_manager_{nullptr};
  void* external_var_addr_{nullptr};
  uint64_t external_var_size_{0U};
  std::unique_ptr<StreamAllocator> stream_allocator_;
  std::unique_ptr<EventAllocator> event_allocator_;
  std::unique_ptr<NotifyAllocator> notify_allocator_;
};
}  // namespace gert
#endif  // AIR_RUNTIME_SESSION_H

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/model/external_allocator_manager.h"
#include "framework/common/debug/ge_log.h"

namespace ge {
std::shared_mutex ExternalAllocatorManager::stream_to_external_allocator_Mutex_;
std::map<const void *const, AllocatorPtr> ExternalAllocatorManager::stream_to_external_allocator_;

void ExternalAllocatorManager::SetExternalAllocator(const void *const stream, AllocatorPtr allocator) {
  const std::unique_lock<std::shared_mutex> lk(stream_to_external_allocator_Mutex_);
  stream_to_external_allocator_[stream] = allocator;
}

void ExternalAllocatorManager::DeleteExternalAllocator(const void *const stream) {
  const std::unique_lock<std::shared_mutex> lk(stream_to_external_allocator_Mutex_);
  (void)stream_to_external_allocator_.erase(stream);
}

AllocatorPtr ExternalAllocatorManager::GetExternalAllocator(const void *const stream) {
  const std::shared_lock<std::shared_mutex> lk(stream_to_external_allocator_Mutex_);
  const auto iter = stream_to_external_allocator_.find(stream);
  if (iter != stream_to_external_allocator_.end()) {
    GELOGI("Get external allocator success, stream:%p, allocator:%p.", stream, iter->second.get());
    return iter->second;
  }
  return nullptr;
}
}  // namespace ge

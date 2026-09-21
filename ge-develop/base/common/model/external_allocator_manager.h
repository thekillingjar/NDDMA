/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_EXTERNAL_ALLOCATOR_MANAGER_H
#define GE_COMMON_EXTERNAL_ALLOCATOR_MANAGER_H

#include <map>
#include <shared_mutex>
#include "ge/ge_allocator.h"
#include "framework/runtime/stream_allocator.h"
#include "framework/runtime/event_allocator.h"
#include "framework/runtime/notify_allocator.h"

namespace ge {
struct DevResourceAllocator {
  gert::EventAllocator event_allocator;
  gert::StreamAllocator stream_allocator;
  gert::NotifyAllocator notify_allocator;
};
class ExternalAllocatorManager {
 public:
  static void SetExternalAllocator(const void *const stream, AllocatorPtr allocator);
  static void DeleteExternalAllocator(const void *const stream);
  static AllocatorPtr GetExternalAllocator(const void *const stream);
 private:
  static std::shared_mutex stream_to_external_allocator_Mutex_;
  static std::map<const void *const, AllocatorPtr> stream_to_external_allocator_;
};
}  // namespace ge
#endif // GE_COMMON_EXECUTOR_H

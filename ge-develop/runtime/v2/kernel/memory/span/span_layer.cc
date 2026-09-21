/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel/memory/span/span_layer.h"
#include "kernel/memory/span/span_allocator.h"
#include "kernel/memory/device/device_allocator.h"
#include "framework/common/debug/ge_log.h"

namespace gert {
void SpanLayer::Recycle(SpanAllocator &span_allocator, DeviceAllocator &device) {
  for (auto span = free_link_.begin(), tmp_span = ++free_link_.begin(); span != free_link_.end();) {
    if (span->IsNewVaSpan()) {
      (void) device.GetExpandableAllocator().FreePhysicalMemory(reinterpret_cast<MemAddr>(span->GetAddr()),
                                                                span->GetPaList(), false, true);
      free_link_.remove(*span);
      span_allocator.Free(*span);
    } else if (device.GetExpandableAllocator().IsValidVirtualAddr(reinterpret_cast<MemAddr>(span->GetAddr()))) {
      (void) device.GetExpandableAllocator().FreePhysicalMemory(reinterpret_cast<MemAddr>(span->GetAddr()),
                                                                span->GetSize(), false, true);
    } else {
      if (!span->HasSplited()) {
        free_link_.remove(*span);
        device.Free(span->GetBlockAddr());
        span_allocator.Free(*span);
      }
    }
    span = tmp_span;
    ++tmp_span;
  }
}

void SpanLayer::Release(SpanAllocator &span_allocator, DeviceAllocator &device) {
  while (!free_link_.empty()) {
    auto span = free_link_.pop_front();
    if (span == nullptr) {
      continue;
    }
    if (span->HasSplited()) {
      GELOGE(ge::INTERNAL_ERROR,
             "[SpanLayer]: releasing splited span [addr : %p, len : %u]", span->GetBlockAddr(), span->GetPageLen());
    }
    if (span->IsBuddyHeader()) {
      if (span->IsNewVaSpan()) {
        (void) device.GetExpandableAllocator().FreePhysicalMemory(reinterpret_cast<MemAddr>(span->GetAddr()),
                                                                  span->GetPaList());
      } else if (device.GetExpandableAllocator().IsValidVirtualAddr(reinterpret_cast<MemAddr>(span->GetAddr()))) {
        (void) device.GetExpandableAllocator().FreePhysicalMemory(reinterpret_cast<MemAddr>(span->GetAddr()),
                                                                  span->GetSize());
      } else {
        device.Free(span->GetBlockAddr());
      }
    }
    span_allocator.Free(*span);
  }
}
}


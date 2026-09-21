/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/runtime/stream_allocator.h"

#include "common/checker.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"

namespace gert {
StreamAllocator::StreamAllocator(int32_t priority, uint32_t flags)
    : streams_holder_(ContinuousVector::Create<rtStream_t>(kMaxStreamNum)),
      default_priority_(priority),
      default_flags_(flags) {
  const auto streams = Streams();
  if (streams != nullptr) {
    streams->MutableData()[0U] = nullptr;
    (void)streams->SetSize(1U);
  }
}

StreamAllocator::~StreamAllocator() {
  const auto streams = Streams();
  for (size_t i = 1U; i < streams->GetSize(); ++i) {
    (void)rtStreamDestroy(streams->MutableData()[i]);
  }
  (void)streams->SetSize(0U);
}

TypedContinuousVector<rtStream_t> *StreamAllocator::AcquireStreams(const size_t stream_num) const {
  GE_ASSERT_TRUE(stream_num < kMaxStreamNum);

  auto streams = Streams();
  for (size_t i = streams->GetSize(); i < stream_num; ++i) {
    rtStream_t stream = nullptr;
    GE_ASSERT_RT_OK(rtStreamCreateWithFlags(&stream, default_priority_, default_flags_));
    streams->MutableData()[i] = stream;
    GE_ASSERT_SUCCESS(streams->SetSize(i + 1U));
  }
  return streams;
}

TypedContinuousVector<rtStream_t> *StreamAllocator::Streams() const {
  return reinterpret_cast<TypedContinuousVector<rtStream_t> *>(streams_holder_.get());
}
}  // namespace gert

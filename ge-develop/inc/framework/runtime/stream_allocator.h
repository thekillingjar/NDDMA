/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_STREAM_ALLOCATOR_H_
#define AIR_CXX_RUNTIME_STREAM_ALLOCATOR_H_

#include <memory>
#include "runtime/stream.h"
#include "common/ge_visibility.h"
#include "framework/common/ge_inner_error_codes.h"
#include "exe_graph/runtime/continuous_vector.h"
namespace gert {
class VISIBILITY_EXPORT StreamAllocator {
 public:
  static constexpr size_t kMaxStreamNum = 2024U;
  explicit StreamAllocator(int32_t priority = RT_STREAM_PRIORITY_DEFAULT, uint32_t flags = RT_STREAM_DEFAULT);
  StreamAllocator(const StreamAllocator &) = delete;
  StreamAllocator &operator=(const StreamAllocator &) = delete;

  ~StreamAllocator();

  TypedContinuousVector<rtStream_t> *AcquireStreams(size_t stream_num) const;

 private:
  TypedContinuousVector<rtStream_t> *Streams() const;

 private:
  std::unique_ptr<uint8_t[]> streams_holder_;
  int32_t default_priority_;
  uint32_t default_flags_;
};
}  // namespace gert

#endif  // AIR_CXX_RUNTIME_STREAM_ALLOCATOR_H_

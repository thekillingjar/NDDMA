/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_SPSC_QUEUE_H
#define AIR_CXX_RUNTIME_V2_SPSC_QUEUE_H

#include <atomic>
#include "core/executor/multi_thread_topological/executor/schedule/queue/cache_utility.h"

namespace gert {
template <typename T>
struct SpscQueue {
  explicit SpscQueue(uint32_t sizeLog2)
      : size_(1 << sizeLog2), mask_(size_ - 1U), items_(new (std::nothrow) T[size_]), head_(0U), tail_(0U) {}

  ~SpscQueue() {
    if (!std::is_trivially_destructible<T>::value) {
      size_t head = head_;
      size_t tail = tail_;
      while (head != tail) {
        items_[(head++) & mask_].~T();
      }
    }

    delete[] items_;
  }

  SpscQueue(const SpscQueue &) = delete;
  SpscQueue &operator=(const SpscQueue &) = delete;

  template <class... Args>
  bool Push(Args &&...itemArgs) {
    const auto tail = tail_.load(std::memory_order_relaxed);

    auto nextTail = tail + 1;
    if (nextTail - head_.load(std::memory_order_acquire) <= GetCapacity()) {
      new (&items_[tail & mask_]) T(std::forward<Args>(itemArgs)...);
      tail_.store(nextTail, std::memory_order_release);
      return true;
    }
    return false;
  }

  bool Pop(T &item) {
    const auto head = head_.load(std::memory_order_relaxed);
    if (head == tail_.load(std::memory_order_acquire)) {
      return false;
    }

    auto nextHead = head + 1;
    item = std::move(items_[head & mask_]);
    items_[head].~T();
    head_.store(nextHead, std::memory_order_release);
    return true;
  }

  bool IsEmpty() const {
    return head_.load(std::memory_order_acquire) == tail_.load(std::memory_order_acquire);
  }

  bool IsFull() const {
    return GetSize() >= GetCapacity();
  }

  size_t GetSize() const {
    return tail_.load(std::memory_order_acquire) - head_.load(std::memory_order_acquire);
  }

  size_t GetCapacity() const {
    return mask_;
  }

 private:
  using QueueIndex = std::atomic<uint32_t>;

  char pad_infer_before_[kHardwareDestructiveInterferenceSize]{};
  const uint32_t size_;
  const uint32_t mask_;
  T *const items_;
  // used by consumer
  alignas(kHardwareDestructiveInterferenceSize) QueueIndex head_;
  // used by producer
  alignas(kHardwareDestructiveInterferenceSize) QueueIndex tail_;
  char pad_infer_after_[kHardwareDestructiveInterferenceSize - sizeof(QueueIndex)]{};
};
}  // namespace gert

#endif // AIR_CXX_RUNTIME_V2_SPSC_QUEUE_H

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_LOWERING_PASS_UTILS_DEDUPLICATE_QUEUE_H_
#define AIR_CXX_RUNTIME_V2_LOWERING_PASS_UTILS_DEDUPLICATE_QUEUE_H_
#include <queue>
#include <set>
namespace gert {
namespace bg {
template <typename T>
class DeduplicateQueue {
 public:
  void push(const T &element) {
    if (elements_set_.insert(element).second) {
      elements_queue_.push(element);
    }
  }
  T pop() {
    auto element = std::move(elements_queue_.front());
    elements_queue_.pop();
    elements_set_.erase(element);
    return element;
  }
  bool empty() const {
    return elements_queue_.empty();
  }

 private:
  std::set<T> elements_set_;
  std::queue<T> elements_queue_;
};
}  // namespace bg
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_LOWERING_PASS_UTILS_DEDUPLICATE_QUEUE_H_

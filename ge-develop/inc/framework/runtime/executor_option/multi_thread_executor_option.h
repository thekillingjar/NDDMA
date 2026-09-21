/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_MULTI_THREAD_EXECUTOR_OPTION_H
#define AIR_CXX_MULTI_THREAD_EXECUTOR_OPTION_H

#include <cstddef>
#include "framework/runtime/executor_option/executor_option.h"

namespace gert {
constexpr size_t kLeastThreadNumber = 2U;  // least new thread num, one for normal worker, one for memory worker

class VISIBILITY_EXPORT MultiThreadExecutorOption : public ExecutorOption {
 public:
  MultiThreadExecutorOption() : MultiThreadExecutorOption(kLeastThreadNumber) {}
  explicit MultiThreadExecutorOption(size_t thread_num)
      : ExecutorOption(ExecutorType::kTopologicalMultiThread), thread_num_(thread_num) {}
  MultiThreadExecutorOption(ExecutorType executor_type, size_t thread_num)
      : ExecutorOption(executor_type), thread_num_(thread_num) {}

  size_t GetThreadNum() const {
    return thread_num_;
  }

 private:
  /**
   * 多线程数量
   * 取值范围：2 <= thread_num
   */
  size_t thread_num_;
};
}  // namespace gert
#endif  // AIR_CXX_MULTI_THREAD_EXECUTOR_OPTION_H

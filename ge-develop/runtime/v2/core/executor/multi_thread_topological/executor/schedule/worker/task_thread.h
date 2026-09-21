/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_TASK_THREAD_H
#define AIR_CXX_RUNTIME_V2_TASK_THREAD_H

#include <thread>
#include <chrono>
#include <string>
#include <memory>
#include <functional>
#include "task_thread_mode.h"
#include "core/executor/multi_thread_topological/executor/schedule/task/exec_task.h"

namespace gert {
class TaskThread {
 public:
  using Runnable = std::function<void()>;

  explicit TaskThread(std::string name, TaskThreadMode threadMode);
  bool Start(Runnable runnable);
  void Stop();
  void Dump() const;

  void Await() {
    if (thread_mode_ == TaskThreadMode::URGENT) {
      return UrgentAwait();
    }
    if (thread_mode_ == TaskThreadMode::MODERATE) {
      return Yield();
    }
    return SleepUs(1);
  }

 public:
  void OnTaskPushFailed() {
    push_nil_count_++;
  }

  void OnTaskPopFailed() {
    pop_nil_count_++;
  }

  void OnTaskExecuted() {
    task_Executed_count_++;
  }

  void OnTaskExecFailed() {
    task_failed_count_++;
  }

  void OnTaskLeaked(ExecTask *leakedTask) {
    leaked_task_ = leakedTask;
  }

  ExecTask *FetchLeakedTask() {
    auto task = leaked_task_;
    leaked_task_ = nullptr;
    return task;
  }

 private:
  void UrgentAwait() noexcept {
    await_count_++;
  }

  void Yield() noexcept {
    yield_count_++;
    std::this_thread::yield();
  }

  void SleepNs(size_t ns) {
    sleep_ns_count_ += ns;
    std::this_thread::sleep_for(std::chrono::nanoseconds(ns));
  }

  void SleepUs(size_t us) {
    sleep_us_count_ += us;
    std::this_thread::sleep_for(std::chrono::microseconds(us));
  }

 private:
  size_t push_nil_count_{0};
  size_t pop_nil_count_{0};

  size_t task_failed_count_{0};
  size_t task_Executed_count_{0};

  size_t await_count_{0};
  size_t yield_count_{0};
  size_t sleep_ns_count_{0};
  size_t sleep_us_count_{0};

 private:
  std::string name_;
  TaskThreadMode thread_mode_;
  std::unique_ptr<std::thread> thread_;

 private:
  ExecTask *leaked_task_{nullptr};
};
}  // namespace gert

#endif // AIR_CXX_RUNTIME_V2_TASK_THREAD_H

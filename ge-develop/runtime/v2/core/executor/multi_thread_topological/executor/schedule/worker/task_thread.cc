/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_thread.h"
#include "framework/common/debug/ge_log.h"
#include "common/compile_profiling/ge_call_wrapper.h"

namespace gert {
TaskThread::TaskThread(std::string name, TaskThreadMode threadMode) : name_(name), thread_mode_(threadMode) {}

bool TaskThread::Start(Runnable runnable) {
  if (thread_) {
    return false;
  }

  thread_.reset(new (std::nothrow) std::thread([=](){
    SET_THREAD_NAME(pthread_self(), "ge_exe_tasktd");
    runnable();
  }));
  return thread_ != nullptr;
}

void TaskThread::Stop() {
  if (!thread_) {
    return;
  }

  if (thread_->joinable()) {
    thread_->join();
  }
  thread_.reset(nullptr);
}

void TaskThread::Dump() const {
  GEEVENT("        |-- Task Thread [ name : %s, mode : %s ]", name_.c_str(), TaskThreadMode_ToString(thread_mode_));

  GEEVENT("            |-- exec succ = %ld, exec failed = %ld", task_Executed_count_, task_failed_count_);

  GEEVENT("            |-- pop none = %ld, push none = %ld, has leak task = %s", pop_nil_count_, push_nil_count_,
          leaked_task_ ? "true" : "false");

  GEEVENT("            |-- await count = %ld, yield count = %ld, sleep ns = %ld, sleep ms = %ld", await_count_,
          yield_count_, sleep_ns_count_, sleep_us_count_);
}

const char *TaskThreadMode_ToString(TaskThreadMode mode) {
  switch (mode) {
    case TaskThreadMode::URGENT:
      return "urgent";
    case TaskThreadMode::MODERATE:
      return "moderate";
    case TaskThreadMode::LOW_LOAD:
      return "low_load";
    default:
      break;
  }
  return "unknown";
}
}  // namespace gert

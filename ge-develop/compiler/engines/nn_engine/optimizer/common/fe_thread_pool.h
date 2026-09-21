/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FE_THREAD_POOL_H_
#define FE_THREAD_POOL_H_

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <queue>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>
#include <string>

#include "common/fe_log.h"

namespace fe {
static constexpr size_t DEFAULT_THREAD_NUM = 8;

using ThreadTask = std::function<void()>;

class ThreadPool {
 public:
  explicit ThreadPool(const std::string &thread_name_prefix, const uint32_t size = DEFAULT_THREAD_NUM);
  ~ThreadPool();

  template <class Func, class... Args>
  auto commit(Func &&func, Args &&... args) -> std::future<decltype(func(args...))> {
    FE_LOGD("commit run task enter.");
    using retType = decltype(func(args...));
    std::future<retType> fail_future;
    if (is_stoped_.load()) {
      FE_LOGE("thread pool has been stopped.");
      return fail_future;
    }

    const auto bindFunc = std::bind(std::forward<Func>(func), std::forward<Args>(args)...);
    const auto task = std::make_shared<std::packaged_task<retType()>>(bindFunc);
    if (task == nullptr) {
      FE_LOGE("Make shared failed.");
      return fail_future;
    }
    std::future<retType> future = task->get_future();
    {
      const std::lock_guard<std::mutex> lock{m_lock_};
      tasks_.emplace([task]() { (*task)(); });
    }
    cond_var_.notify_one();
    FE_LOGD("commit run task end");
    return future;
  }

  static void ThreadFunc(ThreadPool *const thread_pool, uint32_t i, const std::string &thread_name_prefix);

 private:
  std::vector<std::thread> pool_;
  std::queue<ThreadTask> tasks_;
  std::mutex m_lock_;
  std::condition_variable cond_var_;
  std::atomic<bool> is_stoped_;
  std::atomic<uint32_t> idle_thrd_num_;
};
}

#endif

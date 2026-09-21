/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PARSER_COMMON_THREAD_POOL_H_
#define PARSER_COMMON_THREAD_POOL_H_

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "ge/ge_api_error_codes.h"
#include "parser/common/acl_graph_parser_util.h"

namespace ge {
namespace parser {
using ThreadTask = std::function<void()>;

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ThreadPool {
 public:
  explicit ThreadPool(uint32_t size = 4);
  explicit ThreadPool(std::string thread_name_prefix, uint32_t size = 4U);
  ~ThreadPool();

  template <class Func, class... Args>
  auto commit(Func &&func, Args &&... args) -> std::future<decltype(func(args...))> {
    GELOGD("commit run task enter.");
    using retType = decltype(func(args...));
    std::future<retType> fail_future;
    if (is_stoped_.load()) {
      GELOGE(ge::FAILED, "thread pool has been stopped.");
      return fail_future;
    }

    auto bindFunc = std::bind(std::forward<Func>(func), std::forward<Args>(args)...);
    auto task = ge::parser::MakeShared<std::packaged_task<retType()>>(bindFunc);
    if (task == nullptr) {
      GELOGE(ge::FAILED, "Make shared failed.");
      return fail_future;
    }
    std::future<retType> future = task->get_future();
    {
      std::lock_guard<std::mutex> lock{m_lock_};
      tasks_.emplace([task]() { (*task)(); });
    }
    cond_var_.notify_one();
    GELOGD("commit run task end");
    return future;
  }

  static void ThreadFunc(ThreadPool *const thread_pool, uint32_t thread_idx);

 private:
  std::string thread_name_prefix_;
  std::vector<std::thread> pool_;
  std::queue<ThreadTask> tasks_;
  std::mutex m_lock_;
  std::condition_variable cond_var_;
  std::atomic<bool> is_stoped_;
  std::atomic<uint32_t> idle_thrd_num_;
};
}  // namespace parser
}  // namespace ge

#endif  // PARSER_COMMON_THREAD_POOL_H_

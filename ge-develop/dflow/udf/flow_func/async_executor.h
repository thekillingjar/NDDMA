/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLOW_FUNC_ASYNC_EXECUTOR_H
#define FLOW_FUNC_ASYNC_EXECUTOR_H

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <queue>
#include <thread>
#include <vector>
#include "common/udf_log.h"
#include "flow_func/flow_func_defines.h"

namespace FlowFunc {
using ThreadTask = std::function<void()>;

class FLOW_FUNC_VISIBILITY AsyncExecutor {
public:
    explicit AsyncExecutor(std::string thread_name_prefix, const uint32_t thread_num = 1U,
        const std::function<int32_t()> &thread_init_func = nullptr);
    ~AsyncExecutor();

    template <class Func, class... Args>
    auto Commit(Func &&func, Args &&... args) -> std::future<decltype(func(args...))> {
        UDF_LOG_DEBUG("commit run task enter.");
        using retType = decltype(func(args...));
        std::future<retType> fail_future;
        if (is_stopped_.load()) {
            UDF_LOG_ERROR("async executor has been stopped.");
            return fail_future;
        }

        const auto bind_func = std::bind(std::forward<Func>(func), std::forward<Args>(args)...);
        try {
            const auto task = std::make_shared<std::packaged_task<retType()>>(bind_func);
            if (task == nullptr) {
                UDF_LOG_ERROR("Make shared failed.");
                return fail_future;
            }
            std::future<retType> future = task->get_future();
            {
                const std::lock_guard<std::mutex> lock{m_lock_};
                tasks_.emplace([task]() { (*task)(); });
            }
            cond_var_.notify_one();
            UDF_LOG_DEBUG("commit run task end");
            return future;
        } catch (std::exception &e) {
            UDF_LOG_ERROR("new packaged task failed, error=%s.", e.what());
            return fail_future;
        }
    }

    void Stop() noexcept;

    static void ThreadFunc(AsyncExecutor *const async_executor, uint32_t thread_idx);

private:
    std::string thread_name_prefix_;
    std::vector<std::thread> pool_;
    std::queue<ThreadTask> tasks_;
    std::mutex m_lock_;
    std::condition_variable cond_var_;
    std::atomic<bool> is_stopped_;
    std::atomic<uint32_t> idle_thread_num_;
    std::function<int32_t()> thread_init_func_;
};
}
#endif  // FLOW_FUNC_ASYNC_EXECUTOR_H

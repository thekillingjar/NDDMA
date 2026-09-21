/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "async_executor.h"

namespace FlowFunc {
AsyncExecutor::AsyncExecutor(std::string thread_name_prefix, const uint32_t thread_num,
    const std::function<int32_t()> &thread_init_func) : thread_name_prefix_(std::move(thread_name_prefix)), is_stopped_(false),
    thread_init_func_(thread_init_func) {
    idle_thread_num_ = (thread_num < 1U) ? 1U : thread_num;
    for (uint32_t i = 0U; i < idle_thread_num_; ++i) {
        pool_.emplace_back(&AsyncExecutor::ThreadFunc, this, i);
    }
}

AsyncExecutor::~AsyncExecutor() {
    Stop();
    for (std::thread &thd : pool_) {
        if (thd.joinable()) {
            try {
                thd.join();
            } catch (const std::system_error &) {
                UDF_LOG_WARN("system_error");
            } catch (...) {
                UDF_LOG_WARN("exception");
            }
        }
    }
}

void AsyncExecutor::Stop() noexcept {
    is_stopped_.store(true);
    {
        const std::unique_lock<std::mutex> lock{m_lock_};
        cond_var_.notify_all();
    }
}

void AsyncExecutor::ThreadFunc(AsyncExecutor *const async_executor, uint32_t thread_idx) {
    if (async_executor == nullptr) {
        return;
    }
    std::string thread_name = async_executor->thread_name_prefix_ + std::to_string(thread_idx);
    int32_t ret = pthread_setname_np(pthread_self(), thread_name.c_str());
    if (ret != 0) {
        UDF_LOG_WARN("pthread_setname_np failed, thread_name=%s, ret=%d", thread_name.c_str(), ret);
    }
    if (async_executor->thread_init_func_ != nullptr) {
        int32_t init_ret = async_executor->thread_init_func_();
        if (init_ret != FLOW_FUNC_SUCCESS) {
            async_executor->Stop();
            return;
        }
    }
    while (!async_executor->is_stopped_) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock{async_executor->m_lock_};
            async_executor->cond_var_.wait(
                lock, [async_executor]() -> bool {
                    return async_executor->is_stopped_.load() || (!async_executor->tasks_.empty());
                });
            if (async_executor->is_stopped_ && async_executor->tasks_.empty()) {
                return;
            }
            task = std::move(async_executor->tasks_.front());
            async_executor->tasks_.pop();
        }
        --async_executor->idle_thread_num_;
        task();
        ++async_executor->idle_thread_num_;
    }
}


}  // namespace FlowFunc

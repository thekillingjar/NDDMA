/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMON_COUNT_DOWN_LATCH_H
#define COMMON_COUNT_DOWN_LATCH_H

#include <condition_variable>
#include <mutex>
#include <cstdint>

namespace FlowFunc {
class CountDownLatch {
public:
    explicit CountDownLatch(int32_t count = 1) : count_(count) {}

    void ResetCount(int32_t count) {
        std::unique_lock<std::mutex> lock(mutex_);
        count_ = count;
    }

    void Await() {
        std::unique_lock<std::mutex> lock(mutex_);
        while (count_ > 0) {
            condition_.wait(lock);
        }
    }

    void CountDown() {
        std::unique_lock<std::mutex> lock(mutex_);
        --count_;
        if (count_ == 0) {
            condition_.notify_all();
        }
    }

private:
    volatile int32_t count_;
    std::mutex mutex_;
    std::condition_variable condition_;
};
}  // namespace FlowFunc
#endif // COMMON_COUNT_DOWN_LATCH_H

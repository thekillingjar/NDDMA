/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMON_SCOPE_GUARD_H
#define COMMON_SCOPE_GUARD_H
#include <functional>

namespace FlowFunc {
class ScopeGuard {
public:
    explicit ScopeGuard(std::function<void()> callback)
        : exit_callback_(callback) {}

    ~ScopeGuard() noexcept {
        if (exit_callback_ != nullptr) {
            exit_callback_();
        }
    }

    void ReleaseGuard() {
        exit_callback_ = nullptr;
    }

    ScopeGuard(ScopeGuard const &) = delete;

    ScopeGuard &operator=(ScopeGuard const &) = delete;

    ScopeGuard(ScopeGuard &&) = delete;

    ScopeGuard &operator=(ScopeGuard &&) = delete;

private:
    std::function<void()> exit_callback_;
};
}

#endif // COMMON_SCOPE_GUARD_H
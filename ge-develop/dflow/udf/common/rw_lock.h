/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UDF_COMMON_RW_LOCK_H
#define UDF_COMMON_RW_LOCK_H

#include "mmpa/mmpa_api.h"

namespace FlowFunc {
class ReadProtect {
public:
    explicit ReadProtect(mmRWLock_t *const rLock) : lock_(rLock) {
        if (lock_ == nullptr) {
            return;
        }
        (void)mmRWLockRDLock(lock_);
    }

    virtual ~ReadProtect() {
        if (lock_ == nullptr) {
            return;
        }
        (void)mmRDLockUnLock(lock_);
    }

private:
    mmRWLock_t *lock_;
};

class WriteProtect {
public:
    explicit WriteProtect(mmRWLock_t *const wLock) : lock_(wLock) {
        if (lock_ == nullptr) {
            return;
        }
        (void)mmRWLockWRLock(lock_);
    }

    virtual ~WriteProtect() {
        if (lock_ == nullptr) {
            return;
        }
        (void)mmWRLockUnLock(lock_);
    }

private:
    mmRWLock_t *lock_;
};
}  // namespace FlowFunc
#endif  // UDF_COMMON_RW_LOCK_H

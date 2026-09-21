/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ge_tensor_cache.h"

namespace acl {
constexpr size_t GE_TENSOR_CACHE_MAX_SIZE = 10000U;

GeTensorDescCache& GeTensorDescCache::GetInstance()
{
    static GeTensorDescCache inst;
    return inst;
}

GeTensorDescCache::~GeTensorDescCache()
{
    for (size_t i = 0U; i < descCache_.size(); ++i) {
        if (descCache_[i] != nullptr) {
            delete descCache_[i];
        }
    }
    descCache_.clear();
}

GeTensorDescVecPtr GeTensorDescCache::GetDescVecPtr(const size_t size)
{
    GeTensorDescVecPtr ptr = nullptr;
    {
        const std::lock_guard<std::mutex> lk(cacheMutex_);
        for (size_t i = 0U; i < descCache_.size(); ++i) {
            GeTensorDescVecPtr &it = descCache_[i];
            if (it == nullptr) {
                continue;
            }
            if (it->size() != size) {
                continue;
            }
            ptr = it;
            it = nullptr;
            return ptr;
        }
    }
    ptr = new(std::nothrow) std::vector<ge::GeTensorDesc>(size);
    return ptr;
}

void GeTensorDescCache::ReleaseDescVecPtr(const GeTensorDescVecPtr ptr)
{
    const std::lock_guard<std::mutex> lk(cacheMutex_);
    for (size_t i = 0U; i < descCache_.size(); ++i) {
        GeTensorDescVecPtr &it = descCache_[i];
        if (it == nullptr) {
            it = ptr;
            return;
        }
    }
    if (descCache_.size() >= GE_TENSOR_CACHE_MAX_SIZE) {
        delete ptr;
    } else {
        descCache_.emplace_back(ptr);
    }
}
} // namespace acl

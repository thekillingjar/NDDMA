/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "resource_manager.h"
#include "common/common_inner.h"
#include "common/log_inner.h"


namespace acl {
ResourceManager::ResourceManager(const aclrtContext context) : context_(context) {}

ResourceManager::~ResourceManager()
{
    ACL_LOG_INFO("ResourceManager::~ResourceManager IN");
    if (current_mem_ == nullptr) {
        return;
    }

    aclrtContext oldCtx = nullptr;
    (void) aclrtGetCurrentContext(&oldCtx);
    (void) aclrtSetCurrentContext(context_);
    if (current_mem_ != nullptr) {
        (void) aclrtFree(current_mem_);
    }
    for (const auto addr : pending_mem_) {
        (void) aclrtFree(addr);
    }
    (void) aclrtSetCurrentContext(oldCtx);
}

aclError ResourceManager::GetMemory(void **const address, const size_t size)
{
    ACL_REQUIRES_NOT_NULL(address);
    if (current_mem_size_ >= size) {
        ACL_LOG_DEBUG("current memory size[%zu] should be smaller than required memory size[%zu]",
            current_mem_size_, size);
        *address = current_mem_;
        return ACL_SUCCESS;
    }

    ACL_REQUIRES_OK(aclrtMalloc(address, size, ACL_MEM_MALLOC_NORMAL_ONLY));
    pending_mem_.push_back(current_mem_);
    current_mem_ = *address;
    current_mem_size_ = size;

    return ACL_SUCCESS;
}
} // namespace acl

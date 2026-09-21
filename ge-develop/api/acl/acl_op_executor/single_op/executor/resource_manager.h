/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRUNK_PROJ_RESOURCE_MANAGER_H
#define TRUNK_PROJ_RESOURCE_MANAGER_H

#include <vector>
#include "acl/acl_base.h"

namespace acl {
class ResourceManager {
public:
    explicit ResourceManager(const aclrtContext context);
    ~ResourceManager();

    ResourceManager(const ResourceManager &) = delete;
    ResourceManager &operator=(const ResourceManager &) = delete;
    ResourceManager &operator=(ResourceManager &&) = delete;
    ResourceManager(ResourceManager &&) = delete;

    aclError GetMemory(void **const address, const size_t size);

private:
    std::vector<void *> pending_mem_;
    void *current_mem_ = nullptr;
    size_t current_mem_size_ = 0UL;

    // used by release proc
    aclrtContext context_;
};
} // namespace acl

#endif // TRUNK_PROJ_RESOURCE_MANAGER_H

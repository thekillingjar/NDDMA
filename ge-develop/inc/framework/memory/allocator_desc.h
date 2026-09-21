/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_FRAMEWORK_MEMORY_ALLOCATOR_DESC_H_
#define INC_FRAMEWORK_MEMORY_ALLOCATOR_DESC_H_
#include <cstdlib>

namespace ge {
using AllocFunc = void *(*)(void *obj, size_t size);
using FreeFunc = void (*)(void *obj, void *block);
using AllocAdviseFunc = void *(*)(void *obj, size_t size, void *addr);
using GetAddrFromBlockFunc = void *(*)(void *block);

using AllocatorDesc = struct AllocatorDesc {
    AllocFunc alloc_func;
    FreeFunc free_func;
    AllocAdviseFunc alloc_advise_func;
    GetAddrFromBlockFunc get_addr_from_block_func;

    void *obj;
};
}  // namespace ge

#endif  // INC_FRAMEWORK_MEMORY_ALLOCATOR_DESC_H_

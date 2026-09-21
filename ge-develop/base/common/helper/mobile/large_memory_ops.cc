/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "large_memory_ops.h"

#include <iostream>
#include <functional>

#include "securec.h"
#include "securectype.h"

namespace {

using BlockOpsFuncType = std::function<bool(const size_t, const size_t)>;

bool LoopBlockOps(const size_t total_size, const BlockOpsFuncType& block_ops_func)
{
    size_t block_count_max = total_size / SECUREC_MEM_MAX_LEN;
    const size_t rest_size = total_size % SECUREC_MEM_MAX_LEN;
    if (rest_size > static_cast<size_t>(0)) {
        block_count_max++;
    }
    size_t src_offset = 0;
    for (size_t block_count = 0; block_count < block_count_max; block_count++) {
        size_t block_size = SECUREC_MEM_MAX_LEN;
        if ((block_count == block_count_max - static_cast<size_t>(1)) && (rest_size > static_cast<size_t>(0))) {
            block_size = rest_size;
        }
        GELOGI("[Mobile] src offset: %d block_size: %d", src_offset, block_size);
        if (!block_ops_func(src_offset, block_size)) {
            GELOGE(ge::FAILED, "[Mobile] block ops func failed, src offset: %d, block size: %d.", src_offset, block_size);
            return false;
        }
        src_offset += block_size;
    }
    return true;
}

} // namespace

namespace ge {

bool LargeMemoryOps::Copy(void* dst_buffer, const size_t dst_size,
    const void* src_buffer, const size_t src_size)
{
    if (dst_size < src_size) {
        GELOGE(ge::FAILED, "[Mobile] dst_size < src_size failed, dst size: %d, src size: %d.", dst_size, src_size);
        return false;
    }
    auto memcpy_func =
        [&dst_buffer, &dst_size, &src_buffer](const size_t src_offset, const size_t block_size) {
            void* dst_buffer_start_addr = static_cast<uint8_t*>(dst_buffer) + src_offset;
            const size_t dst_rest_size = dst_size - src_offset;
            if (dst_rest_size < block_size) {
                GELOGE(ge::FAILED, "[Mobile] dst_rest_size < block_size failed, dst rest size: %d, block size: %d.",
                    dst_rest_size, block_size);
                return false;
            }
            const void* src_buffer_start_addr = static_cast<const uint8_t*>(src_buffer) + src_offset;
            const auto ret = memcpy_s(dst_buffer_start_addr, block_size, src_buffer_start_addr, block_size);
            if (ret != EOK) {
                GELOGE(ge::FAILED, "[Mobile] memcpy_s failed!, ret %d", ret);
                return false;
            }
            return true;
        };
    return LoopBlockOps(src_size, memcpy_func);
}

} // namespace ge


/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BASE_COMMON_HELPER_MOBILE_LARGE_MEMORY_OPS_H
#define BASE_COMMON_HELPER_MOBILE_LARGE_MEMORY_OPS_H

#include <cstdint>
#include <cstddef>
#include "ge_common/ge_api_error_codes.h"
#include "framework/common/debug/ge_log.h"

namespace ge {

class LargeMemoryOps {
public:
    static bool Copy(void* dst_buffer, const size_t dst_size,
        const void* src_buffer, const size_t src_size);
};

} // namespace ge

#endif // BASE_COMMON_HELPER_MOBILE_LARGE_MEMORY_OPS_H

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACL_UTILS_ARRAY_UTILS_H
#define ACL_UTILS_ARRAY_UTILS_H

#include <utility>
#include <vector>
#include <map>
#include "types/tensor_desc_internal.h"
#include "types/data_buffer_internal.h"
#include "common/log_inner.h"

namespace acl {
namespace array_utils {
using DynamicInputIndexPair = std::pair<std::vector<int32_t>, std::vector<int32_t>>;
template<typename T>
aclError CheckPtrArray(const int32_t size, const T *const *const arr)
{
    if (size == 0) {
        return ACL_SUCCESS;
    }

    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(arr);
    for (int32_t i = 0; i < size; ++i) {
        ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(arr[i]);
    }

    return ACL_SUCCESS;
}

template<typename T>
aclError CheckPtrArray(const int32_t size, T *const *const arr)
{
    if (size == 0) {
        return ACL_SUCCESS;
    }

    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(arr);
    for (int32_t idx = 0; idx < size; ++idx) {
        ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(arr[idx]);
    }

    return ACL_SUCCESS;
}

ACL_FUNC_VISIBILITY aclError CheckDataBufferArry(const int32_t size, const aclDataBuffer *const *const arr);

ACL_FUNC_VISIBILITY bool IsAllTensorEmpty(const int32_t size, const aclTensorDesc *const *const arr);

bool IsAllTensorEmpty(const int32_t size, const aclDataBuffer *const *const arr);

ACL_FUNC_VISIBILITY aclError IsHostMemTensorDesc(const int32_t size, const aclTensorDesc *const *const arr);

bool GetDynamicInputIndex(const int32_t size, const aclTensorDesc *const *const arr, DynamicInputIndexPair &indexPair);
} // namespace array_utils
} // acl
#endif // ACL_UTILS_ARRAY_UTILS_H

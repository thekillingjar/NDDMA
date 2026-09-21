/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACL_OP_GE_TENSOR_CACHE_H
#define ACL_OP_GE_TENSOR_CACHE_H

#include <vector>
#include <mutex>
#include "graph/ge_tensor.h"
#include "graph/utils/attr_utils.h"
#include "graph/small_vector.h"
#include "graph/ascend_limits.h"

namespace acl {

inline void WrapGeShape(ge::GeShape &geShape,
    const ge::SmallVector<int64_t, static_cast<size_t>(ge::kDefaultMaxInputNum)> &dims)
{
    geShape.SetDimNum(dims.size());
    for (size_t i = 0U; i < dims.size(); ++i) {
        (void)geShape.SetDim(i, dims[i]);
    }
}

using GeTensorDescVecPtr = std::vector<ge::GeTensorDesc>*;

class GeTensorDescCache {
public:
    GeTensorDescCache() = default;
    ~GeTensorDescCache();
    // must use GetInstance when we need use this calss, otherwise memory error may happen
    static GeTensorDescCache& GetInstance();
    GeTensorDescVecPtr GetDescVecPtr(const size_t size);
    void ReleaseDescVecPtr(const GeTensorDescVecPtr ptr);
private:
    std::vector<GeTensorDescVecPtr> descCache_;
    std::mutex cacheMutex_;
};
} // namespace acl

#endif // ACL_OP_GE_TENSOR_CACHE_H

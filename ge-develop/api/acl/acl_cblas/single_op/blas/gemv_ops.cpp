/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "acl/ops/acl_cblas.h"
#include "common/log_inner.h"
#include "common/prof_api_reg.h"

aclError aclblasGemvEx(aclTransType transA, int m, int n, const void *alpha,
    const void *a, int lda, aclDataType dataTypeA,
    const void *x, int incx, aclDataType dataTypeX,
    const void *beta, void *y, int incy, aclDataType dataTypeY,
    aclComputeType type, aclrtStream stream)
{
    ACL_PROFILING_REG(acl::AclProfType::AclblasGemvEx);
    ACL_LOG_INFO("start to execute aclblasGemvEx");
    return aclblasGemmEx(transA, ACL_TRANS_N, ACL_TRANS_N,
        m, 1, n, alpha, a, lda, dataTypeA, x, incx,
        dataTypeX, beta, y, incy, dataTypeY, type, stream);
}

aclError aclblasCreateHandleForGemvEx(aclTransType transA, int m, int n,
                                      aclDataType dataTypeA,
                                      aclDataType dataTypeX,
                                      aclDataType dataTypeY,
                                      aclComputeType type,
                                      aclopHandle **handle)
{
    ACL_PROFILING_REG(acl::AclProfType::AclblasCreateHandleForGemvEx);
    ACL_LOG_INFO("start to execute aclblasCreateHandleForGemvEx");
    return aclblasCreateHandleForGemmEx(transA, ACL_TRANS_N, ACL_TRANS_N, m, 1, n,
        dataTypeA, dataTypeX, dataTypeY, type, handle);
}

aclError aclblasHgemv(aclTransType transA,
    int m, int n, const aclFloat16 *alpha, const aclFloat16 *a,
    int lda, const aclFloat16 *x, int incx, const aclFloat16 *beta,
    aclFloat16 *y, int incy, aclComputeType type, aclrtStream stream)
{
    ACL_PROFILING_REG(acl::AclProfType::AclblasHgemv);
    ACL_LOG_INFO("start to execute aclblasHgemv");
    return aclblasGemvEx(transA, m, n, alpha, a, lda, ACL_FLOAT16, x, incx,
        ACL_FLOAT16, beta, y, incy, ACL_FLOAT16, type, stream);
}

aclError aclblasCreateHandleForHgemv(aclTransType transA,
                                     int m,
                                     int n,
                                     aclComputeType type,
                                     aclopHandle **handle)
{
    ACL_PROFILING_REG(acl::AclProfType::AclblasCreateHandleForHgemv);
    ACL_LOG_INFO("start to execute aclblasCreateHandleForHgemv");
    return aclblasCreateHandleForGemvEx(transA, m, n, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, type, handle);
}

aclError aclblasCreateHandleForS8gemv(aclTransType transA,
                                      int m,
                                      int n,
                                      aclComputeType type,
                                      aclopHandle **handle)
{
    ACL_PROFILING_REG(acl::AclProfType::AclblasCreateHandleForS8gemv);
    ACL_LOG_INFO("start to execute aclblasCreateHandleForS8gemv");
    return aclblasCreateHandleForGemvEx(transA, m, n, ACL_INT8, ACL_INT8, ACL_INT32, type, handle);
}

aclError aclblasS8gemv(aclTransType transA,
    int m, int n, const int32_t *alpha, const int8_t *a,
    int lda, const int8_t *x, int incx, const int32_t *beta,
    int32_t *y, int incy, aclComputeType type, aclrtStream stream)
{
    ACL_PROFILING_REG(acl::AclProfType::AclblasS8gemv);
    ACL_LOG_INFO("start to execute aclblasS8gemv");
    return aclblasGemvEx(transA, m, n, alpha, a, lda, ACL_INT8, x, incx,
        ACL_INT8, beta, y, incy, ACL_INT32, type, stream);
}

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
#include "acl/acl_op.h"
#include "common/log_inner.h"
#include "types/op_attr.h"
#include "common/prof_api_reg.h"

namespace {
constexpr int32_t GEMM_NUM_INPUTS = 5;
constexpr int32_t GEMM_NUM_OUTPUTS = 1;
constexpr char_t const *MATMUL_OP_TYPE = "GEMM";
constexpr char_t const *MATMUL_OP_TRANSPOSE_A = "transpose_a";
constexpr char_t const *MATMUL_OP_TRANSPOSE_B = "transpose_b";
constexpr size_t SIZE_ALPHA_BETA = 4U;
constexpr int64_t BLOCK_REDUCE = 16;
constexpr int64_t BLOCK_REDUCE_INT8 = 32;
constexpr int64_t BLOCK = 16;

int64_t Ceil(const int64_t n1, const int64_t n2)
{
    if (n1 == 0) {
        return 0;
    }
    return (n2 != 0) ? (((n1 - 1) / n2) + 1) : 0;
}

class MatMulTemplate {
public:
    MatMulTemplate(const int64_t m,
        const int64_t n,
        const int64_t k,
        const aclDataType dataTypeA,
        const aclDataType dataTypeB,
        const aclDataType dataTypeC,
        const aclTransType transA,
        const aclTransType transB,
        const aclTransType transC)
        : matrixDescA_(dataTypeA, {m, k}, ACL_FORMAT_ND),
          matrixDescB_(dataTypeB, {k, n}, ACL_FORMAT_ND),
          matrixDescC_(dataTypeC, {m, n}, ACL_FORMAT_ND),
          alphaBetaDesc_(dataTypeC, 0U, nullptr, ACL_FORMAT_ND),
          transposeA_(transA),
          transposeB_(transB)
        {
            switch (transA) {
                case ACL_TRANS_T:
                    matrixDescA_.dims = {k, m};
                    break;
                case ACL_TRANS_NZ:
                    matrixDescA_.format = ACL_FORMAT_FRACTAL_NZ;
                    if (dataTypeA == ACL_INT8) {
                        matrixDescA_.dims = {Ceil(k, BLOCK_REDUCE_INT8), Ceil(m, BLOCK), BLOCK, BLOCK_REDUCE_INT8};
                    } else {
                        matrixDescA_.dims = {Ceil(k, BLOCK_REDUCE), Ceil(m, BLOCK), BLOCK, BLOCK_REDUCE};
                    }
                    break;
                default:
                    break;
            }

            switch (transB) {
                case ACL_TRANS_T:
                    matrixDescB_.dims = {n, k};
                    break;
                case ACL_TRANS_NZ:
                    matrixDescB_.format = ACL_FORMAT_FRACTAL_NZ;
                    if (dataTypeB == ACL_INT8) {
                        matrixDescB_.format = ACL_FORMAT_FRACTAL_Z;
                        matrixDescB_.dims = {Ceil(k, BLOCK_REDUCE_INT8), Ceil(n, BLOCK), BLOCK, BLOCK_REDUCE_INT8};
                    } else {
                        matrixDescB_.dims = {Ceil(n, BLOCK), Ceil(k, BLOCK_REDUCE), BLOCK_REDUCE, BLOCK};
                    }

                    break;
                default:
                    break;
            }

            if (transC == ACL_TRANS_NZ) {
                matrixDescC_.format = ACL_FORMAT_FRACTAL_NZ;
                matrixDescC_.dims = {Ceil(n, BLOCK), Ceil(m, BLOCK), BLOCK, BLOCK};
            }
            ACL_LOG_INFO("tensorDesc of matrixA is %s, tensorDesc of matrixB is %s, tensorDesc of matrixC is %s",
                         matrixDescA_.DebugString().c_str(),
                         matrixDescB_.DebugString().c_str(),
                         matrixDescC_.DebugString().c_str());
        }

    ~MatMulTemplate() = default;

    aclError ExecuteAsync(const void *const alpha,
                          const void *const matrixA,
                          const void *const matrixB,
                          const void *const beta,
                          void *const matrixC,
                          const aclrtStream stream)
    {
        // create data buffer
        const size_t lenA = aclGetTensorDescSize(&matrixDescA_);
        const size_t lenB = aclGetTensorDescSize(&matrixDescB_);
        const size_t lenC = aclGetTensorDescSize(&matrixDescC_);
        aclDataBuffer dataA{const_cast<void *>(matrixA), lenA};
        aclDataBuffer dataB{const_cast<void *>(matrixB), lenB};
        aclDataBuffer dataC{matrixC, lenC};
        aclDataBuffer dataAlpha{const_cast<void *>(alpha), SIZE_ALPHA_BETA};
        aclDataBuffer dataBeta{const_cast<void *>(beta), SIZE_ALPHA_BETA};

        aclDataBuffer *devInputs[] = {&dataA, &dataB, &dataC, &dataAlpha, &dataBeta};
        aclDataBuffer *devOutputs[] = {&dataC};

        aclopAttr attr;
        const uint8_t a = ((transposeA_ == ACL_TRANS_T) ? 1U : 0U);
        const uint8_t b = ((transposeB_ == ACL_TRANS_T) ? 1U : 0U);
        (void)aclopSetAttrBool(&attr, MATMUL_OP_TRANSPOSE_A, a);
        (void)aclopSetAttrBool(&attr, MATMUL_OP_TRANSPOSE_B, b);
        return aclopExecuteV2(MATMUL_OP_TYPE,
                              GEMM_NUM_INPUTS, inputDesc_, devInputs,
                              GEMM_NUM_OUTPUTS, outputDesc_, devOutputs,
                              &attr, stream);
    }

    aclError CreateHandle(aclopHandle **const handle) const
    {
        aclopAttr attr;
        const uint8_t a = ((transposeA_ == ACL_TRANS_T) ? 1U : 0U);
        const uint8_t b = ((transposeB_ == ACL_TRANS_T) ? 1U : 0U);
        (void)aclopSetAttrBool(&attr, MATMUL_OP_TRANSPOSE_A, a);
        (void)aclopSetAttrBool(&attr, MATMUL_OP_TRANSPOSE_B, b);
        return aclopCreateHandle(MATMUL_OP_TYPE,
                                 GEMM_NUM_INPUTS, inputDesc_,
                                 GEMM_NUM_OUTPUTS, outputDesc_,
                                 &attr, handle);
    }

private:
    aclTensorDesc matrixDescA_;
    aclTensorDesc matrixDescB_;
    aclTensorDesc matrixDescC_;
    aclTensorDesc alphaBetaDesc_;
    aclTensorDesc *inputDesc_[GEMM_NUM_INPUTS] = {&matrixDescA_, &matrixDescB_,
        &matrixDescC_, &alphaBetaDesc_, &alphaBetaDesc_};
    aclTensorDesc *outputDesc_[GEMM_NUM_OUTPUTS] = {&matrixDescC_};
    aclTransType transposeA_;
    aclTransType transposeB_;
};
} // namespace

aclError aclblasGemmEx(aclTransType transA,
    aclTransType transB, aclTransType transC, int m, int n,
    int k, const void *alpha, const void *matrixA, int lda,
    aclDataType dataTypeA, const void *matrixB, int ldb,
    aclDataType dataTypeB, const void *beta, void *matrixC,
    int ldc, aclDataType dataTypeC, aclComputeType type, aclrtStream stream)
{
    ACL_PROFILING_REG(acl::AclProfType::AclblasGemmEx);
    ACL_LOG_INFO("start to execute aclblasGemmEx");
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(alpha);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(beta);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(matrixA);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(matrixB);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(matrixC);
    if ((m <= 0) || (n <= 0) || (k <= 0)) {
        ACL_LOG_ERROR("[Check][Params]mnk must > 0. m = %d, n = %d, k = %d", m, n, k);
        const std::string values = acl::AclErrorLogManager::FormatStr("m = %d, n = %d, k = %d", m, n, k);
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
            std::vector<const char *>({"param", "value", "reason"}),
            std::vector<const char *>({"m, n, k", values.c_str(), "mnk must > 0"}));
        return ACL_ERROR_INVALID_PARAM;
    }

    if ((transA < ACL_TRANS_N) ||
        (transA >= ACL_TRANS_NZ_T) ||
        (transB < ACL_TRANS_N) ||
        (transB >= ACL_TRANS_NZ_T)) {
        ACL_LOG_ERROR("[Check][Params]transA or transB is error. "
            "only support ACL_TRANS_N, ACL_TRANS_T and ACL_TRANS_NZ");
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
            std::vector<const char *>({"param", "value", "reason"}),
            std::vector<const char *>({"transA or transB", "",
            "only support ACL_TRANS_N, ACL_TRANS_T and ACL_TRANS_NZ"}));
        return ACL_ERROR_INVALID_PARAM;
    }

    if ((transC != ACL_TRANS_N) && (transC != ACL_TRANS_NZ)) {
        ACL_LOG_ERROR("[Check][Trans]transC is error, only support ACL_TRANS_N and ACL_TRANS_NZ");
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
            std::vector<const char *>({"param", "value", "reason"}),
            std::vector<const char *>({"transC", "", "only support ACL_TRANS_N and ACL_TRANS_NZ"}));
        return ACL_ERROR_INVALID_PARAM;
    }

    if ((lda != -1) || (ldb != -1) || (ldc != -1)) {
        ACL_LOG_ERROR("[Check][Params]lda, ldb(incx) or ldc(incy) is error, only support -1.");
        const std::string values = acl::AclErrorLogManager::FormatStr("lda = %d, ldb = %d, ldc = %d", lda, ldb, ldc);
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
            std::vector<const char *>({"param", "value", "reason"}),
            std::vector<const char *>({"lda, ldb, ldc", values.c_str(), "only support -1"}));
        return ACL_ERROR_INVALID_PARAM;
    }

    if (type != ACL_COMPUTE_HIGH_PRECISION) {
        ACL_LOG_ERROR("[Check][Type]precision type is error, only support ACL_COMPUTE_HIGH_PRECISION");
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
            std::vector<const char *>({"param", "value", "reason"}),
            std::vector<const char *>({"type", "", "only support ACL_COMPUTE_HIGH_PRECISION"}));
        return ACL_ERROR_INVALID_PARAM;
    }

    MatMulTemplate matMul(static_cast<int64_t>(m), static_cast<int64_t>(n), static_cast<int64_t>(k),
        dataTypeA, dataTypeB, dataTypeC, transA, transB, transC);
    return matMul.ExecuteAsync(alpha, matrixA, matrixB, beta, matrixC, stream);
}

aclError aclblasCreateHandleForGemmEx(aclTransType transA,
    aclTransType transB, aclTransType transC, int m, int n, int k,
    aclDataType dataTypeA, aclDataType dataTypeB, aclDataType dataTypeC,
    aclComputeType type, aclopHandle **handle)
{
    ACL_PROFILING_REG(acl::AclProfType::AclblasCreateHandleForGemmEx);
    ACL_LOG_INFO("start to execute aclblasCreateHandleForGemmEx");
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(handle);
    if ((m <= 0) || (n <= 0) || (k <= 0)) {
        ACL_LOG_ERROR("[Check][Params]The value of m,n,k must be larger than zero.m = %d, n = %d, k = %d",
            m, n, k);
        const std::string errMsg = acl::AclErrorLogManager::FormatStr("m = %d, n = %d, k = %d", m, n, k);
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
            std::vector<const char *>({"param", "value", "reason"}),
            std::vector<const char *>({"m, n, k", errMsg.c_str(), "mnk must > 0"}));
        return ACL_ERROR_INVALID_PARAM;
    }

    if ((transA < ACL_TRANS_N) ||
        (transA >= ACL_TRANS_NZ_T) ||
        (transB < ACL_TRANS_N) ||
        (transB >= ACL_TRANS_NZ_T)) {
        ACL_LOG_ERROR("[Check][Params]transA, transB is error. "
            "only support ACL_TRANS_N, ACL_TRANS_T and ACL_TRANS_NZ");
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
            std::vector<const char *>({"param", "value", "reason"}),
            std::vector<const char *>({"transA or transB", "",
            "only support ACL_TRANS_N, ACL_TRANS_T and ACL_TRANS_NZ"}));
        return ACL_ERROR_INVALID_PARAM;
    }

    if ((transC != ACL_TRANS_N) && (transC != ACL_TRANS_NZ)) {
        ACL_LOG_ERROR("[Check][transC]transC is error, only support ACL_TRANS_N and ACL_TRANS_NZ");
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
            std::vector<const char *>({"param", "value", "reason"}),
            std::vector<const char *>({"transC", "", "only support ACL_TRANS_N and ACL_TRANS_NZ"}));
        return ACL_ERROR_INVALID_PARAM;
    }

    if (type != ACL_COMPUTE_HIGH_PRECISION) {
        ACL_LOG_ERROR("[Check][Type]precision type is error, only support ACL_COMPUTE_HIGH_PRECISION");
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
            std::vector<const char *>({"param", "value", "reason"}),
            std::vector<const char *>({"type", "", "only support ACL_COMPUTE_HIGH_PRECISION"}));
        return ACL_ERROR_INVALID_PARAM;
    }

    const MatMulTemplate matMul(static_cast<int64_t>(m), static_cast<int64_t>(n), static_cast<int64_t>(k),
        dataTypeA, dataTypeB, dataTypeC, transA, transB, transC);
    return matMul.CreateHandle(handle);
}

aclError aclblasCreateHandleForHgemm(aclTransType transA,
                                     aclTransType transB,
                                     aclTransType transC,
                                     int m,
                                     int n,
                                     int k,
                                     aclComputeType type,
                                     aclopHandle **handle)
{
    ACL_PROFILING_REG(acl::AclProfType::AclblasCreateHandleForHgemm);
    ACL_LOG_INFO("start to execute aclblasCreateHandleForHgemm");

    return aclblasCreateHandleForGemmEx(transA, transB, transC,
                                        m, n, k,
                                        ACL_FLOAT16,
                                        ACL_FLOAT16,
                                        ACL_FLOAT16, type, handle);
}

aclError aclblasHgemm(aclTransType transA,
    aclTransType transB, aclTransType transC, int m, int n, int k,
    const aclFloat16 *alpha, const aclFloat16 *matrixA, int lda,
    const aclFloat16 *matrixB, int ldb, const aclFloat16 *beta,
    aclFloat16 *matrixC, int ldc, aclComputeType type, aclrtStream stream)
{
    ACL_PROFILING_REG(acl::AclProfType::AclblasHgemm);
    ACL_LOG_INFO("start to execute aclblasHgemm");
    return aclblasGemmEx(transA, transB, transC,
                         m, n, k,
                         alpha, matrixA, lda, ACL_FLOAT16,
                         matrixB, ldb, ACL_FLOAT16,
                         beta, matrixC, ldc, ACL_FLOAT16,
                         type, stream);
}

aclError aclblasS8gemm(aclTransType transA,
    aclTransType transB, aclTransType transC, int m, int n, int k,
    const int32_t *alpha, const int8_t *matrixA, int lda,
    const int8_t *matrixB, int ldb, const int32_t *beta, int32_t *matrixC,
    int ldc, aclComputeType type, aclrtStream stream)
{
    ACL_PROFILING_REG(acl::AclProfType::AclblasS8gemm);
    ACL_LOG_INFO("start to execute aclblasS8gemm");
    return aclblasGemmEx(transA, transB, transC,
                         m, n, k,
                         alpha, matrixA, lda, ACL_INT8,
                         matrixB, ldb, ACL_INT8,
                         beta, matrixC, ldc, ACL_INT32,
                         type, stream);
}

aclError aclblasCreateHandleForS8gemm(aclTransType transA,
                                      aclTransType transB,
                                      aclTransType transC,
                                      int m,
                                      int n,
                                      int k,
                                      aclComputeType type,
                                      aclopHandle **handle)
{
    ACL_PROFILING_REG(acl::AclProfType::AclblasCreateHandleForS8gemm);
    ACL_LOG_INFO("start to execute aclblasCreateHandleForS8gemm");
    return aclblasCreateHandleForGemmEx(transA, transB, transC, m, n, k, ACL_INT8, ACL_INT8, ACL_INT32, type, handle);
}

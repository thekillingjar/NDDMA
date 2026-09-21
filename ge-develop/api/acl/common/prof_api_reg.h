/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACL_PROF_API_REG_H_
#define ACL_PROF_API_REG_H_

#include "acl/acl_base.h"
#include "aprof_pub.h"

#define OP_EXEC_PROF_TYPE_START_OFFSET 0x008000U
#define OP_COMPILE_PROF_TYPE_START_OFFSET 0x000000U
#define CBLAS_PROF_TYPE_START_OFFSET 0x006000U

namespace acl {
    enum class AclProfType {
        // start with 0x010000U
        OpCompileProfTypeStart = MSPROF_REPORT_ACL_OP_BASE_TYPE + OP_COMPILE_PROF_TYPE_START_OFFSET,
        AclopCompile,
        AclopCompileAndExecute,
        AclopCompileAndExecuteV2,
        AclGenGraphAndDumpForOp,
        OpCompile,
        OpCompileAndDump,
        // start with 0x018000U
        OpExecProfTypeStart = MSPROF_REPORT_ACL_OP_BASE_TYPE + OP_EXEC_PROF_TYPE_START_OFFSET,
        AclopLoad,
        AclopExecute,
        AclopCreateHandle,
        AclopDestroyHandle,
        AclopExecWithHandle,
        AclopExecuteV2,
        AclopCreateKernel,
        AclopUpdateParams,
        AclopInferShape,
        AclopCast,
        AclopCreateHandleForCast,
        AclopCreateAttr,
        AclopDestroyAttr,
        // start with 0x020000U
        ModelProfTypeStart = MSPROF_REPORT_ACL_MODEL_BASE_TYPE,
        AclmdlExecute,
        AclmdlLoadFromMemWithQ,
        AclmdlLoadFromMemWithMem,
        AclmdlGetDesc,
        AclmdlLoadFromFile,
        AclmdlLoadFromFileWithMem,
        AclmdlLoadFromMem,
        AclmdlBundleLoadFromFile,
        AclmdlBundleLoadFromMem,
        AclmdlBundleLoadModelWithMem,
        AclmdlBundleLoadModelWithConfig,
        AclmdlBundleUnload,
        AclmdlBundleUnloadModel,
        AclmdlSetInputAIPP,
        AclmdlSetAIPPByInputIndex,
        AclmdlExecuteAsync,
        AclmdlQuerySize,
        AclmdlQuerySizeFromMem,
        AclmdlSetDynamicBatchSize,
        AclmdlSetDynamicHWSize,
        AclmdlSetInputDynamicDims,
        AclmdlLoadWithConfig,
        AclmdlLoadFromFileWithQ,
        AclmdlUnload,
        AclCreateTensorDesc,
        AclDestroyTensorDesc,
        AclTransTensorDescFormat,
        // start with 0x046000U
        CblasProfTypeStart = MSPROF_REPORT_ACL_OTHERS_BASE_TYPE + CBLAS_PROF_TYPE_START_OFFSET,
        AclblasGemmEx,
        AclblasCreateHandleForGemmEx,
        AclblasCreateHandleForHgemm,
        AclblasHgemm,
        AclblasS8gemm,
        AclblasCreateHandleForS8gemm,
        AclblasGemvEx,
        AclblasCreateHandleForGemvEx,
        AclblasHgemv,
        AclblasCreateHandleForHgemv,
        AclblasCreateHandleForS8gemv,
        AclblasS8gemv,
        // this is the end, can not add after ProfTypeEnd
        ProfTypeEnd
    };

    class AclProfilingReporter {
    public:
        explicit AclProfilingReporter(const AclProfType apiId);
        virtual ~AclProfilingReporter() noexcept;
    private:
        uint64_t startTime_ = 0UL;
        const AclProfType aclApi_;
    };
}  // namespace acl

#define ACL_PROFILING_REG(apiId) \
    const acl::AclProfilingReporter profilingReporter(apiId)
#endif

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_task.h"
#include "error_codes_inner.h"

namespace acl {

namespace {
    using rtSmDesc_t = void *;
    using rtStream_t = void *;

    using rtError_t = int32_t;
    extern "C" rtError_t rtKernelLaunch(const void *stubFunc, uint32_t blockDim, void *args, uint32_t argsSize,
                                 rtSmDesc_t *smDesc, rtStream_t stm);
}

TbeOpTask::TbeOpTask(const void *const stubFunction, const uint32_t block) : OpTask(),
    stubFunc_(stubFunction), blockDim_(block)
{
}

aclError TbeOpTask::ExecuteAsync(const int32_t numInputs,
                                 const aclDataBuffer *const inputs[],
                                 const int32_t numOutputs,
                                 aclDataBuffer *const outputs[],
                                 const aclrtStream stream)
{
    // update args
    auto *const ptrArgs = reinterpret_cast<uintptr_t *>(args_.get());
    int32_t argIndex = 0;
    for (int32_t i = 0; i < numInputs; ++i) {
        ptrArgs[argIndex] = reinterpret_cast<uintptr_t>(inputs[i]->data);
        argIndex++;
    }
    for (int32_t i = 0; i < numOutputs; ++i) {
        ptrArgs[argIndex] = reinterpret_cast<uintptr_t>(outputs[i]->data);
        argIndex++;
    }

    // launch kernel
    ACL_LOG_DEBUG("To launch kernel, stubFunc = %s, block dim = %u, arg size = %u",
                  reinterpret_cast<const char_t *>(stubFunc_), blockDim_, argSize_);
    const auto ret = rtKernelLaunch(stubFunc_, blockDim_, const_cast<uint8_t *>(args_.get()),
        argSize_, nullptr, stream);
    return ACL_GET_ERRCODE_RTS(ret);
}

void OpTask::SetArgs(std::unique_ptr<uint8_t[]> &&args, const uint32_t argSize)
{
    args_ = std::move(args);
    argSize_ = argSize;
}
} // namespace acl
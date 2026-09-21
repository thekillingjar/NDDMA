/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rt_error_codes.h"

#include <stdlib.h>
#include <string.h>
#include "securec.h"
#include "acl_stub.h"

rtError_t aclStub::rtFunctionRegister(void *binHandle,
                            const void *stubFunc,
                            const char *stubName,
                            const void *devFunc,
                            uint32_t funcMode)
{
    return RT_ERROR_NONE;
}

rtError_t aclStub::rtKernelLaunch(const void *stubFunc,
                        uint32_t blockDim,
                        void *args,
                        uint32_t argsSize,
                        rtSmDesc_t *smDesc,
                        rtStream_t stream)
{
    return RT_ERROR_NONE;
}

rtError_t aclStub::rtDevBinaryRegister(const rtDevBinary_t *bin, void **handle)
{
    return RT_ERROR_NONE;
}

rtError_t aclStub::rtDevBinaryUnRegister(void *handle)
{
    return RT_ERROR_NONE;
}

rtError_t aclStub::rtGetSocSpec(const char *label, const char *key, char *value, const uint32_t maxLen)
{
    if ((strcmp(label, "version") == 0) && (strcmp(key, "NpuArch") == 0)) {
        (void)strcpy_s(value, maxLen, "2201");
    }
    return RT_ERROR_NONE;
}

rtError_t rtFunctionRegister(void *binHandle,
                            const void *stubFunc,
                            const char *stubName,
                            const void *devFunc,
                            uint32_t funcMode)
{
    return MockFunctionTest::aclStubInstance().rtFunctionRegister(binHandle, stubFunc, stubName, devFunc, funcMode);
}

rtError_t rtDevBinaryUnRegister(void *handle)
{
    return MockFunctionTest::aclStubInstance().rtDevBinaryUnRegister(handle);
}

rtError_t rtKernelLaunch(const void *stubFunc,
                        uint32_t blockDim,
                        void *args,
                        uint32_t argsSize,
                        rtSmDesc_t *smDesc,
                        rtStream_t stream)
{
    return MockFunctionTest::aclStubInstance().rtKernelLaunch(stubFunc, blockDim, args, argsSize, smDesc, stream);
}

rtError_t rtDevBinaryRegister(const rtDevBinary_t *bin, void **handle)
{
    return MockFunctionTest::aclStubInstance().rtDevBinaryRegister(bin, handle);
}

rtError_t rtGetSocSpec(const char *label, const char *key, char *value, const uint32_t maxLen)
{
    if ((strcmp(label, "version") == 0) && (strcmp(key, "NpuArch") == 0)) {
        (void)strcpy_s(value, maxLen, "2201");
    }
    return MockFunctionTest::aclStubInstance().rtGetSocSpec(label, key, value, maxLen);
}

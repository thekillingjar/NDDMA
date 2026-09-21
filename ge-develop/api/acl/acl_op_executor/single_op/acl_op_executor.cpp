/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "acl_op_executor_impl.h"

#ifdef __cplusplus
extern "C" {
#endif

aclError aclopSetModelDir(const char *modelDir)
{
    return aclopSetModelDirImpl(modelDir);
}

aclError aclopLoad(const void *model, size_t modelSize)
{
    return aclopLoadImpl(model, modelSize);
}

aclopAttr *aclopCreateAttr()
{
    return aclopCreateAttrImpl();
}

void aclopDestroyAttr(const aclopAttr *attr)
{
    return aclopDestroyAttrImpl(attr);
}

aclError aclopSetAttrBool(aclopAttr *attr, const char *attrName, uint8_t attrValue)
{
    return aclopSetAttrBoolImpl(attr, attrName, attrValue);
}

aclError aclopSetAttrInt(aclopAttr *attr, const char *attrName, int64_t attrValue)
{
    return aclopSetAttrIntImpl(attr, attrName, attrValue);
}

aclError aclopSetAttrFloat(aclopAttr *attr, const char *attrName, float attrValue)
{
    return aclopSetAttrFloatImpl(attr, attrName, attrValue);
}

aclError aclopSetAttrString(aclopAttr *attr, const char *attrName, const char *attrValue)
{
    return aclopSetAttrStringImpl(attr, attrName, attrValue);
}

aclError aclopSetAttrDataType(aclopAttr *attr, const char *attrName, aclDataType attrValue)
{
    return aclopSetAttrDataTypeImpl(attr, attrName, attrValue);
}

aclError aclopSetAttrListDataType(aclopAttr *attr, const char *attrName, int numValues,
    const aclDataType values[])
{
    return aclopSetAttrListDataTypeImpl(attr, attrName, numValues, values);
}

aclError aclopSetAttrListBool(aclopAttr *attr, const char *attrName, int numValues,
    const uint8_t *values)
{
    return aclopSetAttrListBoolImpl(attr, attrName, numValues, values);
}

aclError aclopSetAttrListInt(aclopAttr *attr, const char *attrName, int numValues,
    const int64_t *values)
{
    return aclopSetAttrListIntImpl(attr, attrName, numValues, values);
}

aclError aclopSetAttrListFloat(aclopAttr *attr, const char *attrName, int numValues,
    const float *values)
{
    return aclopSetAttrListFloatImpl(attr, attrName, numValues, values);
}

aclError aclopSetAttrListString(aclopAttr *attr, const char *attrName, int numValues,
    const char **values)
{
    return aclopSetAttrListStringImpl(attr, attrName, numValues, values);
}

aclError aclopSetAttrListListInt(aclopAttr *attr,
                                                     const char *attrName,
                                                     int numLists,
                                                     const int *numValues,
                                                     const int64_t *const values[])
{
    return aclopSetAttrListListIntImpl(attr, attrName, numLists, numValues, values);
}

aclError aclopExecute(const char *opType,
                                          int numInputs,
                                          const aclTensorDesc *const inputDesc[],
                                          const aclDataBuffer *const inputs[],
                                          int numOutputs,
                                          const aclTensorDesc *const outputDesc[],
                                          aclDataBuffer *const outputs[],
                                          const aclopAttr *attr,
                                          aclrtStream stream)
{
    return aclopExecuteImpl(opType, numInputs, inputDesc, inputs, numOutputs, outputDesc, outputs, attr, stream);
}

aclError aclopExecuteV2(const char *opType,
                                            int numInputs,
                                            aclTensorDesc *inputDesc[],
                                            aclDataBuffer *inputs[],
                                            int numOutputs,
                                            aclTensorDesc *outputDesc[],
                                            aclDataBuffer *outputs[],
                                            aclopAttr *attr,
                                            aclrtStream stream)
{
    return aclopExecuteV2Impl(opType, numInputs, inputDesc, inputs, numOutputs, outputDesc, outputs, attr, stream);
}

aclError aclopCreateHandle(const char *opType,
                                               int numInputs,
                                               const aclTensorDesc *const inputDesc[],
                                               int numOutputs,
                                               const aclTensorDesc *const outputDesc[],
                                               const aclopAttr *opAttr,
                                               aclopHandle **handle)
{
    return aclopCreateHandleImpl(opType, numInputs, inputDesc, numOutputs, outputDesc, opAttr, handle);
}

void aclopDestroyHandle(aclopHandle *handle)
{
    return aclopDestroyHandleImpl(handle);
}

aclError aclopExecWithHandle(aclopHandle *handle,
                                                 int numInputs,
                                                 const aclDataBuffer *const inputs[],
                                                 int numOutputs,
                                                 aclDataBuffer *const outputs[],
                                                 aclrtStream stream)
{
    return aclopExecWithHandleImpl(handle, numInputs, inputs, numOutputs, outputs, stream);
}

aclError aclopCast(const aclTensorDesc *srcDesc,
                                       const aclDataBuffer *srcBuffer,
                                       const aclTensorDesc *dstDesc,
                                       aclDataBuffer *dstBuffer,
                                       uint8_t truncate,
                                       aclrtStream stream)
{
    return aclopCastImpl(srcDesc, srcBuffer, dstDesc, dstBuffer, truncate, stream);
}

aclError aclopCreateHandleForCast(aclTensorDesc *srcDesc,
                                                      aclTensorDesc *dstDesc,
                                                      uint8_t truncate,
                                                      aclopHandle **handle)
{
    return aclopCreateHandleForCastImpl(srcDesc, dstDesc, truncate, handle);
}

aclError aclopCreateKernel(const char *opType,
                                               const char *kernelId,
                                               const char *kernelName,
                                               void *binData,
                                               int binSize,
                                               aclopEngineType enginetype,
                                               aclDataDeallocator deallocator)
{
    return aclopCreateKernelImpl(opType, kernelId, kernelName, binData, binSize, enginetype, deallocator);
}

aclError aclopRegisterCompileFunc(const char *opType, aclopCompileFunc func)
{
    return aclopRegisterCompileFuncImpl(opType, func);
}

aclError aclopUnregisterCompileFunc(const char *opType)
{
    return aclopUnregisterCompileFuncImpl(opType);
}

aclError aclopSetKernelArgs(aclopKernelDesc *kernelDesc,
                                                const char *kernelId,
                                                uint32_t blockDim,
                                                const void *args,
                                                uint32_t argSize)
{
    return aclopSetKernelArgsImpl(kernelDesc, kernelId, blockDim, args, argSize);
}

aclError aclopSetKernelWorkspaceSizes(aclopKernelDesc *kernelDesc, int numWorkspaces,
                                                          size_t *workspaceSizes)
{
    return aclopSetKernelWorkspaceSizesImpl(kernelDesc, numWorkspaces, workspaceSizes);
}

aclError aclopUpdateParams(const char *opType,
                                               int numInputs,
                                               const aclTensorDesc *const inputDesc[],
                                               int numOutputs,
                                               const aclTensorDesc *const outputDesc[],
                                               const aclopAttr *attr)
{
    return aclopUpdateParamsImpl(opType, numInputs, inputDesc, numOutputs, outputDesc, attr);
}

aclError aclopSetMaxOpQueueNum(uint64_t maxOpNum)
{
    return aclopSetMaxOpQueueNumImpl(maxOpNum);
}

aclError aclopInferShape(const char *opType,
                                             int numInputs,
                                             aclTensorDesc *inputDesc[],
                                             aclDataBuffer *inputs[],
                                             int numOutputs,
                                             aclTensorDesc *outputDesc[],
                                             aclopAttr *attr)
{
    return aclopInferShapeImpl(opType, numInputs, inputDesc, inputs, numOutputs, outputDesc, attr);
}

#ifdef __cplusplus
}
#endif
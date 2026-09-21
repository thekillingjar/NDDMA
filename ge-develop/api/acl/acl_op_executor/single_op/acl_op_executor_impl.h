/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACL_OP_EXECUTOR_SRC_SINGLE_OP_ACL_OP_EXECUTOR_IMPL_H_
#define ACL_OP_EXECUTOR_SRC_SINGLE_OP_ACL_OP_EXECUTOR_IMPL_H_

#include "acl_base.h"
#include "acl_rt.h"
#include "acl/acl_op.h"

#ifdef __cplusplus
extern "C" {
#endif

ACL_FUNC_VISIBILITY aclError aclopSetModelDirImpl(const char *modelDir);

ACL_FUNC_VISIBILITY aclError aclopLoadImpl(const void *model, size_t modelSize);

ACL_FUNC_VISIBILITY aclopAttr *aclopCreateAttrImpl();

ACL_FUNC_VISIBILITY void aclopDestroyAttrImpl(const aclopAttr *attr);

ACL_FUNC_VISIBILITY aclError aclopSetAttrBoolImpl(aclopAttr *attr, const char *attrName, uint8_t attrValue);

ACL_FUNC_VISIBILITY aclError aclopSetAttrIntImpl(aclopAttr *attr, const char *attrName, int64_t attrValue);

ACL_FUNC_VISIBILITY aclError aclopSetAttrFloatImpl(aclopAttr *attr, const char *attrName, float attrValue);

ACL_FUNC_VISIBILITY aclError aclopSetAttrStringImpl(aclopAttr *attr, const char *attrName, const char *attrValue);

ACL_FUNC_VISIBILITY aclError aclopSetAttrDataTypeImpl(aclopAttr *attr, const char *attrName, aclDataType attrValue);

ACL_FUNC_VISIBILITY aclError aclopSetAttrListDataTypeImpl(aclopAttr *attr, const char *attrName, int numValues,
    const aclDataType values[]);

ACL_FUNC_VISIBILITY aclError aclopSetAttrListBoolImpl(aclopAttr *attr, const char *attrName, int numValues,
    const uint8_t *values);

ACL_FUNC_VISIBILITY aclError aclopSetAttrListIntImpl(aclopAttr *attr, const char *attrName, int numValues,
    const int64_t *values);

ACL_FUNC_VISIBILITY aclError aclopSetAttrListFloatImpl(aclopAttr *attr, const char *attrName, int numValues,
    const float *values);

ACL_FUNC_VISIBILITY aclError aclopSetAttrListStringImpl(aclopAttr *attr, const char *attrName, int numValues,
    const char **values);

ACL_FUNC_VISIBILITY aclError aclopSetAttrListListIntImpl(aclopAttr *attr,
                                                     const char *attrName,
                                                     int numLists,
                                                     const int *numValues,
                                                     const int64_t *const values[]);

ACL_FUNC_VISIBILITY aclError aclopExecuteImpl(const char *opType,
                                          int numInputs,
                                          const aclTensorDesc *const inputDesc[],
                                          const aclDataBuffer *const inputs[],
                                          int numOutputs,
                                          const aclTensorDesc *const outputDesc[],
                                          aclDataBuffer *const outputs[],
                                          const aclopAttr *attr,
                                          aclrtStream stream);

ACL_FUNC_VISIBILITY aclError aclopExecuteV2Impl(const char *opType,
                                            int numInputs,
                                            aclTensorDesc *inputDesc[],
                                            aclDataBuffer *inputs[],
                                            int numOutputs,
                                            aclTensorDesc *outputDesc[],
                                            aclDataBuffer *outputs[],
                                            aclopAttr *attr,
                                            aclrtStream stream);

ACL_FUNC_VISIBILITY aclError aclopCreateHandleImpl(const char *opType,
                                               int numInputs,
                                               const aclTensorDesc *const inputDesc[],
                                               int numOutputs,
                                               const aclTensorDesc *const outputDesc[],
                                               const aclopAttr *opAttr,
                                               aclopHandle **handle);

ACL_FUNC_VISIBILITY void aclopDestroyHandleImpl(aclopHandle *handle);

ACL_FUNC_VISIBILITY aclError aclopExecWithHandleImpl(aclopHandle *handle,
                                                 int numInputs,
                                                 const aclDataBuffer *const inputs[],
                                                 int numOutputs,
                                                 aclDataBuffer *const outputs[],
                                                 aclrtStream stream);

ACL_FUNC_VISIBILITY aclError aclopCastImpl(const aclTensorDesc *srcDesc,
                                       const aclDataBuffer *srcBuffer,
                                       const aclTensorDesc *dstDesc,
                                       aclDataBuffer *dstBuffer,
                                       uint8_t truncate,
                                       aclrtStream stream);

ACL_FUNC_VISIBILITY aclError aclopCreateHandleForCastImpl(aclTensorDesc *srcDesc,
                                                      aclTensorDesc *dstDesc,
                                                      uint8_t truncate,
                                                      aclopHandle **handle);

ACL_FUNC_VISIBILITY aclError aclopCreateKernelImpl(const char *opType,
                                               const char *kernelId,
                                               const char *kernelName,
                                               void *binData,
                                               int binSize,
                                               aclopEngineType enginetype,
                                               aclDataDeallocator deallocator);

ACL_FUNC_VISIBILITY aclError aclopRegisterCompileFuncImpl(const char *opType, aclopCompileFunc func);

ACL_FUNC_VISIBILITY aclError aclopUnregisterCompileFuncImpl(const char *opType);

ACL_FUNC_VISIBILITY aclError aclopSetKernelArgsImpl(aclopKernelDesc *kernelDesc,
                                                const char *kernelId,
                                                uint32_t blockDim,
                                                const void *args,
                                                uint32_t argSize);

ACL_FUNC_VISIBILITY aclError aclopSetKernelWorkspaceSizesImpl(aclopKernelDesc *kernelDesc, int numWorkspaces,
                                                          size_t *workspaceSizes);

ACL_FUNC_VISIBILITY aclError aclopUpdateParamsImpl(const char *opType,
                                               int numInputs,
                                               const aclTensorDesc *const inputDesc[],
                                               int numOutputs,
                                               const aclTensorDesc *const outputDesc[],
                                               const aclopAttr *attr);

ACL_FUNC_VISIBILITY aclError aclopSetMaxOpQueueNumImpl(uint64_t maxOpNum);

ACL_FUNC_VISIBILITY aclError aclopInferShapeImpl(const char *opType,
                                             int numInputs,
                                             aclTensorDesc *inputDesc[],
                                             aclDataBuffer *inputs[],
                                             int numOutputs,
                                             aclTensorDesc *outputDesc[],
                                             aclopAttr *attr);

ACL_FUNC_VISIBILITY aclError aclopStartDumpArgsImpl(uint32_t dumpType, const char *path);

ACL_FUNC_VISIBILITY aclError aclopStopDumpArgsImpl(uint32_t dumpType);

#ifdef __cplusplus
}
#endif

#endif // ACL_OP_EXECUTOR_SRC_SINGLE_OP_ACL_OP_EXECUTOR_IMPL_H_
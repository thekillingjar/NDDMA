/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACL_MODEL_SRC_MODEL_ACL_MODEL_IMPL_H_
#define ACL_MODEL_SRC_MODEL_ACL_MODEL_IMPL_H_

#include <stddef.h>
#include <stdint.h>

#include "acl_base_rt.h"
#include "acl_rt.h"
#include "acl_mdl.h"

#ifdef __cplusplus
extern "C" {
#endif

ACL_FUNC_VISIBILITY aclmdlDesc *aclmdlCreateDescImpl();

ACL_FUNC_VISIBILITY aclError aclmdlDestroyDescImpl(aclmdlDesc *modelDesc);

ACL_FUNC_VISIBILITY aclError aclmdlGetDescImpl(aclmdlDesc *modelDesc, uint32_t modelId);

ACL_FUNC_VISIBILITY aclError aclmdlGetDescFromFileImpl(aclmdlDesc *modelDesc, const char *modelPath);

ACL_FUNC_VISIBILITY aclError aclmdlGetDescFromMemImpl(aclmdlDesc *modelDesc, const void *model, size_t modelSize);

ACL_FUNC_VISIBILITY size_t aclmdlGetNumInputsImpl(aclmdlDesc *modelDesc);

ACL_FUNC_VISIBILITY size_t aclmdlGetNumOutputsImpl(aclmdlDesc *modelDesc);

ACL_FUNC_VISIBILITY size_t aclmdlGetInputSizeByIndexImpl(aclmdlDesc *modelDesc, size_t index);

ACL_FUNC_VISIBILITY size_t aclmdlGetOutputSizeByIndexImpl(aclmdlDesc *modelDesc, size_t index);

ACL_FUNC_VISIBILITY aclmdlExecConfigHandle *aclmdlCreateExecConfigHandleImpl();

ACL_FUNC_VISIBILITY aclError aclmdlDestroyExecConfigHandleImpl(const aclmdlExecConfigHandle *handle);

ACL_FUNC_VISIBILITY aclmdlDataset *aclmdlCreateDatasetImpl();

ACL_FUNC_VISIBILITY aclError aclmdlDestroyDatasetImpl(const aclmdlDataset *dataset);

ACL_FUNC_VISIBILITY aclError aclmdlAddDatasetBufferImpl(aclmdlDataset *dataset, aclDataBuffer *dataBuffer);

ACL_FUNC_VISIBILITY aclError aclmdlSetDatasetTensorDescImpl(aclmdlDataset *dataset,
                                                        aclTensorDesc *tensorDesc,
                                                        size_t index);

ACL_FUNC_VISIBILITY aclTensorDesc *aclmdlGetDatasetTensorDescImpl(const aclmdlDataset *dataset, size_t index);

ACL_FUNC_VISIBILITY size_t aclmdlGetDatasetNumBuffersImpl(const aclmdlDataset *dataset);

ACL_FUNC_VISIBILITY aclDataBuffer *aclmdlGetDatasetBufferImpl(const aclmdlDataset *dataset, size_t index);

ACL_FUNC_VISIBILITY aclError aclmdlLoadFromFileImpl(const char *modelPath, uint32_t *modelId);

ACL_FUNC_VISIBILITY aclError aclmdlBundleLoadFromFileImpl(const char *modelPath, uint32_t *bundleId);

ACL_FUNC_VISIBILITY aclError aclmdlBundleLoadFromMemImpl(const void *model,  size_t modelSize, uint32_t *bundleId);

ACL_FUNC_VISIBILITY aclError aclmdlBundleUnloadImpl(uint32_t bundleId);

ACL_FUNC_VISIBILITY aclError aclmdlBundleGetModelNumImpl(uint32_t bundleId, size_t *modelNum);

ACL_FUNC_VISIBILITY aclError aclmdlBundleGetModelIdImpl(uint32_t bundleId, size_t index, uint32_t *modelId);

ACL_FUNC_VISIBILITY aclmdlBundleQueryInfo *aclmdlBundleCreateQueryInfoImpl();

ACL_FUNC_VISIBILITY aclError aclmdlBundleDestroyQueryInfoImpl(aclmdlBundleQueryInfo *queryInfo);

ACL_FUNC_VISIBILITY aclError aclmdlBundleQueryInfoFromFileImpl(const char* fileName, aclmdlBundleQueryInfo *queryInfo);

ACL_FUNC_VISIBILITY aclError aclmdlBundleQueryInfoFromMemImpl(const void *model, size_t modelSize,
                                                              aclmdlBundleQueryInfo *queryInfo);

ACL_FUNC_VISIBILITY aclError aclmdlBundleGetQueryModelNumImpl(const aclmdlBundleQueryInfo *queryInfo, size_t *modelNum);


ACL_FUNC_VISIBILITY aclError aclmdlBundleGetVarWeightSizeImpl(const aclmdlBundleQueryInfo *queryInfo,
                                                              size_t *variableWeightSize);


ACL_FUNC_VISIBILITY aclError aclmdlBundleGetSizeImpl(const aclmdlBundleQueryInfo *queryInfo, size_t index,
                                                    size_t *workSize, size_t *constWeightSize);

ACL_FUNC_VISIBILITY aclError aclmdlBundleInitFromFileImpl(const char* modelPath, void *varWeightPtr,
                                                          size_t varWeightSize, uint32_t *bundleId);

ACL_FUNC_VISIBILITY aclError aclmdlBundleInitFromMemImpl(const void* model, size_t modelSize, void *varWeightPtr,
                                                          size_t varWeightSize, uint32_t *bundleId);

ACL_FUNC_VISIBILITY aclError aclmdlBundleLoadModelImpl(uint32_t bundleId, size_t index, uint32_t *modelId);

ACL_FUNC_VISIBILITY aclError aclmdlBundleLoadModelWithMemImpl(uint32_t bundleId, size_t index, void *workPtr,
                                                              size_t workSize, void *weightPtr,
                                                              size_t weightSize, uint32_t *modelId);

ACL_FUNC_VISIBILITY aclError aclmdlBundleLoadModelWithConfigImpl(uint32_t bundleId, size_t index,
                                                                aclmdlConfigHandle *handle, uint32_t *modelId);

ACL_FUNC_VISIBILITY aclError aclmdlBundleUnloadModelImpl(uint32_t bundleId, uint32_t modelId);

ACL_FUNC_VISIBILITY aclError aclmdlLoadFromMemImpl(const void *model,  size_t modelSize, uint32_t *modelId);

ACL_FUNC_VISIBILITY aclError aclmdlLoadFromFileWithMemImpl(const char *modelPath,
                                                       uint32_t *modelId, void *workPtr, size_t workSize,
                                                       void *weightPtr, size_t weightSize);

ACL_FUNC_VISIBILITY aclError aclmdlLoadFromMemWithMemImpl(const void *model, size_t modelSize,
                                                      uint32_t *modelId, void *workPtr, size_t workSize,
                                                      void *weightPtr, size_t weightSize);

ACL_FUNC_VISIBILITY aclError aclmdlLoadFromFileWithQImpl(const char *modelPath, uint32_t *modelId, const uint32_t *inputQ,
                                                     size_t inputQNum, const uint32_t *outputQ, size_t outputQNum);

ACL_FUNC_VISIBILITY aclError aclmdlLoadFromMemWithQImpl(const void *model, size_t modelSize, uint32_t *modelId,
                                                    const uint32_t *inputQ, size_t inputQNum,
                                                    const uint32_t *outputQ, size_t outputQNum);

ACL_FUNC_VISIBILITY aclError aclmdlExecuteImpl(uint32_t modelId, const aclmdlDataset *input, aclmdlDataset *output);

ACL_FUNC_VISIBILITY aclError aclmdlExecuteV2Impl(uint32_t modelId, const aclmdlDataset *input, aclmdlDataset *output,
                                             aclrtStream stream, const aclmdlExecConfigHandle *handle);

ACL_FUNC_VISIBILITY  aclError aclmdlExecuteAsyncV2Impl(uint32_t modelId, const aclmdlDataset *input, aclmdlDataset *output,
                                                   aclrtStream stream, const aclmdlExecConfigHandle *handle);

ACL_FUNC_VISIBILITY aclError aclmdlExecuteAsyncImpl(uint32_t modelId, const aclmdlDataset *input,
                                                aclmdlDataset *output, aclrtStream stream);

ACL_FUNC_VISIBILITY aclError aclmdlUnloadImpl(uint32_t modelId);

ACL_FUNC_VISIBILITY aclError aclmdlQuerySizeImpl(const char *fileName, size_t *workSize, size_t *weightSize);

ACL_FUNC_VISIBILITY aclError aclmdlQueryExeOMDescImpl(const char *fileName, aclmdlExeOMDesc *mdlPartitionSize);

ACL_FUNC_VISIBILITY aclError aclmdlQuerySizeFromMemImpl(const void *model, size_t modelSize, size_t *workSize,
                                                    size_t *weightSize);

ACL_FUNC_VISIBILITY aclError aclmdlSetDynamicBatchSizeImpl(uint32_t modelId, aclmdlDataset *dataset, size_t index,
                                                       uint64_t batchSize);

ACL_FUNC_VISIBILITY aclError aclmdlSetDynamicHWSizeImpl(uint32_t modelId, aclmdlDataset *dataset, size_t index,
                                                    uint64_t height, uint64_t width);

ACL_FUNC_VISIBILITY aclError aclmdlSetInputDynamicDimsImpl(uint32_t modelId, aclmdlDataset *dataset, size_t index,
                                                       const aclmdlIODims *dims);

ACL_FUNC_VISIBILITY aclError aclmdlGetInputDimsImpl(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims);

ACL_FUNC_VISIBILITY aclError aclmdlGetInputDimsV2Impl(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims);

ACL_FUNC_VISIBILITY aclError aclmdlGetInputDimsRangeImpl(const aclmdlDesc *modelDesc, size_t index,
                                                     aclmdlIODimsRange *dimsRange);

ACL_FUNC_VISIBILITY aclError aclmdlGetOutputDimsImpl(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims);

ACL_FUNC_VISIBILITY aclError aclmdlGetCurOutputDimsImpl(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims);

ACL_FUNC_VISIBILITY const char *aclmdlGetOpAttrImpl(aclmdlDesc *modelDesc, const char *opName, const char *attr);

ACL_FUNC_VISIBILITY const char *aclmdlGetInputNameByIndexImpl(const aclmdlDesc *modelDesc, size_t index);

ACL_FUNC_VISIBILITY const char *aclmdlGetOutputNameByIndexImpl(const aclmdlDesc *modelDesc, size_t index);

ACL_FUNC_VISIBILITY aclFormat aclmdlGetInputFormatImpl(const aclmdlDesc *modelDesc, size_t index);

ACL_FUNC_VISIBILITY aclFormat aclmdlGetOutputFormatImpl(const aclmdlDesc *modelDesc, size_t index);

ACL_FUNC_VISIBILITY aclDataType aclmdlGetInputDataTypeImpl(const aclmdlDesc *modelDesc, size_t index);

ACL_FUNC_VISIBILITY aclDataType aclmdlGetOutputDataTypeImpl(const aclmdlDesc *modelDesc, size_t index);

ACL_FUNC_VISIBILITY aclError aclmdlGetInputIndexByNameImpl(const aclmdlDesc *modelDesc, const char *name, size_t *index);

ACL_FUNC_VISIBILITY aclError aclmdlGetOutputIndexByNameImpl(const aclmdlDesc *modelDesc, const char *name, size_t *index);

ACL_FUNC_VISIBILITY aclError aclmdlGetDynamicBatchImpl(const aclmdlDesc *modelDesc, aclmdlBatch *batch);

ACL_FUNC_VISIBILITY aclError aclmdlGetDynamicHWImpl(const aclmdlDesc *modelDesc, size_t index, aclmdlHW *hw);

ACL_FUNC_VISIBILITY aclError aclmdlGetInputDynamicGearCountImpl(const aclmdlDesc *modelDesc, size_t index,
                                                            size_t *gearCount);

ACL_FUNC_VISIBILITY aclError aclmdlGetInputDynamicDimsImpl(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims,
                                                       size_t gearCount);

ACL_FUNC_VISIBILITY aclmdlAIPP *aclmdlCreateAIPPImpl(uint64_t batchSize);

ACL_FUNC_VISIBILITY aclError aclmdlDestroyAIPPImpl(const aclmdlAIPP *aippParmsSet);

ACL_FUNC_VISIBILITY aclError aclmdlGetAippDataSizeImpl(uint64_t batchSize, size_t *size);

ACL_FUNC_VISIBILITY aclError aclmdlSetAIPPInputFormatImpl(aclmdlAIPP *aippParmsSet, aclAippInputFormat inputFormat);

ACL_FUNC_VISIBILITY aclError aclmdlSetAIPPCscParamsImpl(aclmdlAIPP *aippParmsSet, int8_t cscSwitch,
                                                    int16_t cscMatrixR0C0, int16_t cscMatrixR0C1, int16_t cscMatrixR0C2,
                                                    int16_t cscMatrixR1C0, int16_t cscMatrixR1C1, int16_t cscMatrixR1C2,
                                                    int16_t cscMatrixR2C0, int16_t cscMatrixR2C1, int16_t cscMatrixR2C2,
                                                    uint8_t cscOutputBiasR0, uint8_t cscOutputBiasR1,
                                                    uint8_t cscOutputBiasR2, uint8_t cscInputBiasR0,
                                                    uint8_t cscInputBiasR1, uint8_t cscInputBiasR2);

ACL_FUNC_VISIBILITY aclError aclmdlSetAIPPRbuvSwapSwitchImpl(aclmdlAIPP *aippParmsSet, int8_t rbuvSwapSwitch);

ACL_FUNC_VISIBILITY aclError aclmdlSetAIPPAxSwapSwitchImpl(aclmdlAIPP *aippParmsSet, int8_t axSwapSwitch);

ACL_FUNC_VISIBILITY aclError aclmdlSetAIPPSrcImageSizeImpl(aclmdlAIPP *aippParmsSet, int32_t srcImageSizeW,
                                                       int32_t srcImageSizeH);

ACL_FUNC_VISIBILITY aclError aclmdlSetAIPPScfParamsImpl(aclmdlAIPP *aippParmsSet,
                                                    int8_t scfSwitch,
                                                    int32_t scfInputSizeW,
                                                    int32_t scfInputSizeH,
                                                    int32_t scfOutputSizeW,
                                                    int32_t scfOutputSizeH,
                                                    uint64_t batchIndex);

ACL_FUNC_VISIBILITY aclError aclmdlSetAIPPCropParamsImpl(aclmdlAIPP *aippParmsSet,
                                                     int8_t cropSwitch,
                                                     int32_t cropStartPosW,
                                                     int32_t cropStartPosH,
                                                     int32_t cropSizeW,
                                                     int32_t cropSizeH,
                                                     uint64_t batchIndex);

ACL_FUNC_VISIBILITY aclError aclmdlSetAIPPPaddingParamsImpl(aclmdlAIPP *aippParmsSet, int8_t paddingSwitch,
                                                        int32_t paddingSizeTop, int32_t paddingSizeBottom,
                                                        int32_t paddingSizeLeft, int32_t paddingSizeRight,
                                                        uint64_t batchIndex);

ACL_FUNC_VISIBILITY aclError aclmdlSetAIPPDtcPixelMeanImpl(aclmdlAIPP *aippParmsSet,
                                                       int16_t dtcPixelMeanChn0,
                                                       int16_t dtcPixelMeanChn1,
                                                       int16_t dtcPixelMeanChn2,
                                                       int16_t dtcPixelMeanChn3,
                                                       uint64_t batchIndex);

ACL_FUNC_VISIBILITY aclError aclmdlSetAIPPDtcPixelMinImpl(aclmdlAIPP *aippParmsSet,
                                                      float dtcPixelMinChn0,
                                                      float dtcPixelMinChn1,
                                                      float dtcPixelMinChn2,
                                                      float dtcPixelMinChn3,
                                                      uint64_t batchIndex);

ACL_FUNC_VISIBILITY aclError aclmdlSetAIPPPixelVarReciImpl(aclmdlAIPP *aippParmsSet,
                                                       float dtcPixelVarReciChn0,
                                                       float dtcPixelVarReciChn1,
                                                       float dtcPixelVarReciChn2,
                                                       float dtcPixelVarReciChn3,
                                                       uint64_t batchIndex);

ACL_FUNC_VISIBILITY aclError aclmdlSetInputAIPPImpl(uint32_t modelId,
                                                aclmdlDataset *dataset,
                                                size_t index,
                                                const aclmdlAIPP *aippParmsSet);

ACL_FUNC_VISIBILITY aclError aclmdlSetAIPPByInputIndexImpl(uint32_t modelId,
                                                       aclmdlDataset *dataset,
                                                       size_t index,
                                                       const aclmdlAIPP *aippParmsSet);

ACL_FUNC_VISIBILITY aclError aclmdlGetAippTypeImpl(uint32_t modelId,
                                               size_t index,
                                               aclmdlInputAippType *type,
                                               size_t *dynamicAttachedDataIndex);

ACL_FUNC_VISIBILITY aclError aclmdlGetFirstAippInfoImpl(uint32_t modelId, size_t index, aclAippInfo *aippInfo);

ACL_FUNC_VISIBILITY aclError aclmdlCreateAndGetOpDescImpl(uint32_t deviceId, uint32_t streamId,
    uint32_t taskId, char *opName, size_t opNameLen, aclTensorDesc **inputDesc, size_t *numInputs,
    aclTensorDesc **outputDesc, size_t *numOutputs);

ACL_FUNC_VISIBILITY aclError aclmdlInitDumpImpl();

ACL_FUNC_VISIBILITY aclError aclmdlSetDumpImpl(const char *dumpCfgPath);

ACL_FUNC_VISIBILITY aclError aclmdlFinalizeDumpImpl();

ACL_FUNC_VISIBILITY aclError aclmdlLoadWithConfigImpl(const aclmdlConfigHandle *handle, uint32_t *modelId);

ACL_FUNC_VISIBILITY aclError aclmdlSetExternalWeightAddressImpl(aclmdlConfigHandle *handle, const char *weightFileName, void *devPtr, size_t size);

ACL_FUNC_VISIBILITY aclmdlConfigHandle *aclmdlCreateConfigHandleImpl();

ACL_FUNC_VISIBILITY aclError aclmdlDestroyConfigHandleImpl(aclmdlConfigHandle *handle);

ACL_FUNC_VISIBILITY aclError aclmdlSetConfigOptImpl(aclmdlConfigHandle *handle, aclmdlConfigAttr attr,
    const void *attrValue, size_t valueSize);

ACL_FUNC_VISIBILITY aclError aclmdlSetExecConfigOptImpl(aclmdlExecConfigHandle *handle, aclmdlExecConfigAttr attr,
                                                    const void *attrValue, size_t valueSize);

ACL_FUNC_VISIBILITY const char *aclmdlGetTensorRealNameImpl(const aclmdlDesc *modelDesc, const char *name);

ACL_FUNC_VISIBILITY aclError aclRecoverAllHcclTasksImpl(int32_t deviceId);

ACL_FUNC_VISIBILITY aclTensorDesc *aclCreateTensorDescImpl(aclDataType dataType,
                                                       int numDims,
                                                       const int64_t *dims,
                                                       aclFormat format);

ACL_FUNC_VISIBILITY void aclDestroyTensorDescImpl(const aclTensorDesc *desc);

ACL_FUNC_VISIBILITY aclError aclSetTensorShapeRangeImpl(aclTensorDesc* desc,
                                                    size_t dimsCount,
                                                    int64_t dimsRange[][ACL_TENSOR_SHAPE_RANGE_NUM]);

ACL_FUNC_VISIBILITY aclError aclSetTensorValueRangeImpl(aclTensorDesc* desc,
                                                    size_t valueCount,
                                                    int64_t valueRange[][ACL_TENSOR_VALUE_RANGE_NUM]);

ACL_FUNC_VISIBILITY aclDataType aclGetTensorDescTypeImpl(const aclTensorDesc *desc);

ACL_FUNC_VISIBILITY aclFormat aclGetTensorDescFormatImpl(const aclTensorDesc *desc);

ACL_FUNC_VISIBILITY size_t aclGetTensorDescSizeImpl(const aclTensorDesc *desc);

ACL_FUNC_VISIBILITY size_t aclGetTensorDescElementCountImpl(const aclTensorDesc *desc);

ACL_FUNC_VISIBILITY size_t aclGetTensorDescNumDimsImpl(const aclTensorDesc *desc);

ACL_FUNC_VISIBILITY int64_t aclGetTensorDescDimImpl(const aclTensorDesc *desc, size_t index);

ACL_FUNC_VISIBILITY aclError aclGetTensorDescDimV2Impl(const aclTensorDesc *desc, size_t index, int64_t *dimSize);

ACL_FUNC_VISIBILITY aclError aclGetTensorDescDimRangeImpl(const aclTensorDesc *desc,
                                                      size_t index,
                                                      size_t dimRangeNum,
                                                      int64_t *dimRange);

ACL_FUNC_VISIBILITY void aclSetTensorDescNameImpl(aclTensorDesc *desc, const char *name);

ACL_FUNC_VISIBILITY const char *aclGetTensorDescNameImpl(aclTensorDesc *desc);

ACL_FUNC_VISIBILITY aclError aclTransTensorDescFormatImpl(const aclTensorDesc *srcDesc, aclFormat dstFormat,
                                                      aclTensorDesc **dstDesc);

ACL_FUNC_VISIBILITY aclError aclSetTensorStorageFormatImpl(aclTensorDesc *desc, aclFormat format);

ACL_FUNC_VISIBILITY aclError aclSetTensorStorageShapeImpl(aclTensorDesc *desc, int numDims, const int64_t *dims);

ACL_FUNC_VISIBILITY aclError aclSetTensorFormatImpl(aclTensorDesc *desc, aclFormat format);

ACL_FUNC_VISIBILITY aclError aclSetTensorShapeImpl(aclTensorDesc *desc, int numDims, const int64_t *dims);

ACL_FUNC_VISIBILITY aclError aclSetTensorOriginFormatImpl(aclTensorDesc *desc, aclFormat format);

ACL_FUNC_VISIBILITY aclError aclSetTensorOriginShapeImpl(aclTensorDesc *desc, int numDims, const int64_t *dims);

ACL_FUNC_VISIBILITY aclTensorDesc *aclGetTensorDescByIndexImpl(aclTensorDesc *desc, size_t index);

ACL_FUNC_VISIBILITY void *aclGetTensorDescAddressImpl(const aclTensorDesc *desc);

ACL_FUNC_VISIBILITY aclError aclSetTensorDynamicInputImpl(aclTensorDesc *desc, const char *dynamicInputName);

ACL_FUNC_VISIBILITY aclError aclSetTensorConstImpl(aclTensorDesc *desc, void *dataBuffer, size_t length);

ACL_FUNC_VISIBILITY aclError aclSetTensorPlaceMentImpl(aclTensorDesc *desc, aclMemType memType);

#ifdef __cplusplus
}
#endif

#endif // ACL_MODEL_SRC_MODEL_ACL_MODEL_IMPL_H_

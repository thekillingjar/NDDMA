/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "acl_model_impl.h"

#ifdef __cplusplus
extern "C" {
#endif

aclmdlDesc *aclmdlCreateDesc()
{
    return aclmdlCreateDescImpl();
}

aclError aclmdlDestroyDesc(aclmdlDesc *modelDesc)
{
    return aclmdlDestroyDescImpl(modelDesc);
}

aclError aclmdlGetDesc(aclmdlDesc *modelDesc, uint32_t modelId)
{
    return aclmdlGetDescImpl(modelDesc, modelId);
}

aclError aclmdlGetDescFromFile(aclmdlDesc *modelDesc, const char *modelPath)
{
    return aclmdlGetDescFromFileImpl(modelDesc, modelPath);
}

aclError aclmdlGetDescFromMem(aclmdlDesc *modelDesc, const void *model, size_t modelSize)
{
    return aclmdlGetDescFromMemImpl(modelDesc, model, modelSize);
}

size_t aclmdlGetNumInputs(aclmdlDesc *modelDesc)
{
    return aclmdlGetNumInputsImpl(modelDesc);
}

size_t aclmdlGetNumOutputs(aclmdlDesc *modelDesc)
{
    return aclmdlGetNumOutputsImpl(modelDesc);
}

size_t aclmdlGetInputSizeByIndex(aclmdlDesc *modelDesc, size_t index)
{
    return aclmdlGetInputSizeByIndexImpl(modelDesc, index);
}

size_t aclmdlGetOutputSizeByIndex(aclmdlDesc *modelDesc, size_t index)
{
    return aclmdlGetOutputSizeByIndexImpl(modelDesc, index);
}

aclmdlExecConfigHandle *aclmdlCreateExecConfigHandle()
{
    return aclmdlCreateExecConfigHandleImpl();
}

aclError aclmdlDestroyExecConfigHandle(const aclmdlExecConfigHandle *handle)
{
    return aclmdlDestroyExecConfigHandleImpl(handle);
}

aclmdlDataset *aclmdlCreateDataset()
{
    return aclmdlCreateDatasetImpl();
}

aclError aclmdlDestroyDataset(const aclmdlDataset *dataset)
{
    return aclmdlDestroyDatasetImpl(dataset);
}

aclError aclmdlAddDatasetBuffer(aclmdlDataset *dataset, aclDataBuffer *dataBuffer)
{
    return aclmdlAddDatasetBufferImpl(dataset, dataBuffer);
}

aclError aclmdlSetDatasetTensorDesc(aclmdlDataset *dataset,
                                                        aclTensorDesc *tensorDesc,
                                                        size_t index)
{
    return aclmdlSetDatasetTensorDescImpl(dataset, tensorDesc, index);
}

aclTensorDesc *aclmdlGetDatasetTensorDesc(const aclmdlDataset *dataset, size_t index)
{
    return aclmdlGetDatasetTensorDescImpl(dataset, index);
}

size_t aclmdlGetDatasetNumBuffers(const aclmdlDataset *dataset)
{
    return aclmdlGetDatasetNumBuffersImpl(dataset);
}

aclDataBuffer *aclmdlGetDatasetBuffer(const aclmdlDataset *dataset, size_t index)
{
    return aclmdlGetDatasetBufferImpl(dataset, index);
}

aclError aclmdlLoadFromFile(const char *modelPath, uint32_t *modelId)
{
    return aclmdlLoadFromFileImpl(modelPath, modelId);
}

aclError aclmdlBundleLoadFromFile(const char *modelPath, uint32_t *bundleId)
{
    return aclmdlBundleLoadFromFileImpl(modelPath, bundleId);
}

aclError aclmdlBundleLoadFromMem(const void *model,  size_t modelSize, uint32_t *bundleId)
{
    return aclmdlBundleLoadFromMemImpl(model, modelSize, bundleId);
}

aclError aclmdlBundleUnload(uint32_t bundleId)
{
    return aclmdlBundleUnloadImpl(bundleId);
}

aclError aclmdlBundleGetModelNum(uint32_t bundleId, size_t *modelNum)
{
    return aclmdlBundleGetModelNumImpl(bundleId, modelNum);
}

aclError aclmdlBundleGetModelId(uint32_t bundleId, size_t index, uint32_t *modelId)
{
    return aclmdlBundleGetModelIdImpl(bundleId, index, modelId);
}

aclmdlBundleQueryInfo *aclmdlBundleCreateQueryInfo()
{
    return aclmdlBundleCreateQueryInfoImpl();
}

aclError aclmdlBundleDestroyQueryInfo(aclmdlBundleQueryInfo *queryInfo)
{
  return aclmdlBundleDestroyQueryInfoImpl(queryInfo);
}

aclError aclmdlBundleQueryInfoFromFile(const char* fileName, aclmdlBundleQueryInfo *queryInfo)
{
  return aclmdlBundleQueryInfoFromFileImpl(fileName, queryInfo);
}

aclError aclmdlBundleQueryInfoFromMem(const void *model, size_t modelSize, aclmdlBundleQueryInfo *queryInfo)
{
  return aclmdlBundleQueryInfoFromMemImpl(model, modelSize, queryInfo);
}

aclError aclmdlBundleGetQueryModelNum(const aclmdlBundleQueryInfo *queryInfo, size_t *modelNum)
{
  return aclmdlBundleGetQueryModelNumImpl(queryInfo, modelNum);
}

aclError aclmdlBundleGetVarWeightSize(const aclmdlBundleQueryInfo *queryInfo, size_t *variableWeightSize)
{
  return aclmdlBundleGetVarWeightSizeImpl(queryInfo, variableWeightSize);
}

aclError aclmdlBundleGetSize(const aclmdlBundleQueryInfo *queryInfo, size_t index,
                             size_t *workSize, size_t *constWeightSize)
{
  return aclmdlBundleGetSizeImpl(queryInfo, index, workSize, constWeightSize);
}

aclError aclmdlBundleInitFromFile(const char* modelPath, void *varWeightPtr,
                                  size_t varWeightSize, uint32_t *bundleId)
{
  return aclmdlBundleInitFromFileImpl(modelPath, varWeightPtr, varWeightSize, bundleId);
}

aclError aclmdlBundleInitFromMem(const void* model, size_t modelSize, void *varWeightPtr,
                                 size_t varWeightSize, uint32_t *bundleId)
{
  return aclmdlBundleInitFromMemImpl(model, modelSize, varWeightPtr, varWeightSize, bundleId);
}

aclError aclmdlBundleLoadModel(uint32_t bundleId, size_t index, uint32_t *modelId)
{
  return aclmdlBundleLoadModelImpl(bundleId, index, modelId);
}

aclError aclmdlBundleLoadModelWithMem(uint32_t bundleId, size_t index, void *workPtr,
                                      size_t workSize, void *weightPtr,
                                      size_t weightSize, uint32_t *modelId)
{
  return aclmdlBundleLoadModelWithMemImpl(bundleId, index, workPtr, workSize, weightPtr, weightSize, modelId);
}

aclError aclmdlBundleLoadModelWithConfig(uint32_t bundleId, size_t index,
                                         aclmdlConfigHandle *handle, uint32_t *modelId)
{
  return aclmdlBundleLoadModelWithConfigImpl(bundleId, index, handle, modelId);
}

aclError aclmdlBundleUnloadModel(uint32_t bundleId, uint32_t modelId)
{
  return aclmdlBundleUnloadModelImpl(bundleId, modelId);
}

aclError aclmdlLoadFromMem(const void *model,  size_t modelSize, uint32_t *modelId)
{
    return aclmdlLoadFromMemImpl(model, modelSize, modelId);
}

aclError aclmdlLoadFromFileWithMem(const char *modelPath,
                                                       uint32_t *modelId, void *workPtr, size_t workSize,
                                                       void *weightPtr, size_t weightSize)
{
    return aclmdlLoadFromFileWithMemImpl(modelPath, modelId, workPtr, workSize, weightPtr, weightSize);
}

aclError aclmdlLoadFromMemWithMem(const void *model, size_t modelSize,
                                                      uint32_t *modelId, void *workPtr, size_t workSize,
                                                      void *weightPtr, size_t weightSize)
{
    return aclmdlLoadFromMemWithMemImpl(model, modelSize, modelId, workPtr, workSize, weightPtr, weightSize);
}

aclError aclmdlLoadFromFileWithQ(const char *modelPath, uint32_t *modelId, const uint32_t *inputQ,
                                                     size_t inputQNum, const uint32_t *outputQ, size_t outputQNum)
{
    return aclmdlLoadFromFileWithQImpl(modelPath, modelId, inputQ, inputQNum, outputQ, outputQNum);
}

aclError aclmdlLoadFromMemWithQ(const void *model, size_t modelSize, uint32_t *modelId,
                                                    const uint32_t *inputQ, size_t inputQNum,
                                                    const uint32_t *outputQ, size_t outputQNum)
{
    return aclmdlLoadFromMemWithQImpl(model, modelSize, modelId, inputQ, inputQNum, outputQ, outputQNum);
}

aclError aclmdlExecute(uint32_t modelId, const aclmdlDataset *input, aclmdlDataset *output)
{
    return aclmdlExecuteImpl(modelId, input, output);
}

aclError aclmdlExecuteV2(uint32_t modelId, const aclmdlDataset *input, aclmdlDataset *output,
                                             aclrtStream stream, const aclmdlExecConfigHandle *handle)
{
    return aclmdlExecuteV2Impl(modelId, input, output, stream, handle);
}

aclError aclmdlExecuteAsync(uint32_t modelId, const aclmdlDataset *input,
                                                aclmdlDataset *output, aclrtStream stream)
{
    return aclmdlExecuteAsyncImpl(modelId, input, output, stream);
}

aclError aclmdlUnload(uint32_t modelId)
{
    return aclmdlUnloadImpl(modelId);
}

aclError aclmdlQuerySize(const char *fileName, size_t *workSize, size_t *weightSize)
{
    return aclmdlQuerySizeImpl(fileName, workSize, weightSize);
}

aclError aclmdlQuerySizeFromMem(const void *model, size_t modelSize, size_t *workSize,
                                                    size_t *weightSize)
{
    return aclmdlQuerySizeFromMemImpl(model, modelSize, workSize, weightSize);
}

aclError aclmdlSetDynamicBatchSize(uint32_t modelId, aclmdlDataset *dataset, size_t index,
                                                       uint64_t batchSize)
{
    return aclmdlSetDynamicBatchSizeImpl(modelId, dataset, index, batchSize);
}

aclError aclmdlSetDynamicHWSize(uint32_t modelId, aclmdlDataset *dataset, size_t index,
                                                    uint64_t height, uint64_t width)
{
    return aclmdlSetDynamicHWSizeImpl(modelId, dataset, index, height, width);
}

aclError aclmdlSetInputDynamicDims(uint32_t modelId, aclmdlDataset *dataset, size_t index,
                                                       const aclmdlIODims *dims)
{
    return aclmdlSetInputDynamicDimsImpl(modelId, dataset, index, dims);
}

aclError aclmdlGetInputDims(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims)
{
    return aclmdlGetInputDimsImpl(modelDesc, index, dims);
}

aclError aclmdlGetInputDimsV2(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims)
{
    return aclmdlGetInputDimsV2Impl(modelDesc, index, dims);
}

aclError aclmdlGetInputDimsRange(const aclmdlDesc *modelDesc, size_t index,
                                                     aclmdlIODimsRange *dimsRange)
{
    return aclmdlGetInputDimsRangeImpl(modelDesc, index, dimsRange);
}

aclError aclmdlGetOutputDims(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims)
{
    return aclmdlGetOutputDimsImpl(modelDesc, index, dims);
}

aclError aclmdlGetCurOutputDims(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims)
{
    return aclmdlGetCurOutputDimsImpl(modelDesc, index, dims);
}

const char *aclmdlGetOpAttr(aclmdlDesc *modelDesc, const char *opName, const char *attr)
{
    return aclmdlGetOpAttrImpl(modelDesc, opName, attr);
}

const char *aclmdlGetInputNameByIndex(const aclmdlDesc *modelDesc, size_t index)
{
    return aclmdlGetInputNameByIndexImpl(modelDesc, index);
}

const char *aclmdlGetOutputNameByIndex(const aclmdlDesc *modelDesc, size_t index)
{
    return aclmdlGetOutputNameByIndexImpl(modelDesc, index);
}

aclFormat aclmdlGetInputFormat(const aclmdlDesc *modelDesc, size_t index)
{
    return aclmdlGetInputFormatImpl(modelDesc, index);
}

aclFormat aclmdlGetOutputFormat(const aclmdlDesc *modelDesc, size_t index)
{
    return aclmdlGetOutputFormatImpl(modelDesc, index);
}

aclDataType aclmdlGetInputDataType(const aclmdlDesc *modelDesc, size_t index)
{
    return aclmdlGetInputDataTypeImpl(modelDesc, index);
}

aclDataType aclmdlGetOutputDataType(const aclmdlDesc *modelDesc, size_t index)
{
    return aclmdlGetOutputDataTypeImpl(modelDesc, index);
}

aclError aclmdlGetInputIndexByName(const aclmdlDesc *modelDesc, const char *name, size_t *index)
{
    return aclmdlGetInputIndexByNameImpl(modelDesc, name, index);
}

aclError aclmdlGetOutputIndexByName(const aclmdlDesc *modelDesc, const char *name, size_t *index)
{
    return aclmdlGetOutputIndexByNameImpl(modelDesc, name, index);
}

aclError aclmdlGetDynamicBatch(const aclmdlDesc *modelDesc, aclmdlBatch *batch)
{
    return aclmdlGetDynamicBatchImpl(modelDesc, batch);
}

aclError aclmdlGetDynamicHW(const aclmdlDesc *modelDesc, size_t index, aclmdlHW *hw)
{
    return aclmdlGetDynamicHWImpl(modelDesc, index, hw);
}

aclError aclmdlGetInputDynamicGearCount(const aclmdlDesc *modelDesc, size_t index,
                                                            size_t *gearCount)
{
    return aclmdlGetInputDynamicGearCountImpl(modelDesc, index, gearCount);
}

aclError aclmdlGetInputDynamicDims(const aclmdlDesc *modelDesc, size_t index, aclmdlIODims *dims,
                                                       size_t gearCount)
{
    return aclmdlGetInputDynamicDimsImpl(modelDesc, index, dims, gearCount);
}

aclmdlAIPP *aclmdlCreateAIPP(uint64_t batchSize)
{
    return aclmdlCreateAIPPImpl(batchSize);
}

aclError aclmdlDestroyAIPP(const aclmdlAIPP *aippParmsSet)
{
    return aclmdlDestroyAIPPImpl(aippParmsSet);
}

aclError aclmdlGetAippDataSize(uint64_t batchSize, size_t *size)
{
    return aclmdlGetAippDataSizeImpl(batchSize, size);
}

aclError aclmdlSetAIPPInputFormat(aclmdlAIPP *aippParmsSet, aclAippInputFormat inputFormat)
{
    return aclmdlSetAIPPInputFormatImpl(aippParmsSet, inputFormat);
}

aclError aclmdlSetAIPPCscParams(aclmdlAIPP *aippParmsSet, int8_t cscSwitch,
                                                    int16_t cscMatrixR0C0, int16_t cscMatrixR0C1, int16_t cscMatrixR0C2,
                                                    int16_t cscMatrixR1C0, int16_t cscMatrixR1C1, int16_t cscMatrixR1C2,
                                                    int16_t cscMatrixR2C0, int16_t cscMatrixR2C1, int16_t cscMatrixR2C2,
                                                    uint8_t cscOutputBiasR0, uint8_t cscOutputBiasR1,
                                                    uint8_t cscOutputBiasR2, uint8_t cscInputBiasR0,
                                                    uint8_t cscInputBiasR1, uint8_t cscInputBiasR2)
{
    return aclmdlSetAIPPCscParamsImpl(aippParmsSet, cscSwitch, cscMatrixR0C0, cscMatrixR0C1, cscMatrixR0C2, cscMatrixR1C0, cscMatrixR1C1, cscMatrixR1C2, cscMatrixR2C0, cscMatrixR2C1, cscMatrixR2C2, cscOutputBiasR0, cscOutputBiasR1, cscOutputBiasR2, cscInputBiasR0, cscInputBiasR1, cscInputBiasR2);
}

aclError aclmdlSetAIPPRbuvSwapSwitch(aclmdlAIPP *aippParmsSet, int8_t rbuvSwapSwitch)
{
    return aclmdlSetAIPPRbuvSwapSwitchImpl(aippParmsSet, rbuvSwapSwitch);
}

aclError aclmdlSetAIPPAxSwapSwitch(aclmdlAIPP *aippParmsSet, int8_t axSwapSwitch)
{
    return aclmdlSetAIPPAxSwapSwitchImpl(aippParmsSet, axSwapSwitch);
}

aclError aclmdlSetAIPPSrcImageSize(aclmdlAIPP *aippParmsSet, int32_t srcImageSizeW,
                                                       int32_t srcImageSizeH)
{
    return aclmdlSetAIPPSrcImageSizeImpl(aippParmsSet, srcImageSizeW, srcImageSizeH);
}

aclError aclmdlSetAIPPScfParams(aclmdlAIPP *aippParmsSet,
                                                    int8_t scfSwitch,
                                                    int32_t scfInputSizeW,
                                                    int32_t scfInputSizeH,
                                                    int32_t scfOutputSizeW,
                                                    int32_t scfOutputSizeH,
                                                    uint64_t batchIndex)
{
    return aclmdlSetAIPPScfParamsImpl(aippParmsSet, scfSwitch, scfInputSizeW, scfInputSizeH, scfOutputSizeW, scfOutputSizeH, batchIndex);
}

aclError aclmdlSetAIPPCropParams(aclmdlAIPP *aippParmsSet,
                                                     int8_t cropSwitch,
                                                     int32_t cropStartPosW,
                                                     int32_t cropStartPosH,
                                                     int32_t cropSizeW,
                                                     int32_t cropSizeH,
                                                     uint64_t batchIndex)
{
    return aclmdlSetAIPPCropParamsImpl(aippParmsSet, cropSwitch, cropStartPosW, cropStartPosH, cropSizeW, cropSizeH, batchIndex);
}

aclError aclmdlSetAIPPPaddingParams(aclmdlAIPP *aippParmsSet, int8_t paddingSwitch,
                                                        int32_t paddingSizeTop, int32_t paddingSizeBottom,
                                                        int32_t paddingSizeLeft, int32_t paddingSizeRight,
                                                        uint64_t batchIndex)
{
    return aclmdlSetAIPPPaddingParamsImpl(aippParmsSet, paddingSwitch, paddingSizeTop, paddingSizeBottom, paddingSizeLeft, paddingSizeRight, batchIndex);
}

aclError aclmdlSetAIPPDtcPixelMean(aclmdlAIPP *aippParmsSet,
                                                       int16_t dtcPixelMeanChn0,
                                                       int16_t dtcPixelMeanChn1,
                                                       int16_t dtcPixelMeanChn2,
                                                       int16_t dtcPixelMeanChn3,
                                                       uint64_t batchIndex)
{
    return aclmdlSetAIPPDtcPixelMeanImpl(aippParmsSet, dtcPixelMeanChn0, dtcPixelMeanChn1, dtcPixelMeanChn2, dtcPixelMeanChn3, batchIndex);
}

aclError aclmdlSetAIPPDtcPixelMin(aclmdlAIPP *aippParmsSet,
                                                      float dtcPixelMinChn0,
                                                      float dtcPixelMinChn1,
                                                      float dtcPixelMinChn2,
                                                      float dtcPixelMinChn3,
                                                      uint64_t batchIndex)
{
    return aclmdlSetAIPPDtcPixelMinImpl(aippParmsSet, dtcPixelMinChn0, dtcPixelMinChn1, dtcPixelMinChn2, dtcPixelMinChn3, batchIndex);
}

aclError aclmdlSetAIPPPixelVarReci(aclmdlAIPP *aippParmsSet,
                                                       float dtcPixelVarReciChn0,
                                                       float dtcPixelVarReciChn1,
                                                       float dtcPixelVarReciChn2,
                                                       float dtcPixelVarReciChn3,
                                                       uint64_t batchIndex)
{
    return aclmdlSetAIPPPixelVarReciImpl(aippParmsSet, dtcPixelVarReciChn0, dtcPixelVarReciChn1, dtcPixelVarReciChn2, dtcPixelVarReciChn3, batchIndex);
}

aclError aclmdlSetInputAIPP(uint32_t modelId,
                                                aclmdlDataset *dataset,
                                                size_t index,
                                                const aclmdlAIPP *aippParmsSet)
{
    return aclmdlSetInputAIPPImpl(modelId, dataset, index, aippParmsSet);
}

aclError aclmdlSetAIPPByInputIndex(uint32_t modelId,
                                                       aclmdlDataset *dataset,
                                                       size_t index,
                                                       const aclmdlAIPP *aippParmsSet)
{
    return aclmdlSetAIPPByInputIndexImpl(modelId, dataset, index, aippParmsSet);
}

aclError aclmdlGetAippType(uint32_t modelId,
                                               size_t index,
                                               aclmdlInputAippType *type,
                                               size_t *dynamicAttachedDataIndex)
{
    return aclmdlGetAippTypeImpl(modelId, index, type, dynamicAttachedDataIndex);
}

aclError aclmdlGetFirstAippInfo(uint32_t modelId, size_t index, aclAippInfo *aippInfo)
{
    return aclmdlGetFirstAippInfoImpl(modelId, index, aippInfo);
}

aclError aclmdlCreateAndGetOpDesc(uint32_t deviceId, uint32_t streamId,
    uint32_t taskId, char *opName, size_t opNameLen, aclTensorDesc **inputDesc, size_t *numInputs,
    aclTensorDesc **outputDesc, size_t *numOutputs)
{
    return aclmdlCreateAndGetOpDescImpl(deviceId, streamId, taskId, opName, opNameLen, inputDesc, numInputs, outputDesc, numOutputs);
}

aclError aclmdlLoadWithConfig(const aclmdlConfigHandle *handle, uint32_t *modelId)
{
    return aclmdlLoadWithConfigImpl(handle, modelId);
}

aclError aclmdlSetExternalWeightAddress(aclmdlConfigHandle *handle, const char *weightFileName,
    void *devPtr, size_t size)
{
    return aclmdlSetExternalWeightAddressImpl(handle, weightFileName, devPtr, size);
}

aclmdlConfigHandle *aclmdlCreateConfigHandle()
{
    return aclmdlCreateConfigHandleImpl();
}

aclError aclmdlDestroyConfigHandle(aclmdlConfigHandle *handle)
{
    return aclmdlDestroyConfigHandleImpl(handle);
}

aclError aclmdlSetConfigOpt(aclmdlConfigHandle *handle, aclmdlConfigAttr attr,
    const void *attrValue, size_t valueSize)
{
    return aclmdlSetConfigOptImpl(handle, attr, attrValue, valueSize);
}

aclError aclmdlSetExecConfigOpt(aclmdlExecConfigHandle *handle, aclmdlExecConfigAttr attr,
                                                    const void *attrValue, size_t valueSize)
{
    return aclmdlSetExecConfigOptImpl(handle, attr, attrValue, valueSize);
}

const char *aclmdlGetTensorRealName(const aclmdlDesc *modelDesc, const char *name)
{
    return aclmdlGetTensorRealNameImpl(modelDesc, name);
}

aclError aclRecoverAllHcclTasks(int32_t deviceId)
{
    return aclRecoverAllHcclTasksImpl(deviceId);
}

aclTensorDesc *aclCreateTensorDesc(aclDataType dataType,
                                                       int numDims,
                                                       const int64_t *dims,
                                                       aclFormat format)
{
    return aclCreateTensorDescImpl(dataType, numDims, dims, format);
}

void aclDestroyTensorDesc(const aclTensorDesc *desc)
{
    return aclDestroyTensorDescImpl(desc);
}

aclError aclSetTensorShapeRange(aclTensorDesc* desc,
                                                    size_t dimsCount,
                                                    int64_t dimsRange[][ACL_TENSOR_SHAPE_RANGE_NUM])
{
    return aclSetTensorShapeRangeImpl(desc, dimsCount, dimsRange);
}

aclError aclSetTensorValueRange(aclTensorDesc* desc,
                                                    size_t valueCount,
                                                    int64_t valueRange[][ACL_TENSOR_VALUE_RANGE_NUM])
{
    return aclSetTensorValueRangeImpl(desc, valueCount, valueRange);
}

aclDataType aclGetTensorDescType(const aclTensorDesc *desc)
{
    return aclGetTensorDescTypeImpl(desc);
}

aclFormat aclGetTensorDescFormat(const aclTensorDesc *desc)
{
    return aclGetTensorDescFormatImpl(desc);
}

size_t aclGetTensorDescSize(const aclTensorDesc *desc)
{
    return aclGetTensorDescSizeImpl(desc);
}

size_t aclGetTensorDescElementCount(const aclTensorDesc *desc)
{
    return aclGetTensorDescElementCountImpl(desc);
}

size_t aclGetTensorDescNumDims(const aclTensorDesc *desc)
{
    return aclGetTensorDescNumDimsImpl(desc);
}

int64_t aclGetTensorDescDim(const aclTensorDesc *desc, size_t index)
{
    return aclGetTensorDescDimImpl(desc, index);
}

aclError aclGetTensorDescDimV2(const aclTensorDesc *desc, size_t index, int64_t *dimSize)
{
    return aclGetTensorDescDimV2Impl(desc, index, dimSize);
}

aclError aclGetTensorDescDimRange(const aclTensorDesc *desc,
                                                      size_t index,
                                                      size_t dimRangeNum,
                                                      int64_t *dimRange)
{
    return aclGetTensorDescDimRangeImpl(desc, index, dimRangeNum, dimRange);
}

void aclSetTensorDescName(aclTensorDesc *desc, const char *name)
{
    return aclSetTensorDescNameImpl(desc, name);
}

const char *aclGetTensorDescName(aclTensorDesc *desc)
{
    return aclGetTensorDescNameImpl(desc);
}

aclError aclTransTensorDescFormat(const aclTensorDesc *srcDesc, aclFormat dstFormat,
                                                      aclTensorDesc **dstDesc)
{
    return aclTransTensorDescFormatImpl(srcDesc, dstFormat, dstDesc);
}

aclError aclSetTensorStorageFormat(aclTensorDesc *desc, aclFormat format)
{
    return aclSetTensorStorageFormatImpl(desc, format);
}

aclError aclSetTensorStorageShape(aclTensorDesc *desc, int numDims, const int64_t *dims)
{
    return aclSetTensorStorageShapeImpl(desc, numDims, dims);
}

aclError aclSetTensorFormat(aclTensorDesc *desc, aclFormat format)
{
    return aclSetTensorFormatImpl(desc, format);
}

aclError aclSetTensorShape(aclTensorDesc *desc, int numDims, const int64_t *dims)
{
    return aclSetTensorShapeImpl(desc, numDims, dims);
}

aclError aclSetTensorOriginFormat(aclTensorDesc *desc, aclFormat format)
{
    return aclSetTensorOriginFormatImpl(desc, format);
}

aclError aclSetTensorOriginShape(aclTensorDesc *desc, int numDims, const int64_t *dims)
{
    return aclSetTensorOriginShapeImpl(desc, numDims, dims);
}

aclTensorDesc *aclGetTensorDescByIndex(aclTensorDesc *desc, size_t index)
{
    return aclGetTensorDescByIndexImpl(desc, index);
}

aclError aclSetTensorDynamicInput(aclTensorDesc *desc, const char *dynamicInputName)
{
    return aclSetTensorDynamicInputImpl(desc, dynamicInputName);
}

aclError aclSetTensorConst(aclTensorDesc *desc, void *dataBuffer, size_t length)
{
    return aclSetTensorConstImpl(desc, dataBuffer, length);
}

aclError aclSetTensorPlaceMent(aclTensorDesc *desc, aclMemType memType)
{
    return aclSetTensorPlaceMentImpl(desc, memType);
}

void *aclGetTensorDescAddress(const aclTensorDesc *desc)
{
    return aclGetTensorDescAddressImpl(desc);
}

#ifdef __cplusplus
}
#endif

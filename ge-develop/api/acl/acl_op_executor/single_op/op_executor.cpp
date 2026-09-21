/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_executor.h"
#include <graph/debug/ge_attr_define.h>
#include "framework/common/profiling_definitions.h"
#include "types/acl_op_inner.h"
#include "common/log_inner.h"
#include "model/acl_resource_manager.h"
#include "single_op/acl_op_resource_manager.h"
#include "ge_tensor_cache.h"
#include "single_op/executor/stream_executor.h"
#include "error_codes_inner.h"
#include "utils/string_utils.h"
#include "utils/array_utils.h"
#include "framework/runtime/gert_api.h"
#include "framework/runtime/subscriber/global_profiler.h"

namespace acl {
namespace {
constexpr int32_t MAX_CACHED_NUM = 128;
static void UpdateExecuteArgs(gert::ModelExecuteArg &arg, aclrtStream stream)
{
    arg.stream = stream;
    auto external_allocator = AclResourceManager::GetInstance().GetAllocators(arg.stream);
    arg.external_allocator = external_allocator.get();
    acl::AclOpResourceManager::GetInstance().UpdateAllocatorsForOp(stream, external_allocator);
}

static void ConstructRT2Tensor(const aclTensorDesc *desc,
                               const aclDataBuffer *buffer,
                               gert::Tensor &rt2Tensor)
{
    // construct gert OriginFormat and StorageFormat
    ge::Format geFormat = ge::FORMAT_RESERVED;
    if (desc->storageFormat != ACL_FORMAT_UNDEFINED) {
        geFormat = static_cast<ge::Format>(desc->storageFormat);
    }
    ge::Format geOriginFormat = ge::FORMAT_RESERVED;
    if (desc->format != ACL_FORMAT_UNDEFINED) {
        geOriginFormat = static_cast<ge::Format>(desc->format);
    }
    // construct gert OriginShape and OriginFormat
    for (const auto &dim : desc->dims) {
        (void) rt2Tensor.MutableOriginShape().AppendDim(dim);
    }
    rt2Tensor.SetOriginFormat(geOriginFormat);
    // construct gert StorageShape and StorageFormat
    // if storage shape and storage format are not defined by user, use user defined
    if ((geFormat != ge::FORMAT_RESERVED) && (geOriginFormat != geFormat)) {
        for (const auto &storageDim : desc->storageDims) {
            (void) rt2Tensor.MutableStorageShape().AppendDim(storageDim);
        }
        rt2Tensor.SetStorageFormat(geFormat);
    } else { // if storage shape and storage format are not defined by user, use origin as storage
        rt2Tensor.MutableStorageShape() = rt2Tensor.GetOriginShape();
        rt2Tensor.SetStorageFormat(geOriginFormat);
    }

    // construct gert placement
    const gert::TensorPlacement placement = desc->memtype == ACL_MEMTYPE_DEVICE ?
                                            gert::TensorPlacement::kOnDeviceHbm : gert::TensorPlacement::kOnHost;

    // construct gert dataType
    ge::DataType dataType = ge::DT_UNDEFINED;
    if (desc->dataType != ACL_DT_UNDEFINED) {
        dataType = static_cast<::ge::DataType>(desc->dataType);
    }

    (void)rt2Tensor.MutableTensorData().SetAddr(buffer->data, nullptr);
    rt2Tensor.MutableTensorData().SetSize(buffer->length);
    rt2Tensor.SetDataType(dataType);
    rt2Tensor.MutableTensorData().SetPlacement(placement);
}

static void ConstructInputAndOutput(const aclTensorDesc* const *desc, const aclDataBuffer* const *buffer,
    const int32_t num, const size_t filterdNum, const bool executeWithExactModel,
    ge::SmallVector<gert::Tensor, static_cast<size_t>(ge::kDefaultMaxInputNum)> &tensor,
    ge::SmallVector<gert::Tensor*, static_cast<size_t>(ge::kDefaultMaxInputNum)> &rt2Tensor)
{
    size_t count = 0U;
    for (int32_t i = 0; i < num; ++i) {
        if (count >= filterdNum) {
            break;
        }
        // skip optional input
        if (desc[i]->IsOptinalTensor()) {
            continue;
        }
        // skip const input
        if (desc[i]->CheckConstTensor(executeWithExactModel)) {
            continue;
        }
        ConstructRT2Tensor(desc[i], buffer[i], tensor[static_cast<size_t>(i)]);
        rt2Tensor[count] = &(tensor[static_cast<size_t>(i)]);
        ++count;
    }
}

static void RefreshOutput(const AclOp &aclOp, const size_t filterOutNum,
    const ge::SmallVector<gert::Tensor*, static_cast<size_t>(ge::kDefaultMaxOutputNum)> &outputTensorPtr,
    const bool executeWithExactModel)
{
    if (aclOp.exeucteType == ACL_OP_EXECUTE_V2) {
        ACL_LOG_DEBUG("exeucteType is ACL_OP_EXECUTE_V2");
        size_t outCnt = 0U;
        for (int32_t i = 0; i < aclOp.numOutputs; ++i) {
            if (outCnt >= filterOutNum) {
                break;
            }
            if (aclOp.outputDesc[i]->CheckConstTensor(executeWithExactModel)) {
                continue;
            }
            const gert::Shape outputShape = outputTensorPtr[outCnt]->GetStorageShape();
            const_cast<aclTensorDesc *>(aclOp.outputDesc[i])->dims.resize(outputShape.GetDimNum());
            for (size_t j = 0U; j < outputShape.GetDimNum(); ++j) {
                ACL_LOG_INFO("update outputDesc[%d], dims[%zu] = %zu", i, j, outputShape.GetDim(j));
                const_cast<aclTensorDesc *>(aclOp.outputDesc[i])->dims[j] = outputShape.GetDim(j);
            }
            ++outCnt;
        }
    } else if (aclOp.exeucteType == ACL_OP_EXECUTE_REFRESH_OUTPUT_ORI_SHAPE) {
        ACL_LOG_DEBUG("exeucteType is ACL_OP_EXECUTE_REFRESH_OUTPUT_ORI_SHAPE");
        size_t outCnt = 0U;
        for (int32_t i = 0; i < aclOp.numOutputs; ++i) {
            if (outCnt >= filterOutNum) {
                break;
            }
            if (aclOp.outputDesc[i]->CheckConstTensor(executeWithExactModel)) {
                continue;
            }
            const gert::Shape outputShape = outputTensorPtr[outCnt]->GetOriginShape();
            const_cast<aclTensorDesc *>(aclOp.outputDesc[i])->dims.resize(outputShape.GetDimNum());
            for (size_t j = 0U; j < outputShape.GetDimNum(); ++j) {
                ACL_LOG_INFO("update outputDesc[%d], dims[%zu] = %zu", i, j, outputShape.GetDim(j));
                const_cast<aclTensorDesc *>(aclOp.outputDesc[i])->dims[j] = outputShape.GetDim(j);
            }
            ++outCnt;
        }
    } else {
        ACL_LOG_DEBUG("exeucteType is neither ACL_OP_EXECUTE_V2 nor ACL_OP_EXECUTE_REFRESH_OUTPUT_ORI_SHAPE,"
                      "No need to refresh output!");
    }
}
} // namespace

aclError OpExecutor::DoExecuteAsync(ge::SingleOp *const singleOp,
                                    const AclOp &aclOp,
                                    const aclDataBuffer *const inputs[],
                                    const aclDataBuffer *const outputs[],
                                    const bool executeWithExactModel)
{
    std::vector<ge::DataBuffer> inputVec;
    std::vector<ge::DataBuffer> outputVec;

    for (int32_t i = 0; i < aclOp.numInputs; ++i) {
        // skip optional input
        if (aclOp.inputDesc[i]->IsOptinalTensor()) {
            continue;
        }
        if (aclOp.inputDesc[i]->CheckConstTensor(executeWithExactModel)) {
            continue;
        }
        ge::DataBuffer buffer;
        buffer.data = inputs[i]->data;
        buffer.length = inputs[i]->length;
        buffer.placement = ((aclOp.inputDesc[i]->memtype == ACL_MEMTYPE_DEVICE) ?
            static_cast<uint32_t>(aclOp.inputDesc[i]->memtype) : static_cast<uint32_t>(ACL_MEMTYPE_HOST));
        inputVec.emplace_back(buffer);
    }

    for (int32_t i = 0; i < aclOp.numOutputs; ++i) {
        if (aclOp.outputDesc[i]->CheckConstTensor(executeWithExactModel)) {
            continue;
        }
        ge::DataBuffer buffer;
        buffer.data = outputs[i]->data;
        buffer.length = outputs[i]->length;
        buffer.placement = ((aclOp.outputDesc[i]->memtype == ACL_MEMTYPE_DEVICE) ?
            static_cast<uint32_t>(aclOp.outputDesc[i]->memtype) : static_cast<uint32_t>(ACL_MEMTYPE_HOST));
        outputVec.emplace_back(buffer);
    }

    ACL_LOG_INFO("To invoke GeExecutor::ExecuteAsync");
    const ge::Status ret = ge::GeExecutor::ExecuteAsync(singleOp, inputVec, outputVec);
    if (ret != ge::SUCCESS) {
        ACL_LOG_CALL_ERROR("[Exec][Op]Execute op failed. op type = %s, ge result = %u", aclOp.opType.c_str(), ret);
        return ACL_GET_ERRCODE_GE(static_cast<int32_t>(ret));
    }

    return ACL_SUCCESS;
}

static void GetInputAndOutputNum(const AclOp &aclOp, const bool executeWithExactModel, size_t &inputNum,
    size_t &outputNum)
{
    inputNum = 0UL;
    for (int32_t i = 0; i < aclOp.numInputs; ++i) {
        // skip optional input
        if (aclOp.inputDesc[i]->IsOptinalTensor()) {
            continue;
        }
        if (aclOp.inputDesc[i]->CheckConstTensor(executeWithExactModel)) {
            continue;
        }
        ++inputNum;
    }

    outputNum = 0UL;
    for (int32_t i = 0; i < aclOp.numOutputs; ++i) {
        if (aclOp.outputDesc[i]->CheckConstTensor(executeWithExactModel)) {
            continue;
        }
        ++outputNum;
    }
}

static void FixGeTensorDesc(const aclTensorDesc &aclDesc, ge::GeTensorDesc &geTensorDesc)
{
    ge::Format geFormat = ge::FORMAT_RESERVED;
    if (aclDesc.storageFormat != ACL_FORMAT_UNDEFINED) {
        geFormat = static_cast<::ge::Format>(aclDesc.storageFormat);
    }
    ge::DataType geDataType = ge::DT_UNDEFINED;
    if (aclDesc.dataType != ACL_DT_UNDEFINED) {
        geDataType = static_cast<::ge::DataType>(aclDesc.dataType);
    }
    ge::Format geOriginFormat = ge::FORMAT_RESERVED;
    if (aclDesc.format != ACL_FORMAT_UNDEFINED) {
        geOriginFormat = static_cast<::ge::Format>(aclDesc.format);
    }
    if ((geFormat != ge::FORMAT_RESERVED) && (geOriginFormat != geFormat)) {
        WrapGeShape(geTensorDesc.MutableShape(), aclDesc.storageDims);
        geTensorDesc.SetFormat(geFormat);
        const ge::GeShape &shape = geTensorDesc.GetOriginShape();
        WrapGeShape(const_cast<ge::GeShape &>(shape), aclDesc.dims);
        geTensorDesc.SetOriginShape(shape);
    } else {
        WrapGeShape(geTensorDesc.MutableShape(), aclDesc.dims);
        geTensorDesc.SetFormat(geOriginFormat);
        geTensorDesc.SetOriginShape(geTensorDesc.GetShape());
    }
    geTensorDesc.SetOriginFormat(geOriginFormat);
    geTensorDesc.SetDataType(geDataType);
}

aclError OpExecutor::DoExecuteAsync(ge::DynamicSingleOp *const singleOp,
                                    const AclOp &aclOp,
                                    const aclDataBuffer *const inputs[],
                                    const aclDataBuffer *const outputs[],
                                    const bool executeWithExactModel)
{
    size_t inputNum = 0U;
    size_t outputNum = 0U;
    GetInputAndOutputNum(aclOp, executeWithExactModel, inputNum, outputNum);
    const GeTensorDescVecPtr inputDesc = GeTensorDescCache::GetInstance().GetDescVecPtr(inputNum);
    ACL_CHECK_WITH_MESSAGE_AND_RETURN(inputDesc != nullptr, ACL_ERROR_BAD_ALLOC, "get input tensor desc failed");
    const GeTensorDescVecPtr outputDesc = GeTensorDescCache::GetInstance().GetDescVecPtr(outputNum);
    if (outputDesc == nullptr) {
        GeTensorDescCache::GetInstance().ReleaseDescVecPtr(inputDesc);
        ACL_LOG_ERROR("get input tensor desc failed");
        return ACL_ERROR_BAD_ALLOC;
    }
    std::vector<ge::DataBuffer> inputVec(inputNum);
    std::vector<ge::DataBuffer> outputVec(outputNum);

    size_t inCnt = 0U;
    for (int32_t i = 0; i < aclOp.numInputs; ++i) {
        if (inCnt >= inputNum) {
            break;
        }
        // skip optional input
        if (aclOp.inputDesc[i]->IsOptinalTensor()) {
            continue;
        }
        if (aclOp.inputDesc[i]->CheckConstTensor(executeWithExactModel)) {
            continue;
        }
        auto &vecDesc = *inputDesc;
        const aclTensorDesc * const aclDesc = aclOp.inputDesc[i];
        FixGeTensorDesc(*aclDesc, vecDesc[inCnt]);
        FixGeDataBuffer(inputs[i], aclDesc->memtype, inputVec[inCnt]);
        ++inCnt;
    }

    size_t outCnt = 0U;
    for (int32_t i = 0; i < aclOp.numOutputs; ++i) {
        if (outCnt >= outputNum) {
            break;
        }
        if (aclOp.outputDesc[i]->CheckConstTensor(executeWithExactModel)) {
            continue;
        }
        auto &vecDesc = *outputDesc;
        const aclTensorDesc *const aclDesc = aclOp.outputDesc[i];
        FixGeTensorDesc(*aclDesc, vecDesc[outCnt]);
        FixGeDataBuffer(outputs[i], aclDesc->memtype, outputVec[outCnt]);
        ++outCnt;
    }

    ACL_LOG_INFO("to invoke GeExecutor::ExecuteAsync");
    const ge::Status ret = ge::GeExecutor::ExecuteAsync(singleOp, *inputDesc, inputVec, *outputDesc, outputVec);
    if (ret != ge::SUCCESS) {
        ACL_LOG_CALL_ERROR("[Exec][Op]Execute op failed. op type = %s, ge result = %u", aclOp.opType.c_str(), ret);
        GeTensorDescCache::GetInstance().ReleaseDescVecPtr(inputDesc);
        GeTensorDescCache::GetInstance().ReleaseDescVecPtr(outputDesc);
        return ACL_GET_ERRCODE_GE(static_cast<int32_t>(ret));
    }

    if (aclOp.exeucteType == ACL_OP_EXECUTE_V2) {
        for (size_t i = 0U; i < outputDesc->size(); ++i) {
            const ge::GeShape outputShape = (*outputDesc)[i].GetShape();
            const std::vector<int64_t> outputDims = outputShape.GetDims();
            ACL_LOG_INFO("update outputDesc[%zu] dims is [%s]", i, StringUtils::VectorToString(outputDims).c_str());
            ConvertVecToSvec(outputDims, const_cast<aclTensorDesc *>(aclOp.outputDesc[i])->dims);
        }
        ACL_LOG_INFO("update outputDesc successfully");
    }

    if (aclOp.exeucteType == ACL_OP_EXECUTE_REFRESH_OUTPUT_ORI_SHAPE) {
        ACL_LOG_INFO("aclOp exeucte type is ACL_OP_EXECUTE_REFRESH_OUTPUT_ORI_SHAPE,"
                    "refresh origin shape of output");
        for (size_t i = 0U; i < outputDesc->size(); ++i) {
            const std::vector<int64_t> outputOriShape = (*outputDesc)[i].GetOriginShape().GetDims();
            ACL_LOG_INFO("update outputDesc[%zu] dims is [%s]", i,
                StringUtils::VectorToString(outputOriShape).c_str());
            ConvertVecToSvec(outputOriShape, const_cast<aclTensorDesc *>(aclOp.outputDesc[i])->dims);
        }
        ACL_LOG_INFO("update outputDesc origin shape successfully");
    }

    GeTensorDescCache::GetInstance().ReleaseDescVecPtr(inputDesc);
    GeTensorDescCache::GetInstance().ReleaseDescVecPtr(outputDesc);
    return ACL_SUCCESS;
}

aclError OpExecutor::DoExecuteAsync(std::shared_ptr<gert::StreamExecutor> &streamExecutor,
                                    const AclOp &aclOp,
                                    const aclrtStream stream,
                                    const bool executeWithExactModel)
{
    size_t filterInNum = 0U;
    size_t filterOutNum = 0U;
    GetInputAndOutputNum(aclOp, executeWithExactModel, filterInNum, filterOutNum);
    ge::SmallVector<gert::Tensor*, static_cast<size_t>(ge::kDefaultMaxInputNum)> inputTensorPtr(filterInNum);
    ge::SmallVector<gert::Tensor*, static_cast<size_t>(ge::kDefaultMaxOutputNum)> outputTensorPtr(filterOutNum);
    ge::SmallVector<gert::Tensor, static_cast<size_t>(ge::kDefaultMaxInputNum)> inputTensor(aclOp.numInputs);
    ge::SmallVector<gert::Tensor, static_cast<size_t>(ge::kDefaultMaxOutputNum)> outputTensor(aclOp.numOutputs);

    ConstructInputAndOutput(aclOp.inputDesc, aclOp.inputs, aclOp.numInputs, filterInNum,
                            executeWithExactModel, inputTensor, inputTensorPtr);
    ConstructInputAndOutput(aclOp.outputDesc, aclOp.outputs, aclOp.numOutputs, filterOutNum,
                            executeWithExactModel, outputTensor, outputTensorPtr);
    gert::ModelExecuteArg arg;
    UpdateExecuteArgs(arg, stream);
    auto currentStream = acl::AclOpResourceManager::GetInstance().GetKeyByStreamOrDefaultStream(stream);
    gert::ModelV2Executor *executor = nullptr;
    const auto aclRet =
            acl::AclOpResourceManager::GetInstance().CreateRT2Executor(streamExecutor, currentStream, arg, executor);
    ACL_REQUIRES_OK_WITH_INNER_MESSAGE(aclRet, "CreateRT2Executor failed, errorCode = %d.", aclRet);
    const ge::graphStatus ret = executor->Execute(arg, inputTensorPtr.data(), filterInNum, outputTensorPtr.data(),
                                                  filterOutNum);
    if (ret != ge::SUCCESS) {
        ACL_LOG_CALL_ERROR("[Exec][Op]Execute op failed. op type = %s, ge result = %u", aclOp.opType.c_str(), ret);
        return ACL_GET_ERRCODE_GE(static_cast<int32_t>(ret));
    }

    if ((aclOp.exeucteType == ACL_OP_EXECUTE_REFRESH_OUTPUT_ORI_SHAPE) || (aclOp.exeucteType == ACL_OP_EXECUTE_V2)) {
        RefreshOutput(aclOp, filterOutNum, outputTensorPtr, executeWithExactModel);
    }
    return ACL_SUCCESS;
}

aclError OpExecutor::LoadSingleOp(const OpModel &modelInfo, const aclrtStream stream, ge::SingleOp **const singleOp)
{
    RT2_PROFILING_SCOPE_CONST(gert::profiling::kUnknownName, gert::profiling::kAclLoadSingleOp);
    const auto allocators = AclResourceManager::GetInstance().GetAllocators(stream);
    ACL_REQUIRES_NOT_NULL(allocators);
    const auto allocator = allocators->GetAllocator(
        gert::TensorPlacement::kOnDeviceHbm, static_cast<size_t>(gert::AllocatorUsage::kAllocNodeWorkspace));
    const auto ret = ge::GeExecutor::SetAllocator(stream, allocator);
    if (ret != ge::SUCCESS) {
        ACL_LOG_CALL_ERROR("[Load][Op]Set allocator failed, ge result = %u", ret);
        return ACL_GET_ERRCODE_GE(ret);
    }
    ge::ModelData modelData;
    modelData.model_data = modelInfo.data.get();
    modelData.model_len = modelInfo.size;
    const auto status = ge::GeExecutor::LoadSingleOpV2(modelInfo.name, modelData,
        stream, singleOp, modelInfo.opModelId);
    if (status != ge::SUCCESS) {
        ACL_LOG_CALL_ERROR("[Load][Op]Load operator failed. model = %s, ge result = %u",
            modelInfo.name.c_str(), status);
        return ACL_GET_ERRCODE_GE(status);
    }

    return ACL_SUCCESS;
}

aclError OpExecutor::LoadDynamicSingleOp(const OpModel &modelInfo, const aclrtStream stream,
                                         ge::DynamicSingleOp **const dynamicSingleOp)
{
    RT2_PROFILING_SCOPE(gert::profiling::kUnknownName, gert::profiling::kAclLoadSingleOp);
    ge::ModelData modelData;
    modelData.model_data = modelInfo.data.get();
    modelData.model_len = modelInfo.size;

    const auto status = ge::GeExecutor::LoadDynamicSingleOpV2(modelInfo.name, modelData, stream, dynamicSingleOp,
                                                              modelInfo.opModelId);
    if (status != ge::SUCCESS) {
        ACL_LOG_CALL_ERROR("[Load][Op]Load dynamic operator failed. model = %s, ge result = %u",
            modelInfo.name.c_str(), status);
        return ACL_GET_ERRCODE_GE(status);
    }

    return ACL_SUCCESS;
}

aclError OpExecutor::ExecuteAsync(const AclOp &aclOp,
                                  const aclDataBuffer *const inputs[],
                                  aclDataBuffer *const outputs[],
                                  const aclrtStream stream)
{
    RT2_PROFILING_SCOPE(gert::profiling::kUnknownName, gert::profiling::kAclExecuteAsync);
    if (OpKernelSelector::GetInstance().HasSelectFunc(aclOp.opType)) {
        ACL_LOG_INFO("Dynamic kernel selector is registered for %s", aclOp.opType.c_str());
        aclrtContext context = nullptr;
        ACL_REQUIRES_OK(aclrtGetCurrentContext(&context));
        auto *const streamExecutor = Executors::GetOrCreate(context, stream);
        ACL_CHECK_MALLOC_RESULT(streamExecutor);
        const auto ret = streamExecutor->ExecuteAsync(aclOp, inputs, outputs);
        if (ret == ACL_SUCCESS) {
            return ACL_SUCCESS;
        }
    }
    ACL_LOG_INFO("OpExecutor::ExecuteAsync aclOp = %s", aclOp.DebugString().c_str());
    OpModel opModel;
    OpModel *opModelPtr = &opModel;
    bool isDynamic = false;

    if (!aclOp.isMatched) {
        ACL_REQUIRES_OK(AclOpResourceManager::GetInstance().MatchOpModel(aclOp, *opModelPtr, isDynamic));
    } else {
        opModelPtr = &((const_cast<AclOp *>(&aclOp))->opModel);
        isDynamic = aclOp.isDynamic;
    }
    const bool isExactModel = (opModelPtr->isStaticModelWithFuzzCompile == 0U) ? true : false;
    if (acl::AclOpResourceManager::GetInstance().IsRuntimeV2Enable(false)) {
         // 临时方案
        // 单算子模型为静态模型，走rt1
        if (!isDynamic) {
            ACL_LOG_DEBUG("optype is %s, is static model, use RT1 default", aclOp.opType.c_str());
            return DoExecuteRT1(aclOp, inputs, outputs, stream, opModelPtr, isDynamic, isExactModel);
        }
        ACL_REQUIRES_OK(PrepareRt2Execute(aclOp, opModelPtr));
        return DoExecuteAsync(opModelPtr->executor, aclOp, stream, isExactModel);
    }
    return DoExecuteRT1(aclOp, inputs, outputs, stream, opModelPtr, isDynamic, isExactModel);
}

aclError OpExecutor::PrepareRt2Execute(const AclOp &aclOp, OpModel *opModelPtr)
{
    // 首次需加载
    if (opModelPtr->executor == nullptr) {
        const auto tmpMu = AclOpResourceManager::GetInstance().GetCacheMutex(opModelPtr->opModelId);
        ACL_REQUIRES_NOT_NULL(tmpMu);
        const std::lock_guard<std::mutex> lk(*tmpMu);
        // 在锁里再次判断下，防止多线程并发
        if (AclOpResourceManager::GetInstance().GetRT2Executor(opModelPtr->opModelId) == nullptr) {
            ACL_LOG_DEBUG("RTV2: executor is nullptr, start load");
            ge::graphStatus ret = ge::GRAPH_FAILED;
            ge::ModelData modelData;
            modelData.model_data = opModelPtr->data.get();
            modelData.model_len = opModelPtr->size;

            gert::LoweringOption oOption;
            // 第三类算子会使用ACL_OP_EXECUTE_REFRESH_OUTPUT_ORI_SHAPE，调用者传入的输出shape并不是准确的
            oOption.trust_shape_on_out_tensor = (aclOp.exeucteType != ACL_OP_EXECUTE_REFRESH_OUTPUT_ORI_SHAPE);
            oOption.always_zero_copy = true;
            oOption.always_external_allocator = true;
            auto streamExecutor = gert::LoadStreamExecutorFromModelData(modelData, oOption, ret).release();
            ACL_REQUIRES_NOT_NULL(streamExecutor);
            const std::shared_ptr<gert::StreamExecutor> exe(streamExecutor);
            if (ret != ge::GRAPH_SUCCESS) {
                ACL_LOG_CALL_ERROR("[Exec][Op]LoadExecutorFromModelData fail. ge result = %u", ret);
                return ACL_GET_ERRCODE_GE(static_cast<int32_t>(ret));
            }
            (void)AclOpResourceManager::GetInstance().UpdateRT2Executor(opModelPtr->opModelId, exe);
            (void)AclOpResourceManager::GetInstance().UnloadModelData(opModelPtr->opModelId);
        }
        // 此处更新，这样下面流程归一调用同一个接口
        opModelPtr->executor = AclOpResourceManager::GetInstance().GetRT2Executor(opModelPtr->opModelId);
    }
    return ACL_SUCCESS;
}

aclError OpExecutor::DoExecuteRT1(const AclOp &aclOp,
                                  const aclDataBuffer *const inputs[],
                                  aclDataBuffer *const outputs[],
                                  const aclrtStream stream,
                                  const OpModel *opModelPtr,
                                  const bool isDynamic,
                                  const bool isExactModel)
{
    aclError ret;
    if (isDynamic) {
        ge::DynamicSingleOp *dynamicSingleOp = nullptr;
        ret = LoadDynamicSingleOp(*opModelPtr, stream, &dynamicSingleOp);
        if ((ret != ACL_SUCCESS) || (dynamicSingleOp == nullptr)) {
            ACL_LOG_INNER_ERROR("[Load][Op]LoadDynamicSingleOp failed or dynamicSingleOp is nullptr"
                                "ret = %d", ret);
            return ret;
        }
        ret = DoExecuteAsync(dynamicSingleOp, aclOp, inputs, outputs, isExactModel);
    } else {
        ge::SingleOp *singleOp = nullptr;
        ret = LoadSingleOp(*opModelPtr, stream, &singleOp);
        if ((ret != ACL_SUCCESS) || (singleOp == nullptr)) {
            ACL_LOG_INNER_ERROR("[Load][Op]LoadSingleOp failed or singleOp is nullptr"
                                "ret = %d", ret);
            return ret;
        }
        ret = DoExecuteAsync(singleOp, aclOp, inputs, outputs, isExactModel);
    }

    return ret;
}

aclError OpExecutor::ExecuteAsync(OpHandle &opHandle,
                                  const aclDataBuffer *const inputs[],
                                  aclDataBuffer *const outputs[],
                                  const aclrtStream stream)
{
    if (opHandle.kernelDesc != nullptr) {
        ACL_LOG_INFO("Get keneldesc is not null");
        aclrtContext context = nullptr;
        ACL_REQUIRES_OK(aclrtGetCurrentContext(&context));
        auto *const streamExecutor = Executors::GetOrCreate(context, stream);
        ACL_CHECK_MALLOC_RESULT(streamExecutor);
        return streamExecutor->ExecuteAsync(*opHandle.kernelDesc,
                                            opHandle.numInputs,
                                            inputs,
                                            opHandle.numOutputs,
                                            outputs);
    }
    aclError ret;
    if (opHandle.isDynamic) {
        ACL_LOG_INFO("begin to load dynamic model. model = %s, stream = %p", opHandle.opModel.name.c_str(), stream);
        auto &cachedExecutors = opHandle.cachedDynamicOperators;
        ge::DynamicSingleOp *dynamicSingleOp = nullptr;
        {
            const std::lock_guard<std::mutex> lk(opHandle.mutexForDynamic);
            const std::map<aclrtStream, ge::DynamicSingleOp *>::const_iterator it = cachedExecutors.find(stream);
            if (it != cachedExecutors.cend()) {
                dynamicSingleOp = it->second;
            } else {
                ret = LoadDynamicSingleOp(opHandle.opModel, stream, &dynamicSingleOp);
                if ((ret != ACL_SUCCESS) || (dynamicSingleOp == nullptr)) {
                    ACL_LOG_INNER_ERROR("[Load][Op]LoadDynamicSingleOp failed or dynamicSingleOp is nullptr,"
                                        "ret = %d", ret);
                    return ret;
                }
                // Just for protection. Preventing from cache grows to large.
                if (cachedExecutors.size() > static_cast<size_t>(MAX_CACHED_NUM)) {
                    std::map<aclrtStream, ge::DynamicSingleOp *>::const_iterator toErase = cachedExecutors.cbegin();
                    ACL_LOG_WARN("cache[%zu] reaches max size[%d], evict one object. stream = %p",
                        cachedExecutors.size(), MAX_CACHED_NUM, toErase->first);
                    (void)cachedExecutors.erase(toErase);
                }

                ACL_LOG_INFO("cache operator executor. model = %s, stream = %p", opHandle.opModel.name.c_str(), stream);
                (void)opHandle.cachedDynamicOperators.emplace(stream, dynamicSingleOp);
            }
        }
        ret = DoExecuteAsync(dynamicSingleOp, opHandle.aclOp, inputs, outputs);
    } else {
        ACL_LOG_INFO("begin to load static model. model = %s, stream = %p", opHandle.opModel.name.c_str(), stream);
        auto &cachedExecutors = opHandle.cachedOperators;
        ge::SingleOp *singleOp = nullptr;
        {
            const std::lock_guard<std::mutex> lk(opHandle.mutexForStatic);
            const auto it = cachedExecutors.find(stream);
            if (it != cachedExecutors.end()) {
                singleOp = it->second;
            } else {
                ret = LoadSingleOp(opHandle.opModel, stream, &singleOp);
                if ((ret != ACL_SUCCESS) || (singleOp == nullptr)) {
                    ACL_LOG_INNER_ERROR("[Load][Op]LoadSingleOp failed or singleOp is nullptr"
                                        "ret = %d", ret);
                    return ret;
                }

                // Just for protection. Preventing from cache grows to large.
                if (cachedExecutors.size() > static_cast<size_t>(MAX_CACHED_NUM)) {
                    const auto toErase = cachedExecutors.begin();
                    ACL_LOG_INFO("cache[%zu] reaches max size[%d], evict one object. stream = %p",
                        cachedExecutors.size(), MAX_CACHED_NUM, toErase->first);
                    (void)cachedExecutors.erase(toErase);
                }

                ACL_LOG_INFO("cache operator executor. model = %s, stream = %p", opHandle.opModel.name.c_str(), stream);
                (void)opHandle.cachedOperators.emplace(stream, singleOp);
            }
        }
        ret = DoExecuteAsync(singleOp, opHandle.aclOp, inputs, outputs);
    }

    return ret;
}

aclError OpExecutor::CreateOpHandle(const AclOp &aclOp, OpHandle **const handle)
{
    std::unique_ptr<OpHandle> handlePtr = std::unique_ptr<OpHandle>(new(std::nothrow) OpHandle);
    ACL_CHECK_MALLOC_RESULT(handlePtr);
    handlePtr->numInputs = aclOp.numInputs;
    handlePtr->numOutputs = aclOp.numOutputs;
    handlePtr->opType = aclOp.opType;
    handlePtr->aclOp = aclOp;

    // try using dynamic shape
    if (OpKernelSelector::GetInstance().HasSelectFunc(aclOp.opType)) {
        ACL_LOG_INFO("Dynamic kernel selector is registered for %s", aclOp.opType.c_str());
        if (OpKernelSelector::GetInstance().GetOpKernelDesc(aclOp, handlePtr->kernelDesc) == ACL_SUCCESS) {
            ACL_LOG_INFO("Got compile kernel desc for handle. opType = %s", aclOp.opType.c_str());
        }
    }
    bool isDynamic = false;

    if (handlePtr->kernelDesc == nullptr) {
        ACL_REQUIRES_OK(AclOpResourceManager::GetInstance().MatchOpModel(aclOp, handlePtr->opModel, isDynamic));
        ACL_LOG_INFO("Match opModel success, opType = %s, isDynamic = %d", aclOp.opType.c_str(),
            static_cast<int32_t>(isDynamic));
    }
    handlePtr->isDynamic = isDynamic;
    *handle = handlePtr.release();
    return ACL_SUCCESS;
}
}

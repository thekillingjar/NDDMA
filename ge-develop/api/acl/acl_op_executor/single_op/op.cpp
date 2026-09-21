/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "acl/acl_op.h"

#include <mutex>
#include <fstream>

#include "common/log_inner.h"
#include "types/acl_op_inner.h"
#include "op_executor.h"
#include "utils/array_utils.h"
#include "utils/string_utils.h"
#include "single_op/acl_op_resource_manager.h"
#include "single_op/compile/op_kernel_registry.h"
#include "single_op/compile/op_kernel_selector.h"
#include "error_codes_inner.h"
#include "common/prof_api_reg.h"
#include "graph/operator.h"
#include "graph/tensor.h"
#include "graph/operator_factory.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/opsproto_manager.h"
#include "graph/utils/tensor_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/common/util.h"
#include "acl_op_executor_impl.h"

namespace {
    static bool g_aclInitFlag = false;
    static std::mutex g_aclInitMutex;
    bool aclLoadOpsProtoFlag = false;
    std::mutex aclLoadOpsProtoMutex;
}

struct aclopHandle {
    aclopHandle() : opHandle(nullptr) {}
    ~aclopHandle()
    {
        ACL_DELETE_AND_SET_NULL(opHandle);
    }

    acl::OpHandle *opHandle;
};

aclError aclopSetModelDirImpl(const char *modelDir)
{
    ACL_LOG_INFO("start to execute aclopSetModelDir");
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(modelDir);
    char_t trustedPath[MMPA_MAX_PATH] = {};
    const int32_t checkPathRet = mmRealPath(modelDir, trustedPath, static_cast<int32_t>(sizeof(trustedPath)));
    if (checkPathRet != EN_OK) {
        const auto formatErrMsg = acl::AclGetErrorFormatMessage(mmGetErrorCode());
        ACL_LOG_ERROR("[Check][modelDir]path %s is invalid, errMessage is %s.", modelDir, formatErrMsg.c_str());
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_PATH_MSG,
            std::vector<const char *>({"path", "reason"}),
            std::vector<const char *>({modelDir,
            (std::string("cannot convert to realpath: ") + formatErrMsg).c_str()}));
        return ACL_ERROR_READ_MODEL_FAILURE;
    }

    const std::unique_lock<std::mutex> lk(g_aclInitMutex);
    if (g_aclInitFlag) {
        ACL_LOG_INNER_ERROR("[Check][InitFlag]repeatedly set model dir.");
        return ACL_ERROR_REPEAT_INITIALIZE;
    }

    const aclError ret = acl::AclOpResourceManager::GetInstance().LoadAllModels(trustedPath);
    if (ret == ACL_SUCCESS) {
        g_aclInitFlag = true;
    }
    return ret;
}

aclError aclopLoadImpl(const void *model, size_t modelSize)
{
    ACL_PROFILING_REG(acl::AclProfType::AclopLoad);

    ACL_LOG_INFO("start to execute aclopLoad");
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(model);
    if (modelSize == 0UL) {
        ACL_LOG_ERROR("[Check][ModelSize]the value of modelSize[%zu] can't be zero", modelSize);
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
            std::vector<const char *>({"param", "value", "reason"}),
            std::vector<const char *>({"modelSize", std::to_string(modelSize).c_str(), "can't be zero"}));
        return ACL_ERROR_INVALID_PARAM;
    }
    // it is static load, true means no need aging
    const aclError ret = acl::AclOpResourceManager::GetInstance().LoadModelFromMem(model, modelSize, true);
    if (ret == ACL_SUCCESS) {
        ACL_LOG_INFO("load opModels from memory successfully");
        return ret;
    }
    ACL_LOG_INNER_ERROR("[Load][opModels]fail to load opModels from memory, result = %d", ret);
    return ret;
}

aclError aclopCreateHandleImpl(const char *opType,
                           int numInputs,
                           const aclTensorDesc *const inputDesc[],
                           int numOutputs,
                           const aclTensorDesc *const outputDesc[],
                           const aclopAttr *opAttr,
                           aclopHandle **handle)
{
    ACL_PROFILING_REG(acl::AclProfType::AclopCreateHandle);
    ACL_LOG_INFO("start to execute aclopCreateHandle");
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(opType);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(handle);
    ACL_REQUIRES_NON_NEGATIVE(numInputs);
    ACL_REQUIRES_NON_NEGATIVE(numOutputs);
    ACL_REQUIRES_OK(acl::array_utils::CheckPtrArray(numInputs, inputDesc));
    ACL_REQUIRES_OK(acl::array_utils::CheckPtrArray(numOutputs, outputDesc));
    if (acl::array_utils::IsHostMemTensorDesc(numInputs, inputDesc) != ACL_SUCCESS) {
        ACL_LOG_INNER_ERROR("[Check][HostMemTensorDesc]aclopCreateHandle or ACL_MEMTYPE_HOST_COMPILE_INDEPENDENT "
            "placeMent in inputDesc not support");
        return ACL_ERROR_API_NOT_SUPPORT;
    }
    if (acl::array_utils::IsHostMemTensorDesc(numOutputs, outputDesc) != ACL_SUCCESS) {
        ACL_LOG_INNER_ERROR("[Check][HostMemTensorDesc]aclopCreateHandle ACL_MEMTYPE_HOST or "
            "ACL_MEMTYPE_HOST_COMPILE_INDEPENDENT placeMent in outputDesc not support");
        return ACL_ERROR_API_NOT_SUPPORT;
    }

    acl::AclOp aclOp;
    aclOp.opType = std::string(opType);
    aclOp.numInputs = numInputs;
    aclOp.inputDesc = inputDesc;
    aclOp.numOutputs = numOutputs;
    aclOp.outputDesc = outputDesc;
    aclOp.opAttr = opAttr;
    aclOp.isCompile = false;

    acl::OpHandle *opHandle = nullptr;
    ACL_REQUIRES_OK(acl::OpExecutor::CreateOpHandle(aclOp, &opHandle));

    auto* const newHandle = new(std::nothrow) aclopHandle();
    ACL_CHECK_MALLOC_RESULT(newHandle);

    newHandle->opHandle = opHandle;
    *handle = newHandle;
    return ACL_SUCCESS;
}

void aclopDestroyHandleImpl(aclopHandle *handle)
{
    ACL_PROFILING_REG(acl::AclProfType::AclopDestroyHandle);
    ACL_DELETE_AND_SET_NULL(handle);
}

aclError aclopExecWithHandleImpl(aclopHandle *handle,
                             int numInputs,
                             const aclDataBuffer *const inputs[],
                             int numOutputs,
                             aclDataBuffer *const outputs[],
                             aclrtStream stream)
{
    ACL_PROFILING_REG(acl::AclProfType::AclopExecWithHandle);
    ACL_LOG_INFO("start to execute aclopExecWithHandle");
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(handle);
    ACL_REQUIRES_NON_NEGATIVE_WITH_INPUT_REPORT(numInputs);
    ACL_REQUIRES_NON_NEGATIVE_WITH_INPUT_REPORT(numOutputs);
    ACL_REQUIRES_OK(acl::array_utils::CheckPtrArray(numInputs, inputs));
    ACL_REQUIRES_OK(acl::array_utils::CheckPtrArray(numOutputs, outputs));
    if (acl::array_utils::IsAllTensorEmpty(numOutputs, outputs)) {
        ACL_LOG_INFO("all ouput tensor are empty");
        return ACL_SUCCESS;
    }

    auto &opHandle = *handle->opHandle;
    if (numInputs != opHandle.numInputs) {
        ACL_LOG_ERROR("[Check][NumInputs]input num mismatch: expect %d, but %d", opHandle.numInputs, numInputs);
        const std::string errMsg =
            acl::AclErrorLogManager::FormatStr("input num mismatch: expect %d", opHandle.numInputs);
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
            std::vector<const char *>({"param", "value", "reason"}),
            std::vector<const char *>({"input num", std::to_string(numInputs).c_str(), errMsg.c_str()}));
        return ACL_ERROR_OP_INPUT_NOT_MATCH;
    }

    if (numOutputs != opHandle.numOutputs) {
        ACL_LOG_ERROR("[Check][NumOutputs]output num mismatch: expect %d, but %d", opHandle.numOutputs, numOutputs);
        const std::string errMsg =
            acl::AclErrorLogManager::FormatStr("input num mismatch: expect %d", opHandle.numOutputs);
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
            std::vector<const char *>({"param", "value", "reason"}),
            std::vector<const char *>({"output num", std::to_string(numOutputs).c_str(), errMsg.c_str()}));
        return ACL_ERROR_OP_OUTPUT_NOT_MATCH;
    }

    return acl::OpExecutor::ExecuteAsync(opHandle, inputs, outputs, stream);
}

aclError aclopExecuteImpl(const char *opType,
                      int numInputs,
                      const aclTensorDesc *const inputDesc[],
                      const aclDataBuffer *const inputs[],
                      int numOutputs,
                      const aclTensorDesc *const outputDesc[],
                      aclDataBuffer *const outputs[],
                      const aclopAttr *attr,
                      aclrtStream stream)
{
    ACL_PROFILING_REG(acl::AclProfType::AclopExecute);
    ACL_LOG_INFO("start to execute aclopExecute");
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(opType);
    ACL_REQUIRES_NON_NEGATIVE_WITH_INPUT_REPORT(numInputs);
    ACL_REQUIRES_NON_NEGATIVE_WITH_INPUT_REPORT(numOutputs);
    ACL_REQUIRES_OK(acl::array_utils::CheckPtrArray(numInputs, inputDesc));
    ACL_REQUIRES_OK(acl::array_utils::CheckPtrArray(numOutputs, outputDesc));
    if (acl::array_utils::IsAllTensorEmpty(numOutputs, outputDesc)) {
        ACL_LOG_INFO("all ouput tensor are empty");
        return ACL_SUCCESS;
    }

    ACL_REQUIRES_OK(acl::array_utils::CheckPtrArray(numInputs, inputs));
    ACL_REQUIRES_OK(acl::array_utils::CheckPtrArray(numOutputs, outputs));

    acl::AclOp aclOp;
    aclOp.opType = std::string(opType);
    aclOp.numInputs = numInputs;
    aclOp.inputDesc = inputDesc;
    aclOp.numOutputs = numOutputs;
    aclOp.outputDesc = outputDesc;
    aclOp.inputs = inputs;
    aclOp.outputs = outputs;
    aclOp.opAttr = attr;
    aclOp.isCompile = false;

    return acl::OpExecutor::ExecuteAsync(aclOp, inputs, outputs, stream);
}

aclError aclopExecuteV2Impl(const char *opType,
                        int numInputs,
                        aclTensorDesc *inputDesc[],
                        aclDataBuffer *inputs[],
                        int numOutputs,
                        aclTensorDesc *outputDesc[],
                        aclDataBuffer *outputs[],
                        aclopAttr *attr,
                        aclrtStream stream)
{
    ACL_PROFILING_REG(acl::AclProfType::AclopExecuteV2);

    ACL_LOG_INFO("start to execute aclopExecuteV2");
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(opType);
    ACL_REQUIRES_NON_NEGATIVE(numInputs);
    ACL_REQUIRES_NON_NEGATIVE(numOutputs);
    ACL_REQUIRES_OK(acl::array_utils::CheckPtrArray(numInputs, inputDesc));
    ACL_REQUIRES_OK(acl::array_utils::CheckPtrArray(numOutputs, outputDesc));
    if (acl::array_utils::IsAllTensorEmpty(numOutputs, outputDesc)) {
        ACL_LOG_INFO("all ouput tensor are empty");
        return ACL_SUCCESS;
    }

    ACL_REQUIRES_OK(acl::array_utils::CheckDataBufferArry(numInputs, inputs));
    ACL_REQUIRES_OK(acl::array_utils::CheckDataBufferArry(numOutputs, outputs));

    acl::AclOp aclOp;
    aclOp.opType = std::string(opType);
    aclOp.numInputs = numInputs;
    aclOp.inputDesc = inputDesc;
    aclOp.numOutputs = numOutputs;
    aclOp.outputDesc = outputDesc;
    aclOp.inputs = inputs;
    aclOp.outputs = outputs;
    aclOp.opAttr = attr;
    aclOp.isCompile = false;
    aclOp.exeucteType = acl::ACL_OP_EXECUTE_V2;
    ACL_REQUIRES_OK(acl::OpExecutor::ExecuteAsync(aclOp, inputs, outputs, stream));
    return ACL_SUCCESS;
}

aclError aclopCreateKernelImpl(const char *opType,
                           const char *kernelId,
                           const char *kernelName,
                           void *binData,
                           int binSize,
                           aclopEngineType enginetype,
                           aclDataDeallocator deallocator)
{
    ACL_PROFILING_REG(acl::AclProfType::AclopCreateKernel);
    ACL_LOG_INFO("start to execute aclopCreateKernel");
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(opType);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(kernelId);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(kernelName);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(binData);

    auto *const registration = new(std::nothrow) acl::OpKernelRegistration();
    ACL_CHECK_MALLOC_RESULT(registration);

    registration->opType = opType;
    registration->kernelId = kernelId;
    registration->kernelName = kernelName;
    registration->binData = binData;
    registration->binSize = static_cast<uint64_t>(binSize);
    registration->enginetype = enginetype;
    registration->deallocator = deallocator;

    ACL_REQUIRES_OK(acl::OpKernelRegistry::GetInstance().Register(
        std::unique_ptr<acl::OpKernelRegistration>(registration)));
    ACL_LOG_DEBUG("Successfully created kernel. opType = %s, kernelId = %s, kernelName = %s", opType,
                  kernelId, kernelName);
    return ACL_SUCCESS;
}

aclError aclopUpdateParamsImpl(const char *opType,
                           int numInputs,
                           const aclTensorDesc *const inputDesc[],
                           int numOutputs,
                           const aclTensorDesc *const outputDesc[],
                           const aclopAttr *attr)
{
    ACL_PROFILING_REG(acl::AclProfType::AclopUpdateParams);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(opType);
    ACL_REQUIRES_NON_NEGATIVE_WITH_INPUT_REPORT(numInputs);
    ACL_REQUIRES_NON_NEGATIVE_WITH_INPUT_REPORT(numOutputs);
    ACL_REQUIRES_OK(acl::array_utils::CheckPtrArray(numInputs, inputDesc));
    ACL_REQUIRES_OK(acl::array_utils::CheckPtrArray(numOutputs, outputDesc));

    ACL_LOG_INFO("start to execute aclopUpdateParams. opType = %s", opType);
    acl::AclOp aclOp;
    aclOp.opType = std::string(opType);
    aclOp.numInputs = numInputs;
    aclOp.inputDesc = inputDesc;
    aclOp.numOutputs = numOutputs;
    aclOp.outputDesc = outputDesc;
    aclOp.opAttr = attr;

    return acl::OpKernelSelector::GetInstance().SelectOpKernel(aclOp);
}

aclError aclopSetMaxOpQueueNumImpl(uint64_t maxOpNum)
{
    if (maxOpNum == 0UL) {
        ACL_LOG_ERROR("[Check][maxOpNum]maxOpNum must be positive, maxOpNum = %lu", maxOpNum);
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
            std::vector<const char *>({"param", "value", "reason"}),
            std::vector<const char *>({"maxOpNum", std::to_string(maxOpNum).c_str(), "must be positive"}));
        return ACL_ERROR_INVALID_PARAM;
    }
    ACL_LOG_INFO("max_op_num is set [%lu]", maxOpNum);
    acl::OpKernelSelector::GetInstance().SetMaxOpNum(maxOpNum);
    acl::AclOpResourceManager::GetInstance().SetMaxOpNum(maxOpNum);
    return ACL_SUCCESS;
}

aclError aclopSetKernelArgsImpl(aclopKernelDesc *kernelDesc,
                            const char *kernelId,
                            uint32_t blockDim,
                            const void *args,
                            uint32_t argSize)
{
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(kernelDesc);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(args);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(kernelId);

    ACL_LOG_DEBUG("start to execute aclopSetKernelArgs, kernelId = %s, blockDim = %u, argSize = %u", kernelId,
                  blockDim, argSize);
    kernelDesc->kernelId = std::string(kernelId);
    kernelDesc->blockDim = blockDim;
    kernelDesc->extendArgs = std::string(static_cast<const char *>(args), static_cast<size_t>(argSize));

    return ACL_SUCCESS;
}

namespace acl {
void SetLoadOpsProtoFlag(const bool value)
{
    aclLoadOpsProtoFlag = value;
}

static aclError GetOppPluginVendors(const std::string &vendorsConfig, std::vector<std::string> &vendors)
{
    ACL_LOG_DEBUG("Enter get opp plugin vendors schedule, config file is '%s'", vendorsConfig.c_str());
    std::ifstream config(vendorsConfig);
    if (!config.good()) {
        ACL_LOG_INFO("Can not open file '%s', %s", vendorsConfig.c_str(), strerror(errno));
        return ACL_ERROR_FAILURE;
    }
    std::string content;
    (void)std::getline(config, content);
    config.close();
    if (content.empty()) {
        ACL_LOG_INNER_ERROR("Content of file '%s' is empty!", vendorsConfig.c_str());
        return ACL_ERROR_FAILURE;
    }
    std::vector<std::string> vParts;
    acl::StringUtils::Split(content, '=', vParts);
    size_t vendorConfigPartsCount = 2U;
    ACL_REQUIRES_TRUE(vParts.size() == vendorConfigPartsCount, ACL_ERROR_FAILURE,
        "Format of file content is invalid!");
    acl::StringUtils::Split(vParts[1], ',', vendors);
    ACL_REQUIRES_TRUE(!vendors.empty(), ACL_ERROR_FAILURE,
        "Format of file content is invalid!");
    (void) for_each(vendors.begin(), vendors.end(), &acl::StringUtils::Trim);
    return ACL_SUCCESS;
}

static void GetOpsProtoPathFromCustomOppPath(std::string &opsProtoPath)
{
    ACL_LOG_DEBUG("Start to get ops proto path from ASCEND_CUSTOM_OPP_PATH schedule.");
    opsProtoPath = "";
    const char_t *customOppPathEnv = nullptr;
    MM_SYS_GET_ENV(MM_ENV_ASCEND_CUSTOM_OPP_PATH, customOppPathEnv);
    if (customOppPathEnv == nullptr) {
        ACL_LOG_INFO("env ASCEND_CUSTOM_OPP_PATH is not defined.");
        return;
    }
    const std::string customOppPath = customOppPathEnv;
    if (customOppPath.empty()) {
        ACL_LOG_WARN("env ASCEND_CUSTOM_OPP_PATH is defined but it's empty.");
        return;
    }
    ACL_LOG_INFO("value of env ASCEND_CUSTOM_OPP_PATH is %s.", customOppPath.c_str());
    std::vector<std::string> customPaths;
    acl::StringUtils::Split(customOppPath, ':', customPaths);
    for (const auto &customPath : customPaths) {
        if ((!customPath.empty()) && (mmIsDir((customPath + "/op_proto").c_str()) == EN_OK)) {
            ACL_LOG_INFO("customPath '%s' is valid.", customPath.c_str());
            opsProtoPath += customPath + "/op_proto/:";
        } else {
            ACL_LOG_WARN("customPath '%s' is invalid, which is skipped.", customPath.c_str());
        }
    }
    ACL_LOG_INFO("Run GetOpsProtoPathFromCustomOppPath finished, current opsProtoPath is %s.", opsProtoPath.c_str());
}

static bool GetOpsProtoPath(std::string &opsProtoPath)
{
    ACL_LOG_DEBUG("Start to get ops proto path schedule.");
    const char_t *pathEnv = nullptr;
    MM_SYS_GET_ENV(MM_ENV_ASCEND_OPP_PATH, pathEnv);
    if (pathEnv != nullptr) {
        const std::string path = pathEnv;
        if (path.empty()) {
            ACL_LOG_INNER_ERROR("[Check][Path]File path is empty string.");
            return false;
        }
        const std::string filePath = ge::RealPath(path.c_str());
        if (filePath.empty()) {
            ACL_LOG_INNER_ERROR("[Check][Path]File path %s is invalid.", path.c_str());
            return false;
        }
        ACL_LOG_INFO("Get opsproto path from: %s", path.c_str());
        const std::string pathBuiltIn = filePath + "/built-in";
        if (mmIsDir(pathBuiltIn.c_str()) != EN_OK) {
            opsProtoPath = (filePath + "/op_proto/custom/" + ":") + (filePath + "/op_proto/built-in/");
        } else {
            GetOpsProtoPathFromCustomOppPath(opsProtoPath);
            std::vector<std::string> vendors;
            const std::string pathVendors = filePath + "/vendors";
            const aclError ret = GetOppPluginVendors(pathVendors + "/config.ini", vendors);
            if (ret != ACL_SUCCESS) {
                ACL_LOG_INFO("Can not get opp plugin vendors!");
                opsProtoPath += filePath + "/op_proto/custom/:";
            } else {
                for (const auto &vendor : vendors) {
                    opsProtoPath += pathVendors + "/" + vendor + "/op_proto/:";
                }
            }
            opsProtoPath += filePath + "/built-in/op_proto/";
        }
        ACL_LOG_INFO("Opsproto path is: %s", opsProtoPath.c_str());
        return true;
    }
    return false;
}

static aclError LoadOpsProto()
{
    ACL_LOG_INFO("LoadOpsProtoLib begin");
    std::string opsprotoPath;
    if (!GetOpsProtoPath(opsprotoPath)) {
        ACL_LOG_INNER_ERROR("[Check][ProtoPath]The environment variable(ASCEND_OPP_PATH) is not set or path is "
                            "invalid.");
        return ACL_ERROR_INVALID_OPP_PATH;
    }
    ge::OpsProtoManager* const protoManager = ge::OpsProtoManager::Instance();
    std::map<std::string, std::string> optionTmp;
    (void)optionTmp.emplace(std::pair<std::string, std::string>(std::string("ge.opsProtoLibPath"), opsprotoPath));
    const bool isProtoInit = protoManager->Initialize(optionTmp);
    if (!isProtoInit) {
        ACL_LOG_INNER_ERROR("[Init][Manager]Load ops_proto lib failed, ops proto path[%s] is invalid.",
            opsprotoPath.c_str());
        return ACL_ERROR_FAILURE;
    }
    ACL_LOG_INFO("LoadOpsProtoLib success");

    std::vector<ge::AscendString> allOp;
    const ge::graphStatus ret = ge::OperatorFactory::GetOpsTypeList(allOp);
    if (ret != ge::GRAPH_SUCCESS) {
        ACL_LOG_CALL_ERROR("[Get][OpsType]GetOpsTypeList failed.");
        return ACL_GET_ERRCODE_GE(ret);
    }
    ACL_LOG_INFO("OpsTypeListSize is %zu", allOp.size());
    return ACL_SUCCESS;
}

static aclError UpdateOutPutDesc(const ge::Operator &inferOp,
    const int32_t numOutputs, aclTensorDesc *const outputDesc[])
{
    ACL_LOG_INFO("Begin to update OutPutDesc, numOutputs is %d", numOutputs);
    // update outputDesc
    std::stringstream ss;
    for (int32_t i = 0; i < numOutputs; ++i) {
        ge::AscendString ascendString;
        // get inferOutputDesc after inferShape
        auto inferOutputDesc = inferOp.GetOutputDesc(static_cast<uint32_t>(i));
        const ge::Format outputFormat = inferOutputDesc.GetFormat();
        const ge::DataType outputDType = inferOutputDesc.GetDataType();
        auto ret = inferOutputDesc.GetName(ascendString);
        if (ret != ge::GRAPH_SUCCESS) {
            ACL_LOG_CALL_ERROR("[Get][Name]the %d tensor GetName failed.", i);
            return ACL_GET_ERRCODE_GE(ret);
        }
        std::string outputName;
        if (ascendString.GetString() != nullptr) {
            outputName = std::string(ascendString.GetString());
        }
        const ge::Shape outputShape = inferOutputDesc.GetShape();
        std::vector<std::pair<int64_t, int64_t>> outputRange;
        ret = inferOutputDesc.GetShapeRange(outputRange);
        if (ret != ge::GRAPH_SUCCESS) {
            ACL_LOG_CALL_ERROR("[Get][ShapeRange]the %d tensor GetShapeRange failed.", i);
            return ACL_GET_ERRCODE_GE(ret);
        }

        // update outputDesc
        outputDesc[i]->dataType = static_cast<aclDataType>(outputDType);
        outputDesc[i]->format = static_cast<aclFormat>(outputFormat);
        outputDesc[i]->name = outputName;
        const std::vector<int64_t> outputDims = outputShape.GetDims();
        ACL_LOG_INFO("inferShapeDimSize is %zu", outputDims.size());
        outputDesc[i]->dims.clear();
        for (size_t j = 0UL; j < outputDims.size(); ++j) {
            outputDesc[i]->dims.emplace_back(outputDims[j]);
        }
        outputDesc[i]->shapeRange.clear();
        for (size_t j = 0UL; j < outputRange.size(); ++j) {
            outputDesc[i]->shapeRange.emplace_back(outputRange[j]);
        }
        if (acl::AclLog::IsLogOutputEnable(ACL_INFO)) {
            ss << "inferOutputDesc[" << i << "]: ";
            ss << outputDesc[i]->DebugString() << " ";
        }
    }

    ACL_LOG_INFO("inferOutputDesc is %s", ss.str().c_str());
    return ACL_SUCCESS;
}

static void AddOpDesc(const aclTensorDesc *const tensorDesc, ge::OpDescPtr &opDesc, const bool isInput)
{
    ge::Format geFormat = ge::FORMAT_RESERVED;
    if (tensorDesc->format != ACL_FORMAT_UNDEFINED) {
        geFormat = static_cast<::ge::Format>(tensorDesc->format);
    }
    ge::DataType geDataType = ge::DT_UNDEFINED;
    if (tensorDesc->dataType != ACL_DT_UNDEFINED) {
        geDataType = static_cast<::ge::DataType>(tensorDesc->dataType);
    }
    std::vector<int64_t> dims;
    ConvertSvecToVec(tensorDesc->dims, dims);
    ge::GeTensorDesc geTensorDesc(ge::GeShape(dims),
                                  geFormat,
                                  geDataType);
    geTensorDesc.SetOriginFormat(geFormat);
    ge::TensorUtils::SetRealDimCnt(geTensorDesc, static_cast<uint32_t>(tensorDesc->dims.size()));
    if (!tensorDesc->valRange.empty()) {
        (void)geTensorDesc.SetValueRange(tensorDesc->valRange);
    }
    if (isInput) {
        ge::TensorUtils::SetInputTensor(geTensorDesc, true);
        ge::TensorUtils::SetOutputTensor(geTensorDesc, false);
    } else {
        ge::TensorUtils::SetInputTensor(geTensorDesc, false);
        ge::TensorUtils::SetOutputTensor(geTensorDesc, true);
    }
    if (!tensorDesc->name.empty()) {
        if (isInput) {
            (void)(opDesc->AddInputDesc(tensorDesc->name, geTensorDesc));
        } else {
            (void)(opDesc->AddOutputDesc(tensorDesc->name, geTensorDesc));
        }
    } else {
        if (isInput) {
            (void)(opDesc->AddInputDesc(geTensorDesc));
        } else {
            (void)(opDesc->AddOutputDesc(geTensorDesc));
        }
    }
}

static aclError AddDataInput(const aclTensorDesc *const inputDesc,
                             const aclDataBuffer *const inputs,
                             std::unique_ptr<uint8_t[]> &constData,
                             ge::Operator &constOp)
{
    // create const operator
    std::vector<int64_t> dims;
    ConvertSvecToVec(inputDesc->dims, dims);
    const ge::TensorDesc constDesc(ge::Shape(dims),
                             static_cast<::ge::Format>(inputDesc->format),
                             static_cast<::ge::DataType >(inputDesc->dataType));
    const size_t tensorSize = inputs->length;
    if (tensorSize == 0UL) {
        ACL_LOG_ERROR("[Check][TensorSize]tensorSize must be positive, tensorSize = %zu", tensorSize);
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
            std::vector<const char *>({"param", "value", "reason"}),
            std::vector<const char *>({"tensorSize", std::to_string(tensorSize).c_str(), "must be positive"}));
        return ACL_ERROR_INVALID_PARAM;
    }
    auto args = std::unique_ptr<uint8_t[]>(new(std::nothrow) uint8_t[tensorSize]);
    if (args == nullptr) {
        ACL_LOG_INNER_ERROR("[Allocate][Mem]Allocate memory failed.");
        return static_cast<int32_t>(ACL_ERROR_BAD_ALLOC);
    }
    constData = std::move(args);
    if (memcpy_s(constData.get(), tensorSize, inputs->data, tensorSize) != EOK) {
        ACL_LOG_INNER_ERROR("[Copy][Mem]Copy input data failed. size = %zu", tensorSize);
        return ACL_ERROR_FAILURE;
    }
    const ge::Tensor constTensor(constDesc, constData.get(), tensorSize);
    const std::string valueName = ge::ATTR_NAME_WEIGHTS;
    (void)constOp.SetAttr(valueName.c_str(), constTensor);
    const auto retConstInfer = constOp.InferShapeAndType();
    if (retConstInfer != ge::GRAPH_SUCCESS) {
        ACL_LOG_CALL_ERROR("the constOp inferShape failed. ge result = %u", retConstInfer);
        return ACL_GET_ERRCODE_GE(retConstInfer);
    }
    return ACL_SUCCESS;
}
}

aclError aclopSetKernelWorkspaceSizesImpl(aclopKernelDesc *kernelDesc, int numWorkspaces, size_t *workspaceSizes)
{
    ACL_LOG_DEBUG("start to execute aclopSetKernelWorkspaceSizes");
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(kernelDesc);
    ACL_REQUIRES_NON_NEGATIVE_WITH_INPUT_REPORT(numWorkspaces);
    if (numWorkspaces != 0) {
        ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(workspaceSizes);
    }

    for (int32_t i = 0; i < numWorkspaces; ++i) {
        kernelDesc->workspaceSizes.emplace_back(workspaceSizes[i]);
    }

    return ACL_SUCCESS;
}

aclError aclopUnregisterCompileFuncImpl(const char *opType)
{
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(opType);
    ACL_LOG_INFO("aclopUnregisterCompileFunc in, opType = %s", opType);
    acl::OpKernelSelector::GetInstance().Unregister(opType);
    ACL_LOG_INFO("Unregistering compile function successfully. op type = %s", opType);
    return ACL_SUCCESS;
}

aclError aclopRegisterCompileFuncImpl(const char *opType, aclopCompileFunc func)
{
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(opType);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(func);
    const bool ret = acl::OpKernelSelector::GetInstance().Register(opType, func);
    if (ret) {
        ACL_LOG_INFO("Registering compile function successfully. op type = %s", opType);
        return ACL_SUCCESS;
    } else {
        ACL_LOG_INNER_ERROR("[Register][Function]Failed to register compile function due to repeat registration. "
            "op type = %s", opType);
        return ACL_ERROR_BIN_SELECTOR_ALREADY_REGISTERED;
    }
}

aclError aclopInferShapeImpl(const char *opType,
                         int numInputs,
                         aclTensorDesc *inputDesc[],
                         aclDataBuffer *inputs[],
                         int numOutputs,
                         aclTensorDesc *outputDesc[],
                         aclopAttr *attr)
{
    ACL_LOG_INFO("start to execute aclopInferShape");
    ACL_PROFILING_REG(acl::AclProfType::AclopInferShape);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(opType);
    ACL_REQUIRES_NON_NEGATIVE(numInputs);
    ACL_REQUIRES_NON_NEGATIVE(numOutputs);
    ACL_REQUIRES_OK(acl::array_utils::CheckPtrArray(numInputs, inputDesc));
    ACL_REQUIRES_OK(acl::array_utils::CheckPtrArray(numOutputs, outputDesc));
    ACL_REQUIRES_OK(acl::array_utils::CheckPtrArray(numInputs, inputs));
    {
        const std::unique_lock<std::mutex> lk(aclLoadOpsProtoMutex);
        if (!aclLoadOpsProtoFlag) {
            const aclError ret = acl::LoadOpsProto();
            if (ret != ACL_SUCCESS) {
                ACL_LOG_INNER_ERROR("[Load][OpsProto]Load opsProto lib fail.");
                return ret;
            }
            acl::SetLoadOpsProtoFlag(true);
        } else {
            ACL_LOG_INFO("OpsProto lib has been successfully loaded");
        }
    }
    ge::Operator inferOp = ge::OperatorFactory::CreateOperator(opType, opType);
    ge::OpDescPtr opDesc = ge::OpDescUtils::GetOpDescFromOperator(inferOp);
    ACL_REQUIRES_NOT_NULL(opDesc);
    const size_t factoryInputSize = opDesc->GetAllInputName().size();
    ACL_LOG_INFO("size of GetAllInputName is %zu, numInputs of entered by user is %d", factoryInputSize, numInputs);
    if (factoryInputSize < static_cast<size_t>(numInputs)) {
        ACL_MAKE_SHARED(opDesc = std::make_shared<ge::OpDesc>(opType, opType), return ACL_ERROR_BAD_ALLOC);
        ACL_CHECK_MALLOC_RESULT(opDesc);
        for (int32_t i = 0; i < numInputs; ++i) {
            acl::AddOpDesc(inputDesc[i], opDesc, true);
        }
        ACL_LOG_INFO("addInputOpDesc success, numInputs is %d", numInputs);
        for (int32_t i = 0; i < numOutputs; ++i) {
            acl::AddOpDesc(outputDesc[i], opDesc, false);
        }
        ACL_LOG_INFO("addOutputOpDesc success numOutputs is %d", numOutputs);
        inferOp.BreakConnect();
        inferOp = ge::OpDescUtils::CreateOperatorFromOpDesc(opDesc);
    }

    if (attr != nullptr) {
        for (const auto &it : attr->Attrs()) {
            (void)(opDesc->SetAttr(it.first, it.second));
        }
    }
    std::unique_ptr<uint8_t[]> *constData = new(std::nothrow) std::unique_ptr<uint8_t[]>[numInputs];
    ACL_CHECK_MALLOC_RESULT(constData);
    std::vector<ge::Operator> constOps;
    for (int32_t i = 0; i < numInputs; ++i) {
        // need to update format because infershape only updte shape and dtype
        const auto desc = opDesc->MutableInputDesc(static_cast<uint32_t>(i));
        const auto &aclFormat = inputDesc[i]->format;
        const ge::Format format = (aclFormat == ACL_FORMAT_UNDEFINED) ?
            ge::FORMAT_RESERVED : static_cast<ge::Format>(aclFormat);
        desc->SetOriginFormat(format);
        desc->SetFormat(format);
        constOps.push_back(ge::OperatorFactory::CreateOperator("Const", "Const"));
        ge::Operator constOp = constOps.back();
        const aclError ret = acl::AddDataInput(inputDesc[i], inputs[i], constData[i], constOp);
        if (ret != ACL_SUCCESS) {
            ACL_LOG_INNER_ERROR("[Add][Data]add data fail, index = %d", i);
            ACL_DELETE_ARRAY_AND_SET_NULL(constData);
            return ret;
        }
        ACL_LOG_INFO("opDesc.GetInputNameByIndex, index = %d", i);
        const std::string inferOpName = opDesc->GetInputNameByIndex(static_cast<uint32_t>(i));
        ACL_LOG_INFO("opDesc.GetInputNameByIndex, inferOpName = %s", inferOpName.c_str());
        (void)inferOp.SetInput(inferOpName.c_str(), constOp, "y");
    }
    ACL_LOG_INFO("create constData success");
    const ge::graphStatus retInfer = inferOp.InferShapeAndType();
    if (retInfer != ge::GRAPH_SUCCESS) {
        ACL_LOG_CALL_ERROR("[Infer][ShapeAndType]the op:%s inferShape failed. ge result = %u",
            opType, retInfer);
        ACL_DELETE_ARRAY_AND_SET_NULL(constData);
        return ACL_GET_ERRCODE_GE(retInfer);
    }
    for (size_t i = 0UL; i < constOps.size(); ++i) {
        constOps[i].BreakConnect();
    }
    ACL_LOG_INFO("the op:%s inferShape success", opType);
    const aclError ret = acl::UpdateOutPutDesc(inferOp, numOutputs, outputDesc);
    if (ret != ACL_SUCCESS) {
        ACL_LOG_INNER_ERROR("[Update][OutPutDesc]the op:%s update outputDesc failed", opType);
        ACL_DELETE_ARRAY_AND_SET_NULL(constData);
        return ret;
    }
    ACL_LOG_INFO("the op:%s update outputDesc success", opType);
    inferOp.BreakConnect();
    ACL_DELETE_ARRAY_AND_SET_NULL(constData);
    return ACL_SUCCESS;
}

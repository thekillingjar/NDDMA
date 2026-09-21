/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "acl_op_resource_manager.h"
#include "awatchdog.h"
#include "framework/runtime/gert_api.h"
#include "framework/common/profiling_definitions.h"
#include "framework/executor/ge_executor.h"
#include "mmpa/mmpa_api.h"
#include "common/log_inner.h"
#include "common/common_inner.h"
#include "common/json_parser.h"
#include "single_op/op_model_parser.h"
#include "single_op/shape_range_utils.h"
#include "single_op/executor/stream_executor.h"
#include "single_op/compile/op_compile_service.h"
#include "utils/attr_utils.h"
#include "utils/file_utils.h"
#include "framework/runtime/subscriber/global_profiler.h"
#include "ge/ge_allocator.h"

namespace {
std::atomic<std::uint64_t> atomicModelId(0UL);
}
namespace acl {
namespace {
constexpr int32_t OM_FILE_SUFFIX_LEN = 3;
constexpr int32_t OM_DIR_MAX_DEPTH = 3;
constexpr int32_t DECIMAL = 10;
const std::string ACL_MAX_OPQUEUE_NUM = "max_opqueue_num";
}

AclOpResourceManager::AclOpResourceManager() {
    GetRuntimeV2Env();
}

void AclOpResourceManager::GetRuntimeV2Env()
{
    const char_t *enableRuntimeV2Flag = nullptr;
    MM_SYS_GET_ENV(MM_ENV_ENABLE_RUNTIME_V2, enableRuntimeV2Flag);
    if (enableRuntimeV2Flag != nullptr) {
        if (enableRuntimeV2Flag[0] == '0') { // 0 both model and singleOp disable
            enableRuntimeV2ForModel_ = false;
            enableRuntimeV2ForSingleOp_ = false;
        } else if (enableRuntimeV2Flag[0] == '2') { // 2: model enable, singleOp disable
            enableRuntimeV2ForModel_ = true;
            enableRuntimeV2ForSingleOp_ = false;
        } else {
            enableRuntimeV2ForModel_ = true;
            enableRuntimeV2ForSingleOp_ = true;
        }
    }
    ACL_LOG_EVENT("runtime v2 flag : model flag = %d, singleOp flag = %d",
                  static_cast<int32_t>(enableRuntimeV2ForModel_),
                  static_cast<int32_t>(enableRuntimeV2ForSingleOp_));
}

void *AclOpResourceManager::GetKeyByStreamOrDefaultStream(const aclrtStream stream)
{
    if (stream != nullptr) {
        return stream;
    }
    // get current context default stream
    aclrtStream curCtxDefaultStream = nullptr;
    const aclError aclErr = aclrtCtxGetCurrentDefaultStream(&curCtxDefaultStream);
    if (aclErr != ACL_ERROR_NONE) {
        ACL_LOG_CALL_ERROR("get current default stream failed, ret:%d", static_cast<int32_t>(aclErr));
        return nullptr;
    }
    return curCtxDefaultStream;
}

void AclOpResourceManager::SetMaxOpNum(const uint64_t maxOpNum)
{
    opModels_.SetMaxOpNum(maxOpNum);
}

aclError AclOpResourceManager::HandleMaxOpQueueConfig(const char_t *const configBuffer)
{
    ACL_LOG_INFO("start to execute HandleMaxOpQueueConfig");
    std::string maxOpNumStr;
    bool found = false;
    const aclError ret =  acl::JsonParser::GetJsonCtxByKeyFromBuffer(configBuffer, maxOpNumStr, ACL_MAX_OPQUEUE_NUM, found);
    if (ret != ACL_SUCCESS) {
        ACL_LOG_INNER_ERROR("[Parse][Config]parse max_opqueue_num config from buffer failed, errorCode = %d", ret);
        return ret;
    }

    if (found) {
        StringUtils::Strip(maxOpNumStr, "\"");
        // if parse failed, maxOpNum is zero
        const int64_t maxOpNum = strtol(maxOpNumStr.c_str(), nullptr, DECIMAL);
        ACL_LOG_INFO("max_opqueue_num is set [%ld].", maxOpNum);
        if (maxOpNum <= 0) {
            ACL_LOG_INNER_ERROR("[Check][MaxOpNum]max_opqueue_num [%s] is invalid from buffer, "
                                "it should be larger than 0.", maxOpNumStr.c_str());
            return ACL_ERROR_INVALID_PARAM;
        }
        if (static_cast<uint64_t>(maxOpNum) < DEFAULT_MAX_OPQUEUE_NUM) {
            ACL_LOG_WARN("max_opqueue_num [%lu] is less than default value DEFAULT_MAX_OPQUEUE_NUM[%lu], "
                         "it may be low performance.", static_cast<uint64_t>(maxOpNum), DEFAULT_MAX_OPQUEUE_NUM);
        }
        opModels_.SetMaxOpNum(static_cast<uint64_t>(maxOpNum));
    } else {
        ACL_LOG_INFO("no max_opqueue_num found, set default DEFAULT_MAX_OPQUEUE_NUM[%lu].",
                     DEFAULT_MAX_OPQUEUE_NUM);
    }

    ACL_LOG_INFO("HandleMaxOpQueueConfig end in HandleMaxOpQueueConfig");
    return ACL_SUCCESS;
}

bool AclOpResourceManager::OmFileFilterFn(const std::string &fileName)
{
    const auto pos = fileName.rfind(".om");
    if (pos == std::string::npos) {
        return false;
    }

    return pos == (fileName.size() - static_cast<size_t>(OM_FILE_SUFFIX_LEN));
}

bool AclOpResourceManager::IsDynamicOpModel(const OpModelDef &modelDef)
{
    if (modelDef.isDynamicModel) {
        ACL_LOG_INFO("the model is dynamic directly");
        return true;
    }
    if (modelDef.isStaticModelWithFuzzCompile == 0U) {
        // 0: ACL_OP_COMPILE_DEFAULT mode
        ACL_LOG_INFO("the model is compiled by exact compile");
        for (size_t i = 0U; i < modelDef.inputDescArr.size(); ++i) {
            if (modelDef.inputDescArr[i].IsDynamicTensor()) {
                return true;
            }
        }
        for (size_t i = 0U; i < modelDef.outputDescArr.size(); ++i) {
            if (modelDef.outputDescArr[i].IsDynamicTensor()) {
                return true;
            }
        }
        return false;
    } else if (modelDef.isStaticModelWithFuzzCompile == 1U) {
        // 1:ACL_OP_COMPILE_FUZZ mode but model is static
        ACL_LOG_INFO("the model is static model with fuzz compile");
        return false;
    } else {
        // 2:ACL_OP_COMPILE_FUZZ mode and model is dynamic
        ACL_LOG_INFO("the model is dynamic model with fuzz compile");
        return true;
    }
}

bool AclOpResourceManager::IsDynamicOpModel(const AclOp &aclOp)
{
    for (int32_t i = 0; i < aclOp.numInputs; ++i) {
        if (aclOp.inputDesc[i]->IsDynamicTensor()) {
            return true;
        }
    }
    for (int32_t i = 0; i < aclOp.numOutputs; ++i) {
        if (aclOp.outputDesc[i]->IsDynamicTensor()) {
            return true;
        }
    }
    return false;
}

aclError AclOpResourceManager::LoadAllModels(const std::string &modelDir)
{
    ACL_LOG_INFO("LoadAllModels begin. config path = %s", modelDir.c_str());
    std::vector<OpModelDef> defList;
    ACL_REQUIRES_OK(ReadModelDefs(modelDir, defList));

    if (defList.empty()) {
        ACL_LOG_WARN("No model is loaded.");
        return ACL_SUCCESS;
    }
    ACL_LOG_INFO("Found %zu model config", defList.size());
    for (auto &modelDef : defList) {
        const bool isDynamic = IsDynamicOpModel(modelDef);
        // it is static load
        bool isRegistered = false;
        ACL_REQUIRES_OK(RegisterModel(std::move(modelDef), opModels_, isDynamic, isRegistered, true));
    }
    return ACL_SUCCESS;
}

aclError AclOpResourceManager::LoadModelFromMem(const void *const model, const size_t modelSize, const bool isStatic)
{
    ACL_LOG_INFO("Load op model begin. modelSize = %zu", modelSize);
    ACL_REQUIRES_NOT_NULL(model);
    ACL_REQUIRES_POSITIVE(modelSize);
    auto *const aclModelData = new (std::nothrow) char_t[modelSize];
    ACL_REQUIRES_NOT_NULL(aclModelData);
    std::shared_ptr<void> modelData;
    modelData.reset(aclModelData, [](const char_t *const p) { delete[]p; });
    if (memcpy_s(aclModelData, modelSize, model, modelSize) != EOK) {
        ACL_LOG_INNER_ERROR("[Copy][Data]Copy model data failed. size = %zu", modelSize);
        return ACL_ERROR_FAILURE;
    }
    return LoadModelFromSharedMem(modelData, modelSize, nullptr, isStatic);
}

aclError AclOpResourceManager::LoadModelFromSharedMem(const std::shared_ptr<void> &model, const size_t modelSize,
                                                    AclOp *aclOp, const bool isStatic)
{
    ACL_LOG_INFO("Load inner op model begin. modelSize = %zu", modelSize);
    ACL_REQUIRES_NOT_NULL(model);
    ACL_REQUIRES_POSITIVE(modelSize);
    OpModel opModel;
    opModel.data = model;
    if (aclOp != nullptr) {
        aclOp->opModel.profilingIndex =
            static_cast<int64_t>(gert::GlobalProfilingWrapper::GetInstance()->RegisterString(aclOp->opType));
        opModel.profilingIndex = aclOp->opModel.profilingIndex;
    }
    opModel.size = static_cast<uint32_t>(modelSize);
    opModel.name = std::to_string(reinterpret_cast<uintptr_t>(model.get()));
    ACL_LOG_INFO("opModel.name = %s", opModel.name.c_str());

    OpModelDef modelDef;
    modelDef.modelPath = opModel.name;
    modelDef.opModelId = atomicModelId++;

    const auto ret = OpModelParser::ParseOpModel(opModel, modelDef);
    if (ret != ACL_SUCCESS) {
        ACL_LOG_INNER_ERROR("parse model failed. errorCode = %d. skip it", ret);
        return ret;
    }
    const bool isDynamic = IsDynamicOpModel(modelDef);
    ACL_LOG_INFO("The name of opModel is %s", opModel.name.c_str());
    bool isRegistered = false;
    const uint64_t opId = modelDef.opModelId;
    ACL_REQUIRES_OK(RegisterModel(std::move(modelDef), opModels_, isDynamic, isRegistered, isStatic));
    if (!isRegistered) {
        ACL_REQUIRES_OK(modelCache_.Add(opId, opModel));
    } else {
        ACL_LOG_INFO("current model %s is registered", opModel.name.c_str());
    }
    return ACL_SUCCESS;
}

aclError AclOpResourceManager::RegisterModel(OpModelDef &&modelConfig,
                                           ModelMap &opModelDefs,
                                           const bool isDynamic,
                                           bool &isRegistered,
                                           const bool isStaticRegister)
{
    const auto numInputs = modelConfig.inputDescArr.size();
    std::vector<const aclTensorDesc *> inputDesc;
    for (size_t i = 0U; i < numInputs; ++i) {
        inputDesc.emplace_back(&modelConfig.inputDescArr[i]);
    }
    std::vector<const aclTensorDesc *> outputDesc;
    const auto numOutputs = modelConfig.outputDescArr.size();
    for (size_t i = 0U; i < numOutputs; ++i) {
        outputDesc.emplace_back(&modelConfig.outputDescArr[i]);
    }

    AclOp aclOp;
    aclOp.opType = modelConfig.opType;
    aclOp.opAttr = &modelConfig.opAttr;
    aclOp.numInputs = static_cast<int32_t>(numInputs);
    aclOp.numOutputs = static_cast<int32_t>(numOutputs);
    aclOp.inputDesc = inputDesc.data();
    aclOp.outputDesc = outputDesc.data();
    if (!isStaticRegister) {
        // only dynamic load need update timestamp, static load is ULLONG_MAX default no need aging
        modelConfig.timestamp = attr_utils::GetCurrentTimestamp();
    }

    const auto modelDefPtr = std::make_shared<OpModelDef>(std::move(modelConfig));
    ACL_REQUIRES_NOT_NULL(modelDefPtr);
    std::shared_ptr<OpModelDef> agingModelDef = nullptr;
    if (isDynamic) {
        ACL_LOG_INFO("The model is dynamic shape");
        uint64_t seq = 0U;
        ShapeRangeUtils::GetInstance().SetTensorShapeInfo(aclOp, seq);
        modelDefPtr->seq = seq;
        ACL_REQUIRES_OK(opModelDefs.InsertDynamic(aclOp, modelDefPtr, agingModelDef, isRegistered));
    } else {
        ACL_LOG_INFO("Insert modeldef to hash map");
        ACL_REQUIRES_OK(opModelDefs.Insert(aclOp, modelDefPtr, agingModelDef, isRegistered));
    }

    if (agingModelDef != nullptr) {
        ACL_REQUIRES_OK(modelCache_.Delete(*agingModelDef, isDynamic));
    }
    const bool castHasTruncate = ((!GetIfCastHasTruncateAttr()) && (aclOp.opType == "Cast")) &&
                                 ((aclOp.opAttr != nullptr) && (aclOp.opAttr->HasAttr("truncate")));
    if (castHasTruncate) {
        ACL_LOG_INFO("Find cast op whose attr contains truncate");
        SetCastHasTruncateAttr(true);
    }
    ACL_LOG_INFO("Register model. OpModelDef = %s", modelDefPtr->DebugString().c_str());
    return ACL_SUCCESS;
}

aclError AclOpResourceManager::SetTensorConst(aclTensorDesc *const desc, const aclDataBuffer *const dataBuffer)
{
    ACL_REQUIRES_NOT_NULL(desc);
    ACL_REQUIRES_NOT_NULL(dataBuffer);
    const size_t length = dataBuffer->length;
    void *const hostMem = dataBuffer->data;
    if (length == 0U) {
        desc->isConst = true;
        return ACL_SUCCESS;
    }

    desc->isConst = true;
    desc->constDataBuf.reset(static_cast<char_t *>(hostMem), [](const char_t *const) {});
    desc->constDataLen = length;
    return ACL_SUCCESS;
}

aclError AclOpResourceManager::SetHostMemToConst(const AclOp &aclopHostMemToConst, bool &isExistConst) const
{
    for (int32_t i = 0; i < aclopHostMemToConst.numInputs; ++i) {
        if ((aclopHostMemToConst.inputDesc[i]->memtype == ACL_MEMTYPE_HOST) &&
            (!aclopHostMemToConst.inputDesc[i]->isConst)) {
            isExistConst = true;
            // HostMem needs to be constructed as constTensor
            ACL_REQUIRES_OK(SetTensorConst(const_cast<aclTensorDesc *>(aclopHostMemToConst.inputDesc[i]),
                                           aclopHostMemToConst.inputs[i]));
        }
    }
    for (int32_t i = 0; i < aclopHostMemToConst.numOutputs; ++i) {
        if ((aclopHostMemToConst.outputDesc[i]->memtype == ACL_MEMTYPE_HOST) &&
            (!aclopHostMemToConst.outputDesc[i]->isConst)) {
            isExistConst = true;
            // HostMem needs to be constructed as constTensor
            ACL_REQUIRES_OK(SetTensorConst(const_cast<aclTensorDesc *>(aclopHostMemToConst.outputDesc[i]),
                                           aclopHostMemToConst.outputs[i]));
        }
    }
    return ACL_SUCCESS;
}

aclError AclOpResourceManager::GetOpModel(AclOp &aclOp)
{
    bool isDynamic = false;
    aclError ret = MatchOpModel(aclOp, aclOp.opModel, isDynamic);
    if (ret == ACL_SUCCESS) {
        aclOp.isMatched = true;
        aclOp.isDynamic = isDynamic;
        ACL_LOG_INFO("operator %s is already registered, isDynamicModel = %d", aclOp.opType.c_str(),
                     static_cast<int32_t>(isDynamic));
        return ret;
    }
    constexpr uint32_t awdTimeout = 300U; // u300 seconds, 5 min
    const AwdHandle hdl = AwdCreateThreadWatchdog(static_cast<uint32_t>(ACL_MODE_ID), awdTimeout, nullptr);
    (void)AwdStartThreadWatchdog(hdl);
    ret = BuildOpModel(aclOp);
    if (ret != ACL_SUCCESS) {
        ACL_LOG_INNER_ERROR("[Build][Op]Fail to build op model");
        (void)AwdStopThreadWatchdog(hdl);
        AwdDestroyThreadWatchdog(hdl);
        return ret;
    }
    (void)AwdStopThreadWatchdog(hdl);
    AwdDestroyThreadWatchdog(hdl);
    return ACL_SUCCESS;
}

aclError AclOpResourceManager::MatchStaticOpModel(const AclOp &aclOp, OpModel &opModel,
                                                bool &isDynamic, bool &isNeedMatchDymaic)
{
    RT2_PROFILING_SCOPE(gert::profiling::kUnknownName, gert::profiling::kAclMatchStaticOpModel);
    std::shared_ptr<OpModelDef> modelDef;
    aclError ret = ACL_SUCCESS;
    isNeedMatchDymaic = false;
    bool isExistConst = false;
    ACL_REQUIRES_OK(SetHostMemToConst(aclOp, isExistConst));
    if (isExistConst) {
        ret = opModels_.Get(aclOp, modelDef, true);
        aclOp.RecoverConst();
        if (ret == ACL_SUCCESS) {
            isDynamic = false;
            ACL_LOG_INFO("Match static model with const memory successfully. opType = %s, opModel = %s",
                         aclOp.opType.c_str(), modelDef->modelPath.c_str());
            ret = modelCache_.GetOpModel(*modelDef, opModel);
            return ret;
        } else {
            ret = opModels_.Get(aclOp, modelDef, true);
            if (ret == ACL_SUCCESS) {
                isDynamic = false;
                ACL_LOG_INFO("Match static model successfully. opType = %s, opModel = %s",
                             aclOp.opType.c_str(), modelDef->modelPath.c_str());
                ret = modelCache_.GetOpModel(*modelDef, opModel);
                return ret;
            }
        }
    } else {
        ret = opModels_.Get(aclOp, modelDef, true);
        if (ret == ACL_SUCCESS) {
            isDynamic = false;
            ACL_LOG_INFO("Match static model successfully. opType = %s, opModel = %s",
                         aclOp.opType.c_str(), modelDef->modelPath.c_str());
            ret = modelCache_.GetOpModel(*modelDef, opModel);
            return ret;
        }
    }
    isNeedMatchDymaic = true;

    return ret;
}

aclError AclOpResourceManager::MatchDynamicOpModel(const AclOp &aclOp, OpModel &opModel, bool &isDynamic)
{
    RT2_PROFILING_SCOPE(gert::profiling::kUnknownName, gert::profiling::kAclMatchDynamicOpModel);
    std::shared_ptr<OpModelDef> modelDef;
    aclError ret = ACL_SUCCESS;
    // check the input shape must be static when executing
    if (!aclOp.isCompile) {
        ACL_CHECK_WITH_INNER_MESSAGE_AND_RETURN(!IsDynamicOpModel(aclOp), ACL_ERROR_INVALID_PARAM,
                                                "[Check][Op]TensorDesc must be static when executing, "
                                                "tensorDesc is %s", aclOp.DebugString().c_str());
    }
    // Need to refresh the shape(-1/-2) and go to map to find model when executing
    std::vector<OpRangeInfo> *opRangeInfoVec;
    const MmRDLockGuard lk(&ShapeRangeUtils::GetInstance().shapeInfoMutex_);
    opRangeInfoVec = ShapeRangeUtils::GetInstance().GetTensorShapeInfo(aclOp);
    if (opRangeInfoVec == nullptr) {
        if (!aclOp.isCompile) {
            ACL_LOG_INFO("No match dynamic model, opName = %s .", aclOp.opType.c_str());
        }
        return ACL_ERROR_OP_NOT_FOUND;
    }
    for (size_t i = 0U; i < opRangeInfoVec->size(); ++i) {
        if (!ShapeRangeUtils::GetInstance().CheckShapeRange(aclOp, (*opRangeInfoVec)[i])) {
            continue;
        }
        ret = opModels_.GetDynamic(aclOp, modelDef, (*opRangeInfoVec)[i].seq, true);
        if (ret == ACL_SUCCESS) {
            ACL_LOG_INFO("Match dynamic model success. opType = %s, opModel = %s",
                         aclOp.opType.c_str(), modelDef->modelPath.c_str());
            isDynamic = true;
            ret = modelCache_.GetOpModel(*modelDef, opModel);
            return ret;
        } else {
            // aicpu is -2 but need to set const
            bool isExistConst = false;
            ACL_REQUIRES_OK(SetHostMemToConst(aclOp, isExistConst));
            if (!isExistConst) {
                continue;
            }
            ret = opModels_.GetDynamic(aclOp, modelDef, (*opRangeInfoVec)[i].seq, true);
            aclOp.RecoverConst();
            if (ret == ACL_SUCCESS) {
                ACL_LOG_INFO("Match dynamic model success. opType = %s, opModel = %s",
                             aclOp.opType.c_str(), modelDef->modelPath.c_str());
                isDynamic = true;
                ret = modelCache_.GetOpModel(*modelDef, opModel);
                return ret;
            }
        }
    }
    return ACL_ERROR_OP_NOT_FOUND;
}

aclError AclOpResourceManager::MatchOpModel(const AclOp &aclOp, OpModel &opModel, bool &isDynamic)
{
    aclError ret;
    RT2_PROFILING_SCOPE(gert::profiling::kUnknownName, gert::profiling::kAclMatchOpModel);
    aclOp.BackupConst();
    if ((GetGlobalJitCompileFlag() == 0) || (GetGlobalCompileFlag() == 1)) {
        ret = MatchDynamicOpModel(aclOp, opModel, isDynamic);
        if (ret != ACL_SUCCESS) {
            bool isNeedMatchDymaic = false;
            ret = MatchStaticOpModel(aclOp, opModel, isDynamic, isNeedMatchDymaic);
        }
    } else {
        bool isNeedMatchDymaic = false;
        ret = MatchStaticOpModel(aclOp, opModel, isDynamic, isNeedMatchDymaic);
        if (!isNeedMatchDymaic) {
            return ret;
        }
        ret = MatchDynamicOpModel(aclOp, opModel, isDynamic);
    }

    if (UNLIKELY((ret != ACL_SUCCESS) && (!aclOp.isCompile))) {
        ACL_LOG_INNER_ERROR("[Match][OpModel]MatchOpModel opName = %s fail from static map or dynamic map. "
                            "Please make sure the op executed and the op compiled is matched, "
                            "you can check the op type, op inputs number, outputs number, "
                            "input format, origin format, datatype, memtype, attr, dim range, "
                            "and so on.", aclOp.opType.c_str());
    }
    return ret;
}

aclError AclOpResourceManager::UpdateRT2Executor(const uint64_t id,
                                               const std::shared_ptr<gert::StreamExecutor> &executor)
{
    return modelCache_.UpdateCachedExecutor(id, executor);
}

std::shared_ptr<std::mutex> AclOpResourceManager::GetCacheMutex(const uint64_t id)
{
    return modelCache_.GetCacheMutex(id);
}

std::shared_ptr<gert::StreamExecutor> AclOpResourceManager::GetRT2Executor(const uint64_t id)
{
    return modelCache_.GetRT2Executor(id);
}

aclError AclOpResourceManager::UnloadModelData(const uint64_t id)
{
    return modelCache_.UnloadCachedModelData(id);
}

aclError AclOpResourceManager::CleanRT2Executor(rtStream_t stream)
{
    return modelCache_.CleanCachedExecutor(stream);
}

aclError AclOpResourceManager::ReadModelDefs(const std::string &configPath,
                                           std::vector<OpModelDef> &configList)
{
    std::vector<std::string> modelFilePaths;
    ACL_REQUIRES_OK(file_utils::ListFiles(configPath, &OmFileFilterFn, modelFilePaths, OM_DIR_MAX_DEPTH));

    for (auto &path : modelFilePaths) {
        OpModel opModel;
        ACL_REQUIRES_OK(ReadOpModelFromFile(path, opModel));
        OpModelDef modelDef;
        modelDef.modelPath = path;
        modelDef.opModelId = atomicModelId++;
        const auto ret = OpModelParser::ParseOpModel(opModel, modelDef);
        if (ret != ACL_SUCCESS) {
            ACL_LOG_WARN("can not parse model, errorCode = %d, model = %s, skip it", ret, modelDef.modelPath.c_str());
            continue;
        }

        configList.emplace_back(std::move(modelDef));
        ACL_LOG_INFO("Add model: %s", configList.back().modelPath.c_str());
        ACL_REQUIRES_OK(modelCache_.Add(configList.back().opModelId, opModel));
    }

    return ACL_SUCCESS;
}

aclError AclOpResourceManager::BuildOpModel(AclOp &aclOp)
{
    RT2_PROFILING_SCOPE(gert::profiling::kUnknownName, gert::profiling::kAclBuildOpModel);
    std::shared_ptr<void> modelData;
    size_t modelSize;
    auto ret = OpCompileService::GetInstance().CompileOp(aclOp, modelData, modelSize);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    // it is dynamic load
    ret = LoadModelFromSharedMem(modelData, modelSize, &aclOp, false);
    if (ret != ACL_SUCCESS) {
        ACL_LOG_INNER_ERROR("[Load][Model]load model from mem failed.");
        return ret;
    }

    ACL_LOG_INFO("Compile operator %s success", aclOp.opType.c_str());
    return ACL_SUCCESS;
}

AclOpResourceManager::~AclOpResourceManager()
{
    // note: op_model的executor的释放依赖allocator，所以先释放executor，再释放allocator
    modelCache_.CleanCachedModels();
    streamAllocators_.clear();
}

void AclOpResourceManager::HandleReleaseSourceByDevice(int32_t deviceId, aclrtDeviceState state, void *args) const
{
    (void)args;
    ACL_LOG_INFO("start to execute HandleReleaseSourceByDevice, devId:%d.", deviceId);
    if (state != ACL_RT_DEVICE_STATE_RESET_PRE) {
        ACL_LOG_INFO("it's not reset pre device callback, currently do nothing.");
        return;
    }
    (void)ge::GeExecutor::ReleaseResource();
    ACL_LOG_INFO("successfully execute HandleReleaseSourceByDevice, devId:%d.", deviceId);
}

void AclOpResourceManager::UpdateAllocatorsForOp(const void * const cacheKey,
                                                 std::shared_ptr<gert::Allocators> &allocators)
{
    const std::unique_lock<std::recursive_mutex> lk(streamAllocatorMutex_);
    streamAllocators_[cacheKey] = allocators;
}

void AclOpResourceManager::CleanAllocatorsForOp(const void * const cacheKey)
{
    if (cacheKey == nullptr) {
      return;
    }
    const std::unique_lock<std::recursive_mutex> lk(streamAllocatorMutex_);
    (void)streamAllocators_.erase(cacheKey);
}

void AclOpResourceManager::HandleReleaseSourceByStream(aclrtStream stream, aclrtStreamState state, void *args)
{
    (void)args;
    ACL_LOG_INFO("start to execute HandleReleaseSourceByStream.");
    if (state != ACL_RT_STREAM_STATE_DESTROY_PRE) {
        ACL_LOG_INFO("it's not destroy stream callback, currently do nothing.");
        return;
    }
    (void)ge::GeExecutor::ReleaseSingleOpResource(stream);
    (void)acl::Executors::RemoveExecutor(stream);
    (void)CleanRT2Executor(stream);
    (void)CleanAllocatorsForOp(stream);
    ACL_LOG_INFO("successfully execute HandleReleaseSourceByStream.");
}

aclError AclOpResourceManager::CreateRT2Executor(std::shared_ptr<gert::StreamExecutor> &streamExecutor, rtStream_t stream,
                                               const gert::ModelExecuteArg &arg, gert::ModelV2Executor *&executor)
{
    return modelCache_.CreateCachedExecutor(streamExecutor, stream, arg, executor);
}

}


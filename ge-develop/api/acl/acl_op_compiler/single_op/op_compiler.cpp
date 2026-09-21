/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "acl/acl_op_compiler.h"
#include "common/log_inner.h"
#include "compile/op_compile_processor.h"
#include "framework/common/ge_format_util.h"
#include "op_executor.h"
#include "acl_op_resource_manager.h"
#include "common/prof_api_reg.h"
#include "types/acl_op_inner.h"
#include "utils/array_utils.h"
#include "acl/acl_rt.h"
#include "platform/platform_info.h"
#include "platform/soc_spec.h"
#include "runtime/base.h"

#define NPUARCH_TO_STR(arch) std::to_string(static_cast<uint32_t>(arch))

namespace {
constexpr size_t COMPILE_OPT_SIZE = 256U;
std::map<aclCompileOpt, std::string> compileOptMap = {{ACL_PRECISION_MODE, ge::PRECISION_MODE},
                                                      {ACL_PRECISION_MODE_V2, ge::PRECISION_MODE_V2},
                                                      {ACL_AICORE_NUM, ge::AICORE_NUM},
                                                      {ACL_OP_SELECT_IMPL_MODE, ge::OP_SELECT_IMPL_MODE},
                                                      {ACL_OPTYPELIST_FOR_IMPLMODE, ge::OPTYPELIST_FOR_IMPLMODE},
                                                      {ACL_OP_DEBUG_LEVEL, ge::OP_DEBUG_LEVEL},
                                                      {ACL_DEBUG_DIR, ge::DEBUG_DIR},
                                                      {ACL_OP_COMPILER_CACHE_MODE, ge::OP_COMPILER_CACHE_MODE},
                                                      {ACL_OP_COMPILER_CACHE_DIR, ge::OP_COMPILER_CACHE_DIR},
                                                      {ACL_OP_PERFORMANCE_MODE, ge::PERFORMANCE_MODE},
                                                      {ACL_OP_JIT_COMPILE, "ge.jit_compile"},
                                                      {ACL_OP_DETERMINISTIC, "ge.deterministic"},
                                                      {ACL_CUSTOMIZE_DTYPES, ge::CUSTOMIZE_DTYPES},
                                                      {ACL_OP_PRECISION_MODE, "ge.exec.op_precision_mode"},
                                                      {ACL_ALLOW_HF32, "ge.exec.allow_hf32"},
                                                      {ACL_OP_DEBUG_OPTION, "op_debug_option"}};

// A set of NPU architecture IDs for which JIT compilation is enabled by default.
const std::set<std::string> kJitCompileEnabledByArch = {
    NPUARCH_TO_STR(NpuArch::DAV_1001), NPUARCH_TO_STR(NpuArch::DAV_2002), NPUARCH_TO_STR(NpuArch::DAV_2102),
    NPUARCH_TO_STR(NpuArch::DAV_3002), NPUARCH_TO_STR(NpuArch::DAV_3004), NPUARCH_TO_STR(NpuArch::DAV_3505),
    NPUARCH_TO_STR(NpuArch::DAV_3102), NPUARCH_TO_STR(NpuArch::DAV_5102),
};

aclError CheckInput(const char *opType, const int32_t numInputs, const aclTensorDesc *const inputDesc[],
                    const aclDataBuffer *const inputs[], const int32_t numOutputs,
                    const aclTensorDesc *const outputDesc[], aclDataBuffer *const outputs[],
                    const aclopCompileType compileFlag)
{
    ACL_REQUIRES_NON_NEGATIVE(numInputs);
    ACL_REQUIRES_NON_NEGATIVE(numOutputs);
    if (compileFlag != ACL_COMPILE_SYS && compileFlag != ACL_COMPILE_UNREGISTERED) {
        ACL_LOG_ERROR("[Check][Type]aclopCompile compile type[%d] not support", static_cast<int32_t>(compileFlag));
        acl::AclErrorLogManager::ReportInputError(
            acl::UNSUPPORTED_FEATURE_MSG, std::vector<const char *>({"feature", "reason"}),
            std::vector<const char *>({"compile type", "must be equal to ACL_COMPILE_SYS or ACL_COMPILE_UNREGISTERED"}));
        return ACL_ERROR_API_NOT_SUPPORT;
    }
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(opType);
    ACL_REQUIRES_OK(acl::array_utils::CheckPtrArray(numInputs, inputDesc));
    ACL_REQUIRES_OK(acl::array_utils::CheckPtrArray(numOutputs, outputDesc));

    ACL_REQUIRES_OK(acl::array_utils::CheckDataBufferArry(numInputs, inputs));
    ACL_REQUIRES_OK(acl::array_utils::CheckDataBufferArry(numOutputs, outputs));
    return ACL_SUCCESS;
}

aclError ConstructAclOp(acl::AclOp &aclOp, const char *opType, const int32_t numInputs,
                        const aclTensorDesc *const inputDesc[], const aclDataBuffer *const inputs[],
                        const int32_t numOutputs, const aclTensorDesc *const outputDesc[],
                        aclDataBuffer *const outputs[], const aclopAttr *attr, const aclopEngineType engineType,
                        const aclopCompileType compileFlag, const char *opPath, const acl::OpExecuteType executeType,
                        const bool isCompile)
{
    aclOp.opType.assign(opType);
    aclOp.numInputs = numInputs;
    aclOp.inputDesc = inputDesc;
    aclOp.numOutputs = numOutputs;
    aclOp.outputDesc = outputDesc;
    aclOp.inputs = inputs;
    aclOp.outputs = outputs;
    aclOp.opAttr = attr;
    aclOp.isCompile = isCompile;
    aclOp.engineType = engineType;
    aclOp.compileType = static_cast<acl::OpCompileType>(compileFlag);
    aclOp.exeucteType = executeType;
    if (compileFlag == ACL_COMPILE_UNREGISTERED) {
        if (opPath == nullptr) {
            ACL_LOG_ERROR("[Check][OpPath]opPath cannot be null while compileFlag is %d",
                          static_cast<int32_t>(compileFlag));
            acl::AclErrorLogManager::ReportInputError(acl::INVALID_NULL_POINTER_MSG,
                                                      std::vector<const char *>({"param"}),
                                                      std::vector<const char *>({"opPath"}));
            return ACL_ERROR_INVALID_PARAM;
        }
        aclOp.opPath.assign(opPath);
    }
    return ACL_SUCCESS;
}

aclError CopyOptValue(char *value, size_t length, const std::string &str)
{
    if (length < str.size() + 1U) {
        ACL_LOG_ERROR("[Check][PARAM] length[%zu] < str_size[%zu] + 1U", length, str.size());
        return ACL_ERROR_FAILURE;
    }
    const auto ret = strncpy_s(value, length, str.c_str(), str.size());
    if (ret != EOK) {
        ACL_LOG_INNER_ERROR("[Copy][Str]call strncpy_s failed, length: %zu, src size: %zu", length, str.size());
        return ACL_ERROR_FAILURE;
    }
    return ACL_SUCCESS;
}

std::string GetDefaultJitCompileValue()
{
    std::string jit_compile_value = "disable";
    constexpr uint32_t kMaxValueLen = 16U;
    char npuArch[kMaxValueLen] = {0};

    const auto ret = rtGetSocSpec("version", "NpuArch", npuArch, kMaxValueLen);
    if (ret != RT_ERROR_NONE) {
        ACL_LOG_WARN("Cannot get NpuArch, using jit_compile = [%s]", jit_compile_value.c_str());
        return jit_compile_value;
    }

    if (kJitCompileEnabledByArch.find(npuArch) != kJitCompileEnabledByArch.end()) {
        jit_compile_value = "enable";
    }
    ACL_LOG_INFO("Current NpuArch is [%s], using default jit_compile = [%s]", npuArch, jit_compile_value.c_str());
    return jit_compile_value;
}
}  // namespace

aclError aclopCompile(const char *opType, int numInputs, const aclTensorDesc *const inputDesc[], int numOutputs,
                      const aclTensorDesc *const outputDesc[], const aclopAttr *attr, aclopEngineType engineType,
                      aclopCompileType compileFlag, const char *opPath)
{
    ACL_PROFILING_REG(acl::AclProfType::AclopCompile);
    ACL_REQUIRES_NON_NEGATIVE(numInputs);
    ACL_REQUIRES_NON_NEGATIVE(numOutputs);
    if (compileFlag != ACL_COMPILE_SYS && compileFlag != ACL_COMPILE_UNREGISTERED) {
        ACL_LOG_ERROR("[Check][CompileFlag]aclopCompileType [%d] not support", static_cast<int32_t>(compileFlag));
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG, std::vector<const char *>({"param", "value",
            "reason"}), std::vector<const char *>({"compile type", std::to_string(compileFlag).c_str(), "not in range"}));
        return ACL_ERROR_API_NOT_SUPPORT;
    }
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(opType);
    ACL_REQUIRES_OK(acl::array_utils::CheckPtrArray(numInputs, inputDesc));
    ACL_REQUIRES_OK(acl::array_utils::CheckPtrArray(numOutputs, outputDesc));
    if (acl::array_utils::IsHostMemTensorDesc(numInputs, inputDesc) != ACL_SUCCESS) {
        ACL_LOG_INNER_ERROR("[Check][TensorDesc]aclopCompile ACL_MEMTYPE_HOST or "
                            "ACL_MEMTYPE_HOST_COMPILE_INDEPENDENT placeMent in inputDesc not support");
        return ACL_ERROR_API_NOT_SUPPORT;
    }
    if (acl::array_utils::IsHostMemTensorDesc(numOutputs, outputDesc) != ACL_SUCCESS) {
        ACL_LOG_INNER_ERROR("[Check][TensorDesc]aclopCompile ACL_MEMTYPE_HOST or "
                            "ACL_MEMTYPE_HOST_COMPILE_INDEPENDENT placeMent in outputDesc not support");
        return ACL_ERROR_API_NOT_SUPPORT;
    }

    ACL_LOG_INFO("start to execute aclopCompile. opType = %s, engineType = %d, compileFlag = %d", opType,
                 static_cast<int32_t>(engineType), static_cast<int32_t>(compileFlag));
    acl::AclOp aclOp;
    aclOp.opType = std::string(opType);
    aclOp.numInputs = numInputs;
    aclOp.inputDesc = inputDesc;
    aclOp.numOutputs = numOutputs;
    aclOp.outputDesc = outputDesc;
    aclOp.opAttr = attr;
    aclOp.isCompile = true;
    aclOp.engineType = engineType;
    aclOp.compileType = static_cast<acl::OpCompileType>(compileFlag);
    if (compileFlag == ACL_COMPILE_UNREGISTERED) {
        if (opPath == nullptr) {
            ACL_LOG_ERROR("[Check][CompileFlag]opPath cannot be null while compileFlag is %d", compileFlag);
            acl::AclErrorLogManager::ReportInputError(acl::INVALID_NULL_POINTER_MSG,
                std::vector<const char *>({"param"}), std::vector<const char *>({"opPath"}));
            return ACL_ERROR_INVALID_PARAM;
        }
        aclOp.opPath = std::string(opPath);
    }
    ACL_LOG_INFO("aclopCompile::aclOp = %s", aclOp.DebugString().c_str());
    return acl::OpCompileProcessor::GetInstance().OpCompile(aclOp);
}

aclError aclopCompileAndExecute(const char *opType, int numInputs, const aclTensorDesc *const inputDesc[],
                                const aclDataBuffer *const inputs[], int numOutputs,
                                const aclTensorDesc *const outputDesc[], aclDataBuffer *const outputs[],
                                const aclopAttr *attr, aclopEngineType engineType, aclopCompileType compileFlag,
                                const char *opPath, aclrtStream stream)
{
    ACL_PROFILING_REG(acl::AclProfType::AclopCompileAndExecute);
    RT2_PROFILING_SCOPE(gert::profiling::kUnknownName, gert::profiling::kAclCompileAndExecute);
    ACL_REQUIRES_OK(CheckInput(opType, numInputs, inputDesc, inputs, numOutputs, outputDesc, outputs, compileFlag));
    if (acl::array_utils::IsAllTensorEmpty(numOutputs, outputDesc)) {
        ACL_LOG_INFO("all ouput tensor are empty");
        return ACL_SUCCESS;
    }

    acl::AclOp aclOp;
    ACL_REQUIRES_OK(ConstructAclOp(aclOp, opType, static_cast<int32_t>(numInputs), inputDesc, inputs,
                                   static_cast<int32_t>(numOutputs), outputDesc, outputs, attr, engineType, compileFlag,
                                   opPath, acl::OpExecuteType::ACL_OP_EXECUTE, true));

    ACL_LOG_INFO("start to execute aclopCompileAndExecute, opType = %s, engineType = %d, compileFlag = %d", opType,
                 static_cast<int32_t>(engineType), static_cast<int32_t>(compileFlag));

    ACL_LOG_INFO("aclopCompile::aclOp = %s", aclOp.DebugString().c_str());
    auto ret = acl::OpCompileProcessor::GetInstance().OpCompile(aclOp);
    RT2_PROFILING_SCOPE_ELEMENT(aclOp.opModel.profilingIndex);
    if (ret != ACL_SUCCESS) {
        ACL_LOG_INNER_ERROR("build op model failed, result = %d", ret);
        return ret;
    }
    ACL_LOG_INFO("ExecuteAsync::aclOp = %s", aclOp.DebugString().c_str());
    aclOp.isCompile = false;
    return acl::OpExecutor::ExecuteAsync(aclOp, inputs, outputs, stream);
}

aclError aclopCompileAndExecuteV2(const char *opType, int numInputs, aclTensorDesc *inputDesc[],
                                  aclDataBuffer *inputs[], int numOutputs, aclTensorDesc *outputDesc[],
                                  aclDataBuffer *outputs[], aclopAttr *attr, aclopEngineType engineType,
                                  aclopCompileType compileFlag, const char *opPath, aclrtStream stream)
{
    ACL_PROFILING_REG(acl::AclProfType::AclopCompileAndExecuteV2);
    RT2_PROFILING_SCOPE(gert::profiling::kUnknownName, gert::profiling::kAclCompileAndExecuteV2);
    ACL_REQUIRES_OK(CheckInput(opType, numInputs, inputDesc, inputs, numOutputs, outputDesc, outputs, compileFlag));
    if (acl::array_utils::IsAllTensorEmpty(numOutputs, outputDesc)) {
        ACL_LOG_INFO("all ouput tensor are empty");
        return ACL_SUCCESS;
    }
    acl::AclOp aclOp;
    ACL_REQUIRES_OK(ConstructAclOp(aclOp, opType, static_cast<int32_t>(numInputs), inputDesc, inputs,
                                   static_cast<int32_t>(numOutputs), outputDesc, outputs, attr, engineType, compileFlag,
                                   opPath, acl::OpExecuteType::ACL_OP_EXECUTE_REFRESH_OUTPUT_ORI_SHAPE, true));

    ACL_LOG_INFO("start to execute aclopCompileAndExecuteV2, opType = %s, engineType = %d, compileFlag = %d", opType,
                 static_cast<int32_t>(engineType), static_cast<int32_t>(compileFlag));

    ACL_LOG_INFO("aclopCompileV2::aclOp = %s", aclOp.DebugString().c_str());
    auto ret = acl::OpCompileProcessor::GetInstance().OpCompile(aclOp);
    RT2_PROFILING_SCOPE_ELEMENT(aclOp.opModel.profilingIndex);
    if (ret != ACL_SUCCESS) {
        ACL_LOG_INNER_ERROR("build op model failed, result = %d", ret);
        return ret;
    }
    ACL_LOG_INFO("ExecuteAsyncV2::aclOp = %s", aclOp.DebugString().c_str());
    aclOp.isCompile = false;
    return acl::OpExecutor::ExecuteAsync(aclOp, inputs, outputs, stream);
}

aclError aclSetCompileopt(aclCompileOpt opt, const char *value)
{
    ACL_LOG_INFO("start to execute aclSetCompileopt");
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(value);

    if (opt == ACL_AUTO_TUNE_MODE) {
        ACL_LOG_INNER_ERROR("The Auto Tune function has been discarded. Please use the AOE tool for tuning.");
        return ACL_ERROR_FEATURE_UNSUPPORTED;
    }

    std::string optStr = compileOptMap.find(opt) != compileOptMap.cend() ? compileOptMap[opt] : "";
    if (optStr.empty()) {
        ACL_LOG_INNER_ERROR("[Check][Opt]Can not find any options[%d] valid in enum aclCompileOpt, "
                            "please check input option.",
                            opt);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    std::string valueStr = std::string(value);
    if (opt == ACL_OP_JIT_COMPILE) {
        valueStr = (valueStr == "enable") ? "1" : "0";
        int32_t flag = (valueStr == "1") ? 1 : 0;
        acl::OpCompileProcessor::GetInstance().SetJitCompileFlag(flag);

        std::string k = "ge.shape_generalized";
        std::string v = (valueStr == "1") ? "0" : "1";
        ACL_REQUIRES_OK(acl::OpCompileProcessor::GetInstance().SetCompileOpt(k, v));
        ACL_LOG_INFO("Set compile option [%s] and value [%s]", k.c_str(), v.c_str());
    }
    ACL_REQUIRES_OK(acl::OpCompileProcessor::GetInstance().SetCompileOpt(optStr, valueStr));
    ACL_LOG_INFO("Set compile option [%s] and value [%s]", optStr.c_str(), valueStr.c_str());
    return ACL_SUCCESS;
}

size_t aclGetCompileoptSize(aclCompileOpt opt)
{
    if (opt == ACL_OP_JIT_COMPILE) {
        (void)opt;
        return COMPILE_OPT_SIZE;
    }
    // 返回value实际长度
    std::string value;
    const std::string &optStr = compileOptMap.find(opt) != compileOptMap.cend() ? compileOptMap[opt] : "";
    if (optStr.empty()) {
        return 0UL;
    }
    (void)acl::OpCompileProcessor::GetInstance().GetCompileOpt(optStr, value);
    return value.empty() ? 0UL : (value.size() + 1UL);
}

aclError aclGetCompileopt(aclCompileOpt opt, char *value, size_t length)
{
    ACL_REQUIRES_NOT_NULL(value);
    if (opt == ACL_OP_DEBUG_OPTION) {
        std::string optValue;
        aclError ret = acl::OpCompileProcessor::GetInstance().GetCompileOpt(compileOptMap[opt], optValue);
        if (ret != ACL_SUCCESS) {
            return ACL_ERROR_API_NOT_SUPPORT;
        }
        return CopyOptValue(value, length, optValue);
    }
    if (opt == ACL_OP_JIT_COMPILE) {
        return CopyOptValue(value, length, GetDefaultJitCompileValue());
    }
    return ACL_ERROR_API_NOT_SUPPORT;
}

aclError aclopSetCompileFlag(aclOpCompileFlag flag)
{
    ACL_LOG_INFO("start to execute aclopSetCompileFlag, flag is %d", static_cast<int32_t>(flag));
    ACL_REQUIRES_OK(acl::OpCompileProcessor::GetInstance().SetCompileFlag(static_cast<int32_t>(flag)));
    return ACL_SUCCESS;
}

aclError aclGenGraphAndDumpForOp(const char *opType, int numInputs, const aclTensorDesc *const inputDesc[],
                                 const aclDataBuffer *const inputs[], int numOutputs,
                                 const aclTensorDesc *const outputDesc[], aclDataBuffer *const outputs[],
                                 const aclopAttr *attr, aclopEngineType engineType, const char *graphDumpPath,
                                 const aclGraphDumpOption *graphDumpOpt)
{
    ACL_PROFILING_REG(acl::AclProfType::AclGenGraphAndDumpForOp);
    ACL_LOG_INFO("start to execute aclGenGraphAndDumpForOp");
    if (graphDumpOpt != nullptr) {
        ACL_LOG_ERROR("[Check][PARAM]graphDumpOpt only support nullptr currently");
        const std::string errMsg = acl::AclErrorLogManager::FormatStr("only support nullptr currently");
        const char_t *argList[] = {"param", "value", "reason"};
        const char_t *argVal[] = {"dstBatchPicDescs height", "not nullptr", errMsg.c_str()};
        acl::AclErrorLogManager::ReportInputErrorWithChar(acl::INVALID_PARAM_MSG, argList, argVal, 3U);
        return ACL_ERROR_INVALID_PARAM;
    }
    ACL_REQUIRES_NON_NEGATIVE(numInputs);
    ACL_REQUIRES_NON_NEGATIVE(numOutputs);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(opType);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(graphDumpPath);
    ACL_REQUIRES_OK(acl::array_utils::CheckPtrArray(numInputs, inputDesc));
    ACL_REQUIRES_OK(acl::array_utils::CheckPtrArray(numOutputs, outputDesc));
    if (acl::array_utils::IsAllTensorEmpty(numOutputs, outputDesc)) {
        ACL_LOG_INFO("all ouput tensor are empty");
        return ACL_SUCCESS;
    }

    ACL_REQUIRES_OK(acl::array_utils::CheckPtrArray(numInputs, inputs));
    ACL_REQUIRES_OK(acl::array_utils::CheckPtrArray(numOutputs, outputs));

    acl::AclOp aclOp;
    aclOp.opType.assign(opType);
    aclOp.numInputs = numInputs;
    aclOp.inputDesc = inputDesc;
    aclOp.numOutputs = numOutputs;
    aclOp.outputDesc = outputDesc;
    aclOp.inputs = inputs;
    aclOp.outputs = outputs;
    aclOp.opAttr = attr;
    aclOp.isCompile = true;
    aclOp.engineType = engineType;

    ACL_LOG_INFO("aclopCompile::aclOp = %s", aclOp.DebugString().c_str());
    return acl::OpCompileProcessor::GetInstance().OpCompileAndDump(aclOp, graphDumpPath, graphDumpOpt);
}

aclGraphDumpOption *aclCreateGraphDumpOpt()
{
    return new (std::nothrow) aclGraphDumpOption();
}

aclError aclDestroyGraphDumpOpt(const aclGraphDumpOption *graphDumpOpt)
{
    if (graphDumpOpt == nullptr) {
        return ACL_ERROR_INVALID_PARAM;
    }

    delete graphDumpOpt;
    return ACL_SUCCESS;
}

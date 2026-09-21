/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "local_compiler.h"
#include <iomanip>
#include "framework/common/util.h"
#include "ge/ge_api.h"
#include "op_compile_service.h"
#include "error_codes_inner.h"
#include "common/ge_types.h"
#include "op_compile_processor.h"
#include "graph/utils/graph_utils.h"
#include "nlohmann/json.hpp"
#include "graph/ge_context.h"
#include "graph/ge_local_context.h"

namespace {
aclError GeFinalizeFunc()
{
    auto geRet = ge::GEFinalize();
    if (geRet != ge::SUCCESS) {
        ACL_LOG_CALL_ERROR("[Finalize][Ge]ge finalize failed. ge result = %u", geRet);
        return ACL_GET_ERRCODE_GE(geRet);
    }
    return ACL_SUCCESS;
}

void SetThreadCompileOpts(const std::map<std::string, std::string> &options)
{
    // set thread local global option and clear session and graph options to avoid interference of graph mode options
    ge::GetThreadLocalContext().SetGlobalOption(options);
    ge::GetThreadLocalContext().SetSessionOption(std::map<std::string, std::string>{});
    ge::GetThreadLocalContext().SetGraphOption(std::map<std::string, std::string>{});
    return;
}
}

namespace acl {
using Json = nlohmann::json;

std::atomic_int LocalCompiler::counter_;

void SetOptionNameMap()
{
    Json optionNameMap;
    optionNameMap.emplace(ge::PRECISION_MODE, "ACL_PRECISION_MODE");
    optionNameMap.emplace(ge::PRECISION_MODE_V2, "ACL_PRECISION_MODE_V2");
    optionNameMap.emplace(ge::AICORE_NUM, "ACL_AICORE_NUM");
    optionNameMap.emplace(ge::AUTO_TUNE_MODE, "ACL_AUTO_TUNE_MODE");
    optionNameMap.emplace(ge::OP_SELECT_IMPL_MODE, "ACL_OP_SELECT_IMPL_MODE");
    optionNameMap.emplace(ge::OPTYPELIST_FOR_IMPLMODE, "ACL_OPTYPELIST_FOR_IMPLMODE");
    optionNameMap.emplace(ge::OP_DEBUG_LEVEL, "ACL_OP_DEBUG_LEVEL");
    optionNameMap.emplace(ge::DEBUG_DIR, "ACL_DEBUG_DIR");
    optionNameMap.emplace(ge::OP_COMPILER_CACHE_MODE, "ACL_OP_COMPILER_CACHE_MODE");
    optionNameMap.emplace(ge::OP_COMPILER_CACHE_DIR, "ACL_OP_COMPILER_CACHE_DIR");
    optionNameMap.emplace(ge::PERFORMANCE_MODE, "ACL_OP_PERFORMANCE_MODE");
    optionNameMap.emplace("ge.jit_compile", "ACL_OP_JIT_COMPILE");
    optionNameMap.emplace("ge.deterministic", "ACL_OP_DETERMINISTIC");
    optionNameMap.emplace(ge::CUSTOMIZE_DTYPES, "ACL_CUSTOMIZE_DTYPES");
    optionNameMap.emplace("ge.exec.op_precision_mode", "ACL_OP_PRECISION_MODE");
    optionNameMap.emplace("ge.exec.allow_hf32", "ACL_ALLOW_HF32");
    ge::GetContext().SetOptionNameMap(optionNameMap.dump());
}

aclError LocalCompiler::Init(const std::map<std::string, std::string> &options)
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (isInitialized_) {
        ACL_LOG_INNER_ERROR("[Check][Compiler]Compiler already initialized");
        return ACL_ERROR_REPEAT_INITIALIZE;
    }

    options_ = options;
    std::map<ge::AscendString, ge::AscendString> ascendOptions;
    for (auto it = options.begin(); it != options.end(); ++it) {
        ascendOptions[ge::AscendString(it->first.c_str())] = ge::AscendString(it->second.c_str());
    }
    auto geRet = ge::GEInitialize(ascendOptions);
    if (geRet != ge::SUCCESS) {
        ACL_LOG_CALL_ERROR("[Initialize][Ge]GEInitialize failed. ge result = %u", geRet);
        return ACL_GET_ERRCODE_GE(geRet);
    }
    SetOptionNameMap();
    SetGeFinalizeCallback(GeFinalizeFunc);
    ACL_LOG_INFO("LocalCompiler Init success.");
    isInitialized_ = true;
    return ACL_SUCCESS;
}

LocalCompiler::~LocalCompiler()
{
}

aclError LocalCompiler::DoCompile(CompileParam &param, std::shared_ptr<void> &modelData, size_t &modelSize)
{
    return OnlineCompileAndDump(param, modelData, modelSize, nullptr, nullptr);
}

aclError LocalCompiler::OnlineCompileAndDump(CompileParam &param, std::shared_ptr<void> &modelData, size_t &modelSize,
    const char_t *const graphDumpPath, const aclGraphDumpOption *const dumpOpt)
{
    aclrtContext savedContext = nullptr;
    bool needReset = (aclrtGetCurrentContext(&savedContext) == ACL_SUCCESS);
    ge::ModelBufferData bufferData;
    std::map<std::string, std::string> options;
    OpCompileProcessor::GetInstance().GetGlobalCompileOpts(options);
    // need add options_ from GEInitialize
    options.insert(options_.cbegin(), options_.cend());
    ge::GeGenerator generator;
    ge::OmgContext omgContext;
    SetThreadCompileOpts(options);
    auto geRet = generator.Initialize(options, omgContext);
    if (geRet != ge::SUCCESS) {
        ACL_LOG_CALL_ERROR("[Initialize][Ge]call ge interface generator.Initialize failed. ge result = %u", geRet);
        return ACL_GET_ERRCODE_GE(geRet);
    }
    ACL_LOG_INFO("call ge interface generator.BuildSingleOpModel");
    aclError aclRet = ACL_SUCCESS;
    if (graphDumpPath == nullptr) {
        geRet = generator.BuildSingleOpModel(param.opDesc, param.inputs, param.outputs,
            param.engineType, param.compileFlag, bufferData);
        if (geRet != ge::SUCCESS) {
            ACL_LOG_CALL_ERROR("[Build][SingleOpModel]call ge interface generator.BuildSingleOpModel failed. "
                "ge result = %u", geRet);
            aclRet = ACL_GET_ERRCODE_GE(geRet);
        }
        modelData = bufferData.data;
        modelSize = static_cast<size_t>(bufferData.length);
    } else {
        aclRet = BuildGraphAndDump(generator, param, graphDumpPath, dumpOpt);
    }
    if (needReset) {
        // need to reset context whatever success of failed
        ACL_REQUIRES_OK(aclrtSetCurrentContext(savedContext));
    }
    if (aclRet != ACL_SUCCESS) {
        ACL_LOG_ERROR("OnlineCompileAndDump failed");
        (void)generator.Finalize();
        return aclRet;
    }
    geRet = generator.Finalize();
    if (geRet != ge::SUCCESS) {
        ACL_LOG_CALL_ERROR("[Finalize][Ge]call ge interface generator.Finalize failed. ge result = %u", geRet);
        return ACL_GET_ERRCODE_GE(geRet);
    }
    ACL_LOG_INFO("OnlineCompileAndDump success");
    return ACL_SUCCESS;
}

aclError LocalCompiler::BuildGraphAndDump(ge::GeGenerator &generator, CompileParam &param,
    const char_t *const graphDumpPath, const aclGraphDumpOption *const dumpOpt) const
{
    (void)dumpOpt;
    ge::GraphStage stage = ge::GraphStage::GRAPH_STAGE_FUZZ; // DEFAULT
    ge::ComputeGraphPtr graph = nullptr;
    ge::ModelBufferData bufferData;
    auto geRet = generator.BuildSingleOpModel(param.opDesc, param.inputs, param.outputs,
        param.engineType, param.compileFlag, bufferData, stage, graph);
    if (geRet != ge::SUCCESS) {
        ACL_LOG_CALL_ERROR("[Build][SingleOpModel]call ge interface generator.BuildSingleOpModel failed. "
            "ge result = %u", geRet);
        return ACL_GET_ERRCODE_GE(geRet);
    }
    const std::string dumpPath(graphDumpPath);
    const std::string fileSuffix = ".txt";
    const std::string::size_type pos = dumpPath.rfind(fileSuffix);
    const int64_t DUMP_ALL = 1LL;
    ge::graphStatus ret;
    if ((pos != std::string::npos) && (dumpPath.substr(pos) == fileSuffix)) {
        ACL_LOG_INFO("dump ge graph by file path %s", dumpPath.c_str());
        ret = ge::GraphUtils::DumpGEGraphByPath(graph, dumpPath, DUMP_ALL); // file path
    } else {
        ACL_LOG_INFO("dump ge graph by dir path %s", dumpPath.c_str());
        static std::atomic<int64_t> atomicIdx(0);
        const auto idx =  atomicIdx.fetch_add(1);
        std::stringstream fileName;
        fileName << dumpPath << MMPA_PATH_SEPARATOR_STR << "ge_proto_" << std::setw(5) << std::setfill('0') << idx
            << "_" << param.opDesc->GetName() << ".txt";
        const std::string protoFile = fileName.str();
        ACL_LOG_INFO("dump graph name path is %s", protoFile.c_str());
        ret = ge::GraphUtils::DumpGEGraphByPath(graph, protoFile, DUMP_ALL); // dir path
    }
    if (ret != ge::SUCCESS) {
        ACL_LOG_CALL_ERROR("[CALL][GEAPI]DumpGEGraphByPath failed");
        return ret;
    }
    ACL_LOG_INFO("BuildGraphAndDump success");
    return ACL_SUCCESS;
}

aclError LocalCompiler::GenGraphAndDump(CompileParam &param, const char_t *const graphDumpPath,
    const aclGraphDumpOption *const dumpOpt)
{
    std::shared_ptr<void> modelData;
    size_t modelSize = 0;
    return OnlineCompileAndDump(param, modelData, modelSize, graphDumpPath, dumpOpt);
}

OpCompiler *LocalCompiler::CreateCompiler()
{
    return new(std::nothrow) LocalCompiler();
}

static OpCompilerRegister g_registerNativeCompiler(NATIVE_COMPILER, &LocalCompiler::CreateCompiler);
} // namespace acl

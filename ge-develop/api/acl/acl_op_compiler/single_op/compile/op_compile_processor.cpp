/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_compile_processor.h"
#include "acl_op_resource_manager.h"
#include "op_compile_service.h"
#include "ge/ge_api.h"
#include "error_codes_inner.h"
#include "common/prof_api_reg.h"
#include "common/common_inner.h"

namespace acl {
namespace {
    std::string JIT_COMPILE = "ge.jit_compile";
    aclError SetJitCompileDefaultValue()
    {
        std::map<std::string, std::string> options;
        acl::OpCompileProcessor::GetInstance().GetGlobalCompileOpts(options);
        if (options.find(JIT_COMPILE) != options.cend()) {
            return ACL_SUCCESS;
        }
        static std::string jitCompileDefaultValue {};
        if (jitCompileDefaultValue.empty()) {
            const auto length = aclGetCompileoptSize(ACL_OP_JIT_COMPILE);
            auto var = std::unique_ptr<char[]>(new (std::nothrow) char[length]);
            ACL_CHECK_MALLOC_RESULT(var);
            ACL_REQUIRES_OK(aclGetCompileopt(ACL_OP_JIT_COMPILE, var.get(), length));
            jitCompileDefaultValue = std::string(var.get()) == "enable" ? "1" : "0";
            ACL_LOG_INFO("ge.jit_compile option does not exist, get default value: %s,",
                         jitCompileDefaultValue.c_str());
        }

        if (jitCompileDefaultValue == "0") {
            acl::OpCompileProcessor::GetInstance().SetJitCompileFlag(0);
            acl::OpCompileProcessor::GetInstance().SetCompileOpt(JIT_COMPILE, jitCompileDefaultValue);
            std::string optKey = "ge.shape_generalized";
            std::string v = "1";
            acl::OpCompileProcessor::GetInstance().SetCompileOpt(optKey, v);
        }
        return ACL_SUCCESS;
    }
}
OpCompileProcessor::OpCompileProcessor()
{
    (void)Init();
}

OpCompileProcessor::~OpCompileProcessor()
{
}

aclError OpCompileProcessor::Init()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (isInit_) {
        return ACL_SUCCESS;
    }
    aclError ret = SetOption();
    if (ret != ACL_SUCCESS) {
        ACL_LOG_INNER_ERROR("[Set][Options]OpCompileProcessor init failed!");
        return ret;
    }

    isInit_ = true;
    return ACL_SUCCESS;
}

aclError OpCompileProcessor::SetOption()
{
    const char *socVersion = aclrtGetSocName();
    if (socVersion == nullptr) {
        ACL_LOG_CALL_ERROR("[Get][SocVersion]get soc version failed, runtime result = %d", ACL_ERROR_INTERNAL_ERROR);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    std::map<std::string, std::string> options = {
        { ge::SOC_VERSION, std::string(socVersion) },
        { ge::OP_SELECT_IMPL_MODE, std::string("high_precision")}
    };
    return OpCompileService::GetInstance().SetCompileStrategy(NATIVE_COMPILER, options);
}

aclError OpCompileProcessor::OpCompile(AclOp &aclOp)
{
    ACL_PROFILING_REG(acl::AclProfType::OpCompile);
    ACL_REQUIRES_TRUE(isInit_, ACL_ERROR_FAILURE, "[Init][Env]init env failed!");
    ACL_REQUIRES_OK(SetJitCompileDefaultValue());
    return AclOpResourceManager::GetInstance().GetOpModel(aclOp);
}

aclError OpCompileProcessor::OpCompileAndDump(AclOp &aclOp, const char_t *const graphDumpPath,
    const aclGraphDumpOption *const dumpOpt) const
{
    ACL_PROFILING_REG(acl::AclProfType::OpCompileAndDump);
    ACL_REQUIRES_TRUE(isInit_, ACL_ERROR_FAILURE, "[Init][Env]init env failed!");
    ACL_REQUIRES_OK(SetJitCompileDefaultValue());
    return OpCompileService::GetInstance().CompileAndDumpGraph(aclOp, graphDumpPath, dumpOpt);
}

aclError OpCompileProcessor::SetCompileOpt(std::string &opt, std::string &value)
{
    ACL_REQUIRES_TRUE(isInit_, ACL_ERROR_FAILURE, "[Init][Env]init env failed!");
    std::lock_guard<std::mutex> lock(globalCompileOptsMutex);
    globalCompileOpts[opt] = value;
    return ACL_SUCCESS;
}

aclError OpCompileProcessor::GetCompileOpt(const std::string &opt, std::string &value)
{
    ACL_REQUIRES_TRUE(isInit_, ACL_ERROR_FAILURE, "[Init][Env]init env failed!");
    std::lock_guard<std::mutex> lock(globalCompileOptsMutex);
    auto iter = globalCompileOpts.find(opt);
    if (iter == globalCompileOpts.end()) {
        return ACL_ERROR_INVALID_PARAM;
    }
    value = iter->second;
    return ACL_SUCCESS;
}

aclError OpCompileProcessor::GetGlobalCompileOpts(std::map<std::string, std::string> &currentOptions)
{
    ACL_REQUIRES_TRUE(isInit_, ACL_ERROR_FAILURE, "[Init][Env]init env failed!");
    currentOptions.clear();
    std::lock_guard<std::mutex> lock(globalCompileOptsMutex);
    currentOptions.insert(globalCompileOpts.cbegin(), globalCompileOpts.cend());
    return ACL_SUCCESS;
}

aclError OpCompileProcessor::SetCompileFlag(int32_t flag) const
{
    ACL_REQUIRES_TRUE(isInit_, ACL_ERROR_FAILURE, "[Init][Env]init env failed!");
    SetGlobalCompileFlag(flag);
    return ACL_SUCCESS;
}

aclError OpCompileProcessor::SetJitCompileFlag(int32_t flag) const
{
    ACL_REQUIRES_TRUE(isInit_, ACL_ERROR_FAILURE, "[Init][Env]init env failed!");
    SetGlobalJitCompileFlag(flag);
    return ACL_SUCCESS;
}
} // namespace acl


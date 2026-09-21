/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_compile_service.h"

namespace acl {
OpCompileService::~OpCompileService()
{
    ACL_DELETE_AND_SET_NULL(compiler_);
}

void OpCompileService::RegisterCreator(const CompileStrategy strategy, const CompilerCreator creatorFn)
{
    (void)creators_.emplace(strategy, creatorFn);
}

aclError OpCompileService::CompileOp(const acl::AclOp &op, std::shared_ptr<void> &modelData, size_t &modelSize) const
{
    if (compiler_ == nullptr) {
        ACL_LOG_DEBUG("compiler is not set");
        return ACL_ERROR_COMPILER_NOT_REGISTERED;
    }

    return compiler_->CompileOp(op, modelData, modelSize);
}

aclError OpCompileService::CompileAndDumpGraph(const acl::AclOp &op,
    const char_t *const graphDumpPath, const aclGraphDumpOption *const dumpOpt) const
{
    if (compiler_ == nullptr) {
        ACL_LOG_DEBUG("compiler is not set");
        return ACL_ERROR_COMPILER_NOT_REGISTERED;
    }

    return compiler_->CompileAndDumpGraph(op, graphDumpPath, dumpOpt);
}

aclError OpCompileService::SetCompileStrategy(const CompileStrategy strategy,
    const std::map<std::string, std::string> &options)
{
    ACL_DELETE_AND_SET_NULL(compiler_);

    ACL_LOG_INFO("Set compile strategy to [%d]", static_cast<int32_t>(strategy));
    if (strategy == NO_COMPILER) {
        return ACL_SUCCESS;
    }

    if ((strategy != NATIVE_COMPILER) && (strategy != REMOTE_COMPILER)) {
        ACL_LOG_INNER_ERROR("[Check][Strategy]The current compile strategy[%d] is invalid.",
            static_cast<int32_t>(strategy));
        return ACL_ERROR_INVALID_PARAM;
    }

    std::map<CompileStrategy, CompilerCreator>::const_iterator it = creators_.find(strategy);
    if (it == creators_.cend()) {
        ACL_LOG_INNER_ERROR("[Find][Strategy]Unsupported compile strategy, compile strategy is %d.",
            static_cast<int32_t>(strategy));
        return ACL_ERROR_COMPILER_NOT_REGISTERED;
    }

    auto newCompiler = it->second();
    ACL_CHECK_MALLOC_RESULT(newCompiler);

    const auto ret = newCompiler->Init(options);
    if (ret != ACL_SUCCESS) {
        ACL_LOG_INNER_ERROR("[Init][Compiler]Init compiler failed");
        ACL_DELETE_AND_SET_NULL(newCompiler);
        return ret;
    }

    compiler_ = newCompiler;
    return ACL_SUCCESS;
}

OpCompilerRegister::OpCompilerRegister(const CompileStrategy strategy, const CompilerCreator creatorFn)
{
    OpCompileService::GetInstance().RegisterCreator(strategy, creatorFn);
}
} // namespace acl

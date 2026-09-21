/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACL_OP_EXEC_COMPILER_OP_COMPILE_SERVICE_H
#define ACL_OP_EXEC_COMPILER_OP_COMPILE_SERVICE_H

#include "op_compiler.h"

namespace acl {
using CompilerCreator = OpCompiler *(*)();

const int32_t ACL_ERROR_COMPILER_NOT_REGISTERED = 16;

enum CompileStrategy {
    NO_COMPILER,
    NATIVE_COMPILER,
    REMOTE_COMPILER
};

class ACL_FUNC_VISIBILITY OpCompileService {
public:
    ~OpCompileService();

    static OpCompileService &GetInstance()
    {
        static OpCompileService instance;
        return instance;
    }

    void RegisterCreator(const CompileStrategy strategy, const CompilerCreator creatorFn);

    aclError SetCompileStrategy(const CompileStrategy strategy, const std::map<std::string, std::string> &options);

    aclError CompileOp(const AclOp &op, std::shared_ptr<void> &modelData, size_t &modelSize) const;

    aclError CompileAndDumpGraph(const acl::AclOp &op, const char_t *const graphDumpPath,
        const aclGraphDumpOption *const dumpOpt) const;

private:
    OpCompileService() = default;

    std::map<CompileStrategy, CompilerCreator> creators_;
    OpCompiler *compiler_ = nullptr;
};

class ACL_FUNC_VISIBILITY OpCompilerRegister {
public:
    OpCompilerRegister(const CompileStrategy strategy, const CompilerCreator creatorFn);
    ~OpCompilerRegister() = default;
};
} // namespace acl

#endif // ACL_OP_EXEC_COMPILER_OP_COMPILE_SERVICE_H

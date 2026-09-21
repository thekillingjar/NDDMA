/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACL_OP_EXEC_COMPILER_LOCAL_COMPILER_H
#define ACL_OP_EXEC_COMPILER_LOCAL_COMPILER_H

#include "op_compiler.h"

#include <mutex>
#include <atomic>

#include "framework/generator/ge_generator.h"

namespace acl {
class LocalCompiler : public OpCompiler {
public:
    LocalCompiler() = default;

    ~LocalCompiler() override;

    static OpCompiler *CreateCompiler();

    aclError Init(const std::map<std::string, std::string> &options) override;

    aclError GenGraphAndDump(CompileParam &param, const char_t *const graphDumpPath,
        const aclGraphDumpOption *const dumpOpt) override;

protected:
    aclError DoCompile(CompileParam &param, std::shared_ptr<void> &modelData, size_t &modelSize) override;

private:
    aclError OnlineCompileAndDump(CompileParam &param, std::shared_ptr<void> &modelData, size_t &modelSize,
        const char_t *const graphDumpPath, const aclGraphDumpOption *const dumpOpt);

    aclError BuildGraphAndDump(ge::GeGenerator &generator, CompileParam &param,
        const char_t *const graphDumpPath, const aclGraphDumpOption *const dumpOpt) const;

    std::map<std::string, std::string> options_;
    std::mutex mutex_;
    bool isInitialized_ = false;
    static std::atomic_int counter_;
};
} // namespace acl

#endif // ACL_OP_EXEC_COMPILER_LOCAL_COMPILER_H

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACL_OP_EXEC_COMPILER_OP_COMPILER_H
#define ACL_OP_EXEC_COMPILER_OP_COMPILER_H

#include "acl/acl_op.h"
#include "acl/acl_op_compiler.h"

#include "graph/op_desc.h"
#include "graph/ge_tensor.h"
#include "common/ge_common/ge_types.h"

#include "common/common_inner.h"
#include "types/acl_op_inner.h"
#include "types/op_model.h"

namespace acl {
struct CompileParam {
    ge::OpDescPtr opDesc;
    std::vector<ge::GeTensor> inputs;
    std::vector<ge::GeTensor> outputs;
    ge::OpEngineType engineType;
    int32_t compileFlag;
};

class OpCompiler {
public:
    OpCompiler() = default;

    virtual ~OpCompiler() = default;

    virtual aclError Init(const std::map<std::string, std::string> &options) = 0;

    aclError CompileOp(const AclOp &op, std::shared_ptr<void> &modelData, size_t &modelSize);

    aclError CompileAndDumpGraph(const AclOp &op, const char_t *const graphDumpPath,
        const aclGraphDumpOption *const dumpOpt);

protected:
    virtual aclError DoCompile(CompileParam &param, std::shared_ptr<void> &modelData, size_t &modelSize) = 0;

    virtual aclError GenGraphAndDump(CompileParam &param, const char_t *const graphDumpPath,
        const aclGraphDumpOption *const dumpOpt) = 0;

private:
    static aclError MakeCompileParam(const AclOp &op, CompileParam &param, const int32_t compileFlag);
};
} // namespace acl

#endif // ACL_OP_EXEC_COMPILER_OP_COMPILER_H

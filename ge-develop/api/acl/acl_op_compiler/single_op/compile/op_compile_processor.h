/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_COMPILE_PROCESSOR_H_
#define OP_COMPILE_PROCESSOR_H_

#include <mutex>

#include "types/acl_op_inner.h"

struct aclGraphDumpOption {
    std::string stage;
};

namespace acl {
class OpCompileProcessor {
public:
    ~OpCompileProcessor();

    static OpCompileProcessor &GetInstance()
    {
        static OpCompileProcessor instance;
        return instance;
    }

    aclError OpCompile(AclOp &aclOp);
    aclError OpCompileAndDump(AclOp &aclOp, const char_t *const graphDumpPath,
        const aclGraphDumpOption *const dumpOpt) const;
    aclError SetCompileOpt(std::string &opt, std::string &value);
    aclError GetCompileOpt(const std::string &opt, std::string &value);
    aclError GetGlobalCompileOpts(std::map<std::string, std::string> &currentOptions);
    aclError SetCompileFlag(int32_t flag) const;
    aclError SetJitCompileFlag(int32_t flag) const;

private:
    OpCompileProcessor();

    aclError SetOption();

    aclError Init();

    bool isInit_ = false;
    std::mutex mutex_;

    std::map<std::string, std::string> globalCompileOpts{};
    std::mutex globalCompileOptsMutex;
};
} // namespace acl

#endif // OP_COMPILE_PROCESSOR_H_

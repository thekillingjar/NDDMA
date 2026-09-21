/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRUNK_PROJ_OP_KERNEL_SELECTOR_H
#define TRUNK_PROJ_OP_KERNEL_SELECTOR_H

#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <climits>

#include "types/acl_op_inner.h"
#include "utils/acl_op_map.h"
#include "types/op_attr.h"

struct aclopKernelDesc {
    std::string kernelId;
    const void *stubFunc = nullptr; // no need for deallocating
    uint32_t blockDim = 0U;
    std::vector<size_t> workspaceSizes;
    std::string extendArgs;
    uint64_t timestamp = ULLONG_MAX;

    std::string opType;
    std::vector<aclTensorDesc> inputDescArr;
    std::vector<aclTensorDesc> outputDescArr;
    uint64_t seq = 0U;
    aclopAttr opAttr;
};

namespace acl {
using OpKernelDesc = aclopKernelDesc;

class OpKernelSelector {
public:
    ~OpKernelSelector() = default;
    static OpKernelSelector &GetInstance()
    {
        static OpKernelSelector instance;
        return instance;
    }

    bool HasSelectFunc(const std::string &opType) const;

    bool Register(const std::string &opType, aclopCompileFunc func);

    void Unregister(const std::string &opType);

    aclError SelectOpKernel(const AclOp &op);

    aclError GetOpKernelDesc(const AclOp &op, std::shared_ptr<OpKernelDesc> &desc);

    void SetMaxOpNum(const uint64_t maxOpNum);

private:
    OpKernelSelector();
    aclopCompileFunc GetSelectFunc(const std::string &opType);

    aclError InsertAclop2KernelDesc(const AclOp &op, const std::shared_ptr<OpKernelDesc> &desc) const;

    std::mutex mu_;
    std::map<std::string, aclopCompileFunc> selectors_;

    AclOpMap<std::shared_ptr<OpKernelDesc>> kernelDescMap_;
};
} // namespace acl

#endif // TRUNK_PROJ_OP_KERNEL_SELECTOR_H

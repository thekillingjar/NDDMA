/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRUNK_PROJ_OP_KERNEL_REGISTRY_H
#define TRUNK_PROJ_OP_KERNEL_REGISTRY_H

#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <vector>
#include <string>

#include "acl/acl_op.h"

namespace acl {
struct OpKernelRegistration {
    OpKernelRegistration() = default;
    ~OpKernelRegistration()
    {
        if (deallocator != nullptr) {
            deallocator(binData, binSize);
        }
    }

    std::string opType;
    std::string kernelId;
    std::string stubName;
    std::string kernelName;
    aclopEngineType enginetype = ACL_ENGINE_SYS;
    void *binData = nullptr;
    uint64_t binSize = 0UL;
    void (*deallocator)(void *data, size_t length) = nullptr;
    void *binHandle = nullptr;
};

class OpKernelRegistry {
public:
    ~OpKernelRegistry();

    static OpKernelRegistry &GetInstance()
    {
        static OpKernelRegistry instance;
        return instance;
    }

    const void *GetStubFunc(const std::string &opType, const std::string &kernelId);

    aclError Register(std::unique_ptr<OpKernelRegistration> &&registration);

private:
    OpKernelRegistry() = default;

    // opType, KernelId, OpKernelRegistration
    std::map<std::string, std::map<std::string, std::unique_ptr<OpKernelRegistration>>> kernels_;
    std::mutex mu_;
};
} // namespace acl

#endif // TRUNK_PROJ_OP_KERNEL_REGISTRY_H

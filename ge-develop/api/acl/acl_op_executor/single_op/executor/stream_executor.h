/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRUNK_PROJ_STREAM_EXECUTOR_H
#define TRUNK_PROJ_STREAM_EXECUTOR_H

#include <memory>
#include <map>

#include "types/acl_op_inner.h"
#include "resource_manager.h"
#include "single_op/compile/op_kernel_selector.h"
#include "op_task.h"

namespace acl {
class StreamExecutor {
public:
    ~StreamExecutor();

    aclError ExecuteAsync(const AclOp &aclOpDesc,
                          const aclDataBuffer *const *const inputs,
                          aclDataBuffer *const *const outputs);

    aclError ExecuteAsync(OpKernelDesc &kernelDesc,
                          const int32_t numInputs,
                          const aclDataBuffer *const *const inputs,
                          const int32_t numOutputs,
                          aclDataBuffer *const *const outputs);

private:
    friend class Executors;

    StreamExecutor(ResourceManager *const resourceMgr, const aclrtStream aclStream);

    aclError InitTbeTask(const OpKernelDesc &desc, const int32_t numInputs, const int32_t numOutputs, TbeOpTask &task);

    aclError AllocateWorkspaces(const std::vector<size_t> &workspaceSizes, std::vector<uintptr_t> &workspaces);

    const std::unique_ptr<ResourceManager> resMgr_;
    const aclrtStream stream_;
    std::mutex mu_;
};

class Executors {
public:
    Executors() = default;

    ~Executors() = default;

    static StreamExecutor *GetOrCreate(const aclrtContext context, const aclrtStream stream);

    static void RemoveExecutor(const aclrtStream stream);

private:
    static std::recursive_mutex mu;
    static std::map<uintptr_t, std::unique_ptr<StreamExecutor>> executors; //lint !e665
};
} // namespace acl


#endif // TRUNK_PROJ_STREAM_EXECUTOR_H

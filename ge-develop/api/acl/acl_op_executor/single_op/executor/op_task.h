/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TRUNK_PROJ_OP_TASK_H
#define TRUNK_PROJ_OP_TASK_H

#include "types/acl_op_inner.h"

namespace acl {
class OpTask {
public:
    virtual aclError ExecuteAsync(const int32_t numInputs,
                                  const aclDataBuffer *const inputs[],
                                  const int32_t numOutputs,
                                  aclDataBuffer *const outputs[],
                                  const aclrtStream stream) = 0;

    void SetArgs(std::unique_ptr<uint8_t[]> &&args, const uint32_t argSize);

protected:
    uint32_t argSize_ = 0U;
    std::unique_ptr<uint8_t[]> args_ = nullptr;
};

class TbeOpTask : public OpTask {
public:
    TbeOpTask(const void *const stubFunction, const uint32_t block);
    ~TbeOpTask() = default;

    aclError ExecuteAsync(const int32_t numInputs,
                          const aclDataBuffer *const inputs[],
                          const int32_t numOutputs,
                          aclDataBuffer *const outputs[],
                          const aclrtStream stream) override;

private:
    const void *stubFunc_;
    uint32_t blockDim_;
};
} // namespace acl
#endif // TRUNK_PROJ_OP_TASK_H

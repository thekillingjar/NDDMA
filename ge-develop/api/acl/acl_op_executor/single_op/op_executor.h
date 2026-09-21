/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACL_OP_EXEC_OP_EXECUTOR_H
#define ACL_OP_EXEC_OP_EXECUTOR_H

#include "framework/executor/ge_executor.h"

#include <memory>
#include <map>
#include <unordered_map>
#include <map>

#include "acl/acl_op.h"
#include "types/acl_op_inner.h"
#include "types/op_model.h"
#include "single_op/compile/op_kernel_selector.h"
#include "framework/runtime/model_v2_executor.h"
#include "framework/runtime/stream_executor.h"

namespace acl {
struct OpHandle {
    std::string opType;
    int32_t numInputs = 0;
    int32_t numOutputs = 0;
    OpModel opModel;
    std::mutex mutexForStatic;
    std::unordered_map<aclrtStream, ge::SingleOp *> cachedOperators;
    std::mutex mutexForDynamic;
    std::map<aclrtStream, ge::DynamicSingleOp *> cachedDynamicOperators;
    std::shared_ptr<aclopKernelDesc> kernelDesc;
    AclOp aclOp;
    bool isDynamic = false;
};


inline void FixGeDataBuffer(const aclDataBuffer *const aclBuf, const aclMemType memType, ge::DataBuffer &geBuf)
{
    geBuf.data = aclBuf->data;
    geBuf.length = aclBuf->length;
    geBuf.placement =
        ((memType == ACL_MEMTYPE_DEVICE) ? static_cast<uint32_t>(memType) : static_cast<uint32_t>(ACL_MEMTYPE_HOST));
}


class ACL_FUNC_VISIBILITY OpExecutor {
public:
    static aclError CreateOpHandle(const AclOp &aclOp, OpHandle **const handle);

    static aclError ExecuteAsync(const AclOp &aclOp,
                                  const aclDataBuffer *const inputs[],
                                  aclDataBuffer *const outputs[],
                                  const aclrtStream stream);

    static aclError ExecuteAsync(OpHandle &opHandle,
                                  const aclDataBuffer *const inputs[],
                                  aclDataBuffer *const outputs[],
                                  const aclrtStream stream);

private:
    static aclError LoadSingleOp(const OpModel &modelInfo, const aclrtStream stream, ge::SingleOp **const singleOp);

    static aclError LoadDynamicSingleOp(const OpModel &modelInfo,
                                        const aclrtStream stream,
                                        ge::DynamicSingleOp **const dynamicSingleOp);

    static aclError DoExecuteAsync(ge::DynamicSingleOp *const singleOp,
                                   const AclOp &aclOp,
                                   const aclDataBuffer *const inputs[],
                                   const aclDataBuffer *const outputs[],
                                   const bool executeWithExactModel = true);

    static aclError DoExecuteAsync(ge::SingleOp *const singleOp,
                                   const AclOp &aclOp,
                                   const aclDataBuffer *const inputs[],
                                   const aclDataBuffer *const outputs[],
                                   const bool executeWithExactModel = true);

    static aclError DoExecuteRT1(const AclOp &aclOp,
                                 const aclDataBuffer *const inputs[],
                                 aclDataBuffer *const outputs[],
                                 const aclrtStream stream,
                                 const OpModel *opModelPtr,
                                 const bool isDynamic,
                                 const bool isExactModel);

    static aclError PrepareRt2Execute(const AclOp &aclOp, OpModel *opModelPtr);

    static aclError DoExecuteAsync(std::shared_ptr<gert::StreamExecutor> &streamExecutor,
                                   const AclOp &aclOp,
                                   const aclrtStream stream,
                                   const bool executeWithExactModel = true);
};
} // namespace acl

#endif // ACL_OP_EXEC_OP_EXECUTOR_H

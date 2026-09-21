/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACL_TYPES_ACL_OP_INNER_H
#define ACL_TYPES_ACL_OP_INNER_H

#include <string>

#include "acl/acl_op.h"
#include "op_attr.h"
#include "op_model.h"

namespace acl {
constexpr uint64_t DEFAULT_MAX_OPQUEUE_NUM = 20000U;

enum OpCompileType :int32_t {
    OP_COMPILE_SYS,
    OP_COMPILE_UNREGISTERED,
};

enum OpExecuteType : int32_t {
    ACL_OP_EXECUTE,
    ACL_OP_EXECUTE_V2,

    // for aclopCompileAndExecuteV2 interface, To ensure compatibility with ACL_OP_EXECUTE_V2,
    // origin shape are update after executed.
    ACL_OP_EXECUTE_REFRESH_OUTPUT_ORI_SHAPE
};

// AclOp does NOT own any of the fields
class ACL_FUNC_VISIBILITY AclOp {
public:
    AclOp() = default;
    ~AclOp();
    AclOp(const AclOp& aclOp);
    AclOp &operator=(const AclOp &aclOp) &;

    std::string opType;
    int32_t numInputs = 0;
    int32_t numOutputs = 0;
    const aclTensorDesc * const *inputDesc = nullptr;
    const aclTensorDesc * const *outputDesc = nullptr;
    const aclDataBuffer *const *inputs = nullptr;
    aclDataBuffer *const *outputs = nullptr;
    const aclopAttr *opAttr = nullptr;
    aclopEngineType engineType = ACL_ENGINE_SYS;
    std::string opPath;
    OpCompileType compileType = OP_COMPILE_SYS;
    bool isCompile = false;
    OpExecuteType exeucteType = ACL_OP_EXECUTE;
    bool isCopyConstructor = false;
    bool isMatched = false;
    bool isDynamic = false;
    OpModel opModel;
    std::string DebugString() const;
    void Init(const AclOp& aclOp);
    void BackupConst() const;
    void RecoverConst() const;
    void BackupDimsAndShapeRanges() const;
    void RecoverDimsAndShapeRanges() const;
};
} // namespace acl

#endif // ACL_TYPES_ACL_OP_INNER_H

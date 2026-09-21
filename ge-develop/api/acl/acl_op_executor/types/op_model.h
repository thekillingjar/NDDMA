/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACL_TYPES_OP_MODEL_H_
#define ACL_TYPES_OP_MODEL_H_

#include <map>
#include <string>
#include <vector>
#include <sstream>
#include <climits>

#include "graph/ge_attr_value.h"
#include "acl/acl_base.h"
#include "types/op_attr.h"
#include "framework/runtime/model_v2_executor.h"
#include "framework/runtime/stream_executor.h"

namespace acl {
struct OpModel {
    OpModel() = default;

    ~OpModel() = default;

    std::shared_ptr<void> data;
    int64_t profilingIndex = -1;
    uint32_t size = 0U;
    std::string name;
    uint64_t opModelId = 0U;
    size_t isStaticModelWithFuzzCompile = 0U;
    std::string cacheKey;
    std::shared_ptr<gert::StreamExecutor> executor = nullptr;
    std::shared_ptr<std::mutex> mtx;
};

struct OpModelDef {
    std::string opType;
    uint64_t opModelId = 0U;
    std::vector<aclTensorDesc> inputDescArr;
    std::vector<aclTensorDesc> outputDescArr;
    aclopAttr opAttr;

    std::string modelPath;
    // 0: ACL_OP_COMPILE_DEFAULT mode
    // 1：ACL_OP_COMPILE_FUZZ mode but model is static
    // 2：ACL_OP_COMPILE_FUZZ mode and model is dynamic
    size_t isStaticModelWithFuzzCompile = 0U;

    // some single op input/output shape is static, but it should be dynamic as its tiling dependence
    bool isDynamicModel = false;

    std::string DebugString() const;

    uint64_t timestamp = static_cast<uint64_t>(ULLONG_MAX);
    uint64_t seq;
};

ACL_FUNC_VISIBILITY aclError ReadOpModelFromFile(const std::string &path, OpModel &opModel);
} // namespace acl

#endif // ACL_TYPES_OP_MODEL_H_

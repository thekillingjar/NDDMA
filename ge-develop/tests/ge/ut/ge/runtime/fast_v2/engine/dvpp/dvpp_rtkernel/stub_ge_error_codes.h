/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STUB_GE_ERROR_CODES_H
#define STUB_GE_ERROR_CODES_H
#include <cstdint>

namespace ge {
using graphStatus = uint32_t;
using Status = uint32_t;

const graphStatus GRAPH_FAILED = 0xFFFFFFFF;
const graphStatus GRAPH_SUCCESS = 0;
const graphStatus GRAPH_NOT_CHANGED = 1343242304;
const graphStatus GRAPH_PARAM_INVALID = 50331649;
const graphStatus GRAPH_NODE_WITHOUT_CONST_INPUT = 50331648;
const graphStatus GRAPH_NODE_NEED_REPASS = 50331647;
const graphStatus GRAPH_INVALID_IR_DEF = 50331646;
const graphStatus OP_WITHOUT_IR_DATATYPE_INFER_RULE = 50331645;

constexpr Status SUCCESS = 0U;
constexpr Status PARAM_INVALID = 103900U;
constexpr Status TIMEOUT = 103901U;
constexpr Status NOT_CONNECTED = 103902U;
constexpr Status ALREADY_CONNECTED = 103903U;
constexpr Status NOTIFY_FAILED = 103904U;
constexpr Status UNSUPPORTED = 103905U;
constexpr Status FAILED = 503900U;
}  // namespace ge
#endif // STUB_GE_ERROR_CODES_H

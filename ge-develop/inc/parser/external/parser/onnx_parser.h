/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_EXTERNAL_PARSER_ONNX_PARSER_H_
#define INC_EXTERNAL_PARSER_ONNX_PARSER_H_

#if defined(_MSC_VER)
#ifdef FUNC_VISIBILITY
#define PARSER_FUNC_VISIBILITY _declspec(dllexport)
#else
#define PARSER_FUNC_VISIBILITY
#endif
#else
#ifdef FUNC_VISIBILITY
#define PARSER_FUNC_VISIBILITY __attribute__((visibility("default")))
#else
#define PARSER_FUNC_VISIBILITY
#endif
#endif

#include <map>
#include "graph/ascend_string.h"
#include "graph/ge_error_codes.h"
#include "graph/graph.h"

namespace ge {
PARSER_FUNC_VISIBILITY graphStatus aclgrphParseONNX(const char *model_file,
    const std::map<ge::AscendString, ge::AscendString> &parser_params, ge::Graph &graph);

PARSER_FUNC_VISIBILITY graphStatus aclgrphParseONNXFromMem(const char *buffer, size_t size,
    const std::map<ge::AscendString, ge::AscendString> &parser_params, ge::Graph &graph);
}  // namespace ge

#endif  // INC_EXTERNAL_PARSER_ONNX_PARSER_H_

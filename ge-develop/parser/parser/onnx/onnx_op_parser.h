/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_PARSER_ONNX_ONNX_OP_PARSER_H_
#define GE_PARSER_ONNX_ONNX_OP_PARSER_H_

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

#include <string>
#include <vector>
#include "framework/omg/parser/op_parser.h"
#include "graph/ge_tensor.h"
#include "graph/node.h"
#include "proto/onnx/ge_onnx.pb.h"

using Status = domi::Status;

namespace ge {
class PARSER_FUNC_VISIBILITY OnnxOpParser : public OpParser {
 public:
  /// @brief parse params
  /// @param [in] op_src        op to be parsed
  /// @param [out] op_dest      the parsed op
  /// @return SUCCESS           parse success
  /// @return FAILED            Parse failed
  Status ParseParams(const Message *op_src, ge::OpDescPtr &op_dest) override {
    (void)op_src;
    (void)op_dest;
    return domi::SUCCESS;
  }

  /// @brief parse params
  /// @param [in] op_src        op to be parsed
  /// @param [out] op_dest      the parsed op
  /// @return SUCCESS           parse success
  /// @return FAILED            Parse failed
  Status ParseParams(const Message *op_src, ge::Operator &op_dest) override {
    (void)op_src;
    (void)op_dest;
    return domi::SUCCESS;
  }

  /// @brief parsie weight
  /// @param [in] op_src        op to be parsed
  /// @param [out] op_dest      the parsed op
  /// @return SUCCESS           parsing success
  /// @return FAILED            parsing failed
  Status ParseWeights(const Message *op_src, ge::NodePtr &node) override {
    (void)op_src;
    (void)node;
    return domi::SUCCESS;
  }
};
}  // namespace ge

#endif  // GE_PARSER_ONNX_ONNX_OP_PARSER_H_

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_PARSER_ONNX_ONNX_FILE_CONSTANT_PARSER_H_
#define GE_PARSER_ONNX_ONNX_FILE_CONSTANT_PARSER_H_

#include "parser/onnx/onnx_op_parser.h"
#include "proto/onnx/ge_onnx.pb.h"

namespace ge {
class PARSER_FUNC_VISIBILITY OnnxFileConstantParser : public OnnxOpParser {
 public:
  Status ParseParams(const Message *op_src, ge::Operator &op_def) override;

 private:
  Status ParsePath(const ge::onnx::TensorProto &tensor_proto, ge::Operator &op_def) const;
  Status ParseDataType(const ge::onnx::TensorProto &tensor_proto, ge::Operator &op_def) const;
  void ParseShape(const ge::onnx::TensorProto &tensor_proto, ge::Operator &op_def) const;
  Status GetTensorProto(const ge::onnx::NodeProto &node_proto, ge::onnx::TensorProto &tensor_proto) const;
  Status SetPathAttr(const ge::onnx::StringStringEntryProto &string_proto, ge::Operator &op_def) const;
};
}  // namespace ge

#endif  // GE_PARSER_ONNX_ONNX_FILE_CONSTANT_PARSER_H_

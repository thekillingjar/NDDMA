/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_PARSER_TENSORFLOW_TENSORFLOW_CONSTANT_PARSER_H_
#define GE_PARSER_TENSORFLOW_TENSORFLOW_CONSTANT_PARSER_H_

#include "common/op_def/constant_operator.h"
#include "parser/common/data_op_parser.h"
#include "parser/tensorflow/tensorflow_op_parser.h"

namespace ge {
using domi::tensorflow::NodeDef;

class PARSER_FUNC_VISIBILITY TensorFlowConstantParser : public TensorFlowOpParser {
 public:
  Status ParseParams(const Message *op_src, ge::OpDescPtr &op_dest) override;

 private:
  static Status ParseDType(const domi::tensorflow::NodeDef *node, ConstantOperator *op);
  static Status ParseValue(const domi::tensorflow::NodeDef *node, const ge::OpDescPtr &opDesc);
};
}  // namespace ge

#endif  // GE_PARSER_TENSORFLOW_TENSORFLOW_CONSTANT_PARSER_H_

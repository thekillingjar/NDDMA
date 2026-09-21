/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "parser/tensorflow/tensorflow_no_op_parser.h"
#include "common/util.h"
#include "framework/common/debug/ge_log.h"
#include "parser/common/op_def/ir_pb_converter.h"
#include "parser/common/op_def/no_op_operator.h"
#include "parser/common/op_parser_factory.h"

using domi::TENSORFLOW;
using namespace ge::parser;

namespace ge {
Status TensorFlowNoOpParser::ParseParams(const Message *op_src, ge::OpDescPtr &op_dest) {
  const domi::tensorflow::NodeDef *node = DOMI_DYNAMIC_CAST<const domi::tensorflow::NodeDef *>(op_src);
  GE_CHECK_NOTNULL(node);
  GELOGD("TF op node name = %s, op type= %s, parse params", node->name().c_str(), node->op().c_str());
  NoOpOperator op;
  op.Name(node->name());

  return ConvertToOpDesc(op, op_dest);
}
}  // namespace ge

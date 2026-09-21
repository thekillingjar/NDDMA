/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/common/debug/ge_log.h"
#include "parser/common/op_def/var_is_initialized_op_operator.h"
#include "parser/common/op_parser_factory.h"
#include "parser/tensorflow/tensorflow_op_parser.h"
#include "parser/tensorflow/tensorflow_parser_register.h"

using namespace ge::parser;

namespace ge {
Status ParseParams(const Message *op_src, VarIsInitializedOpOperator *const op) {
  GE_CHECK_NOTNULL(op_src);
  const domi::tensorflow::NodeDef *node = ge::PtrToPtr<Message, domi::tensorflow::NodeDef>(op_src);
  GE_CHECK_NOTNULL(node);
  GELOGD("TF op node name = %s, op type= %s, parse params", node->name().c_str(), node->op().c_str());
  op->Name(node->name());

  return SUCCESS;
}

DOMI_REGISTER_TENSORFLOW_PARSER(VARISINITIALIZEDOP, VarIsInitializedOpOperator, ParseParams);

DOMI_REGISTER_TENSORFLOW_PARSER(ISVARIABLEINITIALIZED, VarIsInitializedOpOperator, ParseParams);
}  // namespace ge

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "parser/onnx/onnx_custom_parser_adapter.h"

#include "common/util.h"
#include "framework/common/debug/ge_log.h"
#include "parser/common/op_parser_factory.h"
#include "register/op_registry.h"
#include "parser/common/parser_utils.h"
#include "graph/def_types.h"

using domi::ONNX;
using domi::ParseParamByOpFunc;
using domi::ParseParamFunc;

namespace ge {
Status OnnxCustomParserAdapter::ParseParams(const Message *op_src, ge::Operator &op_dest) {
  GE_CHECK_NOTNULL(op_src);
  const ge::onnx::NodeProto *node_src = PtrToPtr<const Message, const ge::onnx::NodeProto>(op_src);
  GE_CHECK_NOTNULL(node_src);
  GELOGI("Onnx op node name = %s, op type= %s, parse params.", node_src->name().c_str(), node_src->op_type().c_str());

  ParseParamFunc custom_op_parser =
      domi::OpRegistry::Instance()->GetParseParamFunc(ParserUtils::GetOperatorType(op_dest), node_src->op_type());
  GE_CHECK_NOTNULL(custom_op_parser);
  if (custom_op_parser(op_src, op_dest) != SUCCESS) {
    GELOGE(FAILED, "[Invoke][Custom_Op_Parser] Custom parser params failed.");
    return FAILED;
  }
  return SUCCESS;
}

Status OnnxCustomParserAdapter::ParseParams(const Operator &op_src, Operator &op_dest) const {
  ParseParamByOpFunc custom_op_parser = domi::OpRegistry::Instance()->GetParseParamByOperatorFunc(
      ParserUtils::GetOperatorType(op_src));
  GE_CHECK_NOTNULL(custom_op_parser);

  if (custom_op_parser(op_src, op_dest) != SUCCESS) {
    GELOGE(FAILED, "[Invoke][Custom_Op_Parser] failed, node name:%s, type:%s",
           ParserUtils::GetOperatorName(op_src).c_str(), ParserUtils::GetOperatorType(op_src).c_str());
    return FAILED;
  }

  return SUCCESS;
}

REGISTER_CUSTOM_PARSER_ADAPTER_CREATOR(ONNX, OnnxCustomParserAdapter);
}  // namespace ge

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "parser/tensorflow/tensorflow_custom_parser_adapter.h"
#include "common/util.h"
#include "framework/common/debug/ge_log.h"
#include "parser/common/op_parser_factory.h"
#include "register/op_registry.h"
#include "parser/common/parser_utils.h"
#include "base/err_msg.h"

using domi::ParseParamFunc;
using domi::ParseParamByOpFunc;

namespace ge {
Status TensorFlowCustomParserAdapter::ParseParams(const Message *op_src, ge::OpDescPtr &op_dest) {
  GE_CHECK_NOTNULL(op_src);
  const domi::tensorflow::NodeDef *node_src = DOMI_DYNAMIC_CAST<const domi::tensorflow::NodeDef *>(op_src);
  GE_CHECK_NOTNULL(node_src);
  GELOGD("TF op node name = %s, op type= %s, parse params", node_src->name().c_str(), node_src->op().c_str());
  GE_CHECK_NOTNULL(op_dest);

  ParseParamFunc custom_op_parser = domi::OpRegistry::Instance()->GetParseParamFunc(
      op_dest->GetType(), node_src->op());
  if (custom_op_parser == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "No ParseParamFunc of node:%s exist in OpRegistry", node_src->name().c_str());
    GELOGE(FAILED, "No ParseParamFunc of node:%s exist in OpRegistry", node_src->name().c_str());
    return FAILED;
  }
  ge::Operator op = ge::OpDescUtils::CreateOperatorFromOpDesc(op_dest);
  GE_CHK_BOOL_RET_STATUS(custom_op_parser(op_src, op) == SUCCESS, FAILED, "Custom parser params failed for node:%s",
                         node_src->name().c_str());

  op.BreakConnect();
  return SUCCESS;
}

Status TensorFlowCustomParserAdapter::ParseParams(const Operator &op_src, ge::OpDescPtr &op_dest) const {
  GELOGI("Tensorflow custom op begin to parse params: op node name = %s, op type = %s.",
         ParserUtils::GetOperatorName(op_src).c_str(), ParserUtils::GetOperatorType(op_src).c_str());
  GE_CHECK_NOTNULL(op_dest);

  ParseParamByOpFunc custom_op_parser = domi::OpRegistry::Instance()->GetParseParamByOperatorFunc(
      ParserUtils::GetOperatorType(op_src));
  if (custom_op_parser == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "No ParseParamByOperatorFunc of node:%s exist in OpRegistry",
                      ParserUtils::GetOperatorName(op_src).c_str());
    GELOGE(FAILED, "No ParseParamByOperatorFunc of node:%s exist in OpRegistry",
           ParserUtils::GetOperatorName(op_src).c_str());
    return FAILED;
  }

  ge::Operator op = ge::OpDescUtils::CreateOperatorFromOpDesc(op_dest);
  GE_CHK_BOOL_RET_STATUS(custom_op_parser(op_src, op) == SUCCESS, FAILED, "Custom parser params failed or node:%s",
                         ParserUtils::GetOperatorName(op_src).c_str());
  op.BreakConnect();
  return SUCCESS;
}
}  // namespace ge

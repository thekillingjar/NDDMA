/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "parser/common/op_def/fill_operator.h"
#include "parser/tensorflow/tensorflow_parser_register.h"
#include "framework/omg/parser/parser_types.h"
#include "base/err_msg.h"

using ge::parser::ALPHA_DEFAULT_VALUE;
using ge::parser::BETA_DEFAULT_VALUE;
using ge::parser::FILL;

namespace ge {
/*
node {
    name: "model_with_buckets/bidirectional_rnn/fw/fw/BasicLSTMCellZeroState/zeros"
    op: "Fill"
    input: "model_with_buckets/bidirectional_rnn/fw/fw/BasicLSTMCellZeroState/concat"
    input: "model_with_buckets/bidirectional_rnn/fw/fw/BasicLSTMCellZeroState/zeros/Const"
    device: "/device:GPU:2"
    attr {
        key: "T"
        value {
            type: DT_FLOAT
        }
    }
}
*/
domi::Status ParseParams(const domi::tensorflow::NodeDef *node, FillOperator *op) {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(op);
  op->Name(node->name());

  domi::tensorflow::DataType data_type;
  GE_RETURN_IF_ERROR(TensorFlowUtil::ParseDataType(node, TENSORFLOW_ATTR_T, data_type));
  ge::DataType type = domi::TensorAssign::ConvertTensorflowDataType(data_type);
  CHECK_FALSE_EXEC(
      type != ge::DataType::DT_UNDEFINED,
      REPORT_INNER_ERR_MSG("E19999", "Data type %s of node %s is not supported",
                        DataType_Name(data_type).c_str(),
                        node->name().c_str());
      GELOGE(PARAM_INVALID, "Data type %s of node %s is not supported.", DataType_Name(data_type).c_str(),
          node->name().c_str());
      return PARAM_INVALID);

  op->DataType(type);

  op->Alpha(ge::parser::ALPHA_DEFAULT_VALUE);
  op->Beta(ge::parser::BETA_DEFAULT_VALUE);

  return domi::SUCCESS;
}

DOMI_REGISTER_TENSORFLOW_PARSER(FILL, FillOperator, ParseParams);
}  // namespace ge

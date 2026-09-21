/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "parser/tensorflow/tensorflow_ref_switch_parser.h"
#include "framework/common/debug/ge_log.h"
#include "parser/common/op_def/ir_pb_converter.h"
#include "parser/common/op_def/ref_switch_operator.h"
#include "parser/common/op_parser_factory.h"
#include "parser/common/util.h"

using domi::tensorflow::DataType;
using domi::tensorflow::DT_FLOAT;
using domi::tensorflow::AttrValue;
using domi::tensorflow::NodeDef;
using domi::TENSORFLOW;
using namespace ge::parser;

namespace ge {
// AUTO GEN PLEASE DO NOT MODIFY IT
Status TensorFlowRefSwitchParser::ParseT(const domi::tensorflow::NodeDef *node, RefSwitchOperator *op) {
  // The upper caller guarantees node is not empty.
  domi::tensorflow::AttrValue attr;

  CHECK_FALSE_EXEC(TensorFlowUtil::FindAttrValue(node, ge::ATTR_NAME_T, attr),
                   op->T(domi::TensorAssign::ConvertTensorflowDataType(DT_FLOAT));
                   return SUCCESS);

  GE_RETURN_WITH_LOG_IF_ERROR(TensorFlowUtil::CheckAttrHasType(attr, "type"), "check Attr T failed");

  const domi::tensorflow::DataType tfType = attr.type();
  const ge::DataType type = domi::TensorAssign::ConvertTensorflowDataType(tfType);
  CHECK_FALSE_EXEC(type != ge::DataType::DT_UNDEFINED,
                   REPORT_INNER_ERR_MSG("E19999", "Data type %s of node %s is not supported",
                                     DataType_Name(tfType).c_str(),
                                     node->name().c_str());
                   GELOGE(FAILED, "Data type %s of node %s is not supported.",
                          DataType_Name(tfType).c_str(), node->name().c_str());
                   return PARAM_INVALID);

  op->T(type);

  return SUCCESS;
}

Status TensorFlowRefSwitchParser::ParseParams(const Message *op_src, ge::OpDescPtr &op_dest) {
  GE_CHECK_NOTNULL(op_src);
  const NodeDef *node = DOMI_DYNAMIC_CAST<const NodeDef *>(op_src);
  GE_CHECK_NOTNULL(node);

  RefSwitchOperator op;
  op.Name(node->name());

  GELOGI("RefSwitch Op %s ParseParams Begin.", node->name().c_str());

  GE_RETURN_WITH_LOG_IF_ERROR(ParseT(node, &op), "Parse T for node %s failed.", node->name().c_str());

  Status status = ConvertToOpDesc(op, op_dest);

  return status;
}

REGISTER_OP_PARSER_CREATOR(TENSORFLOW, REFSWITCH, TensorFlowRefSwitchParser);
}  // namespace ge

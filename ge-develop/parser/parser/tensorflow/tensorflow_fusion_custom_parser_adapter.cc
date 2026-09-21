/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "parser/tensorflow/tensorflow_fusion_custom_parser_adapter.h"
#include "common/util.h"
#include "framework/common/debug/ge_log.h"
#include "parser/common/op_parser_factory.h"
#include "register/op_registry.h"
#include "base/err_msg.h"

using domi::FusionParseParamFunc;
using domi::FusionParseParamByOpFunc;

namespace ge {
Status TensorFlowFusionCustomParserAdapter::ParseParams(const vector<const NodeDef *> &v_input_const,
                                                        ge::NodePtr &node) const {
  GE_CHECK_NOTNULL(node);
  auto op_dest = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_dest);

  std::vector<const google::protobuf::Message *> inside_nodes;
  for (auto inside_node : v_input_const) {
    GE_CHECK_NOTNULL(inside_node);
    const google::protobuf::Message *node_src = dynamic_cast<const google::protobuf::Message *>(inside_node);
    inside_nodes.push_back(node_src);
  }
  std::string ori_type = op_dest->GetType();
  (void)ge::AttrUtils::GetStr(node->GetOpDesc(), ge::ATTR_NAME_FUSIONOP_ORIGINAL_TYPE, ori_type);
  FusionParseParamFunc
      custom_op_parser = domi::OpRegistry::Instance()->GetFusionParseParamFunc(op_dest->GetType(), ori_type);
  if (custom_op_parser == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "No FusionParseParamFunc of node:%s(%s) exist in OpRegistry",
                      node->GetName().c_str(), node->GetType().c_str());
    GELOGE(FAILED, "No FusionParseParamFunc of node:%s(%s) exist in OpRegistry",
           node->GetName().c_str(), node->GetType().c_str());
    return FAILED;
  }
  GELOGI("Get fusion parser succ, node: %s.", node->GetName().c_str());
  ge::Operator op = ge::OpDescUtils::CreateOperatorFromOpDesc(op_dest);
  GE_CHK_BOOL_RET_STATUS(custom_op_parser(inside_nodes, op) == SUCCESS, FAILED,
                         "Custom parse params failed for node:%s(%s)",
                         node->GetName().c_str(), node->GetType().c_str());

  op.BreakConnect();
  GELOGI("Run fusion parser succ, node: %s.", node->GetName().c_str());
  return SUCCESS;
}

Status TensorFlowFusionCustomParserAdapter::ParseParams(const std::vector<ge::Operator> &v_input_const,
                                                        ge::NodePtr &node) const {
  GE_CHECK_NOTNULL(node);
  auto op_dest = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_dest);

  GELOGI("Custom fusion begin to parse params, node: %s.", node->GetName().c_str());
  std::string ori_type = op_dest->GetType();
  (void)ge::AttrUtils::GetStr(node->GetOpDesc(), ge::ATTR_NAME_FUSIONOP_ORIGINAL_TYPE, ori_type);
  FusionParseParamByOpFunc
      custom_op_parser = domi::OpRegistry::Instance()->GetFusionParseParamByOpFunc(op_dest->GetType(), ori_type);
  if (custom_op_parser == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "No FusionParseParamByOpFunc of node:%s(%s) exist in OpRegistry",
                      node->GetName().c_str(), node->GetType().c_str());
    GELOGE(FAILED, "No FusionParseParamByOpFunc of node:%s(%s) exist in OpRegistry",
           node->GetName().c_str(), node->GetType().c_str());
    return FAILED;
  }
  ge::Operator op = ge::OpDescUtils::CreateOperatorFromOpDesc(op_dest);
  GE_CHK_BOOL_RET_STATUS(custom_op_parser(v_input_const, op) == SUCCESS, FAILED,
                         "Custom parser params failedfor node:%s(%s)",
                         node->GetName().c_str(), node->GetType().c_str());

  for (const auto &op_src : v_input_const) {
    op_src.BreakConnect();
  }
  op.BreakConnect();
  GELOGI("Run fusion parser succ, node: %s.", node->GetName().c_str());
  return SUCCESS;
}
}  // namespace ge

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "parser/tensorflow/tensorflow_enter_parser.h"
#include "framework/common/debug/ge_log.h"
#include "common/util.h"
#include "base/err_msg.h"
#include "graph/debug/ge_attr_define.h"
#include "parser/common/op_parser_factory.h"
#include "framework/omg/parser/parser_types.h"

using domi::TENSORFLOW;
using ge::parser::ENTER;
using ge::parser::REFENTER;

namespace ge {
Status TensorFlowEnterParser::ParseParams(const Message *op_src, ge::OpDescPtr &op_desc) {
  GE_CHECK_NOTNULL(op_src);
  GE_CHECK_NOTNULL(op_desc);
  const std::string name = op_desc->GetName();

  const domi::tensorflow::NodeDef *node = PtrToPtr<const Message, const domi::tensorflow::NodeDef>(op_src);
  domi::tensorflow::AttrValue str_attr;
  if (!TensorFlowUtil::FindAttrValue(node, ENTER_ATTR_FRAME_NAME, str_attr)) {
    REPORT_INNER_ERR_MSG("E19999", "In NodeDef:%s attr:%s does not exist, check invalid",
                      name.c_str(), ENTER_ATTR_FRAME_NAME.c_str());
    GELOGE(FAILED, "In NodeDef %s attr [%s] does not exist.", name.c_str(), ENTER_ATTR_FRAME_NAME.c_str());
    return FAILED;
  }
  std::string frame_name = str_attr.s();
  GELOGI("Enter node: %s, attr frame_name: %s", name.c_str(), frame_name.c_str());
  if (!ge::AttrUtils::SetStr(op_desc, ENTER_ATTR_FRAME_NAME, frame_name)) {
    REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s to op:%s(%s) failed", ENTER_ATTR_FRAME_NAME.c_str(),
                      op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(FAILED, "Set attr ENTER_ATTR_FRAME_NAME fail, node: %s", name.c_str());
    return FAILED;
  }

  domi::tensorflow::AttrValue bool_attr;
  if (!TensorFlowUtil::FindAttrValue(node, ENTER_ATTR_CONSTANT_FLAG, bool_attr)) {
    REPORT_INNER_ERR_MSG("E19999", "In NodeDef:%s attr:%s does not exist, check invalid",
                      name.c_str(), ENTER_ATTR_CONSTANT_FLAG.c_str());
    GELOGE(FAILED, "In NodeDef %s attr [%s] does not exist.", name.c_str(), ENTER_ATTR_CONSTANT_FLAG.c_str());
    return FAILED;
  }
  bool is_constant = bool_attr.b();
  GELOGI("Enter node: %s, attr is_constant: %s", name.c_str(), is_constant ? "true" : "false");
  if (!ge::AttrUtils::SetBool(op_desc, ENTER_ATTR_CONSTANT_FLAG, is_constant)) {
    REPORT_INNER_ERR_MSG("E19999", "Set Attr:%s to op:%s(%s) failed", ENTER_ATTR_CONSTANT_FLAG.c_str(),
                      op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(FAILED, "Set attr ENTER_ATTR_CONSTANT_FLAG fail, node: %s", name.c_str());
    return FAILED;
  }

  return SUCCESS;
}

REGISTER_OP_PARSER_CREATOR(TENSORFLOW, ENTER, TensorFlowEnterParser);
REGISTER_OP_PARSER_CREATOR(TENSORFLOW, REFENTER, TensorFlowEnterParser);
}  // namespace ge

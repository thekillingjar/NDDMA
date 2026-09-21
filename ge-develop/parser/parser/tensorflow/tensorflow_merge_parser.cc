/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "parser/tensorflow/tensorflow_merge_parser.h"

#include "framework/common/debug/ge_log.h"
#include "common/util.h"
#include "graph/debug/ge_attr_define.h"
#include "parser/common/op_parser_factory.h"
#include "framework/omg/parser/parser_types.h"
#include "graph/def_types.h"

using domi::TENSORFLOW;
using ge::parser::MERGE;

namespace ge {
Status TensorFlowMergeParser::ParseParams(const Message *op_src, ge::OpDescPtr &op_desc) {
  GE_CHECK_NOTNULL(op_src);
  GE_CHECK_NOTNULL(op_desc);

  const domi::tensorflow::NodeDef *node = PtrToPtr<const Message, const domi::tensorflow::NodeDef>(op_src);
  domi::tensorflow::AttrValue attr_num;
  if (!(TensorFlowUtil::FindAttrValue(node, ATTR_NAME_N, attr_num))) {
    GELOGW("In NodeDef %s dynamic attr [%s] does not exist.", op_desc->GetName().c_str(), ATTR_NAME_N.c_str());
  }
  int32_t input_tensor_num = attr_num.i();

  // add dynamic input
  const graphStatus ret = op_desc->AddDynamicInputDesc("x", input_tensor_num);
  if (ret != GRAPH_SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Add Dynamic InputDesc name:x to node:%s(%s) failed",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(FAILED, "Add dynamic input:x for node:%s failed.", op_desc->GetName().c_str());
    return FAILED;
  }
  GELOGI("add dynamic input for Merge op [%s], num:%d", op_desc->GetName().c_str(), input_tensor_num);

  return SUCCESS;
}

REGISTER_OP_PARSER_CREATOR(TENSORFLOW, MERGE, TensorFlowMergeParser);
}

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "parser/common/op_def/framework_op_operator.h"
#include "framework/common/debug/ge_log.h"
#include "parser/common/op_parser_factory.h"
#include "framework/omg/parser/parser_types.h"
#include "parser/tensorflow/tensorflow_op_parser.h"
#include "base/err_msg.h"
#include "parser/tensorflow/tensorflow_parser_register.h"
#include "proto/tensorflow/tensor_shape.pb.h"

using domi::tensorflow::TensorShapeProto;
using domi::tensorflow::AttrValue;
using domi::TENSORFLOW;
using ge::parser::FRAMEWORKOP;

namespace ge {
Status ParseParams(const Message *op_src, FrameworkOpOperator *op) {
  GE_CHECK_NOTNULL(op_src);
  GE_CHECK_NOTNULL(op);
  const domi::tensorflow::NodeDef *node = static_cast<const domi::tensorflow::NodeDef *>(op_src);
  GELOGD("TF op node name = %s, op type= %s, parse params", node->name().c_str(), node->op().c_str());
  string type = node->op();

  // Parsing input / output desc in attr
  domi::tensorflow::AttrValue input_attr_value;
  domi::tensorflow::AttrValue output_attr_value;
  if (TensorFlowUtil::FindAttrValue(node, ge::ATTR_NAME_INPUT_TENSOR_DESC, input_attr_value)) {
    GE_CHK_STATUS_RET(
        TensorFlowUtil::TransTensorDescriptor(input_attr_value, op, TENSORFLOW_NORMAL_INPUT_TENSOR_FLAG, type),
        "trans input_attr_value failed, op: %s", node->name().c_str());
  } else {
    GELOGD("Frameworkop has no input tensor desc, name:%s, type:%s.", node->name().c_str(), type.c_str());
    /// _Retval constructed from inference function do not has input_tensor_dec
    /// set input tensor desc for adding input tensor desc for op desc
    if (type == "_Retval") {
      ge::GeTensorDesc tensor_desc;
      op->InputTensorDesc(tensor_desc);
    }
  }
  if (TensorFlowUtil::FindAttrValue(node, ge::ATTR_NAME_OUTPUT_TENSOR_DESC, output_attr_value)) {
    GE_CHK_STATUS_RET(
        TensorFlowUtil::TransTensorDescriptor(output_attr_value, op, TENSORFLOW_NORMAL_OUTPUT_TENSOR_FLAG, type),
        "trans output_attr_value failed, op: %s", node->name().c_str());
  } else {
    GELOGD("Frameworkop has no output tensor desc, name:%s, type:%s.", node->name().c_str(), type.c_str());
  }

  // Add index attribute, only Retval needs to be added
  domi::tensorflow::AttrValue index_attr_value;
  GE_IF_BOOL_EXEC(((type == "_Retval") && (TensorFlowUtil::FindAttrValue(node, ATTR_NAME_INDEX, index_attr_value))),
                  op->Index(index_attr_value.i()));

  domi::tensorflow::NodeDef *pkg_node = new (std::nothrow) domi::tensorflow::NodeDef();
  GE_CHECK_NOTNULL(pkg_node);

  pkg_node->CopyFrom(*node);

  domi::tensorflow::AttrValue attr_v;
  // Get the property opdef, if the property does not exist, return failure
  if (TensorFlowUtil::FindAttrValue(pkg_node, ge::ATTR_NAME_FRAMEWORK_OP_DEF, attr_v)) {
    op->TfOpDef(attr_v.s());
  } else {
    GE_CHK_BOOL_EXEC(type == "_Retval",
                     REPORT_INNER_ERR_MSG("E19999", "In NodeDef:%s Attr:opdef does not exist, check invalid",
                                        pkg_node->name().c_str());
                     GE_DELETE_NEW_SINGLE(pkg_node);
                     return PARAM_INVALID, "In NodeDef %s Attr opdef does not exist.", pkg_node->name().c_str());
  }

  pkg_node->mutable_attr()->erase(ge::ATTR_NAME_FRAMEWORK_OP_DEF);
  pkg_node->mutable_attr()->erase(ge::ATTR_NAME_OUTPUT_TENSOR_DESC);
  pkg_node->mutable_attr()->erase(ge::ATTR_NAME_INPUT_TENSOR_DESC);
  pkg_node->mutable_attr()->erase(ge::VAR_ATTR_NAME);

  // Get property func def
  domi::tensorflow::AttrValue func_attr_v;
  GE_IF_BOOL_EXEC(TensorFlowUtil::FindAttrValue(pkg_node, ge::ATTR_NAME_FRAMEWORK_FUNC_DEF, func_attr_v),
                  op->FuncDefPkg(func_attr_v.s());
                  pkg_node->mutable_attr()->erase(ge::ATTR_NAME_FRAMEWORK_FUNC_DEF));
  GELOGD("pkg_node name is %s, op is %s.", pkg_node->name().c_str(), pkg_node->op().c_str());
  if (pkg_node->op() == "DPOP") {
    pkg_node->set_op(pkg_node->name());
  }

  // Serialize nodedef into string and package as a whole
  string serialized_node;
  GE_IF_BOOL_EXEC(!pkg_node->SerializeToString(&serialized_node),
                  REPORT_INNER_ERR_MSG("E19999", "Trans NodeDef:%s(%s) to string failed",
                                    pkg_node->name().c_str(), pkg_node->op().c_str());
                  GELOGE(PARAM_INVALID, "In FrameworkOp trans NodeDef to string failed.");
                  GE_DELETE_NEW_SINGLE(pkg_node);
                  return PARAM_INVALID);

  op->NodeDefPkg(serialized_node);

  string node_def_pkg = op->GetNodeDefPkg();

  GELOGD("In FrameworkOp trans NodeDef to string success.op name : %s. nodedef_pkg [%s]", node->name().c_str(),
         node_def_pkg.c_str());

  // The framework operator of tensorflow preserves its framework type
  op->Frameworktype(TENSORFLOW);

  op->OriginalType(type);

  // Add shape attribute, only variables need to be added
  domi::tensorflow::AttrValue shape_value;
  if (TensorFlowUtil::FindAttrValue(node, VAR_ATTR_SHAPE, shape_value)) {
    vector<int64_t> shape_v;
    TensorShapeProto shape_proto = shape_value.shape();
    for (auto dim : shape_proto.dim()) {
      shape_v.push_back(dim.size());
    }
    op->AttrVector(VAR_ATTR_SHAPE, shape_v);
  }

  GE_DELETE_NEW_SINGLE(pkg_node);
  return SUCCESS;
}

DOMI_REGISTER_TENSORFLOW_PARSER(FRAMEWORKOP, FrameworkOpOperator, ParseParams);
}  // namespace ge

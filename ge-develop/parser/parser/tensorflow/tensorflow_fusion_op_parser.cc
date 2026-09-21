/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "parser/tensorflow/tensorflow_fusion_op_parser.h"
#include <memory>
#include "parser/common/acl_graph_parser_util.h"
#include "common/util.h"
#include "framework/common/debug/ge_log.h"
#include "parser/common/parser_fp16_t.h"
#include "parser/tensorflow/tensorflow_op_parser.h"
#include "register/tensor_assign.h"
#include "base/err_msg.h"

using domi::tensorflow::DataType;
using domi::tensorflow::NodeDef;

namespace ge {
#define GET_CONST_VALUE(tensor, param, index, FIELD)                                                        \
  do {                                                                                                      \
    google::protobuf::RepeatedField<FIELD> val_vec;                                                         \
    int32_t val_size = 0;                                                                                   \
    val_vec = (tensor).FIELD##_val();                                                                       \
    val_size = val_vec.size();                                                                              \
    if ((index) < val_size) {                                                                               \
      (param) = val_vec.Get(index);                                                                         \
    } else if ((tensor).has_tensor_shape()) {                                                               \
      const std::string &tensor_content = (tensor).tensor_content();                                        \
      const char *buf = tensor_content.data();                                                              \
      const FIELD *buf_v = reinterpret_cast<const FIELD *>(buf);                                            \
      if (static_cast<uint32_t>(index) >= tensor_content.length() / sizeof(FIELD)) {                        \
        REPORT_INNER_ERR_MSG("E19999", "Const data size of node:%s is smaller than index: %d, not supported!",\
                           node_def->name().c_str(), index);                                                \
        GELOGE(domi::PARAM_INVALID, "Const data size is smaller than index: %d, not supported!", index);    \
        return domi::PARAM_INVALID;                                                                         \
      }                                                                                                     \
      (param) = buf_v[index];                                                                               \
    } else {                                                                                                \
      REPORT_INNER_ERR_MSG("E19999", "Const data size of node:%s is smaller than index: %d, not supported!",  \
                         node_def->name().c_str(), index);                                                  \
      GELOGE(domi::PARAM_INVALID, "Const data size is smaller than index: %d, not supported!", index);      \
      return domi::PARAM_INVALID;                                                                           \
    }                                                                                                       \
  } while (false)

Status TensorFlowFusionOpParser::GetTensorFromNode(const NodeDef *node_def, TensorProto &tensor) {
  GE_CHECK_NOTNULL(node_def);

  string node_name = node_def->name();
  GELOGI("Convert NodeDef %s.", node_name.c_str());

  domi::tensorflow::AttrValue attr_value;
  // Check that the attribute value must exist and get the value.
  if (!TensorFlowUtil::FindAttrValue(node_def, TENSORFLOW_ATTR_VALUE, attr_value)) {
    REPORT_INNER_ERR_MSG("E19999", "In NodeDef:%s attr:%s does not exist, check invalid",
                      node_def->name().c_str(), TENSORFLOW_ATTR_VALUE.c_str());
    GELOGE(domi::PARAM_INVALID, "NodeDef %s Attr %s does not exist.", node_name.c_str(), TENSORFLOW_ATTR_VALUE.c_str());
    return domi::PARAM_INVALID;
  }
  // Check that the value attribute must be tensor.
  GE_RETURN_WITH_LOG_IF_ERROR(TensorFlowUtil::CheckAttrHasType(attr_value, TENSORFLOW_ATTR_TYPE_TENSOR),
                              "check Attr %s failed", TENSORFLOW_ATTR_VALUE.c_str());
  tensor = attr_value.tensor();
  return SUCCESS;
}

Status TensorFlowFusionOpParser::ParseParams(const std::vector<const NodeDef *> &v_input_const,
                                             NodePtr &op_dest) const {
  (void)v_input_const;
  (void)op_dest;
  return SUCCESS;
}

Status TensorFlowFusionOpParser::ParseParams(const Message *op_src, OpDescPtr &op_dest) {
  (void)op_src;
  (void)op_dest;
  return SUCCESS;
}

Status TensorFlowFusionOpParser::ParseParamFromConst(const NodeDef *node_def, int32_t &param) {
  GE_CHECK_NOTNULL(node_def);
  TensorProto tensor;
  GetTensorFromNode(node_def, tensor);
  GET_CONST_VALUE(tensor, param, 0, int);
  return SUCCESS;
}
Status TensorFlowFusionOpParser::ParseParamFromConst(const NodeDef *node_def, int32_t &param, int index) {
  GE_CHECK_NOTNULL(node_def);
  TensorProto tensor;
  GetTensorFromNode(node_def, tensor);
  GET_CONST_VALUE(tensor, param, index, int);
  return SUCCESS;
}
Status TensorFlowFusionOpParser::ParseParamFromConst(const NodeDef *node_def, float &param) {
  GE_CHECK_NOTNULL(node_def);
  TensorProto tensor;
  GetTensorFromNode(node_def, tensor);
  GET_CONST_VALUE(tensor, param, 0, float);
  return SUCCESS;
}

Status TensorFlowFusionOpParser::ParseParamFromConst(const NodeDef *node_def, float &param, int index) {
  GE_CHECK_NOTNULL(node_def);
  TensorProto tensor;
  GetTensorFromNode(node_def, tensor);
  GET_CONST_VALUE(tensor, param, index, float);
  return SUCCESS;
}

Status TensorFlowFusionOpParser::ParseHalfFromConst(const NodeDef *node_def, float &param, int index) {
  GE_CHECK_NOTNULL(node_def);
  TensorProto tensor;
  GetTensorFromNode(node_def, tensor);
  if (tensor.half_val().size() > 0) {
    auto val_vec = tensor.half_val();
    int32_t val_size = val_vec.size();
    if (index < val_size) {
      ge::parser::fp16_t fp16_value = static_cast<parser::fp16_t>(val_vec.Get(index));
      param = fp16_value.ToFloat();
    } else {
      REPORT_INNER_ERR_MSG("E19999", "Const data size:%d of node:%s <= index:%d, not supported!",
                         val_size, node_def->name().c_str(), index);
      GELOGE(domi::PARAM_INVALID, "Const data size is smaller than index:%d, not supported.", index);
      return domi::PARAM_INVALID;
    }
  } else {
    REPORT_INNER_ERR_MSG("E19999", "Node %s does not have half value, index:%d.", node_def->name().c_str(), index);
    GELOGE(domi::PARAM_INVALID, "Node %s does not have half value, index:%d.", node_def->name().c_str(), index);
    return domi::PARAM_INVALID;
  }
  return SUCCESS;
}

Status TensorFlowFusionOpParser::ParseWeightFromConst(const NodeDef *node_def, ge::GeTensorPtr &weight) {
  GE_CHECK_NOTNULL(node_def);
  TensorProto tensor;
  GE_CHK_STATUS_RET(GetTensorFromNode(node_def, tensor), "get tensor failed.");
  weight = ge::parser::MakeShared<ge::GeTensor>();
  if (weight == nullptr) {
    REPORT_INNER_ERR_MSG("E19999", "New GeTensor failed for node:%s", node_def->name().c_str());
    GELOGE(domi::PARAM_INVALID, "New GeTensor failed for node:%s", node_def->name().c_str());
    return domi::PARAM_INVALID;
  }
  domi::tensorflow::DataType data_type = tensor.dtype();
  GE_CHK_STATUS_RET(
      domi::TensorAssign::SetGeTensorDataType(domi::TensorAssign::ConvertTensorflowDataType(data_type), weight),
      "set ge tensor data type fail");
  GE_CHK_STATUS_RET(domi::TensorAssign::SetGeTensor(tensor, weight), "set ge tensor fail");
  return SUCCESS;
}
}  // namespace ge

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/auto_mapping_util.h"
#include "graph_metadef/graph/debug/ge_util.h"

namespace {
constexpr int32_t kMaxFuncRecursiveDepth = 30;
}  // namespace
namespace ge {

// Convert tensorflow property to ge property
bool AutoMappingUtil::FindAttrValue(const domi::tensorflow::NodeDef *const nodeDef, const string &attr_name,
                                    domi::tensorflow::AttrValue &attr_value) {
  if (nodeDef == nullptr) {
    GE_LOGE("nodeDef is nullptr.");
    return false;
  }
  const google::protobuf::Map<std::string, domi::tensorflow::AttrValue> &attr = nodeDef->attr();
  const google::protobuf::Map<std::string, domi::tensorflow::AttrValue>::const_iterator it = attr.find(attr_name);
  if (it != attr.end()) {
    attr_value = it->second;
    return true;
  }
  return false;
}

// Get the attribute shape of tensorflow
void AutoMappingUtil::ConvertShape(const domi::tensorflow::TensorShapeProto &shape,
                                   vector<int64_t>& shape_dims) {
  shape_dims.clear();
  if (!shape.unknown_rank()) {
    for (auto &dim : shape.dim()) {
      shape_dims.push_back(dim.size());
    }
  } else {
    shape_dims = ge::UNKNOWN_SHAPE;
  }
}

graphStatus AutoMappingUtil::ConvertTensor(const domi::tensorflow::TensorProto &tensor, ge::GeTensorPtr &weight) {
  weight = ComGraphMakeShared<ge::GeTensor>();
  if (weight == nullptr) {
    GE_LOGE("Weight is nullptr.");
    return GRAPH_FAILED;
  }
  const domi::tensorflow::DataType tf_data_type = tensor.dtype();
  const ge::DataType ge_data_type = domi::TensorAssign::ConvertTensorflowDataType(static_cast<uint32_t>(tf_data_type));
  if (domi::TensorAssign::SetGeTensorDataType(ge_data_type, weight) != domi::SUCCESS) {
    GE_LOGE("Set Ge tensor data type failed.");
    return GRAPH_FAILED;
  }
  if (domi::TensorAssign::SetGeTensor(tensor, weight) != domi::SUCCESS) {
    GE_LOGE("Set Ge tensor failed.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

void AutoMappingUtil::ConvertTensorList(const domi::tensorflow::AttrValue_ListValue &list,
                                        std::vector<ge::GeTensorPtr> &vec) {
  vec.clear();
  for (auto &tensor : list.tensor()) {
    ge::GeTensorPtr ge_tensor = nullptr;
    if (ConvertTensor(tensor, ge_tensor) != GRAPH_SUCCESS) {
      GE_LOGE("Convert tensor failed.");
      return;
    }
    vec.push_back(ge_tensor);
  }
}

void AutoMappingUtil::ConvertFunc(const domi::tensorflow::NameAttrList& tf_func,
                                  ge::NamedAttrs& ge_func, const int32_t recursive_depth) {
  if (recursive_depth >= kMaxFuncRecursiveDepth) {
    GELOGW("The call stack has exceeded the maximum recursive depth");
    return;
  }
  ge_func.SetName(tf_func.name());
  auto& attrs = tf_func.attr();
  for (auto &item : attrs) {
    ConvertValue(item.first, item.second, ge_func, recursive_depth + 1);
  }
}

void AutoMappingUtil::ConvertDataTypeList(const domi::tensorflow::AttrValue_ListValue &list,
                                          std::vector<ge::DataType> &vec) {
  vec.clear();
  for (auto &e : list.type()) {
    vec.push_back(domi::TensorAssign::ConvertTensorflowDataType(static_cast<uint32_t>(e)));
  }
}

void AutoMappingUtil::ConvertShapeList(const domi::tensorflow::AttrValue_ListValue &list,
                                       std::vector<vector<int64_t>> &vec) {
  vec.clear();
  for (const auto &e : list.shape()) {
    vector<int64_t> shape_dims;
    ConvertShape(e, shape_dims);
    vec.push_back(shape_dims);
  }
}

void AutoMappingUtil::ConvertFuncList(const domi::tensorflow::AttrValue_ListValue &list,
                                      std::vector<ge::NamedAttrs> &vec, const int32_t recursive_depth) {
  if (recursive_depth >= kMaxFuncRecursiveDepth) {
    GELOGW("The call stack has exceeded the maximum recursive depth");
    return;
  }
  vec.clear();
  for (const auto &e : list.func()) {
    ge::NamedAttrs func;
    ConvertFunc(e, func, recursive_depth + 1);
    vec.push_back(func);
  }
}

} // namespace domi

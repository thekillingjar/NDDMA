/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_PY_TYPE_MAPPER_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_PY_TYPE_MAPPER_H_

#include <unordered_map>
#include <string>
#include "utils.h"
#include "py_gen_constants.h"

namespace ge {
namespace es {
struct PythonTypeMapping {
  std::string ctypes_type;
  std::string type_annotation;
  std::string type_description;
};

class PythonValueHelper {
 public:
  explicit PythonValueHelper(const OpDescPtr &op) {
    for (const auto& ir_out: op->GetIrOutputs()) {
      if (ir_out.second == kIrOutputDynamic) {
        return ;
      }
    }
    for (const auto &ir_in : op->GetIrInputs()) {
      if (ir_in.second == kIrInputOptional) {
        input_names_to_default_[ir_in.first] = true;
      } else {
        input_names_to_default_.clear();
      }
    }
  }

  bool IsInputHasDefault(const std::string &name) const {
    return input_names_to_default_.count(name) > 0;
  }

  bool IsAnyInputHasDefault() const {
    return !input_names_to_default_.empty();
  }

 private:
  std::unordered_map<std::string, bool> input_names_to_default_;
};

class PythonTypeMapper {
 public:
  static const PythonTypeMapping &GetTypeMapping(const std::string &av_type) {
    static const std::unordered_map<std::string, PythonTypeMapping> type_map = {
        {"VT_INT", {PyGenConstants::kCtypesInt64, "int", "int value"}},
        {"VT_FLOAT", {PyGenConstants::kCtypesFloat, "float", "float value"}},
        {"VT_STRING", {PyGenConstants::kCtypesCharP, "str", "string"}},
        {"VT_BOOL", {PyGenConstants::kCtypesBool, "bool", "boolean value"}},
        {"VT_DATA_TYPE", {PyGenConstants::kCtypesInt, "DataType", "DataType enum"}},
        {"VT_LIST_INT", {PyGenConstants::kCtypesInt64, "List[int]", "list of integers"}},
        {"VT_LIST_FLOAT", {PyGenConstants::kCtypesFloat, "List[float]", "list of floats"}},
        {"VT_LIST_BOOL", {PyGenConstants::kCtypesBool, "List[bool]", "list of booleans"}},
        {"VT_TENSOR", {PyGenConstants::kEsCTensor, "Tensor", "Tensor value"}},
        {"VT_LIST_DATA_TYPE", {PyGenConstants::kCtypesInt, "List[DataType]", "list of DataType"}},
        {"VT_LIST_LIST_INT", {PyGenConstants::kCtypesInt64, "List[List[int]]", "list of list of integers"}},
        {"VT_LIST_STRING", {PyGenConstants::kCtypesCharP, "List[str]", "list of strings"}}};

    auto it = type_map.find(av_type);
    if (it == type_map.end()) {
      throw std::runtime_error("Unsupported Python type: " + av_type);
    }
    return it->second;
  }

  static std::string GetCtypesType(const TypeInfo &type_info) {
    return GetTypeMapping(type_info.av_type).ctypes_type;
  }

  static std::string GetTypeAnnotation(const TypeInfo &type_info) {
    return GetTypeMapping(type_info.av_type).type_annotation;
  }

  static std::string GetTypeDescription(const TypeInfo &type_info) {
    return GetTypeMapping(type_info.av_type).type_description;
  }
};

}  // namespace es
}  // namespace ge

#endif  // AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_EAGER_STYLE_GRAPH_BUILDER_PY_TYPE_MAPPER_H_
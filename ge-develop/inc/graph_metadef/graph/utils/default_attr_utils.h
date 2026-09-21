/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __INC_METADEF_DEFAULT_ATTR_UTILS_H
#define __INC_METADEF_DEFAULT_ATTR_UTILS_H

#include <sstream>

#include "graph/node.h"
#include "graph/utils/type_utils.h"

namespace ge {
class AttrString {
 public:
  /**
   * @brief 以字符串类型获取属性的值，对于Tensor类型属性会返回"EsMakeUnique<ge::Tensor>(ge::Tensor())"
   * @param op_desc OpDesc信息
   * @param attr_name 属性IR名称
   * @param av_type 属性IR类型
   * @return Op中的属性值
   */
  static std::string GetDefaultValueString(const OpDescPtr &op_desc, const std::string &attr_name,
                                           const std::string &av_type, const bool skip_empty = false) {
    std::unordered_map<std::string, std::function<std::string(const OpDescPtr &, const std::string &, const bool)>>
        av_types_to_default{
            {"VT_INT", GetInt},
            {"VT_FLOAT", GetFloat},
            {"VT_STRING", GetStr},
            {"VT_BOOL", GetBool},
            {"VT_DATA_TYPE", GetDataType},
            {"VT_TENSOR", GetTensor},

            {"VT_LIST_INT", GetListInt},
            {"VT_LIST_FLOAT", GetListFloat},
            {"VT_LIST_BOOL", GetListBool},
            {"VT_LIST_DATA_TYPE", GetListDataType},
            {"VT_LIST_LIST_INT", GetListListInt},
            {"VT_LIST_STRING", GetListStr},
        };
    const auto iter = av_types_to_default.find(av_type);
    if (iter == av_types_to_default.end()) {
      return "";
    }
    return iter->second(op_desc, attr_name, skip_empty);
  }

 private:
  static std::string ToString(const int64_t value, const bool skip_empty = false) {
    (void) skip_empty;
    return std::to_string(value);
  }
  static std::string ToString(const float value, const bool skip_empty = false) {
    (void) skip_empty;
    return std::to_string(value);
  }
  static std::string ToString(const std::string &value, const bool skip_empty = false) {
    if (skip_empty && value.empty()) {
      return "";
    }
    return "\"" + value + "\"";
  }
  static std::string ToString(const bool value, const bool skip_empty = false) {
    (void) skip_empty;
    return value != 0U ? "true" : "false";
  }
  static std::string ToString(const ge::DataType value, const bool skip_empty = false) {
    (void) skip_empty;
    auto dt_str = TypeUtils::DataTypeToSerialString(value);
    if (dt_str == "UNDEFINED") {
      throw std::runtime_error("Unexpected data type: " + std::to_string(static_cast<int>(value)));
    }
    return "ge::" + dt_str;
  }
  static std::string ToString(const ge::ConstGeTensorPtr &value, const bool skip_empty = false) {
    (void) skip_empty;
    if (value == nullptr) {
      return "";
    }
    return "EsMakeUnique<ge::Tensor>(ge::Tensor())";
  }
  template <typename T>
  static std::string ToString(const std::vector<T> vector_value, const bool skip_empty = false) {
    std::stringstream ss;
    std::stringstream value_ss;
    bool first = true;
    for (const auto &v : vector_value) {
      std::string value = ToString(v, skip_empty);
      if (value.empty()) {
        continue;
      }
      if (first) {
        first = false;
      } else {
        value_ss << ", ";
      }
      value_ss << value;
    }

    if (skip_empty && value_ss.str().empty()) {
      return "";
    }
    ss << "{";
    ss << value_ss.str();
    ss << "}";
    return ss.str();
  }

#define GetFunc(AttrUtilType, CppType)                                                           \
  static std::string Get##AttrUtilType(const OpDescPtr &op_desc, const std::string &attr_name, const bool skip_empty) { \
    CppType default_value;                                                                       \
    if (!AttrUtils::Get##AttrUtilType(op_desc, attr_name, default_value)) {                      \
      throw std::runtime_error("Failed to get default value of attr: " + attr_name);             \
    }                                                                                            \
    return ToString(default_value, skip_empty);                                                  \
  }
  GetFunc(Int, int64_t);
  GetFunc(Float, float);
  static std::string GetStr(const OpDescPtr &op_desc, const std::string &attr_name, const bool skip_empty) {
    const std::string *ptr = AttrUtils::GetStr(op_desc, attr_name); 
    if (ptr == nullptr) {
      throw std::runtime_error("Failed to get default value of attr: " + attr_name);
    }
    return ToString(*ptr, skip_empty);
  }
  GetFunc(Bool, bool);
  GetFunc(DataType, ge::DataType);
  GetFunc(Tensor, ge::ConstGeTensorPtr);

  GetFunc(ListInt, std::vector<int64_t>);
  GetFunc(ListFloat, std::vector<float>);
  GetFunc(ListBool, std::vector<bool>);
  GetFunc(ListDataType, std::vector<ge::DataType>);
  GetFunc(ListListInt, std::vector<std::vector<int64_t>>);
  GetFunc(ListStr, std::vector<std::string>);

#undef GetFunc
};
}  // namespace ge
#endif  // __INC_METADEF_DEFAULT_ATTR_UTILS_H

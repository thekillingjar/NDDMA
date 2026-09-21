/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/utils/enum_attr_utils.h"
#include <algorithm>

namespace ge {
void EnumAttrUtils::GetEnumAttrName(vector<string> &enum_attr_names, const string &attr_name, string &enum_attr_name,
                                    bool &is_new_attr) {
  uint32_t position;
  const auto iter = std::find(enum_attr_names.begin(), enum_attr_names.end(), attr_name);
  if (iter != enum_attr_names.end()) {
    is_new_attr = false;
    position = static_cast<uint32_t>(std::distance(enum_attr_names.begin(), iter));
  } else {
    is_new_attr = true;
    position = static_cast<uint32_t>(enum_attr_names.size());
    enum_attr_names.emplace_back(attr_name);
  }
  Encode(position, enum_attr_name);
}

void EnumAttrUtils::GetEnumAttrValue(vector<string> &enum_attr_values, const string &attr_value,
                                     int64_t &enum_attr_value) {
  const auto iter = std::find(enum_attr_values.begin(), enum_attr_values.end(), attr_value);
  if (iter != enum_attr_values.end()) {
    enum_attr_value = static_cast<int64_t>(std::distance(enum_attr_values.begin(), iter));
  } else {
    enum_attr_value = static_cast<int64_t>(enum_attr_values.size());
    enum_attr_values.emplace_back(attr_value);
  }
}

void EnumAttrUtils::GetEnumAttrValues(vector<string> &enum_attr_values, const vector<string> &attr_values,
                                      vector<int64_t> &enum_values) {
  int64_t enum_attr_value;
  for (const auto &attr_value : attr_values) {
    GetEnumAttrValue(enum_attr_values, attr_value, enum_attr_value);
    enum_values.emplace_back(enum_attr_value);
  }
}

graphStatus EnumAttrUtils::GetAttrName(const vector<string> &enum_attr_names, const vector<bool> name_use_string_values,
                                       const string &enum_attr_name, string &attr_name, bool &is_value_string) {
  if (enum_attr_name.empty()) {
    GELOGE(GRAPH_FAILED, "enum_attr_name is empty.");
    return GRAPH_FAILED;
  }
  static std::string prefix_value(kAppendNum, prefix);
  // 判断enum_attr_name字符串是否Enum化，Enum字符串以'\0'字符开始
  if (enum_attr_name.rfind(prefix_value, 0U) == 0U) {
    size_t position = 0U;
    Decode(enum_attr_name, position);
    if (position < enum_attr_names.size() && position < name_use_string_values.size()) {
      attr_name = enum_attr_names[position];
      is_value_string = name_use_string_values[position];
      return GRAPH_SUCCESS;
    } else {
      GELOGE(GRAPH_FAILED,
             "position[%zu] is not less than enum_attr_names size[%zu] or name_use_string_values size[%zu].",
             position, enum_attr_names.size(), name_use_string_values.size());
      return GRAPH_FAILED;
    }
  } else {
    attr_name = enum_attr_name;
    is_value_string = false;
    return GRAPH_SUCCESS;
  }
  return GRAPH_SUCCESS;
}

graphStatus EnumAttrUtils::GetAttrValue(const vector<string> &enum_attr_values, const int64_t enum_attr_value,
                                        string &attr_value) {
  if (static_cast<size_t>(enum_attr_value) < enum_attr_values.size()) {
    attr_value = enum_attr_values[enum_attr_value];
    return GRAPH_SUCCESS;
  } else {
    GELOGE(GRAPH_FAILED, "enum_attr_value[%lld] is not less than enum_attr_values size[%zu].",
           enum_attr_value, enum_attr_values.size());
    return GRAPH_FAILED;
  }
}

graphStatus EnumAttrUtils::GetAttrValues(const vector<string> &enum_attr_values, const vector<int64_t> &enum_values,
                                         vector<string> &attr_values) {
  string attr_value;
  for (const auto enum_attr_value : enum_values) {
    if (GetAttrValue(enum_attr_values, enum_attr_value, attr_value) == GRAPH_SUCCESS) {
      attr_values.emplace_back(attr_value);
    } else {
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

// 属性名称定义为string类型，此处编码用Assci编码， Assci码的第一位为结束符，不用；可使用127位
// 1位字符范围：[0, 126]; 两位字符范围：[0, 127^2 - 1]; N位字符的范围：[0, 127^N - 1]
void EnumAttrUtils::Encode(const uint32_t src, string &dst) {
  // 按照上述字符范围获取源数据的位数
  uint32_t src_num = static_cast<uint32_t>(log(src) / log(kMaxValueOfEachDigit)) + 1U;

  // 每个ENUM化字符串编码的前缀为'\0', 用于区分哪些字符串未做ENUM化
  dst.append(kAppendNum, prefix);
  char_t data;
  for (uint32_t i = 0U; i < src_num; i++) {
    // 获取每一位的值，取位数后会加1，防止编码中出现'\0'字符
    data = static_cast<char_t>((src / static_cast<uint32_t>(pow(kMaxValueOfEachDigit, i))) % kMaxValueOfEachDigit);
    dst.append(kAppendNum, data + 1);
  }
}

// 将127位编码的src转换成实际的数字
void EnumAttrUtils::Decode(const string &src, size_t &dst) {
  // 解码从第2位开始，第一位是标志符'\0'
  for (size_t i = 1U; i < src.size(); i++) {
    dst += (static_cast<size_t>(static_cast<uint8_t>(src[i])) - 1UL) * 
            static_cast<size_t>(pow(kMaxValueOfEachDigit, (i - 1U)));
  }
}

} // namespace ge


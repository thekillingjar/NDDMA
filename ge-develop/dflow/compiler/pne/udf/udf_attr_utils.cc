/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "udf_attr_utils.h"
#include "framework/common/debug/log.h"
#include "graph/types.h"
#include "graph_metadef/common/ge_common/util.h"

namespace ge {
const std::map<AnyValue::ValueType, UdfAttrUtils::SetAttrFunc> UdfAttrUtils::set_attr_funcs_ = {
    {AnyValue::ValueType::VT_STRING, &UdfAttrUtils::SetStringAttr},
    {AnyValue::ValueType::VT_INT, &UdfAttrUtils::SetIntAttr},
    {AnyValue::ValueType::VT_FLOAT, &UdfAttrUtils::SetFloatAttr},
    {AnyValue::ValueType::VT_BOOL, &UdfAttrUtils::SetBoolAttr},
    {AnyValue::ValueType::VT_DATA_TYPE, &UdfAttrUtils::SetDataTypeAttr},
    {AnyValue::ValueType::VT_LIST_STRING, &UdfAttrUtils::SetListStringAttr},
    {AnyValue::ValueType::VT_LIST_INT, &UdfAttrUtils::SetListIntAttr},
    {AnyValue::ValueType::VT_LIST_FLOAT, &UdfAttrUtils::SetListFloatAttr},
    {AnyValue::ValueType::VT_LIST_BOOL, &UdfAttrUtils::SetListBoolAttr},
    {AnyValue::ValueType::VT_LIST_DATA_TYPE, &UdfAttrUtils::SetListDataTypeAttr},
    {AnyValue::ValueType::VT_LIST_LIST_INT, &UdfAttrUtils::SetListListIntAttr}};

Status UdfAttrUtils::SetStringAttr(const AnyValue &value, udf::AttrValue &attr) {
  std::string attr_value;
  GE_CHK_STATUS_RET(value.GetValue(attr_value), "Get string from AnyValue failed.");
  attr.set_s(attr_value);
  return SUCCESS;
}

Status UdfAttrUtils::SetIntAttr(const AnyValue &value, udf::AttrValue &attr) {
  int64_t attr_value = 0;
  GE_CHK_STATUS_RET(value.GetValue(attr_value), "Get int64_t from AnyValue failed.");
  attr.set_i(attr_value);
  return SUCCESS;
}

Status UdfAttrUtils::SetFloatAttr(const AnyValue &value, udf::AttrValue &attr) {
  float attr_value = 0;
  GE_CHK_STATUS_RET(value.GetValue(attr_value), "Get float from AnyValue failed.");
  attr.set_f(attr_value);
  return SUCCESS;
}

Status UdfAttrUtils::SetBoolAttr(const AnyValue &value, udf::AttrValue &attr) {
  bool attr_value = false;
  GE_CHK_STATUS_RET(value.GetValue(attr_value), "Get bool from AnyValue failed.");
  attr.set_b(attr_value);
  return SUCCESS;
}

Status UdfAttrUtils::SetDataTypeAttr(const AnyValue &value, udf::AttrValue &attr) {
  DataType attr_value = DataType::DT_FLOAT;
  GE_CHK_STATUS_RET(value.GetValue(attr_value), "Get data type from AnyValue failed.");
  attr.set_type(static_cast<int32_t>(attr_value));
  return SUCCESS;
}

Status UdfAttrUtils::SetListStringAttr(const AnyValue &value, udf::AttrValue &attr) {
  std::vector<std::string> attr_value;
  GE_CHK_STATUS_RET(value.GetValue(attr_value), "Get vector<string> from AnyValue failed.");
  auto *array_attr = attr.mutable_array();
  GE_CHECK_NOTNULL(array_attr);
  for (const auto &element : attr_value) {
    array_attr->add_s(element);
  }
  return SUCCESS;
}

Status UdfAttrUtils::SetListIntAttr(const AnyValue &value, udf::AttrValue &attr) {
  std::vector<int64_t> attr_value;
  GE_CHK_STATUS_RET(value.GetValue(attr_value), "Get vector<int64_t> from AnyValue failed.");
  auto *array_attr = attr.mutable_array();
  GE_CHECK_NOTNULL(array_attr);
  for (const auto element : attr_value) {
    array_attr->add_i(element);
  }
  return SUCCESS;
}

Status UdfAttrUtils::SetListFloatAttr(const AnyValue &value, udf::AttrValue &attr) {
  std::vector<float> attr_value;
  GE_CHK_STATUS_RET(value.GetValue(attr_value), "Get vector<float> from AnyValue failed.");
  auto *array_attr = attr.mutable_array();
  GE_CHECK_NOTNULL(array_attr);
  for (const auto element : attr_value) {
    array_attr->add_f(element);
  }
  return SUCCESS;
}

Status UdfAttrUtils::SetListBoolAttr(const AnyValue &value, udf::AttrValue &attr) {
  std::vector<bool> attr_value;
  GE_CHK_STATUS_RET(value.GetValue(attr_value), "Get vector<bool> from AnyValue failed.");
  auto *array_attr = attr.mutable_array();
  GE_CHECK_NOTNULL(array_attr);
  for (const auto element : attr_value) {
    array_attr->add_b(element);
  }
  return SUCCESS;
}

Status UdfAttrUtils::SetListDataTypeAttr(const AnyValue &value, udf::AttrValue &attr) {
  std::vector<DataType> attr_value;
  GE_CHK_STATUS_RET(value.GetValue(attr_value), "Get vector<DataType> from AnyValue failed.");
  auto *array_attr = attr.mutable_array();
  GE_CHECK_NOTNULL(array_attr);
  for (const auto element : attr_value) {
    array_attr->add_type(static_cast<int32_t>(element));
  }
  return SUCCESS;
}

Status UdfAttrUtils::SetListListIntAttr(const AnyValue &value, udf::AttrValue &attr) {
  std::vector<std::vector<int64_t>> attr_value;
  GE_CHK_STATUS_RET(value.GetValue(attr_value), "Get vector<vector<int64_t>> from AnyValue failed.");
  auto *list_list_i = attr.mutable_list_list_i();
  GE_CHECK_NOTNULL(list_list_i);
  for (const auto &elements : attr_value) {
    auto *list_i = list_list_i->add_list_i();
    GE_CHECK_NOTNULL(list_i);
    for (const auto element : elements) {
      list_i->add_i(element);
    }
  }
  return SUCCESS;
}
}  // namespace ge

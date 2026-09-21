/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "get_attr_by_type.h"
#include "common/math_util.h"

namespace fe {
template <typename T>
Status GetAttrValue(const AttrInfoPtr& attrs_info, T &value) {
  if (!attrs_info->GetDefaultValueDefinedFlag()) {
    FE_LOGW("[SubGraphOpt][PreCompileOp] No default value, get attr value failed.");
    return FAILED;
  } else {
    if (attrs_info->GetDefaultValue().GetValue<T>(value) != ge::GRAPH_SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][PreCompileOp] Get default value failed.");
      return FAILED;
    }
  }
  return SUCCESS;
}

Status GetStrAttrValue(const ge::OpDesc &op_desc, const string &attr_name,
                       te::TbeAttrValue &attr_value, const AttrInfoPtr& attrs_info) {
  string str_value;
  string str_default_value;
  if (attrs_info != nullptr && !attrs_info->GetIsRequired()) {
    if (GetAttrValue(attrs_info, str_default_value)!= SUCCESS) {
      FE_LOGW("[SubGraphOpt][PreCompileOp] Op [%s] get default string attr [%s] value failed.",
              op_desc.GetName().c_str(), attr_name.c_str());
    }
  }

  if (!ge::AttrUtils::GetStr(&op_desc, attr_name, str_value)) {
    if (attrs_info == nullptr || attrs_info->GetIsRequired() || !attrs_info->GetDefaultValueDefinedFlag()) {
      REPORT_FE_ERROR("[SubGraphOpt][PreCompileOp] Op [%s] get string attr [%s] value failed.",
                      op_desc.GetName().c_str(), attr_name.c_str());
      return FAILED;
    }
    str_value = str_default_value;
  }

  attr_value = te::TbeAttrValue(attr_name, str_value);
  attr_value.SetIsDefaultValue(str_value == str_default_value);
  return SUCCESS;
}

Status GetIntAttrValue(const ge::OpDesc &op_desc, const string &attr_name,
                       te::TbeAttrValue &attr_value, const AttrInfoPtr& attrs_info) {
  int64_t int_value = 0;
  int64_t int_default_value = 0;
  if (attrs_info != nullptr && !attrs_info->GetIsRequired()) {
    if (GetAttrValue(attrs_info, int_default_value)!= SUCCESS) {
      FE_LOGW("[SubGraphOpt][PreCompileOp] Op [%s] get default int attr [%s] value failed.",
              op_desc.GetName().c_str(), attr_name.c_str());
    }
  }

  if (!ge::AttrUtils::GetInt(&op_desc, attr_name, int_value)) {
    if (attrs_info == nullptr || attrs_info->GetIsRequired() || !attrs_info->GetDefaultValueDefinedFlag()) {
      REPORT_FE_ERROR("[SubGraphOpt][PreCompileOp] Op [%s] get int attr [%s] value failed.",
                      op_desc.GetName().c_str(), attr_name.c_str());
      return FAILED;
    }
    FE_LOGD("attr[%s] not config.", attr_name.c_str());
    int_value = int_default_value;
  }

  attr_value = te::TbeAttrValue(attr_name, int_value);
  attr_value.SetIsDefaultValue(int_value == int_default_value);
  return SUCCESS;
}

Status GetFloatAttrValue(const ge::OpDesc &op_desc, const string &attr_name,
                         te::TbeAttrValue &attr_value, const AttrInfoPtr& attrs_info) {
  float float_value = 0.0;
  float float_default_value = 0.0;
  if (attrs_info != nullptr && !attrs_info->GetIsRequired()) {
    if (GetAttrValue(attrs_info, float_default_value)!= SUCCESS) {
      FE_LOGW("[SubGraphOpt][PreCompileOp] Op [%s] get default float attr [%s] value failed.",
              op_desc.GetName().c_str(), attr_name.c_str());
    }
  }

  if (!ge::AttrUtils::GetFloat(&op_desc, attr_name, float_value)) {
    if (attrs_info == nullptr || attrs_info->GetIsRequired() || !attrs_info->GetDefaultValueDefinedFlag()) {
      REPORT_FE_ERROR("[SubGraphOpt][PreCompileOp] Op [%s] get float attr [%s] value failed.",
                      op_desc.GetName().c_str(), attr_name.c_str());
      return FAILED;
    }
    float_value = float_default_value;
  }
  attr_value = te::TbeAttrValue(attr_name, float_value);
  attr_value.SetIsDefaultValue(CompareValue(float_value, float_default_value));
  return SUCCESS;
}

Status GetBoolAttrValue(const ge::OpDesc &op_desc, const string &attr_name,
                        te::TbeAttrValue &attr_value, const AttrInfoPtr& attrs_info) {
  bool bool_value = true;
  bool bool_default_value = false;
  if (attrs_info != nullptr && !attrs_info->GetIsRequired()) {
    if (GetAttrValue(attrs_info, bool_default_value)!= SUCCESS) {
      FE_LOGW("[SubGraphOpt][PreCompileOp] Op [%s] get default bool attr [%s] value failed.",
              op_desc.GetName().c_str(), attr_name.c_str());
    }
    FE_LOGD("attr[%s] bool_default_value is %d.", attr_name.c_str(), bool_default_value);
  }

  if (!ge::AttrUtils::GetBool(&op_desc, attr_name, bool_value)) {
    if (attrs_info == nullptr || attrs_info->GetIsRequired() || !attrs_info->GetDefaultValueDefinedFlag()) {
      REPORT_FE_ERROR("[SubGraphOpt][PreCompileOp] Op [%s] get bool attr [%s] value failed.",
                      op_desc.GetName().c_str(), attr_name.c_str());
      return FAILED;
    }
    bool_value = bool_default_value;
  }
  attr_value = te::TbeAttrValue(attr_name, bool_value);
  attr_value.SetIsDefaultValue(bool_value == bool_default_value);
  return SUCCESS;
}

Status GetListStrAttrValue(const ge::OpDesc &op_desc, const string &attr_name,
                           te::TbeAttrValue &attr_value, const AttrInfoPtr& attrs_info) {
  vector<string> vec_str_value;
  vector<string> vec_str_default_value;
  if (attrs_info != nullptr && !attrs_info->GetIsRequired()) {
    if (GetAttrValue(attrs_info, vec_str_default_value)!= SUCCESS) {
      FE_LOGW("[SubGraphOpt][PreCompileOp] Op [%s] get default vec_str attr [%s] value failed.",
              op_desc.GetName().c_str(), attr_name.c_str());
    }
  }

  if (!ge::AttrUtils::GetListStr(&op_desc, attr_name, vec_str_value)) {
    if (attrs_info == nullptr || attrs_info->GetIsRequired() || !attrs_info->GetDefaultValueDefinedFlag()) {
      REPORT_FE_ERROR("[SubGraphOpt][PreCompileOp] Op [%s] get vec_str attr [%s] value failed.",
                      op_desc.GetName().c_str(), attr_name.c_str());
      return FAILED;
    }
    vec_str_value.assign(vec_str_default_value.begin(), vec_str_default_value.end());
  }
  attr_value = te::TbeAttrValue(attr_name, vec_str_value);
  attr_value.SetIsDefaultValue(vec_str_value == vec_str_default_value);
  return SUCCESS;
}

Status GetListIntAttrValue(const ge::OpDesc &op_desc, const string &attr_name,
                           te::TbeAttrValue &attr_value, const AttrInfoPtr& attrs_info) {
  vector<int64_t> vec_int_value;
  vector<int64_t> vec_int_default_value;
  if (attrs_info != nullptr && !attrs_info->GetIsRequired()) {
    if (GetAttrValue(attrs_info, vec_int_default_value)!= SUCCESS) {
      FE_LOGW("[SubGraphOpt][PreCompileOp] Op [%s] get default vec_int attr [%s] value failed.",
              op_desc.GetName().c_str(), attr_name.c_str());
    }
  }

  if (!ge::AttrUtils::GetListInt(&op_desc, attr_name, vec_int_value)) {
    if (attrs_info == nullptr || attrs_info->GetIsRequired() || !attrs_info->GetDefaultValueDefinedFlag()) {
      REPORT_FE_ERROR("[SubGraphOpt][PreCompileOp] Op [%s] get vec_int attr [%s] value failed.",
                      op_desc.GetName().c_str(), attr_name.c_str());
      return FAILED;
    }
    vec_int_value.assign(vec_int_default_value.begin(), vec_int_default_value.end());
  }
  attr_value = te::TbeAttrValue(attr_name, vec_int_value);
  attr_value.SetIsDefaultValue(vec_int_value == vec_int_default_value);
  return SUCCESS;
}

Status GetListListIntAttrValue(const ge::OpDesc &op_desc, const string &attr_name,
                               te::TbeAttrValue &attr_value, const AttrInfoPtr& attrs_info) {
  vector<vector<int64_t>> vecvec_int_value;
  vector<vector<int64_t>> vecvec_int_default_value;
  if (attrs_info != nullptr && !attrs_info->GetIsRequired()) {
    if (GetAttrValue(attrs_info, vecvec_int_default_value)!= SUCCESS) {
      FE_LOGW("[SubGraphOpt][PreCompileOp] Op [%s] get default list_list_int attr [%s] value failed.",
              op_desc.GetName().c_str(), attr_name.c_str());
    }
  }

  if (!ge::AttrUtils::GetListListInt(&op_desc, attr_name, vecvec_int_value)) {
    if (attrs_info == nullptr || attrs_info->GetIsRequired() || !attrs_info->GetDefaultValueDefinedFlag()) {
      REPORT_FE_ERROR("[SubGraphOpt][PreCompileOp] Op [%s] get list_list_int attr [%s] value failed.",
                      op_desc.GetName().c_str(), attr_name.c_str());
      return FAILED;
    }
    vecvec_int_value.assign(vecvec_int_default_value.begin(), vecvec_int_default_value.end());
  }
  attr_value = te::TbeAttrValue(attr_name, vecvec_int_value);
  attr_value.SetIsDefaultValue(vecvec_int_value == vecvec_int_default_value);
  return SUCCESS;
}

Status GetListFloatAttrValue(const ge::OpDesc &op_desc, const string &attr_name,
                             te::TbeAttrValue &attr_value, const AttrInfoPtr& attrs_info) {
  vector<float> vec_float_value;
  vector<float> vec_float_default_value;
  if (attrs_info != nullptr && !attrs_info->GetIsRequired()) {
    if (GetAttrValue(attrs_info, vec_float_default_value)!= SUCCESS) {
      FE_LOGW("[SubGraphOpt][PreCompileOp] Op [%s] get default vec_float attr [%s] value failed.",
              op_desc.GetName().c_str(), attr_name.c_str());
    }
  }

  if (!ge::AttrUtils::GetListFloat(&op_desc, attr_name, vec_float_value)) {
    if (attrs_info == nullptr || attrs_info->GetIsRequired() || !attrs_info->GetDefaultValueDefinedFlag()) {
      REPORT_FE_ERROR("[SubGraphOpt][PreCompileOp] Op [%s] get vec_float attr [%s] value failed.",
                      op_desc.GetName().c_str(), attr_name.c_str());
      return FAILED;
    }
    vec_float_value.assign(vec_float_default_value.begin(), vec_float_default_value.end());
  }
  attr_value = te::TbeAttrValue(attr_name, vec_float_value);
  attr_value.SetIsDefaultValue(CompareValue(vec_float_value, vec_float_default_value));
  return SUCCESS;
}

Status GetListBoolAttrValue(const ge::OpDesc &op_desc, const string &attr_name,
                            te::TbeAttrValue &attr_value, const AttrInfoPtr& attrs_info) {
  vector<bool> vec_bool_value;
  vector<bool> vec_bool_default_value;
  if (attrs_info != nullptr && !attrs_info->GetIsRequired()) {
    if (GetAttrValue(attrs_info, vec_bool_default_value)!= SUCCESS) {
      FE_LOGW("[SubGraphOpt][PreCompileOp] Op [%s] get default vec_bool attr [%s] value failed.",
              op_desc.GetName().c_str(), attr_name.c_str());
    }
  }

  if (!ge::AttrUtils::GetListBool(&op_desc, attr_name, vec_bool_value)) {
    if (attrs_info == nullptr || attrs_info->GetIsRequired() || !attrs_info->GetDefaultValueDefinedFlag()) {
      REPORT_FE_ERROR("[SubGraphOpt][PreCompileOp] Op [%s] get vec_bool attr [%s] value failed.",
                      op_desc.GetName().c_str(), attr_name.c_str());
      return FAILED;
    }
    vec_bool_value.assign(vec_bool_default_value.begin(), vec_bool_default_value.end());
  }
  attr_value = te::TbeAttrValue(attr_name, vec_bool_value);
  attr_value.SetIsDefaultValue(vec_bool_value == vec_bool_default_value);
  return SUCCESS;
}

Status GetTensorAttrValue(const ge::OpDesc &op_desc, const string &attr_name,
                          const te::TbeAttrValue &attr_value, const AttrInfoPtr& attrs_info) {
  REPORT_FE_ERROR("[SubGraphOpt][PreCompileOp] Tensor attr [%s] value not supported in op [%s].",
                  attr_name.c_str(), op_desc.GetName().c_str());
  (void)attr_value;
  (void)attrs_info;
  return FAILED;
}

Status GetListTensorAttrValue(const ge::OpDesc &op_desc, const string &attr_name,
                              const te::TbeAttrValue &attr_value, const AttrInfoPtr& attrs_info) {
  REPORT_FE_ERROR("[SubGraphOpt][PreCompileOp] ListTensor attr [%s] value not supported in op [%s].",
                  attr_name.c_str(), op_desc.GetName().c_str());
  (void)attr_value;
  (void)attrs_info;
  return FAILED;
}

void GetStrPrivateAttrValue(const ge::OpDesc &op_desc, const ge::AnyValue &value_type,
                            te::TbeAttrValue &tbe_attr_value, const string &attr_name) {
  string value;
  value_type.GetValue<string>(value);
  if (!ge::AttrUtils::GetStr(&op_desc, attr_name, value)) {
    FE_LOGD("Op[%s, %s] Can not to get private %s from graph, and to keep default value",
            op_desc.GetTypePtr(), op_desc.GetNamePtr(), attr_name.c_str());
  }
  tbe_attr_value = te::TbeAttrValue(attr_name, value);
}

void GetIntPrivateAttrValue(const ge::OpDesc &op_desc, const ge::AnyValue &value_type,
                            te::TbeAttrValue &tbe_attr_value, const string &attr_name) {
  int64_t value = 0;
  value_type.GetValue<int64_t>(value);
  if (!ge::AttrUtils::GetInt(&op_desc, attr_name, value)) {
    FE_LOGD("Op[%s, %s] Can not to get private %s from graph, and to keep default value",
            op_desc.GetTypePtr(), op_desc.GetNamePtr(), attr_name.c_str());
  }
  tbe_attr_value = te::TbeAttrValue(attr_name, value);
}

void GetFloatPrivateAttrValue(const ge::OpDesc &op_desc, const ge::AnyValue &value_type,
                                te::TbeAttrValue &tbe_attr_value, const string &attr_name) {
  float value = 0.0;
  value_type.GetValue<float>(value);
  if (!ge::AttrUtils::GetFloat(&op_desc, attr_name, value)) {
    FE_LOGD("Op[%s, %s] Can not to get private %s from graph, and to keep default value",
            op_desc.GetTypePtr(), op_desc.GetNamePtr(), attr_name.c_str());
  }
  tbe_attr_value = te::TbeAttrValue(attr_name, value);
}

void GetBoolPrivateAttrValue(const ge::OpDesc &op_desc, const ge::AnyValue &value_type,
                             te::TbeAttrValue &tbe_attr_value, const string &attr_name) {
  bool value = false;
  value_type.GetValue<bool>(value);
  if (!ge::AttrUtils::GetBool(&op_desc, attr_name, value)) {
    FE_LOGD("Op[%s, %s] Can not to get private %s from graph, and to keep default value",
            op_desc.GetTypePtr(), op_desc.GetNamePtr(), attr_name.c_str());
  }
  tbe_attr_value = te::TbeAttrValue(attr_name, value);
}

void GetListStrPrivateAttrValue(const ge::OpDesc &op_desc, const ge::AnyValue &value_type,
                              te::TbeAttrValue &tbe_attr_value, const string &attr_name) {
  vector<string> value;
  value_type.GetValue<vector<string>>(value);
  if (!ge::AttrUtils::GetListStr(&op_desc, attr_name, value)) {
    FE_LOGD("Op[%s, %s] Can not to get private %s from graph, and to keep default value",
            op_desc.GetTypePtr(), op_desc.GetNamePtr(), attr_name.c_str());
  }
  tbe_attr_value = te::TbeAttrValue(attr_name, value);
}

void GetListIntPrivateAttrValue(const ge::OpDesc &op_desc, const ge::AnyValue &value_type,
                              te::TbeAttrValue &tbe_attr_value, const string &attr_name) {
  vector<int64_t> value;
  value_type.GetValue<vector<int64_t>>(value);
  if (!ge::AttrUtils::GetListInt(&op_desc, attr_name, value)) {
    FE_LOGD("Op[%s, %s] Can not to get private %s from graph, and to keep default value",
            op_desc.GetTypePtr(), op_desc.GetNamePtr(), attr_name.c_str());
  }
  tbe_attr_value = te::TbeAttrValue(attr_name, value);
}


void GetListFloatPrivateAttrValue(const ge::OpDesc &op_desc, const ge::AnyValue &value_type,
                              te::TbeAttrValue &tbe_attr_value, const string &attr_name) {
  vector<float> value;
  value_type.GetValue<vector<float>>(value);
  if (!ge::AttrUtils::GetListFloat(&op_desc, attr_name, value)) {
    FE_LOGD("Op[%s, %s] Can not to get private %s from graph, and to keep default value",
            op_desc.GetTypePtr(), op_desc.GetNamePtr(), attr_name.c_str());
  }
  tbe_attr_value = te::TbeAttrValue(attr_name, value);
}

void GetListListIntPrivateAttrValue(const ge::OpDesc &op_desc, const ge::AnyValue &value_type,
                              te::TbeAttrValue &tbe_attr_value, const string &attr_name) {
  vector<vector<int64_t>> value;
  value_type.GetValue<vector<vector<int64_t>>>(value);
  if (!ge::AttrUtils::GetListListInt(&op_desc, attr_name, value)) {
    FE_LOGD("Op[%s, %s] Can not to get private %s from graph, and to keep default value",
            op_desc.GetTypePtr(), op_desc.GetNamePtr(), attr_name.c_str());
  }
  tbe_attr_value = te::TbeAttrValue(attr_name, value);
}

void GetListBoolPrivateAttrValue(const ge::OpDesc &op_desc, const ge::AnyValue &value_type,
                              te::TbeAttrValue &tbe_attr_value, const string &attr_name) {
  vector<bool> value;
  value_type.GetValue<vector<bool>>(value);
  if (!ge::AttrUtils::GetListBool(&op_desc, attr_name, value)) {
    FE_LOGD("Op[%s, %s] Can not to get private %s from graph, and to keep default value",
            op_desc.GetTypePtr(), op_desc.GetNamePtr(), attr_name.c_str());
  }
  tbe_attr_value = te::TbeAttrValue(attr_name, value);
}

} // namespace fe

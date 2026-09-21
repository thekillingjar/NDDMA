/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fusion_rule_manager/fusion_rule_parser/attr_assignment_expression.h"

#include <cstdlib>
#include <map>
#include <string>
#include <vector>
#include "common/util/op_info_util.h"
#include "ops_store/ops_kernel_manager.h"
#include "fusion_rule_manager/fusion_rule_parser/fusion_rule_parser_utils.h"

using std::map;
using std::string;
using std::vector;

namespace fe {

Status AttrAssignmentExpression::ParseJson(const nlohmann::json &json_object) {
  if (!json_object.is_object()) {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseJs] Type of graph.Attrs should be object, actually is %s",
                    GetJsonType(json_object).c_str());
    return ILLEGAL_JSON;
  }
  // Key map in attribute assignment expression, "attr", "value", "expr"
  map<string, bool> check_map {
      {ATTR, false},
      {VALUE, false},
      {EXPR, false},
  };

  for (const auto &item : json_object.items()) {
    if (item.key() == ATTR) {
      // parse "attr":"node_name.attr_name" from json
      string str;
      if (GetFromJson(item.value(), str) != SUCCESS) {
        REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseJs] Get graph.Attrs.%s from json failed.", ATTR.c_str());
        return ILLEGAL_JSON;
      }
      if (SplitKeyValueByDot(str, attr_.node_name, attr_.attr_name) != SUCCESS) {
        REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseJs] Convert json Attrs.%s to FusionRuleAttr failed.",
                        ATTR.c_str());
        return ILLEGAL_JSON;
      }
      check_map[ATTR] = true;
    } else if (item.key() == VALUE) {
      // Parse "value":"node_name.attr_name"/ "18.12" from json
      // Now can't parse json to actual value, so temporarily save original json
      // object
      tmp_value_ = item.value();
      check_map[VALUE] = true;
    } else if (item.key() == EXPR) {
      // parse "expr":"=" from json
      string str;
      if (GetFromJson(item.value(), str) != SUCCESS) {
        REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseJs] Get graph.Attrs.%s from json failed.", EXPR.c_str());
        return ILLEGAL_JSON;
      }
      if (str != OPERATOR_ASSIGN) {
        REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseJs] Not support operator:%s in graph.Attrs assign expression",
                        str.c_str());
        return ILLEGAL_JSON;
      }
      check_map[EXPR] = true;
    } else {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseJs] Not support key:%s in Attrs", item.key().c_str());
      return ILLEGAL_JSON;
    }
  }

  if (!AnalyseCheckMap(check_map)) {
    return ILLEGAL_JSON;
  }
  parse_to_attr_success_ = true;
  return SUCCESS;
}

Status AttrAssignmentExpression::ParseToJsonAttr(const nlohmann::json &json_object) {
  if (!json_object.is_object()) {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseToJsAttr] Type of graph.Attrs should be object, actually is %s",
                    GetJsonType(json_object).c_str());
    return ILLEGAL_JSON;
  }
  // Key map in attribute assignment expression, "attr", "value", "expr"
  map<string, bool> check_map {
      {ATTR, false},
      {VALUE, false},
      {EXPR, false},
  };

  for (const auto &item : json_object.items()) {
    if (item.key() == ATTR) {
      // parse "attr":"node_name.attr_name" from json
      string str;
      if (GetFromJson(item.value(), str) != SUCCESS) {
        REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseToJsAttr] Get graph.Attrs.%s from json failed.", ATTR.c_str());
        return ILLEGAL_JSON;
      }
      if (SplitKeyValueByDot(str, attr_.node_name, attr_.attr_name) != SUCCESS) {
        REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseToJsAttr] Convert json Attrs.%s to FusionRuleAttr failed.",
                        ATTR.c_str());
        return ILLEGAL_JSON;
      }
      check_map[ATTR] = true;
    } else if (item.key() == VALUE) {
      // Parse "value":"node_name.attr_name"/ "18.12" from json
      // Now can't parse json to actual value, so temporarily save original json
      // object
      tmp_value_ = item.value();
      check_map[VALUE] = true;
    } else if (item.key() == EXPR) {
      // parse "expr":"=" from json
      string str;
      if (GetFromJson(item.value(), str) != SUCCESS) {
        REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseToJsAttr] Get graph.Attrs.%s from json failed.", EXPR.c_str());
        return ILLEGAL_JSON;
      }
      if (str != OPERATOR_ASSIGN) {
        REPORT_FE_ERROR(
            "[GraphOpt][FusionRuleInit][ParseToJsAttr] No supported operator %s in graph.Attrs assign expression",
            str.c_str());
        return ILLEGAL_JSON;
      }
      check_map[EXPR] = true;
    } else {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseToJsAttr] No supported key %s in Attrs", item.key().c_str());
      return ILLEGAL_JSON;
    }
  }

  if (!AnalyseCheckMap(check_map)) {
    return ILLEGAL_JSON;
  }
  parse_to_attr_success_ = true;
  return SUCCESS;
}

Status AttrAssignmentExpression::GetValueType(const std::map<string, std::vector<string>> &node_map,
                                              const FusionRuleAttr &attr,
                                              ge::GeAttrValue::ValueType &value_type) const {
  // 1. get OpType according to input node map
  auto node_map_iter = node_map.find(attr.node_name);
  if (node_map_iter == node_map.end()) {
    FE_LOGW("Can't find node:%s in nodes map", attr.node_name.c_str());
    return ILLEGAL_JSON;
  }
  // 2. Get OpKernelInfo from OpKernelInfoStore by OpType
  for (const auto &op_type : node_map_iter->second) {
    std::string engine_name = FusionRuleParserUtils::Instance()->GetEngineName();
    FE_CHECK(engine_name.empty(), REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][GetValType] The engine_name is empty."),
             return ILLEGAL_JSON);
    
    OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(engine_name).GetHighPrioOpKernelInfoPtr(op_type);
    if (op_kernel_info_ptr != nullptr) {
      // 3. Get MapAttrType from OpKernelInfo
      if (op_kernel_info_ptr->GetAttrTypeByAttrName(attr.attr_name, value_type) != SUCCESS) {
        REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][GetValType] Can't find attr:%s in OP:%s from OpKernelInfoStore",
                        attr.attr_name.c_str(), op_type.c_str());
        return ILLEGAL_JSON;
      }
      return SUCCESS;
    }
  }

  FE_LOGW("Can't find node:%s's op types in OpKernelInfoStore.", attr.node_name.c_str());
  return INVALID_RULE;
}

Status AttrAssignmentExpression::ParseToAttrValue(const std::map<string, std::vector<string>> &node_map) {
  if (!parse_to_attr_success_) {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseToAttrVal] First step: parse json to attr already failed.");
    return ILLEGAL_JSON;
  }
  // 1. Get ge::ValueType of attr from OpKernelInfoStore
  ge::GeAttrValue::ValueType value_type;
  Status ret = GetValueType(node_map, attr_, value_type);
  if (ret != SUCCESS) {
    FE_LOGW("Get ValueType of %s.%s from OpKernelInfoStore not successfully", attr_.node_name.c_str(),
            attr_.attr_name.c_str());
    return ret;
  }
  // 2. Check if value is FusionRuleAttr
  FE_MAKE_SHARED(value_ = make_shared<FusionRuleAttrValue>(), return INTERNAL_ERROR);
  if (tmp_value_.is_string()) {
    string str;
    ret = GetFromJson(tmp_value_, str);
    if (ret != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][ParseToAttrVal] Get Attrs.value(string) from json failed.");
      return ret;
    }
    FusionRuleAttr tmp_attr;
    if (SplitKeyValueByDot(str, tmp_attr.node_name, tmp_attr.attr_name) == SUCCESS) {
      // if value satisfied format of FusionRuleAttr, then convert to
      // FusionRuleAttr
      ge::GeAttrValue::ValueType tmp_value_type;
      if (GetValueType(node_map, tmp_attr, tmp_value_type) == SUCCESS) {
        if (value_type != tmp_value_type) {
          REPORT_FE_ERROR(
              "[GraphOpt][FusionRuleInit][ParseToAttrVal] In AttrAssignmentExpression, ValueType of OpKernelInfoStore"
              ":%s.%s is not equal to json file:%s.%s",
              attr_.node_name.c_str(), attr_.attr_name.c_str(), tmp_attr.node_name.c_str(), tmp_attr.attr_name.c_str());
          return ILLEGAL_JSON;
        }
        value_->SetAttrValue(tmp_attr);
        return SUCCESS;
      }
    }
  }
  // 3. Else parse value by attr_.value_type
  ret = ConvertToAttrValue(tmp_value_, value_type, value_);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR(
        "[GraphOpt][FusionRuleInit][ParseToAttrVal] Convert fix attr value of AttrAssignmentExpression failed.");
    return ILLEGAL_JSON;
  }

  return SUCCESS;
}

Status AttrAssignmentExpression::ConvertToAttrValue(const nlohmann::json &json_object,
                                                    ge::GeAttrValue::ValueType value_type,
                                                    FusionRuleAttrValuePtr value) {
  switch (value_type) {
    case ge::GeAttrValue::VT_INT: {
      int64_t tmp_value;
      return GetStrAndConvert(json_object, tmp_value, value);
    }
    case ge::GeAttrValue::VT_FLOAT: {
      float tmp_value;
      return GetStrAndConvert(json_object, tmp_value, value);
    }
    case ge::GeAttrValue::VT_BOOL: {
      bool tmp_value = false;
      return GetStrAndConvert(json_object, tmp_value, value);
    }
    case ge::GeAttrValue::VT_STRING: {
      string tmp_value;
      return GetStrAndConvert(json_object, tmp_value, value);
    }
    case ge::GeAttrValue::VT_LIST_INT: {
      vector<int64_t> tmp_value;
      return GetStrAndConvert(json_object, tmp_value, value);
    }
    case ge::GeAttrValue::VT_LIST_FLOAT: {
      vector<float> tmp_value;
      return GetStrAndConvert(json_object, tmp_value, value);
    }
    case ge::GeAttrValue::VT_LIST_BOOL: {
      vector<bool> tmp_value;
      return GetStrAndConvert(json_object, tmp_value, value);
    }
    case ge::GeAttrValue::VT_LIST_STRING: {
      vector<string> tmp_value;
      return GetStrAndConvert(json_object, tmp_value, value);
    }
    default: {
      REPORT_FE_ERROR(
          "[GraphOpt][FusionRuleInit][ConvtToAttrVal] Not supported ge::ValueType %d in AttrAssignmentExpression.",
          value_type);
      return ILLEGAL_JSON;
    }
  }
}

Status AttrAssignmentExpression::GetStrAndConvert(const nlohmann::json &json_object, vector<int64_t> &value,
                                                  FusionRuleAttrValuePtr attr_value) const {
  FE_CHECK(attr_value == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][GetStrConvt] Input attr value is null."),
           return ILLEGAL_JSON);

  // List int in json should be array type
  if (!json_object.is_array()) {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][GetStrConvt] Convert json to list int should from array actually is %s",
                    GetJsonType(json_object).c_str());
    return ILLEGAL_JSON;
  }

  FusionRuleAttrValuePtr tmp_attr_value = nullptr;
  FE_MAKE_SHARED(tmp_attr_value = make_shared<FusionRuleAttrValue>(), return INTERNAL_ERROR);

  for (const auto &iter : json_object) {
    int64_t tmp_value;
    Status ret = GetStrAndConvert(iter, tmp_value, tmp_attr_value);
    if (ret != SUCCESS) {
      REPORT_FE_ERROR(
          "[GraphOpt][FusionRuleInit][GetStrConvt] Get element int list int of Attrs.value from json array failed.");
      return ret;
    }
    value.push_back(tmp_value);
  }
  attr_value->SetAttrValue(value);
  return SUCCESS;
}

Status AttrAssignmentExpression::GetStrAndConvert(const nlohmann::json &json_object, vector<float> &value,
                                                  FusionRuleAttrValuePtr attr_value) const {
  FE_CHECK(attr_value == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][GetStrConvt] Input attr value is null."),
           return ILLEGAL_JSON);

  // List float in json should be array type
  if (!json_object.is_array()) {
    REPORT_FE_ERROR(
        "[GraphOpt][FusionRuleInit][GetStrConvt] Convert json to list float should from array actually is %s",
        GetJsonType(json_object).c_str());
    return ILLEGAL_JSON;
  }

  FusionRuleAttrValuePtr tmp_attr_value = nullptr;
  FE_MAKE_SHARED(tmp_attr_value = make_shared<FusionRuleAttrValue>(), return INTERNAL_ERROR);

  for (const auto &iter : json_object) {
    float tmp_value;
    Status ret = GetStrAndConvert(iter, tmp_value, tmp_attr_value);
    if (ret != SUCCESS) {
      REPORT_FE_ERROR(
          "[GraphOpt][FusionRuleInit][GetStrConvt] Get element in list float of Attrs.value from json array failed.");
      return ret;
    }
    value.push_back(tmp_value);
  }
  attr_value->SetAttrValue(value);
  return SUCCESS;
}

Status AttrAssignmentExpression::GetStrAndConvert(const nlohmann::json &json_object, vector<bool> &value,
                                                  FusionRuleAttrValuePtr attr_value) {
  FE_CHECK(attr_value == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][GetStrConvt] Input attr value is null."),
           return ILLEGAL_JSON);

  // List bool in json should be array type
  if (!json_object.is_array()) {
    REPORT_FE_ERROR(
        "[GraphOpt][FusionRuleInit][GetStrConvt] Convert json to list bool should from array actually is %s",
        GetJsonType(json_object).c_str());
    return ILLEGAL_JSON;
  }

  FusionRuleAttrValuePtr tmp_attr_value = nullptr;
  FE_MAKE_SHARED(tmp_attr_value = make_shared<FusionRuleAttrValue>(), return INTERNAL_ERROR);
  for (const auto &iter : json_object) {
    bool tmp_value = false;
    Status ret = GetStrAndConvert(iter, tmp_value, tmp_attr_value);
    if (ret != SUCCESS) {
      REPORT_FE_ERROR(
          "[GraphOpt][FusionRuleInit][GetStrConvt] Get element in list bool of Attrs.value from json array failed.");
      return ret;
    }
    value.push_back(tmp_value);
  }
  attr_value->SetAttrValue(value);
  return SUCCESS;
}

Status AttrAssignmentExpression::GetStrAndConvert(const nlohmann::json &json_object, vector<string> &value,
                                                  const FusionRuleAttrValuePtr &attr_value) const {
  FE_CHECK(attr_value == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][GetStrConvt] Input attr value is null."),
           return ILLEGAL_JSON);

  // List string in json should be array type
  if (!json_object.is_array()) {
    REPORT_FE_ERROR(
        "[GraphOpt][FusionRuleInit][GetStrConvt] Convert json to list string should from array actually is %s",
        GetJsonType(json_object).c_str());
    return ILLEGAL_JSON;
  }

  try {
    value = json_object.get<vector<string>>();
  } catch (const nlohmann::json::exception &e) {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][GetStrConvt] Get list string from json failed:%s", e.what());
    return ILLEGAL_JSON;
  }
  attr_value->SetAttrValue(value);
  return SUCCESS;
}

Status AttrAssignmentExpression::GetStrAndConvert(const nlohmann::json &json_object, int64_t &value,
                                                  FusionRuleAttrValuePtr attr_value) const {
  FE_CHECK(attr_value == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][GetStrConvt] Input attr value is null."),
           return ILLEGAL_JSON);

  string str;
  Status ret = GetFromJson(json_object, str);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][GetStrConvt] Get Attrs.value(string) from json failed");
    return ret;
  }

  ret = StringToInt64(str, value);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][GetStrConvt] Convert %s to int value failed.", str.c_str());
    return ret;
  }
  attr_value->SetAttrValue(value);
  return SUCCESS;
}

Status AttrAssignmentExpression::GetStrAndConvert(const nlohmann::json &json_object, float &value,
                                                  FusionRuleAttrValuePtr attr_value) const {
  FE_CHECK(attr_value == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][GetStrConvt] Input attr value is null."),
           return ILLEGAL_JSON);

  string str;
  Status ret = GetFromJson(json_object, str);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][GetStrConvt] Get attr.value(string) from json failed");
    return ret;
  }

  ret = StringToFloat(str, value);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][GetStrConvt] Convert %s to float value failed.", str.c_str());
    return ret;
  }
  attr_value->SetAttrValue(value);
  return SUCCESS;
}

Status AttrAssignmentExpression::GetStrAndConvert(const nlohmann::json &json_object, bool &value,
                                                  FusionRuleAttrValuePtr attr_value) {
  FE_CHECK(attr_value == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][GetStrConvt] Input attr value is null."),
           return ILLEGAL_JSON);

  string str;
  Status ret = GetFromJson(json_object, str);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][GetStrConvt] Get attr.value(string) from json failed");
    return ret;
  }

  if (str == BOOL_TRUE) {
    value = true;
    attr_value->SetAttrValue(true);
  } else if (str == BOOL_FALSE) {
    value = false;
    attr_value->SetAttrValue(false);
  } else {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][GetStrConvt] Convert %s to bool value failed.", str.c_str());
    return ILLEGAL_JSON;
  }

  return SUCCESS;
}

Status AttrAssignmentExpression::GetStrAndConvert(const nlohmann::json &json_object, string &value,
                                                  FusionRuleAttrValuePtr attr_value) const {
  FE_CHECK(attr_value == nullptr, REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][GetStrConvt] Input attr value is null."),
           return ILLEGAL_JSON);

  Status ret = GetFromJson(json_object, value);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][GetStrConvt] Get attr.value(string) from json failed");
    return ILLEGAL_JSON;
  }
  attr_value->SetAttrValue(value);
  return SUCCESS;
}
}  // namespace fe

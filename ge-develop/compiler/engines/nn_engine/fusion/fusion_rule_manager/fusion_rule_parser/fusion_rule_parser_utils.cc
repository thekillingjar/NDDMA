/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fusion_rule_manager/fusion_rule_parser/fusion_rule_parser_utils.h"

#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>
#include "common/string_utils.h"

using std::string;
using std::vector;

namespace fe {
namespace {
const string SYMBOL_COMMA_WHITESPACE = ", ";
const string SYMBOL_BRACKET_LEFT = "[";
const string SYMBOL_BRACKET_RIGHT = "]";
}  // namespace

Status GetFromJson(const nlohmann::json &json_object, string &str) {
  if (!json_object.is_string()) {
    ostringstream oss;
    oss << json_object;
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][GetFmJs] Parse %s to string failed.", oss.str().c_str());
    return ILLEGAL_JSON;
  }
  string tmp_str = json_object.get<string>();
  str = StringUtils::Trim(tmp_str);
  if (str.empty()) {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][GetFmJs] String get from json is null.");
    return ILLEGAL_JSON;
  }

  return SUCCESS;
}

bool AnalyseCheckMap(const std::map<string, bool> &check_map) {
  bool result = true;
  for (const auto &iter : check_map) {
    if (!iter.second) {
      REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][AnlyChkMap] Key:%s miss in the check_map", iter.first.c_str());
      result = false;
      break;
    }
  }

  return result;
}

Status SplitKeyValueByColon(const string &str, string &key, string &value) {
  auto pos = str.find(':');
  if (pos == string::npos) {
    return ILLEGAL_JSON;
  }
  string tmp_key = str.substr(0, pos);
  key = StringUtils::Trim(tmp_key);
  string tmp_value = str.substr(pos + 1, (str.size() - pos - 1));
  value = StringUtils::Trim(tmp_value);

  return SUCCESS;
}

Status SplitKeyValueByDot(const string &str, string &key, string &value) {
  auto pos = str.find('.');
  if (pos == string::npos) {
    return ILLEGAL_JSON;
  }
  try {
    // try to convert input to string, if success, means input is a float nummer
    (void)stof(str);
    return ILLEGAL_JSON;
  } catch (...) {
    string tmp_key = str.substr(0, pos);
    key = StringUtils::Trim(tmp_key);
    string tmp_value = str.substr(pos + 1, (str.size() - pos - 1));
    value = StringUtils::Trim(tmp_value);

    return SUCCESS;
  }

  return ILLEGAL_JSON;
}

Status StringToInt(const string &str, int &value) {
  // check if str have '.', stoi() will auto convert float to int
  if (str.find('.') != string::npos) {
    FE_LOGE("Convert %s to int value failed.", str.c_str());
    return ILLEGAL_JSON;
  }
  // check if str have ':', stoi() will auto ignored content after ':'
  if (str.find(':') != string::npos) {
    FE_LOGE("Convert %s to int value failed.", str.c_str());
    return ILLEGAL_JSON;
  }
  int tmp_value;
  try {
    tmp_value = stoi(str);
  } catch (const exception &e) {
    FE_LOGE("Convert %s to int value failed:%s", str.c_str(), e.what());
    return ILLEGAL_JSON;
  }
  value = tmp_value;
  return SUCCESS;
}

Status StringToInt64(const string &str, int64_t &value) {
  // check if str have '.', stol() will auto convert float to int
  if (str.find('.') != string::npos) {
    FE_LOGE("Convert %s to int64_t value failed.", str.c_str());
    return ILLEGAL_JSON;
  }
  // check if str have ':', stol() will auto ignored content after ':'
  if (str.find(':') != string::npos) {
    FE_LOGE("Convert %s to int64_t value failed.", str.c_str());
    return ILLEGAL_JSON;
  }
  int64_t tmp_value;
  try {
    tmp_value = static_cast<int64_t>(stol(str));
  } catch (const exception &e) {
    FE_LOGE("Convert %s to int64_t value failed:%s", str.c_str(), e.what());
    return ILLEGAL_JSON;
  }
  value = tmp_value;

  return SUCCESS;
}

Status StringToFloat(const string &str, float &value) {
  // check if str have ':', stof() will auto ignored content after ':'
  if (str.find(':') != string::npos) {
    FE_LOGE("Convert %s to float value failed.", str.c_str());
    return ILLEGAL_JSON;
  }
  float tmp_value;
  try {
    tmp_value = stof(str);
  } catch (const exception &e) {
    FE_LOGE("Convert %s to float value failed:%s", str.c_str(), e.what());
    return ILLEGAL_JSON;
  }
  value = tmp_value;

  return SUCCESS;
}

string GetStrOfInt(ge::GeAttrValue &attr_value) {
  int64_t tmp_value;
  if (attr_value.GetValue<int64_t>(tmp_value) != ge::GRAPH_SUCCESS) {
    FE_LOGW("Not found int value.");
    return "";
  }

  return to_string(tmp_value);
}

string GetStrOfFloat(ge::GeAttrValue &attr_value) {
  float tmp_value;
  if (attr_value.GetValue<float>(tmp_value) != ge::GRAPH_SUCCESS) {
    FE_LOGW("Not found float value.");
    return "";
  }

  return to_string(tmp_value);
}

string GetStrOfBool(ge::GeAttrValue &attr_value) {
  bool tmp_value = false;
  if (attr_value.GetValue<bool>(tmp_value) != ge::GRAPH_SUCCESS) {
    FE_LOGW("Not found bool value.");
    return "";
  }

  return (tmp_value) ? "true" : "false";
}

string GetStrOfListInt(ge::GeAttrValue &attr_value) {
  vector<int64_t> tmp_value;
  if (attr_value.GetValue<vector<int64_t>>(tmp_value) != ge::GRAPH_SUCCESS) {
    FE_LOGW("Not found list int value.");
    return "";
  }
  string str = SYMBOL_BRACKET_LEFT;
  for (const auto &value : tmp_value) {
    str += to_string(value) + SYMBOL_COMMA_WHITESPACE;
  }
  str += SYMBOL_BRACKET_RIGHT;

  return str;
}

string GetStrOfListFloat(ge::GeAttrValue &attr_value) {
  vector<float> tmp_value;
  if (attr_value.GetValue<vector<float>>(tmp_value) != ge::GRAPH_SUCCESS) {
    FE_LOGW("Not found list float value.");
    return "";
  }
  string str = SYMBOL_BRACKET_LEFT;
  for (const auto &value : tmp_value) {
    str += to_string(value) + SYMBOL_COMMA_WHITESPACE;
  }
  str += SYMBOL_BRACKET_RIGHT;

  return str;
}

string GetStrOfListBool(ge::GeAttrValue &attr_value) {
  vector<bool> tmp_value;
  if (attr_value.GetValue<vector<bool>>(tmp_value) != ge::GRAPH_SUCCESS) {
    FE_LOGW("Not found list bool value.");
    return "";
  }
  string str = SYMBOL_BRACKET_LEFT;
  for (const auto &value : tmp_value) {
    string bool_value = (value) ? "true" : "false";
    str += bool_value + SYMBOL_COMMA_WHITESPACE;
  }
  str += SYMBOL_BRACKET_RIGHT;

  return str;
}

string GetStrFromAttrValue(ge::GeAttrValue &attr_value) {
  auto value_type = attr_value.GetValueType();
  switch (value_type) {
    case ge::GeAttrValue::VT_INT:
      return GetStrOfInt(attr_value);
    case ge::GeAttrValue::VT_FLOAT:
      return GetStrOfFloat(attr_value);
    case ge::GeAttrValue::VT_BOOL:
      return GetStrOfBool(attr_value);
    case ge::GeAttrValue::VT_STRING: {
      string tmp_value;
      if (attr_value.GetValue<string>(tmp_value) != ge::GRAPH_SUCCESS) {
        FE_LOGE("Get string value failed.");
        return "";
      }
      return "\"" + tmp_value + "\"";
    }
    case ge::GeAttrValue::VT_LIST_INT:
      return GetStrOfListInt(attr_value);
    case ge::GeAttrValue::VT_LIST_FLOAT:
      return GetStrOfListFloat(attr_value);
    case ge::GeAttrValue::VT_LIST_BOOL:
      return GetStrOfListBool(attr_value);
    case ge::GeAttrValue::VT_LIST_STRING: {
      vector<string> tmp_value;
      if (attr_value.GetValue<vector<string>>(tmp_value) != ge::GRAPH_SUCCESS) {
        FE_LOGE("Get list string value failed.");
        return "";
      }
      string str = "[";
      for (const auto &value : tmp_value) {
        str += "\"" + value + "\", ";
      }
      str += "]";
      return str;
    }
    default: {
      FE_LOGW("Not supported ge::ValueType %d.", value_type);
      return "";
    }
  }
}

}  // namespace fe

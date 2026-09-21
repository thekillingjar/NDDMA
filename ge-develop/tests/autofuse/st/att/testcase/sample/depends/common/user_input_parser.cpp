/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "user_input_parser.h"
#include <fstream>
#include "common/checker.h"
#include "nlohmann/json.hpp"
namespace att {
#define USED_BY_JSON __attribute__((unused)) static
using Json = nlohmann::json;
ge::Status StringToJson(const std::string &json_str, Json &json) {
  std::stringstream ss;
  ss << json_str;
  try {
    ss >> json;
  } catch (const nlohmann::json::exception &e) {
    GELOGE(ge::PARAM_INVALID, "Failed to init json object, err = %s, json_str = %s", e.what(), json_str.c_str());
    return ge::PARAM_INVALID;
  }
  return ge::SUCCESS;
}

template <typename T>
ge::Status ParseFromJson(const std::string &type, const std::string &json_str, T &value) {
  Json json;
  const auto ret = StringToJson(json_str, json);
  if (ret != 0) {
    return ret;
  }
  try {
    value = json.get<T>();
  } catch (const nlohmann::json::exception &e) {
    GELOGE(ge::PARAM_INVALID, "Failed to parse json object, type = %s, err = %s, json_str = %s", type.c_str(), e.what(),
            json_str.c_str());
    return ge::PARAM_INVALID;
  }
  return ge::SUCCESS;
}

template <typename T>
std::string ToJsonString(const T &obj, const int32_t indent) {
  try {
    const Json j = obj;
    return j.dump(indent);
  } catch (const nlohmann::json::exception &e) {
    GELOGE(ge::FAILED, "Failed to dump object, err = %s", e.what());
    return "";
  }
}

template <typename T>
void GetValue(const Json &j, const std::string &key, T &value) {
  value = j.at(key).template get<T>();
}

USED_BY_JSON void from_json(const Json &j, InputShape &input_shape) {
  GetValue(j, "tiling_key", input_shape.tiling_key);
  GetValue(j, "axes", input_shape.axes);
}

USED_BY_JSON void to_json(Json &j, const InputShape &input_shape) {
  j = Json();
  j["tiling_key"] = input_shape.tiling_key;
  j["axes"] = input_shape.axes;
}

USED_BY_JSON void from_json(const Json &j, UserInput &user_input) {
  GetValue(j, "input_shapes", user_input.input_shapes);
}

USED_BY_JSON void to_json(Json &j, const UserInput &user_input) {
  j = Json();
  j["input_shapes"] = user_input.input_shapes;
}

std::string UserInputParser::ToJson(const InputShape &input_shape, const int32_t indent) {
  return ToJsonString(input_shape, indent);
}

ge::Status UserInputParser::FromJson(const std::string &json_str, InputShape &input_shape) {
  return ParseFromJson("InputShape", json_str, input_shape);
}

std::string UserInputParser::ToJson(const UserInput &user_input, const int32_t indent) {
  return ToJsonString(user_input, indent);
}

ge::Status UserInputParser::FromJson(const std::string &json_str, UserInput &user_input) {
  return ParseFromJson("UserInput", json_str, user_input);
}

bool UserInputParser::CheckInputValid(const std::string &input_file) {
  std::ifstream file(input_file);
  return file.good();
}
std::string UserInputParser::ReadFileToString(const std::string &file_path) {
  std::ifstream file(file_path);
  GE_ASSERT_TRUE(file.is_open(), "Read file[%s] failed, as file is not open.", file_path.c_str());
  std::stringstream buffer;
  buffer << file.rdbuf();
  file.close();
  return buffer.str();
}
}  // namespace att

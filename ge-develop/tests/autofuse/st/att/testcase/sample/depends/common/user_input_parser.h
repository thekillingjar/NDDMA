/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SAMPLEPROJECT_ATT_SAMPLE_TESTS_UTILS_USER_INPUT_PARSER_H_
#define SAMPLEPROJECT_ATT_SAMPLE_TESTS_UTILS_USER_INPUT_PARSER_H_
#include <vector>
#include <string>
#include "common/checker.h"
namespace att {
struct InputShape {
  uint32_t tiling_key;
  std::vector<std::pair<std::string, uint32_t>> axes;
};
struct UserInput {
  std::vector<InputShape> input_shapes;
};
class UserInputParser {
 public:
  static std::string ToJson(const InputShape &input_shape, const int32_t indent = 4);
  static ge::Status FromJson(const std::string &json_str, InputShape &input_shape);
  static std::string ToJson(const UserInput &user_input, const int32_t indent = 4);
  static ge::Status FromJson(const std::string &json_str, UserInput &user_input);
  static bool CheckInputValid(const std::string &input_file);
  static std::string ReadFileToString(const std::string& file_path);
};
}  // namespace att

#endif  // SAMPLEPROJECT_ATT_SAMPLE_TESTS_UTILS_USER_INPUT_PARSER_H_

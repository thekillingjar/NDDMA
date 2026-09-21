/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RUNTIME_COMMON_JSON_PARSER_H_
#define RUNTIME_COMMON_JSON_PARSER_H_

#include <string>
#include "nlohmann/json.hpp"
#include "ge/ge_api_error_codes.h"

namespace ge {
class JsonParser {
 public:
  static Status ReadConfigFile(const std::string &file_path, nlohmann::json &js);

 private:
  static bool CheckFilePath(const std::string &file_path);
};
}  // namespace ge
#endif  // RUNTIME_COMMON_JSON_PARSER_H_

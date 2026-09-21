/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "auto_fuse_config_parser.h"
#include <regex>
#include <string>
#include <fstream>
#include <unordered_map>
#include "common/checker.h"
#include "mmpa/mmpa_api.h"
namespace ge {
constexpr size_t FLAG_PREFIX_LENGTH = 2;

inline std::vector<std::string> SplitString(const std::string &input, char delimiter) {
  std::vector<std::string> result;
  std::stringstream ss(input);
  std::string token;

  while (std::getline(ss, token, delimiter)) {
    result.push_back(token);
  }

  return result;
}

inline void ParseFlags(const std::string &input, const std::vector<std::string> &white_list,
                       std::unordered_map<std::string, std::string> &res) {
  std::unordered_map<std::string, std::string> tmp_flags;
  auto params = SplitString(input, ';');
  for (const auto &param : params) {
    auto keyValue = SplitString(param, '=');
    if (keyValue.size() == FLAG_PREFIX_LENGTH) {
      std::string key = keyValue[0];
      // 去掉--前缀
      if (key.size() >= FLAG_PREFIX_LENGTH && key[0] == '-' && key[1] == '-') {
        key = key.substr(FLAG_PREFIX_LENGTH);
      }
      tmp_flags[key] = keyValue[1];
    }
  }
  for (const auto &key : white_list) {
    const auto it = tmp_flags.find(key);
    if (it != tmp_flags.cend()) {
      res[key] = it->second;
    }
  }
}

std::unordered_map<std::string, std::string> AutoFuseEnvConfigParser::Parse() {
  std::unordered_map<std::string, std::string> res;
  const char_t *env_value = nullptr;
  // 解析AUTOFUSE_FLAGS
  MM_SYS_GET_ENV(MM_ENV_AUTOFUSE_FLAGS, env_value);
  if ((env_value != nullptr) && (strlen(env_value) > 0U)) {
    ParseFlags(env_value, flags_config_keys_, res);
  }
  // 解析AUTOFUSE_DFX_FLAGS
  env_value = nullptr;
  MM_SYS_GET_ENV(MM_ENV_AUTOFUSE_DFX_FLAGS, env_value);
  if ((env_value != nullptr) && (strlen(env_value) > 0U)) {
    ParseFlags(env_value, dfx_flags_config_keys_, res);
  }
  return res;
}
}  // namespace ge
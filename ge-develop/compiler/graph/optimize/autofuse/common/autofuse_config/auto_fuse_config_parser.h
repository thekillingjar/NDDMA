/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMON_AUTOFUSE_CONFIG_AUTO_FUSE_CONFIG_PARSER_H_
#define COMMON_AUTOFUSE_CONFIG_AUTO_FUSE_CONFIG_PARSER_H_
#include <cstdint>
#include <cstdlib>
#include <string>
#include <memory>
#include <utility>
#include "common/autofuse_base_type.h"
#include "ge_common/ge_api_types.h"
#include "auto_fuse_config.h"

namespace ge {

class AutoFuseConfigParser {
 public:
  explicit AutoFuseConfigParser(const AutoFuseConfigType type) : type_(type) {};
  virtual ~AutoFuseConfigParser() {
    configs_.clear();
  }
  virtual std::unordered_map<std::string, std::string> Parse() = 0;

 private:
  AutoFuseConfigType type_;
  std::vector<std::unique_ptr<att::AutoFuseConfigBase>> configs_;
};

class AutoFuseEnvConfigParser : public AutoFuseConfigParser {
 public:
  explicit AutoFuseEnvConfigParser(const std::vector<std::string> &flags_keys,
                                   const std::vector<std::string> &dfx_flags_keys)
      : AutoFuseConfigParser(AutoFuseConfigType::ENV_CONFIG), flags_config_keys_(flags_keys),
        dfx_flags_config_keys_(dfx_flags_keys) {};
  ~AutoFuseEnvConfigParser() override = default;
  std::unordered_map<std::string, std::string> Parse() override;

 private:
  std::vector<std::string> flags_config_keys_;
  std::vector<std::string> dfx_flags_config_keys_;
};
}  // namespace ge

#endif  // COMMON_AUTOFUSE_CONFIG_AUTO_FUSE_CONFIG_PARSER_H_

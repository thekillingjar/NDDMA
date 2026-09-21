/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_UTILS_COMMON_MODIFY_MIXLIST_CONFIG_PARSER_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_UTILS_COMMON_MODIFY_MIXLIST_CONFIG_PARSER_

#include <mutex>
#include <nlohmann/json.hpp>
#include "common/aicore_util_types.h"
#include "common/base_config_parser.h"
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"

namespace fe {
using std::string;
class ModifyMixlistConfigParser : public BaseConfigParser {
 public:
  ModifyMixlistConfigParser();
  ~ModifyMixlistConfigParser() override;

  Status InitializeFromOptions(const std::map<std::string, std::string> &options) override;
  using BaseConfigParser::InitializeFromContext;
  Status InitializeFromContext(const std::string &combined_params_key) override;
  PrecisionPolicy GetPrecisionPolicy(const std::string &op_type, const PrecisionPolicy &op_kernel_policy) const;

 private:
  Status InitializeModifyMixlist();
  Status LoadModifyMixlistJson(const std::string &modify_mixlist_path);
  Status ReadMixlistJson(const std::string &file_path, nlohmann::json &json_obj) const;
  void AddMixList(const nlohmann::json &op_json_file, const std::string &list_type, const std::string &update_type);
  Status VerifyMixlist() const;
  static std::string GetMixlistDesc(const uint8_t &op_mix_list);

  std::string modify_mixlist_path_;
  std::map<std::string, std::uint8_t> mixlist_map_;
};
using ModifyMixlistConfigParserPtr = std::shared_ptr<ModifyMixlistConfigParser>;
}  // namespace fe
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_UTILS_COMMON_MODIFY_MIXLIST_CONFIG_PARSER_

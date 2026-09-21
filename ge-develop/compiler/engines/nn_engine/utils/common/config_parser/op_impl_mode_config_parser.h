/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_UTILS_COMMON_OP_IMPL_MODE_CONFIG_PARSER_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_UTILS_COMMON_OP_IMPL_MODE_CONFIG_PARSER_

#include <mutex>
#include "common/base_config_parser.h"
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"

namespace fe {
class OpImplModeConfigParser : public BaseConfigParser {
 public:
  OpImplModeConfigParser(const std::string &ascend_opp_path);
  ~OpImplModeConfigParser() override;

  Status InitializeFromOptions(const std::map<std::string, std::string> &options) override;
  using BaseConfigParser::InitializeFromContext;
  Status InitializeFromContext() override;

  bool GetOpImplModeByOpName(const std::string &op_name, std::string &op_impl_mode) const;
  bool GetOpImplModeByOpType(const std::string &op_type, std::string &op_impl_mode) const;
  bool GetOpImplMode(const std::string &op_name, const std::string &op_type, std::string &op_impl_mode) const;
  bool IsEnableCustomImplMode() const;
  std::string EmplaceHf32ModeForAclnn(const std::string &hf32_mode) const;

 private:
  Status Initialize(const std::string &op_precision_mode,
                    const std::string &op_select_impl_mode,
                    const std::string &op_type_list_for_impl_mode,
                    const std::string &allow_hf32);
  void UpDateDefaultValue(const std::string &op_precision_mode,
                          std::string &op_select_impl_mode,
                          std::string &allow_hf32);
  Status InitOpPrecisionMode(const std::string &op_precision_mode,
                             const std::string &op_select_impl_mode,
                             const std::string &op_type_list_str);
  Status InitOpPrecisionModeByPrecisionMode(const std::string &op_precision_mode);
  Status InitOpPrecisionModeByImplModeAll(const std::string &op_select_impl_mode_all);
  Status InitOpPrecisionModeByImplMode(const std::string &op_select_impl_mode,
                                       const std::string &op_type_list_str);
  Status InitAllowHF32Mode(const std::string &allow_hf32);
  void ParseLineContentWithMode(const std::string &line_content, bool parse_by_op_type,
                                const size_t &pos_of_equal);
  Status GetOpPrecisonModeStrFromConfigFile(const std::string &file_path);
  bool CheckConfigImplType(const std::string &impl_mode) const;
  std::string ascend_opp_path_;
  std::string op_precision_mode_;
  std::string op_select_impl_mode_;
  std::string op_type_list_for_impl_mode_;
  std::string allow_hf32_;
  std::map<std::string, std::string> op_name_select_impl_mode_map_;
  std::map<std::string, std::string> op_type_select_impl_mode_map_;
  mutable std::mutex op_impl_mode_mutex_;
};
using OpImplModeConfigParserPtr = std::shared_ptr<OpImplModeConfigParser>;
}  // namespace fe
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_UTILS_COMMON_OP_IMPL_MODE_CONFIG_PARSER_

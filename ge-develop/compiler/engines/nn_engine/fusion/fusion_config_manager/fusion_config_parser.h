/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_FUSION_FUSION_CONFIG_MANAGER_FUSION_CONFIG_PARSER_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_FUSION_FUSION_CONFIG_MANAGER_FUSION_CONFIG_PARSER_H_

#include <string>
#include <map>
#include <unordered_set>
#include <set>
#include "common/util/json_util.h"
#include "common/string_utils.h"
#include "register/graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

namespace fe {
extern const int32_t CUSTOM_CFG_TOP_PRIORITY_MIN;
extern const int32_t CUSTOM_CFG_MAIN_PRIORITY_MIN;
extern const int32_t CUSTOM_CFG_DOWN_PRIORITY_MIN;
extern const int32_t BUILT_IN_CFG_TOP_PRIORITY_MIN;
extern const int32_t BUILT_IN_CFG_MAIN_PRIORITY_MIN;
extern const int32_t BUILT_IN_CFG_DOWN_PRIORITY_MIN;
extern const int32_t CUSTOM_PASS_PRIORITY_MIN;
extern const int32_t CUSTOM_RULE_PRIORITY_MIN;
extern const int32_t BUILT_IN_PASS_PRIORITY_MIN;
extern const int32_t BUILT_IN_RULE_PRIORITY_MIN;
extern const int32_t RESERVED_FOR_DOWN_PRIORITY;

using FusionSwitchMap = std::map<std::string, std::map<std::string, bool>>;
using FusionPriorityMap = std::map<std::string, std::map<std::string, int32_t>>;

class FusionConfigParser {
 public:
  explicit FusionConfigParser(std::string engine_name);
  virtual ~FusionConfigParser();
  FusionConfigParser(const FusionConfigParser &) = delete;
  FusionConfigParser &operator=(const FusionConfigParser &) = delete;

  Status ParseFusionConfigFile();

  void ParseSupportFusionPassFile();

  bool GetPassSwitchByExceptionalCfg(const std::string &pass_name) const;

  bool GetFusionSwitchByName(const std::string &pass_name, const std::string &pass_type,
                             const uint64_t pass_attr = 0, const bool can_close_fusion = false);

  bool CheckPassIsWhiteListControl(const std::string &pass_name, const GraphFusionPassType &pass_type) const;

  Status GetFusionPriorityByFusionType(const std::string &fusion_type,
                                       std::map<std::string, int32_t> &fusion_priority_map) const;

  void GetFusionPassNameBySwitch(const std::string &pass_type, const bool &is_switch_on,
                                 std::vector<std::string> &pass_name_vec) const;

 private:
  void ConstructFusionSwitchMap(const nlohmann::json &custom_pass_config_json,
                                const nlohmann::json &builtin_pass_config_json,
                                const std::map<string, bool> &old_fusion_switch_map);

  void ModifyFusionSwitchMap(const std::string &fusion_type, const nlohmann::json &fusion_pass_config_json,
                             std::map<std::string, bool> &fusion_switch_map) const;

  void ConstructExceptionalPassMap(const nlohmann::json &fusion_pass_config_json,
                                  std::set<std::string> &exceptional_pass) const;

  Status ConstructFusionPriorityMap(const nlohmann::json &custom_pass_config_json,
                                    const nlohmann::json &builtin_pass_config_json);

  Status ModifyFusionPriorityMap(const std::string &fusion_type,
                                 const nlohmann::json &fusion_pass_config_json,
                                 std::map<std::string, int32_t> &fusion_priority_map) const;

  Status CheckFusionConfigJsonFormat(nlohmann::json pass_switch_file_json, const string &owner_type) const;

  Status CheckLastLayerFormatForPriorityPart(string type_elem, string fusion_elem,
                                             nlohmann::json pass_switch_file_json, const string &owner_type) const;

  Status CheckConfigFileFormat(const nlohmann::json &custom_fusion_config_json,
                               const nlohmann::json &build_in_fusion_config_json,
                               const string &built_in_fusion_config_json_file,
                               const string &custom_fusion_config_json_file) const;

  Status CheckFusionConfigJsonLastLayerFormat(string type_elem, string fusion_elem,
                                              nlohmann::json pass_switch_file_json, const string &owner_type) const;

  Status JudgePriority(string level_elem, int32_t priority_value, const string &owner_type) const;

  Status VerifyAndParserCustomFile(const string& custom_fusion_config_json_file,
                                   nlohmann::json &custom_fusion_config_json,
                                   std::map<string, bool> &old_fusion_switch_map) const;

  Status LoadOldFormatFusionSwitchFile(const string &custom_fusion_config_json_file,
                                       std::map<string, bool> &old_fusion_switch_map) const;

  bool IsForbiddenClosedPass(const std::string &pass_name, const std::string &pass_type,
                             const uint64_t &pass_attr) const;

  bool IsPassAllowedByLicense(const std::string &pass_name) const;

  bool GetPassSwitchByPassName(const std::string &pass_name, const std::string &pass_type, bool &pass_switch) const;

  Status GetKeyAndValueFromJson(const std::string &line, const string &custom_fusion_config_json_file,
                                std::string &key, std::string &value) const;

  Status GetFusionSwitchByOptimizationOption(const std::string &pass_name, bool &enable_flag,
                                             const std::string &pass_type, const uint64_t pass_attr = 0);

  bool is_first_init_;
  bool has_support_fusion_pass_;
  std::string engine_name_;
  std::string fusion_switch_file_path_;
  FusionSwitchMap fusion_switch_map_;
  FusionPriorityMap fusion_priority_map_;
  std::set<std::string> exceptional_pass_;
  std::vector<std::string> support_fusion_pass_vec_;
};
}  // namespace fe
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_FUSION_FUSION_CONFIG_MANAGER_FUSION_CONFIG_PARSER_H_

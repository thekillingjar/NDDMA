/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_FUSION_FUSION_RULE_MANAGER_FUSION_RULE_MANAGER_H_
#define FUSION_ENGINE_FUSION_FUSION_RULE_MANAGER_FUSION_RULE_MANAGER_H_

#include <memory>
#include <string>
#include <vector>
#include "fusion_rule_manager/fusion_rule_data/fusion_rule_pattern.h"
#include "fusion_rule_manager/fusion_rule_parser/fusion_rule_pattern_constructor.h"
#include "common/opskernel/ops_kernel_info_store.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "common/op_info_common.h"

namespace fe {
struct FusionRuleFinder {
  std::string rule_name;

  explicit FusionRuleFinder(std::string name) : rule_name(std::move(name)) {}

  bool operator()(const vector<FusionRulePatternPtr>::value_type &rule_pattern_ptr) {
    return rule_pattern_ptr->GetRuleName() == rule_name;
  }
};

/** @brief implement the manage of fusion rules module.
*        Provide the following methods:
*        initializing the management of fusion rules,
*        finalizing the management of fusion rules,
*        and obtaining fusion rules based on rule types. */
class FusionRuleManager {
 public:
  explicit FusionRuleManager(OpsKernelInfoStorePtr ops_kernel_info_store_ptr);

  virtual ~FusionRuleManager();

  /*
   * obtain fusion rules from jsons
   * sort fusion rules based on the number of nodes
   */
  Status Initialize(const std::string &engine_name);

  Status Finalize();

  /*
   * obtain fusion rules based on rule types
   * param[in] RuleType ruletype, which is defined by the users
   * param[out] vector<FusionRulePattern> &out_fusion_rules, which is get from the
   *            initialize func
   * return Status
   */
  Status GetFusionRulesByRuleType(RuleType rule_type, vector<FusionRulePatternPtr> &out_fusion_rules) const;

  Status RunGraphFusionRuleByType(ge::ComputeGraph &graph, RuleType rule_type, const std::string &rule_name);

 private:
  FusionRuleManager(const FusionRuleManager &) = delete;

  FusionRuleManager &operator=(const FusionRuleManager &) = delete;

  Status InitGraphRules(const std::string &engine_name);
  /*
   * Parse and load each fusion rule from json file to fusion_rule_patterns's
   * element
   */
  static Status LoadFusionRule(const std::string &file_path, std::vector<FusionRulePatternPtr> &fusion_rule_patterns);

  /*
   * Try to open json file, and split whole file to single fusion rules
   */
  static Status OpenJsonFileToItems(const std::string &file_path,
                                    std::vector<nlohmann::json> &fusion_rule_json_objects);

  Status MatchAndReplaceByRules(ge::ComputeGraph &graph, const std::string &rule_name,
                                const vector<FusionRulePatternPtr>::iterator &rule_pattern_ptr_iter);
  /* Store op info */
  OpsKernelInfoStorePtr ops_kernel_info_store_ptr_;

  bool init_flag_;
  /* Save graph rules loaded from json */
  vector<FusionRulePatternPtr> graph_rule_vector_;
  /* Save custom rules loaded from json */
  vector<FusionRulePatternPtr> custom_rule_vector_;
};
using FusionRuleManagerPtr = std::shared_ptr<FusionRuleManager>;
}  // namespace fe
#endif  // FUSION_ENGINE_FUSION_FUSION_RULE_MANAGER_FUSION_RULE_MANAGER_H_

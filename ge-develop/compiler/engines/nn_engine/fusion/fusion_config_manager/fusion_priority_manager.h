/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_FUSION_FUSION_CONFIG_MANAGER_FUSION_PRIORITY_MANAGER_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_FUSION_FUSION_CONFIG_MANAGER_FUSION_PRIORITY_MANAGER_H_

#include <map>
#include <unordered_map>
#include <memory>
#include <string>
#include <utility>

#include "base/err_msg.h"
#include "common/fe_error_code.h"
#include "fusion_config_manager/fusion_config_parser.h"
#include "fusion_rule_manager/fusion_rule_manager.h"
#include "register/graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "register/graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

namespace fe {
struct FusionPassOrRule;
struct BufferFusionInfo;

const string PASS_METHOD = "PASS_METHOD";
const string RULE_METHOD = "RULE_METHOD";

const std::vector<GraphFusionPassType> GRAPH_FUSION_PASS_TYPE_AICORE_VEC = {
        BUILT_IN_GRAPH_PASS,
        SECOND_ROUND_BUILT_IN_GRAPH_PASS,
        BUILT_IN_BEFORE_TRANSNODE_INSERTION_GRAPH_PASS,
        BUILT_IN_PREPARE_GRAPH_PASS,
        BUILT_IN_BEFORE_QUANT_OPTIMIZATION_GRAPH_PASS,
        BUILT_IN_TF_TAG_NO_CONST_FODING_GRAPH_PASS,
        BUILT_IN_TF_MERGE_SUB_GRAPH_PASS,
        BUILT_IN_QUANT_OPTIMIZATION_GRAPH_PASS,
        BUILT_IN_EN_ISA_ARCH_EXC_V300_AND_V220_GRAPH_PASS,
        BUILT_IN_EN_ISA_ARCH_V100_GRAPH_PASS,
        BUILT_IN_EN_ISA_ARCH_V200_GRAPH_PASS,
        BUILT_IN_DELETE_NO_CONST_FOLDING_GRAPH_PASS,
        BUILT_IN_AFTER_OPTIMIZE_STAGE1,
        BUILT_IN_AFTER_MULTI_DIMS_PASS,
        BUILT_IN_AFTER_OP_JUDGE,
        BUILT_IN_AFTER_BUFFER_OPTIMIZE};

const std::vector<GraphFusionPassType> GRAPH_FUSION_PASS_TYPE_VECTORCORE_VEC = { BUILT_IN_VECTOR_CORE_GRAPH_PASS };

const std::vector<GraphFusionPassType> GRAPH_FUSION_QUANT_PASS_VEC = {
        BUILT_IN_TF_MERGE_SUB_GRAPH_PASS,
        BUILT_IN_QUANT_OPTIMIZATION_GRAPH_PASS,
        BUILT_IN_EN_ISA_ARCH_EXC_V300_AND_V220_GRAPH_PASS,
        BUILT_IN_EN_ISA_ARCH_V100_GRAPH_PASS,
        BUILT_IN_EN_ISA_ARCH_V200_GRAPH_PASS,
        BUILT_IN_DELETE_NO_CONST_FOLDING_GRAPH_PASS};

using FusionConfigParserPtr = std::unique_ptr<FusionConfigParser>;
using BufferFusionPassBasePtr = std::unique_ptr<BufferFusionPassBase>;
using FusionPassOrRuleMap = std::map<size_t, std::vector<FusionPassOrRule>>;
using BufferFusionInfoMap = std::map<size_t, std::vector<BufferFusionInfo>>;

struct FusionPassOrRule {
  std::string name;
  int32_t type; // different pass and rule may have same value of type
  std::string method;
  int32_t priority;
  bool need_topo_sort = false;
  FusionPassRegistry::PassDesc pass_desc;

  FusionPassOrRule(std::string name_param, int32_t type_param,
                   std::string method_param, int32_t priority_param,
                   FusionPassRegistry::PassDesc pass_desc_param)
    : name(std::move(name_param)),
      type(type_param),
      method(std::move(method_param)),
      priority(priority_param),
      pass_desc(pass_desc_param) {}
};

struct FusionPassOrRuleFinder {
  std::string fusion_name;

  explicit FusionPassOrRuleFinder(std::string name) : fusion_name(std::move(name)) {}

  bool operator()(const vector<FusionPassOrRule>::value_type &fusion_pass_or_rule) {
    return fusion_pass_or_rule.name == fusion_name;
  }
};

struct BufferFusionInfo {
  std::string name;
  int32_t priority;
  bool is_auto_fusion;
  bool is_fusion_check;
  BufferFusionPassRegistry::CreateFn buffer_fusion_pass_create_fn;

  BufferFusionInfo(const std::string &name_param, const int32_t priority_param, const bool is_auto_fusion_param,
                   const bool is_fusion_check_param, BufferFusionPassRegistry::CreateFn buffer_fusion_pass_create_fn_param)
    : name(name_param),
      priority(priority_param),
      is_auto_fusion(is_auto_fusion_param),
      is_fusion_check(is_fusion_check_param),
      buffer_fusion_pass_create_fn(buffer_fusion_pass_create_fn_param) {}
};

struct BufferFusionFinder {
  std::string fusion_name;

  explicit BufferFusionFinder(std::string name) : fusion_name(std::move(name)) {}

  bool operator()(const vector<BufferFusionInfo>::value_type &BufferFusionInfo) {
    return BufferFusionInfo.name == fusion_name;
  }
};

class FusionPriorityManager {
 public:
  FusionPriorityManager(std::string engine_name, FusionRuleManagerPtr fusion_rule_mgr_ptr);
  virtual ~FusionPriorityManager();

  FusionPriorityManager(const FusionPriorityManager &) = delete;
  FusionPriorityManager &operator=(const FusionPriorityManager &) = delete;

  Status Initialize();

  Status SortGraphFusion();

  Status SortBufferFusion();

  static int32_t GetRealPriority(const int32_t priority);

  BufferFusionPassBasePtr GetBufferFusionByPassName(const std::string &pass_name) const;

  const std::vector<FusionPassOrRule>& GetSortedGraphFusionList(const bool is_single_scene = false);
  const std::vector<BufferFusionInfo>& GetSortedBufferFusionList(const ge::ComputeGraph &graph);

  bool GetFusionSwitchByName(const std::string &pass_name, const std::string &pass_type,
                             const uint64_t pass_attr = 0, const bool can_close_fusion = false) const;

  void GetFusionPassNameBySwitch(const std::string &pass_type, const bool &is_switch_on,
                                 std::vector<std::string> &pass_name_vec) const;

 private:
  Status SortGraphFusionForScene(const bool &is_single_op_scene,
                                 const std::map<std::string, int32_t> &graph_fusion_priority_map,
                                 std::vector<FusionPassOrRule> &sorted_graph_fusion_vec) const;
  Status SortBufferFusionForScene(const bool &can_close_buffer_fusion,
                                  const std::map<std::string, int32_t> &buffer_fusion_priority_map,
                                  std::vector<BufferFusionInfo> &sorted_buffer_fusion_vec) const;
  Status LoadGraphPriorityCfg(std::map<std::string, int32_t> &graph_fusion_priority_map) const;

  Status LoadBufferPriorityCfg(std::map<std::string, int32_t> &buffer_fusion_priority_map) const;

  Status GetGraphFusionPassesAndRules(const bool &is_single_op_scene,
                                      std::vector<FusionPassOrRule> &custom_pass_or_rule_vec,
                                      std::vector<FusionPassOrRule> &built_in_pass_or_rule_vec) const;

  void ModifyGraphFusionPriority(const std::map<std::string, int32_t> &graph_fusion_priority_map,
                                 bool &has_configured_custom_priority,
                                 std::vector<FusionPassOrRule> &custom_pass_or_rule_vec,
                                 std::vector<FusionPassOrRule> &built_in_pass_or_rule_vec) const;

  void SortFusionPassOrRule(const bool &has_configured_custom_priority,
                            std::vector<FusionPassOrRule> &custom_pass_or_rule_vec,
                            std::vector<FusionPassOrRule> &built_in_pass_or_rule_vec,
                            std::vector<FusionPassOrRule> &sorted_graph_fusion_vec) const;

  Status GetGraphFusionPassInfosByType(const GraphFusionPassType &pass_type,
                                       const bool &is_single_op_scene,
                                       vector<FusionPassOrRule> &graph_fusion_pass_vector) const;

  Status GetGraphFusionRuleInfosByType(RuleType type, vector<FusionPassOrRule> &graph_fusion_rule_vector) const;

  Status GetBufferFusionPassInfosByType(const BufferFusionPassType &pass_type,
                                        const bool &can_close_buffer_fusion,
                                        vector<BufferFusionInfo> &buffer_fusion_pass_infos) const;

  static int32_t AdjustDownStagePriority(int32_t priority);

  bool CanCloseBufferFusion(const ge::ComputeGraph& graph) const;

  static size_t GetCurrentHashedKey();

  std::string engine_name_;
  std::mutex fusion_pass_lock_;
  FusionRuleManagerPtr rule_manager_ptr_{nullptr};
  FusionConfigParserPtr fusion_config_parser_ptr_{nullptr};
  FusionPassOrRuleMap sorted_graph_fusion_map_;
  BufferFusionInfoMap sorted_buffer_fusion_map_;
  FusionPassOrRuleMap sorted_graph_fusion_single_scene_map_;
  BufferFusionInfoMap sorted_buffer_fusion_single_scene_map_;
};
using FusionPriorityMgrPtr = std::shared_ptr<FusionPriorityManager>;
}  // namespace fe
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_FUSION_FUSION_CONFIG_MANAGER_FUSION_PRIORITY_MANAGER_H_

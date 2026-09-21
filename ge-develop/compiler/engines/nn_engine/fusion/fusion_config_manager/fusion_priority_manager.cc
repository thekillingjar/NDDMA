/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fusion_config_manager/fusion_priority_manager.h"
#include <map>
#include <memory>
#include <string>
#include <utility>
#include<functional>
#include "common/configuration.h"
#include "graph/ge_local_context.h"
#include "common/fe_context_utils.h"

namespace fe {
namespace {
bool GraphFusionPrioritySort(const FusionPassOrRule &fusion_pass_or_rule1,
                             const FusionPassOrRule &fusion_pass_or_rule2) {
  return fusion_pass_or_rule1.priority < fusion_pass_or_rule2.priority;
}

bool BufferFusionPrioritySort(const BufferFusionInfo &buffer_fusion_info1,
                              const BufferFusionInfo &buffer_fusion_info2) {
  return buffer_fusion_info1.priority < buffer_fusion_info2.priority;
}
}

FusionPriorityManager::FusionPriorityManager(std::string engine_name, FusionRuleManagerPtr fusion_rule_mgr_ptr)
    : engine_name_(std::move(engine_name)),
      rule_manager_ptr_(std::move(fusion_rule_mgr_ptr)) {
  fusion_config_parser_ptr_ = std::unique_ptr<FusionConfigParser>(new (std::nothrow) FusionConfigParser(engine_name_));
}

FusionPriorityManager::~FusionPriorityManager() = default;

Status FusionPriorityManager::Initialize() {
  FE_CHECK_NOTNULL(fusion_config_parser_ptr_);
  fusion_config_parser_ptr_->ParseSupportFusionPassFile();
  return fusion_config_parser_ptr_->ParseFusionConfigFile();
}

const std::vector<FusionPassOrRule>& FusionPriorityManager::GetSortedGraphFusionList(const bool is_single_scene) {
  size_t hash_key = GetCurrentHashedKey();
  std::lock_guard<std::mutex> lock(fusion_pass_lock_);

  if (is_single_scene) {
    auto scene_itor = sorted_graph_fusion_single_scene_map_.find(hash_key);
    if (scene_itor != sorted_graph_fusion_single_scene_map_.end()) {
      return scene_itor->second;
    }
  } else {
    auto itor = sorted_graph_fusion_map_.find(hash_key);
    if (itor != sorted_graph_fusion_map_.end()) {
      return itor->second;
    }
  }
  
  static std::vector<FusionPassOrRule> empty_result;
  return empty_result;
}
const std::vector<BufferFusionInfo>& FusionPriorityManager::GetSortedBufferFusionList(
                                                            const ge::ComputeGraph &graph) {
  size_t hash_key = GetCurrentHashedKey();
  std::lock_guard<std::mutex> lock(fusion_pass_lock_);

  if (CanCloseBufferFusion(graph)) {
    auto scene_itor = sorted_buffer_fusion_single_scene_map_.find(hash_key);
    if (scene_itor != sorted_buffer_fusion_single_scene_map_.end()) {
      return scene_itor->second;
    }
  } else {
    auto itor = sorted_buffer_fusion_map_.find(hash_key);
    if (itor != sorted_buffer_fusion_map_.end()) {
      return itor->second;
    }
  }
  
  static std::vector<BufferFusionInfo> empty_result;
  return empty_result;
}

Status FusionPriorityManager::SortGraphFusion() {
  FE_LOGD("SortGraphFusion start.");
  FE_TIMECOST_START(SortGraphFusion);
  size_t hash_key = GetCurrentHashedKey();
  std::lock_guard<std::mutex> lock(fusion_pass_lock_);

  if (sorted_graph_fusion_map_.find(hash_key) != sorted_graph_fusion_map_.end()
        && sorted_graph_fusion_single_scene_map_.find(hash_key) != sorted_graph_fusion_single_scene_map_.end()) {
    FE_LOGD("SortGraphFusion has been inited.");
    return SUCCESS;
  }
  // 1.load configured priority from configuration,
  std::map<std::string, int32_t> graph_fusion_priority_map;
  if (LoadGraphPriorityCfg(graph_fusion_priority_map) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][FusionConfig][SortGphFus] Failed to load configured fusion priority for engine:%s.",
                    engine_name_.c_str());
    return FAILED;
  }
  // 2. sort non single op scene graph fusion pass
  std::vector<FusionPassOrRule> sorted_graph_fusion_vec;
  if (SortGraphFusionForScene(false, graph_fusion_priority_map, sorted_graph_fusion_vec) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][FusionConfig][SortGphFus] Failed to sort graph fusion pass for non single op scene.");
    return FAILED;
  }
  // 2. sort single op scene graph fusion pass
  std::vector<FusionPassOrRule> sorted_graph_fusion_single_scene_vec;
  if (SortGraphFusionForScene(true, graph_fusion_priority_map, sorted_graph_fusion_single_scene_vec) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][FusionConfig][SortGphFus] Failed to sort graph fusion pass for single op scene.");
    return FAILED;
  }
  
  sorted_graph_fusion_map_.emplace(hash_key, sorted_graph_fusion_vec);
  sorted_graph_fusion_single_scene_map_.emplace(hash_key, sorted_graph_fusion_single_scene_vec);
  FE_LOGD("SortGraphFusion success.");
  FE_TIMECOST_END(SortGraphFusion, "FusionPriorityManager::SortGraphFusion");
  return SUCCESS;
}

Status FusionPriorityManager::SortGraphFusionForScene(const bool &is_single_op_scene,
                                                      const std::map<std::string, int32_t> &graph_fusion_priority_map,
                                                      std::vector<FusionPassOrRule> &sorted_graph_fusion_vec) const {
  // 1.init sorted_custom_pass_or_rule_vector_ and sorted_built_in_pass_or_rule_vector_
  std::vector<FusionPassOrRule> custom_pass_or_rule_vec;
  std::vector<FusionPassOrRule> built_in_pass_or_rule_vec;
  if (GetGraphFusionPassesAndRules(is_single_op_scene, custom_pass_or_rule_vec, 
                                    built_in_pass_or_rule_vec) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][FusionConfig][SortGphFus] Failed to init passes and rules for engine:%s.",
                    engine_name_.c_str());
    return FAILED;
  }

  // 2.find the configured fusion in sorted_custom_pass_or_rule_vector_ and sorted_built_in_pass_or_rule_vector_
  // then change the priority
  bool has_configured_custom_priority = false;
  ModifyGraphFusionPriority(graph_fusion_priority_map, has_configured_custom_priority,
                            custom_pass_or_rule_vec, built_in_pass_or_rule_vec);

  // 3.sort sorted_custom_pass_or_rule_vector_ and sorted_built_in_pass_or_rule_vector_ by priority,
  // then combine them to init sorted_graph_fusion_map_
  sorted_graph_fusion_vec.clear();
  SortFusionPassOrRule(has_configured_custom_priority, custom_pass_or_rule_vec, built_in_pass_or_rule_vec,
                       sorted_graph_fusion_vec);
  return SUCCESS;
}

Status FusionPriorityManager::SortBufferFusion() {
  size_t hash_key = GetCurrentHashedKey();
  std::lock_guard<std::mutex> lock(fusion_pass_lock_);

  if (sorted_buffer_fusion_map_.find(hash_key) != sorted_buffer_fusion_map_.end()
        && sorted_buffer_fusion_single_scene_map_.find(hash_key) != sorted_buffer_fusion_single_scene_map_.end()) {
    FE_LOGD("SortBufferFusion has been inited.");
    return SUCCESS;
  }
  // 1.Load ub priority config
  std::map<std::string, int32_t> buffer_fusion_priority_map;
  if (LoadBufferPriorityCfg(buffer_fusion_priority_map) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][FusionConfig][SortBufFus] Failed to load configured buffer fusion priority in engine %s.",
                    engine_name_.c_str());
    return FAILED;
  }
  // 2. sort non single op scene buffer fusion pass
  std::vector<BufferFusionInfo> sorted_buffer_fusion_vec;
  if (SortBufferFusionForScene(false, buffer_fusion_priority_map, sorted_buffer_fusion_vec) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][FusionConfig][SortUbFus] Failed to sort buffer fusion pass for non single op scene.");
    return FAILED;
  }
  // 3. sort single op scene buffer fusion pass
  std::vector<BufferFusionInfo> sorted_buffer_fusion_single_scene_vec;
  if (SortBufferFusionForScene(true, buffer_fusion_priority_map, sorted_buffer_fusion_single_scene_vec) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][FusionConfig][SortUbFus] Failed to sort buffer fusion pass for single op scene.");
    return FAILED;
  }

  sorted_buffer_fusion_map_.emplace(hash_key, sorted_buffer_fusion_vec);
  sorted_buffer_fusion_single_scene_map_.emplace(hash_key, sorted_buffer_fusion_single_scene_vec);
  FE_LOGD("End to order buffer pass.");
  return SUCCESS;
}

Status FusionPriorityManager::SortBufferFusionForScene(const bool &can_close_buffer_fusion,
                                                       const std::map<std::string, int32_t> &buffer_fusion_priority_map,
                                                       std::vector<BufferFusionInfo> &sorted_buffer_fusion_vec) const {
  // 1.Get ub fusion pass from register
  sorted_buffer_fusion_vec.clear();
  if (engine_name_ == fe::AI_CORE_NAME) {
    // init built-in ai core pass
    if (GetBufferFusionPassInfosByType(BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, can_close_buffer_fusion,
                                       sorted_buffer_fusion_vec) != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][FusionConfig][SortGphFus] Failed to get buffer pass from this engine [%s]",
                      engine_name_.c_str());
      return FAILED;
    }
  } else {
    // init built-in vector pass
    if (GetBufferFusionPassInfosByType(BUILT_IN_VECTOR_CORE_BUFFER_FUSION_PASS, can_close_buffer_fusion,
                                       sorted_buffer_fusion_vec) != SUCCESS) {
      REPORT_FE_ERROR("[GraphOpt][FusionConfig][SortGphFus] Failed to get buffer pass from this engine [%s]",
                      engine_name_.c_str());
      return FAILED;
    }
  }

  // 2.Find the configured fusion in sorted_buffer_fusion_info_vector_ and configured_buffer_fusion_priority_map_
  // then change the priority
  for (const auto &iter : buffer_fusion_priority_map) {
    auto buffer_fusion_iter = find_if(sorted_buffer_fusion_vec.begin(), sorted_buffer_fusion_vec.end(),
                                      BufferFusionFinder(iter.first));
    if (buffer_fusion_iter != sorted_buffer_fusion_vec.end()) {
      buffer_fusion_iter->priority = AdjustDownStagePriority(iter.second);
      FE_LOGD("The priority of fusion:%s has been set to %d", iter.first.c_str(), iter.second);
      continue;
    }
    FE_LOGW("Could not find this buffer fusion: %s in engine:%s.", iter.first.c_str(), engine_name_.c_str());
  }
  FE_LOGD("End to add priority and register buffer fusion size: %zu", sorted_buffer_fusion_vec.size());
  sort(sorted_buffer_fusion_vec.begin(), sorted_buffer_fusion_vec.end(), BufferFusionPrioritySort);
  return SUCCESS;
}

Status FusionPriorityManager::LoadGraphPriorityCfg(std::map<std::string, int32_t> &graph_fusion_priority_map) const {
  return fusion_config_parser_ptr_->GetFusionPriorityByFusionType(GRAPH_FUSION, graph_fusion_priority_map);
}

Status FusionPriorityManager::LoadBufferPriorityCfg(std::map<std::string, int32_t> &buffer_fusion_priority_map) const {
  return fusion_config_parser_ptr_->GetFusionPriorityByFusionType(UB_FUSION, buffer_fusion_priority_map);
}

Status FusionPriorityManager::GetGraphFusionPassesAndRules(const bool &is_single_op_scene,
                                                           vector<FusionPassOrRule> &custom_pass_or_rule_vec,
                                                           vector<FusionPassOrRule> &built_in_pass_or_rule_vec) const {
  // init custom pass
  if (engine_name_ == fe::AI_CORE_NAME) {
    if (GetGraphFusionPassInfosByType(CUSTOM_AI_CORE_GRAPH_PASS, is_single_op_scene,
                                      custom_pass_or_rule_vec) != SUCCESS) {
      REPORT_FE_ERROR(
          "[GraphOpt][FusionConfig][InitPassRule] Failed to init sorted custom graph fusion passes for engine:%s",
          engine_name_.c_str());
      return FAILED;
    }
  } else {
    if (GetGraphFusionPassInfosByType(CUSTOM_VECTOR_CORE_GRAPH_PASS, is_single_op_scene,
                                      custom_pass_or_rule_vec) != SUCCESS) {
      REPORT_FE_ERROR(
          "[GraphOpt][FusionConfig][InitPassRule] Failed to init sorted custom graph fusion passes for engine:%s",
          engine_name_.c_str());
      return FAILED;
    }
  }

  // init custom rule
  vector<FusionPassOrRule> custom_graph_fusion_rule_infos;
  if (GetGraphFusionRuleInfosByType(RuleType::CUSTOM_GRAPH_RULE, custom_graph_fusion_rule_infos) != SUCCESS) {
    REPORT_FE_ERROR(
        "[GraphOpt][FusionConfig][InitPassRule] Failed to init sorted custom graph fusion rules for engine:%s",
        engine_name_.c_str());
    return FAILED;
  }
  custom_pass_or_rule_vec.insert(custom_pass_or_rule_vec.cend(),
                                 custom_graph_fusion_rule_infos.cbegin(), custom_graph_fusion_rule_infos.cend());

  // init built-in pass
  vector<GraphFusionPassType> pass_type_vec;
  if (engine_name_ == fe::AI_CORE_NAME) {
    pass_type_vec = GRAPH_FUSION_PASS_TYPE_AICORE_VEC;
  } else if (engine_name_ == fe::VECTOR_CORE_NAME) {
    pass_type_vec = GRAPH_FUSION_PASS_TYPE_VECTORCORE_VEC;
  }
  for (auto pass_type : pass_type_vec) {
    if (GetGraphFusionPassInfosByType(pass_type, is_single_op_scene, built_in_pass_or_rule_vec) != SUCCESS) {
      REPORT_FE_ERROR(
          "[GraphOpt][FusionConfig][InitPassRule] Failed to init sorted built-in graph fusion passes for engine:%s",
          engine_name_.c_str());
      return FAILED;
    }
  }

  // init built-in rule
  vector<FusionPassOrRule> built_in_graph_fusion_rule_infos;
  if (GetGraphFusionRuleInfosByType(RuleType::BUILT_IN_GRAPH_RULE,
                                    built_in_graph_fusion_rule_infos) != SUCCESS) {
    REPORT_FE_ERROR(
        "[GraphOpt][FusionConfig][InitPassRule] Failed to init sorted built-in graph fusion rules for engine:%s",
        engine_name_.c_str());
    return FAILED;
  }
  built_in_pass_or_rule_vec.insert(built_in_pass_or_rule_vec.cend(),
                                   built_in_graph_fusion_rule_infos.cbegin(), built_in_graph_fusion_rule_infos.cend());
  return SUCCESS;
}

Status FusionPriorityManager::GetGraphFusionPassInfosByType(const GraphFusionPassType &pass_type,
                                                            const bool &is_single_op_scene,
                                                            vector<FusionPassOrRule> &graph_fusion_pass_vector) const {
  std::map<std::string, FusionPassRegistry::PassDesc> graph_fusion_descs =
          FusionPassRegistry::GetInstance().GetPassDesc(pass_type);
  if (graph_fusion_descs.empty()) {
    FE_LOGD("No registered graph fusion pass was found, type[%s], engine[%s].", GetPassTypeString(pass_type).c_str(),
            engine_name_.c_str());
    return SUCCESS;
  }
  std::string method;
  int32_t priority;
  switch (pass_type) {
    case CUSTOM_AI_CORE_GRAPH_PASS:
    case CUSTOM_VECTOR_CORE_GRAPH_PASS:
      method = PASS_METHOD;
      priority = CUSTOM_PASS_PRIORITY_MIN;
      break;

    case BUILT_IN_PREPARE_GRAPH_PASS:
    case BUILT_IN_AFTER_MULTI_DIMS_PASS:
    case BUILT_IN_BEFORE_QUANT_OPTIMIZATION_GRAPH_PASS:
    case BUILT_IN_TF_TAG_NO_CONST_FODING_GRAPH_PASS:
    case BUILT_IN_TF_MERGE_SUB_GRAPH_PASS:
    case BUILT_IN_QUANT_OPTIMIZATION_GRAPH_PASS:
    case BUILT_IN_EN_ISA_ARCH_EXC_V300_AND_V220_GRAPH_PASS:
    case BUILT_IN_EN_ISA_ARCH_V100_GRAPH_PASS:
    case BUILT_IN_EN_ISA_ARCH_V200_GRAPH_PASS:
    case BUILT_IN_DELETE_NO_CONST_FOLDING_GRAPH_PASS:
    case BUILT_IN_AFTER_OPTIMIZE_STAGE1:
    case BUILT_IN_GRAPH_PASS:
    case BUILT_IN_VECTOR_CORE_GRAPH_PASS:
    case SECOND_ROUND_BUILT_IN_GRAPH_PASS:
    case BUILT_IN_AFTER_OP_JUDGE:
    case BUILT_IN_BEFORE_TRANSNODE_INSERTION_GRAPH_PASS:
    case BUILT_IN_AFTER_BUFFER_OPTIMIZE:
      method = PASS_METHOD;
      priority = BUILT_IN_PASS_PRIORITY_MIN;
      break;

    default:
      REPORT_FE_ERROR("[GraphOpt][FusionConfig][GetGphFusPassInfo] The pass type[%s] does not support priority order.",
                      GetPassTypeString(pass_type).c_str());
      return FAILED;
  }

  for (const auto &iter : graph_fusion_descs) {
    if (fusion_config_parser_ptr_->CheckPassIsWhiteListControl(iter.first, pass_type)) {
      continue;
    }

    if (!fusion_config_parser_ptr_->GetFusionSwitchByName(iter.first, GRAPH_FUSION, iter.second.attr,
                                                          is_single_op_scene)) {
      FE_LOGD("The graph fusion pass[%s] switch is off, single op scene value is %d.", iter.first.c_str(), is_single_op_scene);
      continue;
    }
    FE_LOGD("Load registered graph fusion pass(switch on): %s", iter.first.c_str());
    graph_fusion_pass_vector.emplace_back(iter.first, static_cast<int32_t>(pass_type), method, priority,
                                          iter.second);
    priority++;
  }
  
  FE_LOGI("The total number of pass(switch on) for type[%s] is %zu.", GetPassTypeString(pass_type).c_str(),
          graph_fusion_pass_vector.size());
  return SUCCESS;
}

Status FusionPriorityManager::GetGraphFusionRuleInfosByType(RuleType type,
                                                            vector<FusionPassOrRule> &graph_fusion_rule_vector) const {
  FE_CHECK_NOTNULL(rule_manager_ptr_);
  std::vector<FusionRulePatternPtr> graph_fusion_rules;
  if (rule_manager_ptr_->GetFusionRulesByRuleType(type, graph_fusion_rules) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][FusionConfig][GetGphFusRuleInfo] Failed to obtain graph fusion rules for type[%s].",
                    GetRuleTypeString(type).c_str());
    return FAILED;
  }
  if (graph_fusion_rules.empty()) {
    FE_LOGD("No graph fusion rule was found, type[%s], engine[%s].", GetRuleTypeString(type).c_str(),
            engine_name_.c_str());
    return SUCCESS;
  }
  std::string method;
  int32_t priority;
  switch (type) {
    case RuleType::CUSTOM_GRAPH_RULE:
      method = RULE_METHOD;
      priority = CUSTOM_RULE_PRIORITY_MIN;
      break;
    case RuleType::BUILT_IN_GRAPH_RULE:
      method = RULE_METHOD;
      priority = BUILT_IN_RULE_PRIORITY_MIN;
      break;
    default:
      REPORT_FE_ERROR("[GraphOpt][FusionConfig][GetGphFusRuleInfo] The rule type[%s] does not support priority order.",
                      GetRuleTypeString(type).c_str());
      return FAILED;
  }
  for (const FusionRulePatternPtr &one_rule : graph_fusion_rules) {
    std::string name = one_rule->GetRuleName();
    if (!fusion_config_parser_ptr_->GetFusionSwitchByName(name, GRAPH_FUSION)) {
      FE_LOGD("The graph fusion rule[%s] switch is off.", name.c_str());
      continue;
    }
    FE_LOGD("Load graph fusion rule(switch on): %s", name.c_str());
    FusionPassRegistry::PassDesc pass_desc = {0, nullptr};
    graph_fusion_rule_vector.emplace_back(name, static_cast<int32_t>(type), method, priority, pass_desc);
    priority++;
  }
  FE_LOGD("The total number of rule(switch on) for type[%s] is %zu.", GetRuleTypeString(type).c_str(),
          graph_fusion_rule_vector.size());
  return SUCCESS;
}

Status FusionPriorityManager::GetBufferFusionPassInfosByType(const BufferFusionPassType &pass_type,
                                                             const bool &can_close_buffer_fusion,
                                                             vector<BufferFusionInfo> &buffer_fusion_pass_infos) const {
  std::map<std::string, BufferFusionPassRegistry::PassDesc> buffer_fusion_descs =
          BufferFusionPassRegistry::GetInstance().GetPassDesc(pass_type);
  if (buffer_fusion_descs.empty()) {
    FE_LOGD("GetUbFusion-PassType[%s]: registered buffer fusion passes are empty.",
            GetBufferFusionPassTypeString(pass_type).c_str());
    return SUCCESS;
  }
  int64_t priority = BUILT_IN_PASS_PRIORITY_MIN;
  for (const auto &iter : buffer_fusion_descs) {
    if (!fusion_config_parser_ptr_->GetFusionSwitchByName(iter.first, UB_FUSION, iter.second.attr,
                                                          can_close_buffer_fusion)) {
      FE_LOGD("The ub fusion pass [%s] switch is off, single op scene value is %d.", iter.first.c_str(), can_close_buffer_fusion);
      continue;
    }
    bool is_auto_fusion = IsPassAttrTypeOn(iter.second.attr, PassAttrType::AUTO_FUSION_FLAG);
    bool is_fusion_check = IsPassAttrTypeOn(iter.second.attr, PassAttrType::FUSION_CHECK_FLAG);
    FE_LOGD("Start to load registered buffer fusion passes (on) : %s, is auto fusion[%d], is fusion check[%d].",
            iter.first.c_str(), is_auto_fusion, is_fusion_check);
    buffer_fusion_pass_infos.emplace_back(iter.first, priority, is_auto_fusion, is_fusion_check, iter.second.create_fn);
    priority++;
  }
  FE_LOGD("GetUbFusion-PassType[%s]: end to get BufferFusionPass.",
          GetBufferFusionPassTypeString(pass_type).c_str());
  return SUCCESS;
}

void FusionPriorityManager::ModifyGraphFusionPriority(const std::map<std::string, int32_t> &graph_fusion_priority_map,
                                                      bool &has_configured_custom_priority,
                                                      std::vector<FusionPassOrRule> &custom_pass_or_rule_vec,
                                                      std::vector<FusionPassOrRule> &built_in_pass_or_rule_vec) const {
  if (graph_fusion_priority_map.empty()) {
    return;
  }
  for (auto iter = graph_fusion_priority_map.cbegin(); iter != graph_fusion_priority_map.cend(); iter++) {
    auto fusion_info_iter = find_if(custom_pass_or_rule_vec.begin(),
                                    custom_pass_or_rule_vec.end(), FusionPassOrRuleFinder(iter->first));
    if (fusion_info_iter != custom_pass_or_rule_vec.end()) {
      has_configured_custom_priority = true;
      fusion_info_iter->priority = AdjustDownStagePriority(iter->second);
      FE_LOGD("The priority of fusion:%s has been set to %d", iter->first.c_str(), iter->second);
      continue;
    }
    fusion_info_iter = find_if(built_in_pass_or_rule_vec.begin(), built_in_pass_or_rule_vec.end(),
                               FusionPassOrRuleFinder(iter->first));
    if (fusion_info_iter != built_in_pass_or_rule_vec.end()) {
      fusion_info_iter->priority = AdjustDownStagePriority(iter->second);
      FE_LOGD("The priority of fusion:%s has been set to %d", iter->first.c_str(), iter->second);
      continue;
    }
    FE_LOGW("Could not find fusion:%s in engine:%s.", iter->first.c_str(), engine_name_.c_str());
  }
}

void FusionPriorityManager::SortFusionPassOrRule(const bool &has_configured_custom_priority,
                                                 std::vector<FusionPassOrRule> &custom_pass_or_rule_vec,
                                                 std::vector<FusionPassOrRule> &built_in_pass_or_rule_vec,
                                                 std::vector<FusionPassOrRule> &sorted_graph_fusion_vec) const {
  if (!has_configured_custom_priority) {
    // if the configuration file only has priority for built-in fusion,
    // only sort the built-in vector and insert the custom vector before the built-in vector.
    sort(built_in_pass_or_rule_vec.begin(), built_in_pass_or_rule_vec.end(), GraphFusionPrioritySort);
    sorted_graph_fusion_vec.insert(sorted_graph_fusion_vec.cend(), custom_pass_or_rule_vec.cbegin(),
                                   custom_pass_or_rule_vec.cend());
    sorted_graph_fusion_vec.insert(sorted_graph_fusion_vec.cend(), built_in_pass_or_rule_vec.cbegin(),
                                   built_in_pass_or_rule_vec.cend());
  } else {
    // if the configuration file has priority both for built-in fusion and custom fusion,
    // combine the built-in vector and the custom vector, then sort it.
    sorted_graph_fusion_vec.insert(sorted_graph_fusion_vec.cend(), custom_pass_or_rule_vec.cbegin(),
                                   custom_pass_or_rule_vec.cend());
    sorted_graph_fusion_vec.insert(sorted_graph_fusion_vec.cend(), built_in_pass_or_rule_vec.cbegin(),
                                   built_in_pass_or_rule_vec.cend());
    sort(sorted_graph_fusion_vec.begin(), sorted_graph_fusion_vec.end(), GraphFusionPrioritySort);
  }
}

int32_t FusionPriorityManager::AdjustDownStagePriority(int32_t priority) {
  int32_t adjusted_priority = priority;
  if ((priority >= CUSTOM_CFG_DOWN_PRIORITY_MIN && priority < BUILT_IN_CFG_TOP_PRIORITY_MIN) ||
      (priority >= BUILT_IN_CFG_DOWN_PRIORITY_MIN && priority < CUSTOM_PASS_PRIORITY_MIN)) {
    adjusted_priority += RESERVED_FOR_DOWN_PRIORITY;
  }
  return adjusted_priority;
}

int32_t FusionPriorityManager::GetRealPriority(const int32_t priority) {
  int32_t real_priority = priority;
  if (priority > RESERVED_FOR_DOWN_PRIORITY) {
    real_priority -= RESERVED_FOR_DOWN_PRIORITY;
  }
  return real_priority;
}

bool FusionPriorityManager::GetFusionSwitchByName(const std::string &pass_name, const std::string &pass_type,
                                                  const uint64_t pass_attr, const bool can_close_fusion) const {
  return fusion_config_parser_ptr_->GetFusionSwitchByName(pass_name, pass_type, pass_attr, can_close_fusion);
}

void FusionPriorityManager::GetFusionPassNameBySwitch(const std::string &pass_type, const bool &is_switch_on,
                                                      std::vector<std::string> &pass_name_vec) const {
  fusion_config_parser_ptr_->GetFusionPassNameBySwitch(pass_type, is_switch_on, pass_name_vec);
}

BufferFusionPassBasePtr FusionPriorityManager::GetBufferFusionByPassName(const std::string &pass_name) const {
  BufferFusionPassType pass_type = engine_name_ == fe::AI_CORE_NAME ? BUILT_IN_AI_CORE_BUFFER_FUSION_PASS :
                                                                      BUILT_IN_VECTOR_CORE_BUFFER_FUSION_PASS;
  std::map<string, BufferFusionPassRegistry::CreateFn> create_fns =
          BufferFusionPassRegistry::GetInstance().GetCreateFnByType(pass_type);
  const std::map<string, BufferFusionPassRegistry::CreateFn>::const_iterator iter = create_fns.find(pass_name);
  if (iter == create_fns.cend()) {
    FE_LOGD("Buffer fusion pass[%s] is not found in registered fusion passes.", pass_name.c_str());
    return nullptr;
  }
  BufferFusionPassBasePtr buffer_fusion_pass_ptr = std::unique_ptr<BufferFusionPassBase>(iter->second());
  return buffer_fusion_pass_ptr;
}

bool FusionPriorityManager::CanCloseBufferFusion(const ge::ComputeGraph& graph) const {
  if (IsSingleOpGraph(graph)) {
    return true;
  }

  return (graph.GetGraphUnknownFlag() &&
      Configuration::Instance(engine_name_).GetJitCompileCfg() == JitCompileCfg::CFG_FALSE);
}

size_t FusionPriorityManager::GetCurrentHashedKey() {
  std::string opt_val = "";
  if (ge::GetThreadLocalContext().GetOption("ge.optimizationSwitch", opt_val) != ge::GRAPH_SUCCESS) {
    FE_LOGD("Get optimization switch failed.");
  }
  std::string opt_oolevel_val = "";
  (void)ge::GetThreadLocalContext().GetOo().GetValue(kComLevelO1Opt, opt_oolevel_val);

  string custom_fusion_config_json_file = FEContextUtils::GetFusionSwitchFilePath();

  std::hash<std::string> hasher;
  size_t res = hasher(opt_val + opt_oolevel_val + custom_fusion_config_json_file);
  FE_LOGI("Get optimization switch: [%s], O3 level: [%s], custom fusion config file: [%s], hashed value: [%zu]", 
          opt_val.c_str(), opt_oolevel_val.c_str(), custom_fusion_config_json_file.c_str(), res);
  return res;
}
}  // namespace fe

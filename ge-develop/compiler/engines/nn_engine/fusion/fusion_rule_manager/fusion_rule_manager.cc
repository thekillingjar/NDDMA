/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fusion_rule_manager/fusion_rule_manager.h"

#include <algorithm>
#include "common/configuration.h"
#include "common/fe_utils.h"
#include "common/fe_type_utils.h"
#include "common/fe_report_error.h"
#include "common/util/json_util.h"
#include "fusion_rule_manager/fusion_rule_parser/fusion_rule_parser_utils.h"
#include "fusion_rule_manager/fusion_cycle_detector.h"
#include "graph_optimizer/fusion_common/graph_node_map_util.h"
#include "graph_optimizer/graph_fusion/graph_matcher.h"
#include "graph_optimizer/graph_fusion/graph_replace.h"

using std::string;
using std::vector;

namespace fe {
const std::unordered_set<std::string> rules_need_cycle_detection = {
    "LayerNormGradFusionRule",
    "LayerNormGradFusionRule2"
};

namespace {
/* compare the priority of fusion rules based on the number of nodes */
bool CompareByRuleNode(FusionRulePatternPtr rule_a, FusionRulePatternPtr rule_b) {
  if (rule_a == nullptr || rule_b == nullptr) {
    return false;
  }
  return rule_a->GetOriginRuleNodes().size() > rule_b->GetOriginRuleNodes().size();
}

const string RULE_KEY = "Rules";

/* if dtype pf one node of matched graph is int64 or uint64,attach invalid_flag to this matched graph */
void CheckDtypeNotExpected(vector<GraphMatchResult> &match_results) {
  FE_LOGD("start check match_results dtype.");
  for (auto &matched_graph : match_results) {
    for (const auto &node : matched_graph.origin_nodes) {
      auto num_intputs = node.second->GetOpDesc()->GetAllInputsSize();
      auto num_outputs = node.second->GetOpDesc()->GetAllOutputsDescSize();
      for (size_t index = 0; index < num_intputs; ++index) {
        auto input_tensor_desc = node.second->GetOpDesc()->MutableInputDesc(index);
        if (input_tensor_desc == nullptr) {
          REPORT_FE_ERROR("input_tensor_desc is nullptr.");
          return;}
        auto dtype = input_tensor_desc->GetDataType();
        if (dtype == ge::DT_INT64 || dtype == ge::DT_UINT64) {
          matched_graph.valid_flag = false;
          break;
        }
      }
      for (uint32_t index = 0; index < num_outputs; ++index) {
        auto output_tensor_desc = node.second->GetOpDesc()->MutableOutputDesc(index);
        auto dtype = output_tensor_desc->GetDataType();
        if (dtype == ge::DT_INT64 || dtype == ge::DT_UINT64) {
          matched_graph.valid_flag = false;
          break;
        }
      }
    }
  }
}

/* If one matched graph will lead to loop, just erase it. */
void CorrectMatchedResult(const string& rule_name,
                          const ge::ComputeGraph &graph,
                          const std::shared_ptr<FusionCycleDetector> &detector,
                          vector<GraphMatchResult> &matched_graphs,
                          size_t &invalid_count) {
  CheckDtypeNotExpected(matched_graphs);
  if (rules_need_cycle_detection.count(rule_name) == 0) {
    return;
  }

  std::vector<std::vector<ge::NodePtr>> all_scope_nodes;
  for (auto &matched : matched_graphs) {
    std::vector<ge::NodePtr> one_scope_nodes;
    for (auto &node : matched.origin_nodes) {
      one_scope_nodes.emplace_back(node.second);
    }
    if (!one_scope_nodes.empty()) {
      FE_LOGD("Correct %s with first node %s.", rule_name.c_str(),
              one_scope_nodes[0]->GetName().c_str());
    }

    all_scope_nodes.emplace_back(one_scope_nodes);
    if (detector->CycleDetection(graph, all_scope_nodes)) {
      FE_LOGD("Loop detected %s.", one_scope_nodes[0]->GetName().c_str());
      matched.valid_flag = false;
      ++invalid_count;
    }
  }
}
}  // namespace

FusionRuleManager::FusionRuleManager(OpsKernelInfoStorePtr ops_kernel_info_store_ptr)
    : ops_kernel_info_store_ptr_(ops_kernel_info_store_ptr),
      init_flag_(false),
      graph_rule_vector_(),
      custom_rule_vector_() {}

FusionRuleManager::~FusionRuleManager() {}

/*
 * obtain fusion rules from jsons
 * sort fusion rules based on the number of nodes
 */
Status FusionRuleManager::Initialize(const std::string &engine_name) {
  FE_LOGD("Initialize FusionRuleManager start.");
  if (init_flag_) {
    FE_LOGW("FusionRuleManager has been initialized.");
    return SUCCESS;
  }

  Status ret = InitGraphRules(engine_name);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][init] Init graph rules failed.");
    graph_rule_vector_.clear();
    custom_rule_vector_.clear();
    return ret;
  }

  FE_LOGI("Initialize FusionRuleManager successfully.");
  init_flag_ = true;
  return SUCCESS;
}

/*
 * obtain fusion rules based on rule types
 * param[in] RuleType ruletype, which is defined by the users
 * param[out] vector<FusionRulePattern> &out_fusion_rules, which is get from the
 * initialize func
 * return Status
 */
Status FusionRuleManager::GetFusionRulesByRuleType(RuleType rule_type,
                                                   vector<FusionRulePatternPtr> &out_fusion_rules) const {
  if (!init_flag_) {
    REPORT_FE_ERROR("[GraphOpt][RunFusionRule][GetFusRule] FusionRuleManager has not been initialized.");
    return FAILED;
  }

  if (rule_type == RuleType::BUILT_IN_GRAPH_RULE) {
    out_fusion_rules = graph_rule_vector_;
    FE_LOGD("Get built_in_graph_rules success.");
    return SUCCESS;
  } else if (rule_type == RuleType::CUSTOM_GRAPH_RULE) {
    out_fusion_rules = custom_rule_vector_;
    FE_LOGD("Get custom_graph_rules success.");
    return SUCCESS;
  } else {
    REPORT_FE_ERROR("[GraphOpt][RunFusionRule][GetFusRule] RuleType is not supported.");
    return FAILED;
  }
}

Status FusionRuleManager::Finalize() {
  FE_LOGD("Finalize start.");
  if (!init_flag_) {
    FE_LOGW("FusionRuleManager has not been initialized.");
    return SUCCESS;
  }
  graph_rule_vector_.clear();
  custom_rule_vector_.clear();
  init_flag_ = false;
  FE_LOGD("Finalize success.");
  return SUCCESS;
}

Status FusionRuleManager::InitGraphRules(const std::string &engine_name) {
  string custom_file_path;
  string graph_file_path;
  if (Configuration::Instance(engine_name).GetCustomFilePath(custom_file_path) == SUCCESS) {
    if (custom_file_path.empty()) {
      FE_LOGW("Input custom graph fusion rule json file path is null.");
    } else {
      FusionRuleParserUtils::Instance()->SetEngineName(engine_name);
      Status ret = LoadFusionRule(custom_file_path, custom_rule_vector_);
      if (ret != SUCCESS) {
        REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][InitGphRule] Load %s failed.", custom_file_path.c_str());
        return ret;
      }
    }
  }
  if (Configuration::Instance(engine_name).GetGraphFilePath(graph_file_path) == SUCCESS) {
    if (graph_file_path.empty()) {
      FE_LOGW("Input built-in graph fusion rule json file path is null.");
    } else {
      FusionRuleParserUtils::Instance()->SetEngineName(engine_name);
      Status ret = LoadFusionRule(graph_file_path, graph_rule_vector_);
      if (ret != SUCCESS) {
        REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][InitGphRule] Load [%s] failed.", graph_file_path.c_str());
        return ret;
      }
    }
  }
  if (!custom_rule_vector_.empty()) {
    std::sort(custom_rule_vector_.begin(), custom_rule_vector_.end(), CompareByRuleNode);
    /* Get every custom graph rule's name in debug mod */
    for (const FusionRulePatternPtr &custom_rule_ptr : custom_rule_vector_) {
      std::string custom_rule_name = custom_rule_ptr->GetRuleName();
      FE_LOGD("The name of loaded custom graph rule is [%s]", custom_rule_name.c_str());
    }
    FE_LOGD("The number of loaded custom graph rules is %zu", custom_rule_vector_.size());
  } else {
    FE_LOGI("No custom graph rules were read.");
  }
  if (!graph_rule_vector_.empty()) {
    std::sort(graph_rule_vector_.begin(), graph_rule_vector_.end(), CompareByRuleNode);
    /* Get every built-in graph rule's name in debug mod */
    for (const FusionRulePatternPtr &graph_rule_ptr : graph_rule_vector_) {
      std::string graph_rule_name = graph_rule_ptr->GetRuleName();
      FE_LOGD("The name of loaded built-in graph rule is [%s]", graph_rule_name.c_str());
    }
    FE_LOGD("The number of loaded built-in graph rules is %zu", graph_rule_vector_.size());
  } else {
    FE_LOGI("No built-in graph rules were read.");
  }
  return SUCCESS;
}

Status FusionRuleManager::LoadFusionRule(const string &file_path, vector<FusionRulePatternPtr> &fusion_rule_patterns) {
  vector<nlohmann::json> fusion_rule_json_objects;
  Status ret = OpenJsonFileToItems(file_path, fusion_rule_json_objects);
  if (ret != SUCCESS) {
    ErrorMessageDetail err_msg(EM_READ_FILE_FAILED, {file_path,
                               "The configuration file is not in JSON format or its content is invalid"});
    ReportErrorMessage(err_msg);
    FE_LOGE("[GraphOpt][FusionRuleInit][LdFusRule] Open json file [%s] to json fusion rule objects failed.",
            file_path.c_str());
    return ret;
  }
  FE_LOGD("Start parsing and loading file:%s", file_path.c_str());

  size_t fusion_rule_cnt = 0;
  for (const auto &fusion_rule_json_object : fusion_rule_json_objects) {
    FusionRuleJsonPatternPtr fusion_rule_json_pattern = nullptr;
    FE_MAKE_SHARED(fusion_rule_json_pattern = make_shared<FusionRuleJsonPattern>(), return INTERNAL_ERROR);
    // Step 1: parse fusion rule from json type to c++ type
    ret = fusion_rule_json_pattern->ParseToJsonPattern(fusion_rule_json_object);
    if (ret == INVALID_RULE) {
      FE_LOGW(
          "Fusion rule:[%s] exist node can't find OpKernelInfo, skip "
          "parser&load procedure.",
          fusion_rule_json_pattern->GetName().c_str());
      continue;
    } else if (ret != SUCCESS) {
      ErrorMessageDetail err_msg(EM_READ_FILE_FAILED, {file_path,
                                 "The configuration file is not in JSON format or its content is invalid"});
      ReportErrorMessage(err_msg);
      FE_LOGE("[GraphOpt][FusionRuleInit][LdFusRule] Parse json type fusion rule to c++ struct failed.");
      return ret;
    }
    // Step 2: check and load fusion rule to FusionRulePattern
    FusionRulePatternPtr fusion_rule_pattern = nullptr;
    FE_MAKE_SHARED(fusion_rule_pattern = make_shared<FusionRulePattern>(), return INTERNAL_ERROR);
    ret = FusionRulePatternConstructor::Construct(fusion_rule_pattern, fusion_rule_json_pattern);
    if (ret != SUCCESS) {
      ErrorMessageDetail err_msg(EM_INVALID_CONTENT, {"fusion_rule", file_path,
                                 "Invalid pattern of fusion rule: " + fusion_rule_json_pattern->GetName() + ""});
      ReportErrorMessage(err_msg);
      FE_LOGE("[GraphOpt][FusionRuleInit][LdFusRule] Check and load fusion rule:[%s] to FusionRulePattern failed.",
              fusion_rule_json_pattern->GetName().c_str());
      return ret;
    }
    fusion_rule_patterns.push_back(fusion_rule_pattern);
    FE_LOGD("Parse and Load fusion rule:[%s] success.", fusion_rule_pattern->GetRuleName().c_str());
    fusion_rule_cnt++;
  }
  FE_LOGD("Parse and Load file:[%s] success, totally load %zu fusion rules.", file_path.c_str(), fusion_rule_cnt);

  return SUCCESS;
}

Status FusionRuleManager::OpenJsonFileToItems(const string &file_path,
                                              vector<nlohmann::json> &fusion_rule_json_objects) {
  // Try open json file
  nlohmann::json input_json;
  if (ReadJsonFile(file_path, input_json) != SUCCESS) {
    ErrorMessageDetail err_msg(EM_READ_FILE_FAILED, {file_path,
                               "The configuration file is not in JSON format or its content is invalid"});
    ReportErrorMessage(err_msg);
    FE_LOGE("[GraphOpt][FusionRuleInit][OpenJsFileToItem] ReadJsonFile in %s failed.", file_path.c_str());
    return FAILED;
  }
  try {
    // Top level of json file should be objects
    if (!input_json.is_object()) {
      REPORT_FE_ERROR(
          "[GraphOpt][FusionRuleInit][OpenJsFileToItem] Top level of json file should be object, actually is %s.",
          GetJsonType(input_json).c_str());
      return ILLEGAL_JSON;
    }
    // Get each item to fusion_rule_json_objects
    for (const auto &item : input_json.items()) {
      if (item.key() == RULE_KEY) {
        // Value of "Rules" must be array type
        if (!item.value().is_array()) {
          REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][OpenJsFileToItem] Item of %s should be array, actually is %s.",
                          RULE_KEY.c_str(), GetJsonType(item.value()).c_str());
          return ILLEGAL_JSON;
        }
        for (const auto &fusion_rule : item.value()) {
          fusion_rule_json_objects.push_back(fusion_rule);
        }
      } else {
        REPORT_FE_ERROR(
            "[GraphOpt][FusionRuleInit][OpenJsFileToItem] Not supported key:%s in top level of fusion rule json file.",
            item.key().c_str());
        return ILLEGAL_JSON;
      }
    }
  } catch (const std::exception &e) {
    REPORT_FE_ERROR("[GraphOpt][FusionRuleInit][OpenJsFileToItem] Error message is %s", e.what());
    return ILLEGAL_JSON;
  }

  return SUCCESS;
}

Status FusionRuleManager::MatchAndReplaceByRules(ge::ComputeGraph &graph, const std::string &rule_name,
                                                 const vector<FusionRulePatternPtr>::iterator &rule_pattern_ptr_iter) {
  GraphMatcher matcher;
  GraphReplace replacer(this->ops_kernel_info_store_ptr_);
  vector<GraphMatchResult> matched_graphs;
  Status ret;
  size_t rule_matched_graphs = 0;
  size_t invalid_count = 0;
  std::shared_ptr<FusionCycleDetector> detector = nullptr;
  FE_MAKE_SHARED(detector = std::make_shared<FusionCycleDetector>(), return FAILED);
  do {
    // clear graph map
    matched_graphs.clear();

    // to match subgraph by one Rule
    ret = matcher.Match(**rule_pattern_ptr_iter, graph, matched_graphs);
    if (ret != SUCCESS) {
      REPORT_FE_ERROR(
          "[GraphOpt][RunFusionRule][MtcRplcByRule] GraphFusion, GraphName[%s], Match Graph Failed, rule_name[%s].",
          graph.GetName().c_str(), rule_name.c_str());
      return ret;
    }

    // if not matched any graph, continue
    if (matched_graphs.empty()) {
      break;
    }

    CorrectMatchedResult(rule_name, graph, detector, matched_graphs, invalid_count);

    rule_matched_graphs = matched_graphs.size();
    // to replace matched graphs in computegraph
    ret = replacer.ReplaceGraph(matched_graphs, **rule_pattern_ptr_iter, graph);
    if (ret == GRAPH_REPLACE_CHECKSUPPORTED_FAILED) {
      break;
    }
    if (ret != SUCCESS) {
      REPORT_FE_ERROR(
          "[GraphOpt][RunFusionRule][MtcRplcByRule] GraphFusion, GraphName[%s], Replace Graph Failed, Rule Name:%s.",
          graph.GetName().c_str(), rule_name.c_str());
      return ret;
    }
  } while (false);

  FE_LOGD("GraphFusion: GraphName:%s, Rule Name:%s, MatchedGraphs:%zu, invalid count %zu.",
          graph.GetName().c_str(), rule_name.c_str(),
          rule_matched_graphs, invalid_count);
  return SUCCESS;
}

Status FusionRuleManager::RunGraphFusionRuleByType(ge::ComputeGraph &graph, RuleType rule_type,
                                                   const std::string &rule_name) {
  Status ret;
  if (!init_flag_) {
    REPORT_FE_ERROR("[GraphOpt][RunFusionRule][RunGphRule] FusionRuleManager has not been initialized.");
    return FAILED;
  }

  // get graph rule
  vector<FusionRulePatternPtr> fusion_rules;
  if (GetFusionRulesByRuleType(rule_type, fusion_rules) != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][RunFusionRule][RunGphRule] GraphFusion, GraphName[%s], RuleType[%d], Get Rules Failed.",
                    graph.GetName().c_str(), static_cast<int>(rule_type));
    return FAILED;
  }

  auto rule_pattern_ptr_iter = find_if(fusion_rules.begin(), fusion_rules.end(), FusionRuleFinder(rule_name));
  if (rule_pattern_ptr_iter == fusion_rules.end()) {
    REPORT_FE_ERROR("[GraphOpt][RunFusionRule][RunGphRule] Could not find rule:%s.", rule_name.c_str());
    return FAILED;
  }

  NodeMapInfoPtr node_map_info;
  if (GraphNodeMapUtil::CreatAndSetOpTypeMap(node_map_info, graph) != SUCCESS) {
    return FAILED;
  }

  ret = MatchAndReplaceByRules(graph, rule_name, rule_pattern_ptr_iter);
  if (ret != SUCCESS) {
    return ret;
  }

  return SUCCESS;
}
}  // namespace fe

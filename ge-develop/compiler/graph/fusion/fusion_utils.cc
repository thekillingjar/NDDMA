/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/op_type_utils.h"
#include "register/graph_optimizer/fusion_common/graph_pass_util.h"
#include "graph/ge_context.h"
#include "common/util.h"
#include "common/types.h"
#include "nlohmann/json.hpp"
#include "fusion_utils.h"

namespace ge {
namespace fusion {
namespace {
constexpr char_t const *kGraphFusionKey = "GraphFusion";
constexpr char_t const *kSwitchKey = "Switch";
constexpr char_t const *kSwitchOn = "on";
constexpr char_t const *kSwitchOff = "off";

std::string SwitchToString(const std::map<string, bool> &pass_name_2_switches) {
  std::stringstream ss;
  for (const auto &pass_name_2_switch : pass_name_2_switches) {
    ss << "[" << pass_name_2_switch.first << ",";
    std::string switch_str = pass_name_2_switch.second ? kSwitchOn : kSwitchOff;
    ss << switch_str << "]";
  }
  return ss.str();
}

Status ReadJsonFile(const std::string &file_real_path, nlohmann::json &json_obj) {
  std::ifstream file_stream(file_real_path);
  GE_MAKE_GUARD(file_guard, [&file_stream]() { file_stream.close(); });

  try {
    GE_WARN_ASSERT(file_stream.is_open(), "Failed to open json file[%s], File is already opened",
                   file_real_path.c_str());
    file_stream >> json_obj;
  } catch (const std::exception &e) {
    GELOGW("Failed to read json file[%s], err msg: %s", file_real_path.c_str(), e.what());
    return FAILED;
  }
  // check whether write file error.
  GE_CHK_BOOL_RET_STATUS(file_stream.good(), FAILED, "Failed to read json file[%s], error msg = %s",
                         file_real_path.c_str(), strerror(errno));
  GELOGD("Read json file[%s] success", file_real_path.c_str());
  return SUCCESS;
}

void CollectFusionSwitchMap(const nlohmann::json &fusion_switch_config_json,
                            std::map<std::string, bool> &fusion_switch_map) {
  if (fusion_switch_config_json == nullptr ||
      fusion_switch_config_json.find(kSwitchKey) == fusion_switch_config_json.end() ||
      fusion_switch_config_json[kSwitchKey].find(kGraphFusionKey) == fusion_switch_config_json[kSwitchKey].end()) {
    return;
  }
  if (!fusion_switch_config_json[kSwitchKey][kGraphFusionKey].is_object()) {
    GELOGW("The third level of json file should be object.");
    return;
  }
  for (auto &item_json : fusion_switch_config_json[kSwitchKey][kGraphFusionKey].items()) {
    string key_inner = item_json.key();
    string value_inner = item_json.value();
    if (!key_inner.empty()) {
      fusion_switch_map.emplace(key_inner, value_inner == kSwitchOn);
    }
  }
}

Status ParseFusionSwitchFile(const string &fusion_switch_json_file, std::map<string, bool> &pass_name_2_switch) {
  nlohmann::json fusion_config_json;
  GE_WARN_ASSERT(ReadJsonFile(fusion_switch_json_file, fusion_config_json) == SUCCESS,
                 "Failed to read json from file %s.", fusion_switch_json_file.c_str());

  if (fusion_config_json != nullptr && !fusion_config_json.is_object()) {
    GELOGE(GRAPH_FAILED, "[GraphOpt][Init][CheckCfgFileFormat] Top level of fusion config file should be object.");
    return FAILED;
  }
  // no need check json format, it was check in fe init stage
  CollectFusionSwitchMap(fusion_config_json, pass_name_2_switch);
  return SUCCESS;
}

std::vector<NodePtr> ToNodePtrs(const std::vector<GNode> &nodes) {
  std::vector<NodePtr> node_ptrs;
  for (const auto &node : nodes) {
    node_ptrs.emplace_back(NodeAdapter::GNode2Node(node));
  }
  return node_ptrs;
}
}  // namespace
std::unordered_map<ComputeGraphPtr, CycleDetectorSharedPtr> FusionUtils::graph_2_cycle_detectors_;
Status FusionUtils::MarkPassNameOnReplacementNodes(const std::unique_ptr<Graph> &replacement,
                                                   const std::unique_ptr<SubgraphBoundary> &subgraph,
                                                   const std::string &pass_name) {
  const auto replacement_graph = GraphUtilsEx::GetComputeGraph(*replacement);
  InnerSubgraphBoundary inner_boundary;
  std::string boundary_invalid_reason;
  GE_ASSERT_SUCCESS(inner_boundary.Init(*subgraph, boundary_invalid_reason), boundary_invalid_reason.c_str());
  for (const auto &node : replacement_graph->GetAllNodes()) {
    GE_ASSERT_SUCCESS(
        fe::GraphPassUtil::StoreAndUpdataOriginFusionPassName(node->GetOpDesc(), inner_boundary.GetNodes(), pass_name));
  }
  return SUCCESS;
}

std::string FusionUtils::ToString(const std::unique_ptr<Graph> &graph) {
  std::stringstream ss;
  auto compute_grpah = GraphUtilsEx::GetComputeGraph(*graph);
  ss << "[";
  for (const auto &node : compute_grpah->GetDirectNode()) {
    if (OpTypeUtils::IsDataNode(node->GetType()) || OpTypeUtils::IsGraphOutputNode(node->GetType())) {
      continue;
    }
    ss << "{" << node->GetTypePtr() << "}";
  }
  ss << "]";
  return ss.str();
}

std::string FusionUtils::GetFusionSwitchFileFromOption() {
  std::string fusion_switch_file_path;
  ge::graphStatus status = GetContext().GetOption(FUSION_SWITCH_FILE, fusion_switch_file_path);
  if (status != ge::GRAPH_SUCCESS) {
    GELOGD("Cannot get option value [%s].", FUSION_SWITCH_FILE.c_str());
    return "";
  }
  GELOGD("The [%s] in ge context is %s.", FUSION_SWITCH_FILE.c_str(), fusion_switch_file_path.c_str());
  return fusion_switch_file_path;
}

std::map<string, bool> FusionUtils::ParseFusionSwitch() {
  const auto fusion_switch_file_path = GetFusionSwitchFileFromOption();
  if (fusion_switch_file_path.empty()) {
    return {};
  }
  const auto fusion_switch_real_path = RealPath(fusion_switch_file_path.c_str());
  if (fusion_switch_real_path.empty()) {
    GELOGD("Fusion switch config real path of %s is empty", fusion_switch_file_path.c_str());
    return {};
  }
  GELOGD("Fusion switch config real path is %s", fusion_switch_real_path.c_str());
  std::map<string, bool> pass_name_2_switches;
  // 可能存在一种较老的配置格式导致解析失败，因此处不再考虑兼容老的配置格式，因此返回warning
  GE_WARN_ASSERT(ParseFusionSwitchFile(fusion_switch_real_path, pass_name_2_switches) == SUCCESS);
  GELOGD("[FusionSwitch] is %s", SwitchToString(pass_name_2_switches).c_str());
  return pass_name_2_switches;
}

void FusionUtils::ResetCycleDetectors() {
  graph_2_cycle_detectors_.clear();
}

CycleDetectorSharedPtr FusionUtils::GetOrCreateCycleDetector(const ComputeGraphPtr &curr_graph) {
  auto iter = graph_2_cycle_detectors_.find(curr_graph);
  if (iter != graph_2_cycle_detectors_.end()) {
    return iter->second;
  }
  GE_ASSERT_SUCCESS(curr_graph->TopologicalSorting());
  auto detector = GraphUtils::CreateSharedCycleDetector(curr_graph);
  graph_2_cycle_detectors_[curr_graph] = detector;
  return detector;
}

bool FusionUtils::WillCauseCycleIfFuse(const std::unique_ptr<MatchResult> &match_result) {
  const auto matched_nodes = ToNodePtrs(match_result->GetMatchedNodes());
  if (matched_nodes.empty()) {
    return false;
  }

  const auto owner_graph = (*matched_nodes.begin())->GetOwnerComputeGraph();
  GE_ASSERT_NOTNULL(owner_graph);
  auto detector = GetOrCreateCycleDetector(owner_graph);
  GE_ASSERT_NOTNULL(detector);
  if (detector->HasDetectedCycle({matched_nodes})) {
    return true;
  }
  return false;
}

Status FusionUtils::UpdateToCycleDetector(const ComputeGraphPtr &curr_graph,
                                          const std::unique_ptr<MatchResult> &match_result,
                                          const unique_ptr<Graph> &replacement) {
  (void)match_result;
  auto replacement_graph = GraphUtilsEx::GetComputeGraph(*replacement);
  std::vector<NodePtr> replacement_nodes;
  for (const auto &node : replacement_graph->GetDirectNode()) {
    if (OpTypeUtils::IsDataNode(node->GetTypePtr()) || (strcmp(node->GetTypePtr(), NETOUTPUT) == 0)) {
      continue;
    }
    replacement_nodes.emplace_back(node);
  }

  auto detector = GetOrCreateCycleDetector(curr_graph);
  GE_ASSERT_NOTNULL(detector);
  detector->Update(curr_graph, replacement_nodes);
  return SUCCESS;
}

void FusionUtils::RecordFusionStatistic(const uint64_t session_id, const std::string graph_id, const std::string pass_name,
  const int match_times, const int effect_times) {
  fe::FusionStatisticRecorder &fusion_statistic_instance = fe::FusionStatisticRecorder::Instance();
  auto fusion_info = fe::FusionInfo(session_id, graph_id, pass_name, match_times, effect_times);
  fusion_statistic_instance.UpdateGraphFusionMatchTimes(fusion_info);
  fusion_statistic_instance.UpdateGraphFusionEffectTimes(fusion_info);
}

} // namespace fusion
}  // namespace ge
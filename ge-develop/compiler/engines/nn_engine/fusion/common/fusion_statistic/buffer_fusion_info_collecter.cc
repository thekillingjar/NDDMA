/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "buffer_fusion_info_collecter.h"
#include "common/graph/fe_graph_utils.h"
#include "common/fe_inner_attr_define.h"
#include "fusion_statistic_writer.h"
#include "graph/ge_attr_value.h"
#include "graph/utils/graph_utils.h"

namespace fe {
namespace {
void GetSingleOpPassName(const ge::ComputeGraph &graph, map<std::string, int32_t> &pass_match_map,
                         map<std::string, int32_t> &pass_effect_map) {
  vector<string> single_op_pass_names;
  for (const ge::NodePtr &node : graph.GetDirectNode()) {
    if (ge::AttrUtils::GetListStr(node->GetOpDesc(), kSingleOpUbPassNameAttr, single_op_pass_names)) {
      for (const auto &pass_name : single_op_pass_names) {
        pass_match_map[pass_name] += 1;
        pass_effect_map[pass_name] += 1;
      }
    }
  }
}
} // namespace

BufferFusionInfoCollecter::BufferFusionInfoCollecter(){};

BufferFusionInfoCollecter::~BufferFusionInfoCollecter(){};

void BufferFusionInfoCollecter::SetPassName(const ge::ComputeGraph &graph,
                                            std::set<std::string> &pass_name_set) const {
  for (const ge::NodePtr &node : graph.GetDirectNode()) {
    std::string pass_name;
    ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
    if (ge::AttrUtils::GetStr(op_desc_ptr, kPassNameUbAttr, pass_name)) {
      pass_name_set.insert(pass_name);
      continue;
    }
  }
}

Status BufferFusionInfoCollecter::GetPassNameOfScopeId(ge::ComputeGraph &graph,
                                                       PassNameIdMap &pass_name_scope_id_map) const {
  int64_t scope_id = 0;
  for (const ge::NodePtr &node : graph.GetDirectNode()) {
    ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
    std::string pass_name;
    if (ge::AttrUtils::GetInt(op_desc_ptr, SCOPE_ID_ATTR, scope_id) && scope_id >= 0 &&
        ge::AttrUtils::GetStr(op_desc_ptr, kPassNameUbAttr, pass_name)) {
      const auto iter = pass_name_scope_id_map.find(pass_name);
      if (iter == pass_name_scope_id_map.end()) {
        std::set<int64_t> id_list_new;
        id_list_new.clear();
        id_list_new.insert(scope_id);
        pass_name_scope_id_map.insert(PassNameIdPair(pass_name, id_list_new));
      } else {
        iter->second.insert(scope_id);
      }
    }
  }
  FE_LOGD("Pass-label map size:%zu, label type: SCOPE_ID_ATTR", pass_name_scope_id_map.size());
  return SUCCESS;
}

Status BufferFusionInfoCollecter::GetPassNameOfFailedId(ge::ComputeGraph &graph,
                                                        PassNameIdMap &pass_name_fusion_failed_id_map) const {
  int64_t fusion_failed_id = 0;
  for (const ge::NodePtr &node : graph.GetDirectNode()) {
    ge::OpDescPtr op_desc_ptr = node->GetOpDesc();
    std::string pass_name;
    if (ge::AttrUtils::GetInt(op_desc_ptr, FUSION_FAILED_ID_ATTR, fusion_failed_id) && fusion_failed_id >= 0 &&
        ge::AttrUtils::GetStr(op_desc_ptr, kPassNameUbAttr, pass_name)) {
      const auto iter = pass_name_fusion_failed_id_map.find(pass_name);
      if (iter == pass_name_fusion_failed_id_map.end()) {
        std::set<int64_t> id_list_new;
        id_list_new.clear();
        id_list_new.insert(fusion_failed_id);
        pass_name_fusion_failed_id_map.insert(PassNameIdPair(pass_name, id_list_new));
      } else {
        iter->second.insert(fusion_failed_id);
      }
    }
  }
  FE_LOGD("Pass-label map size:%zu, label type: FAILED_ID_ATTR", pass_name_fusion_failed_id_map.size());
  return SUCCESS;
}

Status BufferFusionInfoCollecter::CountBufferFusionTimes(ge::ComputeGraph &graph) const {
  PassNameIdMap pass_name_scope_id_map;
  PassNameIdMap pass_name_failed_id_map;
  std::set<std::string> pass_name_set;
  map<std::string, int32_t> pass_match_map;
  map<std::string, int32_t> pass_effect_map;
  // set pass_name
  SetPassName(graph, pass_name_set);
  // get pass_name - ScopeId Map
  (void)GetPassNameOfScopeId(graph, pass_name_scope_id_map);
  // get pass_name - FailedId Map
  (void)GetPassNameOfFailedId(graph, pass_name_failed_id_map);
  FE_LOGD("Pass name set size=%zu.", pass_name_set.size());
  for (const auto &pass_name : pass_name_set) {
    // fusion success
    if (pass_name_scope_id_map.find(pass_name) != pass_name_scope_id_map.end()) {
      pass_match_map[pass_name] += static_cast<int32_t>(pass_name_scope_id_map[pass_name].size());
      pass_effect_map[pass_name] += static_cast<int32_t>(pass_name_scope_id_map[pass_name].size());
    }
    // fusion failed
    if (pass_name_failed_id_map.find(pass_name) != pass_name_failed_id_map.end()) {
      pass_match_map[pass_name] += pass_name_failed_id_map[pass_name].size();
    }
  }
  GetSingleOpPassName(graph, pass_match_map, pass_effect_map);

  string graph_id_string = "";
  FeGraphUtils::GetGraphIdFromAttr(graph, graph_id_string);
  for (const auto &iter : pass_match_map) {
    FusionInfo fusion_info(graph.GetSessionID(), graph_id_string, iter.first, iter.second, 0);
    FusionStatisticRecorder::Instance().UpdateBufferFusionMatchTimes(fusion_info);
    FE_LOGD("SessionId %lu graph_id %s buffer_fusion_pass[%s]: matched_times=%d.", graph.GetSessionID(),
            graph_id_string.c_str(), iter.first.c_str(), iter.second);
  }
  for (const auto &iter : pass_effect_map) {
    FusionInfo fusion_info(graph.GetSessionID(), graph_id_string, iter.first, 0, iter.second);
    FusionStatisticRecorder::Instance().UpdateBufferFusionEffectTimes(fusion_info);
    FE_LOGD("SessionId %lu graph_id %s buffer_fusion_pass[%s]: effect_times=%d.", graph.GetSessionID(),
            graph_id_string.c_str(), iter.first.c_str(), iter.second);
  }
  string session_graph_id = "";
  if (ge::AttrUtils::GetStr(graph, ge::ATTR_NAME_SESSION_GRAPH_ID, session_graph_id) && !session_graph_id.empty()) {
    FE_LOGD("ub session_graph_id=%s", session_graph_id.c_str());
  }
  return SUCCESS;
}
}

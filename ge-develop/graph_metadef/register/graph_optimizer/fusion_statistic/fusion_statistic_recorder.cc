/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/graph_optimizer/fusion_common/fusion_statistic_recorder.h"
#include <algorithm>
#include "framework/common/debug/ge_log.h"

namespace fe {

FusionStatisticRecorder::FusionStatisticRecorder(){};

FusionStatisticRecorder::~FusionStatisticRecorder(){};

FusionStatisticRecorder &FusionStatisticRecorder::Instance() {
  static FusionStatisticRecorder fusion_statistic_recoder;
  return fusion_statistic_recoder;
}

void FusionStatisticRecorder::UpdateGraphFusionMatchTimes(const FusionInfo &fusion_info) {
  const std::lock_guard<std::recursive_mutex> my_lock(mutex_);
  if (fusion_info.GetMatchTimes() != 0) {
    const std::string session_and_graph_id = std::to_string(fusion_info.GetSessionId()) + "_" + \
                                             fusion_info.GetGraphId();
    graph_fusion_info_map_[session_and_graph_id][fusion_info.GetPassName()].AddMatchTimes(fusion_info.GetMatchTimes());
    GELOGD("session %lu, graph %s, pass %s, match_times value: %d", fusion_info.GetSessionId(),
           fusion_info.GetGraphId().c_str(), fusion_info.GetPassName().c_str(),
           graph_fusion_info_map_[session_and_graph_id][fusion_info.GetPassName()].GetMatchTimes());
  }
}

void FusionStatisticRecorder::UpdateGraphFusionEffectTimes(const FusionInfo &fusion_info) {
  const std::lock_guard<std::recursive_mutex> my_lock(mutex_);
  if (fusion_info.GetEffectTimes() != 0) {
    const std::string session_and_graph_id = std::to_string(fusion_info.GetSessionId()) + "_" + \
                                             fusion_info.GetGraphId();
    graph_fusion_info_map_[session_and_graph_id][fusion_info.GetPassName()].AddEffectTimes(
        fusion_info.GetEffectTimes());
    GELOGD("session %lu, graph %s, pass %s, effect_times value: %d", fusion_info.GetSessionId(),
           fusion_info.GetGraphId().c_str(), fusion_info.GetPassName().c_str(),
           graph_fusion_info_map_[session_and_graph_id][fusion_info.GetPassName()].GetEffectTimes());
  }
}

void FusionStatisticRecorder::UpdateBufferFusionMatchTimes(const FusionInfo &fusion_info) {
  const std::lock_guard<std::recursive_mutex> my_lock(mutex_);
  const std::string session_graph_id = std::to_string(fusion_info.GetSessionId()) + "_" + fusion_info.GetGraphId();
  if (fusion_info.GetMatchTimes() != 0) {
    buffer_fusion_info_map_[session_graph_id][fusion_info.GetPassName()].AddMatchTimes(fusion_info.GetMatchTimes());
  }

  if (fusion_info.GetRepoHitTimes() != 0) {
    buffer_fusion_info_map_[session_graph_id][fusion_info.GetPassName()].SetRepoHitTimes(fusion_info.GetRepoHitTimes());
  }
  GELOGD("Updated match time of pass [%s] for graph [%s] and session [%lu].",
         fusion_info.GetPassName().c_str(), fusion_info.GetGraphId().c_str(), fusion_info.GetSessionId());
  GELOGD("Match times is [%d] and repo match times is [%d].",
         buffer_fusion_info_map_[session_graph_id][fusion_info.GetPassName()].GetMatchTimes(),
         buffer_fusion_info_map_[session_graph_id][fusion_info.GetPassName()].GetRepoHitTimes());
}

void FusionStatisticRecorder::UpdateBufferFusionEffectTimes(const FusionInfo &fusion_info) {
  const std::lock_guard<std::recursive_mutex> my_lock(mutex_);
  if (fusion_info.GetEffectTimes() != 0) {
    const std::string session_and_graph_id = std::to_string(fusion_info.GetSessionId()) + "_" + \
                                             fusion_info.GetGraphId();
    buffer_fusion_info_map_[session_and_graph_id][fusion_info.GetPassName()].AddEffectTimes(
        fusion_info.GetEffectTimes());
    GELOGD("ub session %lu, graph %s, pass %s, effect_times value: %d", fusion_info.GetSessionId(),
           fusion_info.GetGraphId().c_str(), fusion_info.GetPassName().c_str(),
           buffer_fusion_info_map_[session_and_graph_id][fusion_info.GetPassName()].GetEffectTimes());
  }
}

void FusionStatisticRecorder::GetAndClearFusionInfo(const std::string &session_graph_id,
                                                    std::map<std::string, FusionInfo> &graph_fusion_info_map,
                                                    std::map<std::string, FusionInfo> &buffer_fusion_info_map) {
  const std::lock_guard<std::recursive_mutex> my_lock(mutex_);
  GELOGD("start to get graph map size %zu", graph_fusion_info_map_.size());
  GELOGD("start to get ub graph map size: %zu", buffer_fusion_info_map_.size());
  GetFusionInfo(session_graph_id, graph_fusion_info_map, buffer_fusion_info_map);
  ClearFusionInfo(session_graph_id);
}

void FusionStatisticRecorder::GetFusionInfo(const std::string &session_graph_id,
                                            std::map<std::string, FusionInfo> &graph_fusion_info_map,
                                            std::map<std::string, FusionInfo> &buffer_fusion_info_map) {
  if (graph_fusion_info_map_.find(session_graph_id) != graph_fusion_info_map_.end()) {
    graph_fusion_info_map = graph_fusion_info_map_[session_graph_id];
  }
  if (buffer_fusion_info_map_.find(session_graph_id) != buffer_fusion_info_map_.end()) {
    buffer_fusion_info_map = buffer_fusion_info_map_[session_graph_id];
  }
}

void FusionStatisticRecorder::ClearFusionInfo(const std::string& session_graph_id) {
  if (graph_fusion_info_map_.find(session_graph_id) != graph_fusion_info_map_.end()) {
    (void)graph_fusion_info_map_.erase(session_graph_id);
  }
  if (buffer_fusion_info_map_.find(session_graph_id) != buffer_fusion_info_map_.end()) {
    (void)buffer_fusion_info_map_.erase(session_graph_id);
  }
}

void FusionStatisticRecorder::GetAllSessionAndGraphIdList(std::vector<std::string> &session_graph_id_vec) {
  if (!graph_fusion_info_map_.empty()) {
    for (auto iter = graph_fusion_info_map_.cbegin(); iter != graph_fusion_info_map_.cend(); iter++) {
      session_graph_id_vec.push_back(iter->first);
    }
  }
  if (!buffer_fusion_info_map_.empty()) {
    for (auto iter = buffer_fusion_info_map_.cbegin(); iter != buffer_fusion_info_map_.cend(); iter++) {
      if (std::find(session_graph_id_vec.begin(), session_graph_id_vec.end(), iter->first)
              == session_graph_id_vec.end()) {
        session_graph_id_vec.push_back(iter->first);
      }
    }
  }
}

FusionInfo::FusionInfo(const uint64_t session_id, const std::string graph_id, const std::string pass_name,
                       const int32_t match_times, const int32_t effect_times, const int32_t repo_hit_times)
    : session_id_(session_id), graph_id_(graph_id), pass_name_(pass_name),
      match_times_(match_times), effect_times_(effect_times), repo_hit_times_(repo_hit_times) {}

FusionInfo::~FusionInfo() {}

void FusionInfo::AddMatchTimes(const int32_t match_times) {
  if (this->match_times_ > std::numeric_limits<int32_t>::max() - match_times) {
    return;
  }
  this->match_times_ += match_times;
}

void FusionInfo::AddEffectTimes(const int32_t effect_times) {
  if (this->effect_times_ > std::numeric_limits<int32_t>::max() - effect_times) {
    return;
  }
  this->effect_times_ += effect_times;
}

int32_t FusionInfo::GetMatchTimes() const { return match_times_; }

void FusionInfo::SetMatchTimes(const int32_t match_times) { this->match_times_ = match_times; }

int32_t FusionInfo::GetEffectTimes() const { return effect_times_; }

void FusionInfo::SetEffectTimes(const int32_t effect_times) { this->effect_times_ = effect_times; }

int32_t FusionInfo::GetRepoHitTimes() const { return repo_hit_times_; }

void FusionInfo::SetRepoHitTimes(const int32_t repo_hit_times) { this->repo_hit_times_ = repo_hit_times; }

std::string FusionInfo::GetGraphId() const { return graph_id_; }

std::string FusionInfo::GetPassName() const { return pass_name_; }

uint64_t FusionInfo::GetSessionId() const { return session_id_; }
}

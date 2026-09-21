/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_REGISTER_GRAPH_OPTIMIZER_FUSION_STATISTIC_RECORDER_H
#define INC_REGISTER_GRAPH_OPTIMIZER_FUSION_STATISTIC_RECORDER_H

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace fe {

class FusionInfo {
 public:
  explicit FusionInfo(const uint64_t session_id = 0, const std::string graph_id = "", const std::string pass_name = "",
                      const int32_t match_times = 0, const int32_t effect_times = 0, const int32_t repo_hit_times = 0);

  virtual ~FusionInfo();

  void AddMatchTimes(const int32_t match_times);

  void AddEffectTimes(const int32_t effect_times);

  int32_t GetMatchTimes() const;

  void SetMatchTimes(const int32_t match_times);

  int32_t GetEffectTimes() const;

  void SetEffectTimes(const int32_t effect_times);

  int32_t GetRepoHitTimes() const;

  void SetRepoHitTimes(const int32_t repo_hit_times);

  std::string GetGraphId() const;

  std::string GetPassName() const;

  uint64_t GetSessionId() const;

 private:
  uint64_t session_id_;
  std::string graph_id_;
  std::string pass_name_;
  int32_t match_times_;
  int32_t effect_times_;
  int32_t repo_hit_times_;
};

using FusionStatisticMap = std::map<std::string, std::map<std::string, FusionInfo>>;

class FusionStatisticRecorder {
 public:
  FusionStatisticRecorder(const FusionStatisticRecorder &) = delete;

  FusionStatisticRecorder &operator=(const FusionStatisticRecorder &) = delete;

  static FusionStatisticRecorder &Instance();

  void UpdateGraphFusionMatchTimes(const FusionInfo &fusion_info);

  void UpdateGraphFusionEffectTimes(const FusionInfo &fusion_info);

  void UpdateBufferFusionMatchTimes(const FusionInfo &fusion_info);

  void UpdateBufferFusionEffectTimes(const FusionInfo &fusion_info);

  void GetAndClearFusionInfo(const std::string &session_graph_id,
                             std::map<std::string, FusionInfo> &graph_fusion_info_map,
                             std::map<std::string, FusionInfo> &buffer_fusion_info_map);

  void GetFusionInfo(const std::string &session_graph_id, std::map<std::string, FusionInfo> &graph_fusion_info_map,
                     std::map<std::string, FusionInfo> &buffer_fusion_info_map);

  void GetAllSessionAndGraphIdList(std::vector<std::string> &session_graph_id_vec);

 private:
  FusionStatisticRecorder();
  virtual ~FusionStatisticRecorder();
  FusionStatisticMap graph_fusion_info_map_;
  FusionStatisticMap buffer_fusion_info_map_;
  void ClearFusionInfo(const std::string& session_graph_id);

  std::recursive_mutex mutex_;
};
}  // namespace fe

#endif  // INC_REGISTER_GRAPH_OPTIMIZER_FUSION_STATISTIC_RECORDER_H

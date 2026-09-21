/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fusion_statistic_writer.h"
#include "common/configuration.h"
#include "common/aicore_util_constants.h"
#include "mmpa/mmpa_api.h"
#include "graph_metadef/graph/utils/file_utils.h"
#include <fcntl.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include "common/fe_utils.h"

namespace fe {
namespace {
const std::string kFusionResultFilename = "fusion_result.json";
constexpr char const *kSessionAndGraphId = "session_and_graph_id";
constexpr char const *kGraphFusionLogType = "graph_fusion";
constexpr char const *kUbFusionLogType = "ub_fusion";
constexpr char const *kMatchTimesLog = "match_times";
constexpr char const *kRepoHitTimesLog = "repository_hit_times";
constexpr char const *kEffectTimesLog = "effect_times";
constexpr int kJsonDumpIndex = 4;
constexpr int64_t kFileOffset = -2;
constexpr uint32_t kJsonPrefix = 2;
}

FusionStatisticWriter::FusionStatisticWriter(){};

FusionStatisticWriter::~FusionStatisticWriter(){};

FusionStatisticWriter &FusionStatisticWriter::Instance() {
  static FusionStatisticWriter fusion_statistic_writer;
  return fusion_statistic_writer;
}

void FusionStatisticWriter::ClearHistoryFile() {
  GetFusionResultFilePath();
  std::lock_guard<std::mutex> lock_guard(file_mutex_);
  int res = remove(fusion_result_path_.c_str());
  FE_LOGI("The result of removing file[%s] is %d", fusion_result_path_.c_str(), res);
}

void FusionStatisticWriter::WriteAllFusionInfoToJsonFile(const std::string &session_graph_id) {
  FE_TIMECOST_START(FusionStatisticWriter);
  json json_data;
  if (session_graph_id.empty()) {
    SaveAllFusionInfoToJson(json_data);
  } else {
    SaveFusionInfoToJson(session_graph_id, json_data);
  }

  if (json_data.empty()) {
    FE_LOGI("SaveFusionInfoToJson is empty");
    return;
  }

  {
    GetFusionResultFilePath();
    std::lock_guard<std::mutex> lock_guard(file_mutex_);
    std::ofstream file;
    if (RealPath(fusion_result_path_).empty()) {
      file.open(fusion_result_path_);
      FE_CHECK(!file.is_open(), FE_LOGW("Failed to open file[%s], errmsg[%s], errno[%d].",
                                        fusion_result_path_.c_str(), strerror(errno), errno), return);
      file << json_data.dump(kJsonDumpIndex);
    } else {
      file.open(fusion_result_path_, std::ios::in | std::ios::out);
      FE_CHECK(!file.is_open(), FE_LOGW("Failed to open file[%s], errmsg[%s], errno[%d].",
                                       fusion_result_path_.c_str(), strerror(errno), errno), return);

      // remove "\n}" of json "{\n  data1:1\n}"
      // after append data: "{\n  data1:1,\n  data2:2\n}"
      std::ignore = file.seekp(static_cast<std::streamoff>(kFileOffset), std::ios::end);
      if (file.tellp() == -1) {
        FE_LOGW("Failed to seek point of file[%s], errmsg[%s]", fusion_result_path_.c_str(), strerror(errno));
        file.close();
        return;
      }
      file << ",\n" << json_data.dump(kJsonDumpIndex).substr(kJsonPrefix);
    }

    json_data.clear();
    file.close();
  }
  FE_LOGI("Write fusion info to file[%s], session[%s]", fusion_result_path_.c_str(), session_graph_id.c_str());
  FE_TIMECOST_END(FusionStatisticWriter, "FusionStatisticWriter.WriteAllFusionInfoToJsonFile");
}

void FusionStatisticWriter::GetFusionResultFilePath() {
  if (!fusion_result_path_.empty()) {
    return;
  }
  const std::string &ascend_work_path = Configuration::Instance(AI_CORE_NAME).GetAscendWorkPath();
  if (ascend_work_path.empty()) {
    fusion_result_path_ = kFusionResultFilename;
    return;
  }

  std::ostringstream oss;
  oss << ascend_work_path << "/" << FE_MODULE_NAME << "/" << mmGetPid();
  oss.flush();
  std::string file_path = oss.str();
  if (ge::CreateDirectory(file_path) != 0) {
    FE_LOGW("Create directory %s for fusion_result.json failed, use current directory to store this file", 
      file_path.c_str());
    fusion_result_path_ = kFusionResultFilename;
  }
  oss << "/" << kFusionResultFilename;
  oss.flush();
  fusion_result_path_ = oss.str();
}

void FusionStatisticWriter::SaveAllFusionInfoToJson(json &json_data) const {
  std::vector<std::string> session_graph_id_vec;
  FusionStatisticRecorder::Instance().GetAllSessionAndGraphIdList(session_graph_id_vec);
  if (session_graph_id_vec.empty()) {
    FE_LOGD("The statistic of graph fusion and buffer fusion is empty.");
    return;
  }
  for (std::string &session_graph_id : session_graph_id_vec) {
    SaveFusionInfoToJson(session_graph_id, json_data);
  }
}

void FusionStatisticWriter::SaveFusionTimesToJson(const std::map<std::string, FusionInfo> &fusion_info_map,
                                                  const string &graph_type, json &json_data) const {
  for (auto &graph_iter : fusion_info_map) {
    json js_object_pass;
    FusionInfo fusion_info = graph_iter.second;
    js_object_pass[kMatchTimesLog] = std::to_string(fusion_info.GetMatchTimes());
    if (graph_type == kUbFusionLogType) {
      js_object_pass[kRepoHitTimesLog] = std::to_string(fusion_info.GetRepoHitTimes());
    }
    js_object_pass[kEffectTimesLog] = std::to_string(fusion_info.GetEffectTimes());
    json_data[graph_type][graph_iter.first] = js_object_pass;
  }
}

void FusionStatisticWriter::SaveFusionInfoToJson(const std::string &session_and_graph_id, json &json_data) const {
  std::map<std::string, FusionInfo> graph_fusion_info_map;
  std::map<std::string, FusionInfo> buffer_fusion_info_map;
  FusionStatisticRecorder::Instance().GetAndClearFusionInfo(session_and_graph_id,
                                                    graph_fusion_info_map, buffer_fusion_info_map);
  if (!graph_fusion_info_map.empty() || !buffer_fusion_info_map.empty()) {
    json cur_fusion_info_json_data;
    SaveFusionTimesToJson(graph_fusion_info_map, kGraphFusionLogType, cur_fusion_info_json_data);
    SaveFusionTimesToJson(buffer_fusion_info_map, kUbFusionLogType, cur_fusion_info_json_data);
    std::ostringstream oss;
    oss << kSessionAndGraphId << "_" << session_and_graph_id;
    json_data[oss.str()] = cur_fusion_info_json_data;
    FE_LOGD("Session_and_graph_id[%s]: Finish saving fusion result to json.", session_and_graph_id.c_str());
  }
}
}

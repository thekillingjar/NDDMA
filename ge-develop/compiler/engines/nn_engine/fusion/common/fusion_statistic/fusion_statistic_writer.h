/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_FUSION_COMMON_FUSION_STATISTIC_FUSION_STATISTIC_WRITER_H_
#define FUSION_ENGINE_FUSION_COMMON_FUSION_STATISTIC_FUSION_STATISTIC_WRITER_H_

#include <fstream>
#include <nlohmann/json.hpp>
#include "common/aicore_util_attr_define.h"
#include "common/fe_inner_error_codes.h"
#include "common/fe_log.h"
#include "common/graph_comm.h"
#include "common/util/json_util.h"
#include "register/graph_optimizer/fusion_common/fusion_statistic_recorder.h"

namespace fe {
using json = nlohmann::json;
class FusionStatisticWriter {
 public:
  static FusionStatisticWriter &Instance();

  FusionStatisticWriter(const FusionStatisticWriter &) = delete;

  FusionStatisticWriter &operator=(const FusionStatisticWriter &) = delete;

  void ClearHistoryFile();

  void WriteAllFusionInfoToJsonFile(const std::string &session_graph_id = "");

 private:
  explicit FusionStatisticWriter();

  virtual ~FusionStatisticWriter();

  void GetFusionResultFilePath();

  void SaveAllFusionInfoToJson(json &json_data) const;

  void SaveFusionInfoToJson(const std::string &session_and_graph_id, json &json_data) const;

  void SaveFusionTimesToJson(const std::map<std::string, FusionInfo> &fusion_info_map,
                             const string &graph_type, json &json_data) const;

  std::mutex file_mutex_;

  std::string fusion_result_path_ = "";
};
}

#endif  // FUSION_ENGINE_FUSION_COMMON_FUSION_STATISTIC_FUSION_STATISTIC_WRITER_H_

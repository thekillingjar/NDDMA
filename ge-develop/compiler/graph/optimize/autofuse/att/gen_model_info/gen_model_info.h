/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GEN_MODEL_INFO_H_
#define GEN_MODEL_INFO_H_

#include <string>
#include <vector>
#include <map>
#include <stdint.h>

#include "base/model_info.h"
#include "parser/tuning_space.h"
#include "autofuse_config/auto_fuse_config.h"

namespace att {
ge::Status GenerateModelInfo(const ge::AscGraph &graph, ModelInfo &model_info, TuningSpacePtr &tuning_space,
                         const uint32_t tiling_case_id = 0U);
ge::Status GenerateModelInfo(const std::vector<ge::AscGraph> &graph_list, std::vector<ModelInfo> &model_info_list);
ge::Status GenerateModelInfo(const std::vector<ge::AscGraph> &graph_list, std::vector<ModelInfo> &model_info_list,
                             const std::map<std::string, std::string> &options,
                             bool enable_group_parallel = false);
ge::Status GetModelInfoMap(const ascir::FusedScheduledResult &schedule_results,
                           const std::map<std::string, std::string> &options,
                           std::map<size_t, std::map<size_t, ParsedScheduleResult>> &out_all_model_infos);
ge::Status GetAllSubImplGraphs(const ascir::FusedScheduledResult &schedule_results,
                               std::vector<std::vector<std::vector<std::vector<ge::AscGraph>>>> &all_graphs,
                               std::map<std::string, std::string> &all_graph_score_funcs);
ge::Status MakeJson(std::vector<ModelInfo> &model_info_list, std::string &json_info);
} // namespace att

#endif // GEN_MODEL_INFO_H_

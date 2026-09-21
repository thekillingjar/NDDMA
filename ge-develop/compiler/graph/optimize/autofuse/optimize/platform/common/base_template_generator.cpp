/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "base_template_generator.h"

namespace optimize {

std::vector<autoschedule::AutoScheduleOutput> BaseTemplateGenerator::GetBasedCasesByGenMode(
    const GenerationMode mode, const std::vector<autoschedule::AutoScheduleOutput> &tiling_cases,
    const std::vector<autoschedule::AutoScheduleOutput> &generated_cases) {
  if (mode == GenerationMode::kBaseCase) {
    return tiling_cases;
  }
  if (mode == GenerationMode::kAppendCase) {
    std::vector<autoschedule::AutoScheduleOutput> merged;
    merged.reserve(tiling_cases.size() + generated_cases.size());  // 预分配内存，避免多次扩容
    merged.insert(merged.end(), tiling_cases.begin(), tiling_cases.end());
    merged.insert(merged.end(), generated_cases.begin(), generated_cases.end());
    return merged;
  }
  GELOGW("Unknown generation mode: %u", mode);
  return {};
}

ge::Status BaseTemplateGenerator::Generate(BaseTemplate &strategy, const ge::AscGraph &origin_graph,
                                           const std::vector<autoschedule::AutoScheduleOutput> &based_cases,
                                           std::vector<autoschedule::AutoScheduleOutput> &generated_cases,
                                           std::unordered_set<std::string> &drop_case_names) {
  for (const auto &based_case : based_cases) {
    autoschedule::AutoScheduleOutput generated_output(strategy.GenName(based_case.scheduled_graph.GetName()).c_str());
    GE_ASSERT_TRUE(generated_output.scheduled_graph.CopyFrom(based_case.scheduled_graph));
    generated_output.var_relations_ = based_case.var_relations_;

    if (strategy.Generate(origin_graph, based_case.scheduled_graph, generated_output.scheduled_graph) != ge::SUCCESS) {
      GELOGD("Generate template failed, %s.", generated_output.scheduled_graph.GetName().c_str());
      continue;
    }

    // 生成打分函数
    const auto score_func = strategy.GetScoreFunc(origin_graph, generated_output.scheduled_graph);
    if (!score_func.empty()) {
      generated_output.score_func = score_func;
    }

    GELOGD("Generate template success, %s.", generated_output.scheduled_graph.GetName().c_str());
    generated_cases.push_back(generated_output);

    if (strategy.NeedDropBasedCase(origin_graph, based_case.scheduled_graph, generated_output.scheduled_graph)) {
      GELOGD("New template is better than original general template, drop it, %s.",
             based_case.scheduled_graph.GetName().c_str());
      drop_case_names.emplace(based_case.scheduled_graph.GetName());
    }
  }
  return ge::SUCCESS;
}

ge::Status BaseTemplateGenerator::GenerateTemplates(const ge::AscGraph &origin_graph,
                                                    std::vector<autoschedule::AutoScheduleOutput> &tiling_cases) {
  if (strategies_.empty()) {
    GELOGD("Not found template strategies.");
    return ge::SUCCESS;
  }
  std::vector<autoschedule::AutoScheduleOutput> generated_cases;
  std::unordered_set<std::string> drop_case_names;

  for (const auto &strategy : strategies_) {
    GE_CHECK_NOTNULL(strategy);
    const auto &based_cases = GetBasedCasesByGenMode(strategy->GetGenerationMode(), tiling_cases, generated_cases);
    GE_ASSERT_SUCCESS(Generate(*strategy, origin_graph, based_cases, generated_cases, drop_case_names));
  }

  if (generated_cases.empty() && drop_case_names.empty()) {
    return ge::SUCCESS;
  }

  // 拼接生成的新模板
  tiling_cases.insert(tiling_cases.end(), generated_cases.begin(), generated_cases.end());

  // 把需要删除的模板删掉
  if (!drop_case_names.empty()) {
    std::vector<autoschedule::AutoScheduleOutput> reserved_cases;
    reserved_cases.reserve(tiling_cases.size());
    for (const auto &c : tiling_cases) {
      if (drop_case_names.count(c.scheduled_graph.GetName()) == 0UL) {
        reserved_cases.push_back(c);
      }
    }
    tiling_cases.swap(reserved_cases);
  }
  return ge::SUCCESS;
}

}  // namespace optimize
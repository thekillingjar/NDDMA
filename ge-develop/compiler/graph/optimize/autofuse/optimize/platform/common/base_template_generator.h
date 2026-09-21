/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTIMIZE_PLATFORM_COMMON_BASE_TEMPLATE_GENERATOR_H
#define OPTIMIZE_PLATFORM_COMMON_BASE_TEMPLATE_GENERATOR_H

#include "ascendc_ir.h"
#include "base_template.h"
#include "optimize/autoschedule/autoschedule_defs.h"

namespace optimize {
class BaseTemplateGenerator {
 public:
  virtual ~BaseTemplateGenerator() = default;
  explicit BaseTemplateGenerator() = default;

  virtual ge::Status Generate(
      BaseTemplate &strategy, const ge::AscGraph &origin_graph,
      const std::vector<autoschedule::AutoScheduleOutput> &based_cases,  // 原模板，基于此生成新的模板
      std::vector<autoschedule::AutoScheduleOutput> &generated_cases,    // 生成后的模板
      std::unordered_set<std::string> &drop_case_names);                 // 需要删除的模板

  virtual ge::Status GenerateTemplates(const ge::AscGraph &origin_graph,
                                       std::vector<autoschedule::AutoScheduleOutput> &tiling_cases);

  virtual std::vector<autoschedule::AutoScheduleOutput> GetBasedCasesByGenMode(
      const GenerationMode mode, const std::vector<autoschedule::AutoScheduleOutput> &tiling_cases,
      const std::vector<autoschedule::AutoScheduleOutput> &generated_cases);

  BaseTemplateGenerator(const BaseTemplateGenerator &) = delete;
  BaseTemplateGenerator &operator=(const BaseTemplateGenerator &) = delete;
  BaseTemplateGenerator(BaseTemplateGenerator &&) = delete;
  BaseTemplateGenerator &operator=(BaseTemplateGenerator &&) = delete;

 protected:
  std::vector<std::unique_ptr<BaseTemplate>> strategies_;
};
}  // namespace optimize

#endif  // OPTIMIZE_PLATFORM_COMMON_BASE_TEMPLATE_GENERATOR_H

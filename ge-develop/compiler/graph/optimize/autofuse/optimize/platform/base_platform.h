/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPTIMIZE_PLATFORM_BASE_PLATFORM_H
#define OPTIMIZE_PLATFORM_BASE_PLATFORM_H

#include <set>
#include "ascendc_ir.h"
#include "optimize/optimize.h"
#include "common/base_alignment_strategy.h"
#include "common/pass_runner.h"
#include "common/base_template_generator.h"
#include "backend/backend_spec.h"

namespace optimize {
struct PlatformConfig {
  size_t max_que_num = 4U;
  bool is_support_compat_mode = false;
};

class BasePlatform {
 public:
  virtual ~BasePlatform() = default;

  // 融合子图划分
  virtual ge::Status PartitionSubFunctions(ge::AscGraph &impl_graph) = 0;
  // 向量化轴对齐策略选择
  virtual std::unique_ptr<BaseAlignmentStrategy> GetAlignmentStrategy() = 0;

  virtual std::unique_ptr<BasePassRunner> GetPassRunner() = 0;
  // 根据通用模板生成其他模板
  virtual std::unique_ptr<BaseTemplateGenerator> GetTemplateGenerator() = 0;
  // 获取平台相关规格
  virtual std::unique_ptr<BackendSpec> GetBackendSpec() const = 0;

  virtual const PlatformConfig& GetPlatformConfig() const = 0;

  virtual Status GenerateTasks(::ascir::ImplGraph &optimize_graph, const OptimizerOptions &options,
                               std::vector<ScheduleTask> &tasks) const = 0;

  virtual std::set<std::string> BroadcastTypes() const = 0;
};
}  // namespace optimize
#endif  // OPTIMIZE_PLATFORM_BASE_PLATFORM_H

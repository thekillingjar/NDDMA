/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPTIMIZE_PLATFORM_V1_PLATFORM_V2_H
#define OPTIMIZE_PLATFORM_V1_PLATFORM_V2_H

#include "platform/base_platform.h"
#include "platform/platform_factory.h"
namespace optimize {
class PlatformV2 : public BasePlatform {
 public:
  PlatformV2();
  ~PlatformV2() override = default;
  ge::Status PartitionSubFunctions(ge::AscGraph &impl_graph) override;
  std::unique_ptr<BaseAlignmentStrategy> GetAlignmentStrategy() override;
  unique_ptr<BasePassRunner> GetPassRunner() override;
  std::unique_ptr<BaseTemplateGenerator> GetTemplateGenerator() override;
  std::unique_ptr<BackendSpec> GetBackendSpec() const override;
  Status GenerateTasks(ascir::ImplGraph &optimize_graph, const OptimizerOptions &options,
                       std::vector<ScheduleTask> &tasks) const override;
  const PlatformConfig &GetPlatformConfig() const override;
  std::set<std::string> BroadcastTypes() const override;

 private:
  PlatformConfig config_;
};
}  // namespace optimize
#endif  // OPTIMIZE_PLATFORM_V1_PLATFORM_V2_H

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "platformv1.h"
#include "alignment_strategy.h"
#include "pass_runner_v1.h"
#include "template_generator_v1.h"
#include "task_generator/schedule_task_generator.h"
#include "task_generator/concat_schedule_case_generator.h"
#include "task_generator/transpose_schedule_case_generator.h"
#include "task_generator/reduce_schedule_case_generator.h"
#include "task_generator/recompute_case_generator.h"
#include "task_generator/split_schedule_case_generator.h"

namespace optimize {
constexpr size_t kMaxVecQueNum = 4UL;

PlatformV1::PlatformV1() {
  config_.max_que_num = kMaxVecQueNum;
}

ge::Status PlatformV1::PartitionSubFunctions([[maybe_unused]] ge::AscGraph &impl_graph) {
  return ge::SUCCESS;
}

std::unique_ptr<BaseAlignmentStrategy> PlatformV1::GetAlignmentStrategy() {
  return ge::ComGraphMakeUnique<AlignmentStrategy>();
}
unique_ptr<BasePassRunner> PlatformV1::GetPassRunner() {
  return std::make_unique<PassRunnerV1>();
}

std::unique_ptr<BaseTemplateGenerator> PlatformV1::GetTemplateGenerator() {
  return ge::ComGraphMakeUnique<TemplateGeneratorV1>();
}

std::unique_ptr<BackendSpec> PlatformV1::GetBackendSpec() const {
  constexpr uint32_t kConcatMaxInputNum = 63;
  constexpr int32_t kConcatAlgTranspose = 0;
  constexpr uint32_t kMaxLoadNum = 4;
  constexpr uint32_t kMaxInputNum = 8U;
  auto ret = ge::ComGraphMakeUnique<BackendSpec>();
  ret->concat_max_input_num = kConcatMaxInputNum;
  ret->concat_alg = kConcatAlgTranspose;
  ret->gather_spec = {false, false, false, false, false};
  ret->slice_split_spec.split_lowered_to_split = false;
  ret->slice_split_spec.slice_fuse_with_end_dim_1 = false;
  ret->slice_split_spec.enable_split_flatten = false;
  ret->max_load_num = kMaxLoadNum;
  ret->max_input_nums_after_fuse = kMaxInputNum;
  ret->enable_matmul_lowering_to_matmul = false;
  GELOGD(
      "platform_v1, enable_non_tail_gather = %d, enable_reduce_gather_fusion = %d, "
      "enable_gather_concat_fusion = %d, enable_gather_broadcast_fusion = %d, "
      "enable_gather_elementwise_forward_fusion = %d, max load_num = %u, max input num = %u",
      ret->gather_spec.enable_non_tail_gather, ret->gather_spec.enable_reduce_gather_fusion,
      ret->gather_spec.enable_gather_concat_fusion, ret->gather_spec.enable_gather_broadcast_fusion,
      ret->gather_spec.enable_gather_elementwise_forward_fusion, ret->max_load_num, ret->max_input_nums_after_fuse);
  ret->transpose_mode = static_cast<uint32_t>(TransposeMode::TRANSPOSE_MODE_NORMAL);
  ret->set_local_memory_size = 0;
  ret->pgo_spec = {true};
  return ret;
}

const PlatformConfig &PlatformV1::GetPlatformConfig() const {
  return config_;
}

Status PlatformV1::GenerateTasks(ascir::ImplGraph &optimize_graph, const OptimizerOptions &options,
                                 std::vector<ScheduleTask> &tasks) const {
  GE_CHK_STATUS_RET(SplitFusionCaseGenerator().GeneratorTask(optimize_graph, tasks, options),
                    "Failed to generate tasks for split");
  GE_CHK_STATUS_RET(ConcatFusionCaseGenerator().GeneratorTask(optimize_graph, tasks, options),
                    "Failed to generate tasks for concat");
  GE_CHK_STATUS_RET(TransposeFusionCaseGenerator().GeneratorTask(optimize_graph, tasks, options),
                    "Failed to generate tasks for Transpose");
  GE_CHK_STATUS_RET(ReducePartitionCaseGenerator().GeneratorTask(optimize_graph, tasks, options),
                    "Failed to generate tasks for Reduce");
  if (tasks.empty()) {
    GE_CHK_STATUS_RET(RecomputeCaseGenerator().GeneratorTask(optimize_graph, tasks, options),
                      "Failed to generate recomputation tasks for graph[%s].", optimize_graph.GetName().c_str());
  }
  return ge::SUCCESS;
}

std::set<std::string> PlatformV1::BroadcastTypes() const {
  return {ge::ascir_op::Broadcast::Type};
}

#define REGISTER_PLATFORM_V1(platform_name, suffix) \
  static PlatformRegistrar<PlatformV1> registrar_##suffix(platform_name)

REGISTER_PLATFORM_V1("2201", v1);
}  // namespace optimize
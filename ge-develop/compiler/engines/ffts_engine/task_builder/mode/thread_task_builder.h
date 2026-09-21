/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FFTS_ENGINE_TASK_BUILDER_MODE_THREAD_TASK_BUILDER_H_
#define FFTS_ENGINE_TASK_BUILDER_MODE_THREAD_TASK_BUILDER_H_

#include <runtime/rt.h>
#include "common/opskernel/ops_kernel_builder.h"
#include "graph/compute_graph.h"
#include "inc/ffts_error_codes.h"
#include "inc/ffts_type.h"
#include "task_builder/fftsplus_task_builder.h"
#include "task_builder/thread_ctx/aic_aiv_auto_task_builder.h"
#include "task_builder/thread_ctx/mix_aic_aiv_auto_task_builder.h"
#include "task_builder/thread_ctx/mix_aic_aiv_dynamic_task_builder.h"
#include "task_builder/data_ctx/prefetch_auto_task_builder.h"
#include "task_builder/data_ctx/out_auto_task_builder.h"
#include "task_builder/data_ctx/prefetch_dynamic_task_builder.h"
#include "task_builder/data_ctx/out_dynamic_task_builder.h"
#include "task_builder/data_ctx/prefetch_manual_task_builder.h"
#include "task_builder/data_ctx/out_manual_task_builder.h"
#include "task_builder/thread_ctx/aic_aiv_manual_task_builder.h"
#include "task_builder/thread_ctx/aic_aiv_dynamic_task_builder.h"
#include "task_builder/thread_ctx/mix_aic_aiv_manual_task_builder.h"
#include "task_builder/thread_ctx/collection_ops_manual_task_builder.h"
#include "task_builder/thread_ctx/runtime_ops_manual_task_builder.h"
#include "task_builder/thread_ctx/runtime_ops_auto_task_builder.h"
#include "task_builder/thread_ctx/dsa_manual_task_builder.h"
#include "task_builder/thread_ctx/aicpu_manual_task_builder.h"
#include "task_builder/thread_ctx/aicpu_auto_task_builder.h"
#include "task_builder/mixl2_ctx/mix_l2_task_builder.h"
#include "task_builder/mode/data_task_builder.h"

namespace ffts {
using FFTSPlusTaskBuilderPtr = std::shared_ptr<FFTSPlusTaskBuilder>;
using AICAIVTaskBuilderPtr = std::shared_ptr<AICAIVTaskBuilder>;
using AICAIVAutoTaskBuilderPtr = std::shared_ptr<AICAIVAutoTaskBuilder>;
using MixAICAIVTaskBuilderPtr = std::shared_ptr<MixAICAIVTaskBuilder>;
using MixAICAIVDynamicTaskBuilderPtr = std::shared_ptr<MixAICAIVDynamicTaskBuilder>;
using AICAIVDynamicTaskBuilderPtr = std::shared_ptr<AICAIVDynamicTaskBuilder>;
using MixL2TaskBuilderPtr = std::shared_ptr<MixL2TaskBuilder>;
using MixAICAIVAutoTaskBuilderPtr = std::shared_ptr<MixAICAIVAutoTaskBuilder>;
using CollectionOpsTaskBuilderPtr = std::shared_ptr<CollectionOpsTaskBuilder>;
using AicpuTaskBuilderPtr = std::shared_ptr<AicpuTaskBuilder>;
using AicpuAutoTaskBuilderPtr = std::shared_ptr<AicpuAutoTaskBuilder>;
using RuntimeOpsTaskBuilderPtr = std::shared_ptr<RuntimeOpsTaskBuilder>;
using RuntimeOpsAutoTaskBuilderPtr = std::shared_ptr<RuntimeOpsAutoTaskBuilder>;
using DSAManualTaskBuilderPtr = std::shared_ptr<DSAManualTaskBuilder>;
class TheadTaskBuilder {
 public:
  TheadTaskBuilder();
  virtual ~TheadTaskBuilder();

  virtual Status Initialize() = 0;

  virtual Status GenFftsPlusContextId(ge::ComputeGraph &sgt_graph, std::vector<ge::NodePtr> &sub_graph_nodes,
                                      uint64_t &ready_context_num, uint64_t &total_context_number,
                                      std::vector<ge::NodePtr> &memset_nodes) = 0;

  virtual Status GenSubGraphTaskDef(std::vector<ge::NodePtr> &memset_nodes, std::vector<ge::NodePtr> &sub_graph_nodes,
                                    domi::TaskDef &task_def) = 0;
  void SetModeType(const ModeType &type);
  Status GenFftsPlusContextIdWithMemSet(std::vector<ge::NodePtr> &pre_sub_graph_nodes,
                                        std::vector<ge::NodePtr> &memset_nodes, ge::ComputeGraph &sgt_graph) const;
 protected:
  Status GetNodeContextTypeByNode(const ge::NodePtr &node, TaskBuilderType &task_builder_type) const;

  FFTSPlusTaskBuilderPtr GetTaskBuilder(TaskBuilderType task_builder_type);

  Status GenerateDataTaskDef(const ge::NodePtr &node, domi::FftsPlusTaskDef *ffts_plus_task_def,
                             const ModeType &mode_type) const;

  bool IsNoCtx(const ge::NodePtr &node) const;

  const std::unordered_set<std::string> NO_NEED_GEN_TASK_OP_TYPE = {"Data", "RefData", "NetOutput", "Variable", "Const",
                                                                    "Constant", "PhonyConcat"};

  ModeType mode_type_{ModeType::MANUAL_MODE_TYPE};
  AICAIVTaskBuilderPtr aic_aiv_task_builder_ptr_;
  AICAIVAutoTaskBuilderPtr aic_aiv_auto_task_builder_ptr_;
  MixAICAIVTaskBuilderPtr mix_aic_aiv_task_builder_ptr_;
  MixAICAIVDynamicTaskBuilderPtr mix_aic_aiv_dynamic_task_builder_ptr_;
  MixAICAIVAutoTaskBuilderPtr mix_aic_aiv_auto_task_builder_ptr_;
  AICAIVDynamicTaskBuilderPtr aic_aiv_dynamic_task_builder_ptr_;
  MixL2TaskBuilderPtr mix_L2_task_builder_ptr_;
  CollectionOpsTaskBuilderPtr collection_ops_task_builder_ptr_;
  AicpuTaskBuilderPtr aicpu_task_builder_ptr_;
  AicpuAutoTaskBuilderPtr aicpu_auto_task_builder_ptr_;
  RuntimeOpsTaskBuilderPtr runtime_ops_task_builder_ptr_;
  RuntimeOpsAutoTaskBuilderPtr runtime_ops_auto_task_builder_ptr_;
  DSAManualTaskBuilderPtr dsa_ops_task_builder_ptr_;

};
}  // namespace ffts
#endif  // FFTS_ENGINE_TASK_BUILDER_MODE_THREAD_TASK_BUILDER_H_

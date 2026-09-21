/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_OPS_KERNEL_BUILDER_TASK_BUILDER_CMO_TASK_BUILDER_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_OPS_KERNEL_BUILDER_TASK_BUILDER_CMO_TASK_BUILDER_H_

#include <vector>
#include "graph_optimizer/graph_optimize_register_error_codes.h"
#include "common/fe_log.h"
#include "ops_kernel_builder/task_builder/cmo_task/generate_cmo_task_base.h"
#include "ops_kernel_builder/task_builder/cmo_task/generate_cmo_barrier_task.h"
#include "ops_kernel_builder/task_builder/cmo_task/generate_cmo_invalid_task.h"
#include "ops_kernel_builder/task_builder/cmo_task/generate_cmo_prefetch_task.h"
#include "ops_kernel_builder/task_builder/cmo_task/generate_cmo_writeback_task.h"
#include "graph/node.h"
#include "proto/task.pb.h"

namespace fe {
using GenerateCMOTaskBasePtr = std::shared_ptr<GenerateCMOTaskBase>;
using GenerateCMOBarrierTaskPtr = std::shared_ptr<GenerateCMOBarrierTask>;
using GenerateCMOInvalidTaskPtr = std::shared_ptr<GenerateCMOInvalidTask>;
using GenerateCMOPrefetchTaskPtr = std::shared_ptr<GenerateCMOPrefetchTask>;
using GenerateCMOWritebackTaskPtr = std::shared_ptr<GenerateCMOWritebackTask>;

class CMOTaskBuilder {
 public:
  CMOTaskBuilder();
  virtual ~CMOTaskBuilder();

  Status GenerateCMOTask(const ge::Node &node, std::vector<domi::TaskDef> &task_defs,
                         TaskBuilderContext &context, const bool &pre_cmo_task);
 private:
  GenerateCMOTaskBasePtr GetCMOTaskType(const ge::Node &node, TaskBuilderContext &context,
                                        const std::string &task_type, const bool &pre_cmo_task);
  std::string GetPosition(const bool &pre_task) const;
  GenerateCMOBarrierTaskPtr generate_cmo_barrier_task_ptr_;
  GenerateCMOInvalidTaskPtr generate_cmo_invalid_task_ptr_;
  GenerateCMOPrefetchTaskPtr generate_cmo_prefetch_task_ptr_;
  GenerateCMOWritebackTaskPtr generate_cmo_writeback_task_ptr_;
};
using CMOTaskBuilderPtr = std::shared_ptr<CMOTaskBuilder>;
}  // namespace fe
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_OPS_KERNEL_BUILDER_TASK_BUILDER_CMO_TASK_BUILDER_H_

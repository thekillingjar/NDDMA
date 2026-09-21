/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_OPS_KERNEL_BUILDER_TASK_BUILDER_SUPERKERNEL_TASK_BUILDER_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_OPS_KERNEL_BUILDER_TASK_BUILDER_SUPERKERNEL_TASK_BUILDER_H_

#include "ops_kernel_builder/task_builder/task_builder.h"

namespace fe {
using ScopeNodeIdMap = std::map<int64_t, std::vector<ge::Node *>>;
class SuperkernelTaskBuilder {
 public:
  static Status GenerateKernelTask(const ge::Node &node, const ge::RunContext &context,
                                   std::vector<domi::TaskDef> &task_defs);
  static Status GenerateSuperKernelTask(const ge::Node &node,
                                        ge::RunContext &context, std::vector<domi::TaskDef> &tasks);
  static Status GenerateSubKernelTask(const ge::ComputeGraphPtr &sub_graph, ge::RunContext &context,
      std::vector<ge::Node *> &sub_nodes, std::vector<std::vector<domi::TaskDef>> &sub_tasks);
 private:
  static Status DoSubKernelCompile(ge::ComputeGraphPtr &sub_graph);
  static Status DoSuperKernelCompile(const ge::Node &parent_node, const ScopeNodeIdMap &fusion_nodes_map);
  static Status GetTaskFusionNodes(const ge::Node &node, const int64_t scope_id,
                                   std::vector<std::vector<ge::NodePtr>> &fusion_nodes);
  static Status GenerateTaskForFusionNodes(const std::vector<std::vector<ge::NodePtr>> &fusion_nodes,
                                           TaskBuilderContext &task_context, std::vector<domi::TaskDef> &task_defs);
  static Status SetTaskArgsAttr(const ge::NodePtr &node, TaskBuilderContext &task_context, domi::TaskDef &task_def);
  static Status DoTaskFusion(const ScopeNodeIdMap &fusion_nodes_map,
                             const std::map<int64_t, std::vector<domi::TaskDef>> &scope_task_defs_map,
                             std::vector<domi::TaskDef> &task_defs);
  static Status FillupFusionTask(const ge::OpDescPtr &op_desc, domi::TaskDef &task_def);
};
}  // namespace fe
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_OPS_KERNEL_BUILDER_TASK_BUILDER_SUPERKERNEL_TASK_BUILDER_H_

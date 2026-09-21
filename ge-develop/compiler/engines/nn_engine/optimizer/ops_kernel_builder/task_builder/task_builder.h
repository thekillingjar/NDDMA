/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_OPS_KERNEL_BUILDER_TASK_BUILDER_TASK_BUILDER_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_OPS_KERNEL_BUILDER_TASK_BUILDER_TASK_BUILDER_H_
#include <memory>
#include <vector>
#include "proto/task.pb.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "adapter/adapter_itf/task_builder_adapter.h"

namespace fe {
class TaskBuilder {
 public:
  /*
   * @ingroup fe
   * @brief   Generate tasks
   * @param   [in] node Node of compute graph
   * @param   [in] context Context for generate tasks
   * @param   [out] task_defs Save the generated tasks.
   * @return  SUCCESS or FAILED
   */
  static Status GenerateKernelTask(const ge::Node &node, const ge::RunContext &context,
                                   std::vector<domi::TaskDef> &task_defs);

  static Status GenerateKernelTask(const ge::Node &node, TaskBuilderContext &task_context,
                                   std::vector<domi::TaskDef> &task_defs);

  static Status DoGenerateTask(const ge::Node &node, TaskBuilderContext &task_context, domi::TaskDef &task_def);

  static Status InitTaskContext(const ge::RunContext &context, TaskBuilderContext &task_context);

  static void StartKernelFusion(const ge::OpDescPtr &op_desc_ptr, const int32_t &stream_id,
                                std::vector<domi::TaskDef> &task_defs);
  static void EndKernelFusion(const ge::OpDescPtr &op_desc_ptr, const int32_t &stream_id,
                              std::vector<domi::TaskDef> &task_defs);
  /*
   * @ingroup fe
   * @brief   Run TaskBuilderAdapter
   * @return  SUCCESS or FAILED
   */
  static Status FillTaskDefAfterGenTask(const ge::OpDescPtr &op_desc, domi::TaskDef &task_def,
                                        std::string &args_format);
 private:
  // follow function for traditional kernel task
  static Status GenTaskForOneNode(const ge::Node &node, TaskBuilderContext &task_context,
                                  std::vector<domi::TaskDef> &task_defs, int64_t stream_id);
  /*
   * @ingroup fe
   * @brief   Create TaskBuilderAdapter
   * @param   [in] node Node of compute graph
   * @return  SUCCESS or FAILED
   */
  static TaskBuilderAdapterPtr CreateTaskAdapter(const ge::Node &node, TaskBuilderContext &task_context);
  static void SetSlowContextSwitchAttr(const ge::OpDescPtr &op_desc);
  static void GenerateCMOTask(const ge::Node &node, const bool pre_cmo_task, TaskBuilderContext &task_context,
                              std::vector<domi::TaskDef> &task_defs);
  static Status GenerateOpArgsFormat(const ge::OpDescPtr &op_desc, std::string &args_format_str);
};
}  // namespace fe
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_OPS_KERNEL_BUILDER_TASK_BUILDER_TASK_BUILDER_H_

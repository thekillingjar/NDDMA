/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_OPS_KERNEL_BUILDER_TASK_BUILDER_CMO_TASK_GENERATE_CMO_INVALID_TASK_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_OPS_KERNEL_BUILDER_TASK_BUILDER_CMO_TASK_GENERATE_CMO_INVALID_TASK_H_

#include "ops_kernel_builder/task_builder/cmo_task/generate_cmo_task_base.h"

namespace fe {
class GenerateCMOInvalidTask : public GenerateCMOTaskBase {
 public:
  GenerateCMOInvalidTask(const ge::Node &node, TaskBuilderContext &context);
  ~GenerateCMOInvalidTask() override;

  Status GenerateTask(std::vector<domi::TaskDef> &task_defs, const int32_t &stream_id,
                      const std::vector<CmoAttr> &cmo_attrs) override;
  Status ParseTensorInfo(const CmoAttr &cmo_attr, const TaskArgs &task_args, ge::DataType &data_type,
                         int64_t &complex_cmo_id, uint64_t &source_addr, uint32_t &length_inner) const;
 private:
  bool GetComplexCmoId(const CmoAttr &cmo_attr, int64_t &complex_cmo_id) const;
};
}  // namespace fe
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_OPS_KERNEL_BUILDER_TASK_BUILDER_CMO_TASK_GENERATE_CMO_INVALID_TASK_H_

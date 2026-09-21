/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_OPS_KERNEL_BUILDER_TASK_BUILDER_CMO_TASK_GENERATE_CMO_TASK_BASE_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_OPS_KERNEL_BUILDER_TASK_BUILDER_CMO_TASK_GENERATE_CMO_TASK_BASE_H_

#include "common/cmo_id_gen_strategy.h"
#include "common/aicore_util_attr_define.h"
#include "common/aicore_util_types.h"
#include "adapter/tbe_adapter/tbe_task_builder_adapter.h"
#include "graph/types.h"
#include "graph/utils/tensor_utils.h"
#include "runtime/rt_model.h"
#include "proto/task.pb.h"

namespace fe {
enum class rtCMOType {
  rtCMOBarrier = 0,
  rtCMOInvalid = 1,
  rtCMOPrefetch = 2,
  rtCMOWriteBack = 3,
};

const std::map<ge::DataType, uint8_t> DATA_TYPE_CODE = {
        {ge::DT_INT8, 0x0},
        {ge::DT_INT16, 0x1},
        {ge::DT_INT32, 0x2},
        {ge::DT_FLOAT16, 0x6},
        {ge::DT_FLOAT, 0x7},
};

class GenerateCMOTaskBase {
 public:
  GenerateCMOTaskBase(const ge::Node &node, TaskBuilderContext &context);
  virtual ~GenerateCMOTaskBase();

  virtual Status GenerateTask(std::vector<domi::TaskDef> &task_defs, const int32_t &stream_id,
                              const std::vector<CmoAttr> &cmo_attrs);
  Status ParseTensorInfo(const CmoAttr &cmo_attr, TaskArgs &task_args, ge::DataType &data_type,
                         uint64_t &source_addr, uint32_t &length_inner) const;
  Status InitCmoNodeAddrs(const ge::NodePtr &node, TaskArgs &task_args);

 protected:
  const ge::Node &node_;
  TaskBuilderContext context_;
};
}  // namespace fe
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_OPS_KERNEL_BUILDER_TASK_BUILDER_CMO_TASK_GENERATE_CMO_TASK_BASE_H_

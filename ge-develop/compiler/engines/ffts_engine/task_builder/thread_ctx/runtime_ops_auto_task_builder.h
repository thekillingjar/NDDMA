/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINE_FFTSENG_TASK_BUILDER_THREAD_CTX_RUNTIME_OPS_AUTO_TASK_BUILDER_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINE_FFTSENG_TASK_BUILDER_THREAD_CTX_RUNTIME_OPS_AUTO_TASK_BUILDER_H_
#include <memory>
#include "runtime_ops_manual_task_builder.h"
#include "task_builder/fftsplus_task_builder.h"
#include "proto/task.pb.h"
#include "common/opskernel/ops_kernel_info_types.h"
namespace ffts {
using FftsPlusCtxDefPtr = std::shared_ptr<domi::FftsPlusCtxDef>;
// used for label switch
class RuntimeOpsAutoTaskBuilder : public RuntimeOpsTaskBuilder {
 public:
  RuntimeOpsAutoTaskBuilder();
  ~RuntimeOpsAutoTaskBuilder() override;
  Status GenContextDef(const ge::NodePtr &node, domi::FftsPlusTaskDef *ffts_plus_task_def) override;
 private:
  RuntimeOpsAutoTaskBuilder(const RuntimeOpsAutoTaskBuilder &builder) = delete;
  RuntimeOpsAutoTaskBuilder &operator=(const RuntimeOpsAutoTaskBuilder &builder) = delete;
};
}  // namespace ffts
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINE_FFTSENG_TASK_BUILDER_THREAD_CTX_RUNTIME_OPS_MANUAL_TASK_BUILDER_H_
 

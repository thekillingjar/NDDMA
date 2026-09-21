/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RTS_ENGINE_OPS_KERNEL_STORE_RTS_FFTS_PLUS_OPS_KERNEL_BUILDER_H
#define RTS_ENGINE_OPS_KERNEL_STORE_RTS_FFTS_PLUS_OPS_KERNEL_BUILDER_H

#include <vector>
#include <map>
#include "common/opskernel/ops_kernel_builder.h"

namespace cce {
namespace runtime {
using domi::TaskDef;
using std::map;
using std::string;
using std::vector;

class RtsFftsPlusOpsKernelBuilder : public ge::OpsKernelBuilder {
 public:
  RtsFftsPlusOpsKernelBuilder() {
    isSupportFftsPlus_ = false;
  }

  ~RtsFftsPlusOpsKernelBuilder() override = default;

  /**
   * Initialize related resources of the rts kernel info store
   * @return status whether this operation success
   */
  ge::Status Initialize(const map<string, string> &options) override;

  /**
   * Release related resources of the rts kernel info store
   * @return status whether this operation success
   */
  ge::Status Finalize() override;

  /**
   * Calc the running size of Operator,
   * then GE will alloc the mem size from runtime
   * @param geNode Node information
   * @return status whether this operation success
   */
  ge::Status CalcOpRunningParam(ge::Node &geNode) override;

  /**
   * call the runtime's interface to generate the task
   * @param geNode Node information
   * @param context run context info
   * @return status whether this operation success
   */
  ge::Status GenerateTask(const ge::Node &geNode, ge::RunContext &context, vector<TaskDef> &tasks) override;

  /**
   * call the runtime's interface to update the task
   * @param geNode Node information
   * @return status whether this operation success
   */
  ge::Status UpdateTask(const ge::Node &geNode, vector<TaskDef> &tasks) override;
  // Copy prohibited
  RtsFftsPlusOpsKernelBuilder(const RtsFftsPlusOpsKernelBuilder &opsKernelBuilder) = delete;

  // Move prohibited
  RtsFftsPlusOpsKernelBuilder(const RtsFftsPlusOpsKernelBuilder &&opsKernelBuilder) = delete;

  // Copy prohibited
  RtsFftsPlusOpsKernelBuilder &operator=(const RtsFftsPlusOpsKernelBuilder &opsKernelBuilder) = delete;

  // Move prohibited
  RtsFftsPlusOpsKernelBuilder &operator=(RtsFftsPlusOpsKernelBuilder &&opsKernelBuilder) = delete;

 private:
  bool isSupportFftsPlus_;
};
}  // namespace runtime
}  // namespace cce

#endif  // RTS_FFTS_PLUS_OPS_KERNEL_BUILDER_H

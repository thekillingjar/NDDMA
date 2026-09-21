/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GE_LOCAL_ENGINE_OPS_KERNEL_UTILS_GE_LOCAL_OPS_KERNEL_UTILS_H_
#define GE_GE_LOCAL_ENGINE_OPS_KERNEL_UTILS_GE_LOCAL_OPS_KERNEL_UTILS_H_

#include "ge/ge_api_error_codes.h"
#include "common/opskernel/ops_kernel_builder.h"
#include "ge_local_ops_kernel_calc_op_param.h"

namespace ge {
namespace ge_local {
class GE_FUNC_VISIBILITY GeLocalOpsKernelBuilder : public OpsKernelBuilder {
 public:
  ~GeLocalOpsKernelBuilder() override;
  Status Initialize(const std::map<std::string, std::string> &options) override;

  Status Finalize() override;

  Status CalcOpRunningParam(Node &node) override;

  Status GenerateTask(const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) override;

 private:
  using CalcOpParamCall = std::function<graphStatus(const Node &node)>;

  graphStatus CalcMemSizeByNodeType(OpDescPtr &op_desc, GeTensorDesc &output_tensor,
                                    int64_t &output_mem_size, const std::string &node_type);
  graphStatus SetSizeForStringInput(Node &node, GeTensorDesc &output_tensor,
                                    size_t output_index);
};
}  // namespace ge_local
}  // namespace ge

#endif  // GE_GE_LOCAL_ENGINE_OPS_KERNEL_UTILS_GE_LOCAL_OPS_KERNEL_UTILS_H_

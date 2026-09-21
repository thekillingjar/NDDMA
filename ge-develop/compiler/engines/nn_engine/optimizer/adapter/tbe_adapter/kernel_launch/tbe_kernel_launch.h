/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_ADAPTER_TBE_ADAPTER_KERNEL_LAUNCH_TBE_KERNEL_LAUNCH_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_ADAPTER_TBE_ADAPTER_KERNEL_LAUNCH_TBE_KERNEL_LAUNCH_H_

#include <securec.h>
#include <memory>
#include <string>
#include <vector>
#include "proto/task.pb.h"
#include "common/resource_def.h"
#include "graph/anchor.h"
#include "graph/node.h"
#include "runtime/base.h"
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"

namespace fe {
class TbeKernelLaunch {
 public:
  explicit TbeKernelLaunch(int32_t input_num) : input_num_(input_num) {};
  virtual ~TbeKernelLaunch() {};
  Status DealKernelLaunch(const ge::Node &node, const void *args, const uint32_t &args_size,
                          const std::string &stub_func, const uint32_t &core_dim, domi::TaskDef &task_def);

  virtual size_t GetAppendArgsSizeOf();
  virtual size_t GetAppendArgsNum();
  virtual Status AddAppendArgs(const ge::Node &node, void *all_args_buff, const uint32_t &args_size);

  static bool KernelLaunch(const std::string &stub_func, const uint32_t block_dim, const void *args, uint32_t args_size,
                           const rtSmDesc_t *sm_desc, domi::TaskDef &task_def);
  static bool KernelLaunchWithHandle(const uint32_t block_dim, const void *args, uint32_t args_size,
                                     const rtSmDesc_t *sm_desc, domi::TaskDef &task_def);
 protected:
  int32_t input_num_;

 private:
  void PrintAllArgs(const string &op_name, const string &op_type, const void *all_args_buff, uint32_t args_size);
};
}  // namespace fe
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_ADAPTER_TBE_ADAPTER_KERNEL_LAUNCH_TBE_KERNEL_LAUNCH_H_

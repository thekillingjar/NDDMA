/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_AICPU_BG_LAUNCH_H_
#define AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_AICPU_BG_LAUNCH_H_
#include "bg_aicpu_arg.h"
#include "exe_graph/lowering/value_holder.h"
#include "graph/op_desc.h"
#include "proto/task.pb.h"
#include "register/node_converter_registry.h"
#include "engine/aicpu/kernel/aicpu_resource_manager.h"

namespace gert {
namespace bg {
struct IoInfo {
  std::vector<DevMemValueHolderPtr> input_addrs;
  std::vector<ValueHolderPtr> input_shapes;
  std::vector<ValueHolderPtr> output_sizes;
  std::vector<ValueHolderPtr> output_shapes;
};

ValueHolderPtr UpdateAicpuIoAddr(const ValueHolderPtr &args_handler,
                                 const std::vector<DevMemValueHolderPtr> input_addrs,
                                 const std::vector<ValueHolderPtr> output_addrs);
ValueHolderPtr AicpuTfLaunchKernel(const ValueHolderPtr &args_handler, const ValueHolderPtr &stream,
                                   const ValueHolderPtr &bin_handler, const ge::NodePtr node);
ValueHolderPtr AicpuCCLaunchKernel(const ValueHolderPtr &args_handler, const ValueHolderPtr &stream,
                                   const ValueHolderPtr &block_dim, const domi::KernelDef &kernel_def,
                                   const ge::OpDescPtr &op_desc, const ValueHolderPtr &ext_info_handler,
                                   const ValueHolderPtr &bin_handle, const ge::NodePtr node);
ValueHolderPtr AicpuHostComputeByCpuKernel(const ge::NodePtr &node, const AicpuArgs &args,
                                           const IoInfo &io_info, LoweringGlobalData &global_data,
                                           std::vector<DevMemValueHolderPtr> &output_addrs);
ValueHolderPtr AicpuHostExecFuncProcess(const AicpuHostProcFunc &func,
                                        const IoInfo &io_info,
                                        const std::vector<DevMemValueHolderPtr> &output_addrs);
ValueHolderPtr AicpuHostCompute(const ge::NodePtr &node, const AicpuArgs &args, const IoInfo &io_info,
                                LoweringGlobalData &global_data, std::vector<DevMemValueHolderPtr> &output_addrs);
ValueHolderPtr GetContainerIdHolder(const LowerInput &lower_input);
}  // namespace bg
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_AICPU_BG_LAUNCH_H_

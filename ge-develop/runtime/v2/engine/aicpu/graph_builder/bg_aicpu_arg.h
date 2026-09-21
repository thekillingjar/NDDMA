/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_AICPU_ARG_H_
#define AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_AICPU_ARG_H_
#include "bg_ext_info.h"
#include "runtime/kernel.h"
#include "fwk_adpt_struct.h"
#include "aicpu_engine_struct.h"
#include "graph_builder/bg_infer_shape.h"

namespace gert {
namespace bg {
struct AicpuArgs {
  ValueHolderPtr args_handler;
  ValueHolderPtr ext_info_handler;
};

struct TfArgsInfo {
  domi::KernelExDef kernel_ex_def;
  size_t io_num;
  ValueHolderPtr session_id;
  ValueHolderPtr step_id;
};

struct RtsArgs {
  ValueHolderPtr bin_handle;
};

AicpuArgs BuildHostCCAicpuArg(const ge::NodePtr node, const domi::KernelDef &kernel_def, const size_t io_num,
                              ValueHolderPtr& session_id);
AicpuArgs BuildCCAicpuArg(const ge::NodePtr node, const domi::KernelDef &kernel_def, const size_t io_num,
                          ValueHolderPtr& session_id, const bool &is_memcpy_task);
AicpuArgs BuildTfAicpuArg(const ge::NodePtr node, const TfArgsInfo &tf_args_info, const bool &is_memcpy_task);

ValueHolderPtr CalcBlockDim(const ge::OpDescPtr &op_desc, const std::vector<ValueHolderPtr> &input_shapes);

std::vector<ValueHolderPtr> GetMemAllocShape(const ge::NodePtr &node, const std::vector<ValueHolderPtr> &input_shapes,
                                             LoweringGlobalData &global_data);

inline bool IsAicpuUnknownShape(const ge::NodePtr &node) {
  bool forced_unknown = false;
  (void)ge::AttrUtils::GetBool(node->GetOpDescBarePtr(), ge::ATTR_NAME_FORCE_UNKNOWN_SHAPE, forced_unknown);
  return forced_unknown || IsUnkownShape(node->GetOpDesc());
}

inline bool IsAicpuOutputUnknownShape(const ge::NodePtr &node) {
  bool forced_unknown = false;
  (void)ge::AttrUtils::GetBool(node->GetOpDescBarePtr(), ge::ATTR_NAME_FORCE_UNKNOWN_SHAPE, forced_unknown);
  return forced_unknown || IsOutputUnkownShape(node->GetOpDesc());
}


RtsArgs BuildTfArgsBinHandle(const ge::NodePtr node);


RtsArgs BuildCCArgsBinHandle(const ge::NodePtr node);
} // bg
} // gert
#endif // AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_AICPU_ARG_H_

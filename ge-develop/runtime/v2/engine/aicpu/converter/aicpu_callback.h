/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_NODE_CONVERTER_AICPU_CALLBACK_H_
#define AIR_CXX_RUNTIME_V2_NODE_CONVERTER_AICPU_CALLBACK_H_
#include "graph/node.h"
#include "exe_graph/lowering/dev_mem_value_holder.h"
#include "exe_graph/lowering/value_holder.h"
#include "exe_graph/lowering/lowering_global_data.h"
#include "engine/aicpu/graph_builder/bg_ext_info.h"

namespace gert {
struct NodeOutput {
  std::vector<bg::ValueHolderPtr> shapes;
  std::vector<bg::ValueHolderPtr> addrs;
};

bg::ValueHolderPtr GetStepId(LoweringGlobalData &global_data);
void SetReleaseAfter(const std::vector<bg::DevMemValueHolderPtr> &addrs, const bg::ValueHolderPtr &launch_holder);
void SetReleaseAfter(const std::vector<bg::ValueHolderPtr> &addrs, const bg::ValueHolderPtr &launch_holder);
NodeOutput GetOutputShapeAndAddr(const ge::NodePtr &node, const std::vector<bg::ValueHolderPtr> &input_shapes,
                                 const std::vector<bg::DevMemValueHolderPtr> &input_addrs,
                                 LoweringGlobalData &global_data);
void AicpuCallback(const ge::NodePtr &node, const bg::ValueHolderPtr &ext_info, bg::ValueHolderPtr &launch_holder,
                   LoweringGlobalData &global_data, NodeOutput &node_output);
} // gert
#endif  // AIR_CXX_RUNTIME_V2_NODE_CONVERTER_AICPU_CALLBACK_H_

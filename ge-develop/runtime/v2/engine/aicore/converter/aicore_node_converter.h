/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_NODE_CONVERTER_AICORE_NODE_CONVERTER_H_
#define AIR_CXX_RUNTIME_V2_NODE_CONVERTER_AICORE_NODE_CONVERTER_H_
#include "graph/node.h"
#include "engine/aicore/converter/bg_kernel_launch.h"
#include "register/node_converter_registry.h"
#include "engine/aicore/fe_rt2_common.h"
namespace gert {
LowerResult LoweringAiCoreNode(const ge::NodePtr &node, const LowerInput &lower_input);
std::vector<bg::ValueHolderPtr> InferAiCoreStorageShape(const ge::NodePtr &node,
    const std::vector<bg::ValueHolderPtr> &input_shapes, LoweringGlobalData &global_data);
std::vector<bg::ValueHolderPtr> SetOutputShape(const ge::NodePtr &node,
                                               const bg::DevMemValueHolderPtr &shapebuffer_addr,
                                               const bg::ValueHolderPtr &launch_arg_ref,
                                               const bg::ValueHolderPtr &stream,
                                               const std::vector<bg::ValueHolderPtr> &output_shapes);
bool IsThirdClassOp(const ge::OpDescPtr &op_desc);
ge::Status GetQosInfo(bg::ValueHolderPtr &qos);
ge::NodePtr BuildAtomicNode(const ge::NodePtr &origin_node, const bg::AtomicLoweringArg &atomic_lowering_arg,
                            std::vector<bg::ValueHolderPtr> &output_clean_sizes,
                            std::vector<bg::DevMemValueHolderPtr> &output_clean_addrs, ge::ComputeGraphPtr &graph);
bool NeedCheckEmptyOutput(const ge::NodePtr &node);
bool IsAllTensorNotEmpty();
bool IsSingleOpScene(const ge::NodePtr &node);
DfxExeArg GetOpDfxExeArg(const ge::NodePtr &node);
ge::Status AddDataNodeForAtomic(ge::ComputeGraphPtr &graph, ge::NodePtr &clean_node, size_t output_size);
LowerResult LoweringStaticAicoreNode(const ge::NodePtr &node, const LowerInput &lower_input);
}
#endif  // AIR_CXX_RUNTIME_V2_NODE_CONVERTER_AICORE_NODE_CONVERTER_H_

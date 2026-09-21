/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_GRAPH_TUNNER_RT2_STUB_H
#define AIR_CXX_GRAPH_TUNNER_RT2_STUB_H
#include "exe_graph/runtime/continuous_vector.h"
#include "exe_graph/lowering/value_holder.h"
#include "graph_builder/value_holder_generator.h"
#include "graph/debug/ge_attr_define.h"
namespace tune {
using namespace gert;
extern "C" ge::graphStatus FFTSNodeThreadV2(const ge::NodePtr &node,
    const std::vector<bg::ValueHolderPtr> &input_shapes, const std::vector<bg::ValueHolderPtr> &output_shapes,
    const bg::ValueHolderPtr thread_dim, std::vector<bg::ValueHolderPtr> &output);
extern "C" ge::graphStatus FFTSGraphPreThreadV2(const ge::ComputeGraphPtr sub_graph,
    const std::vector<bg::ValueHolderPtr> &input_shapes, std::vector<bg::ValueHolderPtr> &output);
extern "C" ge::graphStatus FFTSGraphPreThreadV2New(const ge::ComputeGraphPtr &sub_graph,
    const std::vector<bg::ValueHolderPtr> &input_shapes, const std::vector<bg::ValueHolderPtr> &input_addrs,
    std::vector<bg::ValueHolderPtr> &output);
}
#endif  // AIR_CXX_GRAPH_TUNNER_RT2_STUB_H

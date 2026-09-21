/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_NODE_CONVERTER_DSACORE_NODE_CONVERTER_H_
#define AIR_CXX_RUNTIME_V2_NODE_CONVERTER_DSACORE_NODE_CONVERTER_H_
#include "graph/node.h"
#include "register/node_converter_registry.h"
#include "runtime/rt.h"
#include "exe_graph/lowering/value_holder.h"
#include "exe_graph/lowering/lowering_global_data.h"

namespace gert {
LowerResult LoweringDsaCoreNode(const ge::NodePtr &node, const LowerInput &lower_input);
void LoweringDsaNodeFillSqeSolidInfo(std::unique_ptr<rtStarsDsaSqe_t> &dsa_sqe,
                                     const domi::DSATaskDef &dsa_task);
void LoweringDsaNodeFillTaskInfo(std::vector<bg::ValueHolderPtr> &inputs,
                                 const std::vector<bg::DevMemValueHolderPtr> &input_addrs,
                                 const std::vector<bg::DevMemValueHolderPtr> &output_addrs,
                                 const domi::DSATaskDef &dsa_task);
}
#endif  // AIR_CXX_RUNTIME_V2_NODE_CONVERTER_DSACORE_NODE_CONVERTER_H_

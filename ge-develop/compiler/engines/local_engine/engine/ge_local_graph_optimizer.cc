/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ge_local_graph_optimizer.h"
#include "graph/debug/ge_op_types.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/common/debug/ge_log.h"

namespace ge {

GeLocalGraphOptimizer::~GeLocalGraphOptimizer() {}

ge::Status GeLocalGraphOptimizer::Initialize(const std::map<std::string, std::string>& options,
    ge::OptimizeUtility* const optimize_utility) {
    (void)options;
    (void)optimize_utility;
    return SUCCESS;
}

ge::Status GeLocalGraphOptimizer::Finalize() {
    return SUCCESS;
}

ge::Status GeLocalGraphOptimizer::OptimizeOriginalGraph(ge::ComputeGraph& graph) {
    (void)graph;
    return SUCCESS;
}

ge::Status GeLocalGraphOptimizer::OptimizeOriginalGraphJudgeInsert(ge::ComputeGraph& graph) {
    GELOGD("GeLocalGraphOptimizer OptimizeOriginalGraphJudgeInsert in.");

    for (auto &node : graph.GetAllNodes()) {
        if (node->GetType() == PHONYCONCAT) {
            ge::OpDescPtr op_desc = node->GetOpDesc();
            (void)ge::AttrUtils::SetBool(op_desc, ge::ATTR_NAME_NOTASK, true);
            (void)ge::AttrUtils::SetBool(op_desc, ge::ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
            (void)ge::AttrUtils::SetBool(op_desc, ge::ATTR_NAME_OUTPUT_REUSE_INPUT, true);
            (void)ge::AttrUtils::SetInt(op_desc, ge::ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);
        }

        if (node->GetType() == PHONYSPLIT) {
            ge::OpDescPtr op_desc = node->GetOpDesc();
            (void)ge::AttrUtils::SetBool(op_desc, ge::ATTR_NAME_NOTASK, true);
            (void)ge::AttrUtils::SetBool(op_desc, ge::ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT, true);
            (void)ge::AttrUtils::SetBool(op_desc, ge::ATTR_NAME_OUTPUT_REUSE_INPUT, true);
            (void)ge::AttrUtils::SetInt(op_desc, ge::ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);
        }
    }

    return SUCCESS;
}

ge::Status GeLocalGraphOptimizer::OptimizeFusedGraph(ge::ComputeGraph& graph) {
    (void)graph;
    return SUCCESS;
}

ge::Status GeLocalGraphOptimizer::OptimizeWholeGraph(ge::ComputeGraph& graph) {
    (void)graph;
    return SUCCESS;
}

ge::Status GeLocalGraphOptimizer::GetAttributes(ge::GraphOptimizerAttribute& attrs) const {
    attrs.engineName = "DNN_VM_GE_LOCAL";
    return SUCCESS;
}

} // namespace ge

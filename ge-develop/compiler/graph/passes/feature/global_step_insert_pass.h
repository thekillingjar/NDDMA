/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_GLOBAL_STEP_INSERT_PASS_H_
#define GE_GRAPH_PASSES_GLOBAL_STEP_INSERT_PASS_H_

#include <string>
#include <vector>

#include "framework/common/ge_inner_error_codes.h"
#include "graph/passes/graph_pass.h"

namespace ge {
/// Add global step op to the computeGraph when needed.
/// [Notice]: this pass must work before graph partitioner start work
///  in order to make the global step variable place in known subgraph
class GlobalStepInsertPass : public GraphPass {
 public:
 /// @param compute_graph graph
 /// @return SUCCESS: do success
 ///         NOT_CHANGED : do nothing
 ///         Other: failed
 Status Run(ComputeGraphPtr compute_graph) override;

 private:
 /// Universal insert node to graph.
 /// @param compute_graph graph
 /// @param node_type inserted node type
 /// @param node_name inserted node name
 /// @param input_list input desc list
 /// @param output_list output desc list
 /// @return the inserted node. if insert failed return nullptr.
 NodePtr InsertOp(ComputeGraphPtr &compute_graph, const std::string &node_type, const std::string &node_name,
                  const std::vector<GeTensorDesc> &input_list, const std::vector<GeTensorDesc> &output_list) const;
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_GLOBAL_STEP_INSERT_PASS_H_
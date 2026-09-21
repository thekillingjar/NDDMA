/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PARSER_COMMON_PARSER_UTILS_H_
#define PARSER_COMMON_PARSER_UTILS_H_

#include <unordered_map>
#include "graph/graph.h"
#include "graph/node.h"
#include "ge/ge_api_error_codes.h"
#include "ge/ge_api_types.h"

namespace ge {
class ParserUtils {
 public:
  using OutputNodeInfo = std::pair<std::string, int32_t>;
  using OutputMapping = std::unordered_map<std::string, OutputNodeInfo>;
  static Status ExpandOneToManyGraph(const Graph &graph, OutputMapping &output_mapping);
  static string GenOutputKey(const OutputNodeInfo &node_info);
  static void UpdateOutputNodeInfo(const OutputMapping &final_output_nodes, OutputNodeInfo &output_node_info);
  static void UpdateOutputCtx(const OutputMapping &final_output_nodes, OutputMapping &tensor_to_nodes);
  static std::string GetOperatorName(const Operator &op);
  static std::string GetOperatorType(const Operator &op);
  static std::string GetGraphName(const Graph &graph);

 private:
  static Status ExpandNodeToSubgraph(const Graph &subgraph, const NodePtr &node, const Graph &graph,
                                     OutputMapping &output_mapping);
  static Status HandleInputContext(const NodePtr &node,
                                   const std::vector<NodePtr> &input_nodes,
                                   const ComputeGraphPtr &compute_graph);
  static Status HandleOutputContext(const NodePtr &node,
                                    const std::vector<std::pair<NodePtr, int32_t>> &out_node_index,
                                    OutputMapping &output_mapping);
};
}  // namespace ge
#endif  // PARSER_COMMON_PARSER_UTILS_H_

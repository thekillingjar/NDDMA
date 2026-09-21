/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_PARSER_ONNX_SUBGRAPH_ADAPTER_IF_SUBGRAPH_ADAPTER_H_
#define GE_PARSER_ONNX_SUBGRAPH_ADAPTER_IF_SUBGRAPH_ADAPTER_H_

#include <set>
#include <string>
#include "subgraph_adapter.h"
#include "parser/onnx/onnx_util.h"

namespace ge {
class PARSER_FUNC_VISIBILITY IfSubgraphAdapter : public SubgraphAdapter {
 public:
  IfSubgraphAdapter() = default;
  virtual ~IfSubgraphAdapter() override = default;
  domi::Status AdaptAndFindAllSubgraphs(ge::onnx::NodeProto *parent_node,
                                        std::vector<ge::onnx::GraphProto *> &onnx_graphs,
                                        std::map<std::string, ge::onnx::GraphProto *> &name_to_onnx_graph,
                                        const std::string &parent_graph_name = "") override;

 private:
  domi::Status ParseIfNodeSubgraphs(ge::onnx::NodeProto &parent_node, std::vector<ge::onnx::GraphProto *> &onnx_graphs,
                                    std::map<std::string, ge::onnx::GraphProto *> &name_to_onnx_graph,
                                    const std::string &parent_graph_name) const;
  domi::Status GetSubgraphsAllInputs(ge::onnx::GraphProto &onnx_graph, std::set<std::string> &all_inputs) const;
  void AddInputNodeForGraph(const std::set<std::string> &all_inputs, ge::onnx::GraphProto &onnx_graph) const;
  void AddInputForParentNode(const std::set<std::string> &all_inputs, ge::onnx::NodeProto &parent_node) const;
};
}  // namespace ge

#endif  // GE_PARSER_ONNX_SUBGRAPH_ADAPTER_IF_SUBGRAPH_ADAPTER_H_

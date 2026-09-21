/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LLT_FUSION_ENGINE_GRAPH_CONSTRUCTOR_GRAPH_BUILDER_UTILS_H_
#define LLT_FUSION_ENGINE_GRAPH_CONSTRUCTOR_GRAPH_BUILDER_UTILS_H_

#include <vector>
#include <string>
#include <set>

#include "graph/node.h"
#include "graph/graph.h"
#include "graph/compute_graph.h"

namespace ffts {
namespace ut {
class ComputeGraphBuilder {
 public:
  explicit ComputeGraphBuilder(const std::string &name) {
    graph_ = std::make_shared<ge::ComputeGraph>(name);
  }
  ge::NodePtr AddNode(const std::string &name,
                  const std::string &type,
                  int in_cnt,
                  int out_cnt,
                  ge::Format format = ge::FORMAT_NCHW,
                  ge::DataType data_type = ge::DT_FLOAT,
                  std::vector<int64_t> shape = {1, 1, 224, 224});
  ge::NodePtr AddNodeWithImplyType(const std::string &name,
                      const std::string &type,
                      int in_cnt,
                      int out_cnt,
                      ge::Format format = ge::FORMAT_NCHW,
                      ge::DataType data_type = ge::DT_FLOAT,
                      std::vector<int64_t> shape = {1, 1, 224, 224});
  void AddDataEdge(ge::NodePtr &src_node, int src_idx, ge::NodePtr &dst_node, int dst_idx);
  void AddControlEdge(ge::NodePtr &src_node, ge::NodePtr &dst_node);
  ge::ComputeGraphPtr GetGraph() {
    graph_->TopologicalSorting();
    return graph_;
  }
  bool SetConstantInputOffset();
 private:
  ge::ComputeGraphPtr graph_;
};

std::set<std::string> GetNames(const ge::Node::Vistor<ge::NodePtr> &nodes);
std::set<std::string> GetNames(const ge::ComputeGraph::Vistor<ge::NodePtr> &nodes);
}
}

#endif  // LLT_FUSION_ENGINE_GRAPH_CONSTRUCTOR_GRAPH_BUILDER_UTILS_H_

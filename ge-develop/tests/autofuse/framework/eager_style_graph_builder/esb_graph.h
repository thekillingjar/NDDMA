/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_ESB_GRAPH_H_
#define AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_ESB_GRAPH_H_
#include "graph/compute_graph.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "common/checker.h"

#include "esb_tensor.h"

class EsbGraph {
 public:
  EsbGraph() : EsbGraph("graph") {}
  explicit EsbGraph(const char *name) : graph_(std::make_shared<ge::ComputeGraph>(name)), nodes_num_(0) {}

  EsbTensor *GetEsbTensorFromNode(ge::NodePtr node, int32_t output_index);

  EsbTensor *AppendGraphInput(const ge::char_t *name=nullptr, const ge::char_t *type=nullptr) {
    return AddGraphInput(static_cast<int32_t>(graph_input_indexes_.size()), name, type);
  }
  EsbTensor *AddGraphInput(int32_t index, const ge::char_t *name=nullptr, const ge::char_t *type=nullptr);

  ge::Status SetGraphOutput(EsbTensor *tensor, int32_t output_index);

  ge::ComputeGraph *GetComputeGraph() {
    return graph_.get();
  }
  ge::ComputeGraphPtr BuildComputeGraph();

  std::unique_ptr<ge::Graph> BuildGraph();

  int64_t NextNodeIndex() {
    return nodes_num_++;
  }

 private:
  bool IsGraphValid() const;
  EsbTensor *GetEsbTensorFromNodeInner(ge::NodePtr node, int32_t output_index);

 private:
  ge::ComputeGraphPtr graph_;
  std::list<std::unique_ptr<EsbTensor>> tensors_holder_;
  std::set<int32_t> graph_input_indexes_;
  std::map<int32_t, EsbTensor *> output_indexes_to_tensor_;
  int64_t nodes_num_;
};

#endif  // AIR_CXX_COMPILER_GRAPH_EAGER_STYLE_GRAPH_BUILDER_ESB_GRAPH_H_

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATT_GRAPH_CONSTRUCT_UTILS_H
#define ATT_GRAPH_CONSTRUCT_UTILS_H
#include <vector>
#include "ascir_ops.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/normal_graph/ge_tensor_impl.h"
namespace att {
class GraphConstructUtils {
 public:
  static void UpdateVectorizedStride(const std::vector<int64_t> &axis, const std::vector<ge::Expression> &strides,
                                     const std::vector<int64_t> &vectorized_axis,
                                     std::vector<ge::Expression> &vectorized_strides);
  static void UpdateGraphVectorizedStride(ge::AscGraph &graph);
  static void UpdateGraphsVectorizedStride(std::vector<ge::AscGraph> &impl_graphs);
  static ge::Status UpdateTensorAxes(const std::vector<ge::Axis> &axes, ge::AscOpOutput &output, int32_t loop_id = -1);
  static ge::Status UpdateOutputTensorAxes(const std::vector<ge::Axis> &axes, std::vector<ge::AscOpOutput> &&outputs,
                                           int32_t loop_id = -1);
  static ge::Status CreateSimpleLoadStoreOp(ge::AscGraph &graph);
  static ge::AscNodePtr ConstructSingleOp(const std::string &op_type, int32_t in_cnt, int32_t out_cnt);
  static ge::Status BuildConcatGroupAscendGraphS0S1ReduceMultiTiling(ge::AscGraph &graph);
};

class GraphBuilder {
 public:
  explicit GraphBuilder(const std::string &name) {
    graph_ = std::make_shared<ge::ComputeGraph>(name);
  }

  GraphBuilder(const std::string &name, const std::string &node_type) {
    graph_ = std::make_shared<ge::ComputeGraph>(name);
    node_type_ = node_type;
  }

  ge::NodePtr AddNode(const std::string &name, const std::string &type, const int in_cnt, const int out_cnt,
                      const std::vector<int64_t> shape = {1, 1, 1, 1}) {
    auto tensor_desc = std::make_shared<ge::GeTensorDesc>();
    tensor_desc->SetShape(ge::GeShape(std::move(shape)));
    tensor_desc->SetFormat(ge::FORMAT_NCHW);
    tensor_desc->SetDataType(ge::DT_FLOAT);

    auto op_desc = std::make_shared<ge::OpDesc>(name, (node_type_.empty()) ? type : "AscGraph");
    for (std::int32_t i = 0; i < in_cnt; ++i) {
      op_desc->AddInputDesc(tensor_desc->Clone());
    }
    for (std::int32_t i = 0; i < out_cnt; ++i) {
      op_desc->AddOutputDesc(tensor_desc->Clone());
    }
    op_desc->AddInferFunc([](ge::Operator &op) { return ge::GRAPH_SUCCESS; });
    return graph_->AddNode(op_desc);
  }

  ge::Status AddDataEdge(const ge::NodePtr &src_node, const std::int32_t src_idx, const ge::NodePtr &dst_node,
                         const std::int32_t dst_idx) {
    return ge::GraphUtils::AddEdge(src_node->GetOutDataAnchor(src_idx), dst_node->GetInDataAnchor(dst_idx));
  }

  ge::ComputeGraphPtr GetGraph() {
    graph_->TopologicalSorting();
    return graph_;
  }

 private:
  ge::ComputeGraphPtr graph_;
  std::string node_type_;
};
}  // namespace att
#endif  // ATT_GRAPH_CONSTRUCT_UTILS_H

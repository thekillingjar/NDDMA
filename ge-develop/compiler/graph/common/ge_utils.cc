/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ge/ge_utils.h"
#include "trans_op_creator.h"
#include "graph/utils/node_adapter.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/op_type_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "common/checker.h"
#include "graph/shape_refiner.h"
#include "graph/passes/base_pass.h"
#include "graph/ir_definitions_recover.h"
#include "graph/passes/shape_optimize/infershape_pass.h"
namespace ge {
namespace {
const std::string kAicoreEngineName = "AIcoreEngine";
bool IsGraphInput(const NodePtr &node, size_t &input_index) {
  if (!OpTypeUtils::IsDataNode(node->GetType())) {
    return false;
  }
  if (!AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_INDEX, input_index)) {
    return false;
  }
  return true;
}

Status RefreshGraphInputShapes(ComputeGraphPtr &graph, const std::vector<Shape> &input_shape) {
  std::map<size_t, NodePtr> index_2_input_node;
  for (const auto &node : graph->GetDirectNode()) {
    size_t input_index = 0u;
    if (IsGraphInput(node, input_index)) {
      GE_ASSERT_TRUE(input_index < input_shape.size(),
                     "Data node[%s][%s] index[%ld] is out of range of given input_shape size[%zu]", node->GetNamePtr(),
                     node->GetTypePtr(), input_index, input_shape.size());
      index_2_input_node[input_index] = node;
    }
  }
  GE_ASSERT_TRUE(index_2_input_node.size() == input_shape.size(),
                 "Input shape size %zu, not equal with input nodes count %zu", input_shape.size(),
                 index_2_input_node.size());

  for (const auto &idx_2_node : index_2_input_node) {
    const auto data_node = idx_2_node.second;
    const auto input_index = idx_2_node.first;
    auto input_desc = data_node->GetOpDesc()->MutableInputDesc(0);
    auto output_desc = data_node->GetOpDesc()->MutableInputDesc(0);
    const auto origin_shape = input_desc->GetShape();
    const auto ge_shape = GeShape(input_shape[input_index].GetDims());
    input_desc->SetShape(ge_shape);
    input_desc->SetOriginShape(ge_shape);
    output_desc->SetShape(ge_shape);
    output_desc->SetOriginShape(ge_shape);
    GELOGI("Update Graph[%s] Data node [%s][%s] input index[%zu] shape from [%s] to [%s]", graph->GetName().c_str(),
           data_node->GetNamePtr(), data_node->GetTypePtr(), input_index, origin_shape.ToString().c_str(),
           ge_shape.ToString().c_str());
  }
  return SUCCESS;
}
} // namespace

Status GeUtils::InferShape(const Graph &graph, const std::vector<Shape> &input_shape) {
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  GE_ASSERT_NOTNULL(compute_graph);
  GE_ASSERT_SUCCESS(RefreshGraphInputShapes(compute_graph, input_shape));
  GE_ASSERT_GRAPH_SUCCESS(ge::RecoverIrDefinitions(compute_graph), "Failed to recover ir definitions");

  GEPass ge_passes(compute_graph);
  NamesToPass names_to_passes;
  InferShapePass infer_shape_pass(nullptr, nullptr);
  names_to_passes.emplace_back("InferShapePass", &infer_shape_pass);

  Status ret = ge_passes.Run(names_to_passes, false);
  ShapeRefiner::ClearContextMap();
  return ret;
}

Status GeUtils::CheckNodeSupportOnAicore(const GNode &node, bool &is_supported, AscendString &unsupported_reason) {
  const auto node_ptr = NodeAdapter::GNode2Node(node);
  GE_ASSERT_NOTNULL(node_ptr);
  std::string reason;
  auto ret = TransOpCreator::CheckAccuracySupported(node_ptr->GetOpDesc(), kAicoreEngineName, is_supported, reason);
  unsupported_reason = AscendString(reason.c_str());
  return ret;
}
} // namespace ge
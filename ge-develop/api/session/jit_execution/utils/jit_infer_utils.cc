/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "jit_infer_utils.h"
#include "common/memory/tensor_trans_utils.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_shape_inference.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_info_pre_processor.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_infer_util.h"
#include "graph/utils/op_type_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "attribute_group/attr_group_symbolic_desc.h"
#include "opt_info/ge_opt_info.h"

#include <stdbool.h>

namespace ge {
namespace {
/**
 * 判断node是否为uninfered node, uninfered node会被切到下一张图
 *
 * 1、if/case节点, 符号化暂不支持对子图的推导, 因此if和case的输出必没有符号, 切到下一张图后可以根据上一张图的输出把cond构造成const,
 *    然后对if/case做剪枝, 消除if/case节点, 把子图展平到根图上, 达成对if/case做符号化推导的目的
 * 2、存在没有符号的输入且所有输出都没符号
 *
 * @param node
 * @return
 */
bool IsUnInferedNode(const NodePtr &node) {
  auto op_desc = node->GetOpDesc();
  size_t empty_input_count = 0u;
  for (size_t i = 0U; i < op_desc->GetAllInputsDescPtr().size(); ++i) {
    auto attr = op_desc->GetInputDesc(i).GetAttrsGroup<SymbolicDescAttr>();
    if (attr == nullptr) {
      empty_input_count++;
    }
  }
  size_t empty_output_count = 0;
  for (size_t i = 0U; i < op_desc->GetAllOutputsDescPtr().size(); ++i) {
    auto attr = op_desc->GetOutputDesc(i).GetAttrsGroup<SymbolicDescAttr>();
    if (attr == nullptr) {
      empty_output_count++;
    }
  }
  // 算子的输出都没有符号
  if (empty_output_count == op_desc->GetAllOutputsDescPtr().size()) {
    // 算子的输入也没符号切到下一张图
    if (empty_input_count != 0) {
      return true;
    }
    // 算子的输入有符号，是if/case节点，且条件输入不是data，切到下一张图
    auto cond_input = SymbolicInferUtil::GetCondInput(node);
    if (cond_input != nullptr && !OpTypeUtils::IsDataNode(cond_input->GetType())) {
      return true;
    }
  }
  return false;
}
}

Status JitInferUtils::InferSymbolForGraph(const ComputeGraphPtr &graph, const std::vector<GeTensor> &inputs,
                                          std::vector<NodePtr> &infered_nodes) {
  PrepareBeforeInferSymbol(graph, inputs);
  InferGraphAndGetInferredNodes(graph, inputs, infered_nodes);
  return SUCCESS;
}

Status JitInferUtils::PrepareBeforeInferSymbol(const ComputeGraphPtr &graph, const std::vector<GeTensor> &inputs) {
  GE_ASSERT_SUCCESS(GeOptInfo::SetOptInfo());

  GraphNodePtr graph_node = make_shared<GraphNode>(0);
  graph_node->SetComputeGraph(graph);
  GraphPtr graph_ptr = std::make_shared<Graph>(GraphUtilsEx::CreateGraphFromComputeGraph(graph));
  graph_node->SetGraph(graph_ptr);

  GraphPrepare graph_prepare;
  GraphOptimize graph_optimize;
  GE_ASSERT_SUCCESS(graph_prepare.PrepareInit(graph_node, graph->GetSessionID()));
  GE_ASSERT_SUCCESS(graph_optimize.HandleSummaryOp(graph));
  graph_prepare.SetGraphNormalized(true);  // to skip GraphOptimize::OptimizeAfterGraphNormalization
  GE_ASSERT_SUCCESS(graph_prepare.NormalizeGraph(graph, graph_node->GetOptions(), inputs));
  GE_ASSERT_SUCCESS(graph_prepare.PrepareDynShape());

  return SUCCESS;
}

static void ClearInferedNodesWithAllDataNodes(std::vector<NodePtr> &infered_nodes) {
  size_t data_node_num = 0;
  for (auto &node : infered_nodes) {
    auto node_type = node->GetType();
    if (OpTypeUtils::IsVariableNode(node_type) || OpTypeUtils::IsConstNode(node_type) ||
         OpTypeUtils::IsDataNode(node_type)) {
      data_node_num++;
    }
  }
  // avoid all infer nodes are data node
  if (data_node_num == infered_nodes.size()) {
    infered_nodes.clear();
  }
}

static bool ParentNodeInfered(const NodePtr &node, std::vector<NodePtr> &infered_nodes) {
  if (!node->GetInDataNodes().empty()) {
    for (auto &in_data_node : node->GetInDataNodes()) {
      // check parents data nodes
      if (std::find(infered_nodes.begin(), infered_nodes.end(),in_data_node) == infered_nodes.end()) {
          return false;
      }
    }
  }
  if (!node->GetInControlNodes().empty()) {
    for (auto &in_control_node : node->GetInControlNodes()) {
      // check parents control nodes
      if (std::find(infered_nodes.begin(), infered_nodes.end(),in_control_node) == infered_nodes.end()) {
          return false;
      }
    }
  }
  return true;
}

static void DeleteNodesWithoutParentNode(std::vector<NodePtr> &infered_nodes) {
  // delete nodes whose parents node not infered
  for (auto it = infered_nodes.begin(); it != infered_nodes.end();) {
    if (!ParentNodeInfered(*it, infered_nodes)) {
      GELOGD("Infer node:%s. Parent node uninfered", (*it)->GetName().c_str());
      it = infered_nodes.erase(it);
    } else {
      ++it;
    }
  }
}

Status JitInferUtils::InferGraphAndGetInferredNodes(const ComputeGraphPtr &graph, const std::vector<GeTensor> &inputs,
                                                    std::vector<NodePtr> &infered_nodes) {
  bool is_single_op_scene = false;
  ge::AttrUtils::GetBool(graph, ge::ATTR_SINGLE_OP_SCENE, is_single_op_scene);
  if (is_single_op_scene) {
    GELOGI("Skip auto fuse for single op scene.");
    return SUCCESS;
  }
  GE_ASSERT_GRAPH_SUCCESS(SymbolicShapeSymbolizer::Symbolize(graph, inputs), "Symbolize graph input failed, graph %s",
                          graph->GetName().c_str());
  SymbolicShapeInference symbolic_shape_inference;
  GE_ASSERT_SUCCESS(symbolic_shape_inference.Infer(graph));
  std::vector<NodePtr> uninfered_nodes;
  for (auto &node : graph->GetDirectNode()) {
    if (IsUnInferedNode(node)) {
      GELOGD("[%s][%s] add to uninfered_nodes.", node->GetNamePtr(), node->GetTypePtr());
      uninfered_nodes.push_back(node);
      continue;
    }
    infered_nodes.push_back(node);
  }
  if (uninfered_nodes.size() == 1 && uninfered_nodes[0]->GetType() == "NetOutput") {
    infered_nodes.push_back(uninfered_nodes[0]);
  }

  // 如果全是data/variable/const节点，清除vector
  ClearInferedNodesWithAllDataNodes(infered_nodes);

  DeleteNodesWithoutParentNode(infered_nodes);

  GELOGD("Infer node size: %d", infered_nodes.size());
  return SUCCESS;
}
}  // namespace ge
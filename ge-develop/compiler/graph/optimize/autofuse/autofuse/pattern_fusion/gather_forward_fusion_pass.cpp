/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gather_forward_fusion_pass.h"
#include "autofuser.h"
#include "ge_common/ge_api_error_codes.h"
#include "graph/debug/ge_attr_define.h"
#include "common/checker.h"
#include "graph/utils/graph_utils.h"
#include "graph/operator_reg.h"
#include "can_fuse/fusion_strategy_solver.h"
#include "utils/autofuse_utils.h"
#include "utils/auto_fuse_config.h"
#include "post_process/scheduler_adapter/adaption_fallback_load.h"
#include "lowering/lowerings.h"
#include "shape_refiner.h"
#include "backend/backend_spec.h"

namespace  ge {
using namespace autofuse;
std::string GetFuseType(const NodePtr &node) {
  LoweringManager::Lowering(node);
  auto node_out_anchor = node->GetOutDataAnchor(0);
  auto node_kernel_box = loop::GetKernelBox(node_out_anchor);
  std::string fuse_type = FuseTypeToString(node_kernel_box.Type());
  GELOGD("node: %s's fuse_type: %s", node->GetName().c_str(), fuse_type.c_str());
  return fuse_type;
}

ge::SymbolicDescAttr* GetNodeMutableOutputAttr(const NodePtr &node) {
  auto node_op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(node_op_desc);
  auto node_output_desc = node_op_desc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(node_output_desc);
  auto node_output_attr = node_output_desc->GetAttrsGroup<ge::SymbolicDescAttr>();
  return node_output_attr;
}

ge::SymbolicDescAttr* GetNodeMutableInputAttr(const NodePtr &node) {
  auto node_op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(node_op_desc);
  auto node_input_desc = node_op_desc->MutableInputDesc(0);
  GE_ASSERT_NOTNULL(node_input_desc);
  auto node_input_attr = node_input_desc->GetAttrsGroup<ge::SymbolicDescAttr>();
  return node_input_attr;
}

gert::SymbolShape GetNodeSymbolShape(const NodePtr &node) {
  auto node_mutable_output_attr = GetNodeMutableOutputAttr(node);
  auto node_origin_symbolic_shape = node_mutable_output_attr->symbolic_tensor.GetOriginSymbolShape();
  return node_origin_symbolic_shape;
}

Status SetNodeSymbolShape(const NodePtr &node, gert::SymbolShape &gather_symbolic_shape) {
  auto node_mutable_output_attr = GetNodeMutableOutputAttr(node);
  node_mutable_output_attr->symbolic_tensor.MutableOriginSymbolShape() = gather_symbolic_shape;
  auto node_mutable_input_attr = GetNodeMutableInputAttr(node);
  node_mutable_input_attr->symbolic_tensor.MutableOriginSymbolShape() = gather_symbolic_shape;
  return SUCCESS;
}

bool CheckShape(const NodePtr &node) {
  auto node_mutable_output_attr = GetNodeMutableOutputAttr(node);
  auto node_mutable_input_attr = GetNodeMutableInputAttr(node);
  if (node_mutable_output_attr == nullptr || node_mutable_input_attr == nullptr) {
    GELOGD("now ele node %s's input or output symbolic shape is null", node->GetType().c_str());
    return false;
  }
  auto node_mutable_output_symbolic_shape = node_mutable_output_attr->symbolic_tensor.MutableOriginSymbolShape() ;
  auto node_mutable_input_symbolic_shape = node_mutable_input_attr->symbolic_tensor.MutableOriginSymbolShape() ;
  if (node_mutable_input_symbolic_shape != node_mutable_output_symbolic_shape) {
    return false;
  }
  return true;
}

bool NodeForwardFusionJudge(const NodePtr &node) {
  if (GetFuseType(node) != "pointwise") {
    GELOGD("now node's is %s, which is not pointwise", node->GetType().c_str());
    return false;
  }
  if (node->GetInNodesSize() != 1 || node->GetOutNodesSize() != 1) {
    GELOGD("now node's input node's num is %lu, output node's num is %u", node->GetInNodesSize(), node->GetOutNodesSize());
    return false;
  }
  if (!CheckShape(node)) {
    GELOGD("now node %s's input shape and output shape do not match exactly or once input or output symbolic shape is null", node->GetType().c_str());
    return false;
  }
  std::set<ge::DataType> gather_param_support_dtype = {DT_FLOAT, DT_FLOAT16, DT_INT32, DT_INT16, DT_UINT16, DT_UINT32, DT_BF16};
  auto node_op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(node_op_desc);
  auto node_input_desc = node_op_desc->MutableInputDesc(0);
  GE_ASSERT_NOTNULL(node_input_desc);
  auto node_input_desc_dtype = node_input_desc->GetDataType();
  if (gather_param_support_dtype.find(node_input_desc_dtype) == gather_param_support_dtype.end())
  {
    std::vector<DataType> node_input_dtypes = {node_input_desc_dtype};
    GELOGD("elementwise_node's name = %s, elementwise_node's type = %s, node's input dtype is %s, which is not suitable for being gather's input",
    node->GetName().c_str(), node->GetType().c_str(), loop::StrJoin(node_input_dtypes).c_str());
    return false;
  }
  auto node_output_desc = node_op_desc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(node_output_desc);
  auto node_output_desc_dtype = node_output_desc->GetDataType();
  if (node_input_desc_dtype != node_output_desc_dtype) {
    GELOGD("elementwise_node's name = %s, elementwise_node's type = %s, input dtype(%d) != output dtype(%d), stop gather forward fusion",
    node->GetName().c_str(), node->GetType().c_str(), static_cast<int32_t>(node_input_desc_dtype), static_cast<int32_t>(node_output_desc_dtype));
    return false;
  }
  return true;
}

graphStatus GatherForwardFusionPass::Run(const ComputeGraphPtr &graph) const {
  GE_CHECK_NOTNULL(graph);
  GELOGD("Begin gather's forward fusion with elementwise");

  const auto backend_spec = optimize::BackendSpec::GetInstance();
  GE_CHECK_NOTNULL(backend_spec);
  const auto do_gather_elementwise_forward_fusion = backend_spec->gather_spec.enable_gather_elementwise_forward_fusion;
  if (!do_gather_elementwise_forward_fusion || !ge::AutoFuseConfig::LoweringConfig().experimental_lowering_gather) {
    GELOGD("No elementwise node can be forward fused with gather node because the env is not suitable or the switch is off");
    return GRAPH_SUCCESS;
  }
  std::vector<NodePtr> forward_fused_pointwise_nodes;
  for (auto node:graph->GetAllNodes()) {
    if ((node->GetType() != "GatherV2" && node->GetType() != "Gather" && node->GetType() != "GatherV2D") || GetNodeMutableOutputAttr(node) == nullptr)
    {
      continue;
    }
    auto gather_origin_symbolic_shape = GetNodeSymbolShape(node);

    auto node_gather_input_anchor_0 = node->GetInDataAnchor(0); //gather param的in_anchor
    auto node_gather_output_anchor_0 = node->GetOutDataAnchor(0);
    auto node_now = node;
    auto node_pre = node->GetInAllNodes().begin()[0];
    while (NodeForwardFusionJudge(node_pre)) {
      GELOGD("elementwise_node's name = %s, elementwise_node's type = %s",
      node_pre->GetName().c_str(), node_pre->GetType().c_str());
      node_now = node_pre;
      node_pre = node_pre->GetInAllNodes().begin()[0];
      forward_fused_pointwise_nodes.push_back(node_now);
      GE_ASSERT_SUCCESS(SetNodeSymbolShape(node_now, gather_origin_symbolic_shape));
    }
    if (node_now == node) {
      GELOGD("No elementwise node can be forward fuse with gather");
      continue;
    }
    auto node_now_input_anchor = node_now->GetInDataAnchor(0); //起始点的输入anchor
    auto node_now_input_peer_anchor = node_now_input_anchor->GetPeerOutAnchor(); //起始点前一个点的输出anchor
    auto node_gather_input_anchor_0_peer = node_gather_input_anchor_0->GetPeerOutAnchor();
    GELOGD("Now gather node has %zu outputs", node_gather_output_anchor_0->GetPeerInDataAnchors().size());
    for (auto node_gather_out_anchor_peer:node_gather_output_anchor_0->GetPeerInDataAnchors()) {
      GE_ASSERT_GRAPH_SUCCESS(node_gather_output_anchor_0->Unlink(node_gather_out_anchor_peer));
      GE_ASSERT_GRAPH_SUCCESS(node_gather_input_anchor_0_peer->LinkTo(node_gather_out_anchor_peer));
    }
    GE_ASSERT_GRAPH_SUCCESS(node_gather_input_anchor_0_peer->Unlink(node_gather_input_anchor_0));
    GE_ASSERT_GRAPH_SUCCESS(node_now_input_peer_anchor->Unlink(node_now_input_anchor));
    GE_ASSERT_GRAPH_SUCCESS(node_gather_output_anchor_0->LinkTo(node_now_input_anchor));
    GE_ASSERT_GRAPH_SUCCESS(node_now_input_peer_anchor->LinkTo(node_gather_input_anchor_0));
    GELOGD("Graph modification is complete. Gather node %s can be forward fused with the elementwise node before it", node->GetName().c_str());
  }
  GE_ASSERT_GRAPH_SUCCESS(graph->TopologicalSorting());
  std::reverse(forward_fused_pointwise_nodes.begin(), forward_fused_pointwise_nodes.end());
  for(const auto &node:forward_fused_pointwise_nodes) {
    GE_ASSERT_GRAPH_SUCCESS(ShapeRefiner::InferShapeAndType(node));
  }
  GELOGD("Graph changing of gather's forward fusion with elementwise is end");
  return GRAPH_SUCCESS;
}
}
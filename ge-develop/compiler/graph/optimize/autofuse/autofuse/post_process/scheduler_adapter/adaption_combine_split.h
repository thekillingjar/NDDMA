/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_POST_PROCESS_SCHEDULER_ADAPTER_ADAPTION_COMBINE_SPLIT_H
#define AUTOFUSE_POST_PROCESS_SCHEDULER_ADAPTER_ADAPTION_COMBINE_SPLIT_H

#include "ge_common/ge_api_types.h"
#include "graph/compute_graph.h"
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "post_process/post_process_util.h"
#include "adaption_fallback_load.h"

namespace ge {
namespace asc_adapt {
inline Status GetSplitInfos(AscGraph &asc_graph, bool &has_split, std::map<NodePtr, uint32_t> &connect_split_load_nodes) {
  for (const auto &node : asc_graph.GetAllNodes()) {
    if ((node->GetType() != kSplitType)) {
      continue;
    }
    const auto in_anchor = node->GetInDataAnchor(0);
    GE_ASSERT_NOTNULL(in_anchor);
    const auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(peer_out_anchor);
    const auto peer_out_node = peer_out_anchor->GetOwnerNode();
    GE_ASSERT_NOTNULL(peer_out_node);
    GE_ASSERT_TRUE(peer_out_node->GetType() == kLoadType);
    auto it = connect_split_load_nodes.find(peer_out_node);
    if (it != connect_split_load_nodes.end()) {
      it->second++;
    } else {
      connect_split_load_nodes[peer_out_node] = 1U;
    }
  }

  if (!connect_split_load_nodes.empty()) {
    has_split = true;
  }
  return SUCCESS;
}

inline NodePtr CreateSplitNode(AscGraph &asc_graph, const NodePtr &load_node, uint32_t out_num) {
  ge::ascir_op::Split split_op(("Split_Combine_"+ load_node->GetName()).c_str());
  split_op.InstanceOutputy(out_num);
  auto split_node = asc_graph.AddNode(split_op);
  return split_node;
}

inline Status UpdateSplitNodeAttrs(const NodePtr &b_node, const std::vector<int64_t> &current_split_axis,
                                   std::vector<ge::Expression> &current_split_repeats,
                                   std::vector<ge::Expression> &current_split_strides,
                                   AscTensorDataType &current_split_dtype, int32_t output_index) {
  const auto b_opdesc = b_node->GetOpDesc();
  GE_ASSERT_NOTNULL(b_opdesc);
  const auto b_output_tensor_desc = b_opdesc->MutableOutputDesc(output_index);
  GE_ASSERT_NOTNULL(b_output_tensor_desc);
  const auto b_o_attr = b_output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
  GE_ASSERT_NOTNULL(b_o_attr);
  b_o_attr->axis = current_split_axis;
  b_o_attr->repeats = current_split_repeats;
  b_o_attr->strides = current_split_strides;
  b_o_attr->dtype = current_split_dtype;

  GELOGI("split node %s(%s) out tensor axis:%s repeats:%s stride:%s.", b_node->GetName().c_str(),
         b_node->GetType().c_str(), AutofuseUtils::VectorToStr(b_o_attr->axis).c_str(),
         AutofuseUtils::VectorToStr(b_o_attr->repeats).c_str(), AutofuseUtils::VectorToStr(b_o_attr->strides).c_str());
  return SUCCESS;
}

inline Status GetSplitNodeFallbackDtype(const AscGraph &asc_graph, const NodePtr &node, TensorAttrInfo &current_node_attr) {
  (void)asc_graph;
  const auto node_opdesc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(node_opdesc);
  const auto node_attr = node_opdesc->GetAttrsGroup<AscNodeAttr>();
  GE_ASSERT_NOTNULL(node_attr);
  const auto output_tensor_desc = node_opdesc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(output_tensor_desc);
  auto output_attr = output_tensor_desc->GetAttrsGroup<AscTensorAttr>();
  GE_ASSERT_NOTNULL(output_attr);
  current_node_attr.dtype = output_attr->dtype;
  return SUCCESS;
}

inline Status UpdateSplitNodeSchedInfo(const NodePtr &b_node, const std::vector<int64_t> &axis) {
  const auto b_opdesc = b_node->GetOpDesc();
  GE_ASSERT_NOTNULL(b_opdesc);
  const auto b_node_attr = b_opdesc->GetOrCreateAttrsGroup<AscNodeAttr>();
  GE_ASSERT_NOTNULL(b_node_attr);
  b_node_attr->sched.axis = axis;

  GELOGI("set node %s(%s) sched axis is %s.", b_node->GetName().c_str(), b_node->GetType().c_str(),
         AutofuseUtils::VectorToStr(b_node_attr->sched.axis).c_str());
  return SUCCESS;
}

inline Status GetSplitNodeIndex(NodePtr &node, int64_t &index_value) {
  auto op_desc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(op_desc);
  const auto &ir_attr = op_desc->GetAttrsGroup<AscNodeAttr>();
  GE_ASSERT_NOTNULL(ir_attr);
  const auto split_attr = dynamic_cast<ascir_op::Split::AscSplitIrAttrDef *>(ir_attr->ir_attr.get());
  GE_ASSERT_NOTNULL(split_attr);
  split_attr->GetIndex(index_value);
  return SUCCESS;
}

inline Status CheckAscSplitIrAttrValidity(std::vector<NodePtr> nodes_lists) {
  for (auto node : nodes_lists) {
    int64_t index_value = -1L;
    GE_ASSERT_SUCCESS(GetSplitNodeIndex(node, index_value));
    GE_ASSERT_TRUE(index_value != -1L);
    GELOGI("split ascir node %s has split_index: %ld", node->GetName().c_str(), index_value);
  }
  return SUCCESS;
}

inline bool SplitIndexCompare(NodePtr &a, NodePtr &b) {
  ge::GeAttrValue attr_value;
  int64_t index_value_a = -1L;
  GetSplitNodeIndex(a, index_value_a);
  int64_t index_value_b = -1L;
  GetSplitNodeIndex(b, index_value_b);
  return index_value_a < index_value_b;
};

inline Status CreateNewSplitNodeAndModifyEdge(AscGraph &asc_graph, const NodePtr &load_node, uint32_t out_num) {
  int32_t output_index = 0;
  std::vector<NodePtr> new_nodes_lists;
  std::vector<NodePtr> old_nodes_lists;
  std::vector<int32_t> inputs_map;
  std::vector<int32_t> outputs_map;
  const auto b_node = CreateSplitNode(asc_graph, load_node, out_num);
  GE_ASSERT_NOTNULL(b_node);

  const auto load_node_opdesc = load_node->GetOpDesc();
  GE_ASSERT_NOTNULL(load_node_opdesc);
  const auto load_node_attr = load_node_opdesc->GetAttrsGroup<AscNodeAttr>();
  GE_ASSERT_NOTNULL(load_node_attr);
  new_nodes_lists.push_back(b_node);
  auto load_outdata_anchor = load_node->GetAllOutDataAnchorsPtr().at(0);
  for (auto peer_in_data_anchor : load_outdata_anchor->GetPeerInDataAnchorsPtr()) {
    TensorAttrInfo current_node_attr;
    auto const peer_node = peer_in_data_anchor->GetOwnerNode();
    if (peer_node->GetType() == kSplitType) {
      old_nodes_lists.push_back(peer_node);
    }
  }

  GE_ASSERT_SUCCESS(CheckAscSplitIrAttrValidity(old_nodes_lists));

  std::sort(old_nodes_lists.begin(), old_nodes_lists.end(), SplitIndexCompare);

  for (auto peer_split_node : old_nodes_lists) {
    TensorAttrInfo current_node_attr;
    outputs_map.push_back(output_index);
    GetNodeFallbackInfo(asc_graph, peer_split_node, current_node_attr);
    GetSplitNodeFallbackDtype(asc_graph, peer_split_node, current_node_attr);
    GE_ASSERT_SUCCESS(UpdateSplitNodeAttrs(b_node, current_node_attr.axis, current_node_attr.repeats,
                                           current_node_attr.strides, current_node_attr.dtype, output_index));
    GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveEdge(load_node->GetOutDataAnchor(0), peer_split_node->GetInDataAnchor(0)));
    output_index++;
  }
  GE_CHK_STATUS(GraphUtils::ReplaceNodesDataAnchors(new_nodes_lists, old_nodes_lists, inputs_map, outputs_map));
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(load_node->GetOutDataAnchor(0), b_node->GetInDataAnchor(0)));
  GELOGD("del node %s(%u).", load_node->GetName().c_str(), load_node->GetAllOutDataAnchorsSize());
  GE_ASSERT_SUCCESS(UpdateSplitNodeSchedInfo(b_node, load_node_attr->sched.axis));
  auto compute_graph = AscGraphUtils::GetComputeGraph(asc_graph);
  for (auto del_node : old_nodes_lists) {
    if (compute_graph->RemoveNode(del_node) != GRAPH_SUCCESS) {
      GELOGD("del node %s(%s).", del_node->GetName().c_str(), del_node->GetType().c_str());
      return GRAPH_FAILED;
    }
  }
  return SUCCESS;
}

inline Status SplitCombine(AscGraph &asc_graph, [[maybe_unused]] const NodePtr &asc_node) {
  bool has_split = false;
  std::map<NodePtr, uint32_t> connect_split_load_nodes;
  GE_ASSERT_SUCCESS(GetSplitInfos(asc_graph, has_split, connect_split_load_nodes));
  if (connect_split_load_nodes.empty()) {
    GELOGI("graph %s fuse type has no split node.", asc_graph.GetName().c_str());
    return SUCCESS;
  }

  /** 将图中的与load相连的split节点合一**/
  for (auto &load_node_info : connect_split_load_nodes) {
    GELOGI("split combine node %s connect num is %d.", load_node_info.first->GetName().c_str(), load_node_info.second);
    if (load_node_info.second > 1U) {
       GE_ASSERT_SUCCESS(CreateNewSplitNodeAndModifyEdge(asc_graph, load_node_info.first, load_node_info.second));
    }
  }

  // 给ascGraph的节点按照topo id排序，补轴以及后端依赖排序后的节点顺序
  asc_adapt::TopologicalSorting(AscGraphUtils::GetComputeGraph(asc_graph));
  return SUCCESS;
}

inline Status SplitCombineForBackend(const ComputeGraphPtr &ge_or_fused_asc_backend_graph) {
  GE_ASSERT_SUCCESS(ProcessAscBackendNodes(ge_or_fused_asc_backend_graph, SplitCombine, "COMBINE_SPLIT"));
  return SUCCESS;
}
}  // namespace asc_adapt
}  // namespace ge
#endif  // AUTOFUSE_POST_PROCESS_SCHEDULER_ADAPTER_ADAPTION_COMBINE_SPLIT_H

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_POST_PROCESS_SCHEDULER_ADAPTER_ADAPTION_FALLBACK_SCALAR_H
#define AUTOFUSE_POST_PROCESS_SCHEDULER_ADAPTER_ADAPTION_FALLBACK_SCALAR_H

#include "ge_common/ge_api_types.h"
#include "graph/compute_graph.h"
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "post_process/post_process_util.h"
#include "adaption_fallback_load.h"

namespace ge {
namespace asc_adapt {
inline Status FallbackScalar(AscGraph &asc_graph, [[maybe_unused]] const NodePtr &asc_node) {
  for (const auto &node : asc_graph.GetAllNodes()) {
    if (node->GetType() != kScalarType) {
      continue;
    }
    auto out_nodes = node->GetOutNodes();
    // scalar直接链接store不需要修改类型
    if (out_nodes.size() != 1U) {
      GELOGW("scalar(%s) link mul nodes, don't fallback.", node->GetName().c_str());
      continue;
    }
    auto next_node = out_nodes.at(0);
    // 考虑白名单，比如add/mul这些输入是scalar的不加broadcast，其他的默认加，目前没有想到除了store外后端不支持的场景，先只处理store
    if (next_node->GetType() != kStoreType) {
      continue;
    }

    asc_adapt::TensorInfo tensor_info;
    GE_ASSERT_SUCCESS(asc_adapt::GetTensorInfo(next_node, tensor_info));

    tensor_info.broadcast_info = tensor_info.axis;
    const auto peer_node_opdesc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(peer_node_opdesc);
    GE_ASSERT_SUCCESS(asc_adapt::UpdateTopoId(asc_graph, node, tensor_info.broadcast_info.size()));
    tensor_info.current_topo_id = peer_node_opdesc->GetId() + tensor_info.broadcast_info.size();

    auto connect_node = next_node;
    GE_ASSERT_SUCCESS(asc_adapt::CreateAndUpdateBroadcastNodeInfo(asc_graph, node, connect_node, tensor_info));
  }
  // 给ascGraph的节点按照topo id排序，补轴以及后端依赖排序后的节点顺序
  asc_adapt::TopologicalSorting(AscGraphUtils::GetComputeGraph(asc_graph));
  return SUCCESS;
}

// 默认都插broadcast,scheduler会判断并删除多余的broadcast
// scalar后面插的broadcast的轴信息来自graph
inline Status FallbackScalarWithoutCheckType(AscGraph &asc_graph, [[maybe_unused]] const NodePtr &asc_node) {
  for (const auto &node : asc_graph.GetAllNodes()) {
    if (node->GetType() != kScalarType) {
      continue;
    }

    asc_adapt::TensorInfo tensor_info;
    GE_ASSERT_SUCCESS(GetTensorInfoFromAscgraph(tensor_info, asc_graph));
    tensor_info.broadcast_info = tensor_info.axis;
    const auto scalar_node_opdesc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(scalar_node_opdesc);
    GE_ASSERT_SUCCESS(asc_adapt::UpdateTopoId(asc_graph, node, tensor_info.broadcast_info.size()));
    tensor_info.current_topo_id = scalar_node_opdesc->GetId() + tensor_info.broadcast_info.size();
    const auto scalar_output_tensor_desc = scalar_node_opdesc->MutableOutputDesc(0);
    GE_ASSERT_NOTNULL(scalar_output_tensor_desc);
    auto scalar_dtype = scalar_output_tensor_desc->GetDataType();
    const auto &broadcast_info = tensor_info.broadcast_info;
    for (auto index = 0U; index < broadcast_info.size(); index++) {
      const auto b_node = CreateBroadcastNode(asc_graph, node, broadcast_info, index);
      GE_ASSERT_NOTNULL(b_node);
      GE_ASSERT_SUCCESS(asc_adapt::ConnectNodeAfterLoad(node, b_node));
      GE_ASSERT_SUCCESS(asc_adapt::UpdateBroadcastNodeAttrs(b_node, tensor_info.axis, tensor_info.repeats,
                                                            tensor_info.strides, broadcast_info[index]));
      GE_ASSERT_SUCCESS(asc_adapt::UpdateBroadcastNodeSchedInfo(b_node, tensor_info.axis));
      GE_ASSERT_SUCCESS(asc_adapt::UpdateNodeTopoInfo(b_node, tensor_info.current_topo_id));
      GE_ASSERT_SUCCESS(asc_adapt::FromDtypeToOtherDtype(b_node, DT_FLOAT, scalar_dtype));  // 默认类型是DT_FLOAT
      tensor_info.current_topo_id--;
    }
  }
  // 给ascGraph的节点按照topo id排序，补轴以及后端依赖排序后的节点顺序
  asc_adapt::TopologicalSorting(AscGraphUtils::GetComputeGraph(asc_graph));
  return SUCCESS;
}

inline Status FallbackScalarToBroadcast(const ComputeGraphPtr &ge_or_fused_asc_backend_graph) {
  GE_ASSERT_SUCCESS(ProcessAscBackendNodes(ge_or_fused_asc_backend_graph, FallbackScalar, "fallback_scalar"));
  return SUCCESS;
}

// scheduler暂时未完成所有节点类型的scalar后面插入的多余的broadcast删除流程修改，因此后处理暂时使用上面的接口FallbackScalarToBroadcast，此接口待scheduleer完成所有合入后使用
inline Status FallbackScalarToBroadcastWithoutCheckType(const ComputeGraphPtr &ge_or_fused_asc_backend_graph) {
  GE_ASSERT_SUCCESS(
      ProcessAscBackendNodes(ge_or_fused_asc_backend_graph, FallbackScalarWithoutCheckType, "fallback_scalar"));
  return SUCCESS;
}
}  // namespace asc_adapt
}  // namespace ge
#endif  // AUTOFUSE_POST_PROCESS_SCHEDULER_ADAPTER_ADAPTION_FALLBACK_SCALAR_H

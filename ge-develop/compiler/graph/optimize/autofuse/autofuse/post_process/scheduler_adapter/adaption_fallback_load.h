/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_POST_PROCESS_SCHEDULER_ADAPTER_ADAPTION_FALLBACK_LOAD_H
#define AUTOFUSE_POST_PROCESS_SCHEDULER_ADAPTER_ADAPTION_FALLBACK_LOAD_H

#include "ge_common/ge_api_types.h"
#include "graph/compute_graph.h"
#include "post_process/post_process_util.h"
#include "adaption_complete_node_attrs.h"

namespace ge {
namespace asc_adapt {
inline Status UpdateLoadNodeOutput(const NodePtr &data_node, AscTensorAttr *load_output_attr) {
  const auto data_node_opdesc = data_node->GetOpDesc();
  GE_ASSERT_NOTNULL(data_node_opdesc);
  const auto data_output_tensor_desc = data_node_opdesc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(data_output_tensor_desc);
  auto data_output_attr = data_output_tensor_desc->GetAttrsGroup<AscTensorAttr>();
  GE_ASSERT_NOTNULL(data_output_attr);
  load_output_attr->axis = data_output_attr->axis;
  load_output_attr->repeats = data_output_attr->repeats;
  load_output_attr->strides = data_output_attr->strides;
  return SUCCESS;
}

inline Status UpdateDataAndLoadNodeOutput(const NodePtr &data_node, const NodePtr &load_node,
                            const std::vector<std::pair<int64_t, int64_t>> &transpose_info) {
  if (transpose_info.empty()) {
    return SUCCESS;
  }
  const auto data_node_opdesc = data_node->GetOpDesc();
  GE_ASSERT_NOTNULL(data_node_opdesc);
  const auto data_output_tensor_desc = data_node_opdesc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(data_output_tensor_desc);
  auto data_output_attr = data_output_tensor_desc->GetAttrsGroup<AscTensorAttr>();
  GE_ASSERT_NOTNULL(data_output_attr);
  const auto load_node_opdesc = load_node->GetOpDesc();
  GE_ASSERT_NOTNULL(load_node_opdesc);
  const auto load_output_tensor_desc = load_node_opdesc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(load_output_tensor_desc);
  auto load_output_attr = load_output_tensor_desc->GetAttrsGroup<AscTensorAttr>();
  GE_ASSERT_NOTNULL(load_output_attr);
  const auto load_node_attr = load_node_opdesc->GetAttrsGroup<AscNodeAttr>();
  GE_ASSERT_NOTNULL(load_node_attr);
  auto &load_sched_axis = load_node_attr->sched.axis;
  for (auto index = 0U; index < transpose_info.size(); index++) {
    std::swap(data_output_attr->axis[transpose_info[index].first], data_output_attr->axis[transpose_info[index].second]);
    std::swap(data_output_attr->repeats[transpose_info[index].first], data_output_attr->repeats[transpose_info[index].second]);
    std::swap(data_output_attr->strides[transpose_info[index].first], data_output_attr->strides[transpose_info[index].second]);
    std::swap(load_sched_axis[transpose_info[index].first], load_sched_axis[transpose_info[index].second]);
  }
  load_output_attr->axis = data_output_attr->axis;
  load_output_attr->repeats = data_output_attr->repeats;
  load_output_attr->strides = data_output_attr->strides;
  GELOGD("after apply, data attr axis:%s repeats:%s strides:%s.",
         AutofuseUtils::VectorToStr(data_output_attr->axis).c_str(),
         AutofuseUtils::VectorToStr(data_output_attr->repeats).c_str(),
         AutofuseUtils::VectorToStr(data_output_attr->strides).c_str());
  return SUCCESS;
}
inline NodePtr CreateTransposeNode(AscGraph &asc_graph, const NodePtr &load_node, const size_t index) {
  OpDescBuilder transpose_op_desc_builder("Transpose_" + std::to_string(index) + "_" +
                                              load_node->GetName() + "_" +
                                              std::to_string(AutofuseUtils::GenUniqueNumber()),
                                          kTransposeType);
  transpose_op_desc_builder.AddInput("x");
  transpose_op_desc_builder.AddOutput("y");
  auto transpose_op_desc = transpose_op_desc_builder.Build();
  GE_ASSERT_NOTNULL(transpose_op_desc);
  transpose_op_desc->AppendIrInput("x", ge::kIrInputRequired);
  transpose_op_desc->AppendIrOutput("y", ge::kIrOutputRequired);
  auto op = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromOpDesc(transpose_op_desc));
  GE_ASSERT_NOTNULL(op);
  return asc_graph.AddNode(*op);
}

inline NodePtr CreateBroadcastNode(AscGraph &asc_graph, const NodePtr &load_node,
                                   const std::vector<int64_t> &broadcast_info, const size_t index) {
  OpDescBuilder broadcast_op_desc_builder("Broadcast_" + std::to_string(broadcast_info.size() - 1 - index) + "_" +
                                          load_node->GetName() + "_" +
                                          std::to_string(AutofuseUtils::GenUniqueNumber()),
                                          kBroadcastType);
  broadcast_op_desc_builder.AddInput("x");
  broadcast_op_desc_builder.AddOutput("y");
  auto broadcast_op_desc = broadcast_op_desc_builder.Build();
  GE_ASSERT_NOTNULL(broadcast_op_desc);
  broadcast_op_desc->AppendIrInput("x", ge::kIrInputRequired);
  broadcast_op_desc->AppendIrOutput("y", ge::kIrOutputRequired);
  auto op = std::make_shared<Operator>(OpDescUtils::CreateOperatorFromOpDesc(broadcast_op_desc));
  GE_ASSERT_NOTNULL(op);
  return asc_graph.AddNode(*op);
}

inline Status ConnectNodeAfterLoad(const NodePtr &load_node, const NodePtr &new_node) {
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::ReplaceNodeDataAnchors(new_node, load_node, {}, {0}));
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(load_node->GetOutDataAnchor(0), new_node->GetInDataAnchor(0)));
  return SUCCESS;
}

inline Status ConnectNodeBeforeNode(const NodePtr &old_node, const NodePtr &new_node, int32_t input_index) {
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::ReplaceNodeDataAnchors(new_node, old_node, {input_index}, {}));
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(new_node->GetOutDataAnchor(0), old_node->GetInDataAnchor(input_index)));
  return SUCCESS;
}

inline Status ConnectNodeBeforeStore(const NodePtr &store_node, const NodePtr &new_node) {
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::ReplaceNodeDataAnchors(new_node, store_node, {0}, {}));
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::AddEdge(new_node->GetOutDataAnchor(0), store_node->GetInDataAnchor(0)));
  return SUCCESS;
}

inline Status ConnectNodeWithLoadOrStore(NodePtr &node, const NodePtr &new_node) {
  if (node->GetType() != kStoreType) {
    GE_ASSERT_SUCCESS(ConnectNodeAfterLoad(node, new_node));
    node = new_node;
    return SUCCESS;
  }
  // if is store
  return ConnectNodeBeforeStore(node, new_node);
}

inline void UpdateStrides(const ge::Expression repeat_old, std::vector<ge::Expression> &current_broadcast_strides,
                          int64_t axis_index) {
  // 更新高维 stride
  for (int64_t j = axis_index - 1; j >= 0; j--) {
    if (!BackendUtils::IsEqZero(current_broadcast_strides[j])) {
      current_broadcast_strides[j] = current_broadcast_strides[j] / repeat_old;
    }
  }
}

inline Status UpdateBroadcastNodeAttrs(const NodePtr &b_node, const std::vector<int64_t> &current_broadcast_axis,
                                       std::vector<ge::Expression> &current_broadcast_repeats,
                                       std::vector<ge::Expression> &current_broadcast_strides, int64_t axis_id) {
  const auto b_opdesc = b_node->GetOpDesc();
  GE_ASSERT_NOTNULL(b_opdesc);
  const auto b_output_tensor_desc = b_opdesc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(b_output_tensor_desc);
  const auto b_o_attr = b_output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
  GE_ASSERT_NOTNULL(b_o_attr);
  b_o_attr->axis = current_broadcast_axis;
  b_o_attr->repeats = current_broadcast_repeats;
  b_o_attr->strides = current_broadcast_strides;

  // Find the index in current_broadcast_axis that matches axis_id
  auto axis_it = std::find(current_broadcast_axis.begin(), current_broadcast_axis.end(), axis_id);
  GE_ASSERT_TRUE(axis_it != current_broadcast_axis.end());
  size_t axis_index = std::distance(current_broadcast_axis.begin(), axis_it);

  GE_ASSERT_TRUE(axis_index < current_broadcast_repeats.size());
  GE_ASSERT_TRUE(axis_index < current_broadcast_strides.size());
  const auto repeat_old = current_broadcast_repeats[axis_index];
  GE_ASSERT_TRUE(!BackendUtils::IsEqZero(repeat_old));
  current_broadcast_repeats[axis_index] = kSymbolOne;
  current_broadcast_strides[axis_index] = kSymbolZero;

  UpdateStrides(repeat_old, current_broadcast_strides, axis_index);

  GELOGI("add node %s(%s) out tensor axis:%s repeats:%s stride:%s.", b_node->GetName().c_str(),
         b_node->GetType().c_str(), AutofuseUtils::VectorToStr(b_o_attr->axis).c_str(),
         AutofuseUtils::VectorToStr(b_o_attr->repeats).c_str(), AutofuseUtils::VectorToStr(b_o_attr->strides).c_str());
  return SUCCESS;
}

inline Status UpdateBroadcastNodeSchedInfo(const NodePtr &b_node, const std::vector<int64_t> &axis) {
  const auto b_opdesc = b_node->GetOpDesc();
  GE_ASSERT_NOTNULL(b_opdesc);
  const auto b_node_attr = b_opdesc->GetOrCreateAttrsGroup<AscNodeAttr>();
  GE_ASSERT_NOTNULL(b_node_attr);
  b_node_attr->sched.axis = axis;

  GELOGI("set node %s(%s) sched axis is %s.", b_node->GetName().c_str(), b_node->GetType().c_str(),
         AutofuseUtils::VectorToStr(b_node_attr->sched.axis).c_str());
  return SUCCESS;
}

inline Status UpdateNodeTopoInfo(const NodePtr &new_node, int64_t topo_id) {
  auto new_opdesc = new_node->GetOpDesc();
  GE_ASSERT_NOTNULL(new_opdesc);
  new_opdesc->SetId(topo_id);

  GELOGI("set node %s(%s) topo_id is %ld.", new_node->GetName().c_str(), new_node->GetType().c_str(),
         new_opdesc->GetId());
  return SUCCESS;
}

// 如果是插在load后面的transpose,获取transpose前的轴信息就从load获取,如果是插在store前面的transpose,获取transpose前的轴信息就从graph获取
inline Status GetNodeFallbackInfo(AscGraph &asc_graph, const NodePtr &node, TensorAttrInfo &current_node_attr) {
  if (node->GetType() != kStoreType) {
    const auto node_opdesc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(node_opdesc);
    const auto node_attr = node_opdesc->GetAttrsGroup<AscNodeAttr>();
    GE_ASSERT_NOTNULL(node_attr);
    const auto output_tensor_desc = node_opdesc->MutableOutputDesc(0);
    GE_ASSERT_NOTNULL(output_tensor_desc);
    auto output_attr = output_tensor_desc->GetAttrsGroup<AscTensorAttr>();
    GE_ASSERT_NOTNULL(output_attr);
    current_node_attr.sched_axis = node_attr->sched.axis;
    current_node_attr.axis = output_attr->axis;
    current_node_attr.repeats = output_attr->repeats;
    current_node_attr.strides = output_attr->strides;
    return SUCCESS;
  }
  // if is store,从graph获取axis和repeats
  TensorAttrInfo temp_graph_attr;
  GE_ASSERT_SUCCESS(BackendUtils::GetGraphAttrInfo(asc_graph, temp_graph_attr));
  current_node_attr.sched_axis = temp_graph_attr.sched_axis;
  current_node_attr.axis = temp_graph_attr.axis;
  current_node_attr.repeats = temp_graph_attr.repeats;
  current_node_attr.strides = temp_graph_attr.repeats;
  return SUCCESS;
}

inline Status UpdateTopoInfo(const NodePtr &old_node, const NodePtr &new_node, int64_t topo_id) {
  if (old_node->GetType() != kStoreType) {
    return UpdateNodeTopoInfo(new_node, topo_id);
  }
  // if is store
  old_node->GetOpDesc()->SetId(old_node->GetOpDesc()->GetId() + 1);
  return UpdateNodeTopoInfo(new_node, topo_id);
}

// Todo, broadcast插入不用load的属性
inline Status InsertViewOpBroadcast(AscGraph &asc_graph, const NodePtr &data_node, const NodePtr &load_node,
                                    ViewOpAttrInfo &attr_info) {
  auto &broadcast_info = attr_info.broadcast_info;
  if (broadcast_info.empty()) {
    GELOGI("node %s(%s) don't need to add broadcast node in graph %s.", load_node->GetName().c_str(),
           load_node->GetType().c_str(), asc_graph.GetName().c_str());
    return SUCCESS;
  }

  GELOGI("node %s(%s) need to add broadcast node with axis id %s in graph %s.", load_node->GetName().c_str(),
         load_node->GetType().c_str(), AutofuseUtils::VectorToStr(broadcast_info).c_str(), asc_graph.GetName().c_str());

  const auto load_node_opdesc = load_node->GetOpDesc();
  GE_ASSERT_NOTNULL(load_node_opdesc);
  const auto load_node_attr = load_node_opdesc->GetAttrsGroup<AscNodeAttr>();
  GE_ASSERT_NOTNULL(load_node_attr);

  const auto load_output_tensor_desc = load_node_opdesc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(load_output_tensor_desc);
  auto load_output_attr = load_output_tensor_desc->GetAttrsGroup<AscTensorAttr>();
  GE_ASSERT_NOTNULL(load_output_attr);

  const auto current_broadcast_axis = load_output_attr->axis;
  auto current_broadcast_repeats = load_output_attr->repeats;
  auto current_broadcast_strides = load_output_attr->strides;

  TensorAttrInfo current_node_attr;
  if (data_node != nullptr) {
    current_node_attr.axis = current_broadcast_axis;
    current_node_attr.repeats = current_broadcast_repeats;
    current_node_attr.sched_axis = current_broadcast_axis;
    current_node_attr.strides = current_broadcast_strides;
  } else {
    GE_ASSERT_SUCCESS(BackendUtils::GetGraphAttrInfo(asc_graph, current_node_attr));
  }

  GE_ASSERT_SUCCESS(asc_adapt::UpdateTopoId(asc_graph, load_node, broadcast_info.size()));
  auto current_topo_id = load_node_opdesc->GetId() + broadcast_info.size();
  auto load_dtype = load_output_tensor_desc->GetDataType();

  if (data_node != nullptr) {
    GE_ASSERT_SUCCESS(UpdateLoadNodeOutput(data_node, load_output_attr));
  }

  for (auto index = 0U; index < broadcast_info.size(); index++) {
    const auto b_node = CreateBroadcastNode(asc_graph, load_node, broadcast_info, index);
    GE_ASSERT_NOTNULL(b_node);
    GE_ASSERT_SUCCESS(ConnectNodeAfterLoad(load_node, b_node));
    GE_ASSERT_SUCCESS(UpdateBroadcastNodeAttrs(b_node, current_node_attr.axis, current_node_attr.repeats,
                                               current_node_attr.strides, broadcast_info[index]));
    GE_ASSERT_SUCCESS(UpdateBroadcastNodeSchedInfo(b_node, load_node_attr->sched.axis));
    GE_ASSERT_SUCCESS(UpdateNodeTopoInfo(b_node, current_topo_id));
    GE_ASSERT_SUCCESS(asc_adapt::FromDtypeToOtherDtype(b_node, DT_FLOAT, load_dtype));  // 默认类型是DT_FLOAT
    current_topo_id--;
  }

  return SUCCESS;
}

inline Status UpdateTransposeNodeAttrs(const NodePtr &t_node, TensorAttrInfo &current_node_attr,
                                       const std::vector<std::pair<int64_t, int64_t>> &transpose_info) {
  const auto t_opdesc = t_node->GetOpDesc();
  GE_ASSERT_NOTNULL(t_opdesc);
  const auto t_node_attr = t_opdesc->GetAttrsGroup<AscNodeAttr>();
  GE_ASSERT_NOTNULL(t_node_attr);
  const auto t_output_tensor_desc = t_opdesc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(t_output_tensor_desc);
  const auto t_o_attr = t_output_tensor_desc->GetOrCreateAttrsGroup<AscTensorAttr>();
  GE_ASSERT_NOTNULL(t_o_attr);

  auto sched_axis = current_node_attr.sched_axis;
  // 此 sched_axis 是冗余的，ApplySwap接口会更新sched_axis，但是此处不希望更新current_node_attr.sched_axis
  BackendUtils::ApplySwaps(current_node_attr.axis, current_node_attr.repeats, current_node_attr.strides, sched_axis,
                           transpose_info);

  t_node_attr->sched.axis = current_node_attr.sched_axis;
  t_o_attr->axis = current_node_attr.axis;
  t_o_attr->repeats = current_node_attr.repeats;
  t_o_attr->strides = current_node_attr.strides;

  // canfuse需要调用插transpose,需要更新后的连续stride,则调用补轴流程刷新srides
  GE_ASSERT_SUCCESS(asc_adapt::UpdateTensorAttrsIfNotEmpty(t_node, t_o_attr->axis, t_o_attr->repeats, t_o_attr));

  GELOGI("add node %s(%s) sched_axis:%s, out tensor axis:%s repeats:%s stride:%s.", t_node->GetName().c_str(),
         t_node->GetType().c_str(), AutofuseUtils::VectorToStr(t_node_attr->sched.axis).c_str(),
         AutofuseUtils::VectorToStr(t_o_attr->axis).c_str(), AutofuseUtils::VectorToStr(t_o_attr->repeats).c_str(),
         AutofuseUtils::VectorToStr(t_o_attr->strides).c_str());
  return SUCCESS;
}

inline Status InsertViewOpTranspose(AscGraph &asc_graph, const NodePtr &data_node, const NodePtr &node,
                                    ViewOpAttrInfo &attr_info) {
  auto &transpose_info = attr_info.transpose_info;
  if (transpose_info.empty()) {
    GELOGI("node %s(%s) don't need to add transpose node in graph %s.", node->GetName().c_str(),
           node->GetType().c_str(), asc_graph.GetName().c_str());
    return SUCCESS;
  }
  // 根据transpose轴对信息更新data与load节点的信息
  if (data_node != nullptr) {
    GE_ASSERT_SUCCESS(UpdateDataAndLoadNodeOutput(data_node, node, transpose_info));
  }
  GELOGI("node %s(%s) need to add transpose node with axis id %s in graph %s.", node->GetName().c_str(),
         node->GetType().c_str(), AutofuseUtils::VectorPairToStr(transpose_info).c_str(), asc_graph.GetName().c_str());

  const auto node_opdesc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(node_opdesc);

  const auto output_tensor_desc = node_opdesc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(output_tensor_desc);

  TensorAttrInfo current_node_attr;
  GE_ASSERT_SUCCESS(GetNodeFallbackInfo(asc_graph, node, current_node_attr));

  GE_ASSERT_SUCCESS(asc_adapt::UpdateTopoId(asc_graph, node, 1));
  auto current_topo_id = node_opdesc->GetId();
  auto dtype = output_tensor_desc->GetDataType();

  // 插入的transpose在load后面或着store前面，多根轴也只插入一个transpose，后端codegen api支持多根轴transpose
  auto connect_node = static_cast<NodePtr>(node);
  auto index = 0U;
  const auto t_node = CreateTransposeNode(asc_graph, node, index);
  GE_ASSERT_NOTNULL(t_node);
  GE_ASSERT_SUCCESS(ConnectNodeWithLoadOrStore(connect_node, t_node));
  GE_ASSERT_SUCCESS(UpdateTransposeNodeAttrs(t_node, current_node_attr, transpose_info));
  GE_ASSERT_SUCCESS(UpdateTopoInfo(node, t_node, current_topo_id));
  GE_ASSERT_SUCCESS(asc_adapt::FromDtypeToOtherDtype(t_node, DT_FLOAT, dtype)); // 默认类型是DT_FLOAT

  return SUCCESS;
}

// 根据反推结果进行节点插入是按照info元素反向的顺序进行的，即每一次插入节点都插入在load节点后面，预期靠后的节点先插入
inline Status InsertViewOpNodes(AscGraph &asc_graph, const NodePtr &data_node, const NodePtr &load_node,
                                ViewOpAttrInfo &attr_info) {
  GE_ASSERT_SUCCESS(InsertViewOpBroadcast(asc_graph, data_node, load_node, attr_info));
  GE_ASSERT_SUCCESS(InsertViewOpTranspose(asc_graph, data_node, load_node, attr_info));
  return SUCCESS;
}

inline Status GeFallbackPro(AscGraph &asc_graph, [[maybe_unused]] const NodePtr &asc_node) {
  ViewOpAttrInfo attr_info;
  auto autofuse_attr = BackendUtils::GetNodeAutoFuseAttr(asc_node);
  GE_ASSERT_NOTNULL(autofuse_attr);
  GE_ASSERT_SUCCESS(BackendUtils::UpdateSubgraphOutputAttr(AscGraphUtils::GetComputeGraph(asc_graph), asc_node));
  for (const auto &node : asc_graph.GetAllNodes()) {
    if (node->GetType() != kLoadType) {  // 这里没有加gather的判断，当前不支持gather和elewise融合两个图graph轴数不相等
      continue;
    }
    // Load前面为Scalar时，Scalar的tensor轴信息为空，不做处理
    NodePtr peer_out_node;
    GE_ASSERT_SUCCESS(asc_adapt::GetPeerOutNode(node, peer_out_node, 0));
    if (peer_out_node->GetType() == kScalarType) {
      continue;
    }
    const auto in_anchor = node->GetInDataAnchor(0);
    GE_ASSERT_NOTNULL(in_anchor);
    const auto out_anchor = in_anchor->GetPeerOutAnchor();
    GE_ASSERT_NOTNULL(out_anchor);
    const auto data_node = out_anchor->GetOwnerNode();
    GE_ASSERT_NOTNULL(data_node);
    GE_ASSERT_SUCCESS(BackendUtils::BackSteppingViewOp(data_node, node, attr_info));
    auto transpose_info = attr_info.transpose_info;
    if (!transpose_info.empty()) {
      autofuse_attr->SetFuseType(loop::FuseType::kTranspose);
    }
    GE_ASSERT_SUCCESS(InsertViewOpNodes(asc_graph, data_node, node, attr_info));
    // 非Data后面的load节点需要删除，Scalar场景前面已跳过
    if (data_node->GetType() != kDataType) {
      // 非Data后面的load节点如果不需要补轴，则预期在canfuse融合，不在此处删除
      if (attr_info.broadcast_info.empty()) {
        GELOGW("node %s %s, which doesnt't need to add broadcast and is not after data in graph %s, is abnormal.",
               node->GetName().c_str(), node->GetType().c_str(), asc_graph.GetName().c_str());
      }
      GELOGI("node %s %s is not after data node in graph %s, it need to be removed.", node->GetName().c_str(),
             node->GetType().c_str(), asc_graph.GetName().c_str());
      GE_ASSERT_SUCCESS(DelNode(asc_graph, node));
    }
    GE_ASSERT_SUCCESS(attr_info.clear());
  }
  // 给ascGraph的节点按照topo id排序，补轴以及后端依赖排序后的节点顺序
  asc_adapt::TopologicalSorting(AscGraphUtils::GetComputeGraph(asc_graph));
  return SUCCESS;
}

inline Status GeFallback(const ComputeGraphPtr &ge_or_fused_asc_backend_graph) {
  GE_ASSERT_SUCCESS(ProcessAscBackendNodes(ge_or_fused_asc_backend_graph, GeFallbackPro, "ge_fallback"));
  return SUCCESS;
}

inline Status CreateAndUpdateBroadcastNodeInfo(AscGraph &asc_graph, const NodePtr &node,
                                               NodePtr &connect_node, TensorInfo &tensor_info) {
  const std::vector<int64_t> &broadcast_info = tensor_info.broadcast_info;
  GE_ASSERT_TRUE(!broadcast_info.empty());
  for (auto index = 0U; index < broadcast_info.size(); index++) {
    const auto b_node = asc_adapt::CreateBroadcastNode(asc_graph, node, broadcast_info, index);
    GE_ASSERT_NOTNULL(b_node);
    GE_ASSERT_SUCCESS(asc_adapt::ConnectNodeBeforeStore(connect_node, b_node));
    GE_ASSERT_SUCCESS(asc_adapt::UpdateBroadcastNodeAttrs(b_node, tensor_info.axis, tensor_info.repeats,
                                                          tensor_info.strides, broadcast_info[index]));
    GE_ASSERT_SUCCESS(asc_adapt::UpdateBroadcastNodeSchedInfo(b_node, tensor_info.sched_axis));
    GE_ASSERT_SUCCESS(asc_adapt::UpdateNodeTopoInfo(b_node, tensor_info.current_topo_id));
    GE_ASSERT_SUCCESS(asc_adapt::FromDtypeToOtherDtype(b_node, DT_FLOAT, tensor_info.dtype));
    tensor_info.current_topo_id--;
    connect_node = b_node;
  }
  return SUCCESS;
}

inline Status GetTensorInfo(const NodePtr &node, asc_adapt::TensorInfo &tensor_info) {
  const auto store_node_opdesc = node->GetOpDesc();
  GE_ASSERT_NOTNULL(store_node_opdesc);
  const auto store_node_attr = store_node_opdesc->GetAttrsGroup<AscNodeAttr>();
  GE_ASSERT_NOTNULL(store_node_attr);
  tensor_info.sched_axis = store_node_attr->sched.axis;

  const auto store_output_tensor_desc = store_node_opdesc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(store_output_tensor_desc);
  auto store_output_attr = store_output_tensor_desc->GetAttrsGroup<AscTensorAttr>();
  GE_ASSERT_NOTNULL(store_output_attr);

  tensor_info.axis = store_output_attr->axis;
  tensor_info.repeats = store_output_attr->repeats;
  tensor_info.strides = store_output_attr->strides;
  tensor_info.dtype = store_output_tensor_desc->GetDataType();
  return SUCCESS;
}
}  // namespace asc_adapt
}  // namespace ge
#endif  // AUTOFUSE_POST_PROCESS_SCHEDULER_ADAPTER_ADAPTION_FALLBACK_LOAD_H

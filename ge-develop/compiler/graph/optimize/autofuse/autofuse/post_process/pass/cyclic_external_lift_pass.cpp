/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cyclic_external_lift_pass.h"
#include <string>
#include <set>
#include "ge_common/ge_api_types.h"
#include "framework/common/debug/ge_log.h"
#include "common/checker.h"
#include "graph/utils/node_utils.h"
#include "utils/autofuse_utils.h"
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "graph/utils/graph_utils.h"
#include "post_process/post_process_util.h"
#include "post_process/scheduler_adapter/adaption_fallback_load.h"
#include "post_process/scheduler_adapter/adaption_optimized_fallback.h"

namespace ge {
namespace {
Status GetBroadcastAxis(const AscGraph &graph, std::set<int64_t> &broadcast_axis) {
  for (const auto &node : graph.GetAllNodes()) {
    if (node->GetType() != kBroadcastType) {
      continue;
    }
    // 不支持中间Broadcast的场景
    NodePtr peer_node = node;
    NodePtr cur_node = node;
    while ((peer_node->GetType() == kBroadcastType) || (peer_node->GetType() == kCastType)) {
      GE_ASSERT_SUCCESS(asc_adapt::GetPeerOutNode(cur_node, peer_node, 0));
      cur_node = peer_node;
    }
    if ((peer_node->GetType() != kLoadType) && (peer_node->GetType() != kGatherType)) {
      GELOGI("Graph %s not support CyclicExternalLift because broadcast node not at the beginning.",
             graph.GetName().c_str());
      return FAILED;
    }
    // 广播输入节点为UbScalar时，也不做循环外提
    AscTensorAttr *output_tensor_attr;
    GE_ASSERT_SUCCESS(asc_adapt::GetOutputTensorAttr(peer_node, output_tensor_attr));
    if (AutofuseUtils::IsUbScalar(output_tensor_attr->repeats)) {
      GELOGI("Graph %s not support CyclicExternalLift because broadcast node's input is a UB scalar.",
             graph.GetName().c_str());
      return FAILED;
    }
    ViewOpAttrInfo attr_info;
    GE_ASSERT_SUCCESS(BackendUtils::BackSteppingViewOp(peer_node, node, attr_info, true));
    broadcast_axis.insert(attr_info.broadcast_info.begin(), attr_info.broadcast_info.end());
  }
  if (broadcast_axis.empty()) {
    GELOGI("Graph %s no broadcast node.", graph.GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status GetBroadcastAxisIndexAndRepeats(const AscGraph &graph, const std::set<int64_t> &broadcast_axis,
                                       std::set<size_t> &axis_index, std::vector<Expression> &graph_repeats) {
  const auto compute_graph = AscGraphUtils::GetComputeGraph(graph);
  GE_ASSERT_NOTNULL(compute_graph);
  const auto graph_attr = compute_graph->GetAttrsGroup<AscGraphAttr>();
  GE_ASSERT_NOTNULL(graph_attr);
  auto index = 0;
  for (const auto &axis_info : graph_attr->axis) {
    auto it = broadcast_axis.find(axis_info->id);
    if (it != broadcast_axis.end()) {
      axis_index.insert(index);
    }
    graph_repeats.push_back(axis_info->size);
    index++;
  }
  GE_ASSERT_TRUE(broadcast_axis.size() == axis_index.size());
  return SUCCESS;
}

Status GetViewOpSpecialIndex(const AscGraph &graph, const std::vector<Expression> &graph_repeats,
                             std::set<size_t> &special_index) {
  for (const auto &node : graph.GetAllNodes()) {
    if ((node->GetType() == kDataType) || (node->GetType() == kScalarType) || (node->GetType() == kOutputType) ||
        (node->GetType() == kBroadcastType) || (node->GetType() == kLoadType) || (node->GetType() == kCastType) ||
        (node->GetType() == kGatherType)) {
      continue;
    }
    auto asc_node_op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(asc_node_op_desc);
    for (auto &output_desc : asc_node_op_desc->GetAllOutputsDescPtr()) {
      GE_ASSERT_NOTNULL(output_desc);
      auto output_desc_tensor_attr = output_desc->GetAttrsGroup<AscTensorAttr>();
      GE_ASSERT_NOTNULL(output_desc_tensor_attr);
      GE_ASSERT_TRUE(graph_repeats.size() == output_desc_tensor_attr->repeats.size());
      for (size_t i = 0U; i < graph_repeats.size(); ++i) {
        if (graph_repeats[i] != output_desc_tensor_attr->repeats[i]) {
          special_index.insert(i);
        }
      }
    }
  }
  return SUCCESS;
}

bool IsDtypeNotSupport(const AscGraph &graph, DataType &output_dtype) {
  bool is_not_support = false;
  for (const auto &node : graph.GetAllNodes()) {
    if (node->GetType() != kStoreType) {
      continue;
    }
    std::vector<DataType> input_dtypes;
    std::vector<DataType> expect_output_dtypes;
    GeTensorDescPtr output_tensor_desc;
    GE_ASSERT_SUCCESS(asc_adapt::GetOutputTensorDesc(node, output_tensor_desc));
    output_dtype = output_tensor_desc->GetDataType();
    expect_output_dtypes.push_back(output_dtype);
    input_dtypes.push_back(output_dtype);
    is_not_support =
      AutofuseUtils::CallAscirCommonInferDtype(kBroadcastType, input_dtypes, expect_output_dtypes) != SUCCESS;
  }
  return is_not_support;
}

Status DelBroadcastInfo(const AscGraph &graph, const std::set<size_t> &axis_index) {
  // 删除broadcast node
  std::vector<NodePtr> del_nodes;
  for (const auto &node : graph.GetAllNodes()) {
    if (node->GetType() == kBroadcastType) {
      del_nodes.push_back(node);
    }
  }
  for (const auto &node : del_nodes) {
    GE_ASSERT_SUCCESS(asc_adapt::DelNode(graph, node));
  }

  // 刷新所有node的repeat和stride
  for (const auto &node : graph.GetAllNodes()) {
    if ((node->GetType() == kDataType) || (node->GetType() == kScalarType) || (node->GetType() == kOutputType) ||
        (node->GetType() == kStoreType)) {
      continue;
    }
    auto asc_node_op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(asc_node_op_desc);
    for (auto &output_desc : asc_node_op_desc->GetAllOutputsDescPtr()) {
      GE_ASSERT_NOTNULL(output_desc);
      auto output_desc_tensor_attr = output_desc->GetAttrsGroup<AscTensorAttr>();
      GE_ASSERT_NOTNULL(output_desc_tensor_attr);
      GE_ASSERT_TRUE(output_desc_tensor_attr->repeats.size() >= axis_index.size());
      for (size_t i = 0U; i < axis_index.size(); ++i) {
        output_desc_tensor_attr->repeats[i] = kSymbolOne;
        output_desc_tensor_attr->strides[i] = kSymbolZero;
      }
    }
  }
  return SUCCESS;
}

Status InsertBroadcastNode(AscGraph &graph, std::set<int64_t> &broadcast_axis) {
  std::vector<int64_t> broadcast_info(broadcast_axis.begin(), broadcast_axis.end());
  for (const auto &node : graph.GetAllNodes()) {
    if (node->GetType() != kStoreType) {
      continue;
    }
    if (!broadcast_info.empty()) {
      ViewOpAttrInfo attr_info;
      attr_info.broadcast_info = broadcast_info;
      NodePtr peer_node;
      GE_ASSERT_SUCCESS(asc_adapt::GetPeerOutNode(node, peer_node, 0));
      TensorAttrInfo temp_cur_attr;
      GE_ASSERT_SUCCESS(BackendUtils::GetNodeTensorAttrInfo(node, temp_cur_attr));
      TensorAttrInfo temp_peer_out_attr;
      GE_ASSERT_SUCCESS(BackendUtils::GetNodeTensorAttrInfo(peer_node, temp_peer_out_attr));
      std::pair<int32_t, ViewOpAttrInfo> input_index_to_view_info(0, attr_info);
      GE_ASSERT_SUCCESS(asc_adapt::InsertBroadcastBeforeNode(graph, node, temp_cur_attr, temp_peer_out_attr,
                                                             input_index_to_view_info));
    }
  }
  return SUCCESS;
}

Status CheckBroadcastAxisInputRepeat(const AscGraph &graph, const std::set<size_t> &axis_index) {
  std::vector<Expression> base_repeats;
  std::vector<Expression> cur_repeats;
  bool is_first = true;
  for (const auto &node : graph.GetAllNodes()) {
    if ((node->GetType() != kLoadType) && (node->GetType() != kGatherType)) {
      continue;
    }
    NodePtr peer_node;
    GE_ASSERT_SUCCESS(asc_adapt::GetPeerOutNode(node, peer_node, 0));
    if (peer_node->GetType() != kDataType) {
      continue;
    }
    auto op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    const auto output_tensor_desc = op_desc->MutableOutputDesc(0);
    GE_ASSERT_NOTNULL(output_tensor_desc);
    auto output_desc_tensor_attr = output_tensor_desc->GetAttrsGroup<AscTensorAttr>();
    GE_ASSERT_NOTNULL(output_desc_tensor_attr);
    for (const auto axis_idx : axis_index) {
      GE_ASSERT_TRUE(output_desc_tensor_attr->repeats.size() > axis_idx);
      if (is_first) {
        base_repeats.push_back(output_desc_tensor_attr->repeats[axis_idx]);
      }
      cur_repeats.push_back(output_desc_tensor_attr->repeats[axis_idx]);
    }
    if (base_repeats != cur_repeats) {
      return FAILED;
    }
    is_first = false;
    cur_repeats.clear();
  }
  return SUCCESS;
}

// 1. 找到图中所有Broadcast node，只有都是load后的Broadcast才能做外提
// 2. 找出Broadcast轴的合集，只有连续外轴的Broadcast才做外提
// 3. 确认reduce、transpose、slice、split相关轴在Broadcast内轴才支持，如果有其余搬运类先不支持，再补充
// 4. 更新除Broadcast外轴repeat合strides
// 5. 移动Broadcast到store，并且跟新topo id
Status CyclicExternalLift(AscGraph &graph, [[maybe_unused]] const NodePtr &asc_node) {
  std::set<int64_t> broadcast_axis;
  if (GetBroadcastAxis(graph, broadcast_axis) != SUCCESS) {
    GELOGI("Graph %s no need CyclicExternalLift.", graph.GetName().c_str());
    return SUCCESS;
  }

  // 找到轴id对应的index
  std::set<size_t> axis_index;
  std::vector<Expression> graph_repeats;
  GE_ASSERT_SUCCESS(GetBroadcastAxisIndexAndRepeats(graph, broadcast_axis, axis_index, graph_repeats));
  // 判断Broadcast轴是不是外轴连续
  if ((*axis_index.rbegin() + 1U) != axis_index.size()) {
    GELOGI("Graph %s broadcast axis not outer continuous.", graph.GetName().c_str());
    return SUCCESS;
  }

  // 检查Broadcast轴的输入数据repeat是否一致
  if (CheckBroadcastAxisInputRepeat(graph, axis_index) != SUCCESS) {
    GELOGI("Graph %s broadcast axis input data repeat not same.", graph.GetName().c_str());
    return SUCCESS;
  }

  // 找出整图中搬运类算子的轴位置
  std::set<size_t> special_index;
  GE_ASSERT_SUCCESS(GetViewOpSpecialIndex(graph, graph_repeats, special_index));
  std::vector<size_t> vec1(axis_index.begin(), axis_index.end());
  std::vector<size_t> vec2(special_index.begin(), special_index.end());
  GELOGD("Graph %s broadcast axis index(%s), special index(%s).", graph.GetName().c_str(),
         AutofuseUtils::VectorToStr(vec1).c_str(), AutofuseUtils::VectorToStr(vec2).c_str());
  // 搬运类算子轴在Broadcast轴中无法循环外提
  if (!special_index.empty() && (*special_index.begin() <= *axis_index.rbegin())) {
    GELOGI("Graph %s have some view op axis in broadcast axis.", graph.GetName().c_str());
    return SUCCESS;
  }

  DataType output_dtype;
  if (IsDtypeNotSupport(graph, output_dtype)) {
    // Broadcast不支持的dtype不进行外提
    GELOGI("Graph %s can not do cyclic external lift with dtype(%s)", graph.GetName().c_str(),
           TypeUtils::DataTypeToSerialString(output_dtype).c_str());
    return SUCCESS;
  }

  // 删除Broadcast node和调度轴轴信息
  GE_ASSERT_SUCCESS(DelBroadcastInfo(graph, axis_index));

  // 在store前面插入Broadcast
  GE_ASSERT_SUCCESS(InsertBroadcastNode(graph, broadcast_axis));
  asc_adapt::TopologicalSorting(AscGraphUtils::GetComputeGraph(graph));
  return SUCCESS;
}
}  // namespace

Status CyclicExternalLiftPass::Run(const ComputeGraphPtr &graph) const {
  GE_ASSERT_SUCCESS(asc_adapt::ProcessAscBackendNodes(graph, CyclicExternalLift, "cyclic_external_lift_pass"));
  GELOGI("Graph %s completed CyclicExternalLiftPass successfully.", graph->GetName().c_str());
  return SUCCESS;
}
}  // namespace ge

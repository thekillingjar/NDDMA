/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "execution_order.h"
#include "graph/utils/graph_utils.h"
#include "common/checker.h"
#include "api/session/jit_execution/utils/jit_infer_utils.h"
#include "api/session/jit_execution/utils/partitioner/binary_partitioner.h"
#include "common/memory/tensor_trans_utils.h"
#include "graph/utils/tensor_adapter.h"
#include "graph/utils/type_utils.h"

namespace ge {
namespace {
Status SetOutputSizeIfNeed(const ComputeGraphPtr &graph) {
  auto netout_node = graph->GetOrUpdateNetOutputNode();
  GE_ASSERT_NOTNULL(netout_node);
  if (graph->GetOutputSize() != netout_node->GetInDataNodesSize()) {
    // 此处follow节点上信息
    GELOGI("Graph %s output_size[%u] not equal with netoutput[%u] shows in graph, follow netoutput",
           graph->GetName().c_str(), graph->GetOutputSize(), netout_node->GetInDataNodesSize());
    graph->SetOutputSize(netout_node->GetInDataNodesSize());
  }
  return SUCCESS;
}
} // namespace
Status ExecutionOrder::FirstPoint(const std::vector<GeTensor> &inputs, ExecutionPoint *&first_ep) {
  std::lock_guard<std::mutex> locker(mutex_);
  if (!slice_graphs_.empty()) {
    first_ep = slice_graphs_[0].get();
    return SUCCESS;
  }
  GE_ASSERT_SUCCESS(AddNewSlice(user_graph_.compute_graph, inputs, first_ep));
  GE_ASSERT_TRUE(!slice_graphs_.empty());
  first_ep = slice_graphs_[0].get();
  return SUCCESS;
}

ExecutionPoint* ExecutionOrder::GetFirstPoint() {
  std::lock_guard<std::mutex> locker(mutex_);
  if (!slice_graphs_.empty()) {
    return slice_graphs_[0].get();
  }
  return nullptr;
}

Status ExecutionOrder::NextPoint(const ExecutionPoint &ep, const std::vector<GeTensor> &inputs,
                                 ExecutionPoint *&next_ep) {
  std::lock_guard<std::mutex> locker(mutex_);
  GE_ASSERT_TRUE(!slice_graphs_.empty());
  if (HasNext(ep)) {
    GE_ASSERT_TRUE(ep.GetId() >= 0L);
    next_ep = slice_graphs_[static_cast<size_t>(ep.GetId()) + 1].get();
    return SUCCESS;
  }
  if (ep.IsLast()) {
    next_ep = nullptr;
    return SUCCESS;
  }
  GE_ASSERT_SUCCESS(AddNewSlice(ep.GetRemainingGraph(), inputs, next_ep));
  return SUCCESS;
}

Status ExecutionOrder::AddNewSlice(const ComputeGraphPtr &graph, const std::vector<GeTensor> &inputs,
                                   ExecutionPoint *&new_ep) {
  std::vector<NodePtr> infered_nodes;
  GE_ASSERT_SUCCESS(JitInferUtils::InferSymbolForGraph(graph, inputs, infered_nodes));
  PartionResult partition_result;
  if (!infered_nodes.empty()) {
    GE_ASSERT_SUCCESS(BinaryPartitioner::Partition(graph, infered_nodes, partition_result));
  } else {
    partition_result.sliced_graph = graph;
  }
  GE_ASSERT_NOTNULL(partition_result.sliced_graph);
  GE_ASSERT_SUCCESS(SetOutputSizeIfNeed(partition_result.sliced_graph));
  GE_DUMP(partition_result.sliced_graph, "SlicedGraph_"+std::to_string(slice_graphs_.size()));
  // use vector index as ep id
  slice_graphs_.emplace_back(MakeUnique<ExecutionPoint>(slice_graphs_.size(), partition_result.sliced_graph,
                                                        partition_result.remaining_graph));
  GE_ASSERT_NOTNULL(slice_graphs_.back());
  new_ep = slice_graphs_.back().get();
  GE_ASSERT_NOTNULL(new_ep);
  GELOGI("Add new slice graph[%ld] of user graph [%u]", new_ep->GetId(), user_graph_.user_graph_id);
  return SUCCESS;
}

bool ExecutionOrder::HasNext(const ExecutionPoint &ep) const {
  return (ep.GetId() < static_cast<int64_t>(slice_graphs_.size()) - 1);
}

Status ExecutionOrder::ConstructInputTensors(const ComputeGraphPtr &compute_graph) {
  std::vector<GeTensor> inputs_temp;
  is_unknown_input_shape_ = false;
  graph_inputs_.clear();
  for (const auto &node : compute_graph->GetInputNodes()) {
    GE_ASSERT_NOTNULL(node);
    GE_ASSERT_NOTNULL(node->GetOpDesc());
    const auto &tensor_desc = node->GetOpDesc()->GetOutputDesc(0U);
    GE_ASSERT_TRUE(!tensor_desc.IsValid(), "The tensor desc of input node %s:%s is invalid", node->GetName().c_str(),
                         node->GetType().c_str());
    GELOGD("Set input node[%s] tensor desc original shape[%s], shape[%s]", node->GetNamePtr(),
           tensor_desc.GetOriginShape().ToString().c_str(), tensor_desc.GetShape().ToString().c_str());
    if (tensor_desc.GetShape().IsUnknownShape()) {
      is_unknown_input_shape_ = true;
    }
    GeTensor tensor_tmp(tensor_desc);
    inputs_temp.emplace_back(TensorAdapter::NormalizeGeTensor(tensor_tmp));
    GELOGD("Normalize graph %s input %s[%s] -> %s[%s] addr %lu size %lu", compute_graph->GetName().c_str(),
            ge::TypeUtils::FormatToSerialString(tensor_tmp.GetTensorDesc().GetFormat()).c_str(),
            tensor_tmp.GetTensorDesc().GetShape().ToString().c_str(),
            ge::TypeUtils::FormatToSerialString(inputs_temp.back().MutableTensorDesc().GetFormat()).c_str(),
            inputs_temp.back().MutableTensorDesc().GetShape().ToString().c_str(),
            PtrToValue(inputs_temp.back().GetData().GetData()),
            inputs_temp.back().GetData().GetSize());
  }
  GE_ASSERT_SUCCESS(TensorTransUtils::GeTensors2GertTensors(inputs_temp, graph_inputs_));
  GE_ASSERT_SUCCESS(NormalizeOutputs(compute_graph));
  return SUCCESS;
}

Status ExecutionOrder::NormalizeOutputs(const ComputeGraphPtr &compute_graph) const {
  std::vector<GeTensor> outputs_temp;
  for (const auto &output_info : compute_graph->GetGraphOutNodesInfo()) {
    GE_ASSERT_NOTNULL(output_info.first);
    GE_ASSERT_NOTNULL(output_info.first->GetOpDesc());
    const auto &tensor_desc = output_info.first->GetOpDesc()->GetOutputDesc(static_cast<uint32_t>(output_info.second));
    GeTensor output_temp(tensor_desc);
    auto normalized_output = TensorAdapter::NormalizeGeTensor(output_temp);
    GELOGD("Normalize graph %s output %s[%s] -> %s[%s] addr %lu size %lu", compute_graph->GetName().c_str(),
           ge::TypeUtils::FormatToSerialString(output_temp.GetTensorDesc().GetFormat()).c_str(),
           output_temp.GetTensorDesc().GetShape().ToString().c_str(),
           ge::TypeUtils::FormatToSerialString(normalized_output.MutableTensorDesc().GetFormat()).c_str(),
           normalized_output.MutableTensorDesc().GetShape().ToString().c_str(),
           PtrToValue(normalized_output.GetData().GetData()),
           normalized_output.GetData().GetSize());
    GE_ASSERT_SUCCESS(output_info.first->GetOpDesc()->UpdateOutputDesc(static_cast<uint32_t>(output_info.second),
        normalized_output.MutableTensorDesc()),
        "Update output desc size failed for op:%s(%s) index:0 ",
        output_info.first->GetOpDesc()->GetName().c_str(), output_info.first->GetOpDesc()->GetType().c_str());
  }
  return SUCCESS;
}

const std::vector<gert::Tensor> &ExecutionOrder::GetInputTensors(bool &is_unknown_input_shape) {
  if (graph_inputs_.size() == 0) {
    (void)ConstructInputTensors(user_graph_.compute_graph);
  }
  is_unknown_input_shape = is_unknown_input_shape_;
  return graph_inputs_;
}

UserGraph ExecutionOrder::GetUserGraph() const {
  return user_graph_;
}
}  // namespace ge
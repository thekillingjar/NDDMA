/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hybrid/executor/subgraph_context.h"
#include "hybrid/executor/hybrid_model_executor.h"
#include "hybrid/executor/hybrid_model_rt_v1_executor.h"
namespace ge {
namespace hybrid {
SubgraphContext::SubgraphContext(const GraphItem *const graph_item, GraphExecutionContext *const execution_context)
    : graph_item_(graph_item), execution_context_(execution_context) {
}

SubgraphContext::~SubgraphContext() {
}

Status SubgraphContext::Init() {
  GE_CHECK_NOTNULL(graph_item_);
  GELOGD("[%s] Start to init subgraph context. total inputs = %d, total outputs = %d",
         graph_item_->GetName().c_str(),
         graph_item_->TotalInputs(),
         graph_item_->TotalOutputs());
  all_inputs_.resize(static_cast<size_t>(graph_item_->TotalInputs()));
  all_outputs_.resize(static_cast<size_t>(graph_item_->TotalOutputs()));
  GE_CHK_STATUS_RET_NOLOG(InitFrameStates());
  GE_CHK_STATUS_RET_NOLOG(InitNodeStates());
  return SUCCESS;
}

Status SubgraphContext::InitFrameStates() {
  for (const auto node_item : graph_item_->GetAllNodes()) {
    frame_states_[node_item->frame_index_].frame_id_ = node_item->frame_index_;
  }
  for (const auto node_item : graph_item_->GetAllNodes()) {
    if (node_item->frame_index_ != -1) {  // -1 is root frame.
      frame_states_[node_item->frame_index_].parent_frame_ = &frame_states_[node_item->parent_frame_];
    }
  }
  return SUCCESS;
}

Status SubgraphContext::InitNodeStates() {
  for (const auto node_item : graph_item_->GetAllNodes()) {
    auto &node_state = node_states_[node_item];
    node_state.reset(new(std::nothrow)NodeState(*node_item, this, GetFrameState(*node_item)));
    if ((node_state == nullptr) || (node_state->Init() != SUCCESS)) {
      GELOGE(INTERNAL_ERROR, "[Create][NodeState] failed for node:[%s(%s)].", node_item->NodeName().c_str(),
             node_item->NodeType().c_str());
      REPORT_INNER_ERR_MSG("E19999", "Create NodeState failed for node:%s(%s).", node_item->NodeName().c_str(),
                        node_item->NodeType().c_str());
      return INTERNAL_ERROR;
    }
  }
  return SUCCESS;
}

void SubgraphContext::SetGroup(const int32_t group) {
  group_ = group;
}

void SubgraphContext::ResetContext(const NodePtr &node) {
  node_done_manager_.Reset(node);
}

NodeState* SubgraphContext::GetNodeState(const NodeItem *node_item) {
  return node_states_[node_item].get();
}

FrameState&  SubgraphContext::GetFrameState(const NodeItem &node_item) {
  return frame_states_[node_item.frame_index_];
}

Status SubgraphContext::SetInput(const int32_t index, const TensorValue &tensor) {
  if (static_cast<size_t>(index) >= all_inputs_.size()) {
    GELOGE(INTERNAL_ERROR,
           "[Check][Param:index]input index out of range. all input num = %zu, input index = %d",
           all_inputs_.size(), index);
    REPORT_INNER_ERR_MSG("E19999", "input param index out of range, all input num = %zu, input index = %d.",
                       all_inputs_.size(), index);
    return INTERNAL_ERROR;
  }
  all_inputs_[static_cast<size_t>(index)] = tensor;
  return SUCCESS;
}

Status SubgraphContext::SetInput(const NodeItem &node_item, const int32_t input_index, const TensorValue &tensor) {
  const auto index = node_item.input_start + input_index;
  return SetInput(index, tensor);
}

Status SubgraphContext::SetOutput(const NodeItem &node_item, const int32_t output_index, const TensorValue &tensor) {
  const auto index = node_item.output_start + output_index;
  if ((output_index >= node_item.num_outputs) || (static_cast<size_t>(index) >= all_outputs_.size())) {
    GELOGE(INTERNAL_ERROR, "[Check][Param:output_index]output index out of range. all output num = %zu,"
           "node_item = %s, output index = %d.",
           all_outputs_.size(), node_item.DebugString().c_str(), output_index);
    REPORT_INNER_ERR_MSG("E19999", "output index out of range. all output num = %zu, node_item = %s, output index = %d.",
                       all_outputs_.size(), node_item.DebugString().c_str(), output_index);
    return INTERNAL_ERROR;
  }

  all_outputs_[static_cast<size_t>(index)] = tensor;
  return SUCCESS;
}

Status SubgraphContext::GetInput(const int32_t index, TensorValue &tensor) {
  GE_CHECK_GE(all_inputs_.size(), static_cast<size_t>(index) + 1UL);
  tensor = all_inputs_[static_cast<size_t>(index)];
  return SUCCESS;
}

Status SubgraphContext::GetOutputs(std::vector<TensorValue> &outputs) {
  if (graph_item_->IsDynamic()) {
    GELOGD("[%s] graph is dynamic, get outputs from net output input tensors", graph_item_->GetName().c_str());
    // get from net output inputs
    const auto output_node = graph_item_->GetOutputNode();
    if (output_node != nullptr) {
      for (int32_t i = 0; i < output_node->num_inputs; ++i) {
        TensorValue tensor;
        GE_CHK_STATUS_RET_NOLOG(GetInput(output_node->input_start + i, tensor));
        GELOGD("[%s] Adding output tensor by input index [%d], tensor = %s",
               graph_item_->GetName().c_str(),
               output_node->input_start + i,
               tensor.DebugString().c_str());
        outputs.emplace_back(std::move(tensor));
      }
    }
  } else {
    GELOGD("[%s] graph is non-dynamic, get outputs from subgraph outputs", graph_item_->GetName().c_str());
    for (auto &tensor : all_outputs_) {
      GELOGD("[%s] Adding output tensor: %s", graph_item_->GetName().c_str(), tensor.DebugString().c_str());
      outputs.emplace_back(tensor);
    }
  }

  return SUCCESS;
}

Status SubgraphContext::Await(const NodePtr &node) {
  if (node_done_manager_.Await(node)) {
    return SUCCESS;
  }

  if (execution_context_->is_eos_) {
    return END_OF_SEQUENCE;
  }

  return FAILED;
}

void SubgraphContext::OnError(const Status error) {
  if (error != END_OF_SEQUENCE) {
    GELOGE(error, "[Check][Param:error][%s] Error:%X occurred while executing graph.",
           graph_item_->GetName().c_str(), error);
    REPORT_INNER_ERR_MSG("E19999", "[%s] Error:%X occurred while executing graph.",
                       graph_item_->GetName().c_str(), error);
  }
  node_done_manager_.Destroy();
}

void SubgraphContext::NodeDone(const NodePtr &node) {
  node_done_manager_.NodeDone(node);
}

int32_t SubgraphContext::ScheduleGroup(const NodeItem *const node_item) const {
  return (group_ != -1) ? node_item->group : -1;
}

void SubgraphContext::Reset() {
  node_done_manager_.Reset();
  for (auto &input : all_inputs_) {
    input.Destroy();
  }

  for (auto &output : all_outputs_) {
    output.Destroy();
  }

  for (auto &iter : node_states_) {
    iter.second->Reset();
  }
  for (auto &iter : frame_states_) {
    iter.second.Reset();
  }
}
}  // namespace hybrid
}  // namespace ge

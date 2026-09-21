/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hybrid/executor/node_state.h"
#include <chrono>
#include "framework/common/debug/log.h"
#include "graph/compute_graph.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "common/profiling/profiling_manager.h"
#include "hybrid/executor/hybrid_execution_context.h"
#include "hybrid/executor/subgraph_context.h"
#include "hybrid/node_executor/task_context.h"
#include "common/profiling_definitions.h"

namespace {
void IncIterationCount(uint64_t &iteration) {
  ++iteration;
  if (iteration == UINT64_MAX) {
      iteration = 1U;
  }
}
}

namespace ge {
namespace hybrid {
NodeState::NodeState(const NodeItem &node_item, SubgraphContext *const subgraph_context, FrameState &frame_state)
    : subgraph_context_(subgraph_context), node_item_(node_item), frame_state_(frame_state) {
  this->op_desc_ = node_item.node->GetOpDesc();
  ProfilingManager::Instance().RegisterElement(name_profiling_index_, node_item.NodeName());
}

NodeState::~NodeState() {
  WaitForScheduleDone();
}

Status NodeState::Init() {
  auto unique_task_context = TaskContext::Create(this, subgraph_context_);
  GE_CHECK_NOTNULL(unique_task_context);
  task_context_ = std::shared_ptr<TaskContext>(unique_task_context.release());

  // cache operator
  const auto op = OpDescUtils::CreateOperatorFromNode(node_item_.node->shared_from_this());
  GE_CHECK_NOTNULL(subgraph_context_->GetExecutionContext());
  OpDescUtils::SetRuntimeContextToOperator(op, &subgraph_context_->GetExecutionContext()->runtime_context_);
  auto tmp_op = std::unique_ptr<Operator>(new(std::nothrow) Operator(op));
  GE_CHECK_NOTNULL(tmp_op);
  stage_op_map_[subgraph_context_->GetExecutionContext()->stage_id] = std::move(tmp_op);
  return SUCCESS;
}

void NodeState::Reset() {
  WaitForScheduleDone();
  active_count_ = 0U;
  iteration_count_ = 0U;
  ctrl_scheduled_ = 0U;
  data_scheduled_ = 0U;
  merge_index_ = -1; // Use for Execute (Reset after Executed).
  switch_index_ = -1; // Use for Schedule (Reset after Prepared).
  skip_infershape_ = false;
  root_tensor_values_.clear();
  task_context_->Reset();
}

Status NodeState::AwaitDependShapes(const GraphExecutionContext &context) const {
  if ((!node_item_.is_dynamic) || node_item_.IsMergeOp()) {
    return SUCCESS;
  }

  PROFILING_SCOPE_CONST(GetProfilingIndex(), profiling::kInferShapeWaitDependShape);
  for (auto &src_node : node_item_.dependents_for_shape_inference) {
    GELOGI("[%s] Start to wait for data dependent node: %s",
           node_item_.NodeName().c_str(),
           src_node->GetName().c_str());
    RECORD_SHAPE_INFERENCE_EVENT(&context,
                                 node_item_.NodeName().c_str(),
                                 "[AwaitNodeDone] [%s] Start",
                                 src_node->GetName().c_str());
    HYBRID_CHK_STATUS_RET(subgraph_context_->Await(src_node), "[Call][Await] [%s] Await node failed.",
                          src_node->GetName().c_str());
    RECORD_SHAPE_INFERENCE_EVENT(&context,
                                 node_item_.NodeName().c_str(),
                                 "[AwaitNodeDone] [%s] End",
                                 src_node->GetName().c_str());
    GELOGI("[%s] Done waiting node.", src_node->GetName().c_str());
  }
  return SUCCESS;
}

Status NodeState::AwaitInputTensors(const GraphExecutionContext &context) const {
  if (node_item_.IsMergeOp()) {
    GELOGD("[%s] merge index %d, input nodes: %zu", GetName().c_str(), merge_index_, node_item_.data_recv_.size());
    return SUCCESS;
  }

  PROFILING_SCOPE_CONST(GetProfilingIndex(), profiling::kInferShapeWaitInputTensor);
  for (auto &src_node : node_item_.dependents_for_execution) {
    GELOGD("[%s] Start to wait for data dependent node: [%s]",
           node_item_.NodeName().c_str(),
           src_node->GetName().c_str());
    RECORD_EXECUTION_EVENT(&context,
                           node_item_.NodeName().c_str(),
                           "[AwaitNodeDone] [%s] Start",
                           src_node->GetName().c_str());

    HYBRID_CHK_STATUS_RET(subgraph_context_->Await(src_node),
                          "[Call][Await] [%s] Await node [%s] failed.",
                          GetName().c_str(),
                          src_node->GetName().c_str());

    RECORD_EXECUTION_EVENT(&context,
                           node_item_.NodeName().c_str(),
                           "[AwaitNodeDone] [%s] End",
                           src_node->GetName().c_str());
    GELOGD("[%s] Done waiting node: [%s]", node_item_.NodeName().c_str(), src_node->GetName().c_str());
  }
  return SUCCESS;
}

Status NodeState::WaitForPrepareDone() {
  if (prepare_future_.valid()) {
    GELOGD("[%s] Start to wait for prepare future.", GetName().c_str());
    GE_CHK_STATUS_RET(prepare_future_.get(), "[Check][Status][%s] PreRun failed.", GetName().c_str());
  }

  return SUCCESS;
}
Status NodeState::UpdateOutputShapes(const int32_t index, const GeShape &shape, const GeShape &ori_shape) const {
  const auto self_tensor_desc = op_desc_->MutableOutputDesc(static_cast<uint32_t>(index));
  GE_CHECK_NOTNULL(self_tensor_desc);
  self_tensor_desc->SetShape(shape);
  self_tensor_desc->SetOriginShape(ori_shape);
  return SUCCESS;
}

std::shared_ptr<TaskContext> NodeState::GetTaskContext() const {
  return task_context_;
}

Operator *NodeState::GetOperator(const int32_t stage_id) const {
  const auto it = stage_op_map_.find(stage_id);
  if (it != stage_op_map_.end()) {
    return it->second.get();
  }
  return nullptr;
}

void NodeState::SavePersistTensor(const int32_t input_idx, const TensorValue &tensor) {
  const auto is_persist_tensor = [](const std::map<const NodeItem *, std::set<int32_t>> &items, const int32_t idx) {
    // Do not modify the for loop to std::any_of which may increase the time consumption by 100 times
    for (const auto &item : items) {
      if (item.second.find(idx) != item.second.end()) {
        return true;
      }
    }
    return false;
  };

  if (root_tensor_values_.count(input_idx) > 0U) {
    return;
  }

  if (is_persist_tensor(node_item_.root_data_, input_idx)) {
    GELOGD("[%s] Save Root input tensor: %d", GetName().c_str(), input_idx);
    root_tensor_values_[input_idx] = tensor;
  } else if (is_persist_tensor(node_item_.enter_data_, input_idx)) {
    GELOGD("[%s] Save Enter input tensor: %d", GetName().c_str(), input_idx);
    root_tensor_values_[input_idx] = tensor;
  } else {
    // add for misra rule 6-4-2
  }
}

void NodeState::UpdatePersistTensor() const {
  const auto update_tensor = [this](const std::map<const NodeItem *, std::set<int32_t>> &items) {
    for (const auto &item : items) {
      for (const auto idx : item.second) {
        UpdatePersistTensor(idx);
      }
    }
  };

  if (root_tensor_values_.empty()) {
    return;
  }

  update_tensor(node_item_.root_data_);
  if (iteration_count_ > 0U) {
    update_tensor(node_item_.enter_data_);
  }
}

void NodeState::UpdatePersistTensor(const int32_t input_idx) const {
  const auto it = root_tensor_values_.find(input_idx);
  if (it == root_tensor_values_.end()) {
    GELOGW("[%s] Not found saved tensor: %d", GetName().c_str(), input_idx);
    return;
  }

  const auto tensor = task_context_->MutableInput(input_idx);
  if (tensor == nullptr) {
    GELOGW("[%s] Not found input tensor: %d", GetName().c_str(), input_idx);
    return;
  }

  *tensor = it->second;
  GELOGD("[%s] Update input tensor: %d", GetName().c_str(), input_idx);
}

void NodeState::ResetContext(const uint64_t iteration) {
  switch_index_ = -1;
  subgraph_context_->ResetContext(node_item_.node);
  auto unique_task_context = TaskContext::Create(this, subgraph_context_);
  GE_CHECK_NOTNULL_JUST_RETURN(unique_task_context);
  task_context_ = std::shared_ptr<TaskContext>(unique_task_context.release());

  data_scheduled_ = static_cast<uint32_t>(node_item_.root_data_.size());
  ctrl_scheduled_ = static_cast<uint32_t>(node_item_.root_ctrl_.size());
  if (iteration > 0U) {
    data_scheduled_ += static_cast<uint32_t>(node_item_.enter_data_.size());
    ctrl_scheduled_ += static_cast<uint32_t>(node_item_.enter_ctrl_.size());
  }

  iteration_count_ = iteration;
  GELOGD("[%s] in while loop, current iteration: %lu, data scheduled: %u, ctrl scheduled: %u, merge index: %d",
         GetName().c_str(), iteration_count_, data_scheduled_, ctrl_scheduled_, merge_index_);
}

void NodeState::ScheduleContext(const NodeState &node_state) {
  if (node_state.node_item_.IsEnterOp()) {
    GELOGD("[%s]{active: %lu, iteration: %lu}, frame{active: %lu, iteration: %lu} [%s]{active: %lu, iteration: %lu}",
           GetName().c_str(), active_count_, iteration_count_, frame_state_.active_count_,
           frame_state_.iteration_count_, node_state.GetName().c_str(), node_state.frame_state_.active_count_,
           node_state.frame_state_.iteration_count_);
    if (frame_state_.active_count_ != active_count_) {
      ResetContext(0U);
      active_count_ = frame_state_.active_count_;
    }
  } else if (node_state.node_item_.IsExitOp()) {
    GELOGD("[%s]{active: %lu, iteration: %lu} frame{active: %lu, iteration: %lu} "
           "[%s]{active: %lu, iteration: %lu} parent{active: %lu, iteration: %lu}",
           GetName().c_str(), active_count_, iteration_count_, frame_state_.active_count_,
           frame_state_.iteration_count_, node_state.GetName().c_str(), node_state.frame_state_.active_count_,
           node_state.frame_state_.iteration_count_, node_state.frame_state_.parent_frame_->active_count_,
           node_state.frame_state_.parent_frame_->iteration_count_);
    if (node_state.frame_state_.parent_frame_->iteration_count_ != iteration_count_) {
      ResetContext(node_state.frame_state_.parent_frame_->iteration_count_);
    }
  } else if (node_state.iteration_count_ != iteration_count_) {
    ResetContext(node_state.iteration_count_);
  } else {
    // add for misra rule 6-4-2
  }
}

Status NodeState::NodeScheduled(const std::function<void(const NodeItem *)> &ready) const {
  // Schedule data output.
  for (const auto &node : node_item_.data_send_) {
    const auto dst_node_state = subgraph_context_->GetNodeState(node);
    GE_CHECK_NOTNULL(dst_node_state);
    dst_node_state->SetDataSchedule(*this, ready);
  }

  // Schedule ctrl output.
  for (const auto &node : node_item_.ctrl_send_) {
    const auto dst_node_state = subgraph_context_->GetNodeState(node);
    GE_CHECK_NOTNULL(dst_node_state);
    dst_node_state->SetCtrlSchedule(*this, ready);
  }

  // Schedule switch group.
  if ((switch_index_ >= 0) && (static_cast<uint32_t>(switch_index_) < node_item_.switch_groups_.size())) {
    GELOGI("After [%s] scheduled, switch index: %d", GetName().c_str(), switch_index_);
    for (const auto &node : node_item_.switch_groups_[static_cast<size_t>(switch_index_)]) {
      const auto dst_node_state = subgraph_context_->GetNodeState(node);
      GE_CHECK_NOTNULL(dst_node_state);
      dst_node_state->SetCtrlSchedule(*this, ready);
    }
  }

  return SUCCESS;
}

bool NodeState::IsScheduleReady() const {
  GELOGD("[%s] iteration[%lu] data[input: %zu, scheduled: %u], ctrl[input: %zu+%zu, scheduled: %u]",
         GetName().c_str(), iteration_count_, node_item_.data_recv_.size(), data_scheduled_,
         node_item_.ctrl_recv_.size(), node_item_.GetMergeCtrl((iteration_count_ == 0U) ? 0U : 1U), ctrl_scheduled_);
  if (node_item_.IsMergeOp()) {
    if (ctrl_scheduled_ != (node_item_.GetMergeCtrl((iteration_count_ == 0U) ? 0U : 1U) +
                            node_item_.ctrl_recv_.size())) {
      return false;
    }

    return data_scheduled_ > 0U;
  }

  if (ctrl_scheduled_ != node_item_.ctrl_recv_.size()) {
    return false;
  }

  // Exit may feed loop times...
  return data_scheduled_ >= node_item_.data_recv_.size();
}

void NodeState::SetDataSchedule(const NodeState &node_state, const std::function<void(const NodeItem *)> &ready) {
  GELOGD("[%s] schedule [%s], iteration[%lu -> %lu], data[num: %zu, scheduled: %u], ctrl[num: %zu+%zu, scheduled: %u]",
         node_state.GetName().c_str(), GetName().c_str(), iteration_count_, node_state.iteration_count_,
         node_item_.data_recv_.size(), data_scheduled_, node_item_.ctrl_recv_.size(),
         node_item_.GetMergeCtrl((iteration_count_ == 0U) ? 0U : 1U), ctrl_scheduled_);

  const std::lock_guard<std::mutex> lk(mu_);
  ScheduleContext(node_state);
  ++data_scheduled_;

  if (node_item_.IsMergeOp()) {
    const auto it = node_item_.data_recv_.find(&node_state.node_item_);
    if (it != node_item_.data_recv_.end()) {
      merge_index_ = it->second;
      (void)AttrUtils::SetInt(op_desc_, ATTR_NAME_MERGE_INPUT_INDEX, it->second);
      GELOGD("[%s] scheduled, [%s] set merge index: %d", node_state.GetName().c_str(), GetName().c_str(), it->second);
    } else {
      GELOGW("[%s] scheduled, [%s] not followed", node_state.GetName().c_str(), GetName().c_str());
    }
  }

  if (IsScheduleReady()) {
    data_scheduled_ = static_cast<uint32_t>(node_item_.root_data_.size()) +
                      static_cast<uint32_t>(node_item_.enter_data_.size());
    ctrl_scheduled_ = static_cast<uint32_t>(node_item_.root_ctrl_.size()) +
                      static_cast<uint32_t>(node_item_.enter_ctrl_.size());
    ready(&node_item_);
  }
}

void NodeState::SetCtrlSchedule(const NodeState &node_state, const std::function<void(const NodeItem *)> &ready) {
  GELOGD("[%s] schedule [%s], iteration[%lu -> %lu], data[num: %zu, scheduled: %u], ctrl[num: %zu+%zu, scheduled: %u]",
         node_state.GetName().c_str(), GetName().c_str(), iteration_count_, node_state.iteration_count_,
         node_item_.data_recv_.size(), data_scheduled_, node_item_.ctrl_recv_.size(),
         node_item_.GetMergeCtrl((iteration_count_ == 0U) ? 0U : 1U), ctrl_scheduled_);

  const std::lock_guard<std::mutex> lk(mu_);
  ScheduleContext(node_state);
  ++ctrl_scheduled_;

  if (IsScheduleReady()) {
    data_scheduled_ = static_cast<uint32_t>(node_item_.root_data_.size()) +
                      static_cast<uint32_t>(node_item_.enter_data_.size());
    ctrl_scheduled_ = static_cast<uint32_t>(node_item_.root_ctrl_.size()) +
                      static_cast<uint32_t>(node_item_.enter_ctrl_.size());
    ready(&node_item_);
  }
}

void NodeState::RunNextIteration() {
  const std::lock_guard<std::mutex> lk(mu_);
  IncIterationCount(iteration_count_);
  ResetContext(iteration_count_);
}

void NodeState::RunStreamActive() {
  const std::lock_guard<std::mutex> lk(mu_);
  if (node_item_.ctrl_send_.empty()) {   // Not for Loop Enter or Loop Next.
    return;
  }
  switch_index_ = 0;
  data_scheduled_ = 0U;
  ctrl_scheduled_ = 0U;
  if (node_item_.is_enter_active_) {
    frame_state_.iteration_count_ = 0U;
    IncIterationCount(frame_state_.active_count_);
  } else {
    IncIterationCount(frame_state_.iteration_count_);
  }
  GELOGD("Node[%s] current iteration: %lu, frame active: %lu, frame iteration: %lu",
         GetName().c_str(), iteration_count_, frame_state_.active_count_, frame_state_.iteration_count_);
}

void NodeState::SetScheduleFuture(std::future<Status> &&future) {
  const std::lock_guard<std::mutex> lk(mu_);
  if (schedule_future_.valid()) {
    GELOGW("[%s] Wait for last schedule future.", GetName().c_str());
    GE_CHK_STATUS(schedule_future_.get(), "[Check][Status][%s] wait thread failed.", GetName().c_str());
  }
  schedule_future_ = std::move(future);
}

void NodeState::WaitForScheduleDone() {
  const std::lock_guard<std::mutex> lk(mu_);
  if (schedule_future_.valid()) {
    GELOGD("[%s] Start to wait for schedule future.", GetName().c_str());
    GE_CHK_STATUS(schedule_future_.get(), "[Check][Status][%s] wait thread failed", GetName().c_str());
  }
}
}  // namespace hybrid
}  // namespace ge

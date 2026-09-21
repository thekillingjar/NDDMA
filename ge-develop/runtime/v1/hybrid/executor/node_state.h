/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_HYBRID_EXECUTOR_NODE_STATE_H_
#define GE_HYBRID_EXECUTOR_NODE_STATE_H_

#include <condition_variable>
#include <future>

#include "common/blocking_queue.h"
#include "hybrid/model/node_item.h"
#include "hybrid/executor/node_done_manager.h"

namespace ge {
namespace hybrid {
class NodeTask;
class TaskContext;
struct NodeState;

using NodeStatePtr = std::shared_ptr<NodeState>;

struct FrameState {
 public:
  FrameState() = default;
  ~FrameState() = default;
  explicit inline FrameState(const int64_t id) : frame_id_(id) {}
  inline void Reset() {
    active_count_ = 0U;
    iteration_count_ = 0U;
  }

  int64_t frame_id_{0};
  uint64_t active_count_{0U};
  uint64_t iteration_count_{0U};
  FrameState* parent_frame_{nullptr};
};

// saving sth. dynamic during execution
struct NodeState  {
public:
  NodeState(const NodeItem &node_item, SubgraphContext *const subgraph_context, FrameState &frame_state);
  ~NodeState();

  Status Init();

  void Reset();

  inline const NodeItem &GetNodeItem() const {
    return node_item_;
  }

  inline const std::string &GetName() const {
    return node_item_.NodeName();
  }

  inline const std::string &GetType() const {
    return node_item_.NodeType();
  }

  inline int64_t GetProfilingIndex() const {
    return name_profiling_index_;
  }

  Status UpdateOutputShapes(const int32_t index, const GeShape &shape, const GeShape &ori_shape) const;

  inline bool IsShapeDependence() const {
    return node_item_.IsControlFlowOp() || (node_item_.shape_inference_type >= DEPEND_SHAPE_RANGE);
  }

  void RunStreamActive();
  void RunNextIteration();

  void SavePersistTensor(const int32_t input_idx, const TensorValue &tensor);
  void UpdatePersistTensor() const;

  Status NodeScheduled(const std::function<void(const NodeItem *)> &ready) const;

  void SetScheduleFuture(std::future<Status> &&future);
  void WaitForScheduleDone();

  void SetSwitchIndex(const int32_t index) {
    switch_index_ = index;
  }

  int32_t GetSwitchIndex() const {
    return switch_index_;
  }

  void SetMergeIndex(const int32_t index) {
    merge_index_ = index;
  }

  int32_t GetMergeIndex() const {
    return merge_index_;
  }

  const shared_ptr<NodeTask> &GetKernelTask() const {
    return kernel_task_;
  }

  void SetKernelTask(const shared_ptr<NodeTask> &kernel_task) {
    kernel_task_ = kernel_task;
  }

  Status WaitForPrepareDone();

  void SetPrepareFuture(std::future<Status> &&prepare_future) {
    this->prepare_future_ = std::move(prepare_future);
  }

  Status AwaitDependShapes(const GraphExecutionContext &context) const;
  Status AwaitInputTensors(const GraphExecutionContext &context) const;

  std::shared_ptr<TaskContext> GetTaskContext() const;

  void SetSkipInferShape(const bool skip_infershape) { skip_infershape_ = skip_infershape; }

  bool MaySkipShapeInference() const { return skip_infershape_; }

  void SetSkipSchedule(const bool skip_schedule) { skip_schedule_ = skip_schedule; }

  bool MaySkipSchedule() const { return skip_schedule_; }

  void SetUserAllocated(const bool user_allocated) { user_allocated_ = user_allocated; }

  bool IsUserAllocated() const { return user_allocated_; }

  Operator *GetOperator(const int32_t stage_id) const;

  uint32_t GetDataScheduledNum() const { return data_scheduled_; }

  uint32_t GetCtrlScheduledNum() const { return ctrl_scheduled_; }

  uint64_t GetIterationCount() const { return iteration_count_; }

 private:
  bool IsScheduleReady() const;
  void SetDataSchedule(const NodeState &node_state, const std::function<void(const NodeItem *)> &ready);
  void SetCtrlSchedule(const NodeState &node_state, const std::function<void(const NodeItem *)> &ready);
  void ResetContext(const uint64_t iteration);
  void ScheduleContext(const NodeState &node_state);
  void UpdatePersistTensor(const int32_t input_idx) const;

  std::shared_ptr<NodeTask> kernel_task_ = nullptr;
  std::future<Status> prepare_future_;
  OpDescPtr op_desc_;
  std::map<int32_t, std::unique_ptr<Operator>> stage_op_map_;
  SubgraphContext *subgraph_context_;
  const NodeItem &node_item_;
  std::shared_ptr<TaskContext> task_context_ = nullptr;
  std::mutex mu_;

  std::future<Status> schedule_future_;
  std::map<int32_t, TensorValue> root_tensor_values_;
  FrameState &frame_state_;
  uint64_t active_count_ = 0U;
  uint64_t iteration_count_ = 0U;
  uint32_t ctrl_scheduled_ = 0U;
  uint32_t data_scheduled_ = 0U;
  int32_t merge_index_ = -1; // Use for Execute (Reset after Executed).
  int32_t switch_index_ = -1; // Use for Schedule (Reset after Prepared).
  int64_t name_profiling_index_ = -1;  // for performance consideration, profiling use a pre-registered index to name
  bool skip_infershape_ = false;
  bool skip_schedule_ = false;
  bool user_allocated_ = false;
};
}  // namespace hybrid
}  // namespace ge

#endif // GE_HYBRID_EXECUTOR_NODE_STATE_H_

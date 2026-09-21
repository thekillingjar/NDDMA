/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_HYBRID_EXECUTOR_SUBGRAPH_CONTEXT_H_
#define GE_HYBRID_EXECUTOR_SUBGRAPH_CONTEXT_H_

#include <vector>
#include "mmpa/mmpa_api.h"
#include "hybrid/common/tensor_value.h"
#include "hybrid/executor/hybrid_execution_context.h"
#include "hybrid/executor/node_state.h"
#include "hybrid/executor/node_done_manager.h"
#include "hybrid/model/graph_item.h"
#include "hybrid/model/node_item.h"

namespace ge {
namespace hybrid {
class SubgraphContext {
 public:
  SubgraphContext(const GraphItem *const graph_item, GraphExecutionContext *const execution_context);
  ~SubgraphContext();

  Status Init(); // should not be declared as virtual
  void SetGroup(const int32_t group);
  void ResetContext(const NodePtr &node);
  void Reset(); // should not be declared as virtual
  NodeState* GetNodeState(const NodeItem *node_item);

  void OnError(const Status error);

  Status SetInput(const NodeItem &node_item, const int32_t input_index, const TensorValue &tensor);
  Status SetOutput(const NodeItem &node_item, const int32_t output_index, const TensorValue &tensor);
  Status SetInput(const int32_t index, const TensorValue &tensor);
  Status GetInput(const int32_t index, TensorValue &tensor);
  Status GetOutputs(std::vector<TensorValue> &outputs); // should not be declared as virtual

  Status Await(const NodePtr &node);
  void NodeDone(const NodePtr &node);
  int32_t ScheduleGroup(const NodeItem *const node_item) const;
  GraphExecutionContext *GetExecutionContext() const {return execution_context_;}

 private:
  Status InitFrameStates();
  Status InitNodeStates();
  FrameState& GetFrameState(const NodeItem &node_item);  // no lock
  friend class TaskContext;
  const GraphItem *graph_item_;
  GraphExecutionContext *execution_context_;
  std::vector<TensorValue> all_inputs_;
  std::vector<TensorValue> all_outputs_;
  NodeDoneManager node_done_manager_;
  std::unordered_map<const NodeItem *, NodeStatePtr> node_states_;
  std::unordered_map<int64_t, FrameState> frame_states_;
  int32_t group_ = -1;
};
}  // namespace hybrid
}  // namespace ge

#endif // GE_HYBRID_EXECUTOR_ITERATION_CONTEXT_H_

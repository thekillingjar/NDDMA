/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXECUTION_ORDER_H
#define EXECUTION_ORDER_H
#include <vector>
#include <mutex>
#include "graph/compute_graph.h"
#include "execution_point.h"
#include "exe_graph/runtime/tensor.h"

namespace ge {
struct UserGraph {
  uint32_t user_graph_id;
  ComputeGraphPtr compute_graph;
};
/**
 * 该类用于确定一张UserGraph中的切图结果
 * 切出来的图叫做sliced_graph，抽象在执行过程中叫做execution point
 */
class ExecutionOrder {
 public:
  ExecutionOrder() = delete;
  explicit ExecutionOrder(const UserGraph &user_graph) : user_graph_(user_graph), is_unknown_input_shape_(false) {}

  // Retrieves the first slice graph to execute.
  // Returns nullptr when failed.
  // Thread-safe implementation, lock-free when first point exists.
  Status FirstPoint(const std::vector<GeTensor> &inputs, ExecutionPoint *&first_ep);

  // Retrieves the next slice graph to execute.
  // Returns nullptr if all slice graphs have been executed.
  // If the next slice graph has not been generated, ExecutionOrder will slice a new one from ep.remaining_graph_.
  // Thread-safe implementation, lock-free when next point exists.
  Status NextPoint(const ExecutionPoint &ep, const std::vector<GeTensor> &inputs, ExecutionPoint *&next_ep);

  ExecutionPoint* GetFirstPoint();
  const std::vector<gert::Tensor> &GetInputTensors(bool &is_unknown_input_shape);
  UserGraph GetUserGraph() const;
 private:
  bool HasNext(const ExecutionPoint &ep) const;
  Status AddNewSlice(const ComputeGraphPtr &graph, const std::vector<GeTensor> &inputs, ExecutionPoint *&new_ep);
  Status ConstructInputTensors(const ComputeGraphPtr &compute_graph);
  Status NormalizeOutputs(const ComputeGraphPtr &compute_graph) const;
  UserGraph user_graph_;
  // 临时实现
  // 若不同输入shape产生不同的切图结果，需要切换为树实现
  std::mutex mutex_;
  std::vector<std::unique_ptr<ExecutionPoint>> slice_graphs_;
  // todo add io relation between slicing graph later
  std::vector<gert::Tensor> graph_inputs_;
  bool is_unknown_input_shape_;
  friend class ExecutionOrderUtil;
};
}  // namespace ge

#endif  // EXECUTION_ORDER_H

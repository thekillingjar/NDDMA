/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_HYBRID_KERNEL_GE_LOCAL_NODE_EXECUTOR_H_
#define GE_HYBRID_KERNEL_GE_LOCAL_NODE_EXECUTOR_H_

#include <unordered_map>
#include <vector>
#include "hybrid/node_executor/node_executor.h"
#include "hybrid/executor/hybrid_data_flow.h"

namespace ge {
namespace hybrid {
class RefInputTask : public NodeTask {
 public:
  explicit RefInputTask(const NodePtr &node)
      : NodeTask(), node_name_(node->GetName()),
        node_type_(node->GetType()) {
  }

  ~RefInputTask() override = default;

  Status UpdateArgs(TaskContext &context) override;
  Status ExecuteAsync(TaskContext &context, const std::function<void()> &done_callback) override;
  static bool IsBelong(const std::string &op_type);
 private:
  Status Execute(const TaskContext &context) const;
  Status RefOneByOne(const TaskContext &context) const;
  Status RefByOrder(const std::vector<uint32_t> &ref_order, const TaskContext &context) const;

 private:
  const std::string node_name_;
  const std::string node_type_;

  // key is op type, value is output ref input index,
  // e.g. {1,0} means out[0] ref input[1], out[1] ref input[0], if vector is empty, it means ref input one by one
  static const std::map<std::string, std::vector<uint32_t>> out_ref_input_index_;
};

class DependInputShapeTask : public NodeTask {
 public:
  explicit DependInputShapeTask(const NodePtr &node) : NodeTask(), node_(node) {
  }

  ~DependInputShapeTask() override = default;

  Status UpdateArgs(TaskContext &context) override;
  Status ExecuteAsync(TaskContext &context, const std::function<void()> &done_callback) override;
  static bool IsBelong(const std::string &op_type);
 private:
  Status Execute(const TaskContext &context) const;
  Status CopyDataToOutput(const size_t output_num,
                          std::vector<GeTensorPtr> &outputs,
                          const std::string &node_type,
                          const TaskContext &context) const;
  const NodePtr node_;

  // ops depend input shape
  static const std::set<std::string> depend_input_shape_ops_;
};

class ConstantNodeTask : public NodeTask {
 public:
  explicit ConstantNodeTask(const TensorValue * const tensor_in);
  ~ConstantNodeTask() override = default;
  Status UpdateArgs(TaskContext &context) override;

  Status ExecuteAsync(TaskContext &context, const std::function<void()> &done_callback) override;
  static bool IsBelong(const std::string &op_type);

 private:
  static const std::set<std::string> constant_like_task_ops_;
  const TensorValue *tensor_;
};

class NoOpNodeTask : public NodeTask {
 public:
  explicit NoOpNodeTask() = default;
  ~NoOpNodeTask() override = default;
  Status UpdateArgs(TaskContext &context) override;
  Status ExecuteAsync(TaskContext &context, const std::function<void()> &done_callback) override;
  static bool IsBelong(const std::string &op_type);

 private:
  static const std::set<std::string> control_only_task_ops_;
};

class DataFlowNodeTask : public NodeTask {
 public:
  explicit DataFlowNodeTask(const std::string &type) : NodeTask(), node_type_(type) {}
  ~DataFlowNodeTask() override = default;

  Status InitTaskBasicInfo(const NodePtr &node) override;
  Status UpdateArgs(TaskContext &context) override;
  Status ExecuteAsync(TaskContext &context, const std::function<void()> &done_callback) override;
  static bool IsBelong(const std::string &op_type);
 private:
  Status Execute(TaskContext &context) const;
 private:
  static const std::set<std::string> data_flow_ops_;
  std::string node_type_;
  int64_t handle_ = -1;
};

class GeLocalNodeExecutor : public NodeExecutor {
 public:
  GeLocalNodeExecutor() noexcept: NodeExecutor() {}
  ~GeLocalNodeExecutor() override = default;

  Status PrepareTask(NodeTask &task, TaskContext &context) const override;

  Status LoadTask(const HybridModel &model,
                  const NodePtr &node,
                  std::shared_ptr<NodeTask> &task) const override;
};
}  // namespace hybrid
}  // namespace ge
#endif // GE_HYBRID_KERNEL_GE_LOCAL_NODE_EXECUTOR_H_

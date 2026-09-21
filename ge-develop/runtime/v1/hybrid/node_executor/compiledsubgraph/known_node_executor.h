/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HYBRID_KNOWN_NODE_EXECUTOR_H_
#define HYBRID_KNOWN_NODE_EXECUTOR_H_
#include "hybrid/node_executor/node_executor.h"
#include "hybrid/model/hybrid_model.h"
#include "graph/op_desc.h"
#include "graph/load/model_manager/davinci_model.h"

namespace ge {
namespace hybrid {

class KnownNodeTask : public NodeTask {
 public:
  explicit KnownNodeTask(std::shared_ptr<DavinciModel> davinci_model)
      : NodeTask(), davinci_model_(davinci_model)
    {}

  ~KnownNodeTask() override = default;

  Status UpdateArgs(TaskContext &context) override;
  Status ExecuteAsync(TaskContext &context, const std::function<void()> &done_callback) override;
  Status Init(TaskContext &context) override;
  Status InitDavinciModel(const HybridModel &model, const TensorBuffer *const weight_buffer);
  Status ReportProfilingData() override;
 protected:
  virtual Status DoInitDavinciModel(const uintptr_t weight, const size_t weight_size);
 private:
  std::shared_ptr<DavinciModel> davinci_model_ = nullptr;
  bool load_flag_ = false;
};

class KnownNodeExecutor : public NodeExecutor {
 public:
  Status LoadTask(const HybridModel &model, const NodePtr &node, shared_ptr<NodeTask> &task) const override;
  Status PrepareTask(NodeTask &task, TaskContext &context) const override;
  Status ExecuteTask(NodeTask &task, TaskContext &context, const std::function<void()> &callback) const override;
  Status ReportProfilingData(const NodeItem &node_item) const override;
  ~KnownNodeExecutor() override {}

 private:
  static Status ParseAttrForAllocatingOutputs(NodeItem &node_item, const ComputeGraph &graph);
  static Status GetDataNodes(const ComputeGraph &graph, std::map<NodePtr, int32_t> &data_indices);
  static Status GetModelAndGraph(const HybridModel &model,
                                 const NodePtr &node,
                                 GeModelPtr &ge_model,
                                 ComputeGraphPtr &graph);
  Status SetDaviciModel(const HybridModel &model, const NodePtr &node,
                        std::shared_ptr<DavinciModel> &davinci_model) const;
};
}  // namespace hybrid
}  // namespace ge

#endif  // HYBRID_KNOWN_NODE_EXECUTOR_H_

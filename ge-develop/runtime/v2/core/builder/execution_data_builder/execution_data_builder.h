/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_CORE_BUILDER_EXECUTION_DATA_BUILDER_EXECUTION_DATA_BUILDER_H_
#define AIR_CXX_RUNTIME_V2_CORE_BUILDER_EXECUTION_DATA_BUILDER_EXECUTION_DATA_BUILDER_H_
#include "core/builder/graph_executor_builder.h"
namespace gert {
class ExecutionDataBuilder {
 public:
  explicit ExecutionDataBuilder(GraphExecutorBuilder &executor_builder) : executor_builder_(executor_builder) {}
  virtual ~ExecutionDataBuilder() = default;

  virtual ResourceGuardPtr Build() = 0;

  virtual std::pair<ExeGraphExecutor::ExecuteFunc, ExeGraphExecutor::ExecuteWithCallbackFunc> GetExecuteFunc()
      const = 0;

  static bool HasOutputsNeedCreate(const ge::FastNode &node);

 protected:
  GraphExecutorBuilder &GetExecutorBuilder() {
    return executor_builder_;
  }
  const GraphExecutorBuilder &GetExecutorBuilder() const {
    return executor_builder_;
  }
  ge::graphStatus CreateKernelOutputs(vector<std::pair<ge::FastNode *, Node *>> &graph_nodes_to_executor_node) const;

 private:
  GraphExecutorBuilder &executor_builder_;
};
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_CORE_BUILDER_EXECUTION_DATA_BUILDER_EXECUTION_DATA_BUILDER_H_

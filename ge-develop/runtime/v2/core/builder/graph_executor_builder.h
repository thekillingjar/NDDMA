/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_CORE_BUILDER_GRAPH_EXECUTOR_BUILDER_H_
#define AIR_CXX_RUNTIME_V2_CORE_BUILDER_GRAPH_EXECUTOR_BUILDER_H_
#include "graph/fast_graph/execute_graph.h"
#include "exe_graph/runtime/continuous_buffer.h"
#include "runtime/exe_graph_executor.h"
#include "graph_async_value.h"
#include "framework/runtime/executor_option/executor_option.h"
namespace gert {
using SymbolsToValue = std::unordered_map<uint64_t, AsyncAnyValue *>;
struct ModelLevelData {
  const ContinuousBuffer *compute_nodes_info;
  const ContinuousBuffer *kernel_extend_info;
};
class GraphExecutorBuilder {
 public:
  GraphExecutorBuilder(const ModelLevelData &model_level_data, ge::ExecuteGraphPtr exe_graph,
                       SymbolsToValue *symbols_to_value);
  ge::graphStatus Build(ExeGraphExecutor &executor);
  ge::graphStatus Build(ExecutorType executor_type, ExeGraphExecutor &executor);

  GraphExecutorBuilder &ExecutorOpt(const ExecutorOption &option);

  ExecutorOption *GetExecutorOption() {
      return executor_option_;
  }

  ge::ExecuteGraphPtr GetExeGraph() {
    return exe_graph_;
  }

  const ModelLevelData &GetModelLevelData() const {
    return model_level_data_;
  }

  SymbolsToValue *GetSymbolsToValue() {
    return symbols_to_value_;
  }
  // symbols_to_value_成员由外部释放，这里仅替换持有的指针
  void SetSymbolsToValue(SymbolsToValue *symbols_to_value) {
    symbols_to_value_ = symbols_to_value;
  }

 private:
  ModelLevelData model_level_data_;
  ge::ExecuteGraphPtr exe_graph_;
  SymbolsToValue *symbols_to_value_; // 指针生命周期由外部控制
  ExecutorOption *executor_option_{nullptr};
};
}  // namespace gert

#endif  // AIR_CXX_RUNTIME_V2_CORE_BUILDER_GRAPH_EXECUTOR_BUILDER_H_

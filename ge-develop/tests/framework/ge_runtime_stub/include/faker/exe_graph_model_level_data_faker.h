/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_FAKER_EXE_GRAPH_MODEL_LEVEL_DATA_FAKER_H_
#define AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_FAKER_EXE_GRAPH_MODEL_LEVEL_DATA_FAKER_H_
#include <utility>

#include "graph/compute_graph.h"
#include "core/builder/graph_async_value.h"
#include "exe_graph/lowering/buffer_pool.h"
#include "core/builder/graph_executor_builder.h"
#include "core/builder/equivalent_data_edges.h"
#include "framework/runtime/exe_graph_resource_guard.h"

namespace gert {
struct ExeGraphModelLevelData {
  GraphAsyncValue gav;
  EquivalentDataEdges eq_data_anchors;
  SymbolsToValue symbols_to_value;
  std::unique_ptr<uint8_t[]> compute_node_info;
  std::unique_ptr<uint8_t[]> kernel_extend_info;
  std::list<std::string> buffer;

  const ContinuousBuffer *GetComputeNodeInf() const {
    return reinterpret_cast<ContinuousBuffer *>(compute_node_info.get());
  }
  const ContinuousBuffer *GetExtendInfo() const {
    return reinterpret_cast<ContinuousBuffer *>(kernel_extend_info.get());
  }
  ModelLevelData GetModelLevelData() const {
    return {GetComputeNodeInf(), GetExtendInfo()};
  }
};
class ExeGraphModelLevelDataFaker {
 public:
  explicit ExeGraphModelLevelDataFaker(ge::ExecuteGraphPtr exe_graph) : exe_graph_(std::move(exe_graph)) {}
  ExeGraphModelLevelData Build();

 private:
  void FakeGraphAsyncValue(GraphAsyncValue &gav) const;
  EquivalentDataEdges FakeEquivalentDataAnchors() const;

  std::unique_ptr<uint8_t[]> GenerateKernelExtendInfo(std::list<std::string> &buffer) const;
  std::unique_ptr<uint8_t[]> GenerateComputeNodeInfo(std::list<std::string> &buffer) const;

 private:
  ge::ExecuteGraphPtr exe_graph_;
};
}

#endif  // AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_FAKER_EXE_GRAPH_MODEL_LEVEL_DATA_FAKER_H_

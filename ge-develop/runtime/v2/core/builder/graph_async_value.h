/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_CORE_BUILDER_GRAPH_ASYNC_VALUE_H_
#define AIR_CXX_RUNTIME_V2_CORE_BUILDER_GRAPH_ASYNC_VALUE_H_
#include <cstddef>
#include <memory>
#include <map>
#include "graph/fast_graph/fast_node.h"
#include "core/execution_data.h"
#include "runtime/model_v2_executor.h"

namespace gert {
struct GraphAsyncValue {
  size_t total_num{0};
  AsyncAnyValue *values{nullptr};
  std::unique_ptr<uint8_t[]> values_guarder;
  std::map<ge::FastNode *, std::vector<AsyncAnyValue *>> nodes_to_output_values;
  std::map<ge::FastNode *, std::vector<AsyncAnyValue *>> nodes_to_input_values;
};

struct ExeGraphAsyncValue {
  GraphAsyncValue root_graph;
  std::array<GraphAsyncValue, kSubExeGraphTypeEnd> subgraphs;
};
}

#endif  // AIR_CXX_RUNTIME_V2_CORE_BUILDER_GRAPH_ASYNC_VALUE_H_

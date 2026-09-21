/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_COMMON_EXE_GRAPH_H_
#define AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_COMMON_EXE_GRAPH_H_
#include "graph/compute_graph.h"
#include "graph/fast_graph/execute_graph.h"
#include "graph/node.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/graph_dump_utils.h"
#include "graph/utils/execute_graph_adapter.h"
#include "common/checker.h"
namespace gert {
namespace bg {
// 执行图封装，也是后续执行图的接口设计，成熟后实现为高性能版本放到metadef中
class ExeGraph {
 public:
  explicit ExeGraph(ge::ExecuteGraphPtr root_graph) : root_graph_(std::move(root_graph)) {}
  ExeGraph(const ExeGraph &) = delete;
  ExeGraph &operator=(const ExeGraph &) = delete;
  ge::FastNode *GetMainNode() const {
    return ge::ExecuteGraphUtils::FindFirstNodeMatchType(root_graph_.get(), "Main");
  }
  ge::ExecuteGraphPtr GetMainGraph() const {
    auto main_node = GetMainNode();
    GE_ASSERT_NOTNULL(main_node);
    return ge::FastNodeUtils::GetSubgraphFromNode(main_node, 0)->shared_from_this();
  }
  ge::ExecuteGraphPtr GetRootGraph() const {
    return root_graph_;
  }

 private:
  ge::ExecuteGraphPtr root_graph_;
};
}  // namespace bg
}  // namespace gert
#endif  // AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_COMMON_EXE_GRAPH_H_

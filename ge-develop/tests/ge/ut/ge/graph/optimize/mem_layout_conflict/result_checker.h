/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_MEM_CHECK_RESULT_CHECKER_H
#define AIR_CXX_MEM_CHECK_RESULT_CHECKER_H
#include <gtest/gtest.h>

#include <utility>
#include "common/mem_conflict_share_graph.h"
#include "graph/compute_graph.h"
#include "compiler/graph/optimize/mem_layout_conflict_optimize/mem_layout_conflict_optimizer.h"

namespace ge {
namespace mem_check {
struct NodeIo {
  NodeIo(std::string node_name, const uint32_t io_index = 0U) :
        name(std::move(node_name)), index(io_index) {}
  const std::string name;
  const uint32_t index;
};
struct ResultChecker {
static graphStatus CheckIdentityBefore(const ComputeGraphPtr &graph, const std::vector<NodeIo> &nodes);
static graphStatus CheckIdentityNum(const ComputeGraphPtr &graph, const size_t num);
static bool CheckGraphEqual(ge::ComputeGraphPtr graph_a, ge::ComputeGraphPtr graph_b);
};
};
} // namespace ge
#endif // AIR_CXX_MEM_CHECK_RESULT_CHECKER_H

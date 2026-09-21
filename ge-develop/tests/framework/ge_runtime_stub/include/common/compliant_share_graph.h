/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_COMMON_COMPLIANT_SHARE_GRAPH_H_
#define AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_COMMON_COMPLIANT_SHARE_GRAPH_H_
#include "graph/compute_graph.h"
namespace ge {
namespace cg {
ComputeGraphPtr BuildSoftmaxGraph(const std::vector<int64_t> &shape);
ComputeGraphPtr BuildAddGraph(const std::vector<int64_t> &shape1, const std::vector<int64_t> &shape2);
ComputeGraphPtr BuildAddReluReluGraph(const std::vector<int64_t> &shape1, const std::vector<int64_t> &shape2);
ComputeGraphPtr BuildAbsAddReluReluGraph(const std::vector<int64_t> &shape);
ComputeGraphPtr BuildNeedInsertCastGraph(const vector<int64_t> &shape);
ComputeGraphPtr BuildGraphWithUnConsistantType(const vector<int64_t> &shape);
ComputeGraphPtr BuildGraphWithUnConsistantTypeWithConstInput(const vector<int64_t> &shape);
ComputeGraphPtr BuildAddFillGraph(const std::vector<int64_t> &shape1, const std::vector<int64_t> &shape2);
ComputeGraphPtr BuildAddNGraph(const std::vector<std::vector<int64_t>> &shapes);
ComputeGraphPtr BuildAddReluTransposeGraph(const vector<int64_t> &shape1);
ComputeGraphPtr BuildReduceCase1Graph(const std::vector<int64_t> &shape1, const std::vector<int64_t> &shape2);
ComputeGraphPtr BuildReduceCase2Graph(const vector<int64_t> &shape1, const vector<int64_t> &shape2);
ComputeGraphPtr BuildReduceCase3Graph(const vector<int64_t> &shape1, const vector<int64_t> &shape2);
ComputeGraphPtr BuildAddReluRsqrtMatMulGraph(const vector<int64_t> &shape1, const vector<int64_t> &shape2,
                                            const vector<int64_t> &shape3);
ComputeGraphPtr BuildReluReluConcatGraph(const vector<int64_t> &shape1, const vector<int64_t> &shape2);
ComputeGraphPtr BuildReluSquareConcatV2Graph(const vector<int64_t> &shape1, const vector<int64_t> &shape2);
ComputeGraphPtr BuildSoftmaxAddNGraph(const std::vector<int64_t> &shape,
                                     const std::vector<std::vector<int64_t>> &shapes);
ComputeGraphPtr BuildSoftmaxAddGraph(const std::vector<int64_t> &shape1, const std::vector<int64_t> &shape2);
ComputeGraphPtr BuildMatMulSquareMulGraph(const vector<int64_t> &shape1, const vector<int64_t> &shape2, const vector<int64_t> &shape3);
ComputeGraphPtr BuildSquareAbsSquareGraph(const vector<int64_t> &shape1);
ComputeGraphPtr BuildAbsCastGraph(const vector<int64_t> &shape1, int32_t dst_type);
ComputeGraphPtr BuildAddAddGraph(const vector<int64_t> &shape1, const vector<int64_t> &shape2, const vector<int64_t> &shape3);
ComputeGraphPtr BuildOnlyOneNodeSkipRealizeGraph(const vector<int64_t> &shape1, const vector<int64_t> &shape2,
                                                const vector<int64_t> &shape3);
ComputeGraphPtr BuildMatMulReduceSquareMulGraph(const vector<int64_t> &shape1, const vector<int64_t> &shape2,
                                               const vector<int64_t> &shape3, const vector<int64_t> &shape4);
ComputeGraphPtr BuildMatMulSquareMulReduceGraph(const vector<int64_t> &shape1, const vector<int64_t> &shape2,
                                               const vector<int64_t> &shape3, const vector<int64_t> &shape4);
}  // namespace cg
}  // namespace ge
#endif  // AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_COMMON_COMPLIANT_SHARE_GRAPH_H_

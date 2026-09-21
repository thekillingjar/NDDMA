/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TEST_MEMORY_SHARED_GRAPH_H
#define AIR_CXX_TEST_MEMORY_SHARED_GRAPH_H
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <vector>
#include "graph/build/memory/block_mem_assigner.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/tensor_utils.h"
#include "graph/manager/graph_var_manager.h"
#include "common/sgt_slice_type.h"
#include "graph/build/memory/binary_block_mem_assigner.h"

using namespace std;
namespace ge {
namespace block_mem_ut {
class GraphBuilder {
 public:
  explicit GraphBuilder(const std::string &name) { graph_ = std::make_shared<ComputeGraph>(name); }
  NodePtr AddNode(const std::string &name, const std::string &type, int in_cnt, int out_cnt, uint32_t thread_dim = 1U,
                  std::vector<int64_t> shape = {1, 1, 224, 224},
                  Format format = FORMAT_NCHW, DataType data_type = DT_FLOAT);
  NodePtr AddNode(const std::string &name, const std::string &type,
                  std::initializer_list<std::string> input_names,
                  std::initializer_list<std::string> output_names,
                  Format format = FORMAT_NCHW, DataType data_type = DT_FLOAT,
                  std::vector<int64_t> shape = {1, 1, 224, 224});
  Status AddDataEdge(const NodePtr &src_node, int src_idx, const NodePtr &dst_node, int dst_idx);
  void AddControlEdge(const NodePtr &src_node, const NodePtr &dst_node);
  ComputeGraphPtr GetGraph() {
    graph_->TopologicalSorting();
    return graph_;
  }

 private:
  ComputeGraphPtr graph_;
};
ComputeGraphPtr BuildPartitionedCallSuspendOut();
ComputeGraphPtr BuildNodeWithMemoSizeCalcTypeAndMemoryTypeL1();
ComputeGraphPtr BuildGraphForNetoutputNotReuseData();
ComputeGraphPtr BuildSubGraphWithDiffStream();
ComputeGraphPtr BuildKnownSubGraph();
ComputeGraphPtr BuildSingleOutputConnectMultiStreamAndRefNode();
ComputeGraphPtr BuildSingleOutputConnectMultiStreamAndRefNode2();
ComputeGraphPtr BuildSingleOutputConnectMultiStreamAndRefNode3();
ComputeGraphPtr BuildRefNodeConnectContinuousInputNode();
ComputeGraphPtr BuildSingleNoPaddingContinuousConnectContinuousInputNode();
ComputeGraphPtr BuildNoPaddingContinuousAndMultiStreamGraph();
ComputeGraphPtr BuildNoPaddingContinuousMultiInputDiffStream();
ComputeGraphPtr BuildDataRefConst();
ComputeGraphPtr BuildFiveNodesGraph();
ComputeGraphPtr BuildFiveNodesDiffStreamGraph();
ComputeGraphPtr BuildReuseWithOutNodeWrapper();
ComputeGraphPtr BuildAtomicCleanNotReuseSubGraphData();
ComputeGraphPtr BuildGraphWithDtVariant();
ComputeGraphPtr BuildGraphWithStreamMerge();
ComputeGraphPtr BuildGraphWithRefVarSrcVarName();
ComputeGraphPtr BuildGraphWithRefVarSrcVarNameAbsent();
ComputeGraphPtr BuildContinuousInputAndOutputOffsetForBufferFusion();
ComputeGraphPtr BuildPhonyConcatCascated();
ComputeGraphPtr BuildPhonyConcatWithOutputOffsetList();
ComputeGraphPtr BuildPhonySplitWithOutputOffsetList();
ComputeGraphPtr BuildPhonySplitSuspendOutputWithOutputOffsetList();
ComputeGraphPtr BuildPhonySplitCascated();
ComputeGraphPtr BuildPhonySplitGraph();
ComputeGraphPtr BuildHcomWithP2pMemoryType();
ComputeGraphPtr BuildPhonyConcatSizeLessThanSumOfAllInputs();
ComputeGraphPtr BuildPhonyConcatCascatedConnectRefNode();
ComputeGraphPtr BuildPhonyConcatWithRefNodeInputs();
ComputeGraphPtr BuildPhonyConcatWithSameInputThrougRefNode();
ComputeGraphPtr BuildPhonyConcatWithSameInputThrougRefNode2();
ComputeGraphPtr AtomicNodeConnectReShapeConnectNetoutput();
ComputeGraphPtr BuildAtomicCleanInputGraph();
ComputeGraphPtr BuildAtomicCleanWorkspaceGraph();
ComputeGraphPtr BuildHcomWithBufferPoolId();
ComputeGraphPtr BuildHcomWithContinuousOutputGraph();
ComputeGraphPtr BuildPartitionedCallWithAtomicNode();
ComputeGraphPtr BuildGraphWithOutputNotAssign();
ComputeGraphPtr BuildNestingWrapperWithSubgraphNodeDiffStream();
ComputeGraphPtr BuildContinuousOutInWithTwoOutIn();
ComputeGraphPtr BuildContinuousOutInWithOutAsHeader();
ComputeGraphPtr BuildContinuousOutInWithOutAsHeaderAndTail();
ComputeGraphPtr BuildContinuousOutInWithInAsHeader();
ComputeGraphPtr BuildContinuousOutInWithInAsHeaderAndTail();
ComputeGraphPtr BuildContinuousOutInWithTwoContinuousInNode();
ComputeGraphPtr BuildContinuousOutInWithTwoContinuousInNodeAndRefNode();
ComputeGraphPtr BuildContinuousOutInWithTwoContinuousOutNodeAndRefNode();
ComputeGraphPtr BuildContinuousOutInWithOneInput();
ComputeGraphPtr BuildContinuousOutInWithOneOutput();
ComputeGraphPtr BuildContinuousOutInWithThreeNode();
ComputeGraphPtr BuildContinuousOutInVisitLastNodeFirst();
ComputeGraphPtr BuildOneNodeConnectTwoPhonyConcat();
} // namespace block_mem_ut
} // namespace ge
#endif  // AIR_CXX_TEST_MEMORY_SHARED_GRAPH_H

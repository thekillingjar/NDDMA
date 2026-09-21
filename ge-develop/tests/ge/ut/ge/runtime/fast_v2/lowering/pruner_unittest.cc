/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "lowering/pass/utils/pruner.h"
#include <gtest/gtest.h>
#include "common/bg_test.h"
#include "common/summary_checker.h"
#include "graph/utils/execute_graph_utils.h"
#include "graph/utils/graph_dump_utils.h"
#include "exe_graph/lowering/frame_selector.h"
#include "stub/gert_runtime_stub.h"
namespace gert {
class PrunerUT : public bg::BgTestAutoCreateFrame {};

namespace {
ge::ExecuteGraphPtr BuildGraphWithInnerData() {
  // create Init
  auto init_node = bg::ValueHolder::CreateVoid<bg::ValueHolder>(GetExecuteGraphTypeStr(ExecuteGraphType::kInit), {});
  (void)bg::ValueHolder::PushGraphFrame(init_node, "Init");
  auto init_frame = bg::ValueHolder::PopGraphFrame();

  // create DeInit
  auto de_init_node = bg::ValueHolder::CreateVoid<bg::ValueHolder>("DeInit", {});
  (void)bg::ValueHolder::PushGraphFrame(de_init_node, "DeInit");
  auto de_init_frame = bg::ValueHolder::PopGraphFrame();

  std::string type = "Add";
  auto infer_func = bg::FrameSelector::OnInitRoot([&type]() -> std::vector<bg::ValueHolderPtr> {
    auto node_type = bg::ValueHolder::CreateConst(type.c_str(), type.size() + 1, true);
    return {bg::ValueHolder::CreateSingleDataOutput("FindInferShapeFunc", {node_type})};
  });

  // create Main
  auto main_node = bg::ValueHolder::CreateVoid<bg::ValueHolder>(GetExecuteGraphTypeStr(ExecuteGraphType::kMain), {});
  (void)bg::ValueHolder::PushGraphFrame(main_node, "Main");
  auto infer_shape = bg::ValueHolder::CreateDataOutput("InferShape", infer_func, 1U);
  auto main_frame = bg::ValueHolder::PopGraphFrame();
  auto root_frame = bg::ValueHolder::PopGraphFrame();

  return root_frame->GetExecuteGraph();
}
}  // namespace
/*
 *  SelectL1Allocator
 *    /  \
 * data0  data1
 */
TEST_F(PrunerUT, Prune_DeleteNode_OneNormalNode) {
  auto data0 = bg::ValueHolder::CreateFeed(0);
  auto data1 = bg::ValueHolder::CreateFeed(0);
  auto allocator = bg::ValueHolder::CreateSingleDataOutput("SelectL1Allocator", {data0, data1});
  auto frame = bg::ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);

  auto select_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph.get(), "SelectL1Allocator");
  ASSERT_NE(select_node, nullptr);
  bool changed = false;
  ASSERT_EQ(bg::Pruner().PruneFromNodes({select_node}, changed), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);
  ASSERT_EQ(ExeGraphSummaryChecker(graph.get()).StrictDirectNodeTypes({{"Data", 2}}), "success");
}
/*
 *
 * SplitTensor  SplitTensor  SplitTensor
 *    |            |            |
 *  data0        data1        data0
 */
TEST_F(PrunerUT, Prune_DeleteNodes_MultipleStartNodes) {
  auto data0 = bg::ValueHolder::CreateFeed(0);
  auto outputs0 = bg::ValueHolder::CreateDataOutput("SplitTensor", {data0}, 2);
  auto data1 = bg::ValueHolder::CreateFeed(1);
  auto outputs1 = bg::ValueHolder::CreateDataOutput("SplitTensor", {data1}, 2);
  auto data2 = bg::ValueHolder::CreateFeed(2);
  auto outputs2 = bg::ValueHolder::CreateDataOutput("SplitTensor", {data2}, 2);
  auto frame = bg::ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);

  auto nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "SplitTensor");
  ASSERT_EQ(nodes.size(), 3U);
  bool changed = false;
  ASSERT_EQ(bg::Pruner().PruneFromNodes(nodes, changed), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);
  ASSERT_EQ(ExeGraphSummaryChecker(graph.get()).StrictDirectNodeTypes({{"Data", 3}}), "success");
}
/*
 *  LaunchKernelWithFlag
 *    /  \
 * data0  data1
 */
TEST_F(PrunerUT, Prune_IgnoreNode_ModifyDevice) {
  auto data0 = bg::ValueHolder::CreateFeed(0);
  auto data1 = bg::ValueHolder::CreateFeed(0);
  bg::ValueHolder::CreateSingleDataOutput("LaunchKernelWithFlag", {data0, data1});
  auto frame = bg::ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);

  auto launch_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph.get(), "LaunchKernelWithFlag");
  ASSERT_NE(launch_node, nullptr);
  bool changed = false;
  ASSERT_EQ(bg::Pruner().PruneFromNodes({launch_node}, changed), ge::GRAPH_SUCCESS);
  ASSERT_FALSE(changed);
  ASSERT_EQ(ExeGraphSummaryChecker(graph.get()).StrictDirectNodeTypes({{"Data", 2}, {"LaunchKernelWithFlag", 1}}),
            "success");
}
/*
 *   FreeMemory
 *        |
 *  AllocMemory
 *     /  \
 * data0  data1
 */
TEST_F(PrunerUT, Prune_DeleteAllocMemAndFree_WhenAllocOutNoUse) {
  auto data0 = bg::ValueHolder::CreateFeed(0);
  auto data1 = bg::ValueHolder::CreateFeed(1);
  auto addr = bg::ValueHolder::CreateSingleDataOutput("AllocMemory", {data0, data1});
  bg::ValueHolder::CreateVoidGuarder("FreeMemory", addr, {});

  auto frame = bg::ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);

  auto alloc_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph.get(), "AllocMemory");
  ASSERT_NE(alloc_node, nullptr);

  bool changed = false;
  ASSERT_EQ(bg::Pruner().PruneFromNodes({alloc_node}, changed), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);
  ASSERT_EQ(ExeGraphSummaryChecker(graph.get()).StrictDirectNodeTypes({{"Data", 2}}), "success");
}
/*
 *   FreeMemory  LaunchKernelWithFlag
 *           \    /
 *         AllocMemory
 *            /  \
 *        data0  data1
 */
TEST_F(PrunerUT, Prune_DoNotDelete_WhenAllocOutInUse) {
  auto data0 = bg::ValueHolder::CreateFeed(0);
  auto data1 = bg::ValueHolder::CreateFeed(1);
  auto addr = bg::ValueHolder::CreateSingleDataOutput("AllocMemory", {data0, data1});
  bg::ValueHolder::CreateVoidGuarder("FreeMemory", addr, {});
  bg::ValueHolder::CreateVoid<bg::ValueHolder>("LaunchKernelWithFlag", {addr});

  auto frame = bg::ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);

  auto alloc_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph.get(), "AllocMemory");
  ASSERT_NE(alloc_node, nullptr);

  bool changed = false;
  ASSERT_EQ(bg::Pruner().PruneFromNodes({alloc_node}, changed), ge::GRAPH_SUCCESS);
  ASSERT_FALSE(changed);
  ASSERT_EQ(
      ExeGraphSummaryChecker(graph.get())
          .StrictDirectNodeTypes({{"Data", 2}, {"AllocMemory", 1}, {"FreeMemory", 1}, {"LaunchKernelWithFlag", 1}}),
      "success");
}
/*
 *   CalcTensorSizeFromShape
 *       /   \
 * const0   SplitTensor
 *             |
 *           data0
 */
TEST_F(PrunerUT, Prune_DeleteNodesFollowTopo_MultipleNodes1) {
  auto dt = ge::DT_FLOAT;
  auto const0 = bg::ValueHolder::CreateConst(&dt, sizeof(dt));
  auto data0 = bg::ValueHolder::CreateFeed(0);
  auto split_results = bg::ValueHolder::CreateDataOutput("SplitTensor", {data0}, 2);
  bg::ValueHolder::CreateSingleDataOutput("CalcTensorSizeFromShape", {const0, split_results[0]});

  auto frame = bg::ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);

  auto calc_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph.get(), "CalcTensorSizeFromShape");
  ASSERT_NE(calc_node, nullptr);

  bool changed = false;
  ASSERT_EQ(bg::Pruner().PruneFromNodes({calc_node}, changed), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);
  ASSERT_EQ(ExeGraphSummaryChecker(graph.get()).StrictDirectNodeTypes({{"Data", 1}}), "success");
}
/*
 *
 *                                CalcTensorSizeFromShape
 *                                      |          |
 *                                    const2       |
 *                                                 |
 *   CalcTensorSizeFromShape    CalcTensorSizeFromShape
 *       /              \        /                   \
 * const0              SplitTensor                 const1
 *                        |
 *                      data0
 */
TEST_F(PrunerUT, Prune_DeleteNodesFollowTopo_MultipleNodes2) {
  auto data0 = bg::ValueHolder::CreateFeed(0);
  auto split_results = bg::ValueHolder::CreateDataOutput("SplitTensor", {data0}, 2);

  auto dt = ge::DT_FLOAT;
  auto const0 = bg::ValueHolder::CreateConst(&dt, sizeof(dt));
  auto output0 = bg::ValueHolder::CreateSingleDataOutput("CalcTensorSizeFromShape", {const0, split_results[0]});

  dt = ge::DT_FLOAT16;
  auto const1 = bg::ValueHolder::CreateConst(&dt, sizeof(dt));
  auto calc1 = bg::ValueHolder::CreateSingleDataOutput("CalcTensorSizeFromShape", {const1, split_results[0]});

  dt = ge::DT_INT64;
  auto const2 = bg::ValueHolder::CreateConst(&dt, sizeof(dt));
  auto output1 = bg::ValueHolder::CreateSingleDataOutput("CalcTensorSizeFromShape", {const2, calc1});

  auto frame = bg::ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);

  auto output0_node = graph->FindNode(output0->GetFastNode()->GetNodeToken());
  ASSERT_EQ(output0_node, output0->GetFastNode());
  auto output1_node = graph->FindNode(output1->GetFastNode()->GetNodeToken());
  ASSERT_EQ(output1_node, output1->GetFastNode());

  bool changed = false;
  std::vector<ge::FastNode *> start_nodes = {output0->GetFastNode(), output1->GetFastNode()};
  ASSERT_EQ(bg::Pruner().PruneFromNodes(start_nodes, changed), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);
  ASSERT_EQ(ExeGraphSummaryChecker(graph.get()).StrictDirectNodeTypes({{"Data", 1}}), "success");
}
/*
 *                FreeMemory
 *                     |
 *                AllocMemory
 *                 /     \
 *                /      CalcTensorSizeFromShape
 *               /          /    \
 *  SelectL1Allocator    const0   SplitTensor
 *     /   \                       |
 *  data0  data1                 data2
 */
TEST_F(PrunerUT, Prune_DeleteNodesFollowTopo_MultipleNodes3) {
  auto data0 = bg::ValueHolder::CreateFeed(0);
  auto data1 = bg::ValueHolder::CreateFeed(1);
  auto allocator = bg::ValueHolder::CreateSingleDataOutput("SelectL1Allocator", {data0, data1});
  auto data2 = bg::ValueHolder::CreateFeed(2);
  auto shape_and_addr = bg::ValueHolder::CreateDataOutput("SplitTensor", {data2}, 2);
  auto size = bg::ValueHolder::CreateSingleDataOutput("CalcTensorSizeFromStorage",
                                                      {bg::ValueHolder::CreateConst("1", 1), shape_and_addr[0]});
  auto addr = bg::ValueHolder::CreateSingleDataOutput("AllocMemory", {allocator, size});
  bg::ValueHolder::CreateVoidGuarder("FreeMemory", addr, {});

  auto frame = bg::ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);

  auto alloc_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph.get(), "AllocMemory");
  ASSERT_NE(alloc_node, nullptr);

  bool changed = false;
  ASSERT_EQ(bg::Pruner().PruneFromNodes({alloc_node}, changed), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);
  ASSERT_EQ(ExeGraphSummaryChecker(graph.get()).StrictDirectNodeTypes({{"Data", 3}}), "success");
}
/*
 *  LaunchKernelWithFlag
 *    /  \
 * data0  data1
 */
TEST_F(PrunerUT, Prune_Failed_WhenStartNodeCannodeDelete) {
  auto data0 = bg::ValueHolder::CreateFeed(0);
  auto data1 = bg::ValueHolder::CreateFeed(0);
  auto launch_holder = bg::ValueHolder::CreateSingleDataOutput("LaunchKernelWithFlag", {data0, data1});
  auto frame = bg::ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);

  auto launch_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph.get(), "LaunchKernelWithFlag");
  ASSERT_NE(launch_node, nullptr);
  bool changed = false;

  GertRuntimeStub stub;
  ASSERT_NE(bg::Pruner().StartNodesMustBeDeleted().PruneFromNodes({launch_node}, changed), ge::GRAPH_SUCCESS);
  auto expect_log = "Failed to delete node " + launch_node->GetName();
  EXPECT_EQ(stub.GetSlogStub().FindErrorLogEndsWith(expect_log.c_str()), 0);
}

TEST_F(PrunerUT, Prune_Ok_RemoveInnerDataNodeFromMaingraph) {
  auto graph = BuildGraphWithInnerData();
  auto infer_shape = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "InferShape");
  ASSERT_EQ(infer_shape.size(), 1);
  ASSERT_NE(infer_shape[0], nullptr);
  auto main_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph.get(), "Main");
  auto main_graph = ge::FastNodeUtils::GetSubgraphFromNode(main_node, 0);
  ASSERT_NE(main_graph, nullptr);
  auto init_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph.get(), "Init");
  auto init_graph = ge::FastNodeUtils::GetSubgraphFromNode(init_node, 0);
  ASSERT_NE(init_graph, nullptr);

  bool changed = false;
  ASSERT_EQ(bg::Pruner().PruneFromNodes({infer_shape}, changed), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);
  ASSERT_EQ(ExeGraphSummaryChecker(main_graph).StrictDirectNodeTypes({}), "success");
  ASSERT_EQ(
      ExeGraphSummaryChecker(init_graph).StrictDirectNodeTypes({{"Const", 1}, {"NoOp", 1}, {"InnerNetOutput", 1}}),
      "success");
}
}  // namespace gert
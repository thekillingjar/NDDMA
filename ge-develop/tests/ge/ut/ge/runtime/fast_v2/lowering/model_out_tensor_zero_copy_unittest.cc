/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "lowering/pass/model_out_tensor_zero_copy.h"
#include <gtest/gtest.h>
#include "common/share_graph.h"
#include "graph/utils/execute_graph_utils.h"
#include "graph/utils/execute_graph_adapter.h"
#include "graph/utils/graph_dump_utils.h"
#include "common/summary_checker.h"
#include "common/topo_checker.h"
#include "faker/global_data_faker.h"
#include "lowering/graph_converter.h"
#include "graph_builder/bg_condition.h"
#include "exe_graph/lowering/exe_graph_attrs.h"
#include "lowering/pass_changed_kernels_info.h"
#include "faker/model_desc_holder_faker.h"

namespace gert {
namespace {
/**
 *
 *           NetOutput
 *              |c            c
 *  EnsureTensorAtOutMemory ------------> FreeMemory
 *       |  |   |     |    \              ^       ^
 * +-----+  |  attrs stream  out_data     |       |c
 * |        |0                            |       |
 * |   AllocMemory -----------------------+--> Launch
 * |   /          \       0
 * | size     allocator
 * |  |
 * shape
 */
std::unique_ptr<bg::GraphFrame> BuildGraph1() {
  bg::ValueHolder::PushGraphFrame();
  auto allocator = bg::ValueHolder::CreateFeed(0);
  auto shape = bg::ValueHolder::CreateFeed(1);
  auto output_data = bg::ValueHolder::CreateSingleDataOutput("OutputData", {});
  auto stream = bg::ValueHolder::CreateFeed(-1);
  auto size = bg::ValueHolder::CreateSingleDataOutput("CalcTensorSizeFromShape", {shape});
  auto mem = bg::ValueHolder::CreateSingleDataOutput("AllocMemHbm", {allocator, size});
  auto free_mem = bg::ValueHolder::CreateVoidGuarder("FreeMemory", mem, {});
  auto launch = bg::ValueHolder::CreateVoid<bg::ValueHolder>("Launch", {mem});
  auto attrs = bg::ValueHolder::CreateConst("Hello", 5);
  bg::ValueHolder::CreateVoid<bg::ValueHolder>("EnsureTensorAtOutMemory", {shape, mem, attrs, stream, output_data});
  return bg::ValueHolder::PopGraphFrame({output_data}, {});
}
/*
 *
 *                           NetOutput
 *                                  |
 *      EnsureTensorAtOutMemory     |
 *       |    |         |    | \    |
 *    attrs   |         |c   |  OutputData
 *           Foo --> launch  |
 *         /    \            |
 * allocator    size         |
 *               |           |
 *            SomeShape -----+
 */
std::unique_ptr<bg::GraphFrame> BuildGraph2() {
  bg::ValueHolder::PushGraphFrame();
  auto allocator = bg::ValueHolder::CreateFeed(0);
  auto shape = bg::ValueHolder::CreateFeed(1);
  auto size = bg::ValueHolder::CreateSingleDataOutput("CalcSize", {shape});
  auto attrs = bg::ValueHolder::CreateConst("Hello", 5);
  auto stream = bg::ValueHolder::CreateFeed(-1);
  auto output_data = bg::ValueHolder::CreateSingleDataOutput("OutputData", {});
  auto mem = bg::ValueHolder::CreateSingleDataOutput("Foo", {allocator, size});
  auto launch = bg::ValueHolder::CreateVoid<bg::ValueHolder>("Launch", {mem});
  auto tensor =
      bg::ValueHolder::CreateVoid<bg::ValueHolder>("EnsureTensorAtOutMemory", {shape, mem, attrs, stream, output_data});
  bg::ValueHolder::AddDependency(launch, tensor);
  return bg::ValueHolder::PopGraphFrame({output_data}, {});
}

/**
 *
 *           NetOutput
 *              |c            c
 *  EnsureTensorAtOutMemory ------------> FreeMemory
 *       |  |   |     |    \              ^       ^
 * +-----+  |  attrs stream  out_data     |       |c
 * |        |0
 * |   IdentityAddr                       |       |
 * |        |                             |       |
 * |   AllocMemory -----------------------+--> Launch
 * |   /          \       0
 * | size     allocator
 * |  |
 * shape
 */
std::unique_ptr<bg::GraphFrame> BuildGraph3() {
  bg::ValueHolder::PushGraphFrame();
  auto allocator = bg::ValueHolder::CreateFeed(0);
  auto shape = bg::ValueHolder::CreateFeed(1);
  auto output_data = bg::ValueHolder::CreateSingleDataOutput("OutputData", {});
  auto stream = bg::ValueHolder::CreateFeed(-1);
  auto size = bg::ValueHolder::CreateSingleDataOutput("CalcTensorSizeFromShape", {shape});
  auto mem = bg::ValueHolder::CreateSingleDataOutput("AllocMemHbm", {allocator, size});
  auto free_mem = bg::ValueHolder::CreateVoidGuarder("FreeMemory", mem, {});
  auto launch = bg::ValueHolder::CreateVoid<bg::ValueHolder>("Launch", {mem});
  auto attrs = bg::ValueHolder::CreateConst("Hello", 5);
  auto identity = bg::ValueHolder::CreateSingleDataOutput("IdentityAddr", {mem});
  bg::ValueHolder::CreateVoid<bg::ValueHolder>("EnsureTensorAtOutMemory",
                                               {shape, identity, attrs, stream, output_data});
  return bg::ValueHolder::PopGraphFrame({output_data}, {});
}

std::unique_ptr<bg::GraphFrame> BuildGraph4() {
  bg::ValueHolder::PushGraphFrame();
  auto allocator = bg::ValueHolder::CreateFeed(0);
  auto shape = bg::ValueHolder::CreateFeed(1);
  auto cond = bg::ValueHolder::CreateFeed(2);
  auto output_data = bg::ValueHolder::CreateSingleDataOutput("OutputData", {});
  auto stream = bg::ValueHolder::CreateFeed(-1);
  auto holder = bg::If<bg::ValueHolder>(
      cond,
      [&]() -> std::vector<bg::ValueHolderPtr> {
        auto size = bg::ValueHolder::CreateSingleDataOutput("CalcTensorSizeFromShape", {shape});
        auto mem = bg::ValueHolder::CreateSingleDataOutput("AllocMemHbm", {allocator, size});
        auto free_mem = bg::ValueHolder::CreateVoidGuarder("FreeMemory", mem, {});
        auto launch = bg::ValueHolder::CreateVoid<bg::ValueHolder>("Launch", {mem});
        return {mem};
      },
      [&]() -> std::vector<bg::ValueHolderPtr> {
        auto size = bg::ValueHolder::CreateSingleDataOutput("CalcTensorSizeFromShape", {shape});
        auto mem = bg::ValueHolder::CreateSingleDataOutput("AllocMemHbm", {allocator, size});
        auto free_mem = bg::ValueHolder::CreateVoidGuarder("FreeMemory", mem, {});
        auto launch = bg::ValueHolder::CreateVoid<bg::ValueHolder>("Launch", {mem});
        return {mem};
      });
  auto attrs = bg::ValueHolder::CreateConst("Hello", 5);
  bg::ValueHolder::CreateVoid<bg::ValueHolder>("EnsureTensorAtOutMemory",
                                               {shape, holder[0U], attrs, stream, output_data});
  return bg::ValueHolder::PopGraphFrame({output_data}, {});
}

std::unique_ptr<bg::GraphFrame> BuildGraph5() {
  bg::ValueHolder::PushGraphFrame();
  auto allocator = bg::ValueHolder::CreateFeed(0);
  auto shape = bg::ValueHolder::CreateFeed(1);
  auto cond = bg::ValueHolder::CreateFeed(2);
  auto output_data = bg::ValueHolder::CreateSingleDataOutput("OutputData", {});
  auto stream = bg::ValueHolder::CreateFeed(-1);
  auto holder = bg::If<bg::ValueHolder>(
      cond,
      [&]() -> std::vector<bg::ValueHolderPtr> {
        auto size = bg::ValueHolder::CreateSingleDataOutput("CalcTensorSizeFromShape", {shape});
        auto mem = bg::ValueHolder::CreateSingleDataOutput("AllocMemHbm", {allocator, size});
        auto free_mem = bg::ValueHolder::CreateVoidGuarder("FreeMemory", mem, {});
        auto launch = bg::ValueHolder::CreateVoid<bg::ValueHolder>("Launch", {mem});
        return {mem};
      },
      [&]() -> std::vector<bg::ValueHolderPtr> {
        auto foo1 = bg::ValueHolder::CreateSingleDataOutput("Foo1", {shape});
        return {foo1};
      });
  auto attrs = bg::ValueHolder::CreateConst("Hello", 5);
  bg::ValueHolder::CreateVoid<bg::ValueHolder>("EnsureTensorAtOutMemory",
                                               {shape, holder[0U], attrs, stream, output_data});
  return bg::ValueHolder::PopGraphFrame({output_data}, {});
}

std::unique_ptr<bg::GraphFrame> BuildGraph6() {
  bg::ValueHolder::PushGraphFrame();
  auto allocator = bg::ValueHolder::CreateFeed(0);
  auto shape = bg::ValueHolder::CreateFeed(1);
  auto cond = bg::ValueHolder::CreateFeed(2);
  auto cond2 = bg::ValueHolder::CreateFeed(3);
  auto output_data = bg::ValueHolder::CreateSingleDataOutput("OutputData", {});
  auto stream = bg::ValueHolder::CreateFeed(-1);
  auto holder = bg::If<bg::ValueHolder>(
      cond,
      [&]() -> std::vector<bg::ValueHolderPtr> {
        auto size = bg::ValueHolder::CreateSingleDataOutput("CalcTensorSizeFromShape", {shape});
        auto mem = bg::ValueHolder::CreateSingleDataOutput("AllocMemHbm", {allocator, size});
        auto free_mem = bg::ValueHolder::CreateVoidGuarder("FreeMemory", mem, {});
        auto launch = bg::ValueHolder::CreateVoid<bg::ValueHolder>("Launch", {mem});
        return {mem};
      },
      [&]() -> std::vector<bg::ValueHolderPtr> {
        auto sub_holder = bg::If<bg::ValueHolder>(
            cond2,
            [&]() -> std::vector<bg::ValueHolderPtr> {
              auto size = bg::ValueHolder::CreateSingleDataOutput("CalcTensorSizeFromShape", {shape});
              auto mem = bg::ValueHolder::CreateSingleDataOutput("AllocMemHbm", {allocator, size});
              auto free_mem = bg::ValueHolder::CreateVoidGuarder("FreeMemory", mem, {});
              auto launch = bg::ValueHolder::CreateVoid<bg::ValueHolder>("Launch", {mem});
              return {mem};
            },
            [&]() -> std::vector<bg::ValueHolderPtr> {
              auto foo1 = bg::ValueHolder::CreateSingleDataOutput("Foo1", {shape});
              return {foo1};
            });
        return {sub_holder};
      });
  auto attrs = bg::ValueHolder::CreateConst("Hello", 5);
  bg::ValueHolder::CreateVoid<bg::ValueHolder>("EnsureTensorAtOutMemory",
                                               {shape, holder[0U], attrs, stream, output_data});
  return bg::ValueHolder::PopGraphFrame({output_data}, {});
}
}  // namespace
class ModelOutTensorZeroCopyUT : public testing::Test {};

/**
 *
 *                                          NetOutput                                                   NetOutput
 *                      c                    /c     \                               c                    /c    |
 *       FreeMemory <---------  EnsureAtUserMemory  |                FreeMemory <---------  EnsureAtUserMemory |
 *       ^      ^              /  |   |     |    \  |      =>        ^      ^              /  |   |     |    \ |
 *      c|      |           shape |  attrs stream output            c|      |           shape |  attrs stream \|
 *       |      |                 |0                                 |      |                 |0               +
 * SomeNodes <--+----------- AllocMemory                       SomeNodes <--+----------- AllocModelOutTensor   |
 *                    0      /          \                                         0      /          |       \  |
 *                       allocator     size                                          allocator     size    output
 */
TEST_F(ModelOutTensorZeroCopyUT, ZeroCopyAllDone) {
  auto frame = BuildGraph1();
  ASSERT_NE(frame, nullptr);

  auto exe_graph = frame->GetExecuteGraph();
  bool changed = false;
  ASSERT_EQ(bg::ModelOutTensorZeroCopy().Run(exe_graph.get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_EQ(exe_graph->TopologicalSorting(), ge::GRAPH_SUCCESS);

  EXPECT_EQ(ExeGraphSummaryChecker(exe_graph.get())
                .StrictDirectNodeTypes({{"Data", 3},
                                        {"Const", 1},
                                        {"CalcTensorSizeFromShape", 1},
                                        {"AllocModelOutTensor", 1},
                                        {"EnsureTensorAtOutMemory", 1},
                                        {"FreeMemory", 1},
                                        {"Launch", 1},
                                        {"OutputData", 1},
                                        {"NetOutput", 1}}),
            "success");

  auto netoutput = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "NetOutput");
  ASSERT_NE(netoutput, nullptr);
  EXPECT_EQ(FastNodeTopoChecker(netoutput).StrictConnectFrom({{"OutputData", 0}}), "success");

  auto output_data_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "OutputData");
  ASSERT_NE(output_data_node, nullptr);
  for (const auto &node : output_data_node->GetOutDataNodes()) {
    if (node->GetType() == "NetOutput") {
      continue;
    }
    if (node->GetType() == "EnsureTensorAtOutMemory") {
      continue;
    }
    ASSERT_EQ(node->GetType(), "AllocModelOutTensor");

    EXPECT_EQ(FastNodeTopoChecker(node).StrictConnectTo(0, {{"Launch"}, {"FreeMemory"}, {"EnsureTensorAtOutMemory"}}),
              "success");
    EXPECT_EQ(FastNodeTopoChecker(node).StrictConnectFrom({{"Data"}, {"CalcTensorSizeFromShape"}, {"OutputData"}}),
              "success");
  }
}

/**
 *
 *                                          NetOutput                                                   NetOutput
 *                      c                    /c     \                               c                    /c    |
 *       FreeMemory <---------  EnsureAtUserMemory  |                FreeMemory <---------  EnsureAtUserMemory |
 *       ^      ^              /  |   |     |    \  |      =>        ^      ^              /  |   |     |    \ |
 *      c|      |           shape |  attrs stream output            c|      |           shape |  attrs stream \|
 *       |      |                 |0                                 |      |                 |0               +
 *       |      |              IdentityAddr                          |      |              IdentityAddr        |
 *       |      |                 |                                  |      |                 |                |
 * SomeNodes <--+----------- AllocMemory                       SomeNodes <--+----------- AllocModelOutTensor   |
 *                    0      /          \                                         0      /          |       \  |
 *                       allocator     size                                          allocator     size    output
 */
TEST_F(ModelOutTensorZeroCopyUT, ZeroCopyAllDoneWithIdentity) {
  auto frame = BuildGraph3();
  ASSERT_NE(frame, nullptr);

  auto exe_graph = frame->GetExecuteGraph();
  bool changed = false;
  ASSERT_EQ(bg::ModelOutTensorZeroCopy().Run(exe_graph.get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_EQ(exe_graph->TopologicalSorting(), ge::GRAPH_SUCCESS);

  EXPECT_EQ(ExeGraphSummaryChecker(exe_graph.get())
                .StrictDirectNodeTypes({{"Data", 3},
                                        {"Const", 1},
                                        {"CalcTensorSizeFromShape", 1},
                                        {"AllocModelOutTensor", 1},
                                        {"EnsureTensorAtOutMemory", 1},
                                        {"IdentityAddr", 1},
                                        {"FreeMemory", 1},
                                        {"Launch", 1},
                                        {"OutputData", 1},
                                        {"NetOutput", 1}}),
            "success");

  auto netoutput = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "NetOutput");
  ASSERT_NE(netoutput, nullptr);
  EXPECT_EQ(FastNodeTopoChecker(netoutput).StrictConnectFrom({{"OutputData", 0}}), "success");

  auto output_data_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "OutputData");
  ASSERT_NE(output_data_node, nullptr);
  for (const auto &node : output_data_node->GetOutDataNodes()) {
    if (node->GetType() == "NetOutput") {
      continue;
    }
    if (node->GetType() == "EnsureTensorAtOutMemory") {
      continue;
    }
    ASSERT_EQ(node->GetType(), "AllocModelOutTensor");

    EXPECT_EQ(FastNodeTopoChecker(node).StrictConnectTo(0, {{"Launch"}, {"FreeMemory"}, {"IdentityAddr"}}), "success");
    EXPECT_EQ(FastNodeTopoChecker(node).StrictConnectFrom({{"Data"}, {"CalcTensorSizeFromShape"}, {"OutputData"}}),
              "success");
  }
}

TEST_F(ModelOutTensorZeroCopyUT, If_ZeroCopyAllDone) {
  auto frame = BuildGraph4();
  ASSERT_NE(frame, nullptr);

  auto exe_graph = frame->GetExecuteGraph();
  bool changed = false;
  ASSERT_EQ(bg::ModelOutTensorZeroCopy().Run(exe_graph.get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_EQ(exe_graph->TopologicalSorting(), ge::GRAPH_SUCCESS);
  EXPECT_EQ(ExeGraphSummaryChecker(exe_graph.get())
                .StrictDirectNodeTypes({{"Data", 4},
                                        {"Const", 1},
                                        {"If", 1},
                                        {"EnsureTensorAtOutMemory", 1},
                                        {"FreeMemory", 1},
                                        {"OutputData", 1},
                                        {"NetOutput", 1}}),
            "success");

  auto if_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "If");
  ASSERT_NE(if_node, nullptr);
  EXPECT_EQ(ExeGraphSummaryChecker(ge::FastNodeUtils::GetSubgraphFromNode(if_node, 1))
                .StrictDirectNodeTypes({{"InnerData", 3},
                                        {"CalcTensorSizeFromShape", 1},
                                        {"AllocModelOutTensor", 1},
                                        {"Launch", 1},
                                        {"InnerNetOutput", 1}}),
            "success");
  EXPECT_EQ(ExeGraphSummaryChecker(ge::FastNodeUtils::GetSubgraphFromNode(if_node, 2))
                .StrictDirectNodeTypes({{"InnerData", 3},
                                        {"CalcTensorSizeFromShape", 1},
                                        {"AllocModelOutTensor", 1},
                                        {"Launch", 1},
                                        {"InnerNetOutput", 1}}),
            "success");

  auto netoutput = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "NetOutput");
  ASSERT_NE(netoutput, nullptr);
  EXPECT_EQ(FastNodeTopoChecker(netoutput).StrictConnectFrom({{"OutputData", 0}}), "success");
  EXPECT_EQ(FastNodeTopoChecker(if_node).StrictConnectFrom(
                {{"Data", 0}, {"Data", 0}, {"Data", 0}, {"OutputData", 0}, {"OutputData", 0}}),
            "success");

  auto if_subgraph1 = ge::FastNodeUtils::GetSubgraphFromNode(if_node, 1);
  auto then_alloc_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(if_subgraph1, "AllocModelOutTensor");
  ASSERT_NE(then_alloc_node, nullptr);
  EXPECT_EQ(FastNodeTopoChecker(then_alloc_node)
                .StrictConnectFrom({{"InnerData"}, {"CalcTensorSizeFromShape"}, {"InnerData"}}),
            "success");

  auto if_subgraph2 = ge::FastNodeUtils::GetSubgraphFromNode(if_node, 2);
  auto else_alloc_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(if_subgraph2, "AllocModelOutTensor");
  ASSERT_NE(else_alloc_node, nullptr);
  EXPECT_EQ(FastNodeTopoChecker(else_alloc_node)
                .StrictConnectFrom({{"InnerData"}, {"CalcTensorSizeFromShape"}, {"InnerData"}}),
            "success");
}

TEST_F(ModelOutTensorZeroCopyUT, If_ZeroCopy_OnlyThenBranch) {
  auto frame = BuildGraph5();
  ASSERT_NE(frame, nullptr);

  auto exe_graph = frame->GetExecuteGraph();
  bool changed = false;
  ASSERT_EQ(bg::ModelOutTensorZeroCopy().Run(exe_graph.get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_EQ(exe_graph->TopologicalSorting(), ge::GRAPH_SUCCESS);
  EXPECT_EQ(ExeGraphSummaryChecker(exe_graph.get())
                .StrictDirectNodeTypes({{"Data", 4},
                                        {"Const", 1},
                                        {"If", 1},
                                        {"EnsureTensorAtOutMemory", 1},
                                        {"FreeMemory", 1},
                                        {"OutputData", 1},
                                        {"NetOutput", 1}}),
            "success");

  auto if_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "If");
  ASSERT_NE(if_node, nullptr);
  EXPECT_EQ(ExeGraphSummaryChecker(ge::FastNodeUtils::GetSubgraphFromNode(if_node, 1))
                .StrictDirectNodeTypes({{"InnerData", 3},
                                        {"CalcTensorSizeFromShape", 1},
                                        {"AllocModelOutTensor", 1},
                                        {"Launch", 1},
                                        {"InnerNetOutput", 1}}),
            "success");
  EXPECT_EQ(ExeGraphSummaryChecker(ge::FastNodeUtils::GetSubgraphFromNode(if_node, 2))
                .StrictDirectNodeTypes({{"InnerData", 1}, {"Foo1", 1}, {"IdentityAddr", 1}, {"InnerNetOutput", 1}}),
            "success");

  auto netoutput = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "NetOutput");
  ASSERT_NE(netoutput, nullptr);
  EXPECT_EQ(FastNodeTopoChecker(netoutput).StrictConnectFrom({{"OutputData", 0}}), "success");
  EXPECT_EQ(FastNodeTopoChecker(if_node).StrictConnectFrom({{"Data", 0}, {"Data", 0}, {"Data", 0}, {"OutputData", 0}}),
            "success");
  auto if_subgraph = ge::FastNodeUtils::GetSubgraphFromNode(if_node, 1);
  auto then_alloc_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(if_subgraph, "AllocModelOutTensor");
  ASSERT_NE(then_alloc_node, nullptr);
  EXPECT_EQ(FastNodeTopoChecker(then_alloc_node)
                .StrictConnectFrom({{"InnerData"}, {"CalcTensorSizeFromShape"}, {"InnerData"}}),
            "success");
}

TEST_F(ModelOutTensorZeroCopyUT, If_ZeroCopy_ZeroCopyAllDown_With_Subgraph_Nesting) {
  auto frame = BuildGraph6();
  ASSERT_NE(frame, nullptr);

  auto exe_graph = frame->GetExecuteGraph();
  bool changed = false;
  ASSERT_EQ(bg::ModelOutTensorZeroCopy().Run(exe_graph.get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_EQ(exe_graph->TopologicalSorting(), ge::GRAPH_SUCCESS);
  EXPECT_EQ(ExeGraphSummaryChecker(exe_graph.get())
                .StrictDirectNodeTypes({{"Data", 5},
                                        {"Const", 1},
                                        {"If", 1},
                                        {"EnsureTensorAtOutMemory", 1},
                                        {"FreeMemory", 1},
                                        {"OutputData", 1},
                                        {"NetOutput", 1}}),
            "success");

  auto if_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "If");
  ASSERT_NE(if_node, nullptr);
  auto then_graph = ge::FastNodeUtils::GetSubgraphFromNode(if_node, 1);
  ASSERT_NE(then_graph, nullptr);
  EXPECT_EQ(ExeGraphSummaryChecker(then_graph)
                .StrictDirectNodeTypes({{"InnerData", 3},
                                        {"CalcTensorSizeFromShape", 1},
                                        {"AllocModelOutTensor", 1},
                                        {"Launch", 1},
                                        {"InnerNetOutput", 1}}),
            "success");
  auto else_graph = ge::FastNodeUtils::GetSubgraphFromNode(if_node, 2);
  ASSERT_NE(else_graph, nullptr);
  EXPECT_EQ(
      ExeGraphSummaryChecker(else_graph).StrictDirectNodeTypes({{"InnerData", 4}, {"If", 1}, {"InnerNetOutput", 1}}),
      "success");

  auto sub_if_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(else_graph, "If");
  ASSERT_NE(sub_if_node, nullptr);
  EXPECT_EQ(ExeGraphSummaryChecker(ge::FastNodeUtils::GetSubgraphFromNode(sub_if_node, 1))
                .StrictDirectNodeTypes({{"InnerData", 3},
                                        {"CalcTensorSizeFromShape", 1},
                                        {"AllocModelOutTensor", 1},
                                        {"Launch", 1},
                                        {"InnerNetOutput", 1}}),
            "success");
  EXPECT_EQ(ExeGraphSummaryChecker(ge::FastNodeUtils::GetSubgraphFromNode(sub_if_node, 2))
                .StrictDirectNodeTypes({{"InnerData", 1}, {"Foo1", 1}, {"IdentityAddr", 1}, {"InnerNetOutput", 1}}),
            "success");

  auto netoutput = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "NetOutput");
  ASSERT_NE(netoutput, nullptr);
  EXPECT_EQ(FastNodeTopoChecker(netoutput).StrictConnectFrom({{"OutputData", 0}}), "success");
  EXPECT_EQ(FastNodeTopoChecker(if_node).StrictConnectFrom(
                {{"Data", 0}, {"Data", 0}, {"Data", 0}, {"Data", 0}, {"OutputData", 0}, {"OutputData", 0}}),
            "success");

  auto then_alloc_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(then_graph, "AllocModelOutTensor");
  ASSERT_NE(then_alloc_node, nullptr);
  EXPECT_EQ(FastNodeTopoChecker(then_alloc_node)
                .StrictConnectFrom({{"InnerData"}, {"CalcTensorSizeFromShape"}, {"InnerData"}}),
            "success");

  auto sub_then_graph = ge::FastNodeUtils::GetSubgraphFromNode(sub_if_node, 1);
  auto sub_then_alloc_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(sub_then_graph, "AllocModelOutTensor");
  ASSERT_NE(sub_then_alloc_node, nullptr);
  EXPECT_EQ(FastNodeTopoChecker(sub_then_alloc_node)
                .StrictConnectFrom({{"InnerData"}, {"CalcTensorSizeFromShape"}, {"InnerData"}}),
            "success");
}

/*
 *
 *   NetOutput FreeMemory
 *           \/
 *   EnsureTensorAtOutMemory  <------ GetNewShape
 *            |     \                    |
 *            |     attrs                |c
 *     AllocMemHbm ------------------> launch
 *         /    \
 * allocator    size
 */
TEST_F(ModelOutTensorZeroCopyUT, DoNotFuse_WhenCircle) {
  bg::ValueHolder::PushGraphFrame();
  auto allocator = bg::ValueHolder::CreateFeed(0);
  auto size = bg::ValueHolder::CreateFeed(1);
  auto shape = bg::ValueHolder::CreateFeed(2);
  auto output_data = bg::ValueHolder::CreateSingleDataOutput("OutputData", {});
  auto stream = bg::ValueHolder::CreateFeed(-1);
  auto attrs = bg::ValueHolder::CreateConst("Hello", 5);
  auto mem = bg::ValueHolder::CreateSingleDataOutput("AllocMemHbm", {allocator, size});
  auto launch = bg::ValueHolder::CreateVoid<bg::ValueHolder>("Launch", {mem});
  auto new_shape = bg::ValueHolder::CreateSingleDataOutput("GetNewShape", {size});
  bg::ValueHolder::AddDependency(launch, new_shape);
  auto tensor = bg::ValueHolder::CreateVoid<bg::ValueHolder>("EnsureTensorAtOutMemory",
                                                             {new_shape, mem, attrs, stream, output_data});
  auto frame = bg::ValueHolder::PopGraphFrame({output_data}, {});
  ASSERT_NE(frame, nullptr);

  auto exe_graph = frame->GetExecuteGraph();
  bool changed = false;
  ASSERT_EQ(bg::ModelOutTensorZeroCopy().Run(exe_graph.get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_EQ(exe_graph->TopologicalSorting(), ge::GRAPH_SUCCESS);

  ASSERT_EQ(ExeGraphSummaryChecker(exe_graph.get())
                .StrictAllNodeTypes({{"Data", 4},
                                     {"AllocModelOutTensor", 1},
                                     {"Const", 1},
                                     {"Launch", 1},
                                     {"GetNewShape", 1},
                                     {"EnsureTensorAtOutMemory", 1},
                                     {"OutputData", 1},
                                     {"NetOutput", 1}}),
            "success");

  auto netoutput = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "NetOutput");
  ASSERT_NE(netoutput, nullptr);
  ASSERT_EQ(FastNodeTopoChecker(netoutput).StrictConnectFrom({{"OutputData", 0}}), "success");
  auto update_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "EnsureTensorAtOutMemory");
  ASSERT_NE(update_node, nullptr);
  ASSERT_EQ(FastNodeTopoChecker(update_node)
                .StrictConnectFrom(
                    {{"GetNewShape", 0}, {"AllocModelOutTensor", 0}, {"Const", 0}, {"Data", 0}, {"OutputData", 0}}),
            "success");
}
/*
 *
 *                           NetOutput
 *                                  |
 *      EnsureTensorAtOutMemory     |
 *       |    |         |    | \    |
 *    attrs   |         |c   |  OutputData
 *           Foo --> launch  |
 *         /    \            |
 * allocator    size         |
 *               |           |
 *            SomeShape -----+
 */
TEST_F(ModelOutTensorZeroCopyUT, DoNotZeroCopy_WhenNoAllocKernel) {
  auto frame = BuildGraph2();
  ASSERT_NE(frame, nullptr);

  auto exe_graph = frame->GetExecuteGraph();
  bool changed = false;
  ASSERT_EQ(bg::ModelOutTensorZeroCopy().Run(exe_graph.get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_EQ(exe_graph->TopologicalSorting(), ge::GRAPH_SUCCESS);

  ASSERT_EQ(ExeGraphSummaryChecker(exe_graph.get())
                .StrictAllNodeTypes({{"Data", 3},
                                     {"Const", 1},
                                     {"CalcSize", 1},
                                     {"Foo", 1},
                                     {"Launch", 1},
                                     {"EnsureTensorAtOutMemory", 1},
                                     {"OutputData", 1},
                                     {"NetOutput", 1}}),
            "success");

  auto netoutput = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "NetOutput");
  ASSERT_NE(netoutput, nullptr);
  ASSERT_EQ(FastNodeTopoChecker(netoutput).StrictConnectFrom({{"OutputData", 0}}), "success");

  auto output_data_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "OutputData");
  ASSERT_NE(output_data_node, nullptr);
  ASSERT_EQ(FastNodeTopoChecker(output_data_node).StrictConnectTo(0, {{"NetOutput"}, {"EnsureTensorAtOutMemory"}}),
            "success");
}

/*
 ****** ComputeGraph ******
 *       netoutput
 *          |
 *         add1
 *         /  \
 *      data1 data2
 *
 ****** ExecuteGraph ******
 * before ModelOutTensorZeroCopy pass:
 *           netoutput
 *            /      \
 *           /   LaunchKernelWithHandle
 *          /        /
 *        BuildTensor
 *              |
 *        AllocMemHbm
 *               ... ...
 *
 *  after ModelOutTensorZeroCopy pass:
 *           netoutput
 *            /      \
 *           /   LaunchKernelWithHandle
 *          /        /
 * AllocModelOutTensor
 *              ... ...
 *
 */
/** ST 用例，需要等 Adaptor 移动到 Pass 后再补全
TEST_F(ModelOutTensorZeroCopyUT, TestSingleAicoreGraph) {
  auto compute_graph = ShareGraph::BuildSingleNodeGraph();
  compute_graph->TopologicalSorting();
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  auto model_desc_holder = ModelDescHolderFaker().Build();
  auto exe_graph = GraphConverter()
      .SetModelDescHolder(&model_desc_holder)
      .ConvertComputeGraphToExecuteGraph(compute_graph, global_data);
  ASSERT_NE(exe_graph, nullptr);

  ge::GraphUtils::DumpGEGraph(exe_graph, "", true, "./single_node_graph.txt");
  ge::GraphUtils::DumpGEGraphToOnnx(*exe_graph, "ModelOutZeroCopy_SingleAicoreNode");
  auto main_graph = ge::NodeUtils::GetSubgraph(*exe_graph->FindFirstNodeMatchType("Main"), 0);
  ASSERT_NE(main_graph, nullptr);
  EXPECT_EQ(SummaryChecker(main_graph)
                .StrictDirectNodeTypes({{"AllocModelOutTensor", 1},
                                        {"EnsureTensorAtOutMemory", 1},
                                        {"AllocBatchHbm", 1},
                                        {"CalcTensorSizeFromShape", 1},
                                        {"CalcTensorSizeFromStorage", 2},
                                        {"Data", 6},
                                        {"FreeBatchHbm", 1},
                                        {"FreeMemory", 3},
                                        {"InferShape", 1},
                                        {"InnerData", 23},
                                        {"LaunchKernelWithHandle", 1},
                                        {"CopyFlowLaunch", 1},
                                        {"OutputData", 1},
                                        {"NetOutput", 1},
                                        {"SplitRtStreams", 1},
                                        {"SelectL1Allocator", 1},
                                        {"SelectL2Allocator", 1},
                                        {"SplitTensor", 2},
                                        {"Tiling", 1}}),
            "success");

  auto netoutput = main_graph->FindNode("NetOutput");
  ASSERT_NE(netoutput, nullptr);
  ASSERT_EQ(NodeTopoChecker(netoutput).StrictConnectFrom({{"OutputData", 0}, {"EnsureTensorAtOutMemory", -1}}),
"success"); auto output_data = main_graph->FindFirstNodeMatchType("OutputData"); for (const auto &node :
output_data->GetOutDataNodes()) { if (node->GetType() == "NetOutput") { continue;
    }
    if (node->GetType() == "EnsureTensorAtOutMemory") {
      continue;
    }
    ASSERT_EQ(node->GetType(), "AllocModelOutTensor");
    EXPECT_EQ(NodeTopoChecker(node).StrictConnectFrom(
                  {{"SelectL2Allocator"}, {"CalcTensorSizeFromShape"}, {"OutputData"}}),
              "success");
  }
}
*/

/**
 *
 *           NetOutput                                                  NetOutput
 *              |c            c                                            |c
 *  EnsureTensorAtOutMemory ------------> FreeMemory            EnsureTensorAtOutMemory
 *       |  |   |     |    \              ^       ^                 |  |   |     |    \
 * +-----+  |  attrs stream  out_data     |       |c     =>  +------+  |  attrs stream \
 * |        |0                            |       |          |         |1               |  1
 * |   AllocMemory -----------------------+--> Launch        | SplitTensorForOutputData-|-----> Launch
 * |   /          \       0                                  |  /     c|     \         /
 * | size     allocator                                      | size allocator out_data
 * |  |                                                      |  |
 * shape                                                     shape
 */
TEST_F(ModelOutTensorZeroCopyUT, AlwaysZeroCopy_ChangedToSplitTensor_WhenOptionIsEnable) {
  auto frame = BuildGraph1();
  ASSERT_NE(frame, nullptr);

  auto exe_graph = frame->GetExecuteGraph();
  // todo 530还没有实现
  // ge::GraphUtils::DumpGEGraphToOnnx(*exe_graph, "ModelOutZeroCopy_BeforePass");
  bool changed = false;
  auto pass = bg::ModelOutTensorZeroCopy();
  pass.SetLoweringOption({.trust_shape_on_out_tensor = false, .always_zero_copy = true});
  ge::DumpGraph(exe_graph.get(), "BuildGraph1");
  ASSERT_EQ(pass.Run(exe_graph.get(), changed), ge::GRAPH_SUCCESS);
  ge::DumpGraph(exe_graph.get(), "BuildGraph1AfterPass");
  ASSERT_EQ(exe_graph->TopologicalSorting(), ge::GRAPH_SUCCESS);
  ge::DumpGraph(exe_graph.get(), "BuildGraph1AfterTopoSort");
  // ge::GraphUtils::DumpGEGraphToOnnx(*exe_graph, "ModelOutZeroCopy_AfterPassModify");

  EXPECT_EQ(ExeGraphSummaryChecker(exe_graph.get())
                .StrictDirectNodeTypes({{"Data", 3},
                                        {"Const", 1},
                                        {"CalcTensorSizeFromShape", 1},
                                        {"SplitTensorForOutputData", 1},
                                        {"EnsureTensorAtOutMemory", 1},
                                        {"Launch", 1},
                                        {"OutputData", 1},
                                        {"NetOutput", 1}}),
            "success");

  auto netoutput = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "NetOutput");
  ASSERT_NE(netoutput, nullptr);
  EXPECT_EQ(FastNodeTopoChecker(netoutput).StrictConnectFrom({{"OutputData", 0}}), "success");

  auto output_data_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "OutputData");
  ASSERT_NE(output_data_node, nullptr);
  EXPECT_EQ(
      FastNodeTopoChecker(output_data_node)
          .StrictConnectTo(0, {{"SplitTensorForOutputData", 0}, {"EnsureTensorAtOutMemory", 4}, {"NetOutput", 0}}),
      "success");

  auto split_tensor_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "SplitTensorForOutputData");
  ASSERT_NE(split_tensor_node, nullptr);
  ASSERT_EQ(split_tensor_node->GetDataOutNum(), 2);
  EXPECT_EQ(FastNodeTopoChecker(split_tensor_node).StrictConnectTo(0, {}), "success");
  EXPECT_EQ(FastNodeTopoChecker(split_tensor_node).StrictConnectTo(1, {{"Launch", 0}, {"EnsureTensorAtOutMemory", 1}}),
            "success");
  EXPECT_EQ(FastNodeTopoChecker(split_tensor_node)
                .StrictConnectFrom({{"OutputData", 0}, {"Data", 0}, {"CalcTensorSizeFromShape", 0}}),
            "success");
  auto changed_info = split_tensor_node->GetOpDescBarePtr()->GetExtAttr<PassChangedKernels>(kPassChangedInfo);
  ASSERT_NE(changed_info, nullptr);
  ASSERT_EQ(changed_info->pass_changed_kernels.size(), 1U);
  EXPECT_EQ(changed_info->pass_changed_kernels.at(0).first.kernel_name, split_tensor_node->GetName());
  EXPECT_EQ(changed_info->pass_changed_kernels.at(0).second.kernel_name, split_tensor_node->GetName());
  EXPECT_EQ(changed_info->pass_changed_kernels.at(0).first.idx, 0);
  EXPECT_EQ(changed_info->pass_changed_kernels.at(0).second.idx, 1);

  auto copy_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "EnsureTensorAtOutMemory");
  ASSERT_NE(copy_node, nullptr);
  ASSERT_EQ(copy_node->GetDataInNum(), 5);
  ASSERT_EQ(copy_node->GetDataOutNum(), 0);
  EXPECT_EQ(FastNodeTopoChecker(copy_node).StrictConnectFrom(
                {{"Data"}, {"SplitTensorForOutputData", 1}, {"Const"}, {"Data"}, {"OutputData"}}),
            "success");
}
/*
 * 本用例借用自 DoNotZeroCopy_WhenNoAllocKernel，仅添加了always_zero_copy选项
 */
TEST_F(ModelOutTensorZeroCopyUT, AlwaysZeroCopy_NotEnable_WhenZeroCopyMissmatch) {
  auto frame = BuildGraph2();
  ASSERT_NE(frame, nullptr);

  auto exe_graph = frame->GetExecuteGraph();
  bool changed = false;
  auto pass = bg::ModelOutTensorZeroCopy();
  pass.SetLoweringOption({.trust_shape_on_out_tensor = false, .always_zero_copy = true});
  ASSERT_EQ(pass.Run(exe_graph.get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_EQ(exe_graph->TopologicalSorting(), ge::GRAPH_SUCCESS);

  ASSERT_EQ(ExeGraphSummaryChecker(exe_graph.get())
                .StrictAllNodeTypes({{"Data", 3},
                                     {"Const", 1},
                                     {"CalcSize", 1},
                                     {"Foo", 1},
                                     {"Launch", 1},
                                     {"EnsureTensorAtOutMemory", 1},
                                     {"OutputData", 1},
                                     {"NetOutput", 1}}),
            "success");

  auto netoutput = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "NetOutput");
  ASSERT_NE(netoutput, nullptr);
  ASSERT_EQ(FastNodeTopoChecker(netoutput).StrictConnectFrom({{"OutputData", 0}}), "success");

  auto output_data_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "OutputData");
  ASSERT_NE(output_data_node, nullptr);
  ASSERT_EQ(FastNodeTopoChecker(output_data_node).StrictConnectTo(0, {{"NetOutput"}, {"EnsureTensorAtOutMemory"}}),
            "success");
}
}  // namespace gert

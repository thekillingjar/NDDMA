/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "common/bg_test.h"
#include "graph/utils/graph_dump_utils.h"

#include "core/builder/graph_node.h"
#include "graph_builder/bg_condition.h"
#include "exe_graph/lowering/value_holder_utils.h"

namespace gert {
class GraphNodeUT : public bg::BgTest {
 protected:
  void SetUp() override {
    bg::ValueHolder::PushGraphFrame();
    auto init = bg::ValueHolder::CreateVoid<bg::ValueHolder>("Init", {});
    auto main = bg::ValueHolder::CreateVoid<bg::ValueHolder>("Main", {});
    auto de_init = bg::ValueHolder::CreateVoid<bg::ValueHolder>("DeInit", {});

    bg::ValueHolder::PushGraphFrame(init, "Init");
    init_frame_ = bg::ValueHolder::PopGraphFrame({}, {});

    bg::ValueHolder::PushGraphFrame(de_init, "DeInit");
    de_init_frame_ = bg::ValueHolder::PopGraphFrame();

    bg::ValueHolder::PushGraphFrame(main, "Main");
  }

  void TearDown() override {
    BgTest::TearDown();
    init_frame_.reset();
    de_init_frame_.reset();
  }

  std::unique_ptr<bg::GraphFrame> init_frame_;
  std::unique_ptr<bg::GraphFrame> de_init_frame_;
};

/*
 * main graph:
 *
 *         NetOutput
 *          /       \
 *      node3(3)   node4(4)
 *         |   \ c  /
 *    node1(1) node2(2)
 *          \  /
 *          data
 */
TEST_F(GraphNodeUT, EnsureNodeExeInOrder_Prioirty_Not_Same) {
  auto data = bg::ValueHolder::CreateFeed(0);
  auto node1 = bg::ValueHolder::CreateSingleDataOutput("foo1", {data});
  ge::AttrUtils::SetInt(bg::ValueHolderUtils::GetNodeOpDescBarePtr(node1), "priority", 1);
  auto node2 = bg::ValueHolder::CreateSingleDataOutput("foo2", {data});
  ge::AttrUtils::SetInt(bg::ValueHolderUtils::GetNodeOpDescBarePtr(node2), "priority", 2);
  auto node3 = bg::ValueHolder::CreateSingleDataOutput("LaunchKernelWithFlag", {node1});
  bg::ValueHolder::AddDependency(node2, node3);
  ge::AttrUtils::SetInt(bg::ValueHolderUtils::GetNodeOpDescBarePtr(node3), "priority", 3);
  auto node4 = bg::ValueHolder::CreateSingleDataOutput("LaunchKernelWithFlag", {node2});
  ge::AttrUtils::SetInt(bg::ValueHolderUtils::GetNodeOpDescBarePtr(node4), "priority", 4);
  auto main_frame = bg::ValueHolder::PopGraphFrame({node3, node4}, {}, "NetOutput");
  GraphNode graph_node;
  EXPECT_EQ(graph_node.EnsureNodeExeInOrder(main_frame->GetExecuteGraph().get()), ge::GRAPH_SUCCESS);

  EXPECT_EQ(graph_node.additional_indegree_info[node1->GetFastNode()], 0);
  EXPECT_EQ(graph_node.additional_indegree_info[node2->GetFastNode()], 0);
  EXPECT_EQ(graph_node.additional_indegree_info[node3->GetFastNode()], 0);
  EXPECT_EQ(graph_node.additional_indegree_info[node4->GetFastNode()], 1);
  EXPECT_EQ(graph_node.additional_add_info[node1->GetFastNode()].size(), 0);
  EXPECT_EQ(graph_node.additional_add_info[node2->GetFastNode()].size(), 0);
  EXPECT_EQ(graph_node.additional_add_info[node3->GetFastNode()].size(), 1);
  EXPECT_EQ(graph_node.additional_add_info[node3->GetFastNode()][0], node4->GetFastNode());
  EXPECT_EQ(graph_node.additional_add_info[node4->GetFastNode()].size(), 0);
}

/*
 * main graph:
 *
 *                     NetOutput
 *                        |
 *                      node5
 *            /         /     \         \
 *     node1(1)  node2(1)  node3(2)  node4(2)
 */
TEST_F(GraphNodeUT, EnsureNodeExeInOrder_Prioirty_With_Same1) {
  auto node1 = bg::ValueHolder::CreateSingleDataOutput("LaunchKernelWithFlag", {});
  ge::AttrUtils::SetInt(bg::ValueHolderUtils::GetNodeOpDescBarePtr(node1), "priority", 1);
  auto node2 = bg::ValueHolder::CreateSingleDataOutput("LaunchKernelWithFlag", {});
  ge::AttrUtils::SetInt(bg::ValueHolderUtils::GetNodeOpDescBarePtr(node2), "priority", 1);
  auto node3 = bg::ValueHolder::CreateSingleDataOutput("LaunchKernelWithFlag", {});
  ge::AttrUtils::SetInt(bg::ValueHolderUtils::GetNodeOpDescBarePtr(node3), "priority", 2);
  auto node4 = bg::ValueHolder::CreateSingleDataOutput("LaunchKernelWithFlag", {});
  ge::AttrUtils::SetInt(bg::ValueHolderUtils::GetNodeOpDescBarePtr(node4), "priority", 2);
  auto node5 = bg::ValueHolder::CreateSingleDataOutput("LaunchKernelWithFlag", {node1, node2, node3, node4});
  ge::AttrUtils::SetInt(bg::ValueHolderUtils::GetNodeOpDescBarePtr(node5), "priority", 3);
  auto main_frame = bg::ValueHolder::PopGraphFrame({node5}, {}, "NetOutput");
  GraphNode graph_node;
  EXPECT_EQ(graph_node.EnsureNodeExeInOrder(main_frame->GetExecuteGraph().get()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(graph_node.additional_indegree_info[node1->GetFastNode()], 0);
  EXPECT_EQ(graph_node.additional_indegree_info[node2->GetFastNode()], 0);
  EXPECT_EQ(graph_node.additional_indegree_info[node3->GetFastNode()], 2);
  EXPECT_EQ(graph_node.additional_indegree_info[node4->GetFastNode()], 2);
  EXPECT_EQ(graph_node.additional_indegree_info[node5->GetFastNode()], 0);

  EXPECT_EQ(graph_node.additional_add_info[node1->GetFastNode()].size(), 2);
  auto &node1_info = graph_node.additional_add_info[node1->GetFastNode()];
  EXPECT_TRUE(std::find(node1_info.begin(), node1_info.end(), node3->GetFastNode()) != node1_info.end());
  EXPECT_TRUE(std::find(node1_info.begin(), node1_info.end(), node4->GetFastNode()) != node1_info.end());
  EXPECT_EQ(graph_node.additional_add_info[node2->GetFastNode()].size(), 2);
  auto &node2_info = graph_node.additional_add_info[node1->GetFastNode()];
  EXPECT_TRUE(std::find(node2_info.begin(), node2_info.end(), node3->GetFastNode()) != node2_info.end());
  EXPECT_TRUE(std::find(node2_info.begin(), node2_info.end(), node4->GetFastNode()) != node2_info.end());
  EXPECT_EQ(graph_node.additional_add_info[node3->GetFastNode()].size(), 0);
  EXPECT_EQ(graph_node.additional_add_info[node4->GetFastNode()].size(), 0);
  EXPECT_EQ(graph_node.additional_add_info[node5->GetFastNode()].size(), 0);
}

/*
 * main graph:
 *
 *         NetOutput
 *           /    \
 *     node4(2)  node5(2)
 *         |   \c  |
 *     node2(1)  node3(1)
 *          \    /
 *         node1(1)
 *            |
 *          data
 */
TEST_F(GraphNodeUT, EnsureNodeExeInOrder_Prioirty_With_Same2) {
  auto data = bg::ValueHolder::CreateFeed(0);
  auto node1 = bg::ValueHolder::CreateSingleDataOutput("LaunchKernelWithFlag", {data});
  ge::AttrUtils::SetInt(bg::ValueHolderUtils::GetNodeOpDescBarePtr(node1), "priority", 1);
  auto node2 = bg::ValueHolder::CreateSingleDataOutput("LaunchKernelWithFlag", {node1});
  ge::AttrUtils::SetInt(bg::ValueHolderUtils::GetNodeOpDescBarePtr(node2), "priority", 1);
  auto node3 = bg::ValueHolder::CreateSingleDataOutput("LaunchKernelWithFlag", {node1});
  ge::AttrUtils::SetInt(bg::ValueHolderUtils::GetNodeOpDescBarePtr(node3), "priority", 1);
  auto node4 = bg::ValueHolder::CreateSingleDataOutput("LaunchKernelWithFlag", {node2});
  ge::AttrUtils::SetInt(bg::ValueHolderUtils::GetNodeOpDescBarePtr(node4), "priority", 2);
  bg::ValueHolder::AddDependency(node3, node4);
  auto node5 = bg::ValueHolder::CreateSingleDataOutput("LaunchKernelWithFlag", {node3});
  ge::AttrUtils::SetInt(bg::ValueHolderUtils::GetNodeOpDescBarePtr(node5), "priority", 2);
  auto main_frame = bg::ValueHolder::PopGraphFrame({node4, node5}, {}, "NetOutput");
  GraphNode graph_node;
  EXPECT_EQ(graph_node.EnsureNodeExeInOrder(main_frame->GetExecuteGraph().get()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(graph_node.additional_indegree_info[node1->GetFastNode()], 0);
  EXPECT_EQ(graph_node.additional_indegree_info[node2->GetFastNode()], 0);
  EXPECT_EQ(graph_node.additional_indegree_info[node3->GetFastNode()], 0);
  EXPECT_EQ(graph_node.additional_indegree_info[node4->GetFastNode()], 0);
  EXPECT_EQ(graph_node.additional_indegree_info[node5->GetFastNode()], 1);

  EXPECT_EQ(graph_node.additional_add_info[node1->GetFastNode()].size(), 0);
  EXPECT_EQ(graph_node.additional_add_info[node2->GetFastNode()].size(), 1);
  EXPECT_EQ(graph_node.additional_add_info[node2->GetFastNode()][0], node5->GetFastNode());
  EXPECT_EQ(graph_node.additional_add_info[node3->GetFastNode()].size(), 0);
  EXPECT_EQ(graph_node.additional_add_info[node4->GetFastNode()].size(), 0);
  EXPECT_EQ(graph_node.additional_add_info[node5->GetFastNode()].size(), 0);
}

/*
 * main graph:
 *
 *         NetOutput
 *           /    \
 *     node4(2)-->node5(2)
 *         |       |
 *     node2(1)  node3(1)
 *          \    /
 *         node1(1)
 *            |
 *          data
 */
TEST_F(GraphNodeUT, EnsureNodeExeInOrder_Prioirty_With_Same3) {
  auto data = bg::ValueHolder::CreateFeed(0);
  auto node1 = bg::ValueHolder::CreateSingleDataOutput("LaunchKernelWithFlag", {data});
  ge::AttrUtils::SetInt(bg::ValueHolderUtils::GetNodeOpDescBarePtr(node1), "priority", 1);
  auto node2 = bg::ValueHolder::CreateSingleDataOutput("LaunchKernelWithFlag", {node1});
  ge::AttrUtils::SetInt(bg::ValueHolderUtils::GetNodeOpDescBarePtr(node2), "priority", 1);
  auto node3 = bg::ValueHolder::CreateSingleDataOutput("LaunchKernelWithFlag", {node1});
  ge::AttrUtils::SetInt(bg::ValueHolderUtils::GetNodeOpDescBarePtr(node3), "priority", 1);
  auto node4 = bg::ValueHolder::CreateSingleDataOutput("LaunchKernelWithFlag", {node2});
  ge::AttrUtils::SetInt(bg::ValueHolderUtils::GetNodeOpDescBarePtr(node4), "priority", 2);
  auto node5 = bg::ValueHolder::CreateSingleDataOutput("LaunchKernelWithFlag", {node3, node4});
  ge::AttrUtils::SetInt(bg::ValueHolderUtils::GetNodeOpDescBarePtr(node5), "priority", 2);
  auto main_frame = bg::ValueHolder::PopGraphFrame({node4, node5}, {}, "NetOutput");
  GraphNode graph_node;
  EXPECT_EQ(graph_node.EnsureNodeExeInOrder(main_frame->GetExecuteGraph().get()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(graph_node.additional_indegree_info[node1->GetFastNode()], 0);
  EXPECT_EQ(graph_node.additional_indegree_info[node2->GetFastNode()], 0);
  EXPECT_EQ(graph_node.additional_indegree_info[node3->GetFastNode()], 0);
  EXPECT_EQ(graph_node.additional_indegree_info[node4->GetFastNode()], 1);
  EXPECT_EQ(graph_node.additional_indegree_info[node5->GetFastNode()], 0);

  EXPECT_EQ(graph_node.additional_add_info[node1->GetFastNode()].size(), 0);
  EXPECT_EQ(graph_node.additional_add_info[node2->GetFastNode()].size(), 0);
  EXPECT_EQ(graph_node.additional_add_info[node3->GetFastNode()].size(), 1);
  EXPECT_EQ(graph_node.additional_add_info[node3->GetFastNode()][0], node4->GetFastNode());
  EXPECT_EQ(graph_node.additional_add_info[node4->GetFastNode()].size(), 0);
  EXPECT_EQ(graph_node.additional_add_info[node5->GetFastNode()].size(), 0);
}

/*
 * main graph:
 * NetOutput
 *     |
 *   node5(5)
 *     |
 *    if(1)
 *     |
 *   node1(0)
 *
 * then graph:
 *      InnerNetOutput
 *        |      |
 *    node2(2)  node3(3)
 *            \   /
 *          InnerData
 *
 * else graph:
 *      InnerNetOutput
 *        |       |
 *    node4(4)    |
 *            \   |
 *          InnerData
 */
// 测试If子图里的节点需要新增的控制关系
TEST_F(GraphNodeUT, EnsureNodeExeInOrder_Prioirty_With_Subgraph1) {
  auto data = bg::ValueHolder::CreateFeed(0);
  auto node1 = bg::ValueHolder::CreateSingleDataOutput("LaunchKernelWithFlag", {data});
  ge::AttrUtils::SetInt(bg::ValueHolderUtils::GetNodeOpDescBarePtr(node1), "priority", 0);
  auto if_outputs = bg::If<bg::ValueHolder>(
      node1,
      [&]() -> std::vector<bg::ValueHolderPtr> {
        auto node2 = bg::ValueHolder::CreateSingleDataOutput("AicpuLaunchCCKernel", {node1});
        ge::AttrUtils::SetInt(bg::ValueHolderUtils::GetNodeOpDescBarePtr(node2), "priority", 2);
        auto node3 = bg::ValueHolder::CreateSingleDataOutput("DavinciModelExecute", {node1});
        ge::AttrUtils::SetInt(bg::ValueHolderUtils::GetNodeOpDescBarePtr(node3), "priority", 3);
        return {node2, node3};
      },
      [&]() -> std::vector<bg::ValueHolderPtr> {
        auto node4 = bg::ValueHolder::CreateSingleDataOutput("LaunchHcomKernel", {node1});
        ge::AttrUtils::SetInt(bg::ValueHolderUtils::GetNodeOpDescBarePtr(node4), "priority", 4);
        return {node1, node4};
      });
  ge::AttrUtils::SetInt(bg::ValueHolderUtils::GetNodeOpDescBarePtr(if_outputs[0]), "priority", 1);
  auto node5 = bg::ValueHolder::CreateSingleDataOutput("LaunchKernelWithFlag", if_outputs);
  ge::AttrUtils::SetInt(bg::ValueHolderUtils::GetNodeOpDescBarePtr(node5), "priority", 5);
  auto main_frame = bg::ValueHolder::PopGraphFrame({node5}, {}, "NetOutput");
  auto root_graph = ge::ExecuteGraphUtils::FindRootGraph(main_frame->GetExecuteGraph().get());
  ge::DumpGraph(root_graph, "test4");

  GraphNode graph_node;
  EXPECT_EQ(graph_node.EnsureNodeExeInOrder(main_frame->GetExecuteGraph().get()), ge::GRAPH_SUCCESS);
  auto if_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(main_frame->GetExecuteGraph().get(), "If");
  ASSERT_NE(if_node, nullptr);
  auto then_graph = ge::FastNodeUtils::GetSubgraphFromNode(if_node, 1U);
  ASSERT_NE(then_graph, nullptr);
  auto else_graph = ge::FastNodeUtils::GetSubgraphFromNode(if_node, 2U);
  ASSERT_NE(else_graph, nullptr);

  auto node2Ptr = ge::ExecuteGraphUtils::FindFirstNodeMatchType(then_graph, "AicpuLaunchCCKernel");
  ASSERT_NE(node2Ptr, nullptr);
  auto node3Ptr = ge::ExecuteGraphUtils::FindFirstNodeMatchType(then_graph, "DavinciModelExecute");
  ASSERT_NE(node3Ptr, nullptr);
  auto node4Ptr = ge::ExecuteGraphUtils::FindFirstNodeMatchType(else_graph, "LaunchHcomKernel");
  ASSERT_NE(node4Ptr, nullptr);
  EXPECT_EQ(graph_node.additional_indegree_info[node1->GetFastNode()], 0);
  EXPECT_EQ(graph_node.additional_indegree_info[node2Ptr], 0);
  EXPECT_EQ(graph_node.additional_indegree_info[node3Ptr], 1);
  EXPECT_EQ(graph_node.additional_indegree_info[node4Ptr], 0);
  EXPECT_EQ(graph_node.additional_indegree_info[node5->GetFastNode()], 0);

  EXPECT_EQ(graph_node.additional_add_info[node1->GetFastNode()].size(), 0);
  EXPECT_EQ(graph_node.additional_add_info[node2Ptr].size(), 1);
  EXPECT_EQ(graph_node.additional_add_info[node2Ptr][0], node3Ptr);
  EXPECT_EQ(graph_node.additional_add_info[node3Ptr].size(), 0);
  EXPECT_EQ(graph_node.additional_add_info[node4Ptr].size(), 0);
  EXPECT_EQ(graph_node.additional_add_info[node5->GetFastNode()].size(), 0);
}

/*
 * main graph:
 * NetOutput
 *     |
 *   node5(6)
 *     |    \
 *    if(3)  node4(5)
 *     |        |
 *   node1(1)  node2(2)
 *
 * then graph:
 *      InnerNetOutput
 *           |
 *        node3(4)
 *           |
 *        InnerData
 *
 * else graph:
 *      InnerNetOutput
 *              |
 *          InnerData
 */
// 测试普通launch节点和If连接需要增加的控制关系
TEST_F(GraphNodeUT, EnsureNodeExeInOrder_Prioirty_With_Subgraph2) {
  auto node1 = bg::ValueHolder::CreateSingleDataOutput("LaunchKernelWithFlag", {});
  ge::AttrUtils::SetInt(bg::ValueHolderUtils::GetNodeOpDescBarePtr(node1), "priority", 1);
  auto node2 = bg::ValueHolder::CreateSingleDataOutput("LaunchKernelWithFlag", {});
  ge::AttrUtils::SetInt(bg::ValueHolderUtils::GetNodeOpDescBarePtr(node2), "priority", 2);
  auto if_outputs = bg::If<bg::ValueHolder>(
      node1,
      [&]() -> std::vector<bg::ValueHolderPtr> {
        auto node3 = bg::ValueHolder::CreateSingleDataOutput("LaunchKernelWithFlag", {node1});
        ge::AttrUtils::SetInt(bg::ValueHolderUtils::GetNodeOpDescBarePtr(node3), "priority", 4);
        return {node3};
      },
      [&]() -> std::vector<bg::ValueHolderPtr> {
        return {node1};
      });
  ge::AttrUtils::SetInt(bg::ValueHolderUtils::GetNodeOpDescBarePtr(if_outputs[0]), "priority", 3);
  auto node4 = bg::ValueHolder::CreateSingleDataOutput("LaunchKernelWithFlag", {node2});
  ge::AttrUtils::SetInt(bg::ValueHolderUtils::GetNodeOpDescBarePtr(node4), "priority", 5);
  auto node5 = bg::ValueHolder::CreateSingleDataOutput("LaunchKernelWithFlag", {if_outputs[0], node4});
  ge::AttrUtils::SetInt(bg::ValueHolderUtils::GetNodeOpDescBarePtr(node5), "priority", 6);
  auto main_frame = bg::ValueHolder::PopGraphFrame({node5}, {}, "NetOutput");

  GraphNode graph_node;
  EXPECT_EQ(graph_node.EnsureNodeExeInOrder(main_frame->GetExecuteGraph().get()), ge::GRAPH_SUCCESS);
  auto if_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(main_frame->GetExecuteGraph().get(), "If");
  ASSERT_NE(if_node, nullptr);
  auto cond_graph = ge::FastNodeUtils::GetSubgraphFromNode(if_node, 0U);
  ASSERT_NE(cond_graph, nullptr);
  auto then_graph = ge::FastNodeUtils::GetSubgraphFromNode(if_node, 1U);
  ASSERT_NE(then_graph, nullptr);
  auto else_graph = ge::FastNodeUtils::GetSubgraphFromNode(if_node, 2U);
  ASSERT_NE(else_graph, nullptr);
  auto node3Ptr = ge::ExecuteGraphUtils::FindFirstNodeMatchType(then_graph, "LaunchKernelWithFlag");
  ASSERT_NE(node3Ptr, nullptr);
  auto switchNotify = ge::ExecuteGraphUtils::FindFirstNodeMatchType(cond_graph, "SwitchNotify");
  ASSERT_NE(switchNotify, nullptr);
  auto waitAnyone = ge::ExecuteGraphUtils::FindFirstNodeMatchType(cond_graph, "WaitAnyone");
  ASSERT_NE(waitAnyone, nullptr);

  EXPECT_EQ(graph_node.additional_indegree_info[node1->GetFastNode()], 0);
  EXPECT_EQ(graph_node.additional_indegree_info[node2->GetFastNode()], 1);
  EXPECT_EQ(graph_node.additional_indegree_info[node3Ptr], 0);
  EXPECT_EQ(graph_node.additional_indegree_info[switchNotify], 1);
  EXPECT_EQ(graph_node.additional_indegree_info[node4->GetFastNode()], 1);
  EXPECT_EQ(graph_node.additional_indegree_info[node5->GetFastNode()], 0);

  EXPECT_EQ(graph_node.additional_add_info[node1->GetFastNode()].size(), 1);
  EXPECT_EQ(graph_node.additional_add_info[node1->GetFastNode()][0], node2->GetFastNode());
  EXPECT_EQ(graph_node.additional_add_info[node2->GetFastNode()].size(), 1);
  EXPECT_EQ(graph_node.additional_add_info[node2->GetFastNode()][0], switchNotify);
  EXPECT_EQ(graph_node.additional_add_info[node3Ptr].size(), 0);
  EXPECT_EQ(graph_node.additional_add_info[waitAnyone].size(), 1);
  EXPECT_EQ(graph_node.additional_add_info[waitAnyone][0], node4->GetFastNode());
  EXPECT_EQ(graph_node.additional_add_info[node4->GetFastNode()].size(), 0);
  EXPECT_EQ(graph_node.additional_add_info[node5->GetFastNode()].size(), 0);
}
}

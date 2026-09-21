/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "lowering/node_priority_calculator.h"
#include <gtest/gtest.h>
#include "common/bg_test.h"
#include "common/share_graph.h"
#include "common/exe_graph.h"
#include "faker/global_data_faker.h"

#include <memory>
#include "core/builder/node_types.h"
#include "lowering/graph_converter.h"
#include "exe_graph/lowering/value_holder.h"
#include "graph/utils/graph_utils.h"
#include "common/checker.h"
#include "graph_builder/bg_condition.h"
#include "faker/model_desc_holder_faker.h"
#include "graph/utils/execute_graph_adapter.h"
#include "graph/utils/graph_dump_utils.h"

namespace gert {
namespace {
/*
 * compute graph:
 *       node3
 *         |
 *       node2
 *         |
 *       node1
 *
 * execute graph:
 *    +-------> FreeMemory1       FreeMemory2         FreeMemory3
 *    |         /        \      c/    |      \       c/    |
 *    |       c/   c      \c    / c   |       \c     /     |
 *    |   Launch1 -----> Launch2  ----|-----> Launch3      |
 *    |    /             |    |       |       |    |       |
 * AllocMemory1 ---------+   AllocMemory2 ----+   AllocMemory3
 */
std::unique_ptr<bg::GraphFrame> BuildGraph1() {
  bg::ValueHolder::PushGraphFrame();
  auto size = bg::ValueHolder::CreateFeed(0);
  auto allocator = bg::ValueHolder::CreateFeed(1);
  auto stream = bg::ValueHolder::CreateFeed(-1);

  auto alloc1 = bg::ValueHolder::CreateSingleDataOutput("AllocMemory", {allocator, size});
  bg::ValueHolder::CreateVoidGuarder("FreeMemory", alloc1, {});
  auto launch1 = bg::ValueHolder::CreateVoid<bg::ValueHolder>("LaunchKernelWithFlag", {stream, alloc1});

  auto alloc2 = bg::ValueHolder::CreateSingleDataOutput("AllocMemory", {allocator, size});
  bg::ValueHolder::CreateVoidGuarder("FreeMemory", alloc2, {});
  auto launch2 = bg::ValueHolder::CreateVoid<bg::ValueHolder>("LaunchKernelWithFlag", {stream, alloc1, alloc2});

  auto alloc3 = bg::ValueHolder::CreateSingleDataOutput("AllocMemory", {allocator, size});
  bg::ValueHolder::CreateVoidGuarder("FreeMemory", alloc3, {});
  auto launch3 = bg::ValueHolder::CreateVoid<bg::ValueHolder>("LaunchKernelWithFlag", {stream, alloc2, alloc3});

  bg::ValueHolder::AddDependency(launch1, launch2);
  bg::ValueHolder::AddDependency(launch2, launch3);

  auto frame = bg::ValueHolder::PopGraphFrame();
  GE_ASSERT_NOTNULL(frame);
  auto graph = frame->GetExecuteGraph();
  GE_ASSERT_NOTNULL(graph);
  GE_ASSERT_SUCCESS(graph->TopologicalSorting());
  ge::DumpGraph(graph.get(), "TestPriority1");

  return frame;
}
/*
 *        node5
 *      /   |   \
 * node2  node3  node4
 *      \   |   /
 *        node1
 *
 *
 *
 *   FreeMemory1              FreeMemory2    FreeMemory3     FreeMemory4
 *   |     |  |                | |  |           | |  |           |    |
 *   |     |  |                | +--|-----------| +--|-----------|----|---------+
 *   |     |  +----------------+----|-----------+----|-----------+    |         |
 *   |    c|                  c|    |          c|    |          c|    |         |
 *   |   Launch1           Launch2  |       Launch3  |       Launch4  |         | FreeMemory5
 *   |     |                |   |   |        |   |   |        |   |   |        c| c|   |
 *   |     |                |   |   +--------|---|---+--------|---|---+-----> Launch5  |
 *   |     |                |   |   |        |   |   |        |   |   |           |    |
 *   |     |                | AllocMemory2   | AllocMemory3   | AllocMemory4   AllocMemory5
 *   |     |                |                |                |
 * AllocMemory1 ------------+----------------+----------------+
 */
std::unique_ptr<bg::GraphFrame> BuildGraph2() {
  bg::ValueHolder::PushGraphFrame();
  auto stream = bg::ValueHolder::CreateFeed(-1);
  auto size = bg::ValueHolder::CreateFeed(0);
  auto allocator = bg::ValueHolder::CreateFeed(1);

  auto alloc1 = bg::ValueHolder::CreateSingleDataOutput("AllocMemory", {allocator, size});
  bg::ValueHolder::CreateVoidGuarder("FreeMemory", alloc1, {});
  auto launch1 = bg::ValueHolder::CreateVoid<bg::ValueHolder>("LaunchKernelWithFlag", {stream, alloc1});

  auto alloc2 = bg::ValueHolder::CreateSingleDataOutput("AllocMemory", {allocator, size});
  bg::ValueHolder::CreateVoidGuarder("FreeMemory", alloc2, {});
  auto launch2 = bg::ValueHolder::CreateVoid<bg::ValueHolder>("LaunchKernelWithFlag", {stream, alloc1, alloc2});
  bg::ValueHolder::AddDependency(launch1, launch2);

  auto alloc3 = bg::ValueHolder::CreateSingleDataOutput("AllocMemory", {allocator, size});
  bg::ValueHolder::CreateVoidGuarder("FreeMemory", alloc3, {});
  auto launch3 = bg::ValueHolder::CreateVoid<bg::ValueHolder>("LaunchKernelWithFlag", {stream, alloc1, alloc3});
  bg::ValueHolder::AddDependency(launch1, launch3);

  auto alloc4 = bg::ValueHolder::CreateSingleDataOutput("AllocMemory", {allocator, size});
  bg::ValueHolder::CreateVoidGuarder("FreeMemory", alloc4, {});
  auto launch4 = bg::ValueHolder::CreateVoid<bg::ValueHolder>("LaunchKernelWithFlag", {stream, alloc1, alloc4});
  bg::ValueHolder::AddDependency(launch1, launch4);

  auto alloc5 = bg::ValueHolder::CreateSingleDataOutput("AllocMemory", {allocator, size});
  bg::ValueHolder::CreateVoidGuarder("FreeMemory", alloc5, {});
  auto launch5 = bg::ValueHolder::CreateVoid<bg::ValueHolder>("LaunchKernelWithFlag", {stream, alloc2, alloc3, alloc4, alloc5});
  bg::ValueHolder::AddDependency(launch2, launch5);
  bg::ValueHolder::AddDependency(launch3, launch5);
  bg::ValueHolder::AddDependency(launch4, launch5);

  auto frame = bg::ValueHolder::PopGraphFrame();
  GE_ASSERT_NOTNULL(frame);
  auto graph = frame->GetExecuteGraph();
  GE_ASSERT_NOTNULL(graph);
  GE_ASSERT_SUCCESS(graph->TopologicalSorting());
  ge::DumpGraph(graph.get(), "TestPriority2");

  return frame;
}
/*
 *
 *    +----------> FreeMemory1       FreeMemory2         FreeMemory3
 *    |           |  /      \         /   |     \          /  |   |
 *    |    +------+ /        \      c/    |      \       c/   |   |
 *    |   c|      c/   c      \c    / c   |       \c     /    |   |c
 *    |  Foo  Launch1 -----> Launch2  ----|-----> Launch3     |  Bar
 *    |   |    /             |    |       |       |    |      |   /
 * AllocMemory1 -------------+   AllocMemory2 ----+   AllocMemory3
 */
std::unique_ptr<bg::GraphFrame> BuildGraph3() {
  bg::ValueHolder::PushGraphFrame();
  auto size = bg::ValueHolder::CreateFeed(0);
  auto allocator = bg::ValueHolder::CreateFeed(1);
  auto stream = bg::ValueHolder::CreateFeed(-1);

  auto alloc1 = bg::ValueHolder::CreateSingleDataOutput("AllocMemory", {allocator, size});
  bg::ValueHolder::CreateVoidGuarder("FreeMemory", alloc1, {});
  bg::ValueHolder::CreateVoid<bg::ValueHolder>("Foo", {alloc1});
  auto launch1 = bg::ValueHolder::CreateVoid<bg::ValueHolder>("LaunchKernelWithFlag", {stream, alloc1});

  auto alloc2 = bg::ValueHolder::CreateSingleDataOutput("AllocMemory", {allocator, size});
  bg::ValueHolder::CreateVoidGuarder("FreeMemory", alloc2, {});
  auto launch2 = bg::ValueHolder::CreateVoid<bg::ValueHolder>("LaunchKernelWithFlag", {stream, alloc1, alloc2});

  auto alloc3 = bg::ValueHolder::CreateSingleDataOutput("AllocMemory", {allocator, size});
  bg::ValueHolder::CreateVoidGuarder("FreeMemory", alloc3, {});
  bg::ValueHolder::CreateVoid<bg::ValueHolder>("Bar", {alloc3});
  auto launch3 = bg::ValueHolder::CreateVoid<bg::ValueHolder>("LaunchKernelWithFlag", {stream, alloc2, alloc3});

  bg::ValueHolder::AddDependency(launch1, launch2);
  bg::ValueHolder::AddDependency(launch2, launch3);

  auto frame = bg::ValueHolder::PopGraphFrame();
  GE_ASSERT_NOTNULL(frame);
  auto graph = frame->GetExecuteGraph();
  GE_ASSERT_NOTNULL(graph);
  GE_ASSERT_SUCCESS(graph->TopologicalSorting());
  ge::DumpGraph(graph.get(), "TestPriority1");

  return frame;
}

/*
 *
 * FreeMemory1         FreeMemory2
 *    |   |c           |     |c
 *    | Launch1        |   Launch2
 *    |   |   \c       |     |
 *    |   |    If      |     |
 *    |   |    |       |     |
 *  AllocMemory1       AllocMemory2
 */
std::unique_ptr<bg::GraphFrame> BuildGraph4() {
  bg::ValueHolder::PushGraphFrame();
  auto size = bg::ValueHolder::CreateFeed(0);
  auto allocator = bg::ValueHolder::CreateFeed(1);
  auto stream = bg::ValueHolder::CreateFeed(-1);

  auto alloc1 = bg::ValueHolder::CreateSingleDataOutput("AllocMemory", {allocator, size});
  bg::ValueHolder::CreateVoidGuarder("FreeMemory", alloc1, {});
  auto launch1 = bg::ValueHolder::CreateVoid<bg::ValueHolder>("LaunchKernelWithFlag", {stream, alloc1});

  auto if_ret = bg::If<bg::ValueHolder>(
      bg::ValueHolder::CreateFeed(2),
      [&]() -> std::vector<bg::ValueHolderPtr> {
        bg::ValueHolder::CreateVoid<bg::ValueHolder>("LaunchKernelWithFlag", {stream, alloc1});
        return {};
      },
      [&]() -> std::vector<bg::ValueHolderPtr> {
        bg::ValueHolder::CreateVoid<bg::ValueHolder>("LaunchKernelWithFlag", {stream, alloc1});
        return {};
      });

  bg::ValueHolder::AddDependency(if_ret[0], launch1);

  auto alloc2 = bg::ValueHolder::CreateSingleDataOutput("AllocMemory", {allocator, size});
  bg::ValueHolder::CreateVoidGuarder("FreeMemory", alloc2, {});
  auto launch2 = bg::ValueHolder::CreateVoid<bg::ValueHolder>("LaunchKernelWithFlag", {stream, alloc2});

  auto frame = bg::ValueHolder::PopGraphFrame();
  GE_ASSERT_NOTNULL(frame);
  auto graph = frame->GetExecuteGraph();
  GE_ASSERT_NOTNULL(graph);
  GE_ASSERT_SUCCESS(graph->TopologicalSorting());
  ge::DumpGraph(graph.get(), "TestPriority1");

  return frame;
}
}  // namespace
class NodePriorityCalculatorUT : public bg::BgTest {
 protected:
  /** 校验没有滞后释放，当前本条不总是成立，如下，Launch1、Launch2分别拥有自己的Alloc，FreeMemory的优先级和AllocMemory2相等
   *       AllocMemory1               AllocMemory2
   *          |  \                      /      |
   *          |  Launch1            Launch2    |
   *          |     \ c               /c       |
   *          |    EnsureTensorAtOutMemory     |
   *          |        /c                    FreeMemory2
   *         FreeMemory1
   * */
  void CheckNoFreeLaggingBehind(const ge::ExecuteGraph *const graph) {
    InitLaunchNodesPrioritiesIfNeed(graph);

    for (const auto free_node : graph->GetAllNodes()) {
      if (!IsFreeNode(free_node->GetTypePtr())) {
        continue;
      }
      // 找到本此Free所依赖的、优先级最低的那个launch
      std::set<int64_t> all_depend_launch_node_priorities;
      for (const auto depend_node : free_node->GetAllInNodes()) {
        if (IsLaunchNode(depend_node->GetTypePtr())) {
          all_depend_launch_node_priorities.insert(GetPriority(depend_node));
        }
      }
      ASSERT_FALSE(all_depend_launch_node_priorities.empty());
      auto lowest_depend_priority = *all_depend_launch_node_priorities.rbegin();

      // 找到最低优先级更低的一个launch（free的执行次序应该在更低launch对应的alloc的前面）
      auto iter = GetAllPrioritiesToLaunchNodes().find(lowest_depend_priority);
      ASSERT_NE(iter, GetAllPrioritiesToLaunchNodes().end());
      if (++iter == GetAllPrioritiesToLaunchNodes().end()) {
        continue;
      }

      for (const auto next_launch_node : iter->second) {
        for (const auto alloc_node : next_launch_node->GetAllInNodes()) {
          if (IsAllocNode(alloc_node->GetTypePtr())) {
            std::set<int64_t> all_use_launch_nodes_priorities;
            for (const auto use_launch_node : alloc_node->GetAllOutNodes()) {
              if (IsLaunchNode(use_launch_node->GetTypePtr())) {
                all_use_launch_nodes_priorities.insert(GetPriority(use_launch_node));
              }
            }
            if (GetPriority(next_launch_node) != *all_use_launch_nodes_priorities.begin()) {
              // 如果一个alloc node被多个launch node使用，并且其中一个launch node比`next_launch_node`优先级高，
              // 这说明这个alloc_node本就应该更早地被申请，因此不参与判断
              continue;
            }
            EXPECT_LT(GetPriority(free_node), GetPriority(alloc_node))
                << "The free node " << free_node->GetName() << " may execute late than alloc node "
                << alloc_node->GetName();
          }
        }
      }
    }
  }

  // Free节点自底向上单独标注，优先级（数值）应该取所有祖先中的最大值
  void CheckFreeNodeLowestPriority(const ge::ExecuteGraph *const graph) {
    for (const auto free_node : graph->GetAllNodes()) {
      if (!IsFreeNode(free_node->GetTypePtr())) {
        continue;
      }
      std::queue<ge::FastNode *> nodes;
      std::set<ge::FastNode *> seen;
      nodes.push(free_node);
      int64_t max_priority = 0;
      while (!nodes.empty()) {
        auto curr_node = nodes.front();
        nodes.pop();
        for (const auto node : curr_node->GetAllInNodes()) {
          if (seen.find(node) == seen.end()) {
            max_priority = std::max(max_priority, GetPriority(node));
            nodes.push(node);
            seen.insert(curr_node);
          }
        }
      }
      EXPECT_EQ(GetPriority(free_node), max_priority)
          << "The free node " << free_node->GetName() << " has a priority " << GetPriority(free_node)
          << "that is not equal to the maximum priority of all its ancestor nodes.";
    }
  }

  // 没有超前申请
  void CheckNoAllocAdvanced(const ge::ExecuteGraph *const graph) {
    InitLaunchNodesPrioritiesIfNeed(graph);

    for (const auto alloc_node : graph->GetAllNodes()) {
      if (!IsAllocNode(alloc_node->GetTypePtr())) {
        continue;
      }
      auto alloc_priority = GetPriority(alloc_node);
      std::map<int64_t, std::vector<ge::FastNode *>> all_priorities_to_use_launch_nodes;
      for (const auto use_launch_node : alloc_node->GetAllOutNodes()) {
        if (IsLaunchNode(use_launch_node->GetTypePtr()) ||
            !use_launch_node->GetOpDescBarePtr()->GetSubgraphInstanceNames().empty()) {
          all_priorities_to_use_launch_nodes[GetPriority(use_launch_node)].push_back(use_launch_node);
        }
      }
      ASSERT_FALSE(all_priorities_to_use_launch_nodes.empty());
      auto highest_use_priority = all_priorities_to_use_launch_nodes.begin()->first;
      ASSERT_LE(alloc_priority, highest_use_priority)
          << "The alloc node " << alloc_node->GetName() << ", priority " << alloc_priority
          << "not less or equal than the highest launch node. Launch node "
          << all_priorities_to_use_launch_nodes.begin()->second.at(0)->GetName() << ", priority "
          << highest_use_priority;
      auto iter = GetAllPrioritiesToLaunchNodes().find(highest_use_priority);
      ASSERT_NE(iter, GetAllPrioritiesToLaunchNodes().end());
      if (iter != GetAllPrioritiesToLaunchNodes().begin()) {
        --iter;
        EXPECT_GE(alloc_priority, iter->first)
            << "The alloc node " << alloc_node->GetName() << " may execute advanced. Alloc node priority "
            << alloc_priority << ", which is higher than no-use-launch nodes priority " << iter->first
            << ", like launch node " << iter->second.at(0);
      }
    }
  }
  void CheckAllNodesHasPriority(const ge::ExecuteGraph *const graph) {
    for (const auto node : graph->GetAllNodes()) {
      if (node->GetAllOutNodes().empty()) {
        continue;
      }
      EXPECT_NE(GetPriority(node), std::numeric_limits<int64_t>::max())
          << "The node " << node->GetName() << " does not have a priority";
    }
  }

  int64_t GetPriority(const ge::FastNode *const node) {
    int64_t priority = std::numeric_limits<int64_t>::max();
    ge::AttrUtils::GetInt(node->GetOpDescBarePtr(), "priority", priority);
    return priority;
  }

  const std::map<int64_t, std::vector<ge::FastNode *>> &GetAllPrioritiesToLaunchNodes() const {
    return *all_priorities_to_launch_nodes_;
  }

 private:
  void TearDown() override {
    all_priorities_to_launch_nodes_ = nullptr;
  }

  void InitLaunchNodesPrioritiesIfNeed(const ge::ExecuteGraph *const graph) {
    if (all_priorities_to_launch_nodes_ != nullptr) {
      return;
    }
    all_priorities_to_launch_nodes_ = std::make_unique<std::map<int64_t, std::vector<ge::FastNode *>>>();
    for (const auto node : graph->GetAllNodes()) {
      if (IsLaunchNode(node->GetTypePtr()) || !node->GetOpDescBarePtr()->GetSubgraphInstanceNames().empty()) {
        (*all_priorities_to_launch_nodes_)[GetPriority(node)].push_back(node);
      }
    }
  }

  std::unique_ptr<std::map<int64_t, std::vector<ge::FastNode *>>> all_priorities_to_launch_nodes_ = nullptr;
};
TEST_F(NodePriorityCalculatorUT, CalcPriority_AllocFreePriorityCorrrect_ThreeLaunch) {
  auto frame = BuildGraph1();
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);
  const auto all_nodes = graph->GetAllNodes();
  ASSERT_EQ(bg::NodePriorityCalculator(*frame).CalcNodeExecutionPriorities(all_nodes, all_nodes.size()),
            ge::GRAPH_SUCCESS);
  CheckNoAllocAdvanced(graph.get());
  CheckNoFreeLaggingBehind(graph.get());
  CheckFreeNodeLowestPriority(graph.get());
  CheckAllNodesHasPriority(graph.get());
}
TEST_F(NodePriorityCalculatorUT, CalcPriority_PriorityCorrect_ThreeLaunch) {
  auto frame = BuildGraph1();
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);
  const auto all_nodes = graph->GetAllNodes();
  ASSERT_EQ(bg::NodePriorityCalculator(*frame).CalcNodeExecutionPriorities(all_nodes, all_nodes.size()),
            ge::GRAPH_SUCCESS);

  // kernel launch nodes should have the different priorities
  std::map<int64_t, ge::FastNode *> priorities_to_launch_node;
  for (const auto node : graph->GetAllNodes()) {
    if (IsLaunchNode(node->GetTypePtr())) {
      auto priority = GetPriority(node);
      ASSERT_EQ(priorities_to_launch_node.count(priority), 0)
          << "The launch node " << node->GetName() << " has the same priority with "
          << priorities_to_launch_node[priority]->GetName() << ", Priority " << priority;
      priorities_to_launch_node[priority] = node;
    }
  }

  auto data_nodes = ge::ExecuteGraphUtils::FindNodesByTypeFromAllNodes(graph.get(), "Data");
  for (const auto data_node : data_nodes) {
    ASSERT_EQ(GetPriority(data_node), 0);
  }

  // alloc free general check
  CheckNoAllocAdvanced(graph.get());
  CheckNoFreeLaggingBehind(graph.get());
  CheckFreeNodeLowestPriority(graph.get());
  CheckAllNodesHasPriority(graph.get());
}
TEST_F(NodePriorityCalculatorUT, CalcPriority_ForcePriorityCorrect_ThreeLaunchDoesNotHaveEdge) {
  auto frame = BuildGraph2();
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);
  const auto all_nodes = graph->GetAllNodes();
  ASSERT_EQ(bg::NodePriorityCalculator(*frame).CalcNodeExecutionPriorities(all_nodes, all_nodes.size()),
            ge::GRAPH_SUCCESS);
  CheckNoAllocAdvanced(graph.get());
  CheckNoFreeLaggingBehind(graph.get());
  CheckFreeNodeLowestPriority(graph.get());
  CheckAllNodesHasPriority(graph.get());
}
TEST_F(NodePriorityCalculatorUT, CalcPriority_FreeNodeDependentsPriorityCorrect) {
  auto frame = BuildGraph3();
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);
  const auto all_nodes = graph->GetAllNodes();
  ASSERT_EQ(bg::NodePriorityCalculator(*frame).CalcNodeExecutionPriorities(all_nodes, all_nodes.size()),
            ge::GRAPH_SUCCESS);
  CheckNoAllocAdvanced(graph.get());
  CheckNoFreeLaggingBehind(graph.get());
  CheckFreeNodeLowestPriority(graph.get());
  CheckAllNodesHasPriority(graph.get());

  auto foo_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph.get(), "Foo");
  ASSERT_NE(foo_node, nullptr);
  auto bar_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph.get(), "Bar");
  ASSERT_NE(bar_node, nullptr);

  EXPECT_EQ(GetPriority(foo_node), 1);  // the free node after launch2 has priority 2
  EXPECT_EQ(GetPriority(bar_node), 2);  // the free node after launch3 has priority 3
}
TEST_F(NodePriorityCalculatorUT, CalcPriority_AllocFreePriorityCorrrect_WhenThereIsSubgraph) {
  auto frame = BuildGraph4();
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);
  const auto all_nodes = graph->GetAllNodes();
  ASSERT_EQ(bg::NodePriorityCalculator(*frame).CalcNodeExecutionPriorities(all_nodes, all_nodes.size()),
            ge::GRAPH_SUCCESS);

  CheckNoAllocAdvanced(graph.get());
  CheckNoFreeLaggingBehind(graph.get());
  CheckFreeNodeLowestPriority(graph.get());
  CheckAllNodesHasPriority(graph.get());
}
TEST_F(NodePriorityCalculatorUT, CalcPriority_CtrlNodesPriorityCorrect_If) {
  auto frame = BuildGraph4();
  auto graph = frame->GetExecuteGraph();
  ASSERT_NE(graph, nullptr);
  const auto all_nodes = graph->GetAllNodes();
  ASSERT_EQ(bg::NodePriorityCalculator(*frame).CalcNodeExecutionPriorities(all_nodes, all_nodes.size()),
            ge::GRAPH_SUCCESS);
  auto if_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(graph.get(), "If");
  auto if_priority = GetPriority(if_node);
  int64_t max_priority = if_priority;
  const auto &sub_graph_indexes = if_node->GetOpDescBarePtr()->GetSubgraphNameIndexes();
  for (const auto &index: sub_graph_indexes) {
    const auto &sub_graph = ge::FastNodeUtils::GetSubgraphFromNode(if_node, index.second);
    for (const auto node : sub_graph->GetAllNodes()) {
      auto cur_priority = GetPriority(node);
      if (cur_priority > max_priority) {
        max_priority = cur_priority;
      }
    }
  }
  auto ctrl_graph = ge::FastNodeUtils::GetSubgraphFromNode(if_node, 0);
  for (const auto node : ctrl_graph->GetAllNodes()){
    const auto &node_type = node->GetTypePtr();
    if (IsSwitchNotifyNode(node_type) || IsBranchPivot(node_type)) {
      ASSERT_EQ(if_priority, GetPriority(node));
    } else if (IsBranchDone(node_type) || IsWaitAnyone(node_type)) {
      ASSERT_EQ(max_priority, GetPriority(node));
    }
  }

  CheckNoAllocAdvanced(graph.get());
  CheckNoFreeLaggingBehind(graph.get());
  CheckFreeNodeLowestPriority(graph.get());
  CheckAllNodesHasPriority(graph.get());
}
// TEST_F(NodePriorityCalculatorUT, CalcPriority_PriorityCorrect_While) {
//   // todo
// }

TEST_F(NodePriorityCalculatorUT, CalcPriority_Correct_LoweringFromLstmp) {
  auto graph = ShareGraph::LstmpGraph();
  graph->TopologicalSorting();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model)
                         .FakeWithoutHandleAiCore("TransData", true)
                         .FakeWithoutHandleAiCore("DynamicRNNV3", false)
                         .Build();

  ASSERT_NE(graph, nullptr);

  auto model_desc_holder = ModelDescHolderFaker().Build();
  auto exe_graph = GraphConverter()
      .SetModelDescHolder(&model_desc_holder)
      .ConvertComputeGraphToExecuteGraph(graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  exe_graph->TopologicalSorting();
  ge::DumpGraph(exe_graph.get(), "TestPriorityLstmp");
  bg::ExeGraph eg(exe_graph);

  CheckNoAllocAdvanced(exe_graph.get());
  CheckFreeNodeLowestPriority(exe_graph.get());
  CheckAllNodesHasPriority(eg.GetMainGraph().get());
}
}  // namespace gert

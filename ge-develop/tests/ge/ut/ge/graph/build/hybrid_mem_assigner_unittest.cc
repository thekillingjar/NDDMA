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
#include <gmock/gmock.h>
#include <vector>

#include "graph/build/memory/hybrid_mem_assigner.h"
#include "graph/build/memory/mem_inplace.h"
#include "common/share_graph.h"
#include "common/mem_conflict_share_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/tensor_utils.h"
#include "graph/node.h"

namespace ge {
namespace {
void SetOutReuseInput(ComputeGraphPtr &root_graph, const std::string &node_name) {
  for (auto &node : root_graph->GetAllNodes()) {
    if (node->GetName() == node_name) {
      (void)ge::AttrUtils::SetBool(node->GetOpDesc(), ATTR_NAME_OUTPUT_REUSE_INPUT, true);
      (void)ge::AttrUtils::SetInt(node->GetOpDesc(), ge::ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);

      // 为了输出能和输入同符号
      auto output_tensor = node->GetOpDesc()->MutableOutputDesc(0);
      if (output_tensor != nullptr) {
        TensorUtils::SetReuseInput(*output_tensor, true);
        TensorUtils::SetReuseInputIndex(*output_tensor, 0U);
      }
    }
  }
}
}
class UtestHybridMemAssigner : public testing::Test {
 protected:
  void SetUp() {
  }
  void TearDown() {
  }
};

TEST_F(UtestHybridMemAssigner, GetAnchorToSymbol_Success) {
  auto graph = gert::ShareGraph::AicoreGraph();
  graph->TopologicalSorting();
  HybridMemAssigner hybridMemAssigner(graph);
  (void)hybridMemAssigner.Assign();
  ASSERT_FALSE(hybridMemAssigner.GetAnchorToSymbol().empty());
  ASSERT_FALSE(hybridMemAssigner.GetSymbolToAnchors().empty());
}

TEST_F(UtestHybridMemAssigner, ProcessInplace_input_data_invalid) {
  // 输入是data节点，不能inplace
  auto graph = gert::ShareGraph::SimpleFooGraph();
  auto node = graph->FindNode("foo");
  auto op_desc = node->GetOpDesc();
  AttrUtils::SetListListInt(op_desc, ATTR_NAME_OUTPUT_INPLACE_ABILITY, {{0, 0}});
  op_desc->AppendIrInput("x", kIrInputRequired);
  op_desc->AppendIrOutput("z", kIrOutputRequired);

  graph->TopologicalSorting();
  HybridMemAssigner hybridMemAssigner(graph);
  (void)hybridMemAssigner.Assign();
  ASSERT_TRUE(hybridMemAssigner.GetAnchorToSymbol().size() == 4U);
  ASSERT_TRUE(hybridMemAssigner.GetSymbolToAnchors().size() == 2U);
}

TEST_F(UtestHybridMemAssigner, ProcessInplace_one_invalid_input_index) {
  auto graph = gert::ShareGraph::BuildAddConditionCalcGraph();
  graph->TopologicalSorting();
  HybridMemAssigner hybridMemAssigner_before(graph);
  ASSERT_TRUE(hybridMemAssigner_before.Assign() == SUCCESS);
  auto before_inplace_size = hybridMemAssigner_before.GetSymbolToAnchors().size();

  auto node = graph->FindNode("condition_calc");
  auto op_desc = node->GetOpDesc();
  AttrUtils::SetListListInt(op_desc, ATTR_NAME_OUTPUT_INPLACE_ABILITY, {{0, 1}});
  op_desc->AppendIrInput("x", kIrInputRequired);
  op_desc->AppendIrOutput("z", kIrOutputRequired);
  auto tensor_desc = op_desc->GetOutputDesc(0);
  bool reuse_flag;
  (void)TensorUtils::GetReuseInput(tensor_desc, reuse_flag);
  ASSERT_TRUE(!reuse_flag);

  const NodeIndexIO inplace_info(node, 0, kOut);
  auto add1_node = graph->FindNode("add1");
  const NodeIndexIO data_info(add1_node, 0, kOut);

  HybridMemAssigner hybridMemAssigner(graph);
  // metadef仓上库之后放开
  // ASSERT_FALSE(hybridMemAssigner.Assign() == SUCCESS);
  // 符号表合并之后会少一个符号
  (void)hybridMemAssigner.Assign();
  ASSERT_TRUE(hybridMemAssigner.GetSymbolToAnchors().size() == before_inplace_size);
}

TEST_F(UtestHybridMemAssigner, ProcessInplace_continous_output_not_inplace) {
  auto graph = gert::ShareGraph::BuildAddConditionCalcGraph();
  graph->TopologicalSorting();
  HybridMemAssigner hybridMemAssigner(graph);

  auto node = graph->FindNode("condition_calc");
  auto op_desc = node->GetOpDesc();
  AttrUtils::SetListListInt(op_desc, ATTR_NAME_OUTPUT_INPLACE_ABILITY, {{0, 0}});
  AttrUtils::SetBool(op_desc, ATTR_NAME_CONTINUOUS_OUTPUT, true);
  op_desc->AppendIrInput("x", kIrInputRequired);
  op_desc->AppendIrOutput("z", kIrOutputRequired);
  ASSERT_TRUE(hybridMemAssigner.Assign() == SUCCESS);
  bool reuse_flag;
  (void)TensorUtils::GetReuseInput(op_desc->GetOutputDesc(0), reuse_flag);
  ASSERT_FALSE(reuse_flag);
}

TEST_F(UtestHybridMemAssigner, ProcessInplace_no_padding_continous_output_not_inplace) {
  auto graph = gert::ShareGraph::BuildAddConditionCalcGraph();
  graph->TopologicalSorting();
  HybridMemAssigner hybridMemAssigner(graph);

  auto node = graph->FindNode("condition_calc");
  auto op_desc = node->GetOpDesc();
  AttrUtils::SetListListInt(op_desc, ATTR_NAME_OUTPUT_INPLACE_ABILITY, {{0, 0}});
  AttrUtils::SetBool(op_desc, ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT, true);
  op_desc->AppendIrInput("x", kIrInputRequired);
  op_desc->AppendIrOutput("z", kIrOutputRequired);
  ASSERT_TRUE(hybridMemAssigner.Assign() == SUCCESS);
  bool reuse_flag;
  (void)TensorUtils::GetReuseInput(op_desc->GetOutputDesc(0), reuse_flag);
  ASSERT_FALSE(reuse_flag);
}

TEST_F(UtestHybridMemAssigner, ProcessInplace_output_variable_not_inplace) {
  auto graph = gert::ShareGraph::BuildAddConditionCalcGraph();
  graph->TopologicalSorting();
  HybridMemAssigner hybridMemAssigner(graph);

  auto node = graph->FindNode("condition_calc");
  auto op_desc = node->GetOpDesc();
  AttrUtils::SetListListInt(op_desc, ATTR_NAME_OUTPUT_INPLACE_ABILITY, {{0, 0}});
  auto tensor_desc = op_desc->MutableOutputDesc(0);
  AttrUtils::SetStr(tensor_desc, REF_VAR_SRC_VAR_NAME, "variable_0");
  op_desc->AppendIrInput("x", kIrInputRequired);
  op_desc->AppendIrOutput("z", kIrOutputRequired);
  ASSERT_TRUE(hybridMemAssigner.Assign() == SUCCESS);
  bool reuse_flag;
  (void)TensorUtils::GetReuseInput(op_desc->GetOutputDesc(0), reuse_flag);
  ASSERT_FALSE(reuse_flag);
}

TEST_F(UtestHybridMemAssigner, ProcessInplace_invalid_tensor_size) {
  auto graph = gert::ShareGraph::BuildAddConditionCalcGraph();
  graph->TopologicalSorting();
  HybridMemAssigner hybridMemAssigner_before(graph);
  ASSERT_TRUE(hybridMemAssigner_before.Assign() == SUCCESS);
  auto before_inplace_size = hybridMemAssigner_before.GetSymbolToAnchors().size();

  auto node = graph->FindNode("condition_calc");
  auto op_desc = node->GetOpDesc();
  AttrUtils::SetListListInt(op_desc, ATTR_NAME_OUTPUT_INPLACE_ABILITY, {{0, 0}});
  op_desc->AppendIrInput("x", kIrInputRequired);
  op_desc->AppendIrOutput("z", kIrOutputRequired);
  auto tensor_desc = op_desc->MutableOutputDesc(0);
  tensor_desc->SetShape(GeShape({1, 2, 3}));
  tensor_desc->SetFormat(FORMAT_ND);

  HybridMemAssigner hybridMemAssigner(graph);
  ASSERT_TRUE(hybridMemAssigner.Assign() == SUCCESS);

  // 符号表合并之后会少一个符号
  ASSERT_TRUE(hybridMemAssigner.GetSymbolToAnchors().size() == before_inplace_size);
}

TEST_F(UtestHybridMemAssigner, ProcessInplace_one_input_Success) {
  auto graph = gert::ShareGraph::BuildAddConditionCalcGraph();
  graph->TopologicalSorting();
  HybridMemAssigner hybridMemAssigner_before(graph);
  ASSERT_TRUE(hybridMemAssigner_before.Assign() == SUCCESS);
  auto before_inplace_size = hybridMemAssigner_before.GetSymbolToAnchors().size();

  auto node = graph->FindNode("condition_calc");
  auto op_desc = node->GetOpDesc();
  AttrUtils::SetListListInt(op_desc, ATTR_NAME_OUTPUT_INPLACE_ABILITY, {{0, 0}});
  op_desc->AppendIrInput("x", kIrInputRequired);
  op_desc->AppendIrOutput("z", kIrOutputRequired);
  auto tensor_desc = op_desc->GetOutputDesc(0);
  bool reuse_flag;
  (void)TensorUtils::GetReuseInput(tensor_desc, reuse_flag);
  ASSERT_TRUE(!reuse_flag);

  const NodeIndexIO inplace_info(node, 0, kOut);
  auto add1_node = graph->FindNode("add1");
  const NodeIndexIO data_info(add1_node, 0, kOut);

  HybridMemAssigner hybridMemAssigner(graph);
  auto symbol_to_anchors_before_inplace = hybridMemAssigner_before.GetSymbolToAnchors();
  auto anchors_before_inplace = symbol_to_anchors_before_inplace[data_info.ToString()];
  ASSERT_TRUE(hybridMemAssigner.Assign() == SUCCESS);
  auto symbol_to_anchors_after_inplace = hybridMemAssigner.GetSymbolToAnchors();
  auto anchors_after_inplace = symbol_to_anchors_after_inplace[data_info.ToString()];
  // 符号表合并之后会少一个符号
  ASSERT_TRUE(hybridMemAssigner.GetSymbolToAnchors().size() == before_inplace_size - 1);
  // inplace的符号已经被删除
  ASSERT_TRUE(hybridMemAssigner.GetSymbolToAnchors().find(inplace_info.ToString())
              == hybridMemAssigner.GetSymbolToAnchors().end());
  ASSERT_TRUE(hybridMemAssigner.GetSymbolToAnchors().find(data_info.ToString())
              != hybridMemAssigner.GetSymbolToAnchors().end());
  ASSERT_TRUE(anchors_after_inplace.size() == anchors_before_inplace.size() + 2U);
  (void)TensorUtils::GetReuseInput(op_desc->GetOutputDesc(0), reuse_flag);
  ASSERT_TRUE(reuse_flag);
  uint32_t idx = 0U;
  (void)TensorUtils::GetReuseInputIndex(op_desc->GetOutputDesc(0), idx);
  ASSERT_TRUE(idx == 0U); // 0表示复用第一个输入
}

/*
          (0,0)
  ┌──────────────────────────────────────┐
  │                                      ∨
┌───────┐  (0,0)   ┌────────┐  (0,1)   ┌──────┐  (0,0)   ┌───────────┐
│ data1 │ ───────> │  add1  │ ───────> │ add2 │ ───────> │ NetOutput │
└───────┘          └────────┘          └──────┘          └───────────┘
                     ∧
                     │ (0,1)
                     │
                   ┌────────┐
                   │ data2  │
                   └────────┘
* 标记add2的输出可以复用data1和add1的输出, 但是data1不可写，所以data1输出不能被复用，最终会复用add1的输出
*/
TEST_F(UtestHybridMemAssigner, ProcessInplace_two_nodes_Success) {
  auto graph = gert::ShareGraph::BuildStaticTwoAddNodeGraph();
  graph->TopologicalSorting();
  HybridMemAssigner hybridMemAssigner_before(graph);
  ASSERT_TRUE(hybridMemAssigner_before.Assign() == SUCCESS);
  auto before_inplace_size = hybridMemAssigner_before.GetSymbolToAnchors().size();

  auto node = graph->FindNode("add2");
  auto op_desc = node->GetOpDesc();
  AttrUtils::SetListListInt(op_desc, ATTR_NAME_OUTPUT_INPLACE_ABILITY, {{0, 0}, {0, 1}});
  op_desc->AppendIrInput("x", kIrInputRequired);
  op_desc->AppendIrInput("y", kIrInputRequired);
  op_desc->AppendIrOutput("z", kIrOutputRequired);
  bool reuse_flag;
  (void)TensorUtils::GetReuseInput(op_desc->GetOutputDesc(0), reuse_flag);
  ASSERT_TRUE(!reuse_flag);

  const NodeIndexIO inplace_info(node, 0, kOut);
  auto add1_node = graph->FindNode("add1");
  const NodeIndexIO data_info(add1_node, 0, kOut);

  auto symbol_to_anchors_before_inplace = hybridMemAssigner_before.GetSymbolToAnchors();
  auto anchors_before_inplace = symbol_to_anchors_before_inplace[data_info.ToString()];

  HybridMemAssigner hybridMemAssigner(graph);
  ASSERT_TRUE(hybridMemAssigner.Assign() == SUCCESS);
  auto symbol_to_anchors_after_inplace = hybridMemAssigner.GetSymbolToAnchors();
  auto anchors_after_inplace = symbol_to_anchors_after_inplace[data_info.ToString()];
  ASSERT_TRUE(hybridMemAssigner.GetSymbolToAnchors().size() == before_inplace_size - 1);
  ASSERT_TRUE(hybridMemAssigner.GetSymbolToAnchors().find(inplace_info.ToString())
              == hybridMemAssigner.GetSymbolToAnchors().end());
  ASSERT_TRUE(hybridMemAssigner.GetSymbolToAnchors().find(data_info.ToString())
              != hybridMemAssigner.GetSymbolToAnchors().end());
  ASSERT_TRUE(anchors_after_inplace.size() == anchors_before_inplace.size() + 2U);
  uint32_t idx = 0U;
  (void)TensorUtils::GetReuseInputIndex(op_desc->GetOutputDesc(0), idx);
  ASSERT_TRUE(idx == 1U); // 1表示复用第一个输入
}

/*
                                       ┌────────┐
                     ┌──────────────── │ data2  │
                     │                 └────────┘
                     │                   │
                     │ (0,2)             │ (0,1)
                     ∨                   ∨
┌───────┐  (0,0)   ┌────────┐  (1,0)   ┌────────┐  (0,1)   ┌───────────┐
│ data0 │ ───────> │        │ ───────> │  add0  │ ───────> │ NetOutput │
└───────┘          │        │          └────────┘          └───────────┘
                   │        │           (0,0)                ∧
                   │   mg   │ ───────────────────────────────┘
                   │        │
                   │        │  (0,1)   ┌────────┐
                   │        │ <─────── │ data1  │
                   └────────┘          └────────┘
* add0的输出可以复用mg和data2的输出, 但是mg的所有输出之中，add0的topoid不是最大，所以不能复用，data2的输出不可写，所以不能复用
* 最终是不会进行inplace
*/
TEST_F(UtestHybridMemAssigner, ProcessInplace_can_not_inplace) {
  auto graph = gert::ShareGraph::BuildStaticMinimumGradAndAddGraph();
  graph->TopologicalSorting();
  HybridMemAssigner hybridMemAssigner(graph);
  auto node = graph->FindNode("add0");
  auto op_desc = node->GetOpDesc();
  AttrUtils::SetListListInt(op_desc, ATTR_NAME_OUTPUT_INPLACE_ABILITY, {{0, 0}, {0, 1}});
  op_desc->AppendIrInput("x", kIrInputRequired);
  op_desc->AppendIrInput("y", kIrInputRequired);
  op_desc->AppendIrOutput("z", kIrOutputRequired);
  auto tensor_desc = op_desc->GetOutputDesc(0);
  ASSERT_TRUE(hybridMemAssigner.Assign() == SUCCESS);
  bool reuse_flag;
  (void)TensorUtils::GetReuseInput(tensor_desc, reuse_flag);
  ASSERT_TRUE(!reuse_flag);
}

TEST_F(UtestHybridMemAssigner, ProcessInplace_input_include_var) {
  auto graph = gert::ShareGraph::BuildVarConnectToSplit();
  graph->TopologicalSorting();
  HybridMemAssigner hybridMemAssigner(graph);
  auto node = graph->FindNode("split");
  auto op_desc = node->GetOpDesc();
  AttrUtils::SetListListInt(op_desc, ATTR_NAME_OUTPUT_INPLACE_ABILITY, {{0, 0}});
  op_desc->AppendIrInput("x", kIrInputRequired);
  op_desc->AppendIrOutput("z", kIrOutputRequired);
  ASSERT_TRUE(hybridMemAssigner.Assign() == SUCCESS);
  bool reuse_flag;
  (void)TensorUtils::GetReuseInput(op_desc->GetOutputDesc(0), reuse_flag);
  ASSERT_TRUE(!reuse_flag);
}

/*
┌──────┐  (0,0)   ┌──────┐  (0,0)   ┌──────┐  (0,0)   ┌───┐  (0,0)   ┌───────────┐
│ data │ ───────> │ swt1 │ ───────> │ swt2 │ ───────> │ a │ ───────> │ netoutput │
└──────┘          └──────┘          └──────┘          └───┘          └───────────┘
dsa1为ref，所以dsa2的输出可以复用dsa1的输出，但是dsa1的输出不可写（因为复用了data），所以不能复用
*/
TEST_F(UtestHybridMemAssigner, ProcessInplace_two_nodes_not_inplace) {
  auto graph = ge::MemConflictShareGraph::BuildUserInAndNotSupportRefreshOutByRefGraph2();
  graph->TopologicalSorting();
  SetOutReuseInput(graph, "swt1");
  HybridMemAssigner hybridMemAssigner(graph);
  auto node = graph->FindNode("swt2");
  auto op_desc = node->GetOpDesc();
  AttrUtils::SetListListInt(op_desc, ATTR_NAME_OUTPUT_INPLACE_ABILITY, {{0, 0}});
  op_desc->AppendIrInput("x", kIrInputRequired);
  op_desc->AppendIrOutput("z", kIrOutputRequired);

  const NodeIndexIO inplace_info(node, 0, kOut);
  auto dsa1_node = graph->FindNode("swt1");
  const NodeIndexIO data_info(dsa1_node, 0, kOut);

  ASSERT_TRUE(hybridMemAssigner.Assign() == SUCCESS);
  bool reuse_flag;
  (void)TensorUtils::GetReuseInput(op_desc->GetOutputDesc(0), reuse_flag);
  ASSERT_TRUE(!reuse_flag);
}

TEST_F(UtestHybridMemAssigner, ProcessInplace_two_nodes_not_break_reuse) {
  auto graph = ge::MemConflictShareGraph::BuildUserInAndNotSupportRefreshOutByRefGraph2();
  graph->TopologicalSorting();
  HybridMemAssigner hybridMemAssigner(graph);

  auto node = graph->FindNode("swt2");
  auto op_desc = node->GetOpDesc();
  AttrUtils::SetListListInt(op_desc, ATTR_NAME_OUTPUT_INPLACE_ABILITY, {{0, 0}});
  op_desc->AppendIrInput("x", kIrInputRequired);
  op_desc->AppendIrOutput("z", kIrOutputRequired);

  SetOutReuseInput(graph, "swt2");

  const NodeIndexIO inplace_info(node, 0, kOut);
  auto dsa1_node = graph->FindNode("swt1");
  const NodeIndexIO data_info(dsa1_node, 0, kOut);

  ASSERT_TRUE(hybridMemAssigner.Assign() == SUCCESS);
  bool reuse_flag;
  (void)TensorUtils::GetReuseInput(op_desc->GetOutputDesc(0), reuse_flag);
  ASSERT_TRUE(reuse_flag);
}

/*
                    ┌────────┐  (0,2)
                    │   b    │ ──────────────────────────┐
                    └────────┘                           │
                      ∧                                  │
                      │ (1,0)                            │
                      │                                  ∨
┌────────┐  (0,0)   ┌────────┐  (0,0)   ┌───┐  (0,1)   ┌───────────┐
│ split1 │ ───────> │ split2 │ ───────> │ a │ ───────> │ netoutput │
└────────┘          └────────┘          └───┘          └───────────┘
  │                                                      ∧
  │ (1,0)                                                │
  ∨                                                      │
┌────────┐  (0,0)                                        │
│   c    │ ──────────────────────────────────────────────┘
└────────┘
* a是inplace算子，split2是输出连续，这样会导致用户输出复用split2的输出，导致内存冲突
*/
TEST_F(UtestHybridMemAssigner, ProcessInplace_two_nodes_continous_output) {
  auto graph = ge::MemConflictShareGraph::BuildContinuousOutGraph();
  graph->TopologicalSorting();
  HybridMemAssigner hybridMemAssigner(graph);

  auto node = graph->FindNode("a");
  auto op_desc = node->GetOpDesc();
  AttrUtils::SetListListInt(op_desc, ATTR_NAME_OUTPUT_INPLACE_ABILITY, {{0, 0}});
  op_desc->AppendIrInput("x", kIrInputRequired);
  op_desc->AppendIrOutput("z1", kIrOutputRequired);

  ASSERT_TRUE(hybridMemAssigner.Assign() == SUCCESS);
  bool reuse_flag;
  (void)TensorUtils::GetReuseInput(op_desc->GetOutputDesc(0), reuse_flag);
  ASSERT_FALSE(reuse_flag);
}

TEST_F(UtestHybridMemAssigner, ProcessInplace_while_conflict) {
  auto graph = ge::MemConflictShareGraph::BuildWhileSplitGraph();
  graph->TopologicalSorting();
  HybridMemAssigner hybridMemAssigner(graph);

  auto node = graph->FindNode("acast");
  auto op_desc = node->GetOpDesc();
  op_desc->AppendIrInput("x", kIrInputRequired);
  op_desc->AppendIrOutput("z1", kIrOutputRequired);
  SetOutReuseInput(graph, "split2");

  ASSERT_TRUE(hybridMemAssigner.Assign() == SUCCESS);
  bool reuse_flag;
  (void)TensorUtils::GetReuseInput(op_desc->GetOutputDesc(0), reuse_flag);
  ASSERT_FALSE(reuse_flag);
}

/*
┌────────┐  (0,0)   ┌────────┐  (0,1)   ┌────────┐  (0,0)   ┌───────────┐
│ data1  │ ───────> │  add1  │ ───────> │  add2  │ ───────> │ NetOutput │
└────────┘          └────────┘          └────────┘          └───────────┘
  │                   ∧                   ∧
  │ (0,0)             │ (0,1)             │ (0,0)
  ∨                   │                   │
┌────────┐          ┌────────┐            │
│  add3  │ ─┐       │ data2  │            │
└────────┘  │       └────────┘            │
            │                             │
            └─────────────────────────────┘
* add1和add3都能复用，优先复用add3的输出
*/
TEST_F(UtestHybridMemAssigner, ProcessInplace_three_add_nodes_Success) {
  auto graph = gert::ShareGraph::BuildThreeAddNodeGraph();
  graph->TopologicalSorting();
  HybridMemAssigner hybridMemAssigner_before(graph);
  ASSERT_TRUE(hybridMemAssigner_before.Assign() == SUCCESS);
  auto before_inplace_size = hybridMemAssigner_before.GetSymbolToAnchors().size();

  auto node = graph->FindNode("add2");
  auto op_desc = node->GetOpDesc();
  AttrUtils::SetListListInt(op_desc, ATTR_NAME_OUTPUT_INPLACE_ABILITY, {{0, 0}, {0, 1}});
  op_desc->AppendIrInput("x", kIrInputRequired);
  op_desc->AppendIrInput("y", kIrInputRequired);
  op_desc->AppendIrOutput("z", kIrOutputRequired);
  auto tensor_desc = op_desc->GetOutputDesc(0);
  bool reuse_flag;
  (void)TensorUtils::GetReuseInput(tensor_desc, reuse_flag);
  ASSERT_TRUE(!reuse_flag);

  const NodeIndexIO inplace_info(node, 0, kOut);
  auto add1_node = graph->FindNode("add3");
  const NodeIndexIO data_info(add1_node, 0, kOut);

  auto symbol_to_anchors_before_inplace = hybridMemAssigner_before.GetSymbolToAnchors();
  auto anchors_before_inplace = symbol_to_anchors_before_inplace[data_info.ToString()];

  HybridMemAssigner hybridMemAssigner(graph);
  ASSERT_TRUE(hybridMemAssigner.Assign() == SUCCESS);
  auto symbol_to_anchors_after_inplace = hybridMemAssigner.GetSymbolToAnchors();
  auto anchors_after_inplace = symbol_to_anchors_after_inplace[data_info.ToString()];
  ASSERT_TRUE(hybridMemAssigner.GetSymbolToAnchors().size() == before_inplace_size - 1);
  ASSERT_TRUE(hybridMemAssigner.GetSymbolToAnchors().find(inplace_info.ToString())
              == hybridMemAssigner.GetSymbolToAnchors().end());
  ASSERT_TRUE(hybridMemAssigner.GetSymbolToAnchors().find(data_info.ToString())
              != hybridMemAssigner.GetSymbolToAnchors().end());
  ASSERT_TRUE(anchors_after_inplace.size() == anchors_before_inplace.size() + 2U);
  uint32_t idx = 0U;
  (void)TensorUtils::GetReuseInputIndex(op_desc->GetOutputDesc(0), idx);
  ASSERT_TRUE(idx == 0U); // 1表示复用第一个输入
}

TEST_F(UtestHybridMemAssigner, ReuseCheckerInit_Failed_ReuseCheckerNull) {
  auto graph = gert::ShareGraph::BuildThreeAddNodeGraph();
  graph->TopologicalSorting();
  auto add1 = graph->FindNode("add1");
  ASSERT_NE(add1, nullptr);
  add1->GetOpDescBarePtr()->SetId(100); // dependency_analyzer init will failed

  HybridMemAssigner assigner(graph);
  ASSERT_TRUE(assigner.Assign() == SUCCESS);
  EXPECT_EQ(assigner.GetReuseChecker(), nullptr);
}
} // namespace ge
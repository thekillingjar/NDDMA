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
#include "ge/ge_api.h"
#include "ge_graph_dsl/assert/check_utils.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "easy_graph/layout/graph_layout.h"
#include "ge_running_env/dir_env.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "ge_running_env/tensor_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/operator.h"
#include "graph/operator_factory.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/node_adapter.h"
#include "framework/common/types.h"
#include "graph/tuning_utils.h"
#include "ge_graph_dsl/assert/graph_assert.h"
using namespace std;
using namespace testing;
using namespace ge;

class STestMemOptRuleTest : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

namespace {
/*
 *                               +------------------+  +----------------------+
 *                               |case1_graph       |  |case2_graph           |
 *                               |     Netoutput    |  |                      |
 *                               |       |   \      |  |      Netoutput       |
 *                               |       |   Add    |  |       /      \       |
 *                               |       |  / |     |  |      /     Assigin   |
 *                               |       Mul  |     |  |     /     /   /      |
 *       NetOutput               |      /  |  |     |  |    /     /   /       |
 *       /      \                |  Const  |  |     |  |  Variable   /        |
 * Variable     Case <---------> |     \c  |  |     |  |       \    /         |
 *             /    \            |      Data(1)     |  |        Data          |
 *      pred(Data)  input(Data)  +------------------+  +----------------------+
 */
ComputeGraphPtr CaseGraphWithRwConflict() {
  auto main_graph = []() {
    DEF_GRAPH(g) {
      auto index = OP_CFG(DATA)
         .InCnt(1)
         .OutCnt(1)
         .TensorDesc(FORMAT_ND, DT_INT32, {1})
         .Build("index");
      auto data = OP_CFG(DATA)
         .InCnt(1)
         .OutCnt(1)
         .Build("data");
      auto case_node = OP_CFG(CASE)
         .InCnt(2)
         .OutCnt(2)
         .Build("case");
      auto var_node = OP_CFG(VARIABLE)
         .InCnt(1)
         .OutCnt(1)
         .TensorDesc(FORMAT_ND, DT_FLOAT, {32, 32})
         .Build("variable");
      auto net_output = OP_CFG(NETOUTPUT)
         .InCnt(3)
         .OutCnt(1);
      CHAIN(NODE(index)->NODE(case_node)->NODE("NetOutput", net_output));
      CHAIN(NODE(case_node)->EDGE(1, 1)->NODE("NetOutput", net_output));
      CHAIN(NODE(data)->EDGE(0, 1)->NODE(case_node));
      CHAIN(NODE(var_node)->EDGE(0, 2)->NODE("NetOutput", net_output));
    };
    return ToComputeGraph(g);
  }();
  main_graph->SetName("main");
  ge::AttrUtils::SetInt(main_graph->FindNode("index")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(main_graph->FindNode("data")->GetOpDesc(), "index", 1);

  auto case1_graph = []() {
    DEF_GRAPH(g) {
      auto data = OP_CFG(DATA)
         .InCnt(1)
         .OutCnt(1)
         .TensorDesc(FORMAT_ND, DT_FLOAT, {32, 32})
         .Build("data_case1");
      auto mul = OP_CFG(MUL)
         .InCnt(2)
         .OutCnt(1)
         .Build("mul_case1");
      auto add = OP_CFG(ADD)
         .InCnt(2)
         .OutCnt(1)
         .Build("add_case1");
     std::vector<int64_t> shape = {32, 32};
     auto data_tensor = GenerateTensor(shape);
     auto const_node = OP_CFG(CONSTANT)
         .InCnt(1)
         .OutCnt(1)
         .Weight(data_tensor)
         .TensorDesc(FORMAT_ND, DT_INT32, shape)
         .Build("constant_case1");
      auto net_output = OP_CFG(NETOUTPUT)
         .InCnt(2)
         .OutCnt(1);
      CHAIN(NODE(data)->EDGE(0, 0)->NODE(mul)->EDGE(0, 0)->NODE("NetOutput_case1", net_output));
      CHAIN(NODE(data)->EDGE(0, 0)->NODE(add)->EDGE(0, 1)->NODE("NetOutput_case1", net_output));
      CHAIN(NODE(mul)->EDGE(0, 1)->NODE(add));
      CHAIN(NODE(const_node)->EDGE(0, 1)->NODE(mul));
      CHAIN(NODE(data)->CTRL_EDGE()->NODE(const_node));
    };
    return ToComputeGraph(g);
  }();
  case1_graph->SetName("case1_graph");
  ge::AttrUtils::SetInt(case1_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(case1_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
  auto net_out_op_desc = case1_graph->FindFirstNodeMatchType("NetOutput")->GetOpDesc();
  ge::AttrUtils::SetInt(net_out_op_desc->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
  ge::AttrUtils::SetInt(net_out_op_desc->MutableInputDesc(1), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);

  auto case2_graph = []() {
    DEF_GRAPH(g) {
      auto data = OP_CFG(DATA)
         .InCnt(1)
         .OutCnt(1)
         .TensorDesc(FORMAT_ND, DT_FLOAT, {32, 32})
         .Build("data_case2");
      auto assign = OP_CFG(ASSIGN)
         .InCnt(2)
         .OutCnt(1)
         .Build("assign_case2");
      auto var_node = OP_CFG(VARIABLE)
         .InCnt(1)
         .OutCnt(1)
         .Build("var1_case2");
      auto cast = OP_CFG(CAST)
         .InCnt(1)
         .OutCnt(1)
         .Build("var1_case2");
      auto net_output = OP_CFG(NETOUTPUT)
         .InCnt(2)
         .OutCnt(1);
      CHAIN(NODE(data)->EDGE(0, 1)->NODE(assign)->EDGE(0, 0)->NODE("netOutput_case2", net_output));
      CHAIN(NODE(var_node)->EDGE(0, 0)->NODE(assign));
      CHAIN(NODE(data)->EDGE(0, 0)->NODE(cast)->EDGE(0, 1)->NODE("netOutput_case2", net_output));
      };
    return ToComputeGraph(g);
  }();
  case2_graph->SetName("case2_graph");
  ge::AttrUtils::SetInt(case2_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(case2_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);

  auto ref_node_op_desc = case2_graph->FindFirstNodeMatchType("Assign")->GetOpDesc();
  ge::AttrUtils::SetInt(ref_node_op_desc, ge::ATTR_NAME_REFERENCE, true);
  ref_node_op_desc->UpdateInputName({{"ref", 0}, {"value", 1}});
  ref_node_op_desc->UpdateOutputName({{"ref", 0}});
  net_out_op_desc = case2_graph->FindFirstNodeMatchType("NetOutput")->GetOpDesc();
  ge::AttrUtils::SetInt(net_out_op_desc->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
  ge::AttrUtils::SetInt(net_out_op_desc->MutableInputDesc(1), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);

  auto case_node = main_graph->FindNode("case");
  case_node->GetOpDesc()->MutableInputDesc(0)->SetDataType(ge::DT_INT32);
  case_node->GetOpDesc()->MutableInputDesc(0)->SetOriginDataType(ge::DT_INT32);
  case_node->GetOpDesc()->MutableInputDesc(1)->SetDataType(ge::DT_FLOAT);
  case_node->GetOpDesc()->MutableInputDesc(1)->SetOriginDataType(ge::DT_FLOAT);
  case_node->GetOpDesc()->AppendIrInput("branch_index", kIrInputRequired);
  case_node->GetOpDesc()->AppendIrInput("input", kIrInputDynamic);

  main_graph->FindNode("NetOutput")->GetOpDesc()->SetSrcName({"case", "var"});
  main_graph->FindNode("NetOutput")->GetOpDesc()->SetSrcIndex({0, 0});

  auto &name_index = case_node->GetOpDesc()->MutableAllInputName();
  name_index.clear();
  name_index["branch_index"] = 0;
  name_index["input0"] = 1;
  case1_graph->SetParentGraph(main_graph);
  case1_graph->SetParentNode(case_node);
  case2_graph->SetParentGraph(main_graph);
  case2_graph->SetParentNode(case_node);

  main_graph->AddSubgraph(case1_graph);
  main_graph->AddSubgraph(case2_graph);
  case_node->GetOpDesc()->AddSubgraphName("case1_graph");
  case_node->GetOpDesc()->AddSubgraphName("case2_graph");
  case_node->GetOpDesc()->SetSubgraphInstanceName(0, "case1_graph");
  case_node->GetOpDesc()->SetSubgraphInstanceName(1, "case2_graph");

  auto net_output = main_graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"case"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  main_graph->TopologicalSorting();

  return main_graph;
}

ComputeGraphPtr GraphWithSubgraphAndFIFORwConflict() {
  auto main_graph = []() {
    DEF_GRAPH(g) {
      auto index = OP_CFG(DATA).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1}).Build("index");
      auto data = OP_CFG(DATA).InCnt(1).OutCnt(1).Build("data");
      auto case_node = OP_CFG(CASE).InCnt(2).OutCnt(1).Build("case");
      auto net_output = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1);
      CHAIN(NODE(index)->NODE(case_node)->NODE("NetOutput", net_output));
      CHAIN(NODE(case_node)->EDGE(1, 1)->NODE("NetOutput", net_output));
      CHAIN(NODE(data)->EDGE(0, 1)->NODE(case_node));
    };
    return ToComputeGraph(g);
  }();
  main_graph->SetName("main");
  ge::AttrUtils::SetInt(main_graph->FindNode("index")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(main_graph->FindNode("data")->GetOpDesc(), "index", 1);

  auto case1_graph = []() {
    DEF_GRAPH(g) {
      auto data = OP_CFG(DATA).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1}).Build("data_case1");
      std::vector<int64_t> shape = {1};
      auto data_tensor = GenerateTensor(shape);
      auto const_node = OP_CFG(CONSTANT).InCnt(1).OutCnt(1).Weight(data_tensor).TensorDesc(FORMAT_ND, DT_INT32, {1}).Build("const_case1");
      auto foo = OP_CFG(RELU).InCnt(1).OutCnt(1).Build("foo_case1");
      auto net_output = OP_CFG(NETOUTPUT).InCnt(2).OutCnt(1);
      auto cast = OP_CFG(CAST).InCnt(1).OutCnt(1).Build("cast_case1");
      CHAIN(NODE(data)->CTRL_EDGE()->NODE(const_node)->NODE(foo)->NODE("NetOutput_case1", net_output));
      CHAIN(NODE(data)->EDGE(0, 0)->NODE(cast)->EDGE(0, 1)->NODE("NetOutput_case1", net_output));
    };
    return ToComputeGraph(g);
  }();
  case1_graph->SetName("case1_graph");
  ge::AttrUtils::SetInt(case1_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(case1_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
  ge::AttrUtils::SetInt(case1_graph->FindFirstNodeMatchType("ReLU")->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_SPECIAL_INPUT_SIZE, 1);
  auto net_out_op_desc = case1_graph->FindFirstNodeMatchType("NetOutput")->GetOpDesc();
  ge::AttrUtils::SetInt(net_out_op_desc->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
  ge::AttrUtils::SetInt(net_out_op_desc->MutableInputDesc(1), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);

  auto case2_graph = []() {
    DEF_GRAPH(g) {
      auto data = OP_CFG(DATA).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {1}).Build("data_case2");
      auto add = OP_CFG(ADD).InCnt(2).OutCnt(1).Build("add_case2");
      auto net_output = OP_CFG(NETOUTPUT).InCnt(2).OutCnt(1);
      std::vector<int64_t> shape = {1};
      auto data_tensor = GenerateTensor(shape);
      auto const_node = OP_CFG(CONSTANT).InCnt(1).OutCnt(1).Weight(data_tensor).TensorDesc(FORMAT_ND, DT_INT32, {1}).Build("const_case2");
      auto cast = OP_CFG(CAST).InCnt(1).OutCnt(1).Build("cast_case2");
      CHAIN(NODE(data)->EDGE(0, 0)->NODE(add)->NODE("NetOutput_case2", net_output));
      CHAIN(NODE(const_node)->EDGE(0, 1)->NODE(add));
      CHAIN(NODE(data)->CTRL_EDGE()->NODE(const_node));
      CHAIN(NODE(data)->EDGE(0, 0)->NODE(cast)->EDGE(0, 1)->NODE("NetOutput_case2", net_output));
    };
    return ToComputeGraph(g);
  }();
  case2_graph->SetName("case2_graph");
  ge::AttrUtils::SetInt(case2_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), "index", 0);
  ge::AttrUtils::SetInt(case2_graph->FindFirstNodeMatchType("Data")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);
  auto netout_op_desc = case2_graph->FindFirstNodeMatchType("NetOutput")->GetOpDesc();
  ge::AttrUtils::SetInt(netout_op_desc->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
  ge::AttrUtils::SetInt(netout_op_desc->MutableInputDesc(1), ge::ATTR_NAME_PARENT_NODE_INDEX, 1);

  auto case_node = main_graph->FindNode("case");
  case_node->GetOpDesc()->MutableInputDesc(0)->SetDataType(ge::DT_INT32);
  case_node->GetOpDesc()->MutableInputDesc(0)->SetOriginDataType(ge::DT_INT32);
  case_node->GetOpDesc()->MutableInputDesc(1)->SetDataType(ge::DT_INT32);
  case_node->GetOpDesc()->MutableInputDesc(1)->SetOriginDataType(ge::DT_INT32);
  case_node->GetOpDesc()->AppendIrInput("branch_index", kIrInputRequired);
  case_node->GetOpDesc()->AppendIrInput("input", kIrInputDynamic);

  main_graph->FindNode("NetOutput")->GetOpDesc()->SetSrcName({"case", "var"});
  main_graph->FindNode("NetOutput")->GetOpDesc()->SetSrcIndex({0, 0});

  auto &name_index = case_node->GetOpDesc()->MutableAllInputName();
  name_index.clear();
  name_index["branch_index"] = 0;
  name_index["input0"] = 1;

  case1_graph->SetParentGraph(main_graph);
  case1_graph->SetParentNode(case_node);
  case2_graph->SetParentGraph(main_graph);
  case2_graph->SetParentNode(case_node);

  main_graph->AddSubgraph(case1_graph);
  main_graph->AddSubgraph(case2_graph);
  case_node->GetOpDesc()->AddSubgraphName("case1_graph");
  case_node->GetOpDesc()->AddSubgraphName("case2_graph");
  case_node->GetOpDesc()->SetSubgraphInstanceName(0, "case1_graph");
  case_node->GetOpDesc()->SetSubgraphInstanceName(1, "case2_graph");

  auto net_output = main_graph->FindNode("NetOutput");
  net_output->GetOpDesc()->SetSrcName({"case"});
  net_output->GetOpDesc()->SetSrcIndex({0});
  main_graph->TopologicalSorting();

  return main_graph;
}

}

/**
 * 用例描述：case子图场景下，存在读写冲突时，需要插入identity节点
 *
 * 预置条件：
 * 1. fake一个case字图，分支1的第0个输出写输出，另一个分支2的第0个输出存在读，不同分支存在读写冲突，但是操作的却是同一块内存。
 *
 * 测试步骤：
 * 1. 按照预制条件构造好case子图。
 * 2. 从入口处调用图编译接口。
 *
 * 预期结果：
 * 1. 图编译执行成功
 * 2. 存在写分支的子图插入了identity节点，同时插入了LabelSet跟StreamActive节点，子图节点个数会增多。
 * 3. 存在写分支的子图在Netoutput与第0个输入node之间插入了identity节点。
 */

/* 修改后图
 *                               +---------------------------+  +---------------------------+
 *                               |case1_graph                |  |case2_graph                |
 *                               |                LabelSet   |  |         LabelSet          |
 *                               |                  |c       |  |             |             |
 *                               |               Netoutput   |  |         Netoutput         |
 *                               |                  |   \    |  |          /    |           |
 *                               |                  |    Add |  |         /  Identity       |
 *                               |                  |  /  |  |  |        /        \         |
 *                               |StreamActive-----Mul    |  |  |     Identity     Assigin  |
 *       NetOutput               |     |c         /   |   |  |  |       |    |c    |c    |  |
 *       /      \                | LabelSet    Const  |   |  |  |  Variable  |     |     |  |
 * Variable     Case <---------> |     |c          \c |   |  |  |     \    StreamActive  |  |
 *             /   \             |LabelSwitchByIndex  Data(1)|  |      \        |        |  |
 *            /     \            |     |c                    |  |       \    LabelSet    |  |
 *           /       \           |    Data                   |  |        +------Data-----+  |
 *      pred(Data)  input(Data)  +---------------------------+  +----------------------------+
*/
TEST_F(STestMemOptRuleTest, test_case_RW_conflict) {
  DUMP_GRAPH_WHEN("PreRunAfterBuild");
  auto compute_graph = CaseGraphWithRwConflict();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);

  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterBuild) {
    for (auto graph : graph->GetAllSubgraphs()) {
      if (graph->GetName() == "case2_graph") {
        // identity节点如果校验失败则需要审视修改
        auto identity = graph->FindFirstNodeMatchType("Identity");
        ASSERT_NE(identity, nullptr);

        auto netoutput_node = graph->FindFirstNodeMatchType("NetOutput");
        auto in_anchor = netoutput_node->GetInDataAnchor(0U);
        auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
        auto node = peer_out_anchor->GetOwnerNode();
        ASSERT_EQ(node->GetType(), "Identity");
      }
    }
  };
}

/*
 * 用例场景：构造根图上variable-hcomallreduce，不带有子图。allreduce带有mutable_input属性，如果输入是variable/cons会插入identity。
 * 步骤：
 * step 1. 按照用例场景构图
 * 期望：构图成功
 * step 2. 执行图编译
 * 期望： 图编译返回成功
 * step 3. 校验
 * 期望：allreduce前插入identity
 *   var
 *    |
 *  allreduce (_input_mutable)
 *
 *   ||
 *   \/
 *
 *   var
 *    |
 *  identity
 *    |
 *  allreduce (_input_mutable)
 *
*/
TEST_F(STestMemOptRuleTest, TestImutableInput_NoSubgraph_CheckInsertIdentity) {
  DUMP_GRAPH_WHEN("PreRunAfterBuild");
  auto allreduce = OP_CFG(HCOMALLREDUCE).Attr("_input_mutable", true);
  DEF_GRAPH(g1) {
                  CHAIN(NODE("var", VARIABLE)->NODE("allreduce", allreduce)->NODE("netoutput", NETOUTPUT));
                };
  auto graph = ToGeGraph(g1);
  map<AscendString, AscendString> options;
  options[OPTION_FEATURE_BASE_REFRESHABLE] = "0";
  Session session(options);
  session.AddGraph(1, graph, options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);

  EXPECT_EQ(ret, SUCCESS);
  size_t count = 0U;
  std::map<std::string, NodePtr> name_to_node;
  CHECK_GRAPH(PreRunAfterBuild) {
    for (auto node : graph->GetAllNodes()) {
      if (node->GetType() == IDENTITY) {
        ++count;
      }
      name_to_node[node->GetName()] = node;
    }
    EXPECT_EQ(name_to_node["allreduce"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
  };
  EXPECT_EQ(count, 2U);
}

TEST_F(STestMemOptRuleTest, TestFIFORwConflictWithoutSubgraph) {
  DUMP_GRAPH_WHEN("PreRunAfterBuild");
  DEF_GRAPH(g1) {
    CHAIN(NODE("const1", CONSTANT)->NODE("foo", RELU)->NODE("netoutput", NETOUTPUT));
  };
  auto computeGraph1 = ToComputeGraph(g1);
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(computeGraph1);
  int64_t special_size = 64; // random value for test
  auto foo_desc = computeGraph1->FindFirstNodeMatchType("ReLU")->GetOpDesc();
  ge::AttrUtils::SetInt(foo_desc->MutableInputDesc(0), ATTR_NAME_SPECIAL_INPUT_SIZE, special_size);
  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);

  EXPECT_EQ(ret, SUCCESS);
  size_t count = 0U;
  std::map<std::string, NodePtr> name_to_node;
  CHECK_GRAPH(PreRunAfterBuild) {
    for (auto node : graph->GetAllNodes()) {
      if (node->GetType() == IDENTITY) {
        ++count;
      }
      name_to_node[node->GetName()] = node;
    }
    EXPECT_EQ(name_to_node["foo"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
  };
  EXPECT_EQ(count, 1U);
}

TEST_F(STestMemOptRuleTest, TestFIFORwConflictWithSubgraph) {
  DUMP_GRAPH_WHEN("PreRunAfterBuild");
  auto compute_graph = GraphWithSubgraphAndFIFORwConflict();
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, graph, options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);

  EXPECT_EQ(ret, SUCCESS);
  size_t count = 0U;
  std::map<std::string, NodePtr> name_to_node;
  CHECK_GRAPH(PreRunAfterBuild) {
    for (auto node : graph->GetAllNodes()) {
      if (node->GetType() == IDENTITY) {
        ++count;
      }
      name_to_node[node->GetName()] = node;
    }
    EXPECT_EQ(name_to_node["foo_case1"]->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetType(), IDENTITY);
  };
  EXPECT_EQ(count, 4U);
}
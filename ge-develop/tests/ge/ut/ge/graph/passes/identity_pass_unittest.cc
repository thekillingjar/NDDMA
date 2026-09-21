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

#include "macro_utils/dt_public_scope.h"
#include "graph/passes/memory_conflict/identity_pass.h"

#include "common/op/ge_op_utils.h"
#include "common/types.h"
#include "graph/anchor.h"
#include "graph/attr_value.h"
#include "graph/compute_graph.h"
#include "graph/op_desc.h"
#include "graph/passes/base_pass.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "graph/utils/tensor_utils.h"
#include "graph_builder_utils.h"
#include "graph/passes/pass_manager.h"
#include "macro_utils/dt_public_unscope.h"

using namespace std;
using namespace testing;
using namespace ge;

class UtestIdentityPass : public Test {
 protected:
  NodePtr AddNode(ComputeGraphPtr graph, const string &name, const string &type, int32_t in_anchors_num = 1,
                  int32_t out_anchors_num = 1) {
    GeTensorDesc tensor_desc;
    OpDescPtr opdesc = std::make_shared<OpDesc>(name, type);
    for (int32_t i = 0; i < in_anchors_num; i++) {
      opdesc->AddInputDesc(tensor_desc);
    }
    for (int32_t i = 0; i < out_anchors_num; i++) {
      opdesc->AddOutputDesc(tensor_desc);
    }

    NodePtr node = graph->AddNode(opdesc);
    return node;
  }
};

///  merge1
///    |
/// identity1
///   |   \c
/// var1  var2
static ComputeGraphPtr BuildGraph1() {
  ge::ut::GraphBuilder builder("g1");
  auto var1 = builder.AddNode("var1", "Variable", 0, 1);
  auto var2 = builder.AddNode("var2", "Variable", 0, 1);
  auto identity1 = builder.AddNode("identity1", "Identity", 1, 1);
  auto merge1 = builder.AddNode("merge1", "Merge", 1, 1);

  builder.AddDataEdge(var1, 0, identity1, 0);
  builder.AddControlEdge(var2, identity1);
  builder.AddDataEdge(identity1, 0, merge1, 0);
  return builder.GetGraph();
}

///   addn1
///    |c
///  identity1
///    |
///  switch1
///    |
///   var1
static ComputeGraphPtr BuildGraph2() {
  ge::ut::GraphBuilder builder("g1");
  auto var1 = builder.AddNode("var1", "Variable", 0, 1);
  auto switch1 = builder.AddNode("switch1", "Switch", 2, 2);
  auto identity1 = builder.AddNode("identity1", "Identity", 1, 1);
  auto addn1 = builder.AddNode("addn1", "AddN", 1, 1);

  builder.AddDataEdge(var1, 0, switch1, 0);
  builder.AddDataEdge(switch1, 0, identity1, 0);
  builder.AddControlEdge(identity1, addn1);
  return builder.GetGraph();
}

///  addn1
///    |
/// identity1
///    |
/// switch1
///    |
///  var1
static ComputeGraphPtr BuildGraph3() {
  ge::ut::GraphBuilder builder("g3");
  auto var1 = builder.AddNode("var1", "Variable", 0, 1);
  auto switch1 = builder.AddNode("switch1", "Switch", 2, 2);
  auto identity1 = builder.AddNode("identity1", "Identity", 1, 1);
  auto addn1 = builder.AddNode("addn1", "AddN", 1, 1);

  builder.AddDataEdge(var1, 0, switch1, 0);
  builder.AddDataEdge(switch1, 0, identity1, 0);
  builder.AddDataEdge(identity1, 0, addn1, 0);
  return builder.GetGraph();
}

/**
 * var---->readvar--->cast--->netoutput
 *    \       ||
 *       \    ||
 *         \  ||
 *  const->assgin
 */
ComputeGraphPtr BuildGraph4(bool read_first_then_write = true) {
  auto builder = ut::GraphBuilder("test");
  // id1 is useful
  auto id1 = builder.AddNode("id1", READVARIABLEOP, 1, 1);
  auto var0 = builder.AddNode("var0", VARIABLE, 1, 1);
  auto const0 = builder.AddNode("const0", CONSTANT, 1, 1);
  auto cast = builder.AddNode("cast", "CAST", 1, 1);
  auto ref_node = builder.AddNode("ref_node", ASSIGN, 2, 1);
  ref_node->GetOpDesc()->UpdateInputName({{"ref", 0}, {"value", 1}});
  ref_node->GetOpDesc()->UpdateOutputName({{"ref", 0}});
  auto netoutput_node = builder.AddNode("netoutput", NETOUTPUT, 1, 1);

  builder.AddDataEdge(var0, 0, id1, 0);
  builder.AddDataEdge(id1, 0, cast, 0);
  builder.AddDataEdge(cast, 0, netoutput_node, 0);
  builder.AddDataEdge(var0, 0, ref_node, 0);
  builder.AddDataEdge(const0, 0, ref_node, 1);
  if (read_first_then_write) {
    builder.AddControlEdge(id1, ref_node); // read first
  } else {
    builder.AddControlEdge(ref_node, id1); // write first
  }
  return builder.GetGraph();
}

/**
 * var---->readvar--->mul--->netoutput
 *    \       ||     /\
 *       \    ||    /
 *        \/  \/  /
 *  const->assgin
 */
ComputeGraphPtr BuildGraph5() {
  auto builder = ut::GraphBuilder("test");
  // id1 is for var cache
  auto id1 = builder.AddNode("id1", READVARIABLEOP, 1, 1);
  auto var0 = builder.AddNode("var0", VARIABLE, 1, 1);
  auto const0 = builder.AddNode("const0", CONSTANT, 1, 1);
  auto mul = builder.AddNode("mul", MUL, 2, 1);
  auto ref_node = builder.AddNode("ref_node", ASSIGN, 2, 1);
  ref_node->GetOpDesc()->UpdateInputName({{"ref", 0}, {"value", 1}});
  ref_node->GetOpDesc()->UpdateOutputName({{"ref", 0}});
  auto netoutput_node = builder.AddNode("netoutput", NETOUTPUT, 1, 1);

  builder.AddDataEdge(var0, 0, id1, 0);
  builder.AddDataEdge(id1, 0, mul, 0);
  builder.AddDataEdge(ref_node, 0, mul, 1); // mul is like (var) * (var+1)
  builder.AddDataEdge(mul, 0, netoutput_node, 0);
  builder.AddDataEdge(var0, 0, ref_node, 0);
  builder.AddDataEdge(const0, 0, ref_node, 1);
  builder.AddControlEdge(id1, ref_node);  // read first
  return builder.GetGraph();
}

TEST_F(UtestIdentityPass, succ) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = AddNode(graph, "Identity", IDENTITY);
  NodePtr reduce_min_node = AddNode(graph, "reduceMin", REDUCEMIN);

  GraphUtils::AddEdge(node->GetOutDataAnchor(0), reduce_min_node->GetInDataAnchor(0));

  IdentityPass pass(true);
  Status status = pass.Run(node);
  EXPECT_EQ(status, SUCCESS);
  NodePtr found_node = graph->FindNode("Identity");
  EXPECT_EQ(found_node, nullptr);

  status = pass.Run(reduce_min_node);
  EXPECT_EQ(status, SUCCESS);

  string type2 = "FrameworkOp";
  auto op_desc = node->GetOpDesc();
  ge::OpDescUtilsEx::SetType(op_desc, type2);
  status = pass.Run(node);

  NodePtr node_err = AddNode(graph, "Identity", IDENTITY, 1, 2);
  status = pass.Run(node_err);
  EXPECT_EQ(status, ge::PARAM_INVALID);
}

TEST_F(UtestIdentityPass, skip_merge) {
  auto graph = BuildGraph1();
  ge::GEPass pass(graph);

  ge::NamesToPass names_to_pass;
  IdentityPass identity_pass(false);
  names_to_pass.emplace_back("IdentityPass", &identity_pass);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  auto identity1 = graph->FindNode("identity1");
  EXPECT_NE(identity1, nullptr);
  EXPECT_EQ(identity1->GetOutNodes().size(), 1);
  EXPECT_EQ(identity1->GetOutDataNodes().at(0)->GetName(), "merge1");
  EXPECT_EQ(identity1->GetInNodes().size(), 2);

  names_to_pass.clear();
  IdentityPass force_pass(true);
  names_to_pass.emplace_back("IdentityPass", &force_pass);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  identity1 = graph->FindNode("identity1");
  EXPECT_EQ(identity1, nullptr);
}

TEST_F(UtestIdentityPass, skip_switch) {
  auto graph = BuildGraph2();
  ge::GEPass pass(graph);

  ge::NamesToPass names_to_pass;
  IdentityPass identity_pass(false);
  names_to_pass.emplace_back("IdentityPass", &identity_pass);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  auto identity1 = graph->FindNode("identity1");
  EXPECT_NE(identity1, nullptr);
  EXPECT_EQ(identity1->GetInNodes().size(), 1);
  EXPECT_EQ(identity1->GetInDataNodes().at(0)->GetName(), "switch1");

  names_to_pass.clear();
  IdentityPass force_pass(true);
  names_to_pass.emplace_back("IdentityPass", &force_pass);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  identity1 = graph->FindNode("identity1");
  EXPECT_EQ(identity1, nullptr);
}

TEST_F(UtestIdentityPass, norm_after_switch) {
  auto graph = BuildGraph3();
  ge::GEPass pass(graph);

  ge::NamesToPass names_to_pass;
  IdentityPass identity_pass(false);
  names_to_pass.emplace_back("IdentityPass", &identity_pass);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  auto identity1 = graph->FindNode("identity1");
  EXPECT_EQ(identity1, nullptr);
  auto switch1 = graph->FindNode("switch1");
  EXPECT_EQ(switch1->GetOutNodes().size(), 1);
  EXPECT_EQ(switch1->GetOutDataNodes().at(0)->GetName(), "addn1");
}

TEST_F(UtestIdentityPass, useful_rw_control) {
  auto graph = BuildGraph4();
  ge::GEPass pass(graph);

  ge::NamesToPass names_to_pass;
  IdentityPass identity_pass(false);
  names_to_pass.emplace_back("IdentityPass", &identity_pass);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  auto identity1 = graph->FindNode("id1");
  // identity is removed
  EXPECT_EQ(identity1, nullptr);
  auto cast_node = graph->FindNode("cast");
  // identity's ctrl output moved to cast
  EXPECT_NE(cast_node, nullptr);
  EXPECT_EQ(cast_node->GetOutControlNodesSize(), 1U);
  auto ref_node = graph->FindNode("ref_node");
  EXPECT_EQ(cast_node->GetOutControlNodes().at(0), ref_node);
}

TEST_F(UtestIdentityPass, useful_wr_control) {
  auto graph = BuildGraph4(false);
  ge::GEPass pass(graph);

  ge::NamesToPass names_to_pass;
  IdentityPass identity_pass(false);
  names_to_pass.emplace_back("IdentityPass", &identity_pass);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  auto identity1 = graph->FindNode("id1");
  // identity is removed
  EXPECT_EQ(identity1, nullptr);
  auto cast_node = graph->FindNode("cast");
  // identity's ctrl in moved to cast
  EXPECT_NE(cast_node, nullptr);
  EXPECT_EQ(cast_node->GetInControlNodesSize(), 1U);
  auto ref_node = graph->FindNode("ref_node");
  EXPECT_EQ(cast_node->GetInControlNodes().at(0), ref_node);
}

TEST_F(UtestIdentityPass, useful_var_cache) {
  auto graph = BuildGraph5();
  ge::GEPass pass(graph);

  ge::NamesToPass names_to_pass;
  IdentityPass identity_pass(false);
  names_to_pass.emplace_back("IdentityPass", &identity_pass);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  auto identity1 = graph->FindNode("id1");
  // identity is keeped for var cache
  EXPECT_NE(identity1, nullptr);
}

TEST_F(UtestIdentityPass, not_delete_identity) {
  auto graph = BuildGraph3();
  auto identity1 = graph->FindNode("identity1");
  (void)AttrUtils::SetBool(identity1->GetOpDesc(), ATTR_NAME_IS_INSERTED_BY_CANN, true);

  ge::GEPass pass(graph);
  ge::NamesToPass names_to_pass;
  IdentityPass identity_pass(false);
  names_to_pass.emplace_back("IdentityPass", &identity_pass);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);
  EXPECT_NE(graph->FindNode("identity1"), nullptr);
}

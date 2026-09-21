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
#include "graph/partition/optimizer/hostcpu_engine_update_pass.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "framework/common/ge_types.h"
#include "engines/manager/opskernel_manager/ops_kernel_manager.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "graph/utils/node_utils.h"
#include "mmpa/mmpa_api.h"
#include "macro_utils/dt_public_unscope.h"
#include "ge/ge_api_types.h"
#include "graph/ge_context.h"
#include "graph/ge_local_context.h"
#include "graph/option/optimization_option.h"
#include "register/optimization_option_registry.h"

using namespace testing;

namespace ge {
namespace {
void SetNoStorage(const ge::OpDescPtr &op_desc, Format format, DataType dt, std::initializer_list<int64_t> in_shape,
                  std::initializer_list<int64_t> out_shape) {
  for (size_t i = 0; i < op_desc->GetInputsSize(); ++i) {
    op_desc->MutableInputDesc(i)->SetFormat(format);
    op_desc->MutableInputDesc(i)->SetOriginFormat(format);
    op_desc->MutableInputDesc(i)->SetShape(GeShape(in_shape));
    op_desc->MutableInputDesc(i)->SetOriginShape(GeShape(in_shape));
    op_desc->MutableInputDesc(i)->SetDataType(dt);
    op_desc->MutableInputDesc(i)->SetOriginDataType(dt);
  }
  for (size_t i = 0; i < op_desc->GetOutputsSize(); ++i) {
    op_desc->MutableOutputDesc(i)->SetFormat(format);
    op_desc->MutableOutputDesc(i)->SetOriginFormat(format);
    op_desc->MutableOutputDesc(i)->SetShape(GeShape(out_shape));
    op_desc->MutableOutputDesc(i)->SetOriginShape(GeShape(out_shape));
    op_desc->MutableOutputDesc(i)->SetDataType(dt);
    op_desc->MutableOutputDesc(i)->SetOriginDataType(dt);
  }
}

void SetShapeRangeNoStorage(const ge::OpDescPtr &op_desc, std::initializer_list<int64_t> min_shape,
                            std::initializer_list<int64_t> max_shape) {
  std::vector<std::pair<int64_t, int64_t>> range;
  for (size_t i = 0; i < min_shape.size(); ++i) {
    range.emplace_back(min_shape.begin()[i], max_shape.begin()[i]);
  }
  for (size_t i = 0; i < op_desc->GetInputsSize(); ++i) {
    op_desc->MutableInputDesc(i)->SetOriginShapeRange(range);
    op_desc->MutableInputDesc(i)->SetShapeRange(range);
  }
  for (size_t i = 0; i < op_desc->GetOutputsSize(); ++i) {
    op_desc->MutableOutputDesc(i)->SetOriginShapeRange(range);
    op_desc->MutableOutputDesc(i)->SetShapeRange(range);
  }
}

/**
 * netoutput
 *    |
 *  reshape
 *    |     \
 *    |      nonzero
 *    |       \ 
 *    |      concat
 *    |     /      \ 
 *    | unsqueeze1  unsqueeze2
 *    |    |          |
 *    |  gather1     gather2
 *    |    |   \      |   \ 
 *    |    |  const1  |  const2
 *    |    |          |
 *    |  shape1      shape2
 *    |        \    /
 *    +---------data
 */
ComputeGraphPtr BuildGraph1() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data", "Data")
              ->NODE("shape1", "Shape")
              ->NODE("gather1", "Gather")
              ->NODE("unsqueeze1", "Unsqueeze")
              ->NODE("concat", "Concat")
              ->NODE("nonzero", "NonZero")
              ->EDGE(0, 1)
              ->NODE("reshape", "Reshape")
              ->NODE("netoutput", "NetOutput"));
    CHAIN(NODE("data", "Data")
              ->EDGE(0, 1)
              ->NODE("shape2", "Shape")
              ->NODE("gather2", "Gather")
              ->NODE("unsqueeze2", "Unsqueeze")
              ->EDGE(0, 1)
              ->NODE("concat", "Concat"));
    CHAIN(NODE("data", "Data")->EDGE(0, 0)->NODE("reshape", "Reshape"));
    CHAIN(NODE("const1", "Const")->EDGE(0, 1)->NODE("gather1", "Gather"));
    CHAIN(NODE("const2", "Const")->EDGE(0, 1)->NODE("gather2", "Gather"));
  };
  auto graph = ToComputeGraph(g1);
  graph->SetGraphUnknownFlag(true);
  graph->FindNode("data")->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  graph->FindNode("const1")->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  graph->FindNode("const2")->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  graph->FindNode("shape1")->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  graph->FindNode("shape2")->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  graph->FindNode("gather1")->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  graph->FindNode("gather2")->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  graph->FindNode("unsqueeze1")->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  graph->FindNode("unsqueeze2")->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  graph->FindNode("concat")->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  graph->FindNode("reshape")->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  graph->FindNode("nonzero")->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  return graph;
}

/**
 *                                       netoutput
 *                                           |
 *             netoutput                   split
 *                |                          |   \ 
 *       partitionedcall2     netoutput      |   shape
 *             /       \          |          |      |
 *  partitionedcall1   data1    const       data1   data2
 */
ComputeGraphPtr BuildGraph2() {
  DEF_GRAPH(g2) {
    CHAIN(NODE("partitionedcall1", "PartitionedCall")
              ->NODE("partitionedcall2", "PartitionedCall")
              ->NODE("netoutput", "NetOutput"));
    CHAIN(NODE("data1", "Data")
              ->EDGE(0, 1)
              ->NODE("partitionedcall2", "PartitionedCall"));
  };
  DEF_GRAPH(g3) {
    CHAIN(NODE("const", "Const")
              ->NODE("netoutput", "NetOutput"));
  };
  DEF_GRAPH(g4) {
    CHAIN(NODE("data1", "Data")
              ->NODE("split", "Split")
              ->NODE("netoutput", "NetOutput"));
    CHAIN(NODE("data2", "Data")
              ->NODE("shape", "Shape")
              ->EDGE(0, 1)
              ->NODE("split", "Split"));
  };
  auto graph = ToComputeGraph(g2);
  graph->SetGraphUnknownFlag(true);
  graph->FindNode("data1")->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  graph->FindNode("netoutput")->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);

  auto sub_graph1 = ToComputeGraph(g3);
  sub_graph1->SetGraphUnknownFlag(true);
  sub_graph1->FindNode("const")->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  sub_graph1->FindNode("netoutput")->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  auto dst_desc = sub_graph1->FindNode("netoutput")->GetOpDesc()->MutableInputDesc(0);
  AttrUtils::SetInt(dst_desc, "_parent_node_index", 0);

  auto sub_graph2 = ToComputeGraph(g4);
  sub_graph2->SetGraphUnknownFlag(true);
  sub_graph2->FindNode("data1")->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  AttrUtils::SetInt(sub_graph2->FindNode("data1")->GetOpDesc(), "_parent_node_index", 0);
  sub_graph2->FindNode("data2")->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  sub_graph2->FindNode("shape")->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  sub_graph2->FindNode("split")->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  sub_graph2->FindNode("netoutput")->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);

  auto partitionedcall1 = graph->FindNode("partitionedcall1");
  partitionedcall1->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  partitionedcall1->GetOpDesc()->RegisterSubgraphIrName("subgraph", SubgraphType::kStatic);
  partitionedcall1->GetOpDesc()->AddSubgraphName(sub_graph1->GetName());
  partitionedcall1->GetOpDesc()->SetSubgraphInstanceName(0, sub_graph1->GetName());

  auto partitionedcall2 = graph->FindNode("partitionedcall2");
  partitionedcall2->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  partitionedcall2->GetOpDesc()->RegisterSubgraphIrName("subgraph", SubgraphType::kStatic);
  partitionedcall2->GetOpDesc()->AddSubgraphName(sub_graph2->GetName());
  partitionedcall2->GetOpDesc()->SetSubgraphInstanceName(0, sub_graph2->GetName());

  sub_graph1->SetParentNode(partitionedcall1);
  sub_graph1->SetParentGraph(graph);
  sub_graph2->SetParentNode(partitionedcall2);
  sub_graph2->SetParentGraph(graph);
  graph->AddSubgraph(sub_graph1);
  graph->AddSubgraph(sub_graph2);

  return graph;
}

/**
 *      netoutput                              netoutput
 *         |                                     /  \ 
 *      while_op                              mul    value_data
 *       |     \                              | \ 
 *     shape    \            netoutput        | one_tensor
 *       |       \               |            |
 *  cond_data   value_data   cond_data   cond_data
 */
ComputeGraphPtr BuildWhileGraph() {
  DEF_GRAPH(cond) {
    auto cond_data = OP_CFG(DATA).InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {});
    auto net_output = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_FLOAT, {});
    CHAIN(NODE("cond_data", cond_data)->NODE("cond_Node_Output", net_output));
  };

  DEF_GRAPH(body) {
    auto cond_data = OP_CFG(DATA).InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {});
    auto value_data = OP_CFG(DATA).InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_FLOAT, {1});
    auto const_data = OP_CFG(CONSTANT).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {});
    auto mul = OP_CFG(MUL).InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {});
    auto net_output = OP_CFG(NETOUTPUT).InCnt(2).OutCnt(2).TensorDesc(FORMAT_ND, DT_FLOAT, {1}).Build("body_Node_Output");

    CHAIN(NODE("body_arg_0", cond_data)->NODE("mul", mul)->NODE(net_output));
    CHAIN(NODE("one_tensor", const_data)->NODE("mul", mul));
    CHAIN(NODE("value_data", value_data)->NODE(net_output));
  };

  DEF_GRAPH(while_graph) {
    auto cond_data = OP_CFG(DATA).InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 0).TensorDesc(FORMAT_ND, DT_INT32, {});
    auto value_data = OP_CFG(DATA).InCnt(1).OutCnt(1).Attr(ATTR_NAME_INDEX, 1).TensorDesc(FORMAT_ND, DT_FLOAT, {1});
    auto while_op = OP_CFG(WHILE).InCnt(2).OutCnt(2).TensorDesc(FORMAT_ND, DT_FLOAT, {-1}).Build("while_op");
    auto net_output = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_FLOAT, {-1});

    CHAIN(NODE("arg_cond", cond_data)->NODE("shape", "Shape")->NODE(while_op));
    CHAIN(NODE("arg_value", value_data)->NODE(while_op)->EDGE(1, 0)->NODE("Node_Output", net_output));
  };

  auto cond_graph = ToComputeGraph(cond);
  cond_graph->SetGraphUnknownFlag(true);
  cond_graph->FindNode("cond_data")->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  cond_graph->FindNode("cond_Node_Output")->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);

  auto body_graph = ToComputeGraph(body);
  body_graph->SetGraphUnknownFlag(true);
  body_graph->FindNode("body_arg_0")->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  body_graph->FindNode("value_data")->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  body_graph->FindNode("one_tensor")->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  body_graph->FindNode("mul")->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  body_graph->FindNode("body_Node_Output")->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);

  auto root_graph = ToComputeGraph(while_graph);
  root_graph->SetGraphUnknownFlag(true);
  root_graph->FindNode("arg_cond")->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  root_graph->FindNode("arg_value")->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  root_graph->FindNode("shape")->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  root_graph->FindNode("Node_Output")->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  auto while_node = root_graph->FindNode("while_op");
  while_node->GetOpDesc()->RegisterSubgraphIrName("cond", SubgraphType::kStatic);
  while_node->GetOpDesc()->RegisterSubgraphIrName("body", SubgraphType::kStatic);
  while_node->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  while_node->GetOpDesc()->AddSubgraphName(cond_graph->GetName());
  while_node->GetOpDesc()->SetSubgraphInstanceName(0, cond_graph->GetName());
  while_node->GetOpDesc()->AddSubgraphName(body_graph->GetName());
  while_node->GetOpDesc()->SetSubgraphInstanceName(1, body_graph->GetName());
  cond_graph->SetParentNode(while_node);
  cond_graph->SetParentGraph(root_graph);
  body_graph->SetParentNode(while_node);
  body_graph->SetParentGraph(root_graph);
  root_graph->AddSubgraph(cond_graph);
  root_graph->AddSubgraph(body_graph);

  return root_graph;
}

/**
 *           netoutput
 *               |
 *            mapindex                             
 *            /      \
 *        data       const
 */
ComputeGraphPtr BuildGraphMapIndex() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data", "Data")
              ->NODE("mapindex", "MapIndex")
              ->NODE("netoutput", "NetOutput"));
    CHAIN(NODE("const", "Const")
              ->EDGE(0, 1)
              ->NODE("mapindex", "MapIndex"));
  };

  auto sub_graph2 = ToComputeGraph(g1);
  sub_graph2->SetGraphUnknownFlag(true);
  sub_graph2->FindNode("data")->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  sub_graph2->FindNode("const")->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  sub_graph2->FindNode("mapindex")->GetOpDesc()->SetOpKernelLibName(kEngineNameAiCore);
  sub_graph2->FindNode("netoutput")->GetOpDesc()->SetOpKernelLibName(kEngineNameGeLocal);
  bool is_execute_host = true;
  (void)ge::AttrUtils::SetBool(sub_graph2->FindNode("data")->GetOpDesc(), "_host_tensor", is_execute_host);
  return sub_graph2;
}

} // namespace
class UtestHostcpuEngineUpdatePass : public Test {
 public:
  void SetUp() {
    OpInfo host_op_info;
    host_op_info.engine = "DNN_VM_HOST_CPU";
    host_op_info.opKernelLib = "DNN_VM_HOST_CPU_OP_STORE";

    OpInfo aicore_op_info;
    aicore_op_info.engine = "AIcoreEngine";
    aicore_op_info.opKernelLib = "AIcoreEngine";
  
    OpInfo gelocal_op_info;
    gelocal_op_info.engine = "DNN_VM_GE_LOCAL_OP_STORE";
    gelocal_op_info.opKernelLib = "DNN_VM_GE_LOCAL_OP_STORE";

    OpsKernelManager::GetInstance().ops_kernel_info_["Data"].emplace_back(gelocal_op_info);
    OpsKernelManager::GetInstance().ops_kernel_info_["Const"].emplace_back(gelocal_op_info);
    OpsKernelManager::GetInstance().ops_kernel_info_["Shape"].emplace_back(gelocal_op_info);
    OpsKernelManager::GetInstance().ops_kernel_info_["Unsqueeze"].emplace_back(gelocal_op_info);
    OpsKernelManager::GetInstance().ops_kernel_info_["Reshape"].emplace_back(gelocal_op_info);

    OpsKernelManager::GetInstance().ops_kernel_info_["Gather"].emplace_back(host_op_info);
    OpsKernelManager::GetInstance().ops_kernel_info_["Gather"].emplace_back(aicore_op_info);

    OpsKernelManager::GetInstance().ops_kernel_info_["Split"].emplace_back(host_op_info);
    OpsKernelManager::GetInstance().ops_kernel_info_["Split"].emplace_back(aicore_op_info);

    OpsKernelManager::GetInstance().ops_kernel_info_["Concat"].emplace_back(host_op_info);
    OpsKernelManager::GetInstance().ops_kernel_info_["Concat"].emplace_back(aicore_op_info);

    OpsKernelManager::GetInstance().ops_kernel_info_["Mul"].emplace_back(host_op_info);
    OpsKernelManager::GetInstance().ops_kernel_info_["Mul"].emplace_back(aicore_op_info);
  }
};

TEST_F(UtestHostcpuEngineUpdatePass, TestSucc) {
  setenv("ENABLE_RUNTIME_V2", "1", 1);
  auto graph = BuildGraph1();
  EXPECT_NE(graph, nullptr);
  HostcpuEngineUpdatePass pass;
  EXPECT_EQ(graph->FindNode("data")->GetOpDesc()->GetOpKernelLibName(), kEngineNameGeLocal);
  EXPECT_EQ(graph->FindNode("const1")->GetOpDesc()->GetOpKernelLibName(), kEngineNameGeLocal);
  EXPECT_EQ(graph->FindNode("const2")->GetOpDesc()->GetOpKernelLibName(), kEngineNameGeLocal);
  EXPECT_EQ(graph->FindNode("shape1")->GetOpDesc()->GetOpKernelLibName(), kEngineNameGeLocal);
  EXPECT_EQ(graph->FindNode("shape2")->GetOpDesc()->GetOpKernelLibName(), kEngineNameGeLocal);
  EXPECT_EQ(graph->FindNode("gather1")->GetOpDesc()->GetOpKernelLibName(), kEngineNameAiCore);
  EXPECT_EQ(graph->FindNode("gather2")->GetOpDesc()->GetOpKernelLibName(), kEngineNameAiCore);
  EXPECT_EQ(graph->FindNode("unsqueeze1")->GetOpDesc()->GetOpKernelLibName(), kEngineNameGeLocal);
  EXPECT_EQ(graph->FindNode("unsqueeze2")->GetOpDesc()->GetOpKernelLibName(), kEngineNameGeLocal);
  EXPECT_EQ(graph->FindNode("concat")->GetOpDesc()->GetOpKernelLibName(), kEngineNameAiCore);
  EXPECT_EQ(graph->FindNode("reshape")->GetOpDesc()->GetOpKernelLibName(), kEngineNameGeLocal);
  EXPECT_EQ(graph->FindNode("nonzero")->GetOpDesc()->GetOpKernelLibName(), kEngineNameAiCore);

  SetNoStorage(graph->FindNode("nonzero")->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {-1, 2, 3, 4}, {-1, 2, 3, 4});
  SetShapeRangeNoStorage(graph->FindNode("nonzero")->GetOpDesc(), {1, 2, 3, 4}, {1, 2, 3, 4});

  NodeEngineMap node_atomic_engine_map;
  NodeEngineMap node_composite_engine_map;
  EXPECT_EQ(pass.Run(graph, node_atomic_engine_map, node_composite_engine_map), SUCCESS);
  EXPECT_EQ(graph->FindNode("data")->GetOpDesc()->GetOpKernelLibName(), kEngineNameGeLocal);
  EXPECT_EQ(graph->FindNode("const1")->GetOpDesc()->GetOpKernelLibName(), kEngineNameGeLocal);
  EXPECT_EQ(graph->FindNode("const2")->GetOpDesc()->GetOpKernelLibName(), kEngineNameGeLocal);
  EXPECT_EQ(graph->FindNode("shape1")->GetOpDesc()->GetOpKernelLibName(), kEngineNameGeLocal);
  EXPECT_EQ(graph->FindNode("shape2")->GetOpDesc()->GetOpKernelLibName(), kEngineNameGeLocal);
  EXPECT_EQ(graph->FindNode("unsqueeze1")->GetOpDesc()->GetOpKernelLibName(), kEngineNameGeLocal);
  EXPECT_EQ(graph->FindNode("unsqueeze2")->GetOpDesc()->GetOpKernelLibName(), kEngineNameGeLocal);
  EXPECT_EQ(graph->FindNode("reshape")->GetOpDesc()->GetOpKernelLibName(), kEngineNameGeLocal);

  EXPECT_EQ(graph->FindNode("gather1")->GetOpDesc()->GetOpKernelLibName(), "DNN_VM_HOST_CPU_OP_STORE");
  EXPECT_EQ(graph->FindNode("gather1")->GetOpDesc()->GetOpEngineName(), "DNN_VM_HOST_CPU");
  EXPECT_EQ(node_atomic_engine_map[graph->FindNode("gather1")], "DNN_VM_HOST_CPU");
  EXPECT_EQ(node_composite_engine_map[graph->FindNode("gather1")], "DNN_VM_HOST_CPU");
  bool is_small_shape = false;
  (void)ge::AttrUtils::GetBool(graph->FindNode("gather1")->GetOpDesc(),  "SmallShapeHostcpu", is_small_shape);
  EXPECT_EQ(is_small_shape, true);

  EXPECT_EQ(graph->FindNode("gather2")->GetOpDesc()->GetOpKernelLibName(), "DNN_VM_HOST_CPU_OP_STORE");
  EXPECT_EQ(graph->FindNode("gather2")->GetOpDesc()->GetOpEngineName(), "DNN_VM_HOST_CPU");
  EXPECT_EQ(node_atomic_engine_map[graph->FindNode("gather2")], "DNN_VM_HOST_CPU");
  EXPECT_EQ(node_composite_engine_map[graph->FindNode("gather2")], "DNN_VM_HOST_CPU");
  is_small_shape = false;
  (void)ge::AttrUtils::GetBool(graph->FindNode("gather2")->GetOpDesc(),  "SmallShapeHostcpu", is_small_shape);
  EXPECT_EQ(is_small_shape, true);

  EXPECT_EQ(graph->FindNode("concat")->GetOpDesc()->GetOpKernelLibName(), "DNN_VM_HOST_CPU_OP_STORE");
  EXPECT_EQ(graph->FindNode("concat")->GetOpDesc()->GetOpEngineName(), "DNN_VM_HOST_CPU");
  EXPECT_EQ(node_atomic_engine_map[graph->FindNode("concat")], "DNN_VM_HOST_CPU");
  EXPECT_EQ(node_composite_engine_map[graph->FindNode("concat")], "DNN_VM_HOST_CPU");
  is_small_shape = false;
  (void)ge::AttrUtils::GetBool(graph->FindNode("concat")->GetOpDesc(),  "SmallShapeHostcpu", is_small_shape);
  EXPECT_EQ(is_small_shape, true);

  auto graph2 = BuildGraph2();
  EXPECT_NE(graph2, nullptr);
  EXPECT_EQ(pass.Run(graph2, node_atomic_engine_map, node_composite_engine_map), SUCCESS);

  auto mapIndex_graph = BuildGraphMapIndex();
  EXPECT_NE(mapIndex_graph, nullptr);
  EXPECT_EQ(pass.Run(mapIndex_graph, node_atomic_engine_map, node_composite_engine_map), SUCCESS);

  auto while_graph = BuildWhileGraph();
  EXPECT_NE(while_graph, nullptr);
  EXPECT_EQ(pass.Run(while_graph, node_atomic_engine_map, node_composite_engine_map), SUCCESS);
  unsetenv("ENABLE_RUNTIME_V2");
}

TEST_F(UtestHostcpuEngineUpdatePass, CheckInputForHostExec) {
  HostcpuEngineUpdatePass pass;
  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("mapindex", "MapIndex");
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{8}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{1}), FORMAT_NCHW, DT_INT32));
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node = graph->AddNode(op_desc);
  EXPECT_EQ(pass.CheckInputForHostExec(node, 0), false);
}

TEST_F(UtestHostcpuEngineUpdatePass, IsExecOnHost) {
  HostcpuEngineUpdatePass pass;
  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("data", "Data");
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  AttrUtils::SetBool(op_desc, "_host_tensor", true);
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node = graph->AddNode(op_desc);
  EXPECT_EQ(pass.IsExecOnHost(node), true);
}

TEST_F(UtestHostcpuEngineUpdatePass, TestOolevelSucc) {
  setenv("ENABLE_RUNTIME_V2", "1", 1);
  std::map<std::string, std::string> ge_options = {{ge::OO_LEVEL, "O1"}};
  const std::unordered_map<std::string, OoInfo> &registered_opt_table =
    ge::OptionRegistry::GetInstance().GetRegisteredOptTable();
  ge::GetThreadLocalContext().GetOo().Initialize(ge_options, registered_opt_table);
  auto graph = BuildGraph1();
  EXPECT_NE(graph, nullptr);
  HostcpuEngineUpdatePass pass;
  EXPECT_EQ(graph->FindNode("data")->GetOpDesc()->GetOpKernelLibName(), kEngineNameGeLocal);
  EXPECT_EQ(graph->FindNode("const1")->GetOpDesc()->GetOpKernelLibName(), kEngineNameGeLocal);
  EXPECT_EQ(graph->FindNode("const2")->GetOpDesc()->GetOpKernelLibName(), kEngineNameGeLocal);
  EXPECT_EQ(graph->FindNode("shape1")->GetOpDesc()->GetOpKernelLibName(), kEngineNameGeLocal);
  EXPECT_EQ(graph->FindNode("shape2")->GetOpDesc()->GetOpKernelLibName(), kEngineNameGeLocal);
  EXPECT_EQ(graph->FindNode("gather1")->GetOpDesc()->GetOpKernelLibName(), kEngineNameAiCore);
  EXPECT_EQ(graph->FindNode("gather2")->GetOpDesc()->GetOpKernelLibName(), kEngineNameAiCore);
  EXPECT_EQ(graph->FindNode("unsqueeze1")->GetOpDesc()->GetOpKernelLibName(), kEngineNameGeLocal);
  EXPECT_EQ(graph->FindNode("unsqueeze2")->GetOpDesc()->GetOpKernelLibName(), kEngineNameGeLocal);
  EXPECT_EQ(graph->FindNode("concat")->GetOpDesc()->GetOpKernelLibName(), kEngineNameAiCore);
  EXPECT_EQ(graph->FindNode("reshape")->GetOpDesc()->GetOpKernelLibName(), kEngineNameGeLocal);
  EXPECT_EQ(graph->FindNode("nonzero")->GetOpDesc()->GetOpKernelLibName(), kEngineNameAiCore);

  SetNoStorage(graph->FindNode("nonzero")->GetOpDesc(), ge::FORMAT_ND, DT_FLOAT, {-1, 2, 3, 4}, {-1, 2, 3, 4});
  SetShapeRangeNoStorage(graph->FindNode("nonzero")->GetOpDesc(), {1, 2, 3, 4}, {1, 2, 3, 4});

  NodeEngineMap node_atomic_engine_map;
  NodeEngineMap node_composite_engine_map;
  EXPECT_EQ(pass.Run(graph, node_atomic_engine_map, node_composite_engine_map), SUCCESS);
  EXPECT_EQ(graph->FindNode("data")->GetOpDesc()->GetOpKernelLibName(), kEngineNameGeLocal);
  EXPECT_EQ(graph->FindNode("const1")->GetOpDesc()->GetOpKernelLibName(), kEngineNameGeLocal);
  EXPECT_EQ(graph->FindNode("const2")->GetOpDesc()->GetOpKernelLibName(), kEngineNameGeLocal);
  EXPECT_EQ(graph->FindNode("shape1")->GetOpDesc()->GetOpKernelLibName(), kEngineNameGeLocal);
  EXPECT_EQ(graph->FindNode("shape2")->GetOpDesc()->GetOpKernelLibName(), kEngineNameGeLocal);
  EXPECT_EQ(graph->FindNode("unsqueeze1")->GetOpDesc()->GetOpKernelLibName(), kEngineNameGeLocal);
  EXPECT_EQ(graph->FindNode("unsqueeze2")->GetOpDesc()->GetOpKernelLibName(), kEngineNameGeLocal);
  EXPECT_EQ(graph->FindNode("reshape")->GetOpDesc()->GetOpKernelLibName(), kEngineNameGeLocal);

  EXPECT_EQ(graph->FindNode("gather1")->GetOpDesc()->GetOpKernelLibName(), kEngineNameAiCore);
  unsetenv("ENABLE_RUNTIME_V2");
}

} // namespace ge

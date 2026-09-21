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
#include <nlohmann/json.hpp>
#include <iostream>
#include <string>
#include <vector>

#define protected public
#define private public
#include "common/graph_comm.h"
#include "common/aicore_util_constants.h"
#include "common/aicore_util_attr_define.h"
#include "compute_graph.h"
#include "graph/node.h"
#include "graph/op_desc.h"
#include "graph/debug/ge_attr_define.h"
#include "../../../../graph_constructor/graph_constructor.h"
#undef protected
#undef private

using namespace std;
using namespace fe;
using namespace ge;

class UTEST_graph_common_unittest : public testing::Test
{
 public:
  std::shared_ptr<fe::GraphComm> graph_comm_ptr_;

 protected:
  void SetUp()
  {
    graph_comm_ptr_ = std::make_shared<fe::GraphComm>(fe::AI_CORE_NAME);
  }

  void TearDown()
  {
  }

  static void CreateGraph(ComputeGraphPtr graph) {
    OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BN");
    OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Cast");
    OpDescPtr out_op = std::make_shared<OpDesc>("out", "NetOutput");

    // add descriptor
    vector<int64_t> dims = {1,2,3,4};
    GeShape shape(dims);

    GeTensorDesc in_desc1(shape);
    in_desc1.SetFormat(FORMAT_FRACTAL_NZ);
    in_desc1.SetDataType(DT_FLOAT16);
    bn_op->AddInputDesc("x", in_desc1);

    GeTensorDesc out_desc1(shape);
    out_desc1.SetFormat(FORMAT_FRACTAL_NZ);
    out_desc1.SetDataType(DT_FLOAT16);
    bn_op->AddOutputDesc("y", out_desc1);

    GeTensorDesc in_desc2(shape);
    in_desc2.SetFormat(FORMAT_FRACTAL_NZ);
    in_desc2.SetDataType(DT_FLOAT16);
    relu_op->AddInputDesc("x", in_desc2);

    GeTensorDesc out_desc2(shape);
    out_desc2.SetFormat(FORMAT_FRACTAL_NZ);
    out_desc2.SetDataType(DT_FLOAT16);
    relu_op->AddOutputDesc("y", out_desc2);

    GeTensorDesc in_desc3(shape);
    in_desc3.SetFormat(FORMAT_FRACTAL_NZ);
    in_desc3.SetDataType(DT_FLOAT16);
    out_op->AddInputDesc("x", in_desc3);

    GeTensorDesc out_desc3(shape);
    out_desc3.SetFormat(FORMAT_FRACTAL_NZ);
    out_desc3.SetDataType(DT_FLOAT16);
    out_op->AddOutputDesc("y", out_desc3);


    ge::AttrUtils::SetInt(bn_op, fe::FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));
    ge::AttrUtils::SetInt(relu_op, fe::FE_IMPLY_TYPE, static_cast<int>(EN_IMPL_HW_TBE));

    NodePtr bn_node = graph->AddNode(bn_op);
    NodePtr relu_node = graph->AddNode(relu_op);
    NodePtr out_node = graph->AddNode(out_op);

    GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0), out_node->GetInDataAnchor(0));
  }

  static void CreateGraphControlEdge(ComputeGraphPtr &graph) {
    ge::AttrUtils::SetStr(graph, ATTR_NAME_SESSION_GRAPH_ID, "0_0");
    ge::GeShape shape = ge::GeShape({3, 12, 5, 6});
    GraphConstructor test(graph, "", ge::FORMAT_NCHW, ge::DT_FLOAT, shape);
    test.AddOpDesc("a", "Test", 1, 1)
        .SetInputs({"Data0"})
        .AddOpDesc("b", "Test", 1, 1)
        .SetInputs({"a"})
        .AddOpDesc("netout", "NetOutput", 0, 0)
        .SetInput("netout:-1", "b:-1");
  }
};

TEST_F(UTEST_graph_common_unittest, EstablishConnection_1)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BN");
  OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Cast");
  vector<int64_t> dims = {1,2,3,4};
  GeShape shape(dims);
  GeTensorDesc desc(shape);
  bn_op->AddInputDesc("x", desc);
  bn_op->AddOutputDesc("y", desc);
  relu_op->AddInputDesc("x", desc);
  relu_op->AddOutputDesc("y", desc);
  NodePtr bn_node = graph->AddNode(bn_op);
  NodePtr relu_node = graph->AddNode(relu_op);
  ge::GraphUtils::AddEdge(bn_node->GetOutControlAnchor(), relu_node->GetInControlAnchor());
  std::unordered_set<std::string> node_name_set = {"test"};
  ge::CompleteGraphBuilder builder("builder", false);
  Status ret = graph_comm_ptr_->EstablishConnection(bn_node, node_name_set, builder);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(UTEST_graph_common_unittest, EstablishConnection_2)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BN");
  OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Cast");
  vector<int64_t> dims = {1,2,3,4};
  GeShape shape(dims);
  GeTensorDesc desc(shape);
  bn_op->AddInputDesc("x", desc);
  bn_op->AddOutputDesc("y", desc);
  relu_op->AddInputDesc("x", desc);
  relu_op->AddOutputDesc("y", desc);
  NodePtr bn_node = graph->AddNode(bn_op);
  NodePtr relu_node = graph->AddNode(relu_op);
  ge::GraphUtils::AddEdge(bn_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
  std::unordered_set<std::string> node_name_set = {"test"};
  ge::CompleteGraphBuilder builder("builder", false);
  Status ret = graph_comm_ptr_->EstablishConnection(bn_node, node_name_set, builder);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(UTEST_graph_common_unittest, EstablishConnection)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BN");
  OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Cast");
  vector<int64_t> dims = {1,2,3,4};
  GeShape shape(dims);
  GeTensorDesc desc(shape);
  bn_op->AddInputDesc("x", desc);
  bn_op->AddOutputDesc("y", desc);
  relu_op->AddInputDesc("x", desc);
  relu_op->AddOutputDesc("y", desc);
  NodePtr bn_node = graph->AddNode(bn_op);
  NodePtr relu_node = graph->AddNode(relu_op);
  ge::GraphUtils::AddEdge(bn_node->GetOutControlAnchor(), relu_node->GetInControlAnchor());
  std::unordered_set<std::string> node_name_set = {"relu"};
  ge::CompleteGraphBuilder builder("builder", false);
  Status ret = graph_comm_ptr_->EstablishConnection(bn_node, node_name_set, builder);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(UTEST_graph_common_unittest, nullptr_case)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr bn_op = std::make_shared<OpDesc>("batchnormal", "BN");
  OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Cast");
  vector<int64_t> dims = {1,2,3,4};
  GeShape shape(dims);
  GeTensorDesc desc(shape);
  bn_op->AddInputDesc("x", desc);
  bn_op->AddOutputDesc("y", desc);
  relu_op->AddInputDesc("x", desc);
  relu_op->AddOutputDesc("y", desc);
  NodePtr bn_node = graph->AddNode(bn_op);
  NodePtr relu_node = graph->AddNode(relu_op);
  ge::GraphUtils::AddEdge(bn_node->GetOutControlAnchor(), relu_node->GetInControlAnchor());
  std::vector<ge::NodePtr> node_vec = {nullptr};
  ge::NodePtr node = graph_comm_ptr_->TransSingleSubGraph(*(graph.get()), node_vec);
  EXPECT_EQ(node, nullptr);
}

TEST_F(UTEST_graph_common_unittest, output_only_control_edges_case)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateGraphControlEdge(graph);
  std::vector<ge::NodePtr> node_vec = {};
  for (auto &node : graph->GetDirectNode()) {
    auto op_desc_ptr = node->GetOpDesc();
    if (op_desc_ptr->GetName() == "a" || op_desc_ptr->GetName() == "b") {
      node_vec.emplace_back(node);
    }
  }

  ge::NodePtr node = graph_comm_ptr_->TransSingleSubGraph(*(graph.get()), node_vec);
  EXPECT_EQ(graph->TopologicalSorting(), fe::SUCCESS);
  EXPECT_NE(node, nullptr);
}

TEST_F(UTEST_graph_common_unittest, CreateFunctionOpSubGraph)
{
  ge::NodePtr funcnode = nullptr;
  std::vector<ge::NodePtr> node_vec = {};
  vector<fe::FusionDataFlow> input_edge_list;
  vector<fe::FusionDataFlow> output_edge_list;
  vector<fe::FusionDataFlow> output_ctrl_edge_list;
  Status ret = graph_comm_ptr_->CreateFunctionOpSubGraph(funcnode, node_vec, input_edge_list,
                                                         output_edge_list, output_ctrl_edge_list);
  EXPECT_EQ(ret, fe::FAILED);
}

TEST_F(UTEST_graph_common_unittest, CreateFunctionOp_1)
{
  std::vector<ge::NodePtr> node_vec = {};
  ge::OpDescPtr opdesc = graph_comm_ptr_->CreateFunctionOp(node_vec);
  EXPECT_EQ(opdesc, nullptr);
}

TEST_F(UTEST_graph_common_unittest, CreateFunctionOp_2)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  string name = "test_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";
  OpDescPtr bn_op = std::make_shared<OpDesc>(name, "BN");
  OpDescPtr cast1_op = std::make_shared<OpDesc>(name, "Cast");
  OpDescPtr cast2_op = std::make_shared<OpDesc>(name, "Cast");
  OpDescPtr cast3_op = std::make_shared<OpDesc>(name, "Cast");
  OpDescPtr cast4_op = std::make_shared<OpDesc>(name, "Cast");
  OpDescPtr cast5_op = std::make_shared<OpDesc>(name, "Cast");
  NodePtr bn_node = graph->AddNode(bn_op);
  NodePtr cast1_node = graph->AddNode(cast1_op);
  NodePtr cast2_node = graph->AddNode(cast2_op);
  NodePtr cast3_node = graph->AddNode(cast3_op);
  NodePtr cast4_node = graph->AddNode(cast4_op);
  NodePtr cast5_node = graph->AddNode(cast5_op);
  std::vector<ge::NodePtr> node_vec = {bn_node, cast1_node, cast2_node, cast3_node, cast4_node, cast5_node};
  ge::OpDescPtr opdesc = graph_comm_ptr_->CreateFunctionOp(node_vec);
  EXPECT_EQ(opdesc->GetName(), name + "_function_op");
}

TEST_F(UTEST_graph_common_unittest, CreateFunctionOp_3)
{
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr bn_op = std::make_shared<OpDesc>("bn", "BN");
  OpDescPtr cast1_op = std::make_shared<OpDesc>("cast", "Cast");
  NodePtr bn_node = graph->AddNode(bn_op);
  NodePtr cast1_node = graph->AddNode(cast1_op);
  ge::AttrUtils::SetStr(bn_node->GetOpDesc(), ge::ATTR_NAME_SESSION_GRAPH_ID, "1");
  std::vector<ge::NodePtr> node_vec = {bn_node, cast1_node};
  ge::OpDescPtr opdesc = graph_comm_ptr_->CreateFunctionOp(node_vec);
  EXPECT_NE(opdesc, nullptr);
}

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// To test the SetInputOutputOffsetPass

#include <string>
#include <gtest/gtest.h>
#include "common/plugin/ge_make_unique_util.h"
#include "runtime/mem.h"
#include "graph/passes/memory_conflict/set_input_output_offset_pass.h"

using namespace domi;
using namespace ge;

class UTEST_graph_passes_set_input_output_offset_pass : public testing::Test
{
 protected:
  void SetUp() {}
  void TearDown() {}
 public:
/*
*
*              Data
*               |
*             Conv2D
*/
  void MakeGraphDataInParent1(ComputeGraphPtr &graph) {
    auto desc_ptr = MakeShared<ge::GeTensorDesc>();
    auto desc = *desc_ptr;

    OpDescPtr op_desc_data = MakeShared<OpDesc>("Data", DATA);
    op_desc_data->AddOutputDesc(desc);

    OpDescPtr op_desc_out = MakeShared<OpDesc>("Conv2D", CONV2D);
    op_desc_out->AddInputDesc(desc);

    vector<int> connect_input = {0};
    AttrUtils::SetListInt(op_desc_out, ATTR_NAME_NODE_CONNECT_INPUT, connect_input);
    std::vector<int64_t> memory_type = {RT_MEMORY_L2};
    ge::AttrUtils::SetListInt(op_desc_out, ATTR_NAME_INPUT_MEM_TYPE_LIST, memory_type);
    op_desc_data->SetOutputOffset({0});
    op_desc_out->SetInputOffset({0});

    NodePtr data_node = graph->AddNode(op_desc_data);
    NodePtr out_node = graph->AddNode(op_desc_out);
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), out_node->GetInDataAnchor(0));
  }

/*
*
*              Data
*               |\
*             AllReduce
*/
  void MakeGraphDataInParent2(ComputeGraphPtr &graph) {
    auto desc_ptr = MakeShared<ge::GeTensorDesc>();
    auto desc = *desc_ptr;

    OpDescPtr op_desc_data = MakeShared<OpDesc>("Data", DATA);
    op_desc_data->AddOutputDesc(desc);
    op_desc_data->SetOutputOffset({0});

    OpDescPtr op_desc_out = MakeShared<OpDesc>("AllReduce", HCOMALLREDUCE);
    op_desc_out->AddInputDesc(desc);
    op_desc_out->AddInputDesc(desc);
    op_desc_out->SetInputOffset({0, 10});

    vector<int> connect_input = {0, 1};
    AttrUtils::SetListInt(op_desc_out, ATTR_NAME_NODE_CONNECT_INPUT, connect_input);
    bool is_input_continuous = true;
    ge::AttrUtils::SetBool(op_desc_out, ATTR_NAME_CONTINUOUS_INPUT, is_input_continuous);

    NodePtr data_node = graph->AddNode(op_desc_data);
    NodePtr out_node = graph->AddNode(op_desc_out);
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), out_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), out_node->GetInDataAnchor(1));
  }

/*
*              Add
*               |
*          Phonyconcat
*             /     \
*           relu   Netoutput
*/
  void MakeGraphDataInParent3(ComputeGraphPtr &graph) {
    auto desc_ptr = MakeShared<ge::GeTensorDesc>();
    auto desc = *desc_ptr;
    OpDescPtr op_desc_data = MakeShared<OpDesc>("Add", ADD);
    op_desc_data->AddOutputDesc(desc);

    OpDescPtr op_desc_input = MakeShared<OpDesc>("Phonyconcat", CONCAT);
    op_desc_input->AddOutputDesc(desc);
    op_desc_input->AddInputDesc(desc);

    OpDescPtr op_desc_out = MakeShared<OpDesc>("Netoutput", NETOUTPUT);
    op_desc_out->AddInputDesc(desc);

    OpDescPtr op_desc_out_relu = MakeShared<OpDesc>("Relu", RELU);
    op_desc_out_relu->AddInputDesc(desc);

    vector<int> connect_output = {0};
    AttrUtils::SetListInt(op_desc_input, ATTR_NAME_NODE_CONNECT_OUTPUT, connect_output);
    bool attr_no_task = true;
    ge::AttrUtils::SetBool(op_desc_input, ATTR_NAME_NOTASK, attr_no_task);
    bool is_input_continuous = true;
    ge::AttrUtils::SetBool(op_desc_input, ATTR_NAME_CONTINUOUS_INPUT, is_input_continuous);
    op_desc_input->SetOutputOffset({0});
    op_desc_data->SetOutputOffset({0});

    std::vector<int64_t> offsets_for_fusion = {0};
    AttrUtils::SetListInt(op_desc_data, ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION, offsets_for_fusion);

    NodePtr data_node = graph->AddNode(op_desc_data);
    NodePtr phony_node = graph->AddNode(op_desc_input);
    NodePtr out_node = graph->AddNode(op_desc_out);
    NodePtr out_relu_node = graph->AddNode(op_desc_out_relu);
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), phony_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(phony_node->GetOutDataAnchor(0), out_relu_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(phony_node->GetOutDataAnchor(0), out_node->GetInDataAnchor(0));
  }

/* 设置一个Graph，使其拥有如下网络结构
*          HcomBroadcast
*               |
*           Netoutput
*/
  void MakeGraphDataInParent4(ComputeGraphPtr &graph) {
    auto desc_ptr = MakeShared<ge::GeTensorDesc>();
    auto desc = *desc_ptr;

    OpDescPtr op_desc_input = MakeShared<OpDesc>("HcomBroadcast", HCOMBROADCAST);
    op_desc_input->AddOutputDesc(desc);

    OpDescPtr op_desc_out = MakeShared<OpDesc>("Netoutput", NETOUTPUT);
    op_desc_out->AddInputDesc(desc);

    vector<int> connect_output = {0};
    AttrUtils::SetListInt(op_desc_input, ATTR_NAME_NODE_CONNECT_OUTPUT, connect_output);
    bool attr_no_task = false;
    ge::AttrUtils::SetBool(op_desc_input, ATTR_NAME_NOTASK, attr_no_task);
    bool is_output_continuous = true;
    ge::AttrUtils::SetBool(op_desc_input, ATTR_NAME_CONTINUOUS_OUTPUT, is_output_continuous);
    op_desc_input->SetOutputOffset({0});
    op_desc_out->SetInputOffset({0});

    NodePtr hcom_node = graph->AddNode(op_desc_input);
    NodePtr out_node = graph->AddNode(op_desc_out);
    GraphUtils::AddEdge(hcom_node->GetOutDataAnchor(0), out_node->GetInDataAnchor(0));
  }
  /* 设置一个Graph，使其拥有如下网络结构
  *          HcomBroadcast
  *           /    |
  *               Netoutput
  */
  void HcomOutputSuspend(ComputeGraphPtr &graph) {
    auto desc_ptr = MakeShared<ge::GeTensorDesc>();
    auto desc = *desc_ptr;

    OpDescPtr op_desc_input = MakeShared<OpDesc>("HcomBroadcast", HCOMBROADCAST);
    op_desc_input->AddOutputDesc(desc);
    op_desc_input->AddOutputDesc(desc);

    OpDescPtr op_desc_out = MakeShared<OpDesc>("Netoutput", NETOUTPUT);
    op_desc_out->AddInputDesc(desc);

    vector<int> connect_output = {1};
    AttrUtils::SetListInt(op_desc_input, ATTR_NAME_NODE_CONNECT_OUTPUT, connect_output);
    bool attr_no_task = false;
    ge::AttrUtils::SetBool(op_desc_input, ATTR_NAME_NOTASK, attr_no_task);
    bool is_output_continuous = true;
    ge::AttrUtils::SetBool(op_desc_input, ATTR_NAME_CONTINUOUS_OUTPUT, is_output_continuous);
    op_desc_input->SetOutputOffset({0, 1024});
    op_desc_out->SetInputOffset({0});

    NodePtr hcom_node = graph->AddNode(op_desc_input);
    NodePtr out_node = graph->AddNode(op_desc_out);
    GraphUtils::AddEdge(hcom_node->GetOutDataAnchor(1), out_node->GetInDataAnchor(0));
  }
};

TEST_F(UTEST_graph_passes_set_input_output_offset_pass, SetInputOffset_succ){
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  MakeGraphDataInParent1(graph);
  graph->TopologicalSorting();
  SetInputOutputOffsetPass setInputOutputOffsetPass;
  Status ret = setInputOutputOffsetPass.Run(graph);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(UTEST_graph_passes_set_input_output_offset_pass, SetInputOffset_failed){
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  MakeGraphDataInParent2(graph);
  graph->TopologicalSorting();
  SetInputOutputOffsetPass setInputOutputOffsetPass;
  Status ret = setInputOutputOffsetPass.Run(graph);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(UTEST_graph_passes_set_input_output_offset_pass, SetOutputOffsetForConcat_succ){
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  MakeGraphDataInParent3(graph);
  graph->TopologicalSorting();
  SetInputOutputOffsetPass setInputOutputOffsetPass;
  Status ret = setInputOutputOffsetPass.Run(graph);
  EXPECT_EQ(ret, ge::SUCCESS);
  auto net_node = graph->FindFirstNodeMatchType(NETOUTPUT);
  EXPECT_NE(net_node, nullptr);
  auto opdesc = net_node->GetOpDesc();
  EXPECT_NE(opdesc, nullptr);
  EXPECT_TRUE(opdesc->HasAttr(ATTR_ZERO_COPY_BASIC_OFFSET));
}

TEST_F(UTEST_graph_passes_set_input_output_offset_pass, SetOutputOffsetForHcom_succ){
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  MakeGraphDataInParent4(graph);
  graph->TopologicalSorting();
  SetInputOutputOffsetPass setInputOutputOffsetPass;
  Status ret = setInputOutputOffsetPass.Run(graph);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(UTEST_graph_passes_set_input_output_offset_pass, SetOutputOffsetForHcom_HcomOutputSuspend){
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  HcomOutputSuspend(graph);
  graph->TopologicalSorting();
  SetInputOutputOffsetPass setInputOutputOffsetPass;
  Status ret = setInputOutputOffsetPass.Run(graph);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(UTEST_graph_passes_set_input_output_offset_pass, SetOutputOffsetForHcom_HcomOutputNotConnectNetoutput){
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  HcomOutputSuspend(graph);
  graph->TopologicalSorting();
  auto hcom = graph->FindNode("HcomBroadcast");
  vector<int> connect_output = {0};
  AttrUtils::SetListInt(hcom->GetOpDescBarePtr(), ATTR_NAME_NODE_CONNECT_OUTPUT, connect_output);
  SetInputOutputOffsetPass setInputOutputOffsetPass;
  Status ret = setInputOutputOffsetPass.Run(graph);
  EXPECT_EQ(ret, ge::SUCCESS);
}

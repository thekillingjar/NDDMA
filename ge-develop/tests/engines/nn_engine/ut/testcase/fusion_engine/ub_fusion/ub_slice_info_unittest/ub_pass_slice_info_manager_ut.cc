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
#include "graph/node.h"
#include "graph/compute_graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "ub_pass_slice_info/ub_pass_slice_info_manager.h"

using namespace std;
using namespace ge;
namespace fe {
class UbPassSliceInfoManagerUT : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "UbPassSliceInfoManagerUT SetUpTestCase" << std::endl;
  }
  static void TearDownTestCase() {
    std::cout << "UbPassSliceInfoManagerUT TearDownTestCase" << std::endl;
  }
  void SetUp() {

  }
  void TearDown() {

  }

  ge::ComputeGraphPtr CreateGraphWithOneInput(const std::string &op_type, const std::string &op_pattern, const bool is_dual_output,
                                              vector<ge::NodePtr> &fusin_nodes) {
    vector<int64_t> dims = {3, 4, 5, 6};
    ge::GeShape shape(dims);
    ge::GeTensorDesc tensor_desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
    tensor_desc.SetOriginShape(shape);
    tensor_desc.SetOriginDataType(ge::DT_FLOAT);
    tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);

    OpDescPtr data1_op = std::make_shared<OpDesc>("data1", "PlaceHolder");
    OpDescPtr data2_op = std::make_shared<OpDesc>("data2", "PlaceHolder");
    OpDescPtr conv_op = std::make_shared<OpDesc>("conv", "Conv2D");
    OpDescPtr relu_op = std::make_shared<OpDesc>("relu", op_type);

    data1_op->AddOutputDesc(tensor_desc);
    data2_op->AddOutputDesc(tensor_desc);
    conv_op->AddInputDesc(tensor_desc);
    conv_op->AddInputDesc(tensor_desc);
    conv_op->AddOutputDesc(tensor_desc);
    relu_op->AddInputDesc(tensor_desc);
    relu_op->AddOutputDesc(tensor_desc);

    AttrUtils::SetInt(conv_op, "_fe_imply_type", 6);
    AttrUtils::SetInt(relu_op, "_fe_imply_type", 6);

    AttrUtils::SetListInt(conv_op, "strides", {1, 1, 1, 1});
    AttrUtils::SetListInt(conv_op, "pads", {1, 1, 1, 1});
    AttrUtils::SetListInt(conv_op, "dilations", {1, 1, 1, 1});

    string op_slice_info = "{\"_op_slice_info\": {\"splitMaps\": [{\"inputList\": [{\"idx\": 0, \"axis\": [0], \"headOverLap\": [-1], \"tailOverLap\": [-1]}], \"outputList\": [{\"idx\": 0, \"axis\": [0]}]}, {\"inputList\": [{\"idx\": 0, \"axis\": [2], \"headOverLap\": [-1], \"tailOverLap\": [-1]}], \"outputList\": [{\"idx\": 0, \"axis\": [2]}]}, {\"inputList\": [{\"idx\": 0, \"axis\": [3], \"headOverLap\": [-1], \"tailOverLap\": [-1]}], \"outputList\": [{\"idx\": 0, \"axis\": [3]}]}]}}";
    AttrUtils::SetStr(conv_op, "_op_slice_info", op_slice_info);
    AttrUtils::SetStr(relu_op, "_pattern", op_pattern);

    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    NodePtr data1_node = graph->AddNode(data1_op);
    NodePtr data2_node = graph->AddNode(data2_op);
    NodePtr conv_node = graph->AddNode(conv_op);
    NodePtr relu_node = graph->AddNode(relu_op);
    GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data2_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));

    fusin_nodes.push_back(conv_node);
    fusin_nodes.push_back(relu_node);

    if (is_dual_output) {
      OpDescPtr relu1_op = std::make_shared<OpDesc>("relu1", "Relu");
      OpDescPtr relu2_op = std::make_shared<OpDesc>("relu2", "Relu");
      relu1_op->AddInputDesc(tensor_desc);
      relu1_op->AddOutputDesc(tensor_desc);
      relu2_op->AddInputDesc(tensor_desc);
      relu2_op->AddOutputDesc(tensor_desc);
      NodePtr relu1_node = graph->AddNode(relu1_op);
      NodePtr relu2_node = graph->AddNode(relu2_op);
      GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0), relu1_node->GetInDataAnchor(0));
      GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0), relu2_node->GetInDataAnchor(0));
      fusin_nodes.push_back(relu1_node);
    }

    graph->TopologicalSorting();
    AttrUtils::SetStr(graph, ge::ATTR_NAME_SESSION_GRAPH_ID, "11_22");
    return graph;
  }

  ge::ComputeGraphPtr CreateGraphWithTwoInput(const std::string &op_type, const std::string &op_pattern, const bool is_dual_output,
                                              vector<ge::NodePtr> &fusin_nodes) {
    vector<int64_t> dims = {3, 4, 5, 6};
    ge::GeShape shape(dims);
    ge::GeTensorDesc tensor_desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
    tensor_desc.SetOriginShape(shape);
    tensor_desc.SetOriginDataType(ge::DT_FLOAT);
    tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);

    vector<int64_t> dims_5hd = {3, 1, 5, 6, 16};
    ge::GeShape shape_5hd(dims_5hd);
    ge::GeTensorDesc tensor_desc_5hd(shape_5hd, ge::FORMAT_NC1HWC0, ge::DT_FLOAT);
    tensor_desc_5hd.SetOriginShape(shape);
    tensor_desc_5hd.SetOriginDataType(ge::DT_FLOAT);
    tensor_desc_5hd.SetOriginFormat(ge::FORMAT_NCHW);

    OpDescPtr data1_op = std::make_shared<OpDesc>("data1", "PlaceHolder");
    OpDescPtr data2_op = std::make_shared<OpDesc>("data2", "PlaceHolder");
    OpDescPtr data3_op = std::make_shared<OpDesc>("data3", "PlaceHolder");
    OpDescPtr conv_op = std::make_shared<OpDesc>("conv", "Conv2D");
    OpDescPtr relu_op = std::make_shared<OpDesc>("relu", op_type);

    data1_op->AddOutputDesc(tensor_desc_5hd);
    data2_op->AddOutputDesc(tensor_desc);
    data3_op->AddOutputDesc(tensor_desc);
    conv_op->AddInputDesc(tensor_desc_5hd);
    conv_op->AddInputDesc(tensor_desc);
    conv_op->AddOutputDesc(tensor_desc_5hd);
    relu_op->AddInputDesc(tensor_desc_5hd);
    relu_op->AddInputDesc(tensor_desc);
    relu_op->AddOutputDesc(tensor_desc_5hd);

    AttrUtils::SetInt(conv_op, "_fe_imply_type", 6);
    AttrUtils::SetInt(relu_op, "_fe_imply_type", 6);

    AttrUtils::SetListInt(conv_op, "strides", {1, 1, 1, 1});
    AttrUtils::SetListInt(conv_op, "pads", {1, 1, 1, 1});
    AttrUtils::SetListInt(conv_op, "dilations", {1, 1, 1, 1});

    string op_slice_info = "{\"_op_slice_info\": {\"splitMaps\": [{\"inputList\": [{\"idx\": 0, \"axis\": [0], \"headOverLap\": [-1], \"tailOverLap\": [-1]}], \"outputList\": [{\"idx\": 0, \"axis\": [0]}]}, {\"inputList\": [{\"idx\": 0, \"axis\": [2], \"headOverLap\": [-1], \"tailOverLap\": [-1]}], \"outputList\": [{\"idx\": 0, \"axis\": [2]}]}, {\"inputList\": [{\"idx\": 0, \"axis\": [3], \"headOverLap\": [-1], \"tailOverLap\": [-1]}], \"outputList\": [{\"idx\": 0, \"axis\": [3]}]}]}}";
    AttrUtils::SetStr(conv_op, "_op_slice_info", op_slice_info);
    AttrUtils::SetStr(relu_op, "_pattern", op_pattern);

    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    NodePtr data1_node = graph->AddNode(data1_op);
    NodePtr data2_node = graph->AddNode(data2_op);
    NodePtr data3_node = graph->AddNode(data3_op);
    NodePtr conv_node = graph->AddNode(conv_op);
    NodePtr relu_node = graph->AddNode(relu_op);
    GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data2_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data3_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(1));

    fusin_nodes.push_back(conv_node);
    fusin_nodes.push_back(relu_node);

    if (is_dual_output) {
      OpDescPtr relu1_op = std::make_shared<OpDesc>("relu1", "Relu");
      OpDescPtr relu2_op = std::make_shared<OpDesc>("relu2", "Relu");
      relu1_op->AddInputDesc(tensor_desc_5hd);
      relu1_op->AddOutputDesc(tensor_desc_5hd);
      relu2_op->AddInputDesc(tensor_desc_5hd);
      relu2_op->AddOutputDesc(tensor_desc_5hd);
      NodePtr relu1_node = graph->AddNode(relu1_op);
      NodePtr relu2_node = graph->AddNode(relu2_op);
      GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0), relu1_node->GetInDataAnchor(0));
      GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0), relu2_node->GetInDataAnchor(0));
      fusin_nodes.push_back(relu2_node);
    }

    graph->TopologicalSorting();
    AttrUtils::SetStr(graph, ge::ATTR_NAME_SESSION_GRAPH_ID, "11_22");
    return graph;
  }
};

TEST_F(UbPassSliceInfoManagerUT, set_slice_info_for_fusion_nodes_1) {
  vector<ge::NodePtr> fusin_nodes;
  ComputeGraphPtr graph = CreateGraphWithTwoInput("Mul", "Broadcast", false, fusin_nodes);
  EXPECT_NE(graph, nullptr);
  EXPECT_EQ(UbPassSliceInfoManager::SetSliceInfoForFusionNodes(fusin_nodes), fe::SUCCESS);
}

TEST_F(UbPassSliceInfoManagerUT, set_slice_info_for_fusion_nodes_2) {
  vector<ge::NodePtr> fusin_nodes;
  ComputeGraphPtr graph = CreateGraphWithOneInput("Relu", "bn_reduce", false, fusin_nodes);
  EXPECT_NE(graph, nullptr);
  EXPECT_EQ(UbPassSliceInfoManager::SetSliceInfoForFusionNodes(fusin_nodes), fe::SUCCESS);
}

TEST_F(UbPassSliceInfoManagerUT, set_slice_info_for_fusion_nodes_aipp) {
  vector<ge::NodePtr> fusin_nodes;
  ComputeGraphPtr graph = CreateGraphWithOneInput("Aipp", "aipp", false, fusin_nodes);
  EXPECT_NE(graph, nullptr);
  EXPECT_EQ(UbPassSliceInfoManager::SetSliceInfoForFusionNodes(fusin_nodes), fe::SUCCESS);
}

TEST_F(UbPassSliceInfoManagerUT, set_slice_info_for_fusion_nodes_write_select) {
  vector<ge::NodePtr> fusin_nodes;
  ComputeGraphPtr graph = CreateGraphWithOneInput("WriteSelect", "write_select", false, fusin_nodes);
  EXPECT_NE(graph, nullptr);
  EXPECT_EQ(UbPassSliceInfoManager::SetSliceInfoForFusionNodes(fusin_nodes), fe::SUCCESS);
}

TEST_F(UbPassSliceInfoManagerUT, set_slice_info_for_fusion_nodes_strided_write) {
  vector<ge::NodePtr> fusin_nodes;
  ComputeGraphPtr graph = CreateGraphWithOneInput("StridedWrite", "strided_write", false, fusin_nodes);
  EXPECT_NE(graph, nullptr);
  EXPECT_EQ(UbPassSliceInfoManager::SetSliceInfoForFusionNodes(fusin_nodes), fe::SUCCESS);
}

TEST_F(UbPassSliceInfoManagerUT, set_slice_info_for_fusion_nodes_dequant_1) {
  vector<ge::NodePtr> fusin_nodes;
  ComputeGraphPtr graph = CreateGraphWithTwoInput("AscendDequant", "dequant", false, fusin_nodes);
  EXPECT_NE(graph, nullptr);
  EXPECT_EQ(UbPassSliceInfoManager::SetSliceInfoForFusionNodes(fusin_nodes), fe::SUCCESS);
}

TEST_F(UbPassSliceInfoManagerUT, set_slice_info_for_fusion_nodes_dequant_2) {
  vector<ge::NodePtr> fusin_nodes;
  ComputeGraphPtr graph = CreateGraphWithTwoInput("AscendDequant", "dequant", true, fusin_nodes);
  EXPECT_NE(graph, nullptr);
  EXPECT_EQ(UbPassSliceInfoManager::SetSliceInfoForFusionNodes(fusin_nodes), fe::SUCCESS);
}

TEST_F(UbPassSliceInfoManagerUT, set_slice_info_for_fusion_nodes_dequant_3) {
  vector<ge::NodePtr> fusin_nodes;
  ComputeGraphPtr graph = CreateGraphWithTwoInput("AscendDequant", "dequant", true, fusin_nodes);
  EXPECT_NE(graph, nullptr);
  for (const ge::NodePtr &node : fusin_nodes) {
    if (node == nullptr || node->GetType() != "AscendDequant") {
      continue;
    }
    string op_slice_info = "{\"_op_slice_info\": {\"splitMaps\": [{\"inputList\": [{\"idx\": 0, \"axis\": [0], \"headOverLap\": [-1], \"tailOverLap\": [-1]}], \"outputList\": [{\"idx\": 0, \"axis\": [0]}]}, {\"inputList\": [{\"idx\": 0, \"axis\": [2], \"headOverLap\": [-1], \"tailOverLap\": [-1]}], \"outputList\": [{\"idx\": 0, \"axis\": [2]}]}, {\"inputList\": [{\"idx\": 0, \"axis\": [3], \"headOverLap\": [-1], \"tailOverLap\": [-1]}], \"outputList\": [{\"idx\": 0, \"axis\": [3]}]}]}}";
    AttrUtils::SetStr(node->GetOpDesc(), "_op_slice_info", op_slice_info);
    size_t input_size = 2;
    UbPassSliceInfoBasePtr slice_info_base_ptr = UbPassSliceInfoManager::SwitchSliceInfoPtrByPattern(UbMatchedType::UBMATCHTYPE_DEQUANT, node, input_size);
    EXPECT_NE(slice_info_base_ptr, nullptr);
    EXPECT_EQ(slice_info_base_ptr->ModifySliceInfoByPattern(node), fe::SUCCESS);
  }
}

TEST_F(UbPassSliceInfoManagerUT, set_slice_info_for_fusion_nodes_requant) {
  vector<ge::NodePtr> fusin_nodes;
  ComputeGraphPtr graph = CreateGraphWithTwoInput("AscendRequant", "requant", false, fusin_nodes);
  EXPECT_NE(graph, nullptr);
  EXPECT_EQ(UbPassSliceInfoManager::SetSliceInfoForFusionNodes(fusin_nodes), fe::SUCCESS);
}

TEST_F(UbPassSliceInfoManagerUT, set_slice_info_for_fusion_nodes_elemwise_1) {
  vector<ge::NodePtr> fusin_nodes;
  ComputeGraphPtr graph = CreateGraphWithOneInput("Relu", "ElemWise", false, fusin_nodes);
  EXPECT_NE(graph, nullptr);
  EXPECT_EQ(UbPassSliceInfoManager::SetSliceInfoForFusionNodes(fusin_nodes), fe::SUCCESS);
}

TEST_F(UbPassSliceInfoManagerUT, set_slice_info_for_fusion_nodes_elemwise_2) {
  vector<ge::NodePtr> fusin_nodes;
  ComputeGraphPtr graph = CreateGraphWithOneInput("Relu", "ElemWise", true, fusin_nodes);
  EXPECT_NE(graph, nullptr);
  EXPECT_EQ(UbPassSliceInfoManager::SetSliceInfoForFusionNodes(fusin_nodes), fe::SUCCESS);
}

TEST_F(UbPassSliceInfoManagerUT, set_slice_info_for_fusion_nodes_Pool2d) {
  vector<ge::NodePtr> fusin_nodes;
  ComputeGraphPtr graph = CreateGraphWithOneInput("Pooling", "Pool2d", false, fusin_nodes);
  EXPECT_NE(graph, nullptr);
  EXPECT_EQ(UbPassSliceInfoManager::SetSliceInfoForFusionNodes(fusin_nodes), fe::SUCCESS);
}

TEST_F(UbPassSliceInfoManagerUT, set_slice_info_for_fusion_nodes_dequant_s16) {
  vector<ge::NodePtr> fusin_nodes;
  ComputeGraphPtr graph = CreateGraphWithTwoInput("AscendDequantS16", "dequant_s16", false, fusin_nodes);
  EXPECT_NE(graph, nullptr);
  EXPECT_EQ(UbPassSliceInfoManager::SetSliceInfoForFusionNodes(fusin_nodes), fe::SUCCESS);
}

TEST_F(UbPassSliceInfoManagerUT, set_slice_info_for_fusion_nodes_requant_s16) {
  vector<ge::NodePtr> fusin_nodes;
  ComputeGraphPtr graph = CreateGraphWithTwoInput("AscendRequantS16", "requant_s16", false, fusin_nodes);
  EXPECT_NE(graph, nullptr);
  EXPECT_EQ(UbPassSliceInfoManager::SetSliceInfoForFusionNodes(fusin_nodes), fe::SUCCESS);
}
}

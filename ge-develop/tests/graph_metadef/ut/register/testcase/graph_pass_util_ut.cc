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
#include "graph/ge_tensor.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/op_desc.h"
#include "graph/compute_graph.h"
#include "graph_optimizer/fusion_common/graph_pass_util.h"

using namespace std;
using namespace ge;

namespace fe {
class GraphPassUtilUT : public testing::Test {
protected:
  void SetUp() {}

  void TearDown() {}
};

bool CheckOriginAttr(const std::vector<ge::NodePtr> &nodes, std::string pass_name,  
                     GraphPassUtil::OriginOpAttrsVec origin_attrs) {
  for (auto node : nodes) {
    auto op_desc = node->GetOpDesc();
    std::shared_ptr<GraphPassUtil::UnorderedMapping> op_attrs_maps_tmp =
        std::make_shared<GraphPassUtil::UnorderedMapping>();
    op_attrs_maps_tmp = op_desc->TryGetExtAttr(ge::ATTR_NAME_ORIGIN_OP_ATTRS_MAP, op_attrs_maps_tmp);
    if (op_attrs_maps_tmp->find(pass_name) == op_attrs_maps_tmp->cend()) {
      return false;
    }
    auto attrs_in_vec = (*op_attrs_maps_tmp)[pass_name];
    if (attrs_in_vec.size() != origin_attrs.size()) {
      return false;
    }
    for (const auto &origin_attr : origin_attrs) {
      bool is_in_vec = false;
      for (const auto &attr_in_vec : attrs_in_vec) {
        if (origin_attr == attr_in_vec) {
          is_in_vec = true;
          break;
        }
      }
      if (is_in_vec == false) {
        return false;
      }
    }
  }
  return true;
}

TEST_F(GraphPassUtilUT, set_output_desc_attr_case1) {
  EXPECT_NO_THROW(
    NodePtr origin_node = nullptr;
    NodePtr fusion_node = nullptr;
    GraphPassUtil::SetOutputDescAttr(0, 0, origin_node, fusion_node);
  );
}

TEST_F(GraphPassUtilUT, set_output_desc_attr_case2) {
  OpDescPtr relu1 = std::make_shared<OpDesc>("relu1", "Relu");
  OpDescPtr relu2 = std::make_shared<OpDesc>("relu2", "Relu");
  vector<int64_t> dim = {4, 4, 1, 4};
  GeShape shape(dim);
  GeTensorDesc tenosr_desc(shape, ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  tenosr_desc.SetOriginFormat(FORMAT_NCHW);
  tenosr_desc.SetOriginDataType(DT_FLOAT);
  relu1->AddInputDesc(tenosr_desc);
  relu1->AddOutputDesc(tenosr_desc);
  relu2->AddInputDesc(tenosr_desc);
  relu2->AddOutputDesc(tenosr_desc);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr relu1_node = graph->AddNode(relu1);
  NodePtr relu2_node = graph->AddNode(relu2);
  GraphPassUtil::SetOutputDescAttr(1, 0, relu1_node, relu2_node);
  EXPECT_EQ(relu2_node->GetOpDesc()->GetOutputDescPtr(0)->HasAttr(ge::ATTR_NAME_DATA_DUMP_ORIGIN_NAME), false);
  GraphPassUtil::SetOutputDescAttr(0, 1, relu1_node, relu2_node);
  EXPECT_EQ(relu2_node->GetOpDesc()->GetOutputDescPtr(0)->HasAttr(ge::ATTR_NAME_DATA_DUMP_ORIGIN_NAME), false);
  GraphPassUtil::SetOutputDescAttr(0, 0, relu1_node, relu2_node);
  EXPECT_EQ(relu2_node->GetOpDesc()->GetOutputDescPtr(0)->HasAttr(ge::ATTR_NAME_DATA_DUMP_ORIGIN_NAME), true);
  string origin_name;
  AttrUtils::GetStr(relu2->GetOutputDescPtr(0), ge::ATTR_NAME_DATA_DUMP_ORIGIN_NAME, origin_name);
  EXPECT_EQ(origin_name, "relu1");
  string origin_dtype;
  AttrUtils::GetStr(relu2->GetOutputDescPtr(0), ge::ATTR_NAME_DATA_DUMP_ORIGIN_DATA_TYPE, origin_dtype);
  EXPECT_EQ(origin_dtype, "DT_FLOAT");
  string origin_format;
  AttrUtils::GetStr(relu2->GetOutputDescPtr(0), ge::ATTR_NAME_DATA_DUMP_ORIGIN_FORMAT, origin_format);
  EXPECT_EQ(origin_format, "NCHW");
}

TEST_F(GraphPassUtilUT, set_output_desc_attr_case3) {
  OpDescPtr relu1 = std::make_shared<OpDesc>("relu1", "Relu");
  OpDescPtr relu2 = std::make_shared<OpDesc>("relu2", "Relu");
  vector<int64_t> dim = {4, 4, 1, 4};
  GeShape shape(dim);
  GeTensorDesc tenosr_desc(shape, ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  tenosr_desc.SetOriginFormat(FORMAT_NCHW);
  tenosr_desc.SetOriginDataType(DT_FLOAT);
  relu1->AddInputDesc(tenosr_desc);
  relu1->AddOutputDesc(tenosr_desc);
  AttrUtils::SetStr(relu1->MutableOutputDesc(0), ge::ATTR_NAME_DATA_DUMP_ORIGIN_NAME, "origin_relu1");
  AttrUtils::SetStr(relu1->MutableOutputDesc(0), ge::ATTR_NAME_DATA_DUMP_ORIGIN_DATA_TYPE, "DT_DOUBLE");
  AttrUtils::SetStr(relu1->MutableOutputDesc(0), ge::ATTR_NAME_DATA_DUMP_ORIGIN_FORMAT, "ND");
  relu2->AddInputDesc(tenosr_desc);
  relu2->AddOutputDesc(tenosr_desc);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr relu1_node = graph->AddNode(relu1);
  NodePtr relu2_node = graph->AddNode(relu2);

  GraphPassUtil::SetOutputDescAttr(0, 0, relu1_node, relu2_node);
  EXPECT_EQ(relu2_node->GetOpDesc()->GetOutputDescPtr(0)->HasAttr(ge::ATTR_NAME_DATA_DUMP_ORIGIN_NAME), true);
  string origin_name;
  AttrUtils::GetStr(relu2->GetOutputDescPtr(0), ge::ATTR_NAME_DATA_DUMP_ORIGIN_NAME, origin_name);
  EXPECT_EQ(origin_name, "origin_relu1");
  string origin_dtype;
  AttrUtils::GetStr(relu2->GetOutputDescPtr(0), ge::ATTR_NAME_DATA_DUMP_ORIGIN_DATA_TYPE, origin_dtype);
  EXPECT_EQ(origin_dtype, "DT_DOUBLE");
  string origin_format;
  AttrUtils::GetStr(relu2->GetOutputDescPtr(0), ge::ATTR_NAME_DATA_DUMP_ORIGIN_FORMAT, origin_format);
  EXPECT_EQ(origin_format, "ND");
}

TEST_F(GraphPassUtilUT, set_output_desc_attr_case4) {
  OpDescPtr relu1 = std::make_shared<OpDesc>("relu1", "Relu");
  OpDescPtr relu2 = std::make_shared<OpDesc>("relu2", "Relu");
  vector<int64_t> dim = {4, 4, 1, 4};
  GeShape shape(dim);
  GeTensorDesc tenosr_desc(shape, ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  tenosr_desc.SetOriginFormat(FORMAT_NCHW);
  tenosr_desc.SetOriginDataType(DT_FLOAT);
  relu1->AddInputDesc(tenosr_desc);
  relu1->AddOutputDesc(tenosr_desc);
  vector<string> names = {"ori_rule1"};
  AttrUtils::SetListStr(relu1, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, names);
  AttrUtils::SetStr(relu1->MutableOutputDesc(0), ge::ATTR_NAME_DATA_DUMP_ORIGIN_DATA_TYPE, "RESERVED");
  AttrUtils::SetStr(relu1->MutableOutputDesc(0), ge::ATTR_NAME_DATA_DUMP_ORIGIN_FORMAT, "RESERVED");
  relu2->AddInputDesc(tenosr_desc);
  relu2->AddOutputDesc(tenosr_desc);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr relu1_node = graph->AddNode(relu1);
  NodePtr relu2_node = graph->AddNode(relu2);

  GraphPassUtil::SetOutputDescAttr(0, 0, relu1_node, relu2_node);
  EXPECT_EQ(relu2_node->GetOpDesc()->GetOutputDescPtr(0)->HasAttr(ge::ATTR_NAME_DATA_DUMP_ORIGIN_NAME), true);
  string origin_name;
  AttrUtils::GetStr(relu2->GetOutputDescPtr(0), ge::ATTR_NAME_DATA_DUMP_ORIGIN_NAME, origin_name);
  EXPECT_EQ(origin_name, "ori_rule1");
  string origin_dtype;
  AttrUtils::GetStr(relu2->GetOutputDescPtr(0), ge::ATTR_NAME_DATA_DUMP_ORIGIN_DATA_TYPE, origin_dtype);
  EXPECT_EQ(origin_dtype, "DT_FLOAT");
  string origin_format;
  AttrUtils::GetStr(relu2->GetOutputDescPtr(0), ge::ATTR_NAME_DATA_DUMP_ORIGIN_FORMAT, origin_format);
  EXPECT_EQ(origin_format, "NCHW");
}

TEST_F(GraphPassUtilUT, set_output_desc_attr_case5) {
  vector<int64_t> dims = {1,2,3,4};
  std::string origin_data_type_str = "RESERVED";
  GeShape shape(dims);
  GeTensorDescPtr tensor_desc_ptr = std::make_shared<GeTensorDesc>(shape, FORMAT_NCHW, DT_FLOAT);
  tensor_desc_ptr->SetDataType((ge::DataType)24);
  (void)ge::AttrUtils::SetStr(tensor_desc_ptr, ge::ATTR_NAME_DATA_DUMP_ORIGIN_DATA_TYPE, "DT_DOUBLE");
  ge::DataType origin_dtype;
  origin_dtype = GraphPassUtil::GetDataDumpOriginDataType(tensor_desc_ptr);
  EXPECT_EQ(origin_dtype, (ge::DataType)11);
}

TEST_F(GraphPassUtilUT, set_output_desc_attr_case6) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::OpDescPtr op = std::make_shared<OpDesc>("test_op", "TestOp");
  auto node = graph->AddNode(op);

  std::map<std::string, ge::NodePtr> inner_map;
  inner_map["test"] = node;
  std::unordered_map<std::string, std::map<std::string, ge::NodePtr>> node_map;
  node_map["test"] = inner_map;

  NodeTypeMapPtr node_type_map = std::make_shared<NodeTypeMap>(node_map);
  EXPECT_NO_THROW(GraphPassUtil::AddNodeToNodeTypeMap(node_type_map, "test", node));
}

TEST_F(GraphPassUtilUT, set_output_desc_attr_case7) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::OpDescPtr op = std::make_shared<OpDesc>("test_op", "TestOp");
  auto node = graph->AddNode(op);

  std::map<std::string, ge::NodePtr> inner_map;
  inner_map["test"] = node;
  std::unordered_map<std::string, std::map<std::string, ge::NodePtr>> node_map;
  node_map["test"] = inner_map;

  NodeTypeMapPtr node_type_map = std::make_shared<NodeTypeMap>(node_map);
  EXPECT_NO_THROW(GraphPassUtil::RemoveNodeFromNodeTypeMap(node_type_map, "test", node));
}

TEST_F(GraphPassUtilUT, set_output_desc_attr_case8) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");
  ge::OpDescPtr op = std::make_shared<OpDesc>("test_op", "TestOp");
  auto node = graph->AddNode(op);

  std::map<std::string, ge::NodePtr> inner_map;
  inner_map["test"] = node;
  std::unordered_map<std::string, std::map<std::string, ge::NodePtr>> node_map;
  node_map["test"] = inner_map;

  NodeTypeMapPtr node_type_map = std::make_shared<NodeTypeMap>(node_map);
  vector<ge::NodePtr> nodes;
  EXPECT_NO_THROW(GraphPassUtil::GetNodesFromNodeTypeMap(node_type_map, "test", nodes));
}

TEST_F(GraphPassUtilUT, set_output_desc_attr_case9) {
  vector<int64_t> dims = {1,2,3,4};
  std::string origin_data_type_str = "RESERVED";
  GeShape shape(dims);
  GeTensorDescPtr tensor_desc_ptr = std::make_shared<GeTensorDesc>(shape, FORMAT_NCHW, DT_FLOAT);
  tensor_desc_ptr->SetDataType((ge::DataType)24);
  (void)ge::AttrUtils::SetStr(tensor_desc_ptr, ge::ATTR_NAME_DATA_DUMP_ORIGIN_FORMAT, "NCHW");
  ge::Format origin_format;
  origin_format = GraphPassUtil::GetDataDumpOriginFormat(tensor_desc_ptr);
  EXPECT_EQ(origin_format, (ge::Format)0);
}

TEST_F(GraphPassUtilUT, set_output_desc_attr_case10) {
  putenv(const_cast<char*>("DUMP_GE_GRAPH=2"));
  OpDescPtr relu1 = std::make_shared<OpDesc>("relu1", "Relu");
  OpDescPtr relu2 = std::make_shared<OpDesc>("relu2", "Relu");
  vector<int64_t> dim = {4, 4, 1, 4};
  GeShape shape(dim);
  GeTensorDesc tenosr_desc(shape, ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  tenosr_desc.SetOriginFormat(FORMAT_NCHW);
  tenosr_desc.SetOriginDataType(DT_FLOAT);
  relu1->AddInputDesc(tenosr_desc);
  relu1->AddOutputDesc(tenosr_desc);
  vector<string> names = {"ori_rule1"};

  relu2->AddInputDesc(tenosr_desc);
  relu2->AddOutputDesc(tenosr_desc);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr relu1_node = graph->AddNode(relu1);
  NodePtr relu2_node = graph->AddNode(relu2);
  std::vector<ge::NodePtr> original_nodes = {relu1_node};
  std::vector<ge::NodePtr> fus_nodes = {relu2_node};

  GraphPassUtil::RecordPassnameAndOriginalAttrs(original_nodes, fus_nodes, "passA");
  GraphPassUtil::OriginOpAttrsVec origin_op_attrs_to_check = {{"relu1", "Relu"}};
  bool oringin_attr_check = false;
  oringin_attr_check = CheckOriginAttr(fus_nodes, "passA", origin_op_attrs_to_check);
  EXPECT_EQ(oringin_attr_check, true);
}

TEST_F(GraphPassUtilUT, set_original_op_names_and_types) {
  putenv(const_cast<char*>("DUMP_GE_GRAPH=2"));
  OpDescPtr relu1 = std::make_shared<OpDesc>("relu1", "Relu");
  OpDescPtr relu2 = std::make_shared<OpDesc>("relu2", "Relu");
  vector<int64_t> dim = {4, 4, 1, 4};
  GeShape shape(dim);
  GeTensorDesc tenosr_desc(shape, ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  tenosr_desc.SetOriginFormat(FORMAT_NCHW);
  tenosr_desc.SetOriginDataType(DT_FLOAT);
  relu1->AddInputDesc(tenosr_desc);
  relu1->AddOutputDesc(tenosr_desc);
  vector<string> names = {"ori_rule1"};

  relu2->AddInputDesc(tenosr_desc);
  relu2->AddOutputDesc(tenosr_desc);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr relu1_node = graph->AddNode(relu1);
  NodePtr relu2_node = graph->AddNode(relu2);
  vector<string> names_tmp = {"A", "B"};
  vector<string> types_tmp = {"typeA", "typeB"};
  const ge::OpDescPtr relu1_node_op_desc_ptr = relu1_node->GetOpDesc();
  const ge::OpDescPtr relu2_node_op_desc_ptr = relu2_node->GetOpDesc();
  (void)ge::AttrUtils::SetListStr(relu1_node_op_desc_ptr, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, names_tmp);
  (void)ge::AttrUtils::SetListStr(relu1_node_op_desc_ptr, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_TYPES, types_tmp);

  std::vector<ge::NodePtr> original_nodes = {relu1_node};

  GraphPassUtil::RecordOriginalNames(original_nodes, relu2_node);
  vector<string> original_names;
  vector<string> original_types;
  ge::AttrUtils::GetListStr(relu2_node_op_desc_ptr, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_names);
  ge::AttrUtils::GetListStr(relu2_node_op_desc_ptr, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_TYPES, original_types);

  vector<string> original_names_check = {"A", "B"};
  vector<string> original_types_check = {"typeA", "typeB"};
  EXPECT_EQ(original_names_check, original_names);
  EXPECT_EQ(original_types_check, original_types);
}

TEST_F(GraphPassUtilUT, set_output_desc_attr_case11) {
  putenv(const_cast<char*>("DUMP_GE_GRAPH=2"));
  OpDescPtr relu1 = std::make_shared<OpDesc>("relu1", "Relu");
  OpDescPtr relu2 = std::make_shared<OpDesc>("relu2", "Relu");
  vector<int64_t> dim = {4, 4, 1, 4};
  GeShape shape(dim);
  GeTensorDesc tenosr_desc(shape, ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  tenosr_desc.SetOriginFormat(FORMAT_NCHW);
  tenosr_desc.SetOriginDataType(DT_FLOAT);
  relu1->AddInputDesc(tenosr_desc);
  relu1->AddOutputDesc(tenosr_desc);
  vector<string> names = {"ori_rule1"};
  std::shared_ptr<GraphPassUtil::UnorderedMapping> op_attrs_maps_tmp =
      std::make_shared<GraphPassUtil::UnorderedMapping>();
  GraphPassUtil::OriginOpAttrsVec origin_op_attrs_vec = {{"nodeA", "typeA"}, {"nodeB", "typeB"}};
  op_attrs_maps_tmp->insert(std::pair<std::string, 
                            GraphPassUtil::OriginOpAttrsVec>("pass_test", origin_op_attrs_vec));
  (void)relu1->SetExtAttr(ge::ATTR_NAME_ORIGIN_OP_ATTRS_MAP, op_attrs_maps_tmp);
  vector<std::string> pass_names = {"pass_test"};
  (void)AttrUtils::SetListStr(relu1, "pass_name", pass_names);

  relu2->AddInputDesc(tenosr_desc);
  relu2->AddOutputDesc(tenosr_desc);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr relu1_node = graph->AddNode(relu1);
  NodePtr relu2_node = graph->AddNode(relu2);
  std::vector<ge::NodePtr> original_nodes = {relu1_node};
  std::vector<ge::NodePtr> fus_nodes = {relu2_node};

  GraphPassUtil::RecordPassnameAndOriginalAttrs(original_nodes, fus_nodes, "passA");
  GraphPassUtil::OriginOpAttrsVec origin_op_attrs_to_check = {{"nodeA", "typeA"}, {"nodeB", "typeB"}};
  bool oringin_attr_check = false;
  oringin_attr_check = CheckOriginAttr(fus_nodes, "passA", origin_op_attrs_to_check);
  EXPECT_EQ(oringin_attr_check, true);
}


void CreateGraph(ComputeGraphPtr &graph, std::vector<ge::NodePtr> &original_nodes,
                 std::vector<ge::NodePtr> &fus_nodes) {
  OpDescPtr relu1 = std::make_shared<OpDesc>("relu1", "Relu");
  OpDescPtr relu2 = std::make_shared<OpDesc>("relu2", "Relu");
  OpDescPtr relu3 = std::make_shared<OpDesc>("relu3", "Relu");
  OpDescPtr fusion_op = std::make_shared<OpDesc>("fusion", "Fusion");
  vector<int64_t> dim = {4, 4, 1, 4};
  GeShape shape(dim);
  GeTensorDesc tenosr_desc(shape, ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  tenosr_desc.SetOriginFormat(FORMAT_NCHW);
  tenosr_desc.SetOriginDataType(DT_FLOAT);
  relu1->AddInputDesc(tenosr_desc);
  relu1->AddOutputDesc(tenosr_desc);


  relu2->AddInputDesc(tenosr_desc);
  relu2->AddOutputDesc(tenosr_desc);

  relu3->AddInputDesc(tenosr_desc);
  relu3->AddOutputDesc(tenosr_desc);

  fusion_op->AddInputDesc(tenosr_desc);
  fusion_op->AddOutputDesc(tenosr_desc);


  NodePtr relu1_node = graph->AddNode(relu1);
  NodePtr relu2_node = graph->AddNode(relu2);
  NodePtr relu3_node = graph->AddNode(relu3);
  NodePtr fusion_node = graph->AddNode(fusion_op);
  ge::GraphUtils::AddEdge(relu1_node->GetOutDataAnchor(0), relu2_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(relu2_node->GetOutDataAnchor(0), relu3_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(relu3_node->GetOutDataAnchor(0), fusion_node->GetInDataAnchor(0));

  original_nodes = {relu2_node, relu3_node};
  fus_nodes = {fusion_node};
}

TEST_F(GraphPassUtilUT, test_get_back_ward_attr_01) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  std::vector<ge::NodePtr> original_nodes;
  std::vector<ge::NodePtr> fus_nodes;
  CreateGraph(graph, original_nodes, fus_nodes);
  bool backward = false;
  GraphPassUtil::GetBackWardAttr(original_nodes, backward, BackWardInheritMode::kInheritTrue);
  EXPECT_EQ(backward, true);
}

TEST_F(GraphPassUtilUT, test_get_back_ward_attr_02) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  std::vector<ge::NodePtr> original_nodes;
  std::vector<ge::NodePtr> fus_nodes;
  CreateGraph(graph, original_nodes, fus_nodes);
  auto ori_node0 = original_nodes[0];
  auto ori_node1 = original_nodes[1];
  ge::AttrUtils::SetBool(ori_node0->GetOpDesc(), "_backward", true);
  ge::AttrUtils::SetBool(ori_node1->GetOpDesc(), "_backward", true);
  bool backward = false;
  GraphPassUtil::GetBackWardAttr(original_nodes, backward, BackWardInheritMode::kFusedNode);
  EXPECT_EQ(backward, true);
}

TEST_F(GraphPassUtilUT, test_get_back_ward_attr_03) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  std::vector<ge::NodePtr> original_nodes;
  std::vector<ge::NodePtr> fus_nodes;
  CreateGraph(graph, original_nodes, fus_nodes);
  auto ori_node0 = original_nodes[0];
  auto ori_node1 = original_nodes[1];
  ge::AttrUtils::SetBool(ori_node1->GetOpDesc(), "_backward", true);
  bool backward = false;
  GraphPassUtil::GetBackWardAttr(original_nodes, backward, BackWardInheritMode::kInsertNode);
  EXPECT_EQ(backward, true);
}

TEST_F(GraphPassUtilUT, test_get_back_ward_attr_03_1) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  std::vector<ge::NodePtr> original_nodes;
  std::vector<ge::NodePtr> fus_nodes;
  CreateGraph(graph, original_nodes, fus_nodes);
  auto ori_node0 = original_nodes[0];
  auto ori_node1 = original_nodes[1];
  ge::AttrUtils::SetBool(ori_node1->GetOpDesc(), "_backward", true);
  bool backward = false;
  GraphPassUtil::GetBackWardAttr(original_nodes, backward, BackWardInheritMode::kFusedNode);
  EXPECT_FALSE(backward);
}

TEST_F(GraphPassUtilUT, test_get_back_ward_attr_04) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  std::vector<ge::NodePtr> original_nodes;
  std::vector<ge::NodePtr> fus_nodes;
  CreateGraph(graph, original_nodes, fus_nodes);
  auto ori_node0 = original_nodes[0];
  auto ori_node1 = original_nodes[1];
  ge::AttrUtils::SetBool(ori_node0->GetOpDesc(), "_backward", true);
  ge::AttrUtils::SetBool(ori_node1->GetOpDesc(), "_backward", true);
  bool backward = false;
  GraphPassUtil::GetBackWardAttr(original_nodes, backward, BackWardInheritMode::kDoNotInherit);
  EXPECT_FALSE(backward);
}

TEST_F(GraphPassUtilUT, test_inherit_attrs_01) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  std::vector<ge::NodePtr> original_nodes;
  std::vector<ge::NodePtr> fus_nodes;
  CreateGraph(graph, original_nodes, fus_nodes);
  auto ori_node0 = original_nodes[0];
  auto ori_node1 = original_nodes[1];
  ge::AttrUtils::SetBool(ori_node0->GetOpDesc(), "_backward", true);
  ge::AttrUtils::SetBool(ori_node1->GetOpDesc(), "_backward", true);
  ge::AttrUtils::SetBool(ori_node1->GetOpDesc(), "_recompute", true);
  ge::AttrUtils::SetBool(ori_node1->GetOpDesc(), "_optimizer", true);
  ge::AttrUtils::SetInt(ori_node1->GetOpDesc(), ge::ATTR_NAME_KEEP_DTYPE, 1);
  ge::AttrUtils::SetStr(ori_node1->GetOpDesc(), ge::ATTR_NAME_OP_COMPILE_STRATEGY, "test");

  GraphPassUtil::InheritAttrFromOriNodes(original_nodes, fus_nodes, BackWardInheritMode::kFusedNode);
  auto fus_op = fus_nodes.at(0)->GetOpDesc();

  bool backward = false;
  ge::AttrUtils::GetBool(fus_op, "_backward", backward);
  EXPECT_EQ(backward, true);

  bool recompute = 0;
  ge::AttrUtils::GetBool(fus_op, "_recompute", recompute);
  EXPECT_EQ(recompute, true);

  bool optimizer = 0;
  ge::AttrUtils::GetBool(fus_op, "_optimizer", optimizer);
  EXPECT_EQ(optimizer, true);

  int64_t keep_dtype = 0;
  ge::AttrUtils::GetInt(fus_op, ge::ATTR_NAME_KEEP_DTYPE, keep_dtype);
  EXPECT_EQ(keep_dtype, 1);

  string strategy = "";
  ge::AttrUtils::GetStr(fus_op, ge::ATTR_NAME_OP_COMPILE_STRATEGY, strategy);
  EXPECT_EQ(strategy, "test");
}


TEST_F(GraphPassUtilUT, test_inherit_attrs_02) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  std::vector<ge::NodePtr> original_nodes;
  std::vector<ge::NodePtr> fus_nodes;
  CreateGraph(graph, original_nodes, fus_nodes);
  auto ori_node0 = original_nodes[0];
  auto ori_node1 = original_nodes[1];
  ge::AttrUtils::SetBool(ori_node1->GetOpDesc(), "_backward", true);

  GraphPassUtil::InheritAttrFromOriNodes(original_nodes, fus_nodes, BackWardInheritMode::kFusedNode);
  auto fus_op = fus_nodes.at(0)->GetOpDesc();

  bool backward = false;
  ge::AttrUtils::GetBool(fus_op, "_backward", backward);
  EXPECT_TRUE(backward == false);


  EXPECT_EQ(fus_op->HasAttr("_recompute"), false);
  EXPECT_EQ(fus_op->HasAttr("_optimizer"), false);
  EXPECT_EQ(fus_op->HasAttr(ge::ATTR_NAME_KEEP_DTYPE), false);
  EXPECT_EQ(fus_op->HasAttr(ge::ATTR_NAME_OP_COMPILE_STRATEGY), false);
}


TEST_F(GraphPassUtilUT, test_inherit_attrs_03) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  std::vector<ge::NodePtr> original_nodes;
  std::vector<ge::NodePtr> fus_nodes;
  CreateGraph(graph, original_nodes, fus_nodes);
  auto ori_node0 = original_nodes[0];
  auto ori_node1 = original_nodes[1];
  ge::AttrUtils::SetBool(ori_node0->GetOpDesc(), "_backward", true);
  ge::AttrUtils::SetBool(ori_node1->GetOpDesc(), "_backward", true);
  ge::AttrUtils::SetBool(ori_node1->GetOpDesc(), "_recompute", true);
  ge::AttrUtils::SetBool(ori_node1->GetOpDesc(), "_optimizer", true);

  auto original_nodes_reverse_order = {ori_node1, ori_node0};
  GraphPassUtil::InheritAttrFromOriNodes(original_nodes_reverse_order, fus_nodes, BackWardInheritMode::kFusedNode);
  auto fus_op = fus_nodes.at(0)->GetOpDesc();

  bool backward = false;
  ge::AttrUtils::GetBool(fus_op, "_backward", backward);
  EXPECT_EQ(backward, true);
  EXPECT_EQ(fus_op->HasAttr("_recompute"), true);
  EXPECT_EQ(fus_op->HasAttr("_optimizer"), true);

  bool recompute = false;
  ge::AttrUtils::GetBool(fus_op, "_recompute", recompute);
  EXPECT_EQ(recompute, true);

  bool optimizer = false;
  ge::AttrUtils::GetBool(fus_op, "_optimizer", optimizer);
  EXPECT_EQ(optimizer, true);

  EXPECT_EQ(fus_op->HasAttr(ge::ATTR_NAME_KEEP_DTYPE), false);
  EXPECT_EQ(fus_op->HasAttr(ge::ATTR_NAME_OP_COMPILE_STRATEGY), false);
}

  // N -> 1
TEST_F(GraphPassUtilUT, test_inherit_attrs_04) {
  auto graph = std::make_shared<ComputeGraph>("test");
  ge::OpDescPtr ori_op_desc1 = std::make_shared<ge::OpDesc>("node1", "Relu");
  ge::OpDescPtr ori_op_desc2 = std::make_shared<ge::OpDesc>("node2", "Relu");
  ge::OpDescPtr fus_op_desc = std::make_shared<ge::OpDesc>("fusion", "Fusion");

  ge::NodePtr ori_node1 = graph->AddNode(ori_op_desc1);
  ge::NodePtr ori_node2 = graph->AddNode(ori_op_desc2);
  ge::NodePtr fus_node = graph->AddNode(fus_op_desc);

  std::vector<ge::NodePtr> ori_nodes = {ori_node1, ori_node2};
  std::vector<ge::NodePtr> fus_nodes = {fus_node};

  ge::AttrUtils::SetInt(ori_op_desc1, "_op_custom_impl_mode_enum", 0x20);
  ge::AttrUtils::SetInt(ori_op_desc2, "_op_custom_impl_mode_enum", 0x40);

  GraphPassUtil::InheritAttrFromOriNodes(ori_nodes, fus_nodes, BackWardInheritMode::kFusedNode);

  int64_t op_impl_mode = -1;
  ge::AttrUtils::GetInt(fus_op_desc, "_op_custom_impl_mode_enum", op_impl_mode);
  EXPECT_EQ(op_impl_mode, 0x40);
}


// 1 -> N
TEST_F(GraphPassUtilUT, test_inherit_attrs_05) {
  auto graph = std::make_shared<ComputeGraph>("test");
  ge::OpDescPtr ori_op_desc = std::make_shared<ge::OpDesc>("node", "Relu");
  ge::OpDescPtr fus_op_desc1 = std::make_shared<ge::OpDesc>("node2", "Relu");
  ge::OpDescPtr fus_op_desc2 = std::make_shared<ge::OpDesc>("node1", "Relu");

  ge::NodePtr ori_node = graph->AddNode(ori_op_desc);
  ge::NodePtr fus_node1 = graph->AddNode(fus_op_desc1);
  ge::NodePtr fus_node2 = graph->AddNode(fus_op_desc2);

  std::vector<ge::NodePtr> ori_nodes = {ori_node};
  std::vector<ge::NodePtr> fus_nodes = {fus_node1, fus_node2};

  ge::AttrUtils::SetInt(ori_op_desc, "_op_custom_impl_mode_enum", 0x10);

  GraphPassUtil::InheritAttrFromOriNodes(ori_nodes, fus_nodes, BackWardInheritMode::kFusedNode);

  int64_t op_impl_mode = -1;
  ge::AttrUtils::GetInt(fus_op_desc1, "_op_custom_impl_mode_enum", op_impl_mode);
  EXPECT_EQ(op_impl_mode, 0x10);
  ge::AttrUtils::GetInt(fus_op_desc2, "_op_custom_impl_mode_enum", op_impl_mode);
  EXPECT_EQ(op_impl_mode, 0x10);
}

// N -> N
TEST_F(GraphPassUtilUT, test_inherit_attrs_06) {
  auto graph = std::make_shared<ComputeGraph>("test");
  ge::OpDescPtr ori_op_desc1 = std::make_shared<ge::OpDesc>("node1", "Relu");
  ge::OpDescPtr ori_op_desc2 = std::make_shared<ge::OpDesc>("node2", "Relu");
  ge::OpDescPtr fus_op_desc1 = std::make_shared<ge::OpDesc>("node2", "Fusion");
  ge::OpDescPtr fus_op_desc2 = std::make_shared<ge::OpDesc>("node1", "Fusion");

  ge::NodePtr ori_node1 = graph->AddNode(ori_op_desc1);
  ge::NodePtr ori_node2 = graph->AddNode(ori_op_desc2);
  ge::NodePtr fus_node1 = graph->AddNode(fus_op_desc1);
  ge::NodePtr fus_node2 = graph->AddNode(fus_op_desc2);

  std::vector<ge::NodePtr> ori_nodes = {ori_node1, ori_node2};
  std::vector<ge::NodePtr> fus_nodes = {fus_node1, fus_node2};

  ge::AttrUtils::SetInt(ori_op_desc1, "_op_custom_impl_mode_enum", 0x4);
  ge::AttrUtils::SetInt(ori_op_desc2, "_op_custom_impl_mode_enum", 0x2);

  GraphPassUtil::InheritAttrFromOriNodes(ori_nodes, fus_nodes, BackWardInheritMode::kFusedNode);

  int64_t op_impl_mode = -1;
  ge::AttrUtils::GetInt(fus_op_desc1, "_op_custom_impl_mode_enum", op_impl_mode);
  EXPECT_EQ(op_impl_mode, 0x2);
  ge::AttrUtils::GetInt(fus_op_desc2, "_op_custom_impl_mode_enum", op_impl_mode);
  EXPECT_EQ(op_impl_mode, 0x4);
}

// N1 -> N2
TEST_F(GraphPassUtilUT, test_inherit_attrs_07) {
  auto graph = std::make_shared<ComputeGraph>("test");
  ge::OpDescPtr ori_op_desc1 = std::make_shared<ge::OpDesc>("node1", "Relu");
  ge::OpDescPtr ori_op_desc2 = std::make_shared<ge::OpDesc>("node2", "Relu");
  ge::OpDescPtr ori_op_desc3 = std::make_shared<ge::OpDesc>("node3", "Relu");
  ge::OpDescPtr fus_op_desc1 = std::make_shared<ge::OpDesc>("node1", "Relu");
  ge::OpDescPtr fus_op_desc2 = std::make_shared<ge::OpDesc>("Fusion", "Fusion");

  ge::NodePtr ori_node1 = graph->AddNode(ori_op_desc1);
  ge::NodePtr ori_node2 = graph->AddNode(ori_op_desc2);
  ge::NodePtr ori_node3 = graph->AddNode(ori_op_desc3);
  ge::NodePtr fus_node1 = graph->AddNode(fus_op_desc1);
  ge::NodePtr fus_node2 = graph->AddNode(fus_op_desc2);

  std::vector<ge::NodePtr> ori_nodes = {ori_node1, ori_node2, ori_node3};
  std::vector<ge::NodePtr> fus_nodes = {fus_node1, fus_node2};

  ge::AttrUtils::SetInt(ori_op_desc1, "_op_custom_impl_mode_enum", 0x8);
  ge::AttrUtils::SetInt(ori_op_desc2, "_op_custom_impl_mode_enum", 0x4);
  ge::AttrUtils::SetInt(ori_op_desc3, "_op_custom_impl_mode_enum", 0x2);

  GraphPassUtil::InheritAttrFromOriNodes(ori_nodes, fus_nodes, BackWardInheritMode::kFusedNode);

  int64_t op_impl_mode = -1;
  ge::AttrUtils::GetInt(fus_op_desc1, "_op_custom_impl_mode_enum", op_impl_mode);
  EXPECT_EQ(op_impl_mode, 0x8);
  ge::AttrUtils::GetInt(fus_op_desc2, "_op_custom_impl_mode_enum", op_impl_mode);
  EXPECT_EQ(op_impl_mode, 0x4);
}

// N -> 1
TEST_F(GraphPassUtilUT, test_inherit_attrs_08) {
  auto graph = std::make_shared<ComputeGraph>("test");
  ge::OpDescPtr ori_op_desc1 = std::make_shared<ge::OpDesc>("node1", "Relu");
  ge::OpDescPtr ori_op_desc2 = std::make_shared<ge::OpDesc>("node2", "Relu");
  ge::OpDescPtr fus_op_desc = std::make_shared<ge::OpDesc>("fusion", "Fusion");

  ge::NodePtr ori_node1 = graph->AddNode(ori_op_desc1);
  ge::NodePtr ori_node2 = graph->AddNode(ori_op_desc2);
  ge::NodePtr fus_node = graph->AddNode(fus_op_desc);

  std::vector<ge::NodePtr> ori_nodes = {ori_node1, ori_node2};
  std::vector<ge::NodePtr> fus_nodes = {fus_node};

  ge::AttrUtils::SetInt(ori_op_desc1, "_op_impl_mode_enum", 0x20);
  ge::AttrUtils::SetInt(ori_op_desc2, "_op_impl_mode_enum", 0x40);

  GraphPassUtil::InheritAttrFromOriNodes(ori_nodes, fus_nodes, BackWardInheritMode::kFusedNode);

  int64_t op_impl_mode = -1;
  ge::AttrUtils::GetInt(fus_op_desc, "_op_impl_mode_enum", op_impl_mode);
  EXPECT_TRUE(op_impl_mode == -1);
}

TEST_F(GraphPassUtilUT, test_set_pair_tensor_attr) {
  auto graph = std::make_shared<ComputeGraph>("test");
  ge::OpDescPtr op_desc1 = std::make_shared<ge::OpDesc>("node1", "Relu");
  ge::OpDescPtr op_desc2 = std::make_shared<ge::OpDesc>("node2", "Relu");
  ge::OpDescPtr op_desc3 = std::make_shared<ge::OpDesc>("node3", "Relu");
  ge::OpDescPtr op_desc4 = std::make_shared<ge::OpDesc>("node4", "Relu");

  vector<int64_t> dim = {4, 4, 1, 4};
  GeShape shape(dim);
  GeTensorDesc tenosr_desc0(shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);
  tenosr_desc0.SetOriginFormat(FORMAT_NCHW);
  tenosr_desc0.SetOriginDataType(DT_FLOAT);

  vector<int64_t> dim2 = {16, 1, 1, 2};
  GeShape shape2(dim2);
  GeTensorDesc tenosr_desc1(shape2, ge::FORMAT_NCHW, ge::DT_FLOAT16);
  tenosr_desc1.SetOriginFormat(FORMAT_NCHW);
  tenosr_desc1.SetOriginDataType(DT_FLOAT);

  GeTensorDesc tenosr_desc2(tenosr_desc1);

  op_desc1->AddInputDesc(tenosr_desc2);
  op_desc1->AddOutputDesc(tenosr_desc1);

  op_desc2->AddInputDesc(tenosr_desc0);
  op_desc2->AddOutputDesc(tenosr_desc2);

  op_desc3->AddInputDesc(tenosr_desc2);
  op_desc3->AddOutputDesc(tenosr_desc1);

  op_desc4->AddInputDesc(tenosr_desc2);
  op_desc4->AddOutputDesc(tenosr_desc1);

  ge::NodePtr node1 = graph->AddNode(op_desc1);
  ge::NodePtr node2 = graph->AddNode(op_desc2);
  ge::NodePtr node3 = graph->AddNode(op_desc3);
  ge::NodePtr node4 = graph->AddNode(op_desc4);

  ge::GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node2->GetOutDataAnchor(0), node3->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node4->GetInDataAnchor(0));
  std::map<std::string, int> attr_val;
  attr_val["_tensor_memory_scope"] = 2;
  GraphPassUtil::SetPairTensorAttr(node2, 0, attr_val);
  GraphPassUtil::SetPairTensorAttr(node2, 0, attr_val, false);
  int64_t scope_0 = -1;
  int64_t scope_1 = -1;
  int64_t scope_2 = -1;
  int64_t scope_3 = -1;
  int64_t scope_4 = -1;
  (void)ge::AttrUtils::GetInt(op_desc1->MutableOutputDesc(0), "_tensor_memory_scope", scope_0);
  (void)ge::AttrUtils::GetInt(op_desc2->MutableInputDesc(0), "_tensor_memory_scope", scope_1);
  (void)ge::AttrUtils::GetInt(op_desc2->MutableOutputDesc(0), "_tensor_memory_scope", scope_2);
  (void)ge::AttrUtils::GetInt(op_desc3->MutableInputDesc(0), "_tensor_memory_scope", scope_3);
  (void)ge::AttrUtils::GetInt(op_desc4->MutableInputDesc(0), "_tensor_memory_scope", scope_4);
  EXPECT_EQ(scope_0, 2);
  EXPECT_EQ(scope_1, 2);
  EXPECT_EQ(scope_2, 2);
  EXPECT_EQ(scope_3, 2);
  EXPECT_EQ(scope_4, 2);
}

TEST_F(GraphPassUtilUT, test_set_pair_tensor_attr_with_ge_local) {
  auto graph = std::make_shared<ComputeGraph>("test");
  ge::OpDescPtr op_desc1 = std::make_shared<ge::OpDesc>("node1", "Relu");
  ge::OpDescPtr op_desc2 = std::make_shared<ge::OpDesc>("node2", "Reshape");
  ge::OpDescPtr op_desc3 = std::make_shared<ge::OpDesc>("node3", "Relu");
  ge::OpDescPtr op_desc4 = std::make_shared<ge::OpDesc>("node4", "Relu");

  vector<int64_t> dim = {4, 4, 1, 4};
  GeShape shape(dim);
  GeTensorDesc tenosr_desc0(shape, ge::FORMAT_NCHW, ge::DT_FLOAT16);
  tenosr_desc0.SetOriginFormat(FORMAT_NCHW);
  tenosr_desc0.SetOriginDataType(DT_FLOAT);

  vector<int64_t> dim2 = {16, 1, 1, 2};
  GeShape shape2(dim2);
  GeTensorDesc tenosr_desc1(shape2, ge::FORMAT_NCHW, ge::DT_FLOAT16);
  tenosr_desc1.SetOriginFormat(FORMAT_NCHW);
  tenosr_desc1.SetOriginDataType(DT_FLOAT);

  GeTensorDesc tenosr_desc2(tenosr_desc1);

  op_desc1->AddInputDesc(tenosr_desc2);
  op_desc1->AddOutputDesc(tenosr_desc1);

  op_desc2->AddInputDesc(tenosr_desc0);
  op_desc2->AddOutputDesc(tenosr_desc2);

  op_desc3->AddInputDesc(tenosr_desc2);
  op_desc3->AddOutputDesc(tenosr_desc1);

  op_desc4->AddInputDesc(tenosr_desc2);
  op_desc4->AddOutputDesc(tenosr_desc1);

  ge::NodePtr node1 = graph->AddNode(op_desc1);
  ge::NodePtr node2 = graph->AddNode(op_desc2);
  ge::NodePtr node3 = graph->AddNode(op_desc3);
  ge::NodePtr node4 = graph->AddNode(op_desc4);

  ge::GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node2->GetOutDataAnchor(0), node3->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node4->GetInDataAnchor(0));
  std::map<std::string, int> attr_val;
  attr_val["_tensor_memory_scope"] = 2;
  GraphPassUtil::SetPairTensorAttr(node3, 0, attr_val);
  int64_t scope_0 = -1;
  int64_t scope_1 = -1;
  int64_t scope_2 = -1;
  int64_t scope_3 = -1;
  int64_t scope_4 = -1;
  (void)ge::AttrUtils::GetInt(op_desc1->MutableOutputDesc(0), "_tensor_memory_scope", scope_0);
  (void)ge::AttrUtils::GetInt(op_desc2->MutableInputDesc(0), "_tensor_memory_scope", scope_1);
  (void)ge::AttrUtils::GetInt(op_desc2->MutableOutputDesc(0), "_tensor_memory_scope", scope_2);
  (void)ge::AttrUtils::GetInt(op_desc3->MutableInputDesc(0), "_tensor_memory_scope", scope_3);
  (void)ge::AttrUtils::GetInt(op_desc4->MutableInputDesc(0), "_tensor_memory_scope", scope_4);
  EXPECT_EQ(scope_0, 2);
  EXPECT_TRUE(scope_1 == -1);
  EXPECT_TRUE(scope_2 == -1);
  EXPECT_EQ(scope_3, 2);
  EXPECT_EQ(scope_4, 2);
}
}

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// #include <gtest/gtest.h>
//
// #include "graph/utils/ge_ir_utils.h"
// #include "graph/utils/attr_utils.h"
// #include "graph/utils/graph_utils_ex.h"
// #include "graph/op_desc.h"
// #include "graph/compute_graph.h"
// #include "graph_builder_utils.h"
// #include "graph/node.h"
// #include "graph/normal_graph/node_impl.h"
// #include "test_std_structs.h"
// #include "graph/attribute_group/attr_group_shape_env.h"
// #include "graph/attribute_group/attr_group_symbolic_desc.h"
//
// namespace ge {
// namespace {
// ComputeGraphPtr BuildGraph0() {
//   auto builder = ut::GraphBuilder("root");
//   const auto &data1 = builder.AddNode("data1", "Data", 1, 1);
//   const auto &data2 = builder.AddNode("data2", "Data", 1, 1);
//   const auto &add1 = builder.AddNode("add1", "AddN", 20, 1);
//   builder.AddDataEdge(data1, 0, add1, 0);
//   builder.AddDataEdge(data2, 0, add1, 1);
//   builder.AddDataEdge(data2, 0, add1, 2);
//   builder.AddDataEdge(data2, 0, add1, 3);
//   builder.AddDataEdge(data2, 0, add1, 4);
//   return builder.GetGraph();
// }
// }
// static ComputeGraphPtr CreateGraph_1_1_224_224(float *tensor_data) {
//   ut::GraphBuilder builder("graph1");
//   auto data1 = builder.AddNode("data1", "Data", {}, {"y"});
//   AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
//   AttrUtils::SetFloat(data1->GetOpDesc(), "index2", 1.0f);
//   auto const1 = builder.AddNode("const1", "Const", {}, {"y"});
//   GeTensorDesc const1_td;
//   const1_td.SetShape(GeShape({1, 1, 224, 224}));
//   const1_td.SetOriginShape(GeShape({1, 1, 224, 224}));
//   const1_td.SetFormat(FORMAT_NCHW);
//   const1_td.SetOriginFormat(FORMAT_NCHW);
//   const1_td.SetDataType(DT_FLOAT);
//   const1_td.SetOriginDataType(DT_FLOAT);
//   GeTensor tensor(const1_td);
//   tensor.SetData(reinterpret_cast<uint8_t *>(tensor_data), sizeof(float) * 224 * 224);
//   AttrUtils::SetTensor(const1->GetOpDesc(), "value", tensor);
//   auto add1 = builder.AddNode("add1", "Add", {"x1", "x2"}, {"y"});
//   auto netoutput1 = builder.AddNode("NetOutputNode", "NetOutput", {"x"}, {});
//   ge::AttrUtils::SetListListInt(add1->GetOpDesc()->MutableOutputDesc(0), "list_list_i", {{1, 0, 0, 0}});
//   ge::AttrUtils::SetListInt(add1->GetOpDesc(), "list_i", {1});
//   ge::AttrUtils::SetListStr(add1->GetOpDesc(), "list_s", {"1"});
//   ge::AttrUtils::SetListFloat(add1->GetOpDesc(), "list_f", {1.0});
//   ge::AttrUtils::SetListBool(add1->GetOpDesc(), "list_b", {false});
//   builder.AddDataEdge(data1, 0, add1, 0);
//   builder.AddDataEdge(const1, 0, add1, 1);
//   builder.AddDataEdge(add1, 0, netoutput1, 0);
//
//   return builder.GetGraph();
// }
//
// class GeIrUtilsUt : public testing::Test {};
//
// TEST_F(GeIrUtilsUt, ModelSerialize) {
//   ge::Model model1("model", "");
//   ut::GraphBuilder builder("void");
//   auto data_node = builder.AddNode("data", "Data", {}, {"y"});
//   auto add_node = builder.AddNode("add", "Add", {}, {"y"});
//   float tensor_data[224 * 224] = {1.0f};
//   ComputeGraphPtr compute_graph = CreateGraph_1_1_224_224(tensor_data);
//   compute_graph->AddInputNode(data_node);
//   compute_graph->AddOutputNode(add_node);
//   model1.SetGraph(compute_graph);
//   onnx::ModelProto model_proto;
//   EXPECT_TRUE(OnnxUtils::ConvertGeModelToModelProto(model1, model_proto));
//   ge::Model model2;
//   EXPECT_TRUE(ge::IsEqual(std::string("test"), std::string("test"), "tag"));
//   EXPECT_FALSE(ge::IsEqual(300, 20, "tag"));
// }
//
// TEST_F(GeIrUtilsUt, ModelSerializeSetSubgraphs) {
//   ge::Model model1("model", "");
//   ut::GraphBuilder builder("test0");
//   auto data_node = builder.AddNode("data", "Data", {}, {"y"});
//   auto add_node = builder.AddNode("add", "Add", {}, {"y"});
//   auto graph = builder.GetGraph();
//
//   ut::GraphBuilder sub_builder("sub1");
//   auto sub_graph_1 = sub_builder.GetGraph();
//   std::vector<std::shared_ptr<ComputeGraph>> subgraphs;
//   subgraphs.push_back(sub_graph_1);
//
//   graph->SetAllSubgraphs(subgraphs);
//   model1.SetGraph(graph);
//   onnx::ModelProto model_proto;
//   bool ret = OnnxUtils::ConvertGeModelToModelProto(model1, model_proto);
//   EXPECT_EQ(ret, true);
// }
//
// TEST_F(GeIrUtilsUt, EncodeDataTypeUndefined) {
//   DataType data_type = DT_DUAL;
//   int ret = OnnxUtils::EncodeDataType(data_type);
//   EXPECT_EQ(ret, onnx::TensorProto_DataType_UNDEFINED);
// }
//
// TEST_F(GeIrUtilsUt, EncodeNodeDescFail) {
//   NodePtr node;
//   onnx::NodeProto *node_proto = nullptr;
//   bool ret = OnnxUtils::EncodeNodeDesc(node, node_proto);
//   EXPECT_EQ(ret, false);
// }
//
// TEST_F(GeIrUtilsUt, EncodeGraphFail) {
//   ConstComputeGraphPtr graph;
//   onnx::GraphProto *graph_proto = nullptr;
//   bool ret = OnnxUtils::EncodeGraph(graph, graph_proto);
//   EXPECT_EQ(ret, false);
// }
//
// TEST_F(GeIrUtilsUt, EncodeNodeFail) {
//   NodePtr node;
//   onnx::NodeProto *node_proto = nullptr;
//   bool ret = OnnxUtils::EncodeNode(node, node_proto);
//   EXPECT_EQ(ret, false);
// }
//
// TEST_F(GeIrUtilsUt, EncodeNodeLinkFail) {
//   NodePtr node;
//   onnx::NodeProto *node_proto = nullptr;
//   bool ret = OnnxUtils::EncodeNodeLink(node, node_proto);
//   EXPECT_EQ(ret, false);
// }
//
// TEST_F(GeIrUtilsUt, ConvertGeModelToModelProtoFail) {
//   ge::Model model;
//   onnx::ModelProto model_proto;
//   bool ret = OnnxUtils::ConvertGeModelToModelProto(model, model_proto);
//   EXPECT_EQ(ret, false);
// }
//
// TEST_F(GeIrUtilsUt, ConvertGeModelToModelProtoGraphProtoIsNull) {
//   ge::Model model("model", "");
//   ComputeGraphPtr compute_graph;
//   model.SetGraph(compute_graph);
//   onnx::ModelProto model_proto;
//   bool ret = OnnxUtils::ConvertGeModelToModelProto(model, model_proto);
//   EXPECT_EQ(ret, false);
// }
//
// TEST_F(GeIrUtilsUt, DecodeNodeLinkImpFail) {
//   OnnxUtils::NodeLinkInfo item;
//   NodePtr node_ptr;
//   bool ret = OnnxUtils::DecodeNodeLinkImp(item, node_ptr);
//   EXPECT_EQ(ret, false);
// }
//
// TEST_F(GeIrUtilsUt, DecodeNodeDescFail) {
//   onnx::NodeProto *node_proto = nullptr;
//   OpDescPtr op_desc;
//   bool ret = OnnxUtils::DecodeNodeDesc(node_proto, op_desc);
//   EXPECT_EQ(ret, false);
// }
//
// TEST_F(GeIrUtilsUt, DecodeGraphFail) {
//   int32_t recursion_depth = 20;
//   onnx::GraphProto graph_proto;
//   ComputeGraphPtr graph;
//   bool ret = OnnxUtils::DecodeGraph(recursion_depth, graph_proto, graph);
//   EXPECT_EQ(ret, false);
// }
//
// TEST_F(GeIrUtilsUt, DecodeNodeLinkImpGetDataAnchorFail) {
//   auto builder = ut::GraphBuilder("test1");
//   const auto &node1 = builder.AddNode("node1", "node", 1, 1);
//   OnnxUtils::NodeLinkInfo item("node0", 1, node1, 1, "node1");
//   bool ret = OnnxUtils::DecodeNodeLinkImp(item, node1);
//   EXPECT_EQ(ret, false);
// }
//
// TEST_F(GeIrUtilsUt, DecodeNodeLinkImpGetDataAnchorTrue) {
//   auto builder = ut::GraphBuilder("test1");
//   const auto &node1 = builder.AddNode("node1", "Data", 1, 1);
//   const auto &node2 = builder.AddNode("node2", "NetOutput", 1, 0);
//   OnnxUtils::NodeLinkInfo item("node1", 0, node2, 0, "node2");
//   bool ret = OnnxUtils::DecodeNodeLinkImp(item, node1);
//   EXPECT_EQ(ret, true);
// }
//
// TEST_F(GeIrUtilsUt, DecodeNodeLinkImpGetDataAnchorImplIsNull) {
//   auto builder = ut::GraphBuilder("test1");
//   const auto &node1 = builder.AddNode("node1", "Data", 1, 1);
//   const auto &node2 = builder.AddNode("node2", "NetOutput", 1, 0);
//   OnnxUtils::NodeLinkInfo item("node1", 0, node2, 5, "node2");
//   bool ret = OnnxUtils::DecodeNodeLinkImp(item, node1);
//   EXPECT_EQ(ret, false);
// }
//
// TEST_F(GeIrUtilsUt, DecodeNodeLinkImpGetControlAnchorTrue) {
//   auto builder = ut::GraphBuilder("test1");
//   const auto &node1 = builder.AddNode("node1", "Data", 1, 1);
//   const auto &node2 = builder.AddNode("node2", "NetOutput", 1, 0);
//   OnnxUtils::NodeLinkInfo item("node1", -1, node2, 0, "node2");
//   bool ret = OnnxUtils::DecodeNodeLinkImp(item, node1);
//   EXPECT_EQ(ret, true);
// }
//
// TEST_F(GeIrUtilsUt, DecodeGraphTrue) {
//   int32_t recursion_depth = 10;
//   auto builder = ut::GraphBuilder("test0");
//   const auto &node1 = builder.AddNode("node1", "Data", 1, 1);
//   const auto &node2 = builder.AddNode("node2", "NetOutput", 1, 0);
//   auto graph = builder.GetGraph();
//   onnx::GraphProto graph_proto;
//   bool ret = OnnxUtils::DecodeGraph(recursion_depth, graph_proto, graph);
//   EXPECT_EQ(ret, true);
// }
//
// TEST_F(GeIrUtilsUt, AddInputAndOutputNodesForGraphAddInputNodeFail) {
//   auto builder = ut::GraphBuilder("test0");
//   const auto &node1 = builder.AddNode("node1", "Data", 1, 1);
//   const auto &node2 = builder.AddNode("node2", "NetOutput", 1, 0);
//   auto graph = builder.GetGraph();
//   onnx::GraphProto graph_proto;
//   graph_proto.add_input();
//   std::map<std::string, NodePtr> node_map;
//   node_map.insert(pair<std::string, NodePtr>("node2", node2));
//   bool ret = OnnxUtils::AddInputAndOutputNodesForGraph(graph_proto, graph, node_map);
//   EXPECT_EQ(ret, false);
// }
//
// TEST_F(GeIrUtilsUt, AddInputAndOutputNodesForGraphAddOutputNodeFail) {
//   auto builder = ut::GraphBuilder("test0");
//   const auto &node1 = builder.AddNode("node1", "Data", 1, 1);
//   const auto &node2 = builder.AddNode("node2", "NetOutput", 1, 0);
//   auto graph = builder.GetGraph();
//   onnx::GraphProto graph_proto;
//   graph_proto.add_output();
//   std::map<std::string, NodePtr> node_map;
//   node_map.insert(pair<std::string, NodePtr>("node1", node1));
//   bool ret = OnnxUtils::AddInputAndOutputNodesForGraph(graph_proto, graph, node_map);
//   EXPECT_EQ(ret, false);
// }
//
// TEST_F(GeIrUtilsUt, DecodeNodeLinkDstNodeIsNull) {
//   auto builder = ut::GraphBuilder("test0");
//   const auto &node1 = builder.AddNode("node1", "Data", 1, 1);
//   const auto &node2 = builder.AddNode("node2", "NetOutput", 1, 0);
//   auto graph = builder.GetGraph();
//   onnx::NodeProto node_proto;
//   node_proto.add_input();
//   std::vector<onnx::NodeProto> node_proto_vector;
//   node_proto_vector.push_back(node_proto);
//   std::map<std::string, NodePtr> node_map;
//   node_map.insert(pair<std::string, NodePtr>("node2", node2));
//   bool ret = OnnxUtils::DecodeNodeLink(node_proto_vector, node_map);
//   EXPECT_EQ(ret, false);
// }
//
// TEST_F(GeIrUtilsUt, DecodeAttributeAttrProtoTypeIsNotStrings) {
//   ge::onnx::AttributeProto attr_proto;
//   std::vector<std::string> strings;
//   strings.push_back("node1");
//   OnnxUtils::DecodeAttribute(attr_proto, strings);
//   EXPECT_EQ(strings.size(), 1);
// }
//
// TEST_F(GeIrUtilsUt, DecodeAttributeAttrProtoTypeIsNotInts) {
//   ge::onnx::AttributeProto attr_proto;
//   std::vector<int64_t> ints;
//   ints.push_back(1);
//   ints.push_back(2);
//   OnnxUtils::DecodeAttribute(attr_proto, ints);
//   EXPECT_EQ(ints.size(), 2);
// }
//
// TEST_F(GeIrUtilsUt, DecodeAttributeAttrProtoTypeIsNotInt) {
//   ge::onnx::AttributeProto attr_proto;
//   int64_t value = 1;
//   OnnxUtils::DecodeAttribute(attr_proto, value);
//   EXPECT_TRUE(value == 1);
// }
//
// TEST_F(GeIrUtilsUt, DecodeAttributeAttrProtoTypeIsNotString) {
//   ge::onnx::AttributeProto attr_proto;
//   std::string value = "1";
//   OnnxUtils::DecodeAttribute(attr_proto, value);
//   EXPECT_EQ(value, "1");
// }
//
// TEST_F(GeIrUtilsUt, OnnxDumpCheck) {
//   auto graph = BuildGraph0();
//   auto add1 = graph->FindNode("add1");
//   auto shape_env_adder = [](auto attr_group_holder) {
//     attr_group_holder->template GetOrCreateAttrsGroup<ShapeEnvAttr>();
//   };
//   auto symbolic_adder = [](auto attr_group_holder) {
//     attr_group_holder->template GetOrCreateAttrsGroup<SymbolicDescAttr>();
//   };
//   auto normal_attr_adder = [](auto attr_holder) {
//     AttrUtils::SetStr(attr_holder, "normal_attr", "for_test");
//   };
//   shape_env_adder(add1->GetOpDesc());
//   symbolic_adder(add1->GetOpDesc());
//   shape_env_adder(add1->GetOpDesc()->MutableInputDesc(0));
//   symbolic_adder(add1->GetOpDesc()->MutableInputDesc(0));
//   shape_env_adder(add1->GetOpDesc()->MutableInputDesc(1));
//   normal_attr_adder(add1->GetOpDesc()->MutableInputDesc(1));
//   ge::onnx::NodeProto node_proto;
//   EXPECT_TRUE(OnnxUtils::EncodeNode(add1, &node_proto));
//   bool tensor_structured_show = false;
//   for (const auto &attr : node_proto.attribute()) {
//     if (attr.name().find("input_desc_0") == 0U) {
//       tensor_structured_show = true;
//       const std::string kExpected = R"PROTO(name: "input_desc_0"
// s: "{\"attr_groups\":\"attr_group_def {\\n  shape_env_attr_group {\\n    shape_setting {\\n    }\\n  }\\n}\\nattr_group_def {\\n  tensor_attr_group {\\n  }\\n}\\n\",\"cmps_size\":0,\"cmps_tab\":\"\",\"cmps_tab_offset\":0,\"data_offset\":0,\"device_type\":\"NPU\",\"dtype\":\"DT_FLOAT\",\"input_tensor\":0,\"layout\":\"NCHW\",\"origin_dtype\":\"DT_FLOAT\",\"origin_layout\":\"NCHW\",\"origin_shape\":[1,1,224,224],\"output_tensor\":0,\"real_dim_cnt\":0,\"reuse_input\":0,\"shape\":[1,1,224,224],\"size\":0,\"weight_size\":0}"
// type: STRING
// )PROTO";
//       EXPECT_EQ(attr.DebugString(), kExpected);
//     }
//     if (attr.name() == "input_desc_1") {
//       tensor_structured_show = true;
//       const std::string kExpected = R"PROTO(name: "input_desc_1"
// s: "{\"attr_groups\":\"attr_group_def {\\n  shape_env_attr_group {\\n    shape_setting {\\n    }\\n  }\\n}\\n\",\"cmps_size\":0,\"cmps_tab\":\"\",\"cmps_tab_offset\":0,\"data_offset\":0,\"device_type\":\"NPU\",\"dtype\":\"DT_FLOAT\",\"input_tensor\":0,\"layout\":\"NCHW\",\"normal_attr\":\"s: \\\"for_test\\\"\\n\",\"origin_dtype\":\"DT_FLOAT\",\"origin_layout\":\"NCHW\",\"origin_shape\":[1,1,224,224],\"output_tensor\":0,\"real_dim_cnt\":0,\"reuse_input\":0,\"shape\":[1,1,224,224],\"size\":0,\"weight_size\":0}"
// type: STRING
// )PROTO";
//       EXPECT_EQ(attr.DebugString(), kExpected);
//     }
//   }
//   EXPECT_TRUE(tensor_structured_show);
// }
// }

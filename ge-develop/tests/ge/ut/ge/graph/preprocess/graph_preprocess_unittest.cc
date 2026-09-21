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
#include <memory>
#include "ge_graph_dsl/graph_dsl.h"

#include "macro_utils/dt_public_scope.h"
#include "common/ge_inner_error_codes.h"
#include "common/types.h"
#include "common/util.h"
#include "api/gelib/gelib.h"
#include "graph/passes/graph_builder_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/manager/graph_var_manager.h"
#include "stub/gert_runtime_stub.h"

#include "graph/node.h"
#include "graph/preprocess/graph_prepare.h"
#include "ge/ge_api.h"
#include "macro_utils/dt_public_unscope.h"
#include "common/share_graph.h"
#include "graph/ge_local_context.h"
#include "external/ge_common/ge_api_types.h"

using namespace std;
namespace ge {
class UtestGraphPreproces : public testing::Test {
 protected:
  void SetUp() {
  }
  void TearDown() {
  }
};

static ge::OpDescPtr CreateOpDesc(string name = "", string type = "") {
  auto op_desc = std::make_shared<ge::OpDesc>(name, type);
  op_desc->SetStreamId(0);
  op_desc->SetId(0);

  op_desc->SetWorkspace({});
  op_desc->SetWorkspaceBytes({});
  op_desc->SetInputOffset({100, 200});
  op_desc->SetOutputOffset({100, 200});
  return op_desc;
}

ComputeGraphPtr BuildGraph1(){
  auto builder = ut::GraphBuilder("g1");
  auto data1 = builder.AddNode("data1",DATA,1,1);
  auto data_opdesc = data1->GetOpDesc();
  AttrUtils::SetInt(data_opdesc, ATTR_NAME_INDEX, 0);
  data1->UpdateOpDesc(data_opdesc);
  return builder.GetGraph();
}

ComputeGraphPtr BuildGraph2() {
  auto builder = ut::GraphBuilder("g2");
  auto data1 = builder.AddNode("data1", DATA, 1, 1, FORMAT_NCHW, DT_FLOAT, std::vector<int64_t>({22, -1}));
  ge::AttrUtils::SetStr(data1->GetOpDesc(), ATTR_ATC_USER_DEFINE_DATATYPE, "DT_INT8");
  auto data_opdesc = data1->GetOpDesc();
  AttrUtils::SetInt(data_opdesc, ATTR_NAME_INDEX, 0);

  data1->UpdateOpDesc(data_opdesc);
  return builder.GetGraph();
}

ComputeGraphPtr BuildGraph3() {
  auto builder = ut::GraphBuilder("g3");
  auto data1 = builder.AddNode("data1", DATA, 1, 1, FORMAT_NCHW, DT_FLOAT);
  ge::AttrUtils::SetStr(data1->GetOpDesc(), ATTR_ATC_USER_DEFINE_DATATYPE, "DT_INT8");
  auto data_opdesc = data1->GetOpDesc();
  AttrUtils::SetInt(data_opdesc, ATTR_NAME_INDEX, 0);

  data1->UpdateOpDesc(data_opdesc);
  return builder.GetGraph();
}

ComputeGraphPtr BuildGraph3Data() {
  auto builder = ut::GraphBuilder("g3");
  auto data1 = builder.AddNode("data1", DATA, 1, 1, FORMAT_NCHW, DT_FLOAT);
  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 1);
  ge::AttrUtils::SetStr(data1->GetOpDesc(), ATTR_ATC_USER_DEFINE_DATATYPE, "DT_INT8");
  auto data_opdesc = data1->GetOpDesc();
  AttrUtils::SetInt(data_opdesc, ATTR_NAME_INDEX, 0);
  AttrUtils::SetStr(data_opdesc, ATTR_ATC_USER_DEFINE_FORMAT, "NC1HWC0");
  builder.AddDataEdge(data1, 0, netoutput, 0);

  data1->UpdateOpDesc(data_opdesc);
  return builder.GetGraph();
}

ComputeGraphPtr BuildGraph5() {
  auto builder = ut::GraphBuilder("g5");
  auto data1 = builder.AddNode("input1", DATA, 1, 1, FORMAT_NCHW, DT_FLOAT, {1, 2, 3});
  auto data2 = builder.AddNode("input2", DATA, 1, 1, FORMAT_NCHW, DT_FLOAT, {4, 10});
  auto add = builder.AddNode("add", ADD, 2, 1);
  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 1);

  builder.AddDataEdge(data1, 0, add, 0);
  builder.AddDataEdge(data2, 0, add, 1);
  builder.AddDataEdge(add, 0,netoutput, 0);
  return builder.GetGraph();
}

/*
 *   MapIndex   Data1          subgraph1        subgraph2
 *         \    /
 *          Case      ===>       Data2            Data3
 *           |
 *       Netoutput
 */
ComputeGraphPtr BuildGraph4() {
  auto builder = ut::GraphBuilder("mbatch_Case");
  auto data1 = builder.AddNode("data1", DATA, 1, 1);
  auto data_desc = data1->GetOpDesc();
  AttrUtils::SetStr(data_desc, ATTR_ATC_USER_DEFINE_DATATYPE, "DT_FLOAT16");
  AttrUtils::SetStr(data_desc, "mbatch-switch-name", "case1");
  AttrUtils::SetInt(data_desc, ATTR_NAME_INDEX, 0);
  auto mapindex1 = builder.AddNode("mapindex1", "MapIndex", 0, 1);
  auto case1 = builder.AddNode("case1", CASE, 2, 1);
  auto netoutput1 = builder.AddNode("netoutput1", NETOUTPUT, 1, 0);
  builder.AddDataEdge(mapindex1, 0, case1, 0);
  builder.AddDataEdge(data1, 0, case1, 1);
  builder.AddDataEdge(case1, 0, netoutput1, 0);
  return builder.GetGraph();
}

/*
 *   MapIndex   Data1          subgraph1        subgraph2
 *         \    /
 *          Merge      ===>       Data2            Data3
 *           |
 *       Netoutput
 */
ComputeGraphPtr BuildGraph4Data() {
  auto builder = ut::GraphBuilder("mbatch_Case");

  auto data1 = builder.AddNode("data1", DATA, 1, 1);
  auto data_desc = data1->GetOpDesc();
  AttrUtils::SetStr(data_desc, ATTR_ATC_USER_DEFINE_DATATYPE, "DT_FLOAT16");
  AttrUtils::SetStr(data_desc, "mbatch-switch-name", "merge1");
  AttrUtils::SetInt(data_desc, ATTR_NAME_INDEX, 0);

  auto mapindex1 = builder.AddNode("mapindex1", "MapIndex", 0, 1);
  auto case1 = builder.AddNode("merge1", MERGE, 2, 1);
  auto netoutput1 = builder.AddNode("netoutput1", NETOUTPUT, 1, 0);
  std::vector<string> output_format_str;
  output_format_str.push_back("0:NC1HWC0");
  std::vector<string> output_datatype_str;
  output_datatype_str.push_back("0:DT_FLOAT16");
  (void)ge::AttrUtils::SetListStr(netoutput1->GetOpDesc(), ATTR_ATC_USER_DEFINE_DATATYPE, output_format_str);
  (void)ge::AttrUtils::SetListStr(netoutput1->GetOpDesc(), ATTR_ATC_USER_DEFINE_FORMAT, output_format_str);
  builder.AddDataEdge(mapindex1, 0, case1, 0);
  builder.AddDataEdge(data1, 0, case1, 1);
  builder.AddDataEdge(case1, 0, netoutput1, 0);

  return builder.GetGraph();
}

ComputeGraphPtr BuildGraph4DynShape() {
  auto builder = ut::GraphBuilder("mbatch_Case");

  auto data1 = builder.AddNode("data1", DATA, 1, 1);
  auto data_desc = data1->GetOpDesc();
  AttrUtils::SetStr(data_desc, ATTR_ATC_USER_DEFINE_DATATYPE, "DT_FLOAT16");
  AttrUtils::SetStr(data_desc, "mbatch-switch-name", "case1");
  AttrUtils::SetInt(data_desc, ATTR_NAME_INDEX, 0);

  auto mapindex1 = builder.AddNode("mapindex1", "MapIndex", 0, 1);
  auto case1 = builder.AddNode("case1", CASE, 2, 1);
  auto netoutput1 = builder.AddNode("netoutput1", NETOUTPUT, 1, 0);
  std::vector<std::string> output_format_str;
  output_format_str.push_back("0:NC1HWC0");
  ge::AttrUtils::SetListStr(netoutput1->GetOpDesc(), ATTR_ATC_USER_DEFINE_FORMAT, output_format_str);

  builder.AddDataEdge(mapindex1, 0, case1, 0);
  builder.AddDataEdge(data1, 0, case1, 1);
  builder.AddDataEdge(case1, 0, netoutput1, 0);

  return builder.GetGraph();
}

ComputeGraphPtr BuildGraph4_Subgraph(string graph_name) {
  auto builder = ut::GraphBuilder(graph_name);
  auto data1 = builder.AddNode(graph_name + "_data1", DATA, 1, 1);
  auto data_desc = data1->GetOpDesc();
  AttrUtils::SetInt(data_desc, ATTR_NAME_PARENT_NODE_INDEX, 1);
  return builder.GetGraph();
}

ComputeGraphPtr BuildGraph6() {
  auto builder = ut::GraphBuilder("g6");
  auto data1 = builder.AddNode("input1", DATA, 1, 1, FORMAT_NCHW, DT_FLOAT, {3, -1, -1, 5});
  auto data2 = builder.AddNode("input2", DATA, 1, 1, FORMAT_NCHW, DT_FLOAT, {});
  AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 0);
  AttrUtils::SetInt(data2->GetOpDesc(), ATTR_NAME_INDEX, 1);
  auto add = builder.AddNode("add", ADD, 2, 1);
  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);

  builder.AddDataEdge(data1, 0, add, 0);
  builder.AddDataEdge(data2, 0, add, 1);
  builder.AddDataEdge(add, 0,netoutput, 0);
  return builder.GetGraph();
}

ComputeGraphPtr BuildGraph7() {
  auto builder = ut::GraphBuilder("g7");
  auto input1 = builder.AddNode("input1", DATA, 1, 1, FORMAT_ND, DT_FLOAT, {2, 2});
  auto input2 = builder.AddNode("input2", CONSTANT, 0, 1, FORMAT_ND, DT_FLOAT, {2, 2});
  AttrUtils::SetInt(input1->GetOpDesc(), ATTR_NAME_INDEX, 0);
  AttrUtils::SetInt(input2->GetOpDesc(), ATTR_NAME_INDEX, 1);
  auto add = builder.AddNode("add", ADD, 2, 1, FORMAT_NCHW, DT_FLOAT, {2, 2});
  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);

  builder.AddDataEdge(input1, 0, add, 0);
  builder.AddDataEdge(input2, 0, add, 1);
  builder.AddDataEdge(add, 0,netoutput, 0);

  GeTensorDesc tensor_desc(GeShape({1, 2}));
  GeTensor tensor(tensor_desc);
  unique_ptr<uint8_t []> data = unique_ptr<uint8_t []>(new uint8_t[8]);
  tensor.SetData(data.get(), 8);
  AttrUtils::SetTensor(input2->GetOpDesc(), ATTR_NAME_WEIGHTS, tensor);
  (void)AttrUtils::SetBool(add->GetOpDesc(), ATTR_SINGLE_OP_SCENE, true);

  ComputeGraphPtr compute_graph = builder.GetGraph();
  (void)AttrUtils::SetBool(compute_graph, ATTR_SINGLE_OP_SCENE, true);
  return compute_graph;
}

/*
 *
 *  netoutput1
 *    |
 *  merge1
 *   |
 * data1
 */
ComputeGraphPtr BuildGraph8() {
  ut::GraphBuilder builder("g1");
  auto data1 = builder.AddNode("data1", "Data", 1, 1, FORMAT_NCHW, DT_FLOAT, {-1, 3, 224, 224});
  auto addn1 = builder.AddNode("merge1", MERGE, 1, 1);
  auto netoutput1 = builder.AddNode("netoutput1", "NetOutput", 1, 0);

  EXPECT_TRUE(AttrUtils::SetInt(addn1->GetOpDesc(), "TEST_OP_ATTR", 512));
  auto input_desc = addn1->GetOpDesc()->GetInputDesc(0);
  EXPECT_TRUE(AttrUtils::SetInt(input_desc, "TEST_ATTR", 1024));
  addn1->GetOpDesc()->UpdateInputDesc(0, input_desc);

  builder.AddDataEdge(data1, 0, addn1, 0);
  builder.AddDataEdge(addn1, 0, netoutput1, 0);
  auto graph = builder.GetGraph();

  graph->SetInputSize(1);
  graph->SetInputsOrder({"data1"});
  graph->AddInputNode(data1);

  return graph;
}

/*
 *
 *  netoutput1
 *    |
 *  while1
 *   |
 * data1
 */
ComputeGraphPtr BuildGraph9() {
  ut::GraphBuilder builder("g1");
  auto data1 = builder.AddNode("data1", "Data", 1, 1, FORMAT_NCHW, DT_FLOAT, {-1, 3, 224, 224});
  auto addn1 = builder.AddNode("while1", WHILE, 1, 1);
  auto netoutput1 = builder.AddNode("netoutput1", "NetOutput", 1, 0);

  EXPECT_TRUE(AttrUtils::SetInt(addn1->GetOpDesc(), "TEST_OP_ATTR", 512));
  auto input_desc = addn1->GetOpDesc()->GetInputDesc(0);
  EXPECT_TRUE(AttrUtils::SetInt(input_desc, "TEST_ATTR", 1024));
  addn1->GetOpDesc()->UpdateInputDesc(0, input_desc);

  builder.AddDataEdge(data1, 0, addn1, 0);
  builder.AddDataEdge(addn1, 0, netoutput1, 0);
  auto graph = builder.GetGraph();

  graph->SetInputSize(1);
  graph->SetInputsOrder({"data1"});
  graph->AddInputNode(data1);

  return graph;
}

/*
 *
 *   netoutput1
 *       |
 *      add
 *    /    \
 * data1  data2
 */
Graph BuildGraph10() {
  const auto const1 = OP_CFG(CONSTANT)
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, {16, 16, 224, 224})
      .Build("const1");
  const auto data1 = OP_CFG(DATA)
      .TensorDesc(FORMAT_NCHW, DT_FLOAT16, {16, 16, 224, 224})
      .Build("data1");
  const auto data2 = OP_CFG(DATA)
      .TensorDesc(FORMAT_NCHW, DT_FLOAT16, {16, 16, 224, 224})
      .Build("data2");
  const auto merge = OP_CFG(MERGE)
      .InCnt(1).OutCnt(2)
      .TensorDesc(FORMAT_NHWC, DT_FLOAT, {16, 224, 224, 16})
      .Build("merge1");
  const auto net_output = OP_CFG(NETOUTPUT)
      .TensorDesc(FORMAT_NHWC, DT_FLOAT, {16, 224, 224, 16})
      .Build("net_output1");
  const auto add = OP_CFG(ADD)
      .TensorDesc(FORMAT_NHWC, DT_FLOAT, {16, 224, 224, 16})
      .Build("add");

  DEF_GRAPH(g1) {
                  CHAIN(NODE(data1)->EDGE(0, 0)->NODE(add));
                  CHAIN(NODE(data2)->EDGE(0, 1)->NODE(add)->EDGE(0, 0)->NODE(net_output));
                };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto node = compute_graph->FindNode("add");
  auto op_desc = node->GetOpDesc();
  op_desc->MutableAllOutputName().erase("__output0");
  op_desc->MutableAllOutputName()["__input0"] = 0;
  return graph;
}

/*
 *
 *   netoutput1
 *       |
 *      add
 *    /    \
 * const1  data2
 */
Graph BuildGraph11() {
  const auto const1 = OP_CFG(CONSTANT)
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, {16, 16, 224, 224})
      .Build("const1");
  const auto data1 = OP_CFG(DATA)
      .TensorDesc(FORMAT_NCHW, DT_FLOAT16, {16, 16, 224, 224})
      .Build("data1");
  const auto data2 = OP_CFG(DATA)
      .TensorDesc(FORMAT_NCHW, DT_FLOAT16, {16, 16, 224, 224})
      .Build("data2");
  const auto merge = OP_CFG(MERGE)
      .InCnt(1).OutCnt(2)
      .TensorDesc(FORMAT_NHWC, DT_FLOAT, {16, 224, 224, 16})
      .Build("merge1");
  const auto net_output = OP_CFG(NETOUTPUT)
      .TensorDesc(FORMAT_NHWC, DT_FLOAT, {16, 224, 224, 16})
      .Build("net_output1");
  const auto add = OP_CFG(ADD)
      .TensorDesc(FORMAT_NHWC, DT_FLOAT, {16, 224, 224, 16})
      .Build("add");

  DEF_GRAPH(g1) {
                  CHAIN(NODE(const1)->EDGE(0, 0)->NODE(add)->EDGE(0, 0)->NODE(net_output));
                  CHAIN(NODE(data2)->EDGE(0, 1)->NODE(add));
                };

  auto graph = ToGeGraph(g1);
  auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  auto node = compute_graph->FindNode("add");
  auto op_desc = node->GetOpDesc();
  op_desc->MutableAllOutputName().erase("__output0");
  op_desc->MutableAllOutputName()["__input0"] = 0;
  return graph;
}

ComputeGraphPtr BuildGraph12() {
  auto builder = ut::GraphBuilder("g12");
  auto input1 = builder.AddNode("input1", DATA, 1, 1, FORMAT_ND, DT_FLOAT, {});
  auto input2 = builder.AddNode("input2", CONSTANT, 0, 1, FORMAT_ND, DT_FLOAT, {});
  AttrUtils::SetInt(input1->GetOpDesc(), ATTR_NAME_INDEX, 0);
  AttrUtils::SetInt(input2->GetOpDesc(), ATTR_NAME_INDEX, 1);
  auto add = builder.AddNode("add", ADD, 2, 1, FORMAT_NCHW, DT_FLOAT, {});
  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);

  builder.AddDataEdge(input1, 0, add, 0);
  builder.AddDataEdge(input2, 0, add, 1);
  builder.AddDataEdge(add, 0, netoutput, 0);

  GeTensorDesc tensor_desc((GeShape()));
  GeTensor tensor(tensor_desc);
  unique_ptr<uint8_t []> data = unique_ptr<uint8_t []>(new uint8_t[8]);
  tensor.SetData(data.get(), 0);
  AttrUtils::SetTensor(input2->GetOpDesc(), ATTR_NAME_WEIGHTS, tensor);
  (void)AttrUtils::SetBool(add->GetOpDesc(), ATTR_SINGLE_OP_SCENE, true);

  ComputeGraphPtr compute_graph = builder.GetGraph();
  (void)AttrUtils::SetBool(compute_graph, ATTR_SINGLE_OP_SCENE, true);
  return compute_graph;
}

TEST_F(UtestGraphPreproces, test_dynamic_input_shape_parse) {
  ge::GraphPrepare graph_prepare;
  graph_prepare.compute_graph_ = BuildGraph6();
  // prepare user_input & graph option
  ge::GeTensorDesc tensor1;
  tensor1.SetFormat(ge::FORMAT_NCHW);
  tensor1.SetShape(ge::GeShape({3, 12, 5, 5}));
  tensor1.SetDataType(ge::DT_FLOAT);
  GeTensor input1(tensor1);
  ge::GeTensorDesc tensor2;
  tensor2.SetFormat(ge::FORMAT_NCHW);
  tensor2.SetShape(ge::GeShape());
  tensor2.SetDataType(ge::DT_FLOAT);
  GeTensor input2(tensor2);
  std::vector<GeTensor> user_input = {input1, input2};
  std::map<string,string> graph_option = {{"ge.exec.dynamicGraphExecuteMode","dynamic_execute"},
                                          {"ge.exec.dataInputsShapeRange","[3,1~20,2~10,5],[]"}};
  auto ret = graph_prepare.UpdateInput(user_input, graph_option);
  EXPECT_EQ(ret, ge::SUCCESS);
  // check data1 node output shape_range and shape
  auto data_node = graph_prepare.compute_graph_->FindNode("input1");
  auto data_output_desc = data_node->GetOpDesc()->GetOutputDescPtr(0);
  vector<int64_t> input1_expect_shape = {3,-1,-1,5};
  vector<std::pair<int64_t, int64_t>> intpu1_expect_shape_range = {{3,3},{1,20},{2,10},{5,5}};
  auto input1_result_shape = data_output_desc->GetShape();
  vector<std::pair<int64_t, int64_t>> input1_result_shape_range;
  data_output_desc->GetShapeRange(input1_result_shape_range);
  EXPECT_EQ(input1_result_shape.GetDimNum(), input1_expect_shape.size());
  EXPECT_EQ(input1_result_shape_range.size(), input1_expect_shape.size());
  for(size_t i =0; i< input1_expect_shape.size(); ++i){
      EXPECT_EQ(input1_result_shape.GetDim(i), input1_expect_shape.at(i));
  }
  for(size_t i =0; i< intpu1_expect_shape_range.size(); ++i){
     EXPECT_EQ(input1_result_shape_range.at(i).first, intpu1_expect_shape_range.at(i).first);
     EXPECT_EQ(input1_result_shape_range.at(i).second, intpu1_expect_shape_range.at(i).second);
  }
  // check data2 node output shape_range and shape
  auto data_node_2 = graph_prepare.compute_graph_->FindNode("input2");
  auto data_output_desc_2 = data_node_2->GetOpDesc()->GetOutputDescPtr(0);
  vector<std::pair<int64_t, int64_t>> intput2_result_shape_range;
  data_output_desc_2->GetShapeRange(intput2_result_shape_range);
  EXPECT_EQ(intput2_result_shape_range.size(), 0);
}

TEST_F(UtestGraphPreproces, test_dynamic_input_shape_parse_fail) {
  ge::GraphPrepare graph_prepare;
  graph_prepare.compute_graph_ = BuildGraph6();
  // prepare user_input & graph option
  ge::GeTensorDesc tensor1;
  tensor1.SetFormat(ge::FORMAT_NCHW);
  tensor1.SetShape(ge::GeShape({3, 12, 5, 5}));
  tensor1.SetDataType(ge::DT_FLOAT);
  GeTensor input1(tensor1);
  ge::GeTensorDesc tensor2;
  tensor2.SetFormat(ge::FORMAT_NCHW);
  tensor2.SetShape(ge::GeShape());
  tensor2.SetDataType(ge::DT_FLOAT);
  GeTensor input2(tensor2);
  std::vector<GeTensor> user_input = {input1, input2};
  std::map<string,string> graph_option = {{"ge.exec.dynamicGraphExecuteMode","dynamic_execute"},
                                          {"ge.exec.dataInputsShapeRange","[3,1~20,5],[]"}};
  auto ret = graph_prepare.UpdateInput(user_input, graph_option);
  EXPECT_EQ(ret, ge::PARAM_INVALID);

  ge::GeTensorDesc tensor3;
  tensor3.SetFormat(ge::FORMAT_NCHW);
  tensor3.SetShape(ge::GeShape({100, 10, 50, 100}));
  tensor3.SetDataType(ge::DT_FLOAT);
  GeTensor input3(tensor3);
  ge::GeTensorDesc tensor4;
  tensor4.SetFormat(ge::FORMAT_NCHW);
  tensor4.SetShape(ge::GeShape());
  tensor4.SetDataType(ge::DT_FLOAT);
  GeTensor input4(tensor4);
  std::vector<GeTensor> user_input1 = {input3, input4};
  std::map<string,string> graph_option1 = {{"ge.exec.dynamicGraphExecuteMode","dynamic_execute"},
                                          {"ge.exec.dataInputsShapeRange","[1~10,10,10,10],[]"}};
  ge::GraphPrepare graph_prepare1;
  graph_prepare1.compute_graph_ = BuildGraph6();
  auto ret1 = graph_prepare.UpdateInput(user_input1, graph_option1);
  EXPECT_EQ(ret1, ge::PARAM_INVALID);

  ge::GeTensorDesc tensor5;
  tensor5.SetFormat(ge::FORMAT_NCHW);
  tensor5.SetShape(ge::GeShape({100, 10, 50, 100}));
  tensor5.SetDataType(ge::DT_FLOAT);
  GeTensor input5(tensor5);
  ge::GeTensorDesc tensor6;
  tensor6.SetFormat(ge::FORMAT_NCHW);
  tensor6.SetShape(ge::GeShape());
  tensor6.SetDataType(ge::DT_FLOAT);
  GeTensor input6(tensor6);
  std::vector<GeTensor> user_input2 = {input5, input6};
  std::map<string,string> graph_option2 = {{"ge.exec.dynamicGraphExecuteMode","dynamic_execute"},
                                          {"ge.exec.dataInputsShapeRange","[10~10,10,10,10],[]"}};
  ge::GraphPrepare graph_prepare2;
  graph_prepare2.compute_graph_ = BuildGraph6();
  auto ret2 = graph_prepare.UpdateInput(user_input2, graph_option2);
  EXPECT_EQ(ret2, ge::PARAM_INVALID);
}

TEST_F(UtestGraphPreproces, test_update_input_fail) {
  ge::GraphPrepare graph_prepare;
  gert::GertRuntimeStub stub;
  stub.GetSlogStub().SetLevel(DLOG_DEBUG);
  stub.GetSlogStub().Clear();
  graph_prepare.compute_graph_ = BuildGraph1();

  ge::GeTensorDesc tensor1;
  tensor1.SetFormat(ge::FORMAT_NCHW);
  tensor1.SetShape(ge::GeShape({3, 12, 5, 5}));
  tensor1.SetDataType(ge::DT_UNDEFINED);
  GeTensor input1(tensor1);
  std::vector<GeTensor> user_input = {input1};
  std::map<string,string> graph_option;
  auto ret = graph_prepare.UpdateInput(user_input, graph_option);
  EXPECT_EQ(ret, ge::FAILED);

  graph_option.emplace("ge.exec.dataInputsShapeRange", "[1~20, 12, 5, 5],[1~20, 12, 5, 5]");
  graph_option.emplace("ge.exec.dynamicGraphExecuteMode", "dynamic_execute");
  ret = graph_prepare.UpdateInput(user_input, graph_option);
  EXPECT_EQ(ret, ge::PARAM_INVALID);
  EXPECT_NE(stub.GetSlogStub().FindLog(DLOG_DEBUG, "Option ge.exec.dataInputsShapeRange's readable name"), -1);
}

TEST_F(UtestGraphPreproces, test_UpdateInput_WithLog_Success) {
  ge::GraphPrepare graph_prepare;
  gert::GertRuntimeStub stub;
  stub.GetSlogStub().SetLevel(DLOG_INFO);
  stub.GetSlogStub().Clear();
  graph_prepare.compute_graph_ = BuildGraph6();
  ge::GeTensorDesc tensor1;
  tensor1.SetFormat(ge::FORMAT_NCHW);
  tensor1.SetShape(ge::GeShape({3, 12, 5, 5}));
  tensor1.SetDataType(ge::DT_FLOAT);
  GeTensor input1(tensor1);
  ge::GeTensorDesc tensor2;
  tensor2.SetFormat(ge::FORMAT_NCHW);
  tensor2.SetShape(ge::GeShape());
  tensor2.SetDataType(ge::DT_FLOAT);
  GeTensor input2(tensor2);
  std::vector<GeTensor> user_input = {input1, input2};
  std::map<string, string> graph_option = {{"ge.exec.dynamicGraphExecuteMode", "dynamic_execute"},
                                           {"ge.exec.dataInputsShapeRange", "[3,1~20,2~10,5],[]"}};
  const char_t *kBeforeLog =
      "[BeforeUpdate]Data [input1] with shape[3,-1,-1,5], origin_shape[], format[NCHW], origin_format[ND], "
      "dtype[DT_FLOAT].";
  const char_t *kAfterUpdateLog =
      "[AfterUpdate]Data [input1] with shape[3,12,5,5], origin_shape[3,12,5,5], format[NCHW], origin_format[ND], "
      "dtype[DT_FLOAT].";
  const char_t *kAfterUpdateDynamicLog =
      "[AfterUpdate]Data [input1] with shape[3,-1,-1,5], origin_shape[3,-1,-1,5], format[NCHW], origin_format[ND], "
      "dtype[DT_FLOAT].";

  auto ret = graph_prepare.UpdateInput(user_input, graph_option);
  EXPECT_EQ(ret, ge::SUCCESS);
  EXPECT_NE(stub.GetSlogStub().FindLog(DLOG_INFO, kBeforeLog), -1);
  EXPECT_NE(stub.GetSlogStub().FindLog(DLOG_INFO, kAfterUpdateLog), -1);
  EXPECT_NE(stub.GetSlogStub().FindLog(DLOG_INFO, kAfterUpdateDynamicLog), -1);
  stub.GetSlogStub().Clear();
}

TEST_F(UtestGraphPreproces, test_check_user_input) {
  ge::GraphPrepare graph_prepare;
  graph_prepare.compute_graph_ = BuildGraph1();

  vector<int64_t> dim = {2, -3};
  GeTensor tensor;
  tensor.SetTensorDesc(GeTensorDesc(GeShape(dim)));
  std::vector<GeTensor> user_input;
  user_input.emplace_back(tensor);

  Status ret = graph_prepare.CheckUserInput(user_input);
  EXPECT_EQ(ret, GE_GRAPH_INIT_FAILED);
}

TEST_F(UtestGraphPreproces, test_update_input_output1) {
  ge::GraphPrepare graph_prepare;
  graph_prepare.compute_graph_ = BuildGraph3();

  Status ret = graph_prepare.UpdateInputOutputByOptions();
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphPreproces, check_ref_op_data_succ) {
  GraphPrepare graph_preparer;
  ComputeGraphPtr graph_test = BuildGraph5();
  NodePtr add_node = nullptr;
  for (auto &node : graph_test->GetAllNodes()) {
    if (node->GetName() == "add") {
      add_node = node;
    }
  }
  EXPECT_NE(add_node, nullptr);
  string input_name = "__input0";
  std::set<NodePtr> ref_nodes;
  auto ret = graph_preparer.CheckRefInputNode(add_node, input_name, ref_nodes);
  auto graph = BuildGraph10();
  ret = graph_preparer.Init(graph, 0U, nullptr, nullptr);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphPreproces, check_ref_op_data_failed) {
  GraphPrepare graph_preparer;
  auto graph = BuildGraph11();
  auto ret = graph_preparer.Init(graph, 0U, nullptr, nullptr);
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(UtestGraphPreproces, test_update_dtype_mbatch_case) {
  ge::GraphPrepare graph_prepare;
  graph_prepare.compute_graph_ = BuildGraph4();
  auto parent_graph = graph_prepare.compute_graph_;
  auto subgraph1 = BuildGraph4_Subgraph("subgraph1");
  auto subgraph2 = BuildGraph4_Subgraph("subgraph2");

  auto data1 = parent_graph->FindNode("data1");
  auto data_desc = data1->GetOpDesc();

  auto case_node = parent_graph->FindNode("case1");
  EXPECT_NE(case_node, nullptr);
  case_node->GetOpDesc()->AddSubgraphName("subgraph1");
  case_node->GetOpDesc()->SetSubgraphInstanceName(0, "subgraph1");
  subgraph1->SetParentNode(case_node);
  subgraph1->SetParentGraph(parent_graph);
  EXPECT_EQ(parent_graph->AddSubgraph("subgraph1", subgraph1), GRAPH_SUCCESS);

  case_node->GetOpDesc()->AddSubgraphName("subgraph2");
  case_node->GetOpDesc()->SetSubgraphInstanceName(1, "subgraph2");
  subgraph2->SetParentNode(case_node);
  subgraph2->SetParentGraph(parent_graph);
  EXPECT_EQ(parent_graph->AddSubgraph("subgraph2", subgraph2), GRAPH_SUCCESS);

  Status ret = graph_prepare.UpdateInputOutputByOptions();
  EXPECT_EQ(ret, SUCCESS);

  auto case_desc = case_node->GetOpDesc();
  auto case_input = case_desc->MutableInputDesc(1);
  EXPECT_EQ(case_input->GetDataType(), 1);

  auto sub1_data1 = subgraph1->FindNode("subgraph1_data1");
  EXPECT_NE(sub1_data1, nullptr);
  auto data1_desc = sub1_data1->GetOpDesc();
  auto data1_input = data1_desc->MutableInputDesc(0);
  EXPECT_EQ(data1_input->GetDataType(), 1);
  auto data1_output = data1_desc->MutableOutputDesc(0);
  EXPECT_EQ(data1_output->GetDataType(), 1);
}

TEST_F(UtestGraphPreproces, test_prepare_dyn_shape) {
  map<string, string> options;
  Status ret = ge::GELib::Initialize(options);
  EXPECT_EQ(ret, SUCCESS);

  ComputeGraphPtr compute_graph = BuildGraph7();
  GraphPtr graph_ptr = std::make_shared<Graph>(GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph));

  GraphNodePtr graph_node = make_shared<GraphNode>(0);
  graph_node->SetComputeGraph(compute_graph);
  graph_node->SetGraph(graph_ptr);

  GeTensorDesc tensor_desc(GeShape({2, 16}));
  GeTensor tensor(tensor_desc);
  std::vector<GeTensor> user_input = {tensor};
  GraphPrepare graph_prepare;
  ret = graph_prepare.PrepareInit(graph_node, 0);
  EXPECT_EQ(ret, SUCCESS);
  ret = graph_prepare.PrepareDynShape();
  EXPECT_EQ(ret, SUCCESS);

  auto input2 = compute_graph->FindNode("input2");
  ASSERT_NE(input2, nullptr);
  ASSERT_NE(input2->GetOpDesc(), nullptr);
  auto out_desc = input2->GetOpDesc()->MutableOutputDesc(0);
  EXPECT_EQ(out_desc->GetShape().GetDims(), vector<int64_t>({1, 2}));
  EXPECT_EQ(out_desc->GetFormat(), FORMAT_NCHW);

  shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  EXPECT_NE(instance_ptr, nullptr);
  ret = instance_ptr->Finalize();
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphPreproces, test_prepare_empty_shape) {
  map<string, string> options;
  Status ret = ge::GELib::Initialize(options);
  EXPECT_EQ(ret, SUCCESS);

  ComputeGraphPtr compute_graph = BuildGraph12();
  GraphPtr graph_ptr = std::make_shared<Graph>(GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph));

  GraphNodePtr graph_node = make_shared<GraphNode>(0);
  graph_node->SetComputeGraph(compute_graph);
  graph_node->SetGraph(graph_ptr);

  GeTensorDesc tensor_desc((GeShape()));
  GeTensor tensor(tensor_desc);
  std::vector<GeTensor> user_input = {tensor};
  GraphPrepare graph_prepare;
  ret = graph_prepare.PrepareInit(graph_node, 0);
  EXPECT_EQ(ret, SUCCESS);
  shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  EXPECT_NE(instance_ptr, nullptr);
  ret = instance_ptr->Finalize();
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphPreproces, test_updar_variable_formats) {
  EXPECT_EQ(VarManager::Instance(0)->Init(0, 0, 0, 0), SUCCESS);
  auto builder = ut::GraphBuilder("g1");
  auto var = builder.AddNode("var", VARIABLE, 1, 1);
  auto g1 = builder.GetGraph();
  g1->SetSessionID(0);
  TransNodeInfo trans_node_info;
  VarTransRoad fusion_road;
  fusion_road.emplace_back(trans_node_info);
  VarManager::Instance(g1->GetSessionID())->SetTransRoad(var->GetName(), fusion_road);
  GraphPrepare graph_prepare;
  // trans_node_info.node_type is empty, return fail
  EXPECT_EQ(graph_prepare.UpdateVariableFormats(g1), INTERNAL_ERROR);

  auto builder1 = ut::GraphBuilder("g2");
  auto var1 = builder1.AddNode("var1", VARIABLE, 1, 1);
  auto g2 = builder1.GetGraph();
  g2->SetSessionID(0);
  GeTensorDesc input_desc(GeShape({1, 2, 3, 4}), static_cast<ge::Format>(GetFormatFromSub(ge::FORMAT_FRACTAL_Z, 240)), DT_INT32);
  GeTensorDesc output_desc(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, DT_INT32);
  TransNodeInfo trans_node_info1 = {.node_type = RESHAPE, .input = input_desc, .output = output_desc};
  VarTransRoad fusion_road1;
  fusion_road1.emplace_back(trans_node_info1);
  VarManager::Instance(g2->GetSessionID())->SetTransRoad(var1->GetName(), fusion_road1);
  AttrUtils::SetStr(var1->GetOpDesc(), REF_VAR_SRC_VAR_NAME, "var1");
  EXPECT_EQ(graph_prepare.UpdateVariableFormats(g2), SUCCESS);
}

TEST_F(UtestGraphPreproces, RecoverTransRoadForVarRef_TRANSDATA) {
  EXPECT_EQ(VarManager::Instance(0)->Init(0, 0, 0, 0), SUCCESS);
  GraphPrepare graph_prepare;
  VarTransRoad fusion_road;
  TransNodeInfo trans_node_info;

  auto builder1 = ut::GraphBuilder("g2");
  auto var1 = builder1.AddNode("var1", VARIABLE, 1, 1);
  auto g2 = builder1.GetGraph();
  g2->SetSessionID(0);
  GeTensorDesc input_desc(GeShape({1, 2, 3, 4}), static_cast<ge::Format>(GetFormatFromSub(ge::FORMAT_FRACTAL_Z, 240)), DT_INT32);
  GeTensorDesc output_desc(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, DT_INT32);
  trans_node_info = {.node_type = TRANSDATA, .input = input_desc, .output = output_desc};
  fusion_road.emplace_back(trans_node_info);
  VarManager::Instance(g2->GetSessionID())->SetTransRoad(var1->GetName(), fusion_road);
  (void)AttrUtils::SetStr(var1->GetOpDesc(), REF_VAR_SRC_VAR_NAME, "var1");
  (void)AttrUtils::SetStr(var1->GetOpDesc(), ATTR_NAME_STREAM_LABEL, "stream1");
  EXPECT_EQ(graph_prepare.UpdateVariableFormats(g2), SUCCESS);
}

TEST_F(UtestGraphPreproces, test_updar_variable_formats_insert_transpose) {
  auto builder = ut::GraphBuilder("g1");
  auto var = builder.AddNode("var_transpose", VARIABLE, 1, 1);
  auto g1 = builder.GetGraph();
  g1->SetSessionID(1); // diffenent from pre test case. cause varmanager is instance
  GeTensorDesc input_desc(GeShape({1, 2, 3, 4}), static_cast<ge::Format>(GetFormatFromSub(ge::FORMAT_FRACTAL_Z, 240)), DT_INT32);
  GeTensorDesc output_desc(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, DT_INT32);
  TransNodeInfo trans_node_info = {.node_type = TRANSDATA, .input = input_desc, .output = output_desc};
  TransNodeInfo transpose_node_info = {.node_type = TRANSPOSED, .input = input_desc, .output = output_desc};
  TransNodeInfo cast_node_info = {.node_type = CAST, .input = input_desc, .output = output_desc};
  VarTransRoad fusion_road;
  fusion_road.emplace_back(trans_node_info);
  fusion_road.emplace_back(transpose_node_info);
  fusion_road.emplace_back(cast_node_info);
  VarManager::Instance(g1->GetSessionID())->Init(0, 1, 0, 1);
  VarManager::Instance(g1->GetSessionID())->SetTransRoad(var->GetName(), fusion_road);
  GraphPrepare graph_prepare;
  EXPECT_EQ(graph_prepare.UpdateVariableFormats(g1), SUCCESS);

  // check new transdata dst_format
}

TEST_F(UtestGraphPreproces, test_update_data_netoutput_by_storageFormat) {
  ComputeGraphPtr compute_graph = BuildGraph5();
  auto set_tensor_desc_nd_to_nz = [](GeTensorDescPtr &desc) {
    desc->SetFormat(FORMAT_ND);
    desc->SetOriginFormat(FORMAT_ND);
    desc->SetShape(GeShape({-1, -1}));
    desc->SetOriginShape(GeShape({-1, -1}));
    AttrUtils::SetInt(desc, ATTR_NAME_STORAGE_FORMAT, static_cast<int64_t>(FORMAT_FRACTAL_NZ));
    vector<int64_t> storage_shape = {1, 1, 1, 16, 16};
    AttrUtils::SetListInt(desc, ATTR_NAME_STORAGE_SHAPE, storage_shape);
  };
  auto set_origin_shape_range = [](GeTensorDescPtr &desc) {
    desc->SetOriginShapeRange({{1, -1}, {1, -1}});
    desc->SetShapeRange({{1, -1}, {1, -1}});
  };

  auto input1 = compute_graph->FindNode("input1");
  ASSERT_NE(input1, nullptr);
  ASSERT_NE(input1->GetOpDesc(), nullptr);
  auto in_desc = input1->GetOpDesc()->MutableInputDesc(0);
  auto out_desc = input1->GetOpDesc()->MutableOutputDesc(0);
  set_tensor_desc_nd_to_nz(in_desc);
  set_tensor_desc_nd_to_nz(out_desc);

  auto net_output = compute_graph->FindNode("netoutput");
  ASSERT_NE(net_output, nullptr);
  ASSERT_NE(net_output->GetOpDesc(), nullptr);
  auto net_output_in_desc = net_output->GetOpDesc()->MutableInputDesc(0);
  auto net_output_out_desc = net_output->GetOpDesc()->MutableOutputDesc(0);
  set_tensor_desc_nd_to_nz(net_output_in_desc);
  set_tensor_desc_nd_to_nz(net_output_out_desc);
  set_origin_shape_range(net_output_in_desc);
  set_origin_shape_range(net_output_out_desc);

  ge::GraphPrepare graph_prepare;
  graph_prepare.compute_graph_ = compute_graph;
  vector<int64_t> storage_shape = {1, 1, 1, 16, 16};

  // 1. not shape generalized mode
  auto ret = graph_prepare.UpdateDataNetOutputByStorageFormat();
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(in_desc->GetFormat(), FORMAT_FRACTAL_NZ);
  EXPECT_EQ(in_desc->GetShape().GetDims(), storage_shape);

  // 2. format is same to storageFormat
  in_desc->SetFormat(FORMAT_ND);
  (void)AttrUtils::SetBool(input1->GetOpDesc(), ATTR_NAME_IS_OP_GENERALIZED, true);
  (void)AttrUtils::SetBool(net_output->GetOpDesc(), ATTR_NAME_IS_OP_GENERALIZED, true);
  (void)AttrUtils::SetInt(in_desc, ATTR_NAME_STORAGE_FORMAT, static_cast<int64_t>(FORMAT_ND));
  set_origin_shape_range(in_desc);
  set_origin_shape_range(out_desc);
  ret = graph_prepare.UpdateDataNetOutputByStorageFormat();
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(in_desc->GetFormat(), FORMAT_ND);
  EXPECT_EQ(in_desc->GetShape().GetDims(), storage_shape);

  // 3. transformer
  set_tensor_desc_nd_to_nz(in_desc);
  set_tensor_desc_nd_to_nz(out_desc);
  ret = graph_prepare.UpdateDataNetOutputByStorageFormat();
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(in_desc->GetFormat(), FORMAT_FRACTAL_NZ);
  EXPECT_EQ(in_desc->GetShape().GetDims(), vector<int64_t>({ -1, -1, 16, 16 }));
}

/*
 *   MapIndex   Data1          subgraph1        subgraph2
 *         \    /
 *          Case      ===>       Data2            Data3
 *           |
 *       Netoutput
 */
TEST_F(UtestGraphPreproces, test_update_data_netoutput_by_storageFormat_in_DynamicBatch) {
  // build graph
  ComputeGraphPtr compute_graph = BuildGraph4();
  auto subgraph1 = BuildGraph4_Subgraph("subgraph1");
  auto subgraph2 = BuildGraph4_Subgraph("subgraph2");

  auto case_node = compute_graph->FindNode("case1");
  EXPECT_NE(case_node, nullptr);
  case_node->GetOpDesc()->AddSubgraphName("subgraph1");
  case_node->GetOpDesc()->SetSubgraphInstanceName(0, "subgraph1");
  subgraph1->SetParentNode(case_node);
  subgraph1->SetParentGraph(compute_graph);
  EXPECT_EQ(compute_graph->AddSubgraph("subgraph1", subgraph1), GRAPH_SUCCESS);

  case_node->GetOpDesc()->AddSubgraphName("subgraph2");
  case_node->GetOpDesc()->SetSubgraphInstanceName(1, "subgraph2");
  subgraph2->SetParentNode(case_node);
  subgraph2->SetParentGraph(compute_graph);
  EXPECT_EQ(compute_graph->AddSubgraph("subgraph2", subgraph2), GRAPH_SUCCESS);

  auto set_tensor_desc_nd_to_nz = [](GeTensorDescPtr &desc) {
    desc->SetFormat(FORMAT_ND);
    desc->SetOriginFormat(FORMAT_ND);
    desc->SetShape(GeShape({-1, -1}));
    desc->SetOriginShape(GeShape({-1, -1}));
    AttrUtils::SetInt(desc, ATTR_NAME_STORAGE_FORMAT, static_cast<int64_t>(FORMAT_FRACTAL_NZ));
    vector<int64_t> storage_shape = {1, 1, 1, 16, 16};
    AttrUtils::SetListInt(desc, ATTR_NAME_STORAGE_SHAPE, storage_shape);
  };

  auto input1 = compute_graph->FindNode("data1");
  ASSERT_NE(input1, nullptr);
  ASSERT_NE(input1->GetOpDesc(), nullptr);
  auto in_desc = input1->GetOpDesc()->MutableInputDesc(0);
  auto out_desc = input1->GetOpDesc()->MutableOutputDesc(0);
  set_tensor_desc_nd_to_nz(in_desc);
  set_tensor_desc_nd_to_nz(out_desc);

  auto branch_data = subgraph1->FindFirstNodeMatchType(DATA);
  auto branch_in_desc = branch_data->GetOpDesc()->MutableInputDesc(0);
  auto branch_out_desc = branch_data->GetOpDesc()->MutableOutputDesc(0);
  set_tensor_desc_nd_to_nz(branch_in_desc);
  set_tensor_desc_nd_to_nz(branch_out_desc);

  ge::GraphPrepare graph_prepare;
  graph_prepare.compute_graph_ = compute_graph;
  vector<int64_t> storage_shape = {1, 1, 1, 16, 16};

  // 1. not shape generalized mode
  auto ret = graph_prepare.UpdateDataNetOutputByStorageFormat();
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(in_desc->GetFormat(), FORMAT_FRACTAL_NZ);
  EXPECT_EQ(in_desc->GetShape().GetDims(), storage_shape);

  auto case1 = compute_graph->FindNode("case1");
  auto case1_input_desc1 = case1->GetOpDesc()->GetInputDescPtr(1);
  EXPECT_EQ(case1_input_desc1->GetFormat(), FORMAT_FRACTAL_NZ);
  EXPECT_EQ(case1_input_desc1->GetShape().GetDims(), storage_shape);
}

TEST_F(UtestGraphPreproces, test_update_data_netoutput_shape_err) {
  ComputeGraphPtr compute_graph = BuildGraph5();
  auto set_tensor_desc_nd_to_nz = [](GeTensorDescPtr &desc) {
    desc->SetFormat(FORMAT_ND);
    desc->SetOriginFormat(FORMAT_ND);
    desc->SetShape(GeShape({-1, -1}));
    desc->SetOriginShape(GeShape({-1, -1}));
    AttrUtils::SetInt(desc, ATTR_NAME_STORAGE_FORMAT, static_cast<int64_t>(FORMAT_FRACTAL_NZ));
  };
  auto set_origin_shape_range = [](GeTensorDescPtr &desc) {
    desc->SetOriginShapeRange({{1, -1}, {1, -1}});
    desc->SetShapeRange({{1, -1}, {1, -1}});
  };

  auto input1 = compute_graph->FindNode("input1");
  ASSERT_NE(input1, nullptr);
  ASSERT_NE(input1->GetOpDesc(), nullptr);
  auto in_desc = input1->GetOpDesc()->MutableInputDesc(0);
  auto out_desc = input1->GetOpDesc()->MutableOutputDesc(0);
  set_tensor_desc_nd_to_nz(in_desc);
  set_tensor_desc_nd_to_nz(out_desc);

  auto net_output = compute_graph->FindNode("netoutput");
  ASSERT_NE(net_output, nullptr);
  ASSERT_NE(net_output->GetOpDesc(), nullptr);
  auto net_output_in_desc = net_output->GetOpDesc()->MutableInputDesc(0);
  auto net_output_out_desc = net_output->GetOpDesc()->MutableOutputDesc(0);
  set_tensor_desc_nd_to_nz(net_output_in_desc);
  set_tensor_desc_nd_to_nz(net_output_out_desc);
  set_origin_shape_range(net_output_in_desc);
  set_origin_shape_range(net_output_out_desc);

  ge::GraphPrepare graph_prepare;
  graph_prepare.compute_graph_ = compute_graph;

  auto ret = graph_prepare.UpdateDataNetOutputByStorageFormat();
  EXPECT_NE(ret, SUCCESS);
}

TEST_F(UtestGraphPreproces, test_updar_variable_formats_insert_transdata) {
  auto builder = ut::GraphBuilder("g1");
  auto var = builder.AddNode("var_transdata", VARIABLE, 1, 1);
  auto g1 = builder.GetGraph();
  g1->SetSessionID(1); // diffenent from pre test case. cause varmanager is instance
  GeTensorDesc input_desc(GeShape({1, 2, 3, 4}), static_cast<ge::Format>(GetFormatFromSub(ge::FORMAT_FRACTAL_Z, 240)), DT_INT32);
  GeTensorDesc output_desc(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, DT_INT32);
  TransNodeInfo trans_node_info = {.node_type = TRANSDATA, .input = input_desc, .output = output_desc};
  VarTransRoad fusion_road;
  fusion_road.emplace_back(trans_node_info);
  VarManager::Instance(g1->GetSessionID())->Init(0, 1, 0, 1);
  VarManager::Instance(g1->GetSessionID())->SetTransRoad(var->GetName(), fusion_road);
  GraphPrepare graph_prepare;
  EXPECT_EQ(graph_prepare.UpdateVariableFormats(g1), SUCCESS);

  // check new transdata dst_format
  for (const auto &node : g1->GetDirectNode()) {
    if (node->GetType() == TRANSDATA) {
      string dst_format;
      AttrUtils::GetStr(node->GetOpDesc(), FORMAT_TRANSFER_DST_FORMAT, dst_format);
      int sub_format = 0;
      AttrUtils::GetInt(node->GetOpDesc(), "groups", sub_format);
      EXPECT_EQ(dst_format, "FRACTAL_Z");
      EXPECT_EQ(sub_format, 240);
    }
  }
}

TEST_F(UtestGraphPreproces, test_UpdateDataInputOutputDesc_skip_refresh) {
  auto builder = ut::GraphBuilder("g1");
  auto data = builder.AddNode("data", DATA, 1, 1, ge::FORMAT_NCHW);
  AttrUtils::SetBool(data->GetOpDesc(), "_skip_refresh_data_format", true);
  auto g1 = builder.GetGraph();
  GeTensorDesc tensor_desc(GeShape({1, 2, 3, 4}), ge::FORMAT_NHWC, DT_INT32);

  // pre check
  bool skip_refresh_data = false;
  AttrUtils::GetBool(data->GetOpDesc(), "_skip_refresh_data_format", skip_refresh_data);
  ASSERT_TRUE(skip_refresh_data);
  // test interface
  GraphPrepare graph_prepare;
  auto op_desc = data->GetOpDesc();
  EXPECT_EQ(graph_prepare.UpdateDataInputOutputDesc(0, op_desc, tensor_desc), SUCCESS);

  // after check
  skip_refresh_data = false;
  ASSERT_FALSE(AttrUtils::HasAttr(data->GetOpDesc(), "_skip_refresh_data_format"));
  ASSERT_EQ(data->GetOpDesc()->GetOutputDescPtr(0)->GetFormat(), ge::FORMAT_NCHW);

  // test normal
  EXPECT_EQ(graph_prepare.UpdateDataInputOutputDesc(0, op_desc, tensor_desc), SUCCESS);
  ASSERT_EQ(data->GetOpDesc()->GetOutputDescPtr(0)->GetFormat(), ge::FORMAT_NHWC);

  // test update_op_desc failed, data without input_desc
  auto builder2 = ut::GraphBuilder("g2");
  auto data2 = builder2.AddNode("data", DATA, 0, 1, ge::FORMAT_NCHW);
  auto g2 = builder2.GetGraph();
  auto op_desc2 = data2->GetOpDesc();
  EXPECT_EQ(graph_prepare.UpdateDataInputOutputDesc(0, op_desc2, tensor_desc), GRAPH_FAILED);
}

TEST_F(UtestGraphPreproces, UpdateUninitializedOriginShape) {
  ge::GraphPrepare graph_prepare;

  ComputeGraphPtr graph = make_shared<ComputeGraph>("default");

  OpDescPtr op_desc = CreateOpDesc("FileConstant", FILECONSTANT);
  std::vector<int64_t> shape = {2,2,2,2};

  EXPECT_TRUE(AttrUtils::SetDataType(op_desc, "dtype", DT_FLOAT));
  EXPECT_TRUE(AttrUtils::SetStr(op_desc, ATTR_NAME_FILE_CONSTANT_ID, "file"));
  EXPECT_TRUE(AttrUtils::SetListInt(op_desc, "shape", shape));
  GeTensorDesc tensor_desc(GeShape(shape), FORMAT_ND, DT_FLOAT);
  TensorUtils::SetSize(tensor_desc, 128);
  op_desc->AddOutputDesc(tensor_desc);
  op_desc->SetOutputOffset({0});
  graph->AddNode(op_desc);

  graph_prepare.compute_graph_ = graph;

  for (const NodePtr &node : graph_prepare.compute_graph_->GetAllNodes()) {
    Status ret = graph_prepare.UpdateUninitializedOriginShape(node);
    EXPECT_EQ(ret, SUCCESS);
  }
}

TEST_F(UtestGraphPreproces, SwitchOpOptimize) {
  ge::GraphPrepare graph_prepare;

  ComputeGraphPtr graph = make_shared<ComputeGraph>("default");

  OpDescPtr op_desc = CreateOpDesc("FileConstant", FILECONSTANT);
  std::vector<int64_t> shape = {2,2,2,2};

  EXPECT_TRUE(AttrUtils::SetDataType(op_desc, "dtype", DT_FLOAT));
  EXPECT_TRUE(AttrUtils::SetStr(op_desc, ATTR_NAME_FILE_CONSTANT_ID, "file"));
  EXPECT_TRUE(AttrUtils::SetListInt(op_desc, "shape", shape));
  GeTensorDesc tensor_desc(GeShape(shape), FORMAT_ND, DT_FLOAT);
  TensorUtils::SetSize(tensor_desc, 128);
  op_desc->AddOutputDesc(tensor_desc);
  op_desc->SetOutputOffset({0});
  graph->AddNode(op_desc);
  graph_prepare.compute_graph_ = graph;

  Status ret = graph_prepare.SwitchOpOptimize(graph);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphPreproces, GenerateInfershapeGraph) {
  ge::GraphPrepare graph_prepare;

  ComputeGraphPtr graph = make_shared<ComputeGraph>("default");

  OpDescPtr op_desc = CreateOpDesc("FileConstant", FILECONSTANT);
  std::vector<int64_t> shape = {2,2,2,2};

  EXPECT_TRUE(AttrUtils::SetDataType(op_desc, "dtype", DT_FLOAT));
  EXPECT_TRUE(AttrUtils::SetStr(op_desc, ATTR_NAME_FILE_CONSTANT_ID, "file"));
  EXPECT_TRUE(AttrUtils::SetListInt(op_desc, "shape", shape));
  GeTensorDesc tensor_desc(GeShape(shape), FORMAT_ND, DT_FLOAT);
  TensorUtils::SetSize(tensor_desc, 128);
  op_desc->AddOutputDesc(tensor_desc);
  op_desc->SetOutputOffset({0});
  op_desc->AddInferFunc([](Operator &op) {return GRAPH_SUCCESS;});
  graph->AddNode(op_desc);

  graph_prepare.compute_graph_ = graph;

  const ge::Graph graph_node = ge::GraphUtilsEx::CreateGraphFromComputeGraph(graph);
  ConstGraphPtr const_graph = MakeShared<const ge::Graph>(graph_node);
  Status ret = graph_prepare.GenerateInfershapeGraph(const_graph);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphPreproces, NeedUpdateFormatByOutputTypeParm) {
  ge::GraphPrepare graph_prepare;
  graph_prepare.compute_graph_ = BuildGraph4DynShape();
  Status ret = graph_prepare.UpdateInputOutputByOptions();
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphPreproces, InitDummyShapeOnControlFlow) {
  ge::GraphPrepare graph_prepare;
  auto compute_graph = BuildGraph8();
  auto ret = graph_prepare.InferShapeForPreprocess(compute_graph, nullptr, nullptr);
  EXPECT_EQ(ret, FAILED);
  compute_graph = BuildGraph9();
  ret = graph_prepare.InferShapeForPreprocess(compute_graph, nullptr, nullptr);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphPreproces, CheckConstOpFramwordOpCheckInvalid) {
  ge::GraphPrepare graph_prepare;

  ComputeGraphPtr graph = make_shared<ComputeGraph>("default");

  OpDescPtr op_desc = CreateOpDesc("framework_op", FRAMEWORKOP);
  std::vector<int64_t> shape = {2,2,2,2};

  EXPECT_TRUE(AttrUtils::SetStr(op_desc, ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, CONSTANT));
  GeTensorDesc tensor_desc(GeShape(shape), FORMAT_ND, DT_FLOAT);
  TensorUtils::SetSize(tensor_desc, 128);
  op_desc->AddOutputDesc(tensor_desc);
  op_desc->SetOutputOffset({0});
  auto node_ptr = graph->AddNode(op_desc);

  graph_prepare.compute_graph_ = graph;
  const auto ret = graph_prepare.CheckConstOp();
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(UtestGraphPreproces, IsDynamicDimsNoPositive) {
  ge::GraphPrepare graph_prepare;

  ComputeGraphPtr graph = make_shared<ComputeGraph>("default");

  OpDescPtr op_desc = CreateOpDesc("framework_op", FRAMEWORKOP);
  std::vector<int64_t> shape = {-1,2,2,2};

  EXPECT_TRUE(AttrUtils::SetStr(op_desc, ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, CONSTANT));
  GeTensorDesc tensor_desc(GeShape(shape), FORMAT_ND, DT_FLOAT);
  TensorUtils::SetSize(tensor_desc, 128);
  op_desc->AddOutputDesc(tensor_desc);
  op_desc->SetOutputOffset({0});
  auto node_ptr = graph->AddNode(op_desc);
  graph_prepare.options_.input_shape = "2,2,2,2";
  graph_prepare.options_.dynamic_dims = "-1,2,2,2";
  graph_prepare.options_.dynamic_node_type = 1;
  const auto ret = graph_prepare.IsDynamicDims(node_ptr);
  EXPECT_EQ(ret, true);
}

// test autofuse dynamic shape scene
TEST_F(UtestGraphPreproces, IsDynamicDimsNoPositiveAndAutofuse) {
  setenv("AUTOFUSE_FLAGS", "--enable_autofuse=true", 1);
  ge::GraphPrepare graph_prepare;

  ComputeGraphPtr graph = make_shared<ComputeGraph>("default");
  OpDescPtr op_desc = CreateOpDesc("framework_op", FRAMEWORKOP);
  std::vector<int64_t> shape = {-1,2,2,2};

  EXPECT_TRUE(AttrUtils::SetStr(op_desc, ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, CONSTANT));
  GeTensorDesc tensor_desc(GeShape(shape), FORMAT_ND, DT_FLOAT);
  TensorUtils::SetSize(tensor_desc, 128);
  op_desc->AddOutputDesc(tensor_desc);
  op_desc->SetOutputOffset({0});
  auto node_ptr = graph->AddNode(op_desc);
  graph_prepare.options_.input_shape = "2,2,2,2";
  std::map<std::string, std::string> options;
  options.insert(std::pair<std::string, std::string>("ge.jit_compile", "1"));
  ge::GetThreadLocalContext().SetGlobalOption(options);
  const auto ret = graph_prepare.IsDynamicDims(node_ptr);
  EXPECT_EQ(ret, true);
  unsetenv("AUTOFUSE_FLAGS");
}

TEST_F(UtestGraphPreproces, CheckUserInputInitFailed) {
  ge::GraphPrepare graph_prepare;

  ComputeGraphPtr graph = make_shared<ComputeGraph>("default");

  OpDescPtr op_desc = CreateOpDesc("data1", DATA);
  std::vector<int64_t> shape = {2,2,2,2};

  EXPECT_TRUE(AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, -1));
  GeTensorDesc tensor_desc(GeShape(shape), FORMAT_ND, DT_FLOAT);
  TensorUtils::SetSize(tensor_desc, 128);
  op_desc->AddOutputDesc(tensor_desc);
  op_desc->SetOutputOffset({0});
  auto node_ptr = graph->AddNode(op_desc);

  graph_prepare.compute_graph_ = graph;
  const std::vector<GeTensor> user_input;
  const auto ret = graph_prepare.CheckUserInput(user_input);
  EXPECT_EQ(ret, GE_GRAPH_INIT_FAILED);
}

TEST_F(UtestGraphPreproces, CheckUserInputInitFaileRefDataAndIoAllocMode) {
  ge::GraphPrepare graph_prepare;

  ComputeGraphPtr graph = make_shared<ComputeGraph>("default");

  OpDescPtr op_desc = CreateOpDesc("data1", REFDATA);
  std::vector<int64_t> shape = {2,2,2,2};

  EXPECT_TRUE(AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, 0));
  GeTensorDesc tensor_desc(GeShape(shape), FORMAT_ND, DT_FLOAT);
  TensorUtils::SetSize(tensor_desc, 128);
  op_desc->AddOutputDesc(tensor_desc);
  op_desc->SetOutputOffset({0});
  auto node_ptr = graph->AddNode(op_desc);
  graph_prepare.compute_graph_ = graph;

  std::map<std::string, std::string> options;
  options.insert(std::pair<std::string, std::string>("ge.exec.graphIOMemAllocMode", "ByGE"));
  ge::GetThreadLocalContext().SetGlobalOption(options);

  vector<int64_t> dim = {2, -3};
  GeTensor tensor;
  tensor.SetTensorDesc(GeTensorDesc(GeShape(dim)));
  std::vector<GeTensor> user_input;
  user_input.emplace_back(tensor);

  const auto ret = graph_prepare.CheckUserInput(user_input);
  EXPECT_EQ(ret, GE_GRAPH_INIT_FAILED);
}

TEST_F(UtestGraphPreproces, TypeConversionOfConstant) {
  ge::GraphPrepare graph_prepare;

  ComputeGraphPtr graph = make_shared<ComputeGraph>("default");

  OpDescPtr op_desc = CreateOpDesc("const1", CONSTANT);
  std::vector<int64_t> shape = {2,2,2,2};

  EXPECT_TRUE(AttrUtils::SetBool(op_desc, ATTR_SINGLE_OP_SCENE, false));
  GeTensorDesc tensor_desc(GeShape(shape), FORMAT_ND, DT_FLOAT);
  TensorUtils::SetSize(tensor_desc, 128);
  op_desc->AddOutputDesc(tensor_desc);
  op_desc->SetOutputOffset({0});
  auto node_ptr = graph->AddNode(op_desc);

  graph_prepare.compute_graph_ = graph;
  graph_prepare.options_.train_graph_flag = true;
  graph_prepare.TypeConversionOfConstant();
  EXPECT_EQ(node_ptr->GetOpDesc()->GetType(), CONSTANTOP);
  graph_prepare.options_.train_graph_flag = false;
  graph_prepare.TypeConversionOfConstant();
  EXPECT_EQ(node_ptr->GetOpDesc()->GetType(), CONSTANT);
}


TEST_F(UtestGraphPreproces, test_updar_variable_formats_insert_reshape) {
  auto builder = ut::GraphBuilder("g1");
  auto var = builder.AddNode("var_reshape", VARIABLE, 1, 1);
  (void)AttrUtils::SetStr(var->GetOpDesc(), REF_VAR_SRC_VAR_NAME, "var1");
  auto g1 = builder.GetGraph();
  g1->SetSessionID(1); // diffenent from pre test case. cause varmanager is instance
  GeTensorDesc input_desc(GeShape({1, 2, 3, 4}), static_cast<ge::Format>(GetFormatFromSub(ge::FORMAT_FRACTAL_Z, 240)), DT_INT32);
  GeTensorDesc output_desc(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, DT_INT32);
  TransNodeInfo trans_node_info = {.node_type = RESHAPE, .input = input_desc, .output = output_desc};
  VarTransRoad fusion_road;
  fusion_road.emplace_back(trans_node_info);
  VarManager::Instance(g1->GetSessionID())->Init(0, 1, 0, 1);
  VarManager::Instance(g1->GetSessionID())->SetTransRoad(var->GetName(), fusion_road);
  GraphPrepare graph_prepare;
  EXPECT_EQ(graph_prepare.UpdateVariableFormats(g1), SUCCESS);

  // check new reshape dst_format
  for (const auto &node : g1->GetDirectNode()) {
    if (node->GetType() == RESHAPE) {
      string dst_format;
      AttrUtils::GetStr(node->GetOpDesc(), FORMAT_TRANSFER_DST_FORMAT, dst_format);
      int sub_format = 0;
      AttrUtils::GetInt(node->GetOpDesc(), "groups", sub_format);
      EXPECT_EQ(dst_format, "");
      EXPECT_TRUE(sub_format == 0);
    }
  }
  std::vector<int> empty_shape;
  GeTensorDesc output_desc_empty(GeShape(), ge::FORMAT_NCHW, DT_INT32);
  trans_node_info.output = output_desc_empty;
  EXPECT_EQ(graph_prepare.UpdateVariableFormats(g1), SUCCESS);
}

TEST_F(UtestGraphPreproces, TryDoAipp) {
  auto builder = ut::GraphBuilder("g1");
  auto var = builder.AddNode("var", VARIABLE, 1, 1);
  auto g1 = builder.GetGraph();
  g1->SetSessionID(1); // diffenent from pre test case. cause varmanager is instance
  GeTensorDesc input_desc(GeShape({1, 2, 3, 4}), static_cast<ge::Format>(GetFormatFromSub(ge::FORMAT_FRACTAL_Z, 240)), DT_INT32);
  GeTensorDesc output_desc(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, DT_INT32);
  TransNodeInfo trans_node_info = {.node_type = RESHAPE, .input = input_desc, .output = output_desc};
  VarTransRoad fusion_road;
  fusion_road.emplace_back(trans_node_info);
  VarManager::Instance(g1->GetSessionID())->Init(0, 1, 0, 1);
  VarManager::Instance(g1->GetSessionID())->SetTransRoad(var->GetName(), fusion_road);
  GraphPrepare graph_prepare;
  graph_prepare.options_.train_graph_flag = false;
  graph_prepare.options_.insert_op_file = "op_file";
  graph_prepare.compute_graph_ = g1;
  EXPECT_EQ(graph_prepare.TryDoAipp(), GE_GRAPH_OPTIMIZE_INSERT_OP_PARSE_FAILED);
  graph_prepare.options_.insert_op_file = "\0";
  EXPECT_EQ(graph_prepare.TryDoAipp(), SUCCESS);
}

TEST_F(UtestGraphPreproces, ProcessInputNC1HWC0DynShape) {
  ge::GraphPrepare graph_prepare;
  graph_prepare.compute_graph_ = BuildGraph3Data();

  Status ret = graph_prepare.UpdateInputOutputByOptions();
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphPreproces, ProcessInputNC1HWC0DynShapeDynBatchFailed) {
  const char *const kMbatchSwitchnName = "mbatch-switch-name";
  ge::GraphPrepare graph_prepare;
  graph_prepare.compute_graph_ = BuildGraph3Data();
  auto data1 = graph_prepare.compute_graph_->FindNode("data1");
  AttrUtils::SetStr(data1->GetOpDesc(), kMbatchSwitchnName, "");
  Status ret = graph_prepare.UpdateInputOutputByOptions();
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestGraphPreproces, ProcessInputNC1HWC0DynShapeDynBatch) {
  const char *const kMbatchSwitchnName = "mbatch-switch-name";
  ge::GraphPrepare graph_prepare;
  graph_prepare.compute_graph_ = BuildGraph3Data();
  auto data1 = graph_prepare.compute_graph_->FindNode("data1");
  AttrUtils::SetStr(data1->GetOpDesc(), kMbatchSwitchnName, "netoutput");
  Status ret = graph_prepare.UpdateInputOutputByOptions();
  EXPECT_EQ(ret, SUCCESS);
}


TEST_F(UtestGraphPreproces, test_update_dtype_mbatch_NeedUpdateDtByOutputTypeParm) {
  ge::GraphPrepare graph_prepare;
  graph_prepare.compute_graph_ = BuildGraph4Data();
  auto parent_graph = graph_prepare.compute_graph_;
  auto subgraph1 = BuildGraph4_Subgraph("subgraph1");
  auto subgraph2 = BuildGraph4_Subgraph("subgraph2");

  auto data1 = parent_graph->FindNode("data1");
  auto data_desc = data1->GetOpDesc();

  auto merge_node = parent_graph->FindNode("merge1");
  EXPECT_NE(merge_node, nullptr);
  merge_node->GetOpDesc()->AddSubgraphName("subgraph1");
  merge_node->GetOpDesc()->SetSubgraphInstanceName(0, "subgraph1");
  subgraph1->SetParentNode(merge_node);
  subgraph1->SetParentGraph(parent_graph);
  EXPECT_EQ(parent_graph->AddSubgraph("subgraph1", subgraph1), GRAPH_SUCCESS);

  merge_node->GetOpDesc()->AddSubgraphName("subgraph2");
  merge_node->GetOpDesc()->SetSubgraphInstanceName(1, "subgraph2");
  subgraph2->SetParentNode(merge_node);
  subgraph2->SetParentGraph(parent_graph);
  EXPECT_EQ(parent_graph->AddSubgraph("subgraph2", subgraph2), GRAPH_SUCCESS);

  Status ret = graph_prepare.UpdateInputOutputByOptions();
  EXPECT_EQ(ret, SUCCESS);
  // need check results
}

TEST_F(UtestGraphPreproces, SaveOriginalGraphToOmModel) {
  ge::GraphPrepare graph_prepare;
  graph_prepare.compute_graph_ = nullptr;
  graph_prepare.options_.save_original_model = "true";
  Status ret = graph_prepare.SaveOriginalGraphToOmModel();
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphPreproces, ProcessAIPPNodesDataFormatTest) {
  ge::GraphPrepare graph_prepare;
  graph_prepare.options_.input_format = "NHWC";
  auto builder = ut::GraphBuilder("graph1");
  auto data = builder.AddNode("data", DATA, 0, 1, ge::FORMAT_NCHW);
  auto aipp_op = builder.AddNode("aipp", AIPP, 1, 1, ge::FORMAT_NCHW);
  auto conv = builder.AddNode("conv", CONV2D, 1, 1, ge::FORMAT_NCHW);
  AttrUtils::SetInt(data->GetOpDesc(), ATTR_NAME_INDEX, 0);

  builder.AddDataEdge(data, 0, aipp_op, 0);
  builder.AddDataEdge(aipp_op, 0, conv, 0);
  ComputeGraphPtr graph = builder.GetGraph();
  graph_prepare.compute_graph_ = graph;
  Status ret = graph_prepare.ProcessAippNodesDataFormat();
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphPreproces, test_IsDynamicDims) {
  ge::GraphPrepare graph_prepare;
  graph_prepare.compute_graph_ = BuildGraph6();
  NodePtr node = graph_prepare.compute_graph_->FindNode("input1");

  std::map<std::string, std::string> options;
  options.insert(std::pair<std::string, std::string>("ge.jit_compile", "0"));
  ge::GetThreadLocalContext().SetGlobalOption(options);
  EXPECT_EQ(graph_prepare.IsDynamicDims(node), true);
}
TEST_F(UtestGraphPreproces, test_internal_format) {
  ge::GraphPrepare graph_prepare;
  auto builder = ut::GraphBuilder("graph1");
  auto data = builder.AddNode("data", DATA, 0, 1, ge::FORMAT_NCHW);
  auto tensor_desc = data->GetOpDesc()->MutableOutputDesc(0U);
  tensor_desc->SetOriginFormat(ge::FORMAT_FRACTAL_Z);
  EXPECT_NE(graph_prepare.CheckInternalFormat(data, *tensor_desc), SUCCESS);
}

// test storage format start
namespace {
void SetStorageFormatAndShape(GeTensorDescPtr &tensor_desc, Format storage_format,
                              const std::vector<int64_t> &storage_shape, const std::string &expand_dims_type = "") {
  AttrUtils::SetInt(tensor_desc, ATTR_NAME_STORAGE_FORMAT, static_cast<int>(storage_format));
  AttrUtils::SetListInt(tensor_desc, ATTR_NAME_STORAGE_SHAPE, storage_shape);
  tensor_desc->SetExpandDimsRule(expand_dims_type);
}
void CheckStorageFormatAndShapeOnAttr(ConstGeTensorDescPtr &tensor_desc, Format expect_storage_format,
                                      const std::vector<int64_t> &expect_storage_shape,
                                      const std::string &expand_dims_type = "") {
  int storage_format = static_cast<int>(FORMAT_RESERVED);
  AttrUtils::GetInt(tensor_desc, ATTR_NAME_STORAGE_FORMAT, storage_format);
  EXPECT_EQ(storage_format, expect_storage_format);

  std::vector<int64_t> storage_shape;
  AttrUtils::GetListInt(tensor_desc, ATTR_NAME_STORAGE_SHAPE, storage_shape);
  EXPECT_STREQ(ToString(storage_shape).c_str(), ToString(expect_storage_shape).c_str());

  if (!expand_dims_type.empty()) {
    EXPECT_STREQ( tensor_desc->GetExpandDimsRule().c_str(), expand_dims_type.c_str());
  }
}

void CheckStorageFormatAndShape(ConstGeTensorDescPtr &tensor_desc, Format expect_storage_format,
                                const std::vector<int64_t> &expect_storage_shape,
                                const std::string &expand_dims_type = "") {
  int storage_format = static_cast<int>(FORMAT_RESERVED);
  AttrUtils::GetInt(tensor_desc, ATTR_NAME_STORAGE_FORMAT, storage_format);
  EXPECT_EQ(storage_format, expect_storage_format);
  EXPECT_EQ(tensor_desc->GetFormat(), expect_storage_format);

  EXPECT_EQ(tensor_desc->GetShape().GetDims(), expect_storage_shape);

  if (!expand_dims_type.empty()) {
    std::string target_expand_dims_type;
    AttrUtils::GetStr(tensor_desc, ATTR_NAME_RESHAPE_INFER_TYPE, target_expand_dims_type);
    EXPECT_STREQ(target_expand_dims_type.c_str(), expand_dims_type.c_str());
  }
}
} // namespace

/**
 *   refdata   data
 *      \      /
 *       conv2d
 */
TEST_F(UtestGraphPreproces, NormalizeGraph_output_storage_format) {
  // build graph
  auto builder = ut::GraphBuilder("graph1");
  auto ref_data = builder.AddNode("refdata", REFDATA, 1, 1, ge::FORMAT_NCHW);
  auto data = builder.AddNode("data", DATA, 1, 1, ge::FORMAT_NCHW);
  auto conv2d = builder.AddNode("conv2d", CONV2D, 2, 1, ge::FORMAT_NCHW);
  auto output_tensor_desc = conv2d->GetOpDesc()->MutableOutputDesc(0);
  SetStorageFormatAndShape(output_tensor_desc, FORMAT_FRACTAL_Z, {}, "1100");
  builder.AddDataEdge(ref_data, 0, conv2d, 0);
  builder.AddDataEdge(data, 0, conv2d, 1);
  auto compute_graph = builder.GetGraph();
  // set conv2d 0th output as graph outputs
  compute_graph->SetUserDefOutput("conv2d:0");
  compute_graph->SetOutputSize(1);
  std::vector<std::pair<NodePtr, int32_t>> out_nodes_info = {{conv2d, 0}};
  compute_graph->SetGraphOutNodesInfo(out_nodes_info);

  // test normalize graph
  map<string, string> options;
  Status ret = ge::GELib::Initialize(options);
  EXPECT_EQ(ret, SUCCESS);
  ge::GraphPrepare graph_prepare;
  EXPECT_EQ(graph_prepare.NormalizeGraph(compute_graph, {}, {}), GRAPH_SUCCESS);

  // check has netoutput on graph, and netoutput has storage format attrs
  auto netoutput = compute_graph->FindFirstNodeMatchType(NETOUTPUT);
  EXPECT_NE(netoutput, nullptr);
  auto netoutput_in_tensordesc = netoutput->GetOpDesc()->GetInputDescPtr(0);
  CheckStorageFormatAndShapeOnAttr(netoutput_in_tensordesc, FORMAT_FRACTAL_Z, {}, "1100");
}

/**
 *   refdata   data
 *      \      /
 *       conv2d
 *         |
 *      Netoutput
 */
TEST_F(UtestGraphPreproces, NormalizeGraph_withNetoutput_storage_format) {
  // build graph
  auto builder = ut::GraphBuilder("graph1");
  auto ref_data = builder.AddNode("refdata", REFDATA, 1, 1, ge::FORMAT_NCHW);
  auto data = builder.AddNode("data", DATA, 1, 1, ge::FORMAT_NCHW);
  auto conv2d = builder.AddNode("conv2d", CONV2D, 2, 1, ge::FORMAT_NCHW);
  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 1, ge::FORMAT_NCHW);
  auto output_tensor_desc = netoutput->GetOpDesc()->MutableInputDesc(0);
  SetStorageFormatAndShape(output_tensor_desc, FORMAT_FRACTAL_Z, {5, 6, 7, 8, 9}, "1100");
  netoutput->GetOpDesc()->UpdateOutputDesc(0, *output_tensor_desc);
  builder.AddDataEdge(ref_data, 0, conv2d, 0);
  builder.AddDataEdge(data, 0, conv2d, 1);
  builder.AddDataEdge(conv2d, 0, netoutput, 0);
  auto compute_graph = builder.GetGraph();
  // set conv2d 0th output as graph outputs
  compute_graph->SetUserDefOutput("netoutput:0");
  compute_graph->SetOutputSize(1);
  std::vector<std::pair<NodePtr, int32_t>> out_nodes_info = {{netoutput, 0}};
  compute_graph->SetGraphOutNodesInfo(out_nodes_info);

  // test normalize graph
  map<string, string> options;
  Status ret = ge::GELib::Initialize(options);
  EXPECT_EQ(ret, SUCCESS);
  ge::GraphPrepare graph_prepare;
  EXPECT_EQ(graph_prepare.NormalizeGraph(compute_graph, {}, {}), GRAPH_SUCCESS);

  // check has netoutput on graph, and netoutput has storage format attrs
  auto netoutput_node = compute_graph->FindFirstNodeMatchType(NETOUTPUT);
  EXPECT_NE(netoutput_node, nullptr);
  auto netoutput_in_tensordesc = netoutput_node->GetOpDesc()->GetInputDescPtr(0);
  CheckStorageFormatAndShapeOnAttr(netoutput_in_tensordesc, FORMAT_FRACTAL_Z, {5, 6, 7, 8, 9}, "1100");
}

/**
 *     data
 *      |
 *    netoutput
 */
TEST_F(UtestGraphPreproces, PrepareRunningFormatRefiner_input_output_storage_format_use_defined_stroage_shape) {
  // build test graph
  auto builder = ut::GraphBuilder("graph1");
  auto data = builder.AddNode("data", DATA, 1, 1, ge::FORMAT_NCHW);
  auto tensor_desc = data->GetOpDesc()->MutableInputDesc(0U);
  SetStorageFormatAndShape(tensor_desc, ge::FORMAT_FRACTAL_Z, {5,6,7,8,9});

  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 1, ge::FORMAT_NCHW);
  auto out_tensor_desc = netoutput->GetOpDesc()->MutableInputDesc(0U);
  SetStorageFormatAndShape(out_tensor_desc, ge::FORMAT_FRACTAL_Z, {5,6,7,8,9});
  builder.AddDataEdge(data, 0, netoutput, 0);

  // test PrepareRunningFormatRefiner
  ge::GraphPrepare graph_prepare;
  graph_prepare.compute_graph_ = builder.GetGraph();
  EXPECT_EQ(graph_prepare.PrepareRunningFormatRefiner(), GRAPH_SUCCESS);

  // check heavy op attr
  auto target_data = graph_prepare.compute_graph_->FindNode("data");
  bool is_data_heavy_op = false;
  AttrUtils::GetBool(target_data->GetOpDesc(), ATTR_NAME_IS_HEAVY_OP, is_data_heavy_op);
  EXPECT_FALSE(is_data_heavy_op);

  auto target_output = graph_prepare.compute_graph_->FindNode("netoutput");
  bool is_out_heavy_op = false;
  AttrUtils::GetBool(target_output->GetOpDesc(), ATTR_NAME_IS_HEAVY_OP, is_out_heavy_op);
  EXPECT_FALSE(is_out_heavy_op);
  // check storage shape
  auto target_tensor_desc = target_data->GetOpDesc()->GetInputDescPtr(0);
  CheckStorageFormatAndShape(target_tensor_desc, FORMAT_FRACTAL_Z, {5,6,7,8,9});
}

/**
 *     data
 *      |
 *    netoutput
 *
 *    origin format: NCHW
 *    origin shape: [3,8,8]
 *    storage format:FORMAT_FRACTAL_Z
 *    storage shape:[64,1,16,16]
 */
TEST_F(UtestGraphPreproces, PrepareRunningFormatRefiner_input_output_storage_format_with_expand_dims_rule) {
  // build test graph
  auto builder = ut::GraphBuilder("graph1");
  auto data = builder.AddNode("data", DATA, 1, 1, ge::FORMAT_NCHW, DT_FLOAT16, {3,8,8});
  auto tensor_desc = data->GetOpDesc()->MutableInputDesc(0U);
  SetStorageFormatAndShape(tensor_desc, ge::FORMAT_FRACTAL_Z, {}, "1000");

  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 1, ge::FORMAT_NCHW, DT_FLOAT16, {3,8,8});
  auto out_tensor_desc = netoutput->GetOpDesc()->MutableInputDesc(0U);
  SetStorageFormatAndShape(out_tensor_desc, ge::FORMAT_FRACTAL_Z, {}, "1000");
  builder.AddDataEdge(data, 0, netoutput, 0);

  // test PrepareRunningFormatRefiner
  ge::GraphPrepare graph_prepare;
  graph_prepare.compute_graph_ = builder.GetGraph();
  EXPECT_EQ(graph_prepare.PrepareRunningFormatRefiner(), GRAPH_SUCCESS);

  // check heavy op attr
  auto target_data = graph_prepare.compute_graph_->FindNode("data");
  bool is_data_heavy_op = false;
  AttrUtils::GetBool(target_data->GetOpDesc(), ATTR_NAME_IS_HEAVY_OP, is_data_heavy_op);
  EXPECT_TRUE(is_data_heavy_op);
  // check storage shape / storage format on tensor
  auto input_desc = target_data->GetOpDesc()->GetInputDescPtr(0);
  auto output_desc = target_data->GetOpDesc()->GetOutputDescPtr(0);

  CheckStorageFormatAndShape(input_desc, FORMAT_FRACTAL_Z, {64,1,16,16}, "CHW");

  auto target_output = graph_prepare.compute_graph_->FindNode("netoutput");
  bool is_out_heavy_op = false;
  AttrUtils::GetBool(target_data->GetOpDesc(), ATTR_NAME_IS_HEAVY_OP, is_out_heavy_op);
  EXPECT_TRUE(is_out_heavy_op);
}

/**
 *     data
 *      |
 *    netoutput
 *
 *    origin format: NCHW
 *    origin shape: [3,8,8]
 *    storage format:FORMAT_FRACTAL_Z
 *    storage shape:[64,1,16,16]
 */
TEST_F(UtestGraphPreproces, PrepareRunningFormatRefiner_input_storage_format_without_spread) {
  // build test graph
  auto builder = ut::GraphBuilder("graph1");
  auto data = builder.AddNode("data", DATA, 1, 1, ge::FORMAT_NCHW, DT_FLOAT16, {3,8,8});
  AttrUtils::SetBool(data->GetOpDesc(), "_enable_storage_format_spread", false);
  auto tensor_desc = data->GetOpDesc()->MutableInputDesc(0U);
  SetStorageFormatAndShape(tensor_desc, ge::FORMAT_FRACTAL_Z, {}, "1000");

  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 1, ge::FORMAT_NCHW, DT_FLOAT16, {3,8,8});
  auto out_tensor_desc = netoutput->GetOpDesc()->MutableInputDesc(0U);
  SetStorageFormatAndShape(out_tensor_desc, ge::FORMAT_FRACTAL_Z, {}, "1000");
  builder.AddDataEdge(data, 0, netoutput, 0);

  // test PrepareRunningFormatRefiner
  ge::GraphPrepare graph_prepare;
  graph_prepare.compute_graph_ = builder.GetGraph();
  EXPECT_EQ(graph_prepare.PrepareRunningFormatRefiner(), GRAPH_SUCCESS);

  // check heavy op attr
  auto target_data = graph_prepare.compute_graph_->FindNode("data");
  bool is_data_heavy_op= true;
  EXPECT_TRUE(AttrUtils::GetBool(target_data->GetOpDesc(), ATTR_NAME_IS_HEAVY_OP, is_data_heavy_op));
  EXPECT_FALSE(is_data_heavy_op);
  // check storage shape / storage format on tensor
  auto input_desc = target_data->GetOpDesc()->GetInputDescPtr(0);
  auto output_desc = target_data->GetOpDesc()->GetOutputDescPtr(0);

  CheckStorageFormatAndShape(input_desc, FORMAT_FRACTAL_Z, {64,1,16,16}, "CHW");

  auto target_output = graph_prepare.compute_graph_->FindNode("netoutput");
  bool is_out_heavy_op = false;
  AttrUtils::GetBool(target_output->GetOpDesc(), ATTR_NAME_IS_HEAVY_OP, is_out_heavy_op);
  EXPECT_TRUE(is_out_heavy_op);
}
/*
 *  refdata   data
 *      \     /
 *      assign
 *        |
 *      netoutput
 */
TEST_F(UtestGraphPreproces, PrepareRunningFormatRefiner_refdata_storage_format) {
  // build graph
  auto builder = ut::GraphBuilder("graph1");
  auto ref_data = builder.AddNode("refdata", REFDATA, 1, 1, ge::FORMAT_NCHW, DT_FLOAT16, {3,8,8});
  auto tensor_desc = ref_data->GetOpDesc()->MutableInputDesc(0U);
  auto out_tensor_desc = ref_data->GetOpDesc()->MutableOutputDesc(0U);
  SetStorageFormatAndShape(tensor_desc, ge::FORMAT_FRACTAL_Z, {}, "1000");
  SetStorageFormatAndShape(out_tensor_desc, ge::FORMAT_FRACTAL_Z, {}, "1000");

  auto data = builder.AddNode("data", DATA, 1, 1, ge::FORMAT_NCHW);
  auto assign = builder.AddNode("assign", ASSIGN, 1, 1, ge::FORMAT_NCHW);
  std::map<std::string, uint32_t> input_names = {{"ref", 0}, {"value", 1}};
  std::map<std::string, uint32_t> output_names = {{"ref", 0}};
  assign->GetOpDesc()->UpdateInputName(input_names);
  assign->GetOpDesc()->UpdateOutputName(output_names);

  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 1, ge::FORMAT_NCHW);
  builder.AddDataEdge(ref_data, 0, assign, 0);
  builder.AddDataEdge(data, 0, assign, 1);
  builder.AddDataEdge(assign, 0, netoutput, 0);

  auto refdata_input_desc = ref_data->GetOpDesc()->GetInputDescPtr(0);
  auto refdata_output_desc = ref_data->GetOpDesc()->GetOutputDescPtr(0);
  CheckStorageFormatAndShapeOnAttr(refdata_input_desc, FORMAT_FRACTAL_Z, {});
  CheckStorageFormatAndShapeOnAttr(refdata_output_desc, FORMAT_FRACTAL_Z, {});

  // test PrepareRunningFormatRefiner
  ge::GraphPrepare graph_prepare;
  graph_prepare.compute_graph_ = builder.GetGraph();
  graph_prepare.PrepareRunningFormatRefiner();

  // check refdata is heavy op
  auto target_data = graph_prepare.compute_graph_->FindNode("refdata");
  auto target_data_input_desc = target_data->GetOpDesc()->GetInputDescPtr(0);
  auto target_data_output_desc = target_data->GetOpDesc()->GetOutputDescPtr(0);
  CheckStorageFormatAndShape(target_data_input_desc, FORMAT_FRACTAL_Z, {64,1,16,16}, "CHW");
  CheckStorageFormatAndShape(target_data_output_desc, FORMAT_FRACTAL_Z, {64,1,16,16}, "CHW");
  bool is_heavy_op = false;
  AttrUtils::GetBool(target_data->GetOpDesc(), ATTR_NAME_IS_HEAVY_OP, is_heavy_op);
  EXPECT_TRUE(is_heavy_op);

  // check refdata_ref is heavy op
  std::string ref_var_name = ref_data->GetName() + "_TO_" + assign->GetName() + "_REF_0";
  auto ref_data_ref = graph_prepare.compute_graph_->FindNode(ref_var_name);
  EXPECT_NE(ref_data_ref, nullptr);
  auto input_desc = ref_data_ref->GetOpDesc()->GetInputDescPtr(0);
  auto output_desc = ref_data_ref->GetOpDesc()->GetOutputDescPtr(0);
  CheckStorageFormatAndShape(input_desc, FORMAT_FRACTAL_Z, {64,1,16,16}, "CHW");
  CheckStorageFormatAndShape(output_desc, FORMAT_FRACTAL_Z, {64,1,16,16}, "CHW");

  bool is_ref_var_heavy_op =false;
  AttrUtils::GetBool(ref_data_ref->GetOpDesc(), ATTR_NAME_IS_HEAVY_OP, is_ref_var_heavy_op);
  EXPECT_FALSE(is_ref_var_heavy_op); // 回边不需要是重型
}
/*
 *  refdata   data
 *      \     /
 *      assign
 *        |
 *      netoutput
 */
TEST_F(UtestGraphPreproces, PrepareRunningFormatRefiner_IS_OP_GENERALIZED_refdata_storage_format_same_with_origin) {
  // build graph
  auto builder = ut::GraphBuilder("graph1");
  auto ref_data = builder.AddNode("refdata", REFDATA, 1, 1, ge::FORMAT_NCHW, DT_FLOAT16, {-1});
  auto tensor_desc = ref_data->GetOpDesc()->MutableInputDesc(0U);
  auto out_tensor_desc = ref_data->GetOpDesc()->MutableOutputDesc(0U);
  SetStorageFormatAndShape(tensor_desc, ge::FORMAT_NCHW, {64}, "");
  SetStorageFormatAndShape(out_tensor_desc, ge::FORMAT_NCHW, {64}, "");
  (void)AttrUtils::SetBool(ref_data->GetOpDesc(), ATTR_NAME_IS_OP_GENERALIZED, true);

  auto data = builder.AddNode("data", DATA, 1, 1, ge::FORMAT_NCHW);
  auto assign = builder.AddNode("assign", ASSIGN, 1, 1, ge::FORMAT_NCHW);
  std::map<std::string, uint32_t> input_names = {{"ref", 0}, {"value", 1}};
  std::map<std::string, uint32_t> output_names = {{"ref", 0}};
  assign->GetOpDesc()->UpdateInputName(input_names);
  assign->GetOpDesc()->UpdateOutputName(output_names);

  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 1, ge::FORMAT_NCHW);
  builder.AddDataEdge(ref_data, 0, assign, 0);
  builder.AddDataEdge(data, 0, assign, 1);
  builder.AddDataEdge(assign, 0, netoutput, 0);

  // test PrepareRunningFormatRefiner
  ge::GraphPrepare graph_prepare;
  graph_prepare.compute_graph_ = builder.GetGraph();
  graph_prepare.PrepareRunningFormatRefiner();

  // check refdata is not heavy op
  auto target_data = graph_prepare.compute_graph_->FindNode("refdata");
  bool is_heavy_op = false;
  AttrUtils::GetBool(target_data->GetOpDesc(), ATTR_NAME_IS_HEAVY_OP, is_heavy_op);
  EXPECT_FALSE(is_heavy_op);
  auto target_output_tensor_desc = target_data->GetOpDesc()->GetOutputDescPtr(0);
  CheckStorageFormatAndShape(target_output_tensor_desc, FORMAT_NCHW, {-1});
}

/*
 *  refdata   data
 *      \     /
 *      assign
 *        |
 *      netoutput
 */
TEST_F(UtestGraphPreproces, PrepareRunningFormatRefiner_OP_NOT_GENERALIZED_refdata_storage_format_same_with_origin) {
  // build graph
  auto builder = ut::GraphBuilder("graph1");
  auto ref_data = builder.AddNode("refdata", REFDATA, 1, 1, ge::FORMAT_NCHW, DT_FLOAT16, {64});
  auto tensor_desc = ref_data->GetOpDesc()->MutableInputDesc(0U);
  auto out_tensor_desc = ref_data->GetOpDesc()->MutableOutputDesc(0U);
  SetStorageFormatAndShape(tensor_desc, ge::FORMAT_NCHW, {}, "");
  SetStorageFormatAndShape(out_tensor_desc, ge::FORMAT_NCHW, {}, "");

  auto data = builder.AddNode("data", DATA, 1, 1, ge::FORMAT_NCHW);
  auto assign = builder.AddNode("assign", ASSIGN, 1, 1, ge::FORMAT_NCHW);
  std::map<std::string, uint32_t> input_names = {{"ref", 0}, {"value", 1}};
  std::map<std::string, uint32_t> output_names = {{"ref", 0}};
  assign->GetOpDesc()->UpdateInputName(input_names);
  assign->GetOpDesc()->UpdateOutputName(output_names);

  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 1, ge::FORMAT_NCHW);
  builder.AddDataEdge(ref_data, 0, assign, 0);
  builder.AddDataEdge(data, 0, assign, 1);
  builder.AddDataEdge(assign, 0, netoutput, 0);

  // test PrepareRunningFormatRefiner
  ge::GraphPrepare graph_prepare;
  graph_prepare.compute_graph_ = builder.GetGraph();
  graph_prepare.PrepareRunningFormatRefiner();

  // check refdata is not heavy op
  auto target_data = graph_prepare.compute_graph_->FindNode("refdata");
  bool is_heavy_op = false;
  AttrUtils::GetBool(target_data->GetOpDesc(), ATTR_NAME_IS_HEAVY_OP, is_heavy_op);
  EXPECT_TRUE(is_heavy_op);
  auto target_output_tensor_desc = target_data->GetOpDesc()->GetOutputDescPtr(0);
  CheckStorageFormatAndShape(target_output_tensor_desc, FORMAT_NCHW, {64});
}

/*
 *  refdata   data
 *      \     /
 *      assign
 *        |
 *      netoutput
 */
TEST_F(UtestGraphPreproces, PrepareRunningFormatRefiner_OP_NOT_GENERALIZED_unsupport_storage_format) {
  // build graph
  auto builder = ut::GraphBuilder("graph1");
  auto ref_data = builder.AddNode("refdata", REFDATA, 1, 1, ge::FORMAT_NCHW, DT_FLOAT16, {64,2,3,4});
  auto tensor_desc = ref_data->GetOpDesc()->MutableInputDesc(0U);
  auto out_tensor_desc = ref_data->GetOpDesc()->MutableOutputDesc(0U);
  SetStorageFormatAndShape(tensor_desc, ge::FORMAT_FRACTAL_ZN_RNN, {}, "");
  SetStorageFormatAndShape(out_tensor_desc, ge::FORMAT_FRACTAL_ZN_RNN, {}, "");

  auto data = builder.AddNode("data", DATA, 1, 1, ge::FORMAT_NCHW);
  auto assign = builder.AddNode("assign", ASSIGN, 1, 1, ge::FORMAT_NCHW);
  std::map<std::string, uint32_t> input_names = {{"ref", 0}, {"value", 1}};
  std::map<std::string, uint32_t> output_names = {{"ref", 0}};
  assign->GetOpDesc()->UpdateInputName(input_names);
  assign->GetOpDesc()->UpdateOutputName(output_names);

  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 1, ge::FORMAT_NCHW);
  builder.AddDataEdge(ref_data, 0, assign, 0);
  builder.AddDataEdge(data, 0, assign, 1);
  builder.AddDataEdge(assign, 0, netoutput, 0);

  // test PrepareRunningFormatRefiner
  ge::GraphPrepare graph_prepare;
  graph_prepare.compute_graph_ = builder.GetGraph();
  EXPECT_NE(graph_prepare.PrepareRunningFormatRefiner(), SUCCESS);
}

TEST_F(UtestGraphPreproces, test_update_constplaceholder_by_storage_format) {
  auto compute_graph = gert::ShareGraph::BuildSingleConstPlaceHolderGraph(nullptr, 100L);
  auto set_op_desc_nd_to_nz = [](OpDescPtr op_desc) {
    AttrUtils::SetInt(op_desc, ATTR_NAME_STORAGE_FORMAT, static_cast<int64_t>(FORMAT_FRACTAL_NZ));
    vector<int64_t> storage_shape = {1, 1, 1, 16, 16};
    AttrUtils::SetListInt(op_desc, ATTR_NAME_STORAGE_SHAPE, storage_shape);
  };

  const auto node = compute_graph->FindNode("constplaceholder1");
  ASSERT_NE(node, nullptr);
  const auto op_desc = node->GetOpDesc();
  ASSERT_NE(op_desc, nullptr);
  set_op_desc_nd_to_nz(op_desc);

  ge::GraphPrepare graph_prepare;
  graph_prepare.compute_graph_ = compute_graph;
  vector<int64_t> storage_shape = {1, 1, 1, 16, 16};
  const auto ret = graph_prepare.UpdateDataNetOutputByStorageFormat();
  EXPECT_EQ(ret, SUCCESS);
  const auto out_desc = node->GetOpDesc()->MutableOutputDesc(0);
  EXPECT_EQ(out_desc->GetFormat(), FORMAT_FRACTAL_NZ);
  EXPECT_EQ(out_desc->GetShape().GetDims(), storage_shape);
}

TEST_F(UtestGraphPreproces, test_verify_const_op) {
  const auto construct_constant_node = [](const std::vector<uint8_t> &data, const std::vector<int64_t> &shape,
                                          DataType data_type) -> NodePtr {
    GeTensorDesc weight_desc(GeShape(shape), FORMAT_ND, data_type);
    GeTensor weight;
    weight.SetData(data);
    weight.SetTensorDesc(weight_desc);
    auto constant = OP_CFG(CONSTANT).TensorDesc(FORMAT_ND, data_type, shape).Attr<GeTensor>(ATTR_NAME_WEIGHTS, weight);
    DEF_GRAPH(g1) {
      CHAIN(NODE("constant", constant)->EDGE(0, 0)->NODE("netoutput", NETOUTPUT));
    };
    auto graph = ToGeGraph(g1);
    auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
    return compute_graph->FindNode("constant");
  };

  GraphPrepare graph_prepare;
  // shape = [2], type = DT_INT32
  NodePtr node = construct_constant_node({1, 1, 1, 1, 1, 1, 1, 1}, {2}, DT_INT32);
  EXPECT_EQ(graph_prepare.VerifyConstOp(node), SUCCESS);

  // shape = [2], type = DT_INT4
  node = construct_constant_node({1}, {2}, DT_INT4);
  EXPECT_EQ(graph_prepare.VerifyConstOp(node), SUCCESS);

  // shape = [] (data_size != 0), type = DT_INT4
  node = construct_constant_node({1}, {}, DT_INT4);
  EXPECT_EQ(graph_prepare.VerifyConstOp(node), SUCCESS);

  // shape = [x, y, 0,...], type = DT_INT4
  node = construct_constant_node({}, {1, 0}, DT_INT4);
  EXPECT_EQ(graph_prepare.VerifyConstOp(node), SUCCESS);
}
// test storage format end
}

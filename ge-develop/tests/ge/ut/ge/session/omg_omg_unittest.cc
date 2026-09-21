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

#include "common/plugin/ge_make_unique_util.h"
#include "proto/ge_ir.pb.h"
#include "framework/omg/omg.h"
#include "ge/ge_api_error_codes.h"
#include "framework/common/types.h"
#include "graph/utils//graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/common/helper/model_helper.h"
#include "framework/omg/parser/parser_factory.h"
#include "framework/common/debug/ge_log.h"
#include "graph/passes/graph_builder_utils.h"
#include "framework/omg/parser/model_parser.h"
#include "framework/omg/parser/weights_parser.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "macro_utils/dt_public_unscope.h"

using namespace std;

namespace ge {
class UtestOmg : public testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

using GetGraphCallback = std::function<std::unique_ptr<google::protobuf::Message>(
  const google::protobuf::Message *root_proto, const std::string &graph)>;

class CaffeModelParser : public domi::ModelParser {
  public:
    CaffeModelParser(){}
    ~CaffeModelParser(){}
    domi::Status Parse(const char *file, ge::Graph &graph){return SUCCESS;}
    domi::Status ParseFromMemory(const char *data, uint32_t size, ge::ComputeGraphPtr &graph){return SUCCESS;}
    domi::Status ParseFromMemory(const char *data, uint32_t size, ge::Graph &graph){return SUCCESS;}
    domi::Status ParseProto(const google::protobuf::Message *proto, ge::ComputeGraphPtr &graph){return SUCCESS;}
    domi::Status ParseProtoWithSubgraph(const google::protobuf::Message *proto, GetGraphCallback callback,
                                              ge::ComputeGraphPtr &graph){return SUCCESS;}
    ge::DataType ConvertToGeDataType(const uint32_t type){return DT_FLOAT;}
    domi::Status ParseAllGraph(const google::protobuf::Message *root_proto,
                                              ge::ComputeGraphPtr &root_graph){return SUCCESS;}
};

class CaffeWeightsParser : public domi::WeightsParser {
  public:
    CaffeWeightsParser(){}
    ~CaffeWeightsParser(){}
    domi::Status Parse(const char *file, ge::Graph &graph){return SUCCESS;}
    domi::Status ParseFromMemory(const char *input, uint32_t lengt, ge::ComputeGraphPtr &graph){return SUCCESS;}
};

std::shared_ptr<domi::ModelParser> FuncTest() {
  std::shared_ptr<domi::ModelParser> ptr = std::make_shared<CaffeModelParser>();
  return ptr;
}

std::shared_ptr<domi::WeightsParser> WeightsFuncTest() {
  std::shared_ptr<domi::WeightsParser> ptr = std::make_shared<CaffeWeightsParser>();
  return ptr;
}

static ComputeGraphPtr BuildSubComputeGraph() {
  ut::GraphBuilder builder = ut::GraphBuilder("subgraph");
  auto data = builder.AddNode("sub_Data", "sub_Data", 0, 1);
  auto netoutput = builder.AddNode("sub_Netoutput", "sub_NetOutput", 1, 0);
  builder.AddDataEdge(data, 0, netoutput, 0);
  auto graph = builder.GetGraph();
  return graph;
}

// construct graph which contains subgraph
static ComputeGraphPtr BuildComputeGraph() {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data = builder.AddNode("Data", "Data", 1, 1);
  auto transdata = builder.AddNode("Transdata", "Transdata", 1, 1);
  transdata->GetOpDesc()->AddSubgraphName("subgraph");
  transdata->GetOpDesc()->SetSubgraphInstanceName(0, "subgraph");
  auto netoutput = builder.AddNode("Netoutput", "NetOutput", 1, 0);
  builder.AddDataEdge(data, 0, transdata, 0);
  builder.AddDataEdge(transdata, 0, netoutput, 0);
  auto graph = builder.GetGraph();
  // add subgraph
  transdata->SetOwnerComputeGraph(graph);
  ComputeGraphPtr subgraph = BuildSubComputeGraph();
  subgraph->SetParentGraph(graph);
  subgraph->SetParentNode(transdata);
  graph->AddSubgraph("subgraph", subgraph);
  auto op_desc = netoutput->GetOpDesc();
  std::vector<std::string> src_name{"out"};
  op_desc->SetSrcName(src_name);
  std::vector<int64_t> src_index{0};
  op_desc->SetSrcIndex(src_index);
  return graph;
}

TEST_F(UtestOmg, display_model_info_failed) {
  ge::proto::ModelDef model_def;
  EXPECT_NO_THROW(PrintModelInfo(&model_def, 1));
}

TEST_F(UtestOmg, display_model_info_success) {
  ge::proto::ModelDef model_def;
  auto attrs = model_def.mutable_attr();
  ge::proto::AttrDef *attr_def_soc = &(*attrs)["soc_version"];
  attr_def_soc->set_s("Ascend310");
  ge::proto::AttrDef *attr_def = &(*attrs)["om_info_list"];
  attr_def->mutable_list()->add_i(1);
  attr_def->mutable_list()->add_i(2);
  attr_def->mutable_list()->add_i(3);
  attr_def->mutable_list()->add_i(4);
  attr_def->mutable_list()->add_i(5);
  EXPECT_NO_THROW(PrintModelInfo(&model_def, 1));
}

TEST_F(UtestOmg, test_set_out_node_info_with_tensor_name) {
  ComputeGraphPtr compute_graph = ge::MakeShared<ComputeGraph>("tmp_graph");
  OpDescPtr data1_desc = ge::MakeShared<OpDesc>("data1", DATA);
  data1_desc->AddOutputDesc(GeTensorDesc());
  OpDescPtr data2_desc = ge::MakeShared<OpDesc>("data2", DATA);
  data2_desc->AddOutputDesc(GeTensorDesc());
  OpDescPtr add_desc = ge::MakeShared<OpDesc>("add", ADD);
  add_desc->AddInputDesc(GeTensorDesc());
  add_desc->AddInputDesc(GeTensorDesc());
  add_desc->AddOutputDesc(GeTensorDesc());
  auto data1 = compute_graph->AddNode(data1_desc);
  auto data2 = compute_graph->AddNode(data2_desc);
  auto add = compute_graph->AddNode(add_desc);
  GraphUtils::AddEdge(data1->GetOutDataAnchor(0), add->GetInDataAnchor(0));
  GraphUtils::AddEdge(data2->GetOutDataAnchor(0), add->GetInDataAnchor(1));
  ge::Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  // Set global context
  UpdateParserCtxWithOmgCtx();
  domi::GetContext().user_out_nodes.clear();
  domi::GetContext().default_out_nodes.clear();
  domi::GetContext().out_tensor_names.clear();
  domi::GetContext().net_out_nodes.clear();
  domi::GetContext().default_out_nodes.push_back({"add", 0});
  domi::GetContext().out_tensor_names.push_back("net_output");

  string out_type;
  auto ret = SetOutputNodeInfo(graph, out_type);
  ASSERT_EQ(ret, SUCCESS);
  auto graph_output_node_info = compute_graph->GetGraphOutNodesInfo();
  ASSERT_EQ(graph_output_node_info.size(), 1);
  EXPECT_EQ(graph_output_node_info.at(0).first->GetName(), "add");
  EXPECT_EQ(graph_output_node_info.at(0).second, 0);
  string origin_output_tensor_name;
  AttrUtils::GetStr(add_desc->GetOutputDesc(0), ATTR_NAME_ORIGIN_OUTPUT_TENSOR_NAME, origin_output_tensor_name);
  EXPECT_EQ(origin_output_tensor_name, "net_output");
  auto &net_out_name = domi::GetContext().net_out_nodes;
  ASSERT_EQ(net_out_name.size(), 1);
  EXPECT_EQ(net_out_name.at(0), "add:0:net_output");

  // Reset global context
  UpdateOmgCtxWithParserCtx();
}

TEST_F(UtestOmg, test_parse_out_node) {
  // Set global context
  UpdateParserCtxWithOmgCtx();

  Graph graph;
  std::map<string, string> atc_params = {{"out_nodes", "tensor_1;tensor_2"}};
  const char *model_file = "stub";
  const char *weights_file = "stub";
  domi::FrameworkType type = domi::ONNX;
  const char *op_conf = "stub";
  const char *target = "stub";
  RunMode run_mode = RunMode::GEN_OM_MODEL;
  bool dynamic_input = false;
  auto ret = ParseGraph(graph, atc_params, model_file, weights_file, type, op_conf, target, run_mode, dynamic_input);
  EXPECT_EQ(ret, FAILED);
  auto out_tensors = domi::GetContext().user_out_tensors;
  ASSERT_EQ(out_tensors.size(), 2);
  EXPECT_EQ(out_tensors.at(0), "tensor_1");
  EXPECT_EQ(out_tensors.at(1), "tensor_2");

  std::map<string, string> atc_params1 = {{"out_nodes", "node_1:0;tensor_2"}};
  ret = ParseGraph(graph, atc_params1, model_file, weights_file, type, op_conf, target, run_mode, dynamic_input);
  EXPECT_NE(ret, SUCCESS);
  out_tensors = domi::GetContext().user_out_tensors;
  auto out_nodes = domi::GetContext().user_out_nodes;
  ASSERT_EQ(out_tensors.size(), 0);
  ASSERT_EQ(out_nodes.size(), 1);
  EXPECT_EQ(out_nodes.at(0).first, "node_1");

  std::map<string, string> atc_params2 = {{"out_nodes", "tensor_3;node_1:0"}};
  ret = ParseGraph(graph, atc_params2, model_file, weights_file, type, op_conf, target, run_mode, dynamic_input);
  EXPECT_NE(ret, SUCCESS);
  out_tensors = domi::GetContext().user_out_tensors;
  out_nodes = domi::GetContext().user_out_nodes;
  ASSERT_EQ(out_tensors.size(), 1);
  ASSERT_EQ(out_nodes.size(), 0);
  EXPECT_EQ(out_tensors.at(0), "tensor_3");

  // Reset global context
  UpdateOmgCtxWithParserCtx();
}

TEST_F(UtestOmg, test_parse_out_node_parse_output_fp16) {
  // Set global context
  UpdateParserCtxWithOmgCtx();

  Graph graph;
  std::map<string, string> atc_params = {{"out_nodes", "tensor_1;tensor_2"}};
  const char *model_file = "stub";
  const char *weights_file = "stub";
  domi::FrameworkType type = domi::ONNX;
  const char *op_conf = "stub";
  const char *target = "stub";
  RunMode run_mode = RunMode::GEN_OM_MODEL;
  bool dynamic_input = false;

  std::map<string, string> atc_params1 = {{"out_nodes", "node_1:0;tensor_2"},
                                          {"is_output_adjust_hw_layout", "true,false"}};
  auto ret = ParseGraph(graph, atc_params1, model_file, weights_file, type, op_conf, target, run_mode, dynamic_input);
  EXPECT_NE(ret, SUCCESS);
  auto out_tensors = domi::GetContext().user_out_tensors;
  auto out_nodes = domi::GetContext().user_out_nodes;
  ASSERT_EQ(out_tensors.size(), 0);
  ASSERT_EQ(out_nodes.size(), 1);
  EXPECT_EQ(out_nodes.at(0).first, "node_1");

  std::map<string, string> atc_params2 = {{"out_nodes", "tensor_3;node_1:0"}};
  ret = ParseGraph(graph, atc_params2, model_file, weights_file, type, op_conf, target, run_mode, dynamic_input);
  EXPECT_NE(ret, SUCCESS);
  out_tensors = domi::GetContext().user_out_tensors;
  out_nodes = domi::GetContext().user_out_nodes;
  ASSERT_EQ(out_tensors.size(), 1);
  ASSERT_EQ(out_nodes.size(), 0);
  EXPECT_EQ(out_tensors.at(0), "tensor_3");

  // Reset global context
  UpdateOmgCtxWithParserCtx();
}

TEST_F(UtestOmg, ParseGraphModelParserIsNull) {
  Graph graph;
  std::map<std::string, std::string> atc_params = {{"inout_nodes", "out_nodes"}};
  const char *model_file = "stub";
  const char *weights_file = "stub";
  domi::FrameworkType type = domi::FRAMEWORK_RESERVED;
  const char *op_conf = nullptr;
  const char *target = "stub";
  RunMode run_mode = RunMode::GEN_OM_MODEL;
  bool dynamic_input = true;
  Status ret = ParseGraph(graph, atc_params, model_file, weights_file, type, op_conf, target, run_mode, dynamic_input);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestOmg, ParseGraphRunModeIsOnlyPreCheck) {
  Graph out_graph;
  std::map<std::string, std::string> atc_params = {{"in_nodes", ""}};
  std::map<std::string, std::string> atc_params2 = {{"in_nodes", ""}, {"check_report", "./check_report.txt"}};

  ComputeGraphPtr cgp = BuildComputeGraph();
  Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(cgp);
  graph.SaveToFile("./ut_graph1.txt");
  graph.SaveToFile("./ut_graph2.txt");
  const char *model_file = "./ut_graph1.txt";
  const char *weights_file = "./ut_graph2.txt";

  domi::FrameworkType type = domi::ONNX;
  domi::ModelParserFactory::Instance()->creator_map_[type] = FuncTest;

  const char *op_conf = nullptr;
  const char *target = "stub";
  RunMode run_mode = RunMode::ONLY_PRE_CHECK;
  bool dynamic_input = true;
  Status ret = ParseGraph(out_graph, atc_params, model_file, weights_file, type, op_conf, target, run_mode, dynamic_input);
  EXPECT_EQ(ret, FAILED);

  ret = ParseGraph(out_graph, atc_params2, model_file, weights_file, type, op_conf, target, run_mode, dynamic_input);
  EXPECT_EQ(ret, SUCCESS);

  domi::ModelParserFactory::Instance()->creator_map_.clear();
  system("rm -rf ./ut_graph1.txt");
  system("rm -rf ./ut_graph2.txt");
  system("rm -rf ./check_report.txt");
}

TEST_F(UtestOmg, ParseGraphSuccess) {
  Graph out_graph;
  std::map<std::string, std::string> atc_params = {{"in_nodes", ""}};

  ComputeGraphPtr cgp = BuildComputeGraph();
  Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(cgp);
  graph.SaveToFile("./ut_graph1.txt");
  graph.SaveToFile("./ut_graph2.txt");
  const char *model_file = "./ut_graph1.txt";
  const char *weights_file = "./ut_graph2.txt";

  domi::FrameworkType type = domi::ONNX;
  domi::ModelParserFactory::Instance()->creator_map_[type] = FuncTest;

  domi::WeightsParserFactory::Instance()->creator_map_[type] = WeightsFuncTest;

  const char *op_conf = nullptr;
  const char *target = "stub";
  RunMode run_mode = RunMode::GEN_OM_MODEL;
  bool dynamic_input = true;
  Status ret = ParseGraph(out_graph, atc_params, model_file, weights_file, type, op_conf, target, run_mode, dynamic_input);
  EXPECT_EQ(ret, SUCCESS);

  domi::ModelParserFactory::Instance()->creator_map_.clear();
  domi::WeightsParserFactory::Instance()->creator_map_.clear();
  system("rm -rf ./ut_graph1.txt");
  system("rm -rf ./ut_graph2.txt");
}

TEST_F(UtestOmg, ParseGraphSuccessCheckInputShapeNode) {
  Graph out_graph;
  std::map<std::string, std::string> atc_params = {{"in_nodes", ""}};
  std::map<std::string, std::string> atc_params_error = {{"in_nodes", ""},
                                                         {"is_output_adjust_hw_layout", "tr,false"}};
  std::map<std::string, std::string> atc_params_ok = {{"in_nodes", ""},
                                                         {"is_output_adjust_hw_layout", "true,false"}};

  ComputeGraphPtr cgp = BuildComputeGraph();
  Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(cgp);
  graph.SaveToFile("./ut_graph1.txt");
  graph.SaveToFile("./ut_graph2.txt");
  const char *model_file = "./ut_graph1.txt";
  const char *weights_file = "./ut_graph2.txt";

  domi::FrameworkType type = domi::ONNX;
  domi::ModelParserFactory::Instance()->creator_map_[type] = FuncTest;

  domi::WeightsParserFactory::Instance()->creator_map_[type] = WeightsFuncTest;

  const char *op_conf = nullptr;
  const char *target = "stub";
  RunMode run_mode = RunMode::GEN_OM_MODEL;
  bool dynamic_input = false;
  Status ret = ParseGraph(out_graph, atc_params_error, model_file, weights_file, type, op_conf, target, run_mode, dynamic_input);
  EXPECT_EQ(ret, PARAM_INVALID);
  ret = ParseGraph(out_graph, atc_params, model_file, weights_file, type, op_conf, target, run_mode, dynamic_input);
  EXPECT_EQ(ret, SUCCESS);

  domi::ModelParserFactory::Instance()->creator_map_.clear();
  domi::WeightsParserFactory::Instance()->creator_map_.clear();
  system("rm -rf ./ut_graph1.txt");
  system("rm -rf ./ut_graph2.txt");
}

TEST_F(UtestOmg, ConvertOmPathINVALID) {
  const char *model_file = "test1";
  const char *json_file = "test2";
  bool is_covert_to_json = true;
  Status ret = ConvertOm(model_file, json_file, is_covert_to_json);
  EXPECT_EQ(ret, ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID);
}

TEST_F(UtestOmg, ConvertOmDataSizeINVALID) {
  ComputeGraphPtr cgp = BuildComputeGraph();
  Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(cgp);
  graph.SaveToFile("./ut_graph1.txt");
  const char *model_file = "./ut_graph1.txt";
  const char *json_file = nullptr;
  bool is_covert_to_json = false;
  Status ret = ConvertOm(model_file, json_file, is_covert_to_json);
  EXPECT_EQ(ret, ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID);
  system("rm -rf ./ut_graph1.txt");
}

TEST_F(UtestOmg, ConvertOmModelPartition) {
  const ComputeGraphPtr graph = BuildComputeGraph();
  const GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  const auto model_task_def = MakeShared<domi::ModelTaskDef>();
  model_task_def->add_task()->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_END_GRAPH));
  ge_model->SetModelTaskDef(model_task_def);

  TBEKernelStore tbe_kernel_store;
  const auto kernel = MakeShared<OpKernelBin>("hello", std::vector<char>(64, 0));
  tbe_kernel_store.AddTBEKernel(kernel);
  tbe_kernel_store.Build();
  ge_model->SetTBEKernelStore(tbe_kernel_store);

  ModelBufferData model;
  ModelHelper model_helper;
  EXPECT_EQ(model_helper.SaveToOmModel(ge_model, "ut_test_model.om", model), SUCCESS);

  EXPECT_EQ(ConvertOm("ut_test_model.om", "ut_test_model.json", false), SUCCESS);
  system("rm -rf ut_test_model.om");
}

TEST_F(UtestOmg, ConvertFwkModelToJsonTest) {
  const char *model_file = "test1";
  char const*json_file = "test2";
  domi::FrameworkType type = domi::ONNX;
  domi::ModelParserFactory::Instance()->creator_map_[type] = FuncTest;
  Status ret = ConvertFwkModelToJson(type, model_file, json_file);
  EXPECT_EQ(ret, SUCCESS);

  domi::ModelParserFactory::Instance()->creator_map_.clear();
  ret = ConvertFwkModelToJson(type, model_file, json_file);
  EXPECT_EQ(ret, FAILED);

  type = domi::MINDSPORE;
  ret = ConvertFwkModelToJson(type, model_file, json_file);
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(UtestOmg, ConvertPbtxtToJsonPathINVALID) {
  char const*model_file = "test1";
  char const*json_file = "test2";
  Status ret = ConvertPbtxtToJson(model_file, json_file);
  EXPECT_NE(ret, SUCCESS);
}

TEST_F(UtestOmg, ConvertPbtxtToJsonParseFromStringFail) {
  ComputeGraphPtr cgp = BuildComputeGraph();
  Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(cgp);
  graph.SaveToFile("./ut_graph1.txt");
  const char *model_file = "./ut_graph1.txt";
  const char *json_file = "test";
  Status ret = ConvertPbtxtToJson(model_file, json_file);
  EXPECT_EQ(ret, FAILED);
  system("rm -rf ./ut_graph1.txt");
}

TEST_F(UtestOmg, DumpInfershapeJsonBufferIsNull) {
  ge::Graph graph;
  const char *json_file = "test2";
  Status ret = DumpInfershapeJson(graph, json_file);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestOmg, InitDomiOmgContextTest) {
  std::string input_shape = "test0";
  std::string input_format = "test1";
  std::string net_format;
  bool is_dynamic_input = false;
  Status ret = InitDomiOmgContext(input_shape, input_format, net_format, is_dynamic_input);
  EXPECT_EQ(ret, PARAM_INVALID);

  input_format = "NCHW";
  ret = InitDomiOmgContext(input_shape, input_format, net_format, is_dynamic_input);
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(UtestOmg, GetOutputLeafTest) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto in_node = builder.AddNode("Data", "Data", 0, 1);
  auto out_node = builder.AddNode("out_node", "NetOutput", 1, 0);
  int32_t index = 0;
  std::pair<ge::NodePtr, int32_t> node_info(out_node, index);
  std::vector<std::pair<ge::NodePtr, int32_t>> output_nodes_info;
  output_nodes_info.push_back(node_info);

  auto node = std::make_shared<Node>();
  Status ret = GetOutputLeaf(node, output_nodes_info);
  EXPECT_EQ(ret, domi::FAILED);

  ret = GetOutputLeaf(in_node, output_nodes_info);
  EXPECT_EQ(ret, SUCCESS);

  ret = GetOutputLeaf(out_node, output_nodes_info);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestOmg, ConvertPbtxtToJsonFlagIsFalse) {
  ComputeGraphPtr cgp = BuildComputeGraph();
  Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(cgp);
  graph.SaveToFile("./ut_graph1.txt");

  ComputeGraphPtr cgp2 = BuildComputeGraph();
  Graph graph2 = ge::GraphUtilsEx::CreateGraphFromComputeGraph(cgp2);
  graph.SaveToFile("./ut_graph2.txt");

  const char *model_file = "./ut_graph1.txt";
  const char *json_file = "./ut_graph2.txt";
  Status ret = ConvertPbtxtToJson(model_file, json_file);
  EXPECT_EQ(ret, FAILED);
  system("rm -rf ./ut_graph1.txt");
  system("rm -rf ./ut_graph2.txt");
}

TEST_F(UtestOmg, ConvertPbtxtToJsonTest) {
  std::string caseDir = __FILE__;
  std::size_t idx = caseDir.find_last_of("/");
  caseDir = caseDir.substr(0, idx);
  std::string modelFile = caseDir + "/ge_proto_00000261_partition0_rank31_new_sub_graph102_SecondPartitioning.txt";

  const char *json_file = nullptr;
  Status ret = ConvertPbtxtToJson(modelFile.c_str(), json_file);
  EXPECT_NE(ret, SUCCESS); // modelfile not exist

  std::string josnFile = "./test_model.json";
  ret = ConvertPbtxtToJson(modelFile.c_str(), josnFile.c_str());
  EXPECT_NE(ret, SUCCESS); // modelfile not exist
  system("rm -rf ./test_model.json");
}

TEST_F(UtestOmg, DumpInfershapeJsonSuccess) {
  ComputeGraphPtr cgp = BuildComputeGraph();
  Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(cgp);
  graph.SaveToFile("./ut_graph1.txt");

  std::string josnFile = "./test_model.json";
  Status ret = DumpInfershapeJson(graph, josnFile.c_str());
  EXPECT_EQ(ret, SUCCESS);
  system("rm -rf ./ut_graph1.txt");
  system("rm -rf ./test_model.json");
}

TEST_F(UtestOmg, SetOutputNodeInfoFail) {
  ComputeGraphPtr cgp = BuildComputeGraph();
  Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(cgp);

  std::string output_type = "NetOutput";
  Status ret = SetOutputNodeInfo(graph, output_type);
  EXPECT_EQ(ret, domi::FAILED);

  output_type = "node1:0:FP16";
  ret = SetOutputNodeInfo(graph, output_type);
  EXPECT_EQ(ret, domi::FAILED);

  std::pair<std::string, int32_t> out_node("node1", 0);
  domi::GetContext().user_out_nodes.push_back(out_node);
  ret = SetOutputNodeInfo(graph, output_type);
  EXPECT_EQ(ret, domi::FAILED);
}

TEST_F(UtestOmg, SetOutputNodeInfoUserOutPutTest) {
  ComputeGraphPtr compute_graph = ge::MakeShared<ComputeGraph>("tmp_graph");
  OpDescPtr data1_desc = ge::MakeShared<OpDesc>("data1", DATA);
  data1_desc->AddOutputDesc(GeTensorDesc());

  OpDescPtr data2_desc = ge::MakeShared<OpDesc>("data2", DATA);
  vector<int64_t> dims({2, 3, 4, 5});
  GeShape ge_shape(dims);
  Format format = FORMAT_NCHW;
  DataType data_type = DT_FLOAT16;
  data2_desc->AddOutputDesc(GeTensorDesc(ge_shape, format, data_type));

  OpDescPtr add_desc = ge::MakeShared<OpDesc>("add", ADD);
  add_desc->AddInputDesc(GeTensorDesc());
  add_desc->AddInputDesc(GeTensorDesc());
  add_desc->AddOutputDesc(GeTensorDesc());

  auto data1 = compute_graph->AddNode(data1_desc);
  auto data2 = compute_graph->AddNode(data2_desc);
  auto add = compute_graph->AddNode(add_desc);
  GraphUtils::AddEdge(data1->GetOutDataAnchor(0), add->GetInDataAnchor(0));
  GraphUtils::AddEdge(data2->GetOutDataAnchor(0), add->GetInDataAnchor(1));
  ge::Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);


  std::string output_type;
  domi::GetContext().user_out_nodes.clear();
  std::pair<std::string, int32_t> out_node("data1", 1);
  domi::GetContext().user_out_nodes.push_back(out_node);
  Status ret = SetOutputNodeInfo(graph, output_type);
  EXPECT_EQ(ret, domi::FAILED);

  domi::GetContext().user_out_nodes.clear();
  domi::GetContext().final_out_nodes_map.clear();
  std::pair<std::string, int32_t> out_node2("data2", 0);
  domi::GetContext().user_out_nodes.push_back(out_node2);
  domi::GetContext().final_out_nodes_map.emplace("data2:0", std::make_pair("data2", 0));
  domi::GetContext().output_formats.push_back(domi::DOMI_TENSOR_NC1HWC0);
  ret = SetOutputNodeInfo(graph, "data2:0:FP16");
  EXPECT_EQ(ret, domi::SUCCESS);
}

TEST_F(UtestOmg, CreateOutputNodesInfoTest) {
  ut::GraphBuilder builder = ut::GraphBuilder("root");
  auto node1 = builder.AddNode("node1", "Data", 0, 1);
  std::pair<ge::NodePtr, int32_t> out_node_info(node1, 0);
  std::vector<std::pair<ge::NodePtr, int32_t>> output_nodes_info;
  output_nodes_info.push_back(out_node_info);
  std::vector<std::string> output_nodes_name;

  domi::GetContext().out_tensor_names.clear();
  CreateOutputNodesInfo(output_nodes_info, output_nodes_name);
  EXPECT_EQ(output_nodes_name.size(), 1);
}

TEST_F(UtestOmg, GetDefaultOutInfoFail) {
  ComputeGraphPtr compute_graph = ge::MakeShared<ComputeGraph>("tmp_graph");
  OpDescPtr data1_desc = ge::MakeShared<OpDesc>("data1", DATA);
  data1_desc->AddInputDesc(GeTensorDesc());

  OpDescPtr add_desc = ge::MakeShared<OpDesc>("add", ADD);
  add_desc->AddInputDesc(GeTensorDesc());

  auto data1 = compute_graph->AddNode(data1_desc);
  auto add = compute_graph->AddNode(add_desc);
  GraphUtils::AddEdge(data1->GetInDataAnchor(0), add->GetInDataAnchor(0));
  ge::Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  domi::GetContext().user_out_nodes.clear();
  domi::GetContext().type = domi::MINDSPORE;
  std::pair<std::string, int32_t> default_node("node1", 1);
  domi::GetContext().default_out_nodes.push_back(default_node);

  std::string output_type;
  Status ret = SetOutputNodeInfo(graph, output_type);
  EXPECT_EQ(ret, domi::FAILED);

  domi::GetContext().default_out_nodes.clear();
  ret = SetOutputNodeInfo(graph, output_type);
  EXPECT_EQ(ret, domi::SUCCESS);
}

TEST_F(UtestOmg, FindParserSo) {
  system("mkdir so_path");
  system("touch so_path/lib_caffe_parser.so");
  string path = "./so_path";
  std::vector<std::string> file_list;
  std::string caffe_path;
  FindParserSo(path, file_list, caffe_path);
  EXPECT_EQ(!caffe_path.empty(), true);
  system("rm -rf ./so_path");
}

TEST_F(UtestOmg, ParseGraphCheckInputFp16NodesTest) {
  Graph out_graph;
  ComputeGraphPtr cgp = BuildComputeGraph();
  Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(cgp);
  graph.SaveToFile("./ut_graph1.txt");
  graph.SaveToFile("./ut_graph2.txt");

  std::map<std::string, std::string> atc_params = {{"in_nodes", ""}, {"input_shape_range", ""},
      {"is_input_adjust_hw_layout", "fal,fal"}};

  std::map<std::string, std::string> atc_params2 = {{"in_nodes", ""}, {"input_shape_range", ""},
      {"input_fp16_nodes", "node1:0;node2:1"}};

  const char *model_file = "./ut_graph1.txt";
  const char *weights_file = "./ut_graph2.txt";

  domi::FrameworkType type = domi::ONNX;
  domi::ModelParserFactory::Instance()->creator_map_[type] = FuncTest;
  domi::WeightsParserFactory::Instance()->creator_map_[type] = WeightsFuncTest;

  const char *op_conf = nullptr;
  const char *target = "stub";
  RunMode run_mode = RunMode::GEN_OM_MODEL;
  bool is_dynamic_input = false;
  Status ret = ParseGraph(out_graph, atc_params, model_file, weights_file, type, op_conf, target, run_mode, is_dynamic_input);
  EXPECT_EQ(ret, PARAM_INVALID);

  ret = ParseGraph(out_graph, atc_params2, model_file, weights_file, type, op_conf, target, run_mode, is_dynamic_input);
  EXPECT_EQ(ret, PARAM_INVALID);

  domi::ModelParserFactory::Instance()->creator_map_.clear();
  domi::WeightsParserFactory::Instance()->creator_map_.clear();
  system("rm -rf ./ut_graph1.txt");
  system("rm -rf ./ut_graph2.txt");
}

TEST_F(UtestOmg, SetOutputNodeInfoParseOutputTypeFail) {
  ComputeGraphPtr cgp = BuildComputeGraph();
  Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(cgp);

  std::string output_type = "node1:0";
  Status ret = SetOutputNodeInfo(graph, output_type);
  EXPECT_EQ(ret, domi::FAILED);

  output_type = "node1:zero:one";
  ret = SetOutputNodeInfo(graph, output_type);
  EXPECT_EQ(ret, domi::FAILED);

  output_type = "node1:0:UIN16";
  ret = SetOutputNodeInfo(graph, output_type);
  EXPECT_EQ(ret, domi::FAILED);
}

TEST_F(UtestOmg, ParseGraphParseOutNodesTest) {
  Graph out_graph;
  ComputeGraphPtr cgp = BuildComputeGraph();
  Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(cgp);
  graph.SaveToFile("./ut_graph1.txt");
  graph.SaveToFile("./ut_graph2.txt");

  std::map<std::string, std::string> atc_params = {{"in_nodes", ""}, {"out_nodes", "node1:0:0;node2:1:1"}};
  std::map<std::string, std::string> atc_params2 = {{"in_nodes", ""}, {"out_nodes", "node1:zero;node2:1"}};

  const char *model_file = "./ut_graph1.txt";
  const char *weights_file = "./ut_graph2.txt";

  domi::FrameworkType type = domi::ONNX;
  domi::ModelParserFactory::Instance()->creator_map_[type] = FuncTest;
  domi::WeightsParserFactory::Instance()->creator_map_[type] = WeightsFuncTest;

  const char *op_conf = nullptr;
  const char *target = "stub";
  RunMode run_mode = RunMode::GEN_OM_MODEL;
  bool is_dynamic_input = false;
  Status ret = ParseGraph(out_graph, atc_params, model_file, weights_file, type, op_conf, target, run_mode, is_dynamic_input);
  EXPECT_EQ(ret, PARAM_INVALID);

  ret = ParseGraph(out_graph, atc_params2, model_file, weights_file, type, op_conf, target, run_mode, is_dynamic_input);
  EXPECT_EQ(ret, PARAM_INVALID);

  domi::ModelParserFactory::Instance()->creator_map_.clear();
  domi::WeightsParserFactory::Instance()->creator_map_.clear();
  system("rm -rf ./ut_graph1.txt");
  system("rm -rf ./ut_graph2.txt");
}

class FakeModelParser : public domi::ModelParser {
  public:
    FakeModelParser(){}
    ~FakeModelParser(){}
    domi::Status Parse(const char *file, ge::Graph &graph){
      (void)file;
      std::vector<int64_t> shape{8, 4, -1};
      DEF_GRAPH(test_graph) {
        auto data = OP_CFG(DATA).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_FLOAT, shape).Attr(ATTR_NAME_INDEX, 0);
        auto square = OP_CFG(ADD).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_FLOAT, shape);
        auto netoutput = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(0).TensorDesc(FORMAT_ND, DT_FLOAT, shape);
        CHAIN(NODE("Data", data)->NODE("Square", square)->NODE("NetOutput", netoutput));
      };
      auto cgp = ToComputeGraph(test_graph);

      graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(cgp);
      return SUCCESS;
    }
    domi::Status ParseFromMemory(const char *data, uint32_t size, ge::ComputeGraphPtr &graph){return SUCCESS;}
    domi::Status ParseFromMemory(const char *data, uint32_t size, ge::Graph &graph){return SUCCESS;}
    domi::Status ParseProto(const google::protobuf::Message *proto, ge::ComputeGraphPtr &graph){return SUCCESS;}
    domi::Status ParseProtoWithSubgraph(const google::protobuf::Message *proto, GetGraphCallback callback,
                                              ge::ComputeGraphPtr &graph){return SUCCESS;}
    ge::DataType ConvertToGeDataType(const uint32_t type){return DT_FLOAT;}
    domi::Status ParseAllGraph(const google::protobuf::Message *root_proto,
                                              ge::ComputeGraphPtr &root_graph){return SUCCESS;}
};

TEST_F(UtestOmg, CheckInputShapeNodeINVALID) {
  Graph out_graph;
  ComputeGraphPtr cgp = BuildComputeGraph();
  Graph graph = ge::GraphUtilsEx::CreateGraphFromComputeGraph(cgp);
  graph.SaveToFile("./ut_graph1.txt");
  graph.SaveToFile("./ut_graph2.txt");

  std::map<std::string, std::string> atc_params = {{"in_nodes", ""}};

  const char *model_file = "./ut_graph1.txt";
  const char *weights_file = "./ut_graph2.txt";

  domi::FrameworkType type = domi::ONNX;
  domi::ModelParserFactory::Instance()->RegisterCreator(type, FuncTest);
  domi::ModelParserFactory::Instance()->creator_map_[type] = []() { 
    std::shared_ptr<domi::ModelParser> ptr = std::make_shared<FakeModelParser>();
    return ptr;
  };
  domi::WeightsParserFactory::Instance()->creator_map_[type] = WeightsFuncTest;

  const char *op_conf = nullptr;
  const char *target = "stub";
  RunMode run_mode = RunMode::GEN_OM_MODEL;
  bool is_dynamic_input = false;
  Status ret = ParseGraph(out_graph, atc_params, model_file, weights_file, type, op_conf, target, run_mode, is_dynamic_input);
  EXPECT_EQ(ret, PARAM_INVALID);

  domi::ModelParserFactory::Instance()->creator_map_.clear();
  domi::WeightsParserFactory::Instance()->creator_map_.clear();
  system("rm -rf ./ut_graph1.txt");
  system("rm -rf ./ut_graph2.txt");
}
}  // namespace ge

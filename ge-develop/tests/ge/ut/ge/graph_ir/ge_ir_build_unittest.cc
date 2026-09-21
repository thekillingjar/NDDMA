/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdio.h>
#include <gtest/gtest.h>
#include "api/aclgrph/option_utils.h"
#include "graph/testcase/ge_graph/graph_builder_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/tensor_adapter.h"
#include "ge/ge_ir_build.h"
#include "graph/ops_stub.h"
#include "graph/ge_global_options.h"
#include "ge/ge_api_types.h"
#include "macro_utils/dt_public_scope.h"
#include "ge/ge_ir_build.h"
#include "api/aclgrph/option_utils.h"
#include "api/aclgrph/attr_options/attr_options.h"
#include "macro_utils/dt_public_unscope.h"
#include "mmpa/mmpa_api.h"
#include "graph_metadef/common/plugin/plugin_manager.h"
#include "graph/ge_local_context.h"
#include "common/util/mem_utils.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "ge_graph_dsl/assert/graph_assert.h"
#include "framework/common/scope_guard.h"
#include "framework/common/types.h"
#include "faker/model_data_faker.h"
#include "faker/ge_model_builder.h"
#include "faker/aicore_taskdef_faker.h"
#include "register/optimization_option_registry.h"
#include "graph/manager/graph_var_manager.h"
#include "stub/gert_runtime_stub.h"
#include "graph/operator_factory_impl.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "ge_running_env/fake_op.h"
#include "ge_running_env/fake_graph_optimizer.h"
#include "ge_running_env/fake_engine.h"

const string AddNYes = "AddNYes";
const char *const kEnvName = "ASCEND_OPP_PATH";
const string kOpsProto = "libopsproto_rt2.0.so";
const string kOpMaster = "libopmaster_rt2.0.so";
const string kInner = "built-in";
const string kOpsProtoPath = "/op_proto/lib/linux/x86_64/";
const string kOpMasterPath = "/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/";
#include "api/gelib/gelib.h"
#include "framework/omg/ge_init.h"

using namespace ge;

namespace ge {
  extern graphStatus CheckVarDesc(const vector<ge::GraphWithOptions> &graph_and_options, const uint64_t session_id);
ge::graphStatus StubInferShape(ge::Operator &op) {
  auto x_input_desc = op.GetInputDesc(0);
  auto x_shape = x_input_desc.GetShape().GetDims();
  auto x_type = x_input_desc.GetDataType();
  std::vector<std::pair<int64_t, int64_t>> x_shape_range;
  (void)x_input_desc.GetShapeRange(x_shape_range);
  TensorDesc op_output_desc = op.GetOutputDesc(0);
  op_output_desc.SetShape(ge::Shape(x_shape));
  op_output_desc.SetOriginShape(ge::Shape(x_shape));
  op_output_desc.SetDataType(x_type);
  if (!x_shape_range.empty()) {
    op_output_desc.SetShapeRange(x_shape_range);
  }
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  return op_desc->UpdateOutputDesc(0, TensorAdapter::TensorDesc2GeTensorDesc(op_output_desc));
}

ge::graphStatus GetShapeInferShape(ge::Operator &op) {
  std::cout << "Enter infershape getshape" << std::endl;
  std::vector<std::string> tiling_inline_engine;
  tiling_inline_engine.push_back("AIcoreEngine");
  vector<std::string> export_shape_engine;
  export_shape_engine.push_back("AIcoreEngine");
  op.SetAttr("_op_tiling_inline_engine", tiling_inline_engine);
  op.SetAttr("_op_export_shape_engine", export_shape_engine);
  return ge::GRAPH_SUCCESS;
}
}
class UtestIrCommon : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

class UtestIrBuild : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

static ge::OpDescPtr CreateOpDesc(const std::string &name, const std::string &type) {
  OpDescPtr op_desc = std::make_shared<ge::OpDesc>(name, type);
  ge::GeTensorDesc ge_tensor_desc;
  op_desc->AddInputDesc("input", ge_tensor_desc);
  op_desc->AddOutputDesc("output", ge_tensor_desc);

  return op_desc;
}

static ComputeGraphPtr BuildComputeGraph() {
  auto builder = ut::GraphBuilder("test");
  auto data1 = builder.AddNode("input1", DATA, 1, 1, FORMAT_NCHW, DT_FLOAT, {1, 2, 3});
  auto data2 = builder.AddNode("input2", DATA, 1, 1, FORMAT_NCHW, DT_FLOAT, {4, 10});
  auto addn1 = builder.AddNode("addn1", AddNYes, 2, 1);
  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);

  builder.AddDataEdge(data1, 0, addn1, 0);
  builder.AddDataEdge(data2, 0, addn1, 1);
  builder.AddDataEdge(addn1, 0,netoutput, 0);

  return builder.GetGraph();
}

static ComputeGraphPtr BuildComputeGraph1() {
  auto builder = ut::GraphBuilder("test");
  auto data1 = builder.AddNode("input1", DATA, 1, 1, FORMAT_NCHW, DT_FLOAT, {1, 2, 3});
  auto data2 = builder.AddNode("input2", DATA, 1, 1, FORMAT_NCHW, DT_FLOAT, {4, 10});
  auto addn1 = builder.AddNode("addn1", AddNYes, 2, 1);
  auto node1 = builder.AddNode("addd", "Mul", 2, 1);
  auto node2 = builder.AddNode("ffm", "FrameworkOp", 2, 1);
  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);

  builder.AddDataEdge(data1, 0, addn1, 0);
  builder.AddDataEdge(data2, 0, addn1, 1);
  builder.AddDataEdge(addn1, 0,netoutput, 0);

  return builder.GetGraph();
}

static Graph BuildIrConstGraph1() {
  auto builder = ut::GraphBuilder("g1");
  auto const_node_1 = builder.AddNode("const1", CONSTANT, 0, 1, FORMAT_ND, DT_FLOAT, {4, 4});
  auto const_node_2 = builder.AddNode("const2", CONSTANT, 0, 1, FORMAT_ND, DT_FLOAT, {4, 4});
  auto cast1 = builder.AddNode("cast1", CAST, 1, 1, FORMAT_ND, DT_FLOAT, std::vector<int64_t>({4, 4}));
  cast1->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
  AttrUtils::SetInt(cast1->GetOpDesc(), "dst_type", DT_FLOAT16);

  auto cast2 = builder.AddNode("cast2", CAST, 1, 1, FORMAT_ND, DT_FLOAT, std::vector<int64_t>({4, 4}));
  cast2->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
  AttrUtils::SetInt(cast2->GetOpDesc(), "dst_type", DT_FLOAT16);

  auto matmul = builder.AddNode("matmul", MATMUL, 2, 1, FORMAT_ND, DT_FLOAT, {});
  GeTensorDesc weight_desc(GeShape({4, 4}), FORMAT_ND, DT_FLOAT);

  GeTensorPtr tensor = nullptr;
  float value_tmp[16] = {1.0};
  tensor = std::make_shared<GeTensor>(weight_desc, (uint8_t *) (&value_tmp[0]), sizeof(value_tmp));
  OpDescUtils::SetWeights(const_node_1, {tensor});
  OpDescUtils::SetWeights(const_node_2, {tensor});

  builder.AddDataEdge(const_node_1, 0, cast1, 0);
  builder.AddDataEdge(const_node_2, 0, cast2, 0);
  builder.AddDataEdge(cast1, 0, matmul, 0);
  builder.AddDataEdge(cast2, 0, matmul, 1);
  auto compute_graph = builder.GetGraph();
  return GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
}

static Graph BuildSubGraph() {
  auto graph = BuildIrConstGraph1();
  auto builder = ut::GraphBuilder("g1_parent");
  auto parent_node = builder.AddNode("parent_node", PERFORMANCE_MODE, 0, 1, FORMAT_ND, DT_FLOAT, {4, 4});
  auto parent_compute_graph = builder.GetGraph();
  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(graph);

  parent_node->GetOpDesc()->AddSubgraphName(compute_graph->GetName());
  parent_node->GetOpDesc()->SetSubgraphInstanceName(0, compute_graph->GetName());
  compute_graph->SetParentGraph(parent_compute_graph);
  compute_graph->SetParentNode(parent_node);
  parent_compute_graph->AddSubGraph(compute_graph);
  return GraphUtilsEx::CreateGraphFromComputeGraph(parent_compute_graph);
}

static Graph BuildIrVariableGraph1() {
  auto builder = ut::GraphBuilder("g1");
  auto var1 = builder.AddNode("var1", VARIABLE, 0, 1, FORMAT_ND, DT_FLOAT, {4, 4});
  auto var2 = builder.AddNode("var2", CONSTANT, 0, 1, FORMAT_ND, DT_FLOAT, {4, 4});
  auto cast1 = builder.AddNode("cast1", CAST, 1, 1, FORMAT_ND, DT_FLOAT, std::vector<int64_t>({4, 4}));
  cast1->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
  AttrUtils::SetInt(cast1->GetOpDesc(), "dst_type", DT_FLOAT16);

  auto cast2 = builder.AddNode("cast2", CAST, 1, 1, FORMAT_ND, DT_FLOAT, std::vector<int64_t>({4, 4}));
  cast2->GetOpDesc()->MutableOutputDesc(0)->SetDataType(DT_FLOAT16);
  AttrUtils::SetInt(cast2->GetOpDesc(), "dst_type", DT_FLOAT16);

  auto matmul = builder.AddNode("matmul", MATMUL, 2, 1, FORMAT_ND, DT_FLOAT, {});
  GeTensorDesc weight_desc(GeShape({4, 4}), FORMAT_ND, DT_FLOAT);

  builder.AddDataEdge(var1, 0, cast1, 0);
  builder.AddDataEdge(var2, 0, cast2, 0);
  builder.AddDataEdge(cast1, 0, matmul, 0);
  builder.AddDataEdge(cast2, 0, matmul, 1);
  auto compute_graph = builder.GetGraph();
  return GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
}

// data not set attr index;
// but becasue of op proto, register attr index. so all data index is zero;
static Graph BuildIrGraph() {
  auto data1 = op::Data("data1");
  auto data2 = op::Data("data2");
  auto data3 = op::Data("data3");
  std::vector<Operator> inputs {data1, data2, data3};
  std::vector<Operator> outputs;

  Graph graph("test_graph");
  graph.SetInputs(inputs).SetOutputs(outputs);
  return graph;
}

// data set attr index, but is not valid
static Graph BuildIrGraph1() {
  auto data1 = op::Data("data1").set_attr_index(0);
  auto data2 = op::Data("data2").set_attr_index(1);
  auto data3 = op::Data("data3");
  auto data4 = op::Data("Test");
  std::vector<Operator> inputs {data1, data2, data3, data4};
  std::vector<Operator> outputs;

  Graph graph("test_graph");
  graph.AddNodeByOp(Operator("gg", "Mul"));
  graph.SetInputs(inputs).SetOutputs(outputs);
  return graph;
}

static Graph BuildIrGraphWithVar() {
  auto data1 = op::Data("data1").set_attr_index(0);
  auto var1 = op::Variable("var1").set_attr_index(0);
  auto add1 = op::Add("add1").set_input_x1(data1).set_input_x2(var1);

  auto data2 = op::Data("data2").set_attr_index(1);
  auto var2 = op::Variable("var2").set_attr_index(0);
  auto add2 = op::Add("add2").set_input_x1(data2).set_input_x2(var2);

  auto var3 = op::Variable("var3").set_attr_index(0);
  auto var4 = op::Variable("var4").set_attr_index(0);
  auto add3 = op::Add("add3").set_input_x1(var3).set_input_x2(var4);
  std::vector<Operator> inputs{data1, data2};
  std::vector<Operator> outputs{add1, add2, add3};

  Graph graph("test_graph_copy1");
  graph.SetInputs(inputs).SetOutputs(outputs);
  return graph;
}

// data set attr index, but is not valid
static Graph BuildIrGraph2() {
  auto data1 = op::Data("data1").set_attr_index(0);
  auto data2 = op::Data("data2");
  auto data3 = op::Data("data3").set_attr_index(2);
  std::vector<Operator> inputs {data1, data2, data3};
  std::vector<Operator> outputs;

  Graph graph("test_graph");
  graph.SetInputs(inputs).SetOutputs(outputs);
  return graph;
}

// data set attr index
static Graph BuildIrGraph3() {
  auto data1 = op::Data("data1").set_attr_index(0);
  auto data2 = op::Data("data2").set_attr_index(1);
  auto data3 = op::Data("data3").set_attr_index(2);
  std::vector<Operator> inputs {data1, data2, data3};
  std::vector<Operator> outputs;

  Graph graph("test_graph");
  graph.SetInputs(inputs).SetOutputs(outputs);
  return graph;
}

static std::string ConstructOppEnv() {
  std::string path = GetModelPath();
  path = path.substr(0, path.rfind('/'));
  path = path.substr(0, path.rfind('/'));
  path = path.substr(0, path.rfind('/') + 1);
  std::string opp_path = path + "opp_oo_test/";
  system(("mkdir -p " + opp_path).c_str());
  mmSetEnv(kEnvName, opp_path.c_str(), 1);
  std::string scene_path = opp_path + "scene.info";
  system(("touch " + scene_path).c_str());
  system(("echo 'os=linux' > " + scene_path).c_str());
  system(("echo 'arch=x86_64' >> " + scene_path).c_str());
  system("pwd");
  std::string inner_proto_path = opp_path + kInner + kOpsProtoPath;
  system(("mkdir -p " + inner_proto_path).c_str());
  std::string inner_tiling_path = opp_path + kInner + kOpMasterPath;
  system(("mkdir -p " + inner_tiling_path).c_str());
  return opp_path;
}

TEST(UtestIrCommon, update_data_op_shape) {
  ge::OpDescPtr op_desc = CreateOpDesc("Data", "Data");
  map<string, vector<int64_t>> shape_map;
  shape_map["Data"] = {{1,2}};

  Status ret = UpdateDataOpShape(op_desc, shape_map);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST(UtestIrCommon, update_data_op_shape_range) {
  ge::OpDescPtr op_desc = CreateOpDesc("Data", "Data");
  std::vector<std::vector<std::pair<int64_t, int64_t>>> index_shape_range_map;

  std::pair<int64_t, int64_t> range_pair(1, 2);
  vector<pair<int64_t, int64_t>> range_pair_tmp = { range_pair };

  index_shape_range_map.push_back(range_pair_tmp);

  AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, 0);
  Status ret = UpdateDataOpShapeRange(op_desc, index_shape_range_map);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST(UtestIrCommon, update_dynamic_shape_range_success) {
  ComputeGraphPtr graph = BuildComputeGraph();
  std::string input_shape_range = "input1:[1, 2~3, -1];input2:[3~5, 10]";

  Status ret = UpdateDynamicInputShapeRange(graph, input_shape_range);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST(UtestIrCommon, update_dynamic_shape_range_failed) {
  ComputeGraphPtr graph = BuildComputeGraph();
  // 1
  std::string input_shape_range = "input1;[1, 2~3, -1]";
  Status ret = UpdateDynamicInputShapeRange(graph, input_shape_range);
  EXPECT_EQ(ret, ge::PARAM_INVALID);

  // 2
  input_shape_range = "input1:[1, 2~3, -1)";
  ret = UpdateDynamicInputShapeRange(graph, input_shape_range);
  EXPECT_EQ(ret, ge::PARAM_INVALID);

  //3
  input_shape_range = "input1:[1, 3~2, -1];input2:[3~5, 10]";
  ret = UpdateDynamicInputShapeRange(graph, input_shape_range);
  EXPECT_EQ(ret, ge::FAILED);

  //4
  input_shape_range = "input1:[1, 2~-3, -1]";
  ret = UpdateDynamicInputShapeRange(graph, input_shape_range);
  EXPECT_EQ(ret, ge::PARAM_INVALID);

  //5
  input_shape_range = "input:[1, 2~3, -1]";
  ret = UpdateDynamicInputShapeRange(graph, input_shape_range);
  EXPECT_EQ(ret, ge::PARAM_INVALID);

  //6
  input_shape_range = "addn1:[1, 2~3, -1]";
  ret = UpdateDynamicInputShapeRange(graph, input_shape_range);
  EXPECT_EQ(ret, ge::PARAM_INVALID);
}

TEST(UtestIrCommon, check_dynamic_image_size_fail) {
  map<string, vector<int64_t>> shape_map;
  shape_map["input1"] = {8, 3, -1, -1};
  string input_format = "NCHW";
  string dynamic_image_size = "@64,64;128,128;";

  bool ret = CheckDynamicImagesizeInputShapeValid(shape_map, input_format, dynamic_image_size);
  EXPECT_EQ(ret, false);
}

TEST(UtestIrCommon, check_dynamic_image_size_and_input_format_fail) {
  map<string, vector<int64_t>> shape_map;
  shape_map["input1"] = {8, 3, -1, -1};
  string input_format = "ND";
  string dynamic_image_size = "64,64;128,128;";

  bool ret = CheckDynamicImagesizeInputShapeValid(shape_map, input_format, dynamic_image_size);
  EXPECT_EQ(ret, false);
}

TEST(UtestIrCommon, check_input_format_failed) {
  std::string format = "invalid";
  Status ret = CheckInputFormat(format);
  EXPECT_EQ(ret, ge::PARAM_INVALID);
}

TEST(UtestIrCommon, check_dynamic_batch_size_input_shape_succ) {
  map<string, vector<int64_t>> shape_map;
  shape_map.insert(std::pair<string, vector<int64_t>>("data", {-1, 2, 3}));
  std::string dynamic_batch_size = "11";

  bool ret = CheckDynamicBatchSizeInputShapeValid(shape_map, dynamic_batch_size);
  EXPECT_EQ(ret, true);
}

TEST(UtestIrCommon, check_dynamic_images_size_input_shape_succ) {
  map<string, vector<int64_t>> shape_map;
  shape_map.insert(std::pair<string, vector<int64_t>>("data", {4, -1, -1, 5}));
  std::string input_format = "NCHW";
  std::string dynamic_image_size = "4,5";

  Status ret = CheckDynamicImagesizeInputShapeValid(shape_map, input_format, dynamic_image_size);
  EXPECT_EQ(ret, ge::SUCCESS);

}

TEST(UtestIrCommon, check_dynamic_input_param_succ) {
  string dynamic_batch_size = "1";
  string dynamic_image_size;
  string dynamic_dims;
  string input_shape = "data:-1,3,244,244";
  string input_shape_range;
  string input_format = "NCHW";
  bool is_dynamic_input = false;

  Status ret = CheckDynamicInputParamValid(dynamic_batch_size, dynamic_image_size, dynamic_dims,
                                           input_shape, input_shape_range, input_format, is_dynamic_input);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST(UtestIrCommon, check_dynamic_input_param_scalar_succ) {
  string dynamic_batch_size;
  string dynamic_image_size;
  string dynamic_dims = "1;2;3";
  string input_shape = "data:-1,3,244,244;data1:";
  string input_shape_range;
  string input_format = "ND";
  bool is_dynamic_input = false;

  Status ret = CheckDynamicInputParamValid(dynamic_batch_size, dynamic_image_size, dynamic_dims,
                                           input_shape, input_shape_range, input_format, is_dynamic_input);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST(UtestIrCommon, transfer_shape_to_range_succ) {
  string dynamic_batch_size;
  string dynamic_image_size;
  string dynamic_dims;
  string input_shape = "data:-1,1~3,244,244;data1:1~2,3,2";
  string input_shape_range;
  Status ret = CheckAndTransferInputShapeToRange(input_shape, input_shape_range,
                                                 dynamic_batch_size, dynamic_image_size,
                                                 dynamic_dims);
  EXPECT_EQ(ret, ge::SUCCESS);
  EXPECT_TRUE(input_shape == "");
  EXPECT_EQ(input_shape_range, "data:[-1,1~3,244,244];data1:[1~2,3,2]");

  input_shape = "data1:-1,1,244,244;data2:-1,3,2";
  input_shape_range = "";
  ret = CheckAndTransferInputShapeToRange(input_shape, input_shape_range,
                                          dynamic_batch_size, dynamic_image_size,
                                          dynamic_dims);
  EXPECT_EQ(ret, ge::SUCCESS);
  EXPECT_TRUE(input_shape == "");
  EXPECT_EQ(input_shape_range, "data1:[-1,1,244,244];data2:[-1,3,2]");

  input_shape = "-1,1~3,244,244;1~2,3,2";
  input_shape_range = "";
  ret = CheckAndTransferInputShapeToRange(input_shape, input_shape_range,
                                          dynamic_batch_size, dynamic_image_size,
                                          dynamic_dims);
  EXPECT_EQ(ret, ge::SUCCESS);
  EXPECT_TRUE(input_shape == "");
  EXPECT_EQ(input_shape_range, "[-1,1~3,244,244],[1~2,3,2]");

  input_shape = "2,244,244;1,3,2";
  input_shape_range = "";
  ret = CheckAndTransferInputShapeToRange(input_shape, input_shape_range,
                                          dynamic_batch_size, dynamic_image_size,
                                          dynamic_dims);
  EXPECT_EQ(ret, ge::SUCCESS);
  EXPECT_TRUE(input_shape == "2,244,244;1,3,2");
  EXPECT_TRUE(input_shape_range == "");

  input_shape = "data:-1,1,244,244;data1:1,3,2";
  input_shape_range = "";
  dynamic_batch_size = "1";
  ret = CheckAndTransferInputShapeToRange(input_shape, input_shape_range,
                                          dynamic_batch_size, dynamic_image_size,
                                          dynamic_dims);
  EXPECT_EQ(ret, ge::SUCCESS);
  EXPECT_TRUE(input_shape == "data:-1,1,244,244;data1:1,3,2");
  EXPECT_TRUE(input_shape_range == "");

  input_shape = "";
  input_shape_range = "data:[-1,1,244,244];data1:[1,3,2]";
  dynamic_batch_size = "";
  ret = CheckAndTransferInputShapeToRange(input_shape, input_shape_range,
                                          dynamic_batch_size, dynamic_image_size,
                                          dynamic_dims);
  EXPECT_EQ(ret, ge::SUCCESS);
  EXPECT_TRUE(input_shape == "");
  EXPECT_TRUE(input_shape_range == "data:[-1,1,244,244];data1:[1,3,2]");

  input_shape = "data1;data2:-1,3,2";
  input_shape_range = "";
  ret = CheckAndTransferInputShapeToRange(input_shape, input_shape_range,
                                          dynamic_batch_size, dynamic_image_size,
                                          dynamic_dims);
  EXPECT_NE(ret, ge::SUCCESS);

  input_shape = "data1:;data2:-1,3,2";
  input_shape_range = "";
  ret = CheckAndTransferInputShapeToRange(input_shape, input_shape_range,
                                          dynamic_batch_size, dynamic_image_size,
                                          dynamic_dims);
  EXPECT_EQ(ret, ge::SUCCESS);

  input_shape = ":;data2:-1,3,2";
  input_shape_range = "";
  ret = CheckAndTransferInputShapeToRange(input_shape, input_shape_range,
                                          dynamic_batch_size, dynamic_image_size,
                                          dynamic_dims);
  EXPECT_NE(ret, ge::SUCCESS);

  input_shape = "data1 3*;data2,-1,3,2";
  input_shape_range = "";
  ret = CheckAndTransferInputShapeToRange(input_shape, input_shape_range,
                                          dynamic_batch_size, dynamic_image_size,
                                          dynamic_dims);
  EXPECT_NE(ret, ge::SUCCESS);

  input_shape = "data1:-1, 2, 3;data2:1~3, 3, 2";
  input_shape_range = "";
  ret = CheckAndTransferInputShapeToRange(input_shape, input_shape_range,
                                          dynamic_batch_size, dynamic_image_size,
                                          dynamic_dims);
  EXPECT_EQ(ret, ge::SUCCESS);

  input_shape = "data1:3*;data2:-1,3, 2data";
  input_shape_range = "";
  ret = CheckAndTransferInputShapeToRange(input_shape, input_shape_range,
                                          dynamic_batch_size, dynamic_image_size,
                                          dynamic_dims);
  EXPECT_NE(ret, ge::SUCCESS);

  input_shape = "data1:-1,1,244,244;data2:-1,3,2";
  input_shape_range = "data1:[-1,1,244,244];data2:[-1,3,2]";
  ret = CheckAndTransferInputShapeToRange(input_shape, input_shape_range,
                                          dynamic_batch_size, dynamic_image_size,
                                          dynamic_dims);
  EXPECT_EQ(ret, ge::SUCCESS);
  EXPECT_TRUE(input_shape == "");
  EXPECT_TRUE(input_shape_range == "data1:[-1,1,244,244];data2:[-1,3,2]");

  input_shape = "data2:1~3,3,2";
  input_shape_range = "";
  dynamic_batch_size = "1,2,3";
  ret = CheckAndTransferInputShapeToRange(input_shape, input_shape_range,
                                          dynamic_batch_size, dynamic_image_size,
                                          dynamic_dims);
  EXPECT_NE(ret, ge::SUCCESS);

  input_shape = "data2:-1,3,2";
  input_shape_range = "";
  dynamic_batch_size = "1,2,3";
  ret = CheckAndTransferInputShapeToRange(input_shape, input_shape_range,
                                          dynamic_batch_size, dynamic_image_size,
                                          dynamic_dims);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST(UtestIrCommon, check_dynamic_input_param_failed) {
  string dynamic_batch_size = "1";
  string dynamic_image_size;
  string dynamic_dims;
  string input_shape = "data:1,3,244,244";
  string input_shape_range;
  string input_format = "NCHW";
  bool is_dynamic_input = false;

  Status ret = CheckDynamicInputParamValid(dynamic_batch_size, dynamic_image_size, dynamic_dims,
                                           input_shape, input_shape_range, input_format,is_dynamic_input);
  EXPECT_EQ(ret, ge::PARAM_INVALID);
}

TEST(UtestIrCommon, check_modify_mixlist_param) {
  std::string precision_mode = "allow_mix_precision";
  std::string precision_mode_v2 = "";
  std::string modify_mixlist = "/mixlist.json";
  Status ret = CheckModifyMixlistParamValid(precision_mode, precision_mode_v2, modify_mixlist);
  EXPECT_EQ(ret, ge::SUCCESS);

  precision_mode = "";
  precision_mode_v2 = "";
  ret = CheckModifyMixlistParamValid(precision_mode, precision_mode_v2, modify_mixlist);
  EXPECT_EQ(ret, ge::PARAM_INVALID);

  precision_mode = "force_fp16";
  precision_mode_v2 = "";
  ret = CheckModifyMixlistParamValid(precision_mode, precision_mode_v2, modify_mixlist);
  EXPECT_EQ(ret, ge::PARAM_INVALID);

  precision_mode = "";
  precision_mode_v2 = "fp16";
  ret = CheckModifyMixlistParamValid(precision_mode, precision_mode_v2, modify_mixlist);
  EXPECT_EQ(ret, ge::PARAM_INVALID);
}

TEST(UtestIrCommon, check_compress_weight) {
  std::string enable_compress_weight = "true";
  std::string compress_weight_conf="./";
  Status ret = CheckCompressWeightParamValid(enable_compress_weight, compress_weight_conf);
  EXPECT_EQ(ret, PARAM_INVALID);

  enable_compress_weight = "yes";
  compress_weight_conf = "./";
  ret = CheckCompressWeightParamValid(enable_compress_weight, compress_weight_conf);
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST(UtestIrCommon, check_sparsity) {
  EXPECT_EQ(CheckSparseParamValid("-1"), PARAM_INVALID);
  EXPECT_EQ(CheckSparseParamValid("1"), SUCCESS);
}

TEST(UtestIrCommon, check_allow_hf32) {
  EXPECT_EQ(CheckAllowHF32ParamValid("1"), PARAM_INVALID);
  EXPECT_EQ(CheckAllowHF32ParamValid("false"), SUCCESS);
}

TEST(UtestIrCommon, check_attr_compression_empty) {
  EXPECT_EQ(CheckAttrCompressionParamValid(""), SUCCESS);
}

TEST(UtestIrCommon, check_attr_compression_valid_lowercase) {
  EXPECT_EQ(CheckAttrCompressionParamValid("true"), SUCCESS);
  EXPECT_EQ(CheckAttrCompressionParamValid("false"), SUCCESS);
}

TEST(UtestIrCommon, check_attr_compression_valid_uppercase) {
  EXPECT_EQ(CheckAttrCompressionParamValid("TRUE"), SUCCESS);
  EXPECT_EQ(CheckAttrCompressionParamValid("FALSE"), SUCCESS);
}

TEST(UtestIrCommon, check_attr_compression_valid_mixed_case) {
  EXPECT_EQ(CheckAttrCompressionParamValid("True"), SUCCESS);
  EXPECT_EQ(CheckAttrCompressionParamValid("False"), SUCCESS);
  EXPECT_EQ(CheckAttrCompressionParamValid("TrUe"), SUCCESS);
  EXPECT_EQ(CheckAttrCompressionParamValid("FaLsE"), SUCCESS);
}

TEST(UtestIrCommon, check_attr_compression_invalid_values) {
  EXPECT_EQ(CheckAttrCompressionParamValid("yes"), PARAM_INVALID);
  EXPECT_EQ(CheckAttrCompressionParamValid("no"), PARAM_INVALID);
  EXPECT_EQ(CheckAttrCompressionParamValid("1"), PARAM_INVALID);
  EXPECT_EQ(CheckAttrCompressionParamValid("0"), PARAM_INVALID);
  EXPECT_EQ(CheckAttrCompressionParamValid("enable"), PARAM_INVALID);
  EXPECT_EQ(CheckAttrCompressionParamValid("disable"), PARAM_INVALID);
  EXPECT_EQ(CheckAttrCompressionParamValid("on"), PARAM_INVALID);
  EXPECT_EQ(CheckAttrCompressionParamValid("off"), PARAM_INVALID);
  EXPECT_EQ(CheckAttrCompressionParamValid("invalid"), PARAM_INVALID);
}

TEST(UtestIrCommon, check_param_failed) {
  std::string param_invalid = "invalid";

  Status ret = CheckOutputTypeParamValid(param_invalid);
  EXPECT_EQ(ret, PARAM_INVALID);

  ret = CheckBufferOptimizeParamValid(param_invalid);
  EXPECT_EQ(ret, PARAM_INVALID);

  //ret = CheckKeepTypeParamValid(param_invalid);
  //EXPECT_EQ(ret, PARAM_INVALID);

  // ret = CheckInsertOpConfParamValid(param_invalid);
  // EXPECT_EQ(ret, PARAM_INVALID);

  //ret = CheckDisableReuseMemoryParamValid(param_invalid);
  //EXPECT_EQ(ret, PARAM_INVALID);

  ret = CheckEnableSingleStreamParamValid(param_invalid);
  EXPECT_EQ(ret, PARAM_INVALID);

  ret = CheckExternalWeightParamValid(param_invalid);
  EXPECT_EQ(ret, PARAM_INVALID);

  ret = CheckAcParallelEnableParamValid(param_invalid);
  EXPECT_NE(ret, SUCCESS);

  ret = CheckTilingScheduleOptimizeParamValid(param_invalid);
  EXPECT_EQ(ret, PARAM_INVALID);

  ret = CheckQuantDumpableParamValid(param_invalid);
  EXPECT_NE(ret, SUCCESS);

  ret = CheckIsWeightClipParamValid(param_invalid);
  EXPECT_NE(ret, SUCCESS);

  std::string optypelist_for_implmode;
  std::string op_select_implmode = "1";
  ret = CheckImplmodeParamValid(optypelist_for_implmode, op_select_implmode);
  EXPECT_EQ(ret, PARAM_INVALID);

  ret = CheckLogParamValidAndSetLogLevel(param_invalid);
}

TEST(UtestIrCommon, check_external_weight_param_succ) {
  Status ret = CheckExternalWeightParamValid("0");
  EXPECT_EQ(ret, SUCCESS);

  ret = CheckExternalWeightParamValid("1");
  EXPECT_EQ(ret, SUCCESS);

  ret = CheckExternalWeightParamValid("2");
  EXPECT_EQ(ret, SUCCESS);
}

TEST(UtestIrBuild, test_subgraph) {
  auto graph = BuildSubGraph();
  WeightRefreshableGraphs split_graphs;
  std::vector<AscendString> lora_names;

  lora_names.emplace_back(AscendString("const1"));
  lora_names.emplace_back(AscendString("const2"));
  auto ret = aclgrphConvertToWeightRefreshableGraphs(graph, lora_names, split_graphs);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST(UtestIrBuild, aclgrphConvertToWeightRefreshableGraphs) {
  Graph graph = BuildIrConstGraph1();
  EXPECT_EQ(graph.GetAllNodes().size(), 5);
  WeightRefreshableGraphs split_graphs;
  std::vector<AscendString> lora_names;
  // empty test
  graphStatus ret = aclgrphConvertToWeightRefreshableGraphs(graph, lora_names, split_graphs);
  EXPECT_EQ(ret, GRAPH_PARAM_INVALID);
  // error name test
  lora_names.clear();
  lora_names.emplace_back(AscendString("const11"));
  ret = aclgrphConvertToWeightRefreshableGraphs(graph, lora_names, split_graphs);
  EXPECT_EQ(ret, GRAPH_PARAM_INVALID);
  // repeated name
  lora_names.clear();
  lora_names.emplace_back(AscendString("const1"));
  lora_names.emplace_back(AscendString("const1"));
  ret = aclgrphConvertToWeightRefreshableGraphs(graph, lora_names, split_graphs);
  EXPECT_EQ(ret, GRAPH_PARAM_INVALID);

  // non const name
  lora_names.clear();
  lora_names.emplace_back(AscendString("matmul"));
  ret = aclgrphConvertToWeightRefreshableGraphs(graph, lora_names, split_graphs);
  EXPECT_EQ(ret, GRAPH_PARAM_INVALID);

  lora_names.clear();
  lora_names.emplace_back(AscendString("const1"));
  lora_names.emplace_back(AscendString("const2"));
  ret = aclgrphConvertToWeightRefreshableGraphs(graph, lora_names, split_graphs);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  size_t variable_cnt = 0;
  size_t assign_cnt = 0;
  size_t if_cnt = 0;
  for (auto &node : split_graphs.infer_graph.GetAllNodes()) {
    AscendString tmp_name;
    node.GetName(tmp_name);
    AscendString tmp_type;
    node.GetType(tmp_type);
    if (std::string(tmp_type.GetString()) == "Variable") {
      ++variable_cnt;
    }
    if (std::string(tmp_type.GetString()) == "Assign") {
      ++assign_cnt;
    }
    if (std::string(tmp_type.GetString()) == "If") {
      ++if_cnt;
    }
  }
  EXPECT_EQ(variable_cnt, 2);
  EXPECT_TRUE(assign_cnt == 0);
  EXPECT_TRUE(if_cnt == 0);

  for (auto &node : split_graphs.var_init_graph.GetAllNodes()) {
    AscendString tmp_name;
    node.GetName(tmp_name);
    AscendString tmp_type;
    node.GetType(tmp_type);
    if (std::string(tmp_type.GetString()) == "Variable") {
      ++variable_cnt;
    }
    if (std::string(tmp_type.GetString()) == "Assign") {
      ++assign_cnt;
    }
    if (std::string(tmp_type.GetString()) == "If") {
      ++if_cnt;
    }
  }
  EXPECT_EQ(variable_cnt, 4);
  EXPECT_EQ(assign_cnt, 2);
  EXPECT_TRUE(if_cnt == 0);

  for (auto &node : split_graphs.var_update_graph.GetAllNodes()) {
    AscendString tmp_name;
    node.GetName(tmp_name);
    AscendString tmp_type;
    node.GetType(tmp_type);
    if (std::string(tmp_type.GetString()) == "Variable") {
      ++variable_cnt;
    }
    if (std::string(tmp_type.GetString()) == "Assign") {
      ++assign_cnt;
    }
    if (std::string(tmp_type.GetString()) == "If") {
      ++if_cnt;
    }
  }
  EXPECT_EQ(variable_cnt, 6);
  EXPECT_EQ(assign_cnt, 4);
  EXPECT_EQ(if_cnt, 2);
}

TEST(UtestIrBuild, check_aclgrphBundle) {
  std::string path = GetModelPath();
  path = path.substr(0, path.rfind('/'));
  path = path.substr(0, path.rfind('/'));
  path = path.substr(0, path.rfind('/') + 1);

  std::string opp_path = path + "opp/";
  system(("mkdir -p " + opp_path).c_str());
  mmSetEnv(kEnvName, opp_path.c_str(), 1);

  std::string scene_path = opp_path + "scene.info";
  system(("touch " + scene_path).c_str());
  system(("echo 'os=linux' > " + scene_path).c_str());
  system(("echo 'arch=x86_64' >> " + scene_path).c_str());

  system("pwd");
  std::string inner_proto_path = opp_path + kInner + kOpsProtoPath;
  system(("mkdir -p " + inner_proto_path).c_str());
  inner_proto_path += kOpsProto;
  system(("touch " + inner_proto_path).c_str());
  system(("echo 'ops proto:123 ' > " + inner_proto_path).c_str());

  std::string inner_tiling_path = opp_path + kInner + kOpMasterPath;
  system(("mkdir -p " + inner_tiling_path).c_str());
  inner_tiling_path += kOpMaster;
  system(("touch " + inner_tiling_path).c_str());
  system(("echo 'op tiling:456 ' > " + inner_tiling_path).c_str());

  std::map<std::string, std::string> global_options;
  global_options[ge::OPTION_EXEC_HCCL_FLAG] = "0";
  global_options[ge::OPTION_HOST_ENV_OS] = "linux";
  global_options[ge::OPTION_HOST_ENV_CPU] = "x86_64";
  ge::aclgrphBuildInitialize(global_options);

  Graph graph = BuildIrConstGraph1();
  WeightRefreshableGraphs split_graphs;
  std::vector<AscendString> lora_names;
  lora_names.emplace_back(AscendString("const1"));
  lora_names.emplace_back(AscendString("const2"));
  auto ret = aclgrphConvertToWeightRefreshableGraphs(graph, lora_names, split_graphs);
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  std::vector<GraphWithOptions> graph_and_options;
  std::map<AscendString, AscendString> options;
  //options.insert({AscendString("ge.socVersion"), AscendString("Ascend910A")});
  options.insert({AscendString("input_format"), AscendString("ND")});
  ModelBufferData model;
  ret = aclgrphBundleBuildModel(graph_and_options, model);
  EXPECT_EQ(ret, GRAPH_PARAM_INVALID);
  graph_and_options.emplace_back(GraphWithOptions{split_graphs.infer_graph, options});
  graph_and_options.emplace_back(GraphWithOptions{split_graphs.var_init_graph, options});

  auto instance_ptr = ge::GELib::GetInstance();
  EXPECT_NE(instance_ptr, nullptr);
  //  SchedulerConf conf;
  SchedulerConf scheduler_conf;
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL"] = std::make_shared<EngineConf>();
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL"]->name = "DNN_VM_GE_LOCAL";
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL"]->id = "DNN_VM_GE_LOCAL";
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL"]->independent = false;
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL"]->attach = true;
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL"]->skip_assign_stream = true;

  scheduler_conf.cal_engines["AIcoreEngine"] = std::make_shared<EngineConf>();
  scheduler_conf.cal_engines["AIcoreEngine"]->name = "AIcoreEngine";
  scheduler_conf.cal_engines["AIcoreEngine"]->id = "AIcoreEngine";
  scheduler_conf.cal_engines["AIcoreEngine"]->independent = false;
  scheduler_conf.cal_engines["AIcoreEngine"]->attach = false;
  scheduler_conf.cal_engines["AIcoreEngine"]->skip_assign_stream = false;

  scheduler_conf.cal_engines["DNN_VM_AICPU"] = std::make_shared<EngineConf>();
  scheduler_conf.cal_engines["DNN_VM_AICPU"]->name = "DNN_VM_AICPU";
  scheduler_conf.cal_engines["DNN_VM_AICPU"]->id = "DNN_VM_AICPU";
  scheduler_conf.cal_engines["DNN_VM_AICPU"]->independent = false;
  scheduler_conf.cal_engines["DNN_VM_AICPU"]->attach = true;
  scheduler_conf.cal_engines["DNN_VM_AICPU"]->skip_assign_stream = false;

  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL_OP_STORE"] = std::make_shared<EngineConf>();
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL_OP_STORE"]->name = "DNN_VM_GE_LOCAL_OP_STORE";
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL_OP_STORE"]->id = "DNN_VM_GE_LOCAL_OP_STORE";
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL_OP_STORE"]->independent = false;
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL_OP_STORE"]->attach = true;
  scheduler_conf.cal_engines["DNN_VM_GE_LOCAL_OP_STORE"]->skip_assign_stream = true;

  instance_ptr->DNNEngineManagerObj().schedulers_["multi_batch"] = scheduler_conf;

  GeRunningEnvFaker ge_env;
  auto multi_dims = MakeShared<FakeMultiDimsOptimizer>();
  ge_env.Install(FakeEngine("AIcoreEngine").KernelInfoStore("AiCoreLib").GraphOptimizer("AIcoreEngine").Priority(PriorityEnum::COST_0));
  ge_env.Install(FakeEngine("VectorEngine").KernelInfoStore("VectorLib").GraphOptimizer("VectorEngine").Priority(PriorityEnum::COST_1));
  ge_env.Install(FakeEngine("DNN_VM_AICPU").KernelInfoStore("AicpuLib").GraphOptimizer("aicpu_tf_optimizer").Priority(PriorityEnum::COST_3));
  ge_env.Install(FakeEngine("DNN_VM_AICPU_ASCEND").KernelInfoStore("AicpuAscendLib").GraphOptimizer("aicpu_ascend_optimizer").Priority(PriorityEnum::COST_2));
  ge_env.Install(FakeEngine("DNN_HCCL").KernelInfoStore("ops_kernel_info_hccl").GraphOptimizer("hccl_graph_optimizer").GraphOptimizer("hvd_graph_optimizer").Priority(PriorityEnum::COST_1));
  ge_env.Install(FakeEngine("DNN_VM_RTS").KernelInfoStore("RTSLib").GraphOptimizer("DNN_VM_RTS_GRAPH_OPTIMIZER_STORE").Priority(PriorityEnum::COST_1));
  ge_env.Install(FakeEngine("DNN_VM_GE_LOCAL").KernelInfoStore("DNN_VM_GE_LOCAL_OP_STORE").GraphOptimizer("DNN_VM_HOST_CPU_OPTIMIZER").Priority(PriorityEnum::COST_9));
  ge_env.Install(FakeEngine("DNN_VM_HOST_CPU").KernelInfoStore("DNN_VM_HOST_CPU_OP_STORE").GraphOptimizer("DNN_VM_HOST_CPU_OPTIMIZER").Priority(PriorityEnum::COST_10));
  ge_env.Install(FakeEngine("DSAEngine").KernelInfoStore("DSAEngine").Priority(PriorityEnum::COST_1));
  ge_env.Install(FakeEngine("AIcoreEngine").GraphOptimizer("MultiDims", multi_dims));
  ge_env.Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("AicpuLib"));
  ge_env.Install(FakeOp(CASE).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(STREAMACTIVE).InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp(SEND).InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp(RECV).InfoStoreAndBuilder("RTSLib"));
  ge_env.Install(FakeOp(CONSTANT).InfoStoreAndBuilder("AicpuLib"));
  ge_env.Install(FakeOp(VARIABLE).InfoStoreAndBuilder("AicpuLib"));
  ge_env.Install(FakeOp(MATMUL).InferShape(StubInferShape).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(DATA).InferShape(StubInferShape).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(ADD).InferShape(StubInferShape).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(ASSIGN).InfoStoreAndBuilder("AiCoreLib"));
  ge_env.Install(FakeOp(CAST).InfoStoreAndBuilder("AiCoreLib"));

  ret = aclgrphBundleBuildModel(graph_and_options, model);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  ge::aclgrphBuildFinalize();
  RuntimeStub::Reset();
  ge_env.Reset();
}

TEST(UtestIrBuild, check_CheckVarDesc) {
  Graph graph_1 = BuildIrVariableGraph1();
  Graph graph_2 = BuildIrVariableGraph1();

  std::vector<GraphWithOptions> graph_and_options;
  std::map<AscendString, AscendString> options;
  graph_and_options.emplace_back(GraphWithOptions{graph_1, options});
  graph_and_options.emplace_back(GraphWithOptions{graph_2, options});
  auto ret = CheckVarDesc(graph_and_options, 0);
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  // check dtype
  auto compute_graph_2 = ge::GraphUtilsEx::GetComputeGraph(graph_2);
  EXPECT_NE(compute_graph_2, nullptr);
  auto var_1 = compute_graph_2->FindNode("var1");
  EXPECT_NE(var_1, nullptr);
  auto var_1_opdesc = var_1->GetOpDesc();
  EXPECT_NE(var_1_opdesc, nullptr);
  EXPECT_NE(var_1_opdesc->MutableOutputDesc(0), nullptr);
  EXPECT_EQ(var_1_opdesc->MutableOutputDesc(0)->GetDataType(), DT_FLOAT);
  var_1_opdesc->MutableOutputDesc(0)->SetDataType(DT_BOOL);
  ret = CheckVarDesc(graph_and_options, 0);
  EXPECT_NE(ret, GRAPH_SUCCESS);

  // check format
  EXPECT_EQ(var_1_opdesc->MutableOutputDesc(0)->GetFormat(), FORMAT_ND);
  var_1_opdesc->MutableOutputDesc(0)->SetDataType(DT_FLOAT);
  var_1_opdesc->MutableOutputDesc(0)->SetFormat(FORMAT_NC1HWC0);
  ret = CheckVarDesc(graph_and_options, 0);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  TransNodeInfo info;
  std::vector<TransNodeInfo> info_vec{info};
  VarManager::Instance(0)->Init(0,0,0,0);
  VarManager::Instance(0)->SetTransRoad("var1", info_vec);
  VarManager::Instance(0)->SetTransRoad("var2", info_vec);
  ret = CheckVarDesc(graph_and_options, 0);
  EXPECT_NE(ret, GRAPH_SUCCESS);
}

// Get attr index failed, when set input shape range
TEST(UtestIrBuild, check_data_op_attr_index_invalid_0) {
  ComputeGraphPtr compute_graph = BuildComputeGraph();
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  const map<string, string> build_options = {
    {"input_shape_range", "[1, 2~3, -1],[4~5, 3~5, 10],[1, 2~3, -1]"},
    {ge::ir_option::SHAPE_GENERALIZED_BUILD_MODE, "shape_precise"}
  };
  ModelBufferData model;
  graphStatus ret = aclgrphBuildModel(graph, build_options, model);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST(UtestIrBuild, check_tensor_attr_success) {
  ComputeGraphPtr compute_graph = BuildComputeGraph();
  GeTensorDesc tensor_desc(GeShape({1024}));
  GeTensor tensor(tensor_desc);
  auto data_node = compute_graph->FindNode("input1");
  auto in_desc = data_node->GetOpDesc()->MutableInputDesc(0);
  AttrUtils::SetTensor(in_desc, ATTR_NAME_VALUE, tensor);
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);

  DUMP_GRAPH_WHEN("PrepareAfterCheckAndUpdateInput");

  const map<string, string> build_options;
  ModelBufferData model;
  graphStatus ret = aclgrphBuildModel(graph, build_options, model);
  EXPECT_NE(ret, GRAPH_SUCCESS);

  CHECK_GRAPH(PrepareAfterCheckAndUpdateInput) {
    auto data_node = compute_graph->FindNode("input1");
    auto in_desc = data_node->GetOpDesc()->MutableInputDesc(0);
    ConstGeTensorPtr tensor = nullptr;
    EXPECT_EQ(AttrUtils::GetTensor(in_desc, ATTR_NAME_VALUE, tensor), true);
    auto dims = tensor->GetTensorDesc().GetShape().GetDims();
    EXPECT_EQ(dims.size(), 1);
    EXPECT_EQ(dims[0], 1024);
  };
}

TEST(UtestIrBuild, check_irbuild_supported_inner_options_invalid) {
  ComputeGraphPtr compute_graph = BuildComputeGraph();
  Graph graph = GraphUtilsEx::CreateGraphFromComputeGraph(compute_graph);
  const map<string, string> ir_build_options = {
    {"ge.inner_options_invalid", "1"},
  };
  ModelBufferData model;
  graphStatus ret = aclgrphBuildModel(graph, ir_build_options, model);
  EXPECT_EQ(ret, GRAPH_PARAM_INVALID);
}

// not set attr index, when set input shape range
TEST(UtestIrBuild, check_data_op_attr_index_invalid_1) {
  Graph graph = BuildIrGraph();
  const map<string, string> build_options = {
    {"input_shape_range", "[1, 2~3, -1],[4~5, 3~5, 10],[1, 2~3, -1]"}
  };
  ModelBufferData model;
  graphStatus ret = aclgrphBuildModel(graph, build_options, model);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

// set attr index, but not valid, when set input shape range
TEST(UtestIrBuild, check_data_op_attr_index_invalid_2) {
  Graph graph = BuildIrGraph1();
  const map<string, string> build_options = {
    {"input_shape_range", "[1, 2~3, -1],[4~5, 3~5, 10],[1, 2~3, -1]"}
  };
  ModelBufferData model;
  graphStatus ret = aclgrphBuildModel(graph, build_options, model);
  EXPECT_NE(ret, GRAPH_SUCCESS);

  Graph graph2 = BuildIrGraph2();
  ret = aclgrphBuildModel(graph2, build_options, model);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

// set attr index valid, when set input shape range
// only check data op attr index valid func.
TEST(UtestIrBuild, check_data_op_attr_index_valid) {
  Graph graph = BuildIrGraph3();
  const map<string, string> build_options = {
    {"input_shape_range", "[1, 2~3, -1],[4~5, 3~5, 10],[1, 2~3, -1]"}
  };
  ModelBufferData model;
  graphStatus ret = aclgrphBuildModel(graph, build_options, model);
  EXPECT_EQ(ret, ge::GE_CLI_GE_NOT_INITIALIZED);
}

// set attr index invalid, when not set input shape range
// only check data op attr index valid func.
TEST(UtestIrBuild, check_data_attr_index_succ_no_input_range) {
  Graph graph = BuildIrGraph1();
  const map<string, string> build_options;
  ModelBufferData model;
  graphStatus ret = aclgrphBuildModel(graph, build_options, model);
  EXPECT_EQ(ret, ge::GE_CLI_GE_NOT_INITIALIZED);
}

TEST(UtestIrBuild, check_modify_mixlist_param) {
  Graph graph = BuildIrGraph1();
  const std::map<std::string, std::string> build_options = {
    {"ge.exec.modify_mixlist", "/modify.json"}
  };
  ModelBufferData model;

  auto ret = aclgrphBuildModel(graph, build_options, model);
  EXPECT_EQ(ret, GRAPH_PARAM_INVALID);
}

TEST(UtestIrBuild, check_op_precision_mode_param) {
  Graph graph = BuildIrGraph1();
  const std::map<std::string, std::string> build_options = {
    {"ge.exec.op_precision_mode", "./op_precision_mode.ini"}
  };
  ModelBufferData model;

  auto ret = aclgrphBuildModel(graph, build_options, model);
  EXPECT_EQ(ret, GRAPH_PARAM_INVALID);
}

TEST(UtestIrBuild, check_input_format_param) {
  Graph graph_1 = BuildIrGraph1();
  ModelBufferData model_1;
  const std::map<std::string, std::string> build_options = {
    {"input_format", "ND"}
  };
  auto ret = ge::aclgrphBuildModel(graph_1, build_options, model_1);
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);
}

TEST(UtestIrBuild, check_customize_dtypes_param) {
  Graph graph_1 = BuildIrGraph1();
  ModelBufferData model_1;
  const std::map<std::string, std::string> build_options = {
    {"ge.customizeDtypes", "./op_precision_mode.ini"}
  };
  auto ret = ge::aclgrphBuildModel(graph_1, build_options, model_1);
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);
}

TEST(UtestIrBuild, CheckPrecisionModeParamValid_Failed) {
  Graph graph_1 = BuildIrGraph1();
  ModelBufferData model_1;
  const std::map<std::string, std::string> build_options = {
      {ge::PRECISION_MODE, "invalid"}
  };
  auto ret = ge::aclgrphBuildModel(graph_1, build_options, model_1);
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);
  std::string ge_option;
  EXPECT_NE(ge::GetThreadLocalContext().GetOption(ge::PRECISION_MODE, ge_option), ge::SUCCESS);
}

TEST(UtestIrBuild, CheckPrecisionModeV2ParamValid_Failed) {
  Graph graph_1 = BuildIrGraph1();
  ModelBufferData model_1;
  const std::map<std::string, std::string> build_options = {
      {ge::PRECISION_MODE_V2, "invalid"}
  };
  auto ret = ge::aclgrphBuildModel(graph_1, build_options, model_1);
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);
  std::string ge_option;
  EXPECT_NE(ge::GetThreadLocalContext().GetOption(ge::PRECISION_MODE_V2, ge_option), ge::SUCCESS);
}

TEST(UtestIrBuild, CheckPrecisionModeV2ParamValid_Failed_WhenConfigPrecisionModeAtTheSameTime) {
  Graph graph_1 = BuildIrGraph1();
  ModelBufferData model_1;
  const std::map<std::string, std::string> build_options = {
      {ge::PRECISION_MODE_V2, "fp16"},
      {ge::PRECISION_MODE, "force_fp16"}
  };
  auto ret = ge::aclgrphBuildModel(graph_1, build_options, model_1);
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);
  std::string ge_option;
  EXPECT_NE(ge::GetThreadLocalContext().GetOption(ge::PRECISION_MODE_V2, ge_option), ge::SUCCESS);
}

TEST(UtestIrBuild, CheckPrecisionModeParamValid_Success) {
  Graph graph_1 = BuildIrGraph1();
  ModelBufferData model_1;
  const std::map<std::string, std::string> build_options = {
      {ge::PRECISION_MODE, "force_fp16"}
  };
  (void)ge::aclgrphBuildModel(graph_1, build_options, model_1);
  std::string ge_option;
  EXPECT_EQ(ge::GetThreadLocalContext().GetOption(ge::PRECISION_MODE, ge_option), ge::SUCCESS);
  EXPECT_STREQ(ge_option.c_str(), "force_fp16");
}

TEST(UtestIrBuild, CheckPrecisionModeV2ParamValid_Success) {
  Graph graph_1 = BuildIrGraph1();
  ModelBufferData model_1;
  const std::map<std::string, std::string> build_options = {
      {ge::PRECISION_MODE_V2, "fp16"}
  };
  (void)ge::aclgrphBuildModel(graph_1, build_options, model_1);
  std::string ge_option;
  EXPECT_EQ(ge::GetThreadLocalContext().GetOption(ge::PRECISION_MODE_V2, ge_option), ge::SUCCESS);
  EXPECT_STREQ(ge_option.c_str(), "fp16");
}

TEST(UtestIrBuild, check_build_model_and_build_step) {
  Graph graph_1 = BuildIrGraph1();
  const std::map<std::string, std::string> build_options_1 = {
    {"ge.buildMode", "xxx"}
  };
  ModelBufferData model_1;
  auto ret_1 = aclgrphBuildModel(graph_1, build_options_1, model_1);
  EXPECT_NE(ret_1, GRAPH_SUCCESS);

  Graph graph_2 = BuildIrGraph1();
  const std::map<std::string, std::string> build_options_2 = {
    {"ge.buildStep", "xxx"}
  };
  ModelBufferData model_2;
  auto ret_2 = aclgrphBuildModel(graph_2, build_options_2, model_2);
  EXPECT_NE(ret_2, GRAPH_SUCCESS);

  Graph graph_3 = BuildIrGraph1();
  const std::map<std::string, std::string> build_options_3 = {
    {"ge.buildMode", "tuning"}
  };
  ModelBufferData model_3;
  auto ret_3 = aclgrphBuildModel(graph_3, build_options_3, model_3);
  EXPECT_NE(ret_3, GRAPH_SUCCESS);
}

TEST(UtestIrBuild, atc_cfg_optype_param) {
  ComputeGraphPtr graph = BuildComputeGraph1();
  FILE *fp = fopen("./keep.txt", "w+");
  if (fp) {
    fprintf(fp, "Test\n");
    fprintf(fp, "OpType::Mul\n");
    fprintf(fp, "Optype::Sub\n");
    fclose(fp);
  }
  auto ret = KeepDtypeFunc(graph, "./keep.txt");
  (void)remove("./keep.txt");
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST(UtestIrBuild, atc_cfg_optype_param_nok) {
  ComputeGraphPtr graph = BuildComputeGraph1();
  FILE *fp = fopen("./keep.txt", "w+");
  if (fp) {
    fprintf(fp, "Test\n");
    fprintf(fp, "OpType::mul\n");
    fprintf(fp, "Optype::sub\n");
    fclose(fp);
  }
  auto ret = KeepDtypeFunc(graph, "./keep.txt");
  (void)remove("./keep.txt");
  EXPECT_EQ(ret, GRAPH_PARAM_INVALID);
}

TEST(UtestIrBuild, aclgrphGenerateForOp_test) {
  ge::AscendString type("Add");
  ge::TensorDesc tensor_desc;
  int32_t *c = new int32_t[10];
  void *d = static_cast<void *>(c);
  std::unique_ptr<uint8_t[]> data = std::unique_ptr<uint8_t[]>(reinterpret_cast<uint8_t *>(d));
  tensor_desc.SetConstData(std::move(data), sizeof(int));
  ge::Graph graph;
  ge::graphStatus ret = ge::aclgrphGenerateForOp(type, vector<ge::TensorDesc>{tensor_desc},
                                                 vector<ge::TensorDesc>{tensor_desc}, graph);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST(UtestIrBuild, aclgrphGenerateForOp_singleop_test) {
  std::string pwd = __FILE__;
  std::size_t idx = pwd.find_last_of("/");
  pwd = pwd.substr(0, idx);
  std::string json_file = pwd + "/../session/add_int.json";
  ge::AscendString json(json_file.c_str());

  std::vector<Graph> graphs;
  auto ret = ge::aclgrphGenerateForOp(json, graphs);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(graphs.size(), 1);
}

TEST(UtestIrBuild, aclgrphGenerateForOp_singleop_const_test) {
  std::string pwd = __FILE__;
  std::size_t idx = pwd.find_last_of("/");
  pwd = pwd.substr(0, idx);
  std::string json_file = pwd + "/../session/add_int_const.json";
  ge::AscendString json(json_file.c_str());

  std::vector<Graph> graphs;
  auto ret = ge::aclgrphGenerateForOp(json, graphs);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
  EXPECT_EQ(graphs.size(), 1);
  bool has_const = false;
  for (auto &node : graphs[0].GetAllNodes()) {
    AscendString str_type("");
    (void)node.GetType(str_type);
    if (std::string(str_type.GetString()) == "Constant") {
      has_const = true;
      break;
    }
  }
  EXPECT_TRUE(has_const);
}

TEST(UtestIrBuild, aclgrphBuildInitialize_test) {
  std::string path = GetModelPath();
  path = path.substr(0, path.rfind('/'));
  path = path.substr(0, path.rfind('/'));
  path = path.substr(0, path.rfind('/') + 1);

  std::string opp_path = path + "opp/";
  system(("mkdir -p " + opp_path).c_str());
  mmSetEnv(kEnvName, opp_path.c_str(), 1);

  std::string scene_path = opp_path + "scene.info";
  system(("touch " + scene_path).c_str());
  system(("echo 'os=linux' > " + scene_path).c_str());
  system(("echo 'arch=x86_64' >> " + scene_path).c_str());

  system("pwd");
  std::string inner_proto_path = opp_path + kInner + kOpsProtoPath;
  system(("mkdir -p " + inner_proto_path).c_str());
  inner_proto_path += kOpsProto;
  system(("touch " + inner_proto_path).c_str());
  system(("echo 'ops proto:123 ' > " + inner_proto_path).c_str());

  std::string inner_tiling_path = opp_path + kInner + kOpMasterPath;
  system(("mkdir -p " + inner_tiling_path).c_str());
  inner_tiling_path += kOpMaster;
  system(("touch " + inner_tiling_path).c_str());
  system(("echo 'op tiling:456 ' > " + inner_tiling_path).c_str());

  std::map<std::string, std::string> global_options;
  global_options[ge::OPTION_EXEC_HCCL_FLAG] = "0";
  global_options[ge::OPTION_HOST_ENV_OS] = "linux";
  global_options[ge::OPTION_HOST_ENV_CPU] = "x86_64";
  ge::graphStatus ret = ge::aclgrphBuildInitialize(global_options);
  ge::aclgrphBuildFinalize();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  std::map<AscendString, AscendString> global_options1;
  global_options1[ge::OPTION_EXEC_HCCL_FLAG] = "0";
  global_options1[ge::OPTION_HOST_ENV_OS] = "linux";
  global_options1[ge::OPTION_HOST_ENV_CPU] = "x86_64";
  global_options1[ge::OPTION_SCREEN_PRINT_MODE] = "enable";
  ret = ge::aclgrphBuildInitialize(global_options1);
  ge::aclgrphBuildFinalize();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  std::map<AscendString, AscendString> global_options2;
  global_options1["ge.autoTuneMode"] = "RA";
  ret = ge::aclgrphBuildInitialize(global_options1);
  ge::aclgrphBuildFinalize();
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);

  std::map<std::string, std::string> global_options3;
  global_options3[ge::DETERMINISTIC] = "1";
  global_options3["ge.deterministicLevel"] = "1";
  ret = ge::aclgrphBuildInitialize(global_options3);
  auto &options = GetMutableGlobalOptions();
  auto it = options.find(ge::DETERMINISTIC);
  EXPECT_NE(it, options.end());
  if (it != options.end()) {
    EXPECT_EQ(it->second == "1", true);
  }
  it = options.find("ge.deterministicLevel");
  EXPECT_NE(it, options.end());
  if (it != options.end()) {
    EXPECT_EQ(it->second == "1", true);
  }
  ge::aclgrphBuildFinalize();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  std::map<std::string, std::string> global_options4;
  global_options4[ge::OP_PRECISION_MODE] = "op_precision.ini";
  ret = ge::aclgrphBuildInitialize(global_options4);
  options = GetMutableGlobalOptions();
  it = options.find(ge::OP_PRECISION_MODE);
  EXPECT_NE(it, options.end());
  if (it != options.end()) {
    EXPECT_EQ(it->second == "op_precision.ini", true);
  }
  ge::aclgrphBuildFinalize();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  std::map<std::string, std::string> global_options5;
  global_options5[ge::ALLOW_HF32] = "1";
  ret = ge::aclgrphBuildInitialize(global_options5);
  ge::aclgrphBuildFinalize();
  EXPECT_EQ(ret, ge::GRAPH_PARAM_INVALID);

  std::map<std::string, std::string> global_options6;
  global_options6[ge::OPTION_SCREEN_PRINT_MODE] = "0";
  ret = ge::aclgrphBuildInitialize(global_options6);
  ge::aclgrphBuildFinalize();
  EXPECT_EQ(ret, ge::GRAPH_PARAM_INVALID);
}

TEST(UtestIrBuild, aclgrphBuildInitialize_test_fail) {
  std::map<std::string, std::string> global_options2;
  global_options2["ge.optionInvalid"] = "invalid";
  EXPECT_EQ(ge::aclgrphBuildInitialize(global_options2), ge::GRAPH_SUCCESS);
  ge::aclgrphBuildFinalize();
}

TEST(UtestIrBuild, check_compression_optimize_conf_test) {
  std::map<std::string, std::string> global_options2;
  global_options2[ge::COMPRESSION_OPTIMIZE_CONF] = "0";
  global_options2[ge::OPTION_HOST_ENV_OS] = "linux";
  global_options2[ge::OPTION_HOST_ENV_CPU] = "x86_64";
  ge::graphStatus ret = ge::aclgrphBuildInitialize(global_options2);
  ge::aclgrphBuildFinalize();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST(UtestIrBuild, check_virtual_type_test_success) {
  std::map<std::string, std::string> global_options2;
  global_options2[ge::VIRTUAL_TYPE ] = "1";
  global_options2[ge::OPTION_HOST_ENV_OS] = "linux";
  global_options2[ge::OPTION_HOST_ENV_CPU] = "x86_64";
  ge::graphStatus ret = ge::aclgrphBuildInitialize(global_options2);
  ge::aclgrphBuildFinalize();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST(UtestIrBuild, aclgrphBuildInitializeCheckJitCompileTrue) {
  std::map<std::string, std::string> global_options2;
  global_options2[ge::VIRTUAL_TYPE ] = "1";
  global_options2[ge::OPTION_HOST_ENV_OS] = "linux";
  global_options2[ge::OPTION_HOST_ENV_CPU] = "x86_64";
  ge::graphStatus ret = ge::aclgrphBuildInitialize(global_options2);
  ge::aclgrphBuildFinalize();
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  std::string jit_compile_option;
  GetThreadLocalContext().GetOption(JIT_COMPILE, jit_compile_option);
  EXPECT_STREQ(jit_compile_option.c_str(), "1");
}

TEST(UtestIrBuild, check_dynamic_dims_input_shape_valid_test) {
  std::map<std::string, std::vector<int64_t>> shape_map;
  std::string dynamic_dims;
  std::vector<int64_t> shape = {1, 3, 224,224};
  shape_map.insert(map<std::string, std::vector<int64_t>>::value_type ("NCHW", shape));
  auto ret = ge::CheckDynamicDimsInputShapeValid(shape_map, dynamic_dims);
  EXPECT_EQ(ret, false);
}

TEST(UtestIrBuild, check_and_parse_dynamic_dims_test) {
  const std::map<std::string, std::vector<int64_t>> shape_map;
  int32_t dynamic_dim_num = 0;
  std::string dynamic_dims = "";
  auto ret = ge::CheckAndParseDynamicDims(dynamic_dim_num, dynamic_dims);
  EXPECT_EQ(ret, false);
  dynamic_dims = "1;3;224;224";
  ret = ge::CheckAndParseDynamicDims(dynamic_dim_num, dynamic_dims);
  EXPECT_EQ(ret, false);
  dynamic_dim_num = 4;
  dynamic_dims = "1,3,-1,-1";
  ret = ge::CheckAndParseDynamicDims(dynamic_dim_num, dynamic_dims);
  EXPECT_EQ(ret, false);
}

TEST(UtestIrBuild, ParseInputShapeRange_Failure_InvalidShape) {
  std::string shape_range;
  std::vector<std::vector<std::pair<int64_t, int64_t>>> range;
  auto ret = ge::ParseInputShapeRange(shape_range, range);
  ASSERT_NE(ret, SUCCESS);
  shape_range = "[1,8-31,512]";
  ret = ge::ParseInputShapeRange(shape_range, range);
  ASSERT_NE(ret, SUCCESS);
}

TEST(UtestIrBuild, ParseInputShapeRange_Success_CheckShapeRange) {
  std::string shape_range;
  std::vector<std::vector<std::pair<int64_t, int64_t>>> range;
  shape_range = "[1,8~31,512]";
  std::vector<std::pair<int64_t, int64_t>> expected_range{{1, 1}, {8, 31}, {512, 512}};
  auto ret = ge::ParseInputShapeRange(shape_range, range);
  ASSERT_EQ(ret, SUCCESS);
  EXPECT_EQ(range[0], expected_range);
  shape_range = "[-1,8~31,512]";
  range.clear();
  expected_range = {{0, -1}, {8, 31}, {512, 512}};
  ret = ge::ParseInputShapeRange(shape_range, range);
  ASSERT_EQ(ret, SUCCESS);
  EXPECT_EQ(range[0], expected_range);
}

TEST(UtestIrBuild, parse_check_dynamic_input_param_valid_test) {
  std::string dynamic_batch_size;
  std::string dynamic_image_size;
  std::string dynamic_dims;
  std::string input_shape;
  std::string input_shape_range = "1:3:224:224";
  std::string input_format;
  bool is_dynamic_input;
  auto ret = ge::CheckDynamicInputParamValid(dynamic_batch_size, dynamic_image_size, dynamic_dims,
      input_shape, input_shape_range, input_format, is_dynamic_input);
  ASSERT_NE(ret, SUCCESS);
  dynamic_batch_size = "1";
  dynamic_image_size = "1";
  dynamic_dims = "1:3";
  ret = ge::CheckDynamicInputParamValid(dynamic_batch_size, dynamic_image_size, dynamic_dims,
      input_shape, input_shape_range, input_format, is_dynamic_input);
  ASSERT_NE(ret, SUCCESS);
  dynamic_image_size = "1";
  dynamic_dims = "";
  input_shape = "1";
  ret = ge::CheckDynamicInputParamValid(dynamic_batch_size, dynamic_image_size, dynamic_dims,
      input_shape, input_shape_range, input_format, is_dynamic_input);
  ASSERT_NE(ret, SUCCESS);
  dynamic_image_size = "";
  dynamic_dims = "1";
  input_shape = "1";
  ret = ge::CheckDynamicInputParamValid(dynamic_batch_size, dynamic_image_size, dynamic_dims,
      input_shape, input_shape_range, input_format, is_dynamic_input);
  ASSERT_NE(ret, SUCCESS);
}

TEST(UtestIrBuild, check_log_param_valid_and_set_log_level_test) {
  std::string log = "";
  auto ret = ge::CheckLogParamValidAndSetLogLevel(log);
  ASSERT_EQ(ret, -1);
  log = "debug";
  ret = ge::CheckLogParamValidAndSetLogLevel(log);
  ASSERT_EQ(ret, 0);
  log = "info";
  ret = ge::CheckLogParamValidAndSetLogLevel(log);
  ASSERT_NE(ret, -1);
  log = "warning";
  ret = ge::CheckLogParamValidAndSetLogLevel(log);
  ASSERT_NE(ret, -1);
  log = "error";
  ret = ge::CheckLogParamValidAndSetLogLevel(log);
  ASSERT_NE(ret, -1);
}

TEST(UtestIrBuild, check_impl_mode_param_valid_test) {
  std::string optypelist_for_implmode = "mode";
  std::string op_select_implmode = "";
  auto ret = ge::CheckImplmodeParamValid(optypelist_for_implmode, op_select_implmode);
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST(UtestIrBuild, update_data_op_shape_range_test) {
  OpDescPtr op = CreateOpDesc("data1", "DATA");
  std::vector<std::vector<std::pair<int64_t, int64_t>>> index_shape_range_map;
  auto ret = ge::UpdateDataOpShapeRange(op, index_shape_range_map);
  EXPECT_EQ(ret, SUCCESS);
  int64_t index = -1;
  AttrUtils::SetInt(op, ATTR_NAME_INDEX, index);
  index = 0;
  AttrUtils::GetInt(op, ATTR_NAME_INDEX, index);
  ret = ge::UpdateDataOpShapeRange(op, index_shape_range_map);
  EXPECT_EQ(ret, SUCCESS);
}

TEST(UtestIrBuild, aclgrphBuildModel_test) {
  Graph graph_1 = BuildIrGraph1();
  ModelBufferData model_1;
  std::map<AscendString, AscendString> global_options1;
  global_options1[ge::OPTION_EXEC_HCCL_FLAG] = "0";
  auto ret = ge::aclgrphBuildModel(graph_1, global_options1, model_1);
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);

  Graph graph_2 = BuildIrGraph1();
  ModelBufferData model_2;
  std::map<AscendString, AscendString> global_options2;
  global_options2["ge.autoTuneMode"] = "RA";
  ret = ge::aclgrphBuildModel(graph_2, global_options2, model_2);
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);
}

TEST(UtestIrBuild, aclgrphBuildModel_test1) {
  ModelBufferData model_1;
  std::string output_file = "./test";
  auto ret = ge::aclgrphSaveModel(output_file, model_1);
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);

  char *output_file1 = nullptr;
  ret = ge::aclgrphSaveModel(output_file1, model_1);
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);
  Graph graph_1 = BuildIrGraph1();
  const std::map<std::string, std::string> build_options_1 = {
    {"ge.buildMode", "xxx"}
  };
  auto ret_1 = aclgrphBuildModel(graph_1, build_options_1, model_1);
  EXPECT_NE(ret_1, GRAPH_SUCCESS);
  ret = ge::aclgrphSaveModel(output_file, model_1);
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);
  ret = ge::aclgrphSaveModel(output_file1, model_1);
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);
  output_file1 = (char *)"./test1";
  ret = ge::aclgrphSaveModel(output_file1, model_1);
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);
}

TEST(UtestIrBuild, aclgrphGetIRVersion_test) {
  int32_t version = 1;
  int32_t *major_version = &version;
  int32_t *minor_version = &version;
  int32_t *patch_version = &version;
  auto ret = ge::aclgrphGetIRVersion(major_version, minor_version, patch_version);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST(UtestIrBuild, aclgrphBuildModel_test2) {
  Graph graph_1 = BuildIrGraph1();
  const char *file = "";
  size_t len = 0U;
  // (len > PATH_MAX || len != strlen(file) || strlen(file) == 0)
  EXPECT_EQ(ge::aclgrphDumpGraph(graph_1, file, len), GRAPH_PARAM_INVALID);
}

TEST(UtestIrBuild, aclgrphSetOpAttr_test) {
  Graph graph_1 = BuildIrGraph1();
  aclgrphAttrType attr_type = ATTR_TYPE_KEEP_DTYPE;
  const char *cfg_path = "./test_aclgrphSetOpAttr_test";
  auto ret = ge::aclgrphSetOpAttr(graph_1, attr_type, cfg_path);
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);
}

TEST(UtestIrBuild, aclgrphSetOpAttr_test_AttrTypeOutOfRange) {
  Graph graph_1 = BuildIrGraph1();
  const char *cfg_path = "./test_aclgrphSetOpAttr_test";
  auto ret = ge::aclgrphSetOpAttr(graph_1, (aclgrphAttrType)4, cfg_path);
  EXPECT_NE(ret, ge::GRAPH_SUCCESS);
}

TEST(UtestIrBuild, check_dynamic_imagesize_input_shapevalid_test) {
  OpDescPtr op = CreateOpDesc("data1", "DATA");
  std::map<std::string, std::vector<int64_t>> shape_map;
  std::string input_format = "HHH";
  std::string dynamic_image_size;
  auto ret = ge::CheckDynamicImagesizeInputShapeValid(shape_map, input_format, dynamic_image_size);
  EXPECT_EQ(ret, SUCCESS);
  shape_map.insert(std::pair<string, vector<int64_t>>("data", {1, 3, 224}));
  input_format = "NCHW";
  ret = ge::CheckDynamicImagesizeInputShapeValid(shape_map, input_format, dynamic_image_size);
  EXPECT_EQ(ret, SUCCESS);
  dynamic_image_size = "1,3,224";
  ret = ge::CheckDynamicImagesizeInputShapeValid(shape_map, input_format, dynamic_image_size);
  EXPECT_EQ(ret, SUCCESS);
}

TEST(UtestIrCommon, check_dynamic_batch_size_input_shape_test1) {
  map<string, vector<int64_t>> shape_map;
  shape_map.insert(std::pair<string, vector<int64_t>>("data", {-1, 2, 3}));
  std::string dynamic_batch_size = "11:11";
  auto ret = CheckDynamicBatchSizeInputShapeValid(shape_map, dynamic_batch_size);
  EXPECT_NE(ret, true);
}

TEST(UtestIrCommon, weight_compress_func_test) {
    ComputeGraphPtr graph = BuildComputeGraph1();
    std::string currentDir = "";
    auto ret = ge::WeightCompressFunc(graph, currentDir);
    ASSERT_EQ(ret, SUCCESS);
    currentDir = "./test_weight_compress_func_test";
    ret = ge::WeightCompressFunc(graph, currentDir);
    ASSERT_NE(ret, SUCCESS);
    currentDir = __FILE__;
    std::size_t idx = currentDir.find_last_of("/");
    currentDir = currentDir.substr(0, idx) + "/weight_compress.cfg";
    ret = ge::WeightCompressFunc(graph, currentDir);
    ASSERT_EQ(ret, SUCCESS);
}

TEST(UtestIrCommon, is_original_op_find_test) {
    ge::OpDescPtr op_desc = CreateOpDesc("Data", "Data");
    std::vector<std::vector<std::pair<int64_t, int64_t>>> index_shape_range_map;
    std::string op_name = "Data";
    vector<pair<int64_t, int64_t>> range_pair_tmp;
    std::vector<std::string> original_op_names = {"DATA"};
    AttrUtils::SetListStr(op_desc, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_op_names);
    index_shape_range_map.push_back(range_pair_tmp);

    AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, 0);
    auto ret = ge::IsOriginalOpFind(op_desc, op_name);
    ASSERT_EQ(ret, SUCCESS);
    op_name = "TEST";
    ret = ge::IsOriginalOpFind(op_desc, op_name);
    ASSERT_EQ(ret, SUCCESS);
}

TEST(UtestIrCommon, is_op_type_equal_test) {
    auto builder = ut::GraphBuilder("test");
    NodePtr data1 = builder.AddNode("input1", DATA, 1, 1, FORMAT_NCHW, DT_FLOAT, {1, 2, 3});
    NodePtr data2 = builder.AddNode("input2", "CASE", 1, 1, FORMAT_NCHW, DT_FLOAT, {1, 2, 3});
    std::string op_type = "DATA1";
    auto ret = ge::IsOpTypeEqual(data1, op_type);
    ASSERT_EQ(ret, SUCCESS);
    data1->SetOrigNode(data2);
    ret = ge::IsOpTypeEqual(data1, op_type);
    ASSERT_EQ(ret, SUCCESS);
}

TEST(UtestIrBuild, aclgrphSaveModelTest) {
  ModelBufferData model_1;
  model_1.length = 1;
  model_1.data = std::make_shared<uint8_t>(1);
  char *output_file = nullptr;
  auto ret = ge::aclgrphSaveModel(output_file, model_1);
  EXPECT_EQ(ret, GRAPH_PARAM_INVALID);

  const char *output_file2 = "./graph1.txt";
  ret = ge::aclgrphSaveModel(output_file2, model_1);
  EXPECT_EQ(ret, ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID);
  system("rm ./graph1.txt");

  std::string output_file3 = "./graph2.txt";
  ret = ge::aclgrphSaveModel(output_file3, model_1);
  EXPECT_EQ(ret, ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID);
  system("rm ./graph2.txt");

  ModelBufferData model_2;
  model_2.length = sizeof(ModelFileHeader);
  auto buff_data = std::make_unique<uint8_t[]>(model_2.length);
  model_2.data.reset(buff_data.release(), std::default_delete<uint8_t[]>());
  auto *header = reinterpret_cast<ModelFileHeader *>(model_2.data.get());
  header->modeltype = ge::MODEL_TYPE_FLOW_MODEL;
  std::string output_file4 = "./graph3.txt";
  ret = ge::aclgrphSaveModel(output_file4, model_2);
  EXPECT_NE(ret, SUCCESS);
  system("rm ./graph3.txt");
}

TEST(UtestIrCommon, CheckDynamicBatchSizeInputShapeValidTest) {
  map<string, vector<int64_t>> shape_map;
  shape_map.insert(std::pair<string, vector<int64_t>>("data", {}));
  std::string dynamic_batch_size = "1";

  bool ret = CheckDynamicBatchSizeInputShapeValid(shape_map, dynamic_batch_size);
  EXPECT_EQ(ret, false);
}

TEST(UtestIrBuild, check_os_err) {
  std::map<std::string, std::string> global_options2;
  global_options2[ge::COMPRESSION_OPTIMIZE_CONF] = "0";
  global_options2[ge::OPTION_HOST_ENV_OS] = "Linux";
  global_options2[ge::OPTION_HOST_ENV_CPU] = "x86_64";
  ge::graphStatus ret = ge::aclgrphBuildInitialize(global_options2);
  ge::aclgrphBuildFinalize();
  EXPECT_EQ(ret, ge::GRAPH_PARAM_INVALID);
}

TEST(UtestIrBuild, check_cpu_err) {
  std::map<std::string, std::string> global_options2;
  global_options2[ge::COMPRESSION_OPTIMIZE_CONF] = "0";
  global_options2[ge::OPTION_HOST_ENV_OS] = "linux";
  global_options2[ge::OPTION_HOST_ENV_CPU] = "aaaaa";
  ge::graphStatus ret = ge::aclgrphBuildInitialize(global_options2);
  ge::aclgrphBuildFinalize();
  EXPECT_EQ(ret, ge::GRAPH_PARAM_INVALID);
}

TEST(UtestIrBuild, bundle_save_successfully) {
  auto graph = BuildIrGraphWithVar();
  auto root_graph = GraphUtilsEx::GetComputeGraph(graph);

  auto var1 = root_graph->FindNode("var1");
  ASSERT_NE(var1, nullptr);
  GeTensorDesc desc(GeShape({1, 1, 4, 4}));
  ge::TensorUtils::SetSize(*var1->GetOpDescBarePtr()->MutableOutputDesc(0), 64);
  var1->GetOpDescBarePtr()->SetOutputOffset({137438953472U});

  auto var2 = root_graph->FindNode("var2");
  ASSERT_NE(var2, nullptr);
  ge::TensorUtils::SetSize(*var2->GetOpDescBarePtr()->MutableOutputDesc(0), 64);
  var2->GetOpDescBarePtr()->SetOutputOffset({137438953572U});

  gert::GeModelBuilder builder(root_graph);
  auto ge_root_model =
      builder.AddTaskDef("Add", gert::AiCoreTaskDefFaker("AddStubName")).FakeTbeBin({"Add"}).BuildGeRootModel();
  auto model_data = gert::ModelDataFaker().GeRootModel(ge_root_model).BuildUnknownShape();
  auto buff_data = MakeUnique<uint8_t[]>(model_data.Get().model_len);
  memcpy_s(buff_data.get(), model_data.Get().model_len, model_data.Get().model_data, model_data.Get().model_len);

  std::vector<ModelBufferData> model_datas;
  ModelBufferData buffer;
  buffer.data.reset(buff_data.release(), std::default_delete<uint8_t[]>());
  buffer.length = model_data.Get().model_len;
  model_datas.push_back(buffer);
  model_datas.push_back(buffer);

  ModelBufferData bundle_model;
  EXPECT_EQ(ModelHelper::SaveBundleModelBufferToMem(model_datas, 2048, bundle_model), SUCCESS);

  EXPECT_NE(aclgrphBundleSaveModel(nullptr, bundle_model), SUCCESS);

  std::string output_file = "saved_bundle_model";
  EXPECT_EQ(aclgrphBundleSaveModel(output_file.c_str(), bundle_model), SUCCESS);

  std::string single_model_out = "saved_single_model";
  EXPECT_EQ(aclgrphSaveModel(single_model_out.c_str(), model_datas[0]), SUCCESS);

  aclgrphBuildFinalize();
  system("rm -f ./saved_bundle_model.om");
  system("rm -f ./saved_single_model.om");

}

namespace ge{
extern Status VerifyVarOffset(const ComputeGraphPtr &root_graph,
                              std::map<std::string, std::pair<int64_t, GeTensorDesc>> &var_name_to_verify_info);
}

TEST(UtestIrBuild, var_desc_verified_failed) {
  auto graph1 = BuildIrGraphWithVar();
  auto root_graph1 = GraphUtilsEx::GetComputeGraph(graph1);

  auto graph2 = BuildIrGraphWithVar();
  auto root_graph2 = GraphUtilsEx::GetComputeGraph(graph2);

  auto var11 = root_graph1->FindNode("var1");
  ASSERT_NE(var11, nullptr);
  ge::TensorUtils::SetSize(*var11->GetOpDescBarePtr()->MutableOutputDesc(0), 64);
  var11->GetOpDescBarePtr()->SetOutputOffset({137438959572U});

  auto var12 = root_graph1->FindNode("var2");
  ASSERT_NE(var12, nullptr);
  ge::TensorUtils::SetSize(*var12->GetOpDescBarePtr()->MutableOutputDesc(0), 64);
  var12->GetOpDescBarePtr()->SetOutputOffset({137438959572U});

  auto var21 = root_graph2->FindNode("var1");
  ASSERT_NE(var21, nullptr);
  ge::TensorUtils::SetSize(*var21->GetOpDescBarePtr()->MutableOutputDesc(0), 64);
  var21->GetOpDescBarePtr()->SetOutputOffset({137438959572U});

  auto var22 = root_graph2->FindNode("var2");
  ASSERT_NE(var22, nullptr);
  ge::TensorUtils::SetSize(*var22->GetOpDescBarePtr()->MutableOutputDesc(0), 64);
  var22->GetOpDescBarePtr()->SetOutputOffset({137438959572U});
  var11->GetOpDescBarePtr()->MutableOutputDesc(0)->SetDataType(DT_COMPLEX128);
  std::map<std::string, std::pair<int64_t, GeTensorDesc>> var_name_to_verify_info;
  EXPECT_EQ(VerifyVarOffset(root_graph1, var_name_to_verify_info), SUCCESS);
  EXPECT_NE(VerifyVarOffset(root_graph2, var_name_to_verify_info), SUCCESS);
}

TEST(UtestIrBuild, varoffset_verified_failed) {
  auto graph1 = BuildIrGraphWithVar();
  auto root_graph1 = GraphUtilsEx::GetComputeGraph(graph1);

  auto graph2 = BuildIrGraphWithVar();
  auto root_graph2 = GraphUtilsEx::GetComputeGraph(graph2);

  auto var11 = root_graph1->FindNode("var1");
  ASSERT_NE(var11, nullptr);
  ge::TensorUtils::SetSize(*var11->GetOpDescBarePtr()->MutableOutputDesc(0), 64);
  var11->GetOpDescBarePtr()->SetOutputOffset({137438959572U});

  auto var12 = root_graph1->FindNode("var2");
  ASSERT_NE(var12, nullptr);
  ge::TensorUtils::SetSize(*var12->GetOpDescBarePtr()->MutableOutputDesc(0), 64);
  var12->GetOpDescBarePtr()->SetOutputOffset({137438959572U});

  auto var21 = root_graph2->FindNode("var1");
  ASSERT_NE(var21, nullptr);
  ge::TensorUtils::SetSize(*var21->GetOpDescBarePtr()->MutableOutputDesc(0), 64);
  var21->GetOpDescBarePtr()->SetOutputOffset({137438953472U});

  auto var22 = root_graph2->FindNode("var2");
  ASSERT_NE(var22, nullptr);
  ge::TensorUtils::SetSize(*var22->GetOpDescBarePtr()->MutableOutputDesc(0), 64);
  var22->GetOpDescBarePtr()->SetOutputOffset({137438953572U});
  std::map<std::string, std::pair<int64_t, GeTensorDesc>> var_name_to_verify_info;
  EXPECT_EQ(VerifyVarOffset(root_graph1, var_name_to_verify_info), SUCCESS);
  EXPECT_NE(VerifyVarOffset(root_graph2, var_name_to_verify_info), SUCCESS);
}

TEST(UtestIrBuild, bundle_save_flow_model_invalid_buffer) {
  ModelBufferData model_1;
  model_1.length = sizeof(ModelFileHeader);
  auto buff_data1 = std::make_unique<uint8_t[]>(model_1.length);
  model_1.data.reset(buff_data1.release(), std::default_delete<uint8_t[]>());
  auto *header1 = reinterpret_cast<ModelFileHeader *>(model_1.data.get());
  header1->modeltype = ge::MODEL_TYPE_FLOW_MODEL;

  ModelBufferData model_2;
  model_2.length = sizeof(ModelFileHeader);
  auto buff_data = std::make_unique<uint8_t[]>(model_2.length);
  model_2.data.reset(buff_data.release(), std::default_delete<uint8_t[]>());
  auto *header = reinterpret_cast<ModelFileHeader *>(model_2.data.get());
  header->modeltype = ge::MODEL_TYPE_FLOW_MODEL;

  std::vector<ModelBufferData> model_datas{model_1, model_2};

  ModelBufferData bundle_model;
  EXPECT_EQ(ModelHelper::SaveBundleModelBufferToMem(model_datas, 2048, bundle_model), SUCCESS);

  std::string output_file = "saved_bundle_flow_model";
  EXPECT_NE(aclgrphBundleSaveModel(output_file.c_str(), bundle_model), SUCCESS);
  aclgrphBuildFinalize();
}

TEST(UtestIrBuild, ir_build_oo_init) {
  const auto opp_path = ConstructOppEnv();
  Graph graph_1 = BuildIrGraph1();
  ModelBufferData model_1;

  std::map<std::string, std::string> global_options = {
      {ge::SOC_VERSION, "Ascend910"},
      {ge::OPTION_HOST_ENV_OS, "linux"},
      {ge::OPTION_HOST_ENV_CPU, "x86_64"},
      {ge::OO_LEVEL, "O1"},
      {OO_CONSTANT_FOLDING, "false"}
  };
  EXPECT_EQ(ge::aclgrphBuildInitialize(global_options), GRAPH_SUCCESS);

  const std::map<std::string, std::string> build_options = {{ge::OO_LEVEL, "O1"}, {OO_CONSTANT_FOLDING, "true"}};
  EXPECT_NE(ge::aclgrphBuildModel(graph_1, build_options, model_1), GRAPH_SUCCESS);
  std::string opt_value;
  EXPECT_EQ(GetThreadLocalContext().GetOo().GetValue(OO_CONSTANT_FOLDING, opt_value), ge::GRAPH_SUCCESS);
  EXPECT_EQ(opt_value, "true");

  aclgrphBuildFinalize();
  system(("rm -rf " + opp_path).c_str());
  GetThreadLocalContext().SetGlobalOption({});
  GetThreadLocalContext().SetSessionOption({});
  GetThreadLocalContext().SetGraphOption({});
  GetThreadLocalContext().GetOo().Initialize({}, OptionRegistry::GetInstance().GetRegisteredOptTable());
}

TEST(UtestIrBuild, ir_build_oo_init_param_invalid) {
  const auto opp_path = ConstructOppEnv();
  Graph graph_1 = BuildIrGraph1();
  ModelBufferData model_1;

  std::map<std::string, std::string> global_options;
  global_options[OO_LEVEL] = "O4";
  EXPECT_NE(ge::aclgrphBuildInitialize(global_options), GRAPH_SUCCESS);

  global_options[OO_LEVEL] = "O1";
  global_options[OO_CONSTANT_FOLDING] = "False";
  EXPECT_NE(ge::aclgrphBuildInitialize(global_options), GRAPH_SUCCESS);

  global_options[OO_CONSTANT_FOLDING] = "0";
  EXPECT_NE(ge::aclgrphBuildInitialize(global_options), GRAPH_SUCCESS);

  std::map<std::string, std::string> build_options;
  build_options[OO_LEVEL] = "O4";
  EXPECT_NE(ge::aclgrphBuildModel(graph_1, build_options, model_1), GRAPH_SUCCESS);

  build_options[OO_LEVEL] = "O1";
  build_options[OO_CONSTANT_FOLDING] = "False";
  EXPECT_NE(ge::aclgrphBuildModel(graph_1, build_options, model_1), GRAPH_SUCCESS);

  build_options[OO_CONSTANT_FOLDING] = "0";
  EXPECT_NE(ge::aclgrphBuildModel(graph_1, build_options, model_1), GRAPH_SUCCESS);

  aclgrphBuildFinalize();
  system(("rm -rf " + opp_path).c_str());

  GetThreadLocalContext().SetGlobalOption({});
  GetThreadLocalContext().SetSessionOption({});
  GetThreadLocalContext().SetGraphOption({});
  GetThreadLocalContext().GetOo().Initialize({}, OptionRegistry::GetInstance().GetRegisteredOptTable());
}

TEST(UtestIrBuild, ir_build_export_compile_stat_valid) {
  const auto opp_path = ConstructOppEnv();
  Graph graph_1 = BuildIrGraph1();
  ModelBufferData model_1;
  std::string opt_value("-1");
  EXPECT_EQ(GetThreadLocalContext().GetOption(OPTION_EXPORT_COMPILE_STAT, opt_value), ge::GRAPH_SUCCESS);
  EXPECT_EQ(opt_value, "1");


  std::map<std::string, std::string> global_options;
  std::map<std::string, std::string> build_options;
  global_options[OPTION_EXPORT_COMPILE_STAT] = "0";
  EXPECT_EQ(ge::aclgrphBuildInitialize(global_options), GRAPH_SUCCESS);
  EXPECT_NE(ge::aclgrphBuildModel(graph_1, build_options, model_1), GRAPH_SUCCESS);
  EXPECT_EQ(GetThreadLocalContext().GetOption(OPTION_EXPORT_COMPILE_STAT, opt_value), ge::GRAPH_SUCCESS);
  EXPECT_EQ(opt_value, "0");

  build_options[OPTION_EXPORT_COMPILE_STAT] = "2";
  Graph graph_2 = BuildIrGraph1();
  EXPECT_NE(ge::aclgrphBuildModel(graph_2, build_options, model_1), GRAPH_SUCCESS);
  EXPECT_EQ(GetThreadLocalContext().GetOption(OPTION_EXPORT_COMPILE_STAT, opt_value), ge::GRAPH_SUCCESS);
  EXPECT_EQ(opt_value, "2");

  aclgrphBuildFinalize();
  system(("rm -rf " + opp_path).c_str());

  GetThreadLocalContext().SetGlobalOption({});
  GetThreadLocalContext().SetSessionOption({});
  GetThreadLocalContext().SetGraphOption({});
  GetThreadLocalContext().GetOo().Initialize({}, OptionRegistry::GetInstance().GetRegisteredOptTable());
}

TEST(UtestIrBuild, ir_build_export_compile_stat_invalid) {
  GetThreadLocalContext().GetOo().Initialize({}, {});
  const auto opp_path = ConstructOppEnv();
  Graph graph_1 = BuildIrGraph1();
  ModelBufferData model_1;
  std::string opt_value("-1");
  std::map<std::string, std::string> global_options;
  global_options[OPTION_EXPORT_COMPILE_STAT] = "3";
  EXPECT_NE(ge::aclgrphBuildInitialize(global_options), GRAPH_SUCCESS);

  aclgrphBuildFinalize();
  system(("rm -rf " + opp_path).c_str());

  GetThreadLocalContext().SetGlobalOption({});
  GetThreadLocalContext().SetSessionOption({});
  GetThreadLocalContext().SetGraphOption({});
  GetThreadLocalContext().GetOo().Initialize({}, OptionRegistry::GetInstance().GetRegisteredOptTable());
}

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
#include "ge/ge_api.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "ge_graph_dsl/assert/graph_assert.h"
#include "graph/ge_local_context.h"
#include "graph/execute/model_executor.h"
#include "graph/tuning_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "ge_running_env/tensor_utils.h"
#include "api/gelib/gelib.h"

namespace ge {
class TestAoe : public testing::Test {
 protected:
  void SetUp() {
    ge_env.InstallDefault();
  }
  void TearDown() {
    auto instance_ptr = ge::GELib::GetInstance();
    if (instance_ptr != nullptr) {
      instance_ptr->Finalize();
    }
  }
  GeRunningEnvFaker ge_env;
};
namespace {
ComputeGraphPtr MakeFunctionGraph1(const std::string &func_node_name, const std::string &func_node_type,
                                   bool is_scalar = false) {
  std::vector<int64_t> shape = {};
  if (!is_scalar) {
    shape = {2, 2, 3, 2};  // HWCN
  }
  auto data_tensor = GenerateTensor(shape);
  auto const1 =
      OP_CFG(CONSTANT).Weight(data_tensor).TensorDesc(FORMAT_HWCN, DT_INT32, shape).InCnt(1).OutCnt(1).Build("const1");
  DEF_GRAPH(g1) {
    CHAIN(NODE(const1)->NODE(func_node_name, func_node_type)->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("_arg_1", DATA)->NODE(func_node_name));
    CHAIN(NODE("_arg_2", DATA)->NODE(func_node_name));
  };
  return ToComputeGraph(g1);
}
ComputeGraphPtr MakeFunctionGraph2(const std::string &func_node_name, const std::string &func_node_type) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("_arg_0", DATA)->NODE(func_node_name, func_node_type)->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("_arg_1", DATA)->NODE(func_node_name));
    CHAIN(NODE("_arg_2", DATA)->NODE(func_node_name));
  };
  return ToComputeGraph(g1);
};

ComputeGraphPtr MakeSubGraph(const std::string &prefix) {
  DEF_GRAPH(g2, prefix.c_str()) {
    auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 0);
    auto data_1 = OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 1);
    auto netoutput = OP_CFG(NETOUTPUT)
                         .InCnt(1)
                         .TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, 3, 16, 16})
                         .OutCnt(1)
                         .TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, 3, 16, 16})
                         .Build("netoutput");
    auto conv_0 = OP_CFG(CONV2D).Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM));
    auto relu_0 = OP_CFG(RELU).Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::AI_CPU));
    CHAIN(NODE(prefix + "_arg_0", data_0)
              ->EDGE(0, 0)
              ->NODE(prefix + "Conv2D", conv_0)
              ->EDGE(0, 0)
              ->NODE(prefix + "Relu", relu_0))
        ->NODE(netoutput);
    CHAIN(NODE(prefix + "_arg_1", data_1)->EDGE(0, 1)->NODE(prefix + "Conv2D", conv_0));
  };
  return ToComputeGraph(g2);
}

ComputeGraphPtr MakeGraph() {
  std::vector<int64_t> shape = {2, 2, 3, 2};  // HWCN
  auto data_tensor = GenerateTensor(shape);
  DEF_GRAPH(g) {
    ge::OpDescPtr data1;
    data1 = OP_CFG(DATA)
                .Attr(ATTR_NAME_INDEX, 0)
                .InCnt(1)
                .TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, 3, 16, 16})
                .OutCnt(1)
                .TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, 3, 16, 16})
                .Build("data1");
    auto data2 = OP_CFG(DATA)
                     .Attr(ATTR_NAME_INDEX, 1)
                     .InCnt(1)
                     .TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, 3, 16, 16})
                     .OutCnt(1)
                     .TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, 3, 16, 16})
                     .Build("data2");
    auto const1 = OP_CFG(CONSTANT)
                      .Weight(data_tensor)
                      .TensorDesc(FORMAT_HWCN, DT_FLOAT, shape)
                      .InCnt(1)
                      .OutCnt(1)
                      .Build("const1");

    auto conv2d1 = OP_CFG(CONV2D).InCnt(2).OutCnt(1).Build("conv2d1");

    auto relu1 = OP_CFG(RELU).InCnt(1).OutCnt(1).Build("relu1");

    auto relu2 = OP_CFG(RELU).InCnt(1).OutCnt(1).Build("relu2");

    auto netoutput1 = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).Build("netoutput1");

    CHAIN(NODE(data1)->NODE(conv2d1)->NODE(relu1))->NODE(netoutput1);
    CHAIN(NODE(data2)->NODE(relu2)->EDGE(0, 1)->NODE(netoutput1));
    CHAIN(NODE(const1)->EDGE(0, 1)->NODE(conv2d1));
  };
  return ToComputeGraph(g);
}

void AddPartitionedCall(const ComputeGraphPtr &graph, const std::string &call_name, const ComputeGraphPtr &subgraph) {
  const auto &call_node = graph->FindNode(call_name);
  if (call_node == nullptr) {
    return;
  }
  call_node->GetOpDesc()->RegisterSubgraphIrName("f", SubgraphType::kStatic);

  size_t index = call_node->GetOpDesc()->GetSubgraphInstanceNames().size();
  call_node->GetOpDesc()->AddSubgraphName(subgraph->GetName());
  call_node->GetOpDesc()->SetSubgraphInstanceName(index, subgraph->GetName());

  subgraph->SetParentNode(call_node);
  subgraph->SetParentGraph(graph);
  GraphUtils::FindRootGraph(graph)->AddSubgraph(subgraph);
}
}  // namespace
TEST_F(TestAoe, test_build_mode_build_step_with_function_node_case_const_input) {
  std::string func_node_name = "Case_0";
  const auto &root_graph = MakeFunctionGraph1(func_node_name, CASE);
  const auto &sub_graph = MakeSubGraph("sub_graph_0/");
  AddPartitionedCall(root_graph, func_node_name, sub_graph);
  const auto &sub_graph_1 = MakeSubGraph("sub_graph_1/");
  AddPartitionedCall(root_graph, func_node_name, sub_graph_1);

  map<std::string, std::string> options;
  options.emplace(BUILD_MODE, BUILD_MODE_TUNING);
  options.emplace(BUILD_STEP, BUILD_STEP_AFTER_BUILDER);
  Status ret = ge::GELib::Initialize(options);
  EXPECT_EQ(ret, SUCCESS);
  Session session(options);
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(root_graph);
  session.AddGraph(1, graph, options);
  std::vector<InputTensorInfo> inputs;
  DUMP_GRAPH_WHEN("PrepareAfterInsertAipp", "PrepareAfterInferFormatAndShape");
  ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PrepareAfterInsertAipp) {
    EXPECT_EQ(graph->GetAllSubgraphs().size(), 2);  // TODO: other check
  };

  CHECK_GRAPH(PrepareAfterInferFormatAndShape) {
    EXPECT_EQ(graph->GetAllSubgraphs().size(), 1);  // TODO: other check
  };
};

TEST_F(TestAoe, test_build_mode_build_step_with_function_node_case_const_input_scalar) {
  std::string func_node_name = "Case_0";
  const auto &root_graph = MakeFunctionGraph1(func_node_name, CASE, true);
  const auto &sub_graph = MakeSubGraph("sub_graph_0/");
  AddPartitionedCall(root_graph, func_node_name, sub_graph);
  const auto &sub_graph_1 = MakeSubGraph("sub_graph_1/");
  AddPartitionedCall(root_graph, func_node_name, sub_graph_1);

  map<std::string, std::string> options;
  options.emplace(BUILD_MODE, BUILD_MODE_TUNING);
  options.emplace(BUILD_STEP, BUILD_STEP_AFTER_BUILDER);
  Status ret = ge::GELib::Initialize(options);
  EXPECT_EQ(ret, SUCCESS);
  Session session(options);
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(root_graph);
  session.AddGraph(1, graph, options);
  std::vector<InputTensorInfo> inputs;
  DUMP_GRAPH_WHEN("PrepareAfterInsertAipp", "PrepareAfterInferFormatAndShape");
  ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PrepareAfterInsertAipp) {
    EXPECT_EQ(graph->GetAllSubgraphs().size(), 2);  // TODO: other check
  };

  CHECK_GRAPH(PrepareAfterInferFormatAndShape) {
    EXPECT_EQ(graph->GetAllSubgraphs().size(), 1);  // TODO: other check
  };
};

TEST_F(TestAoe, test_build_mode_build_step_with_function_if_const_input) {
  std::string func_node_name = "If_0";
  const auto &root_graph = MakeFunctionGraph1(func_node_name, IF);
  const auto &sub_graph = MakeSubGraph("sub_graph_0/");
  AddPartitionedCall(root_graph, func_node_name, sub_graph);
  const auto &sub_graph_1 = MakeSubGraph("sub_graph_1/");
  AddPartitionedCall(root_graph, func_node_name, sub_graph_1);

  map<std::string, std::string> options;
  options.emplace(BUILD_MODE, BUILD_MODE_TUNING);
  options.emplace(BUILD_STEP, BUILD_STEP_AFTER_BUILDER);
  Status ret = ge::GELib::Initialize(options);
  EXPECT_EQ(ret, SUCCESS);
  Session session(options);
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(root_graph);
  session.AddGraph(1, graph, options);
  std::vector<InputTensorInfo> inputs;
  DUMP_GRAPH_WHEN("PrepareAfterInsertAipp", "PrepareAfterInferFormatAndShape");
  ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PrepareAfterInsertAipp) {
    EXPECT_EQ(graph->GetAllSubgraphs().size(), 2);  // TODO: other check
  };

  CHECK_GRAPH(PrepareAfterInferFormatAndShape) {
    EXPECT_EQ(graph->GetAllSubgraphs().size(), 1);  // TODO: other check
  };
};

TEST_F(TestAoe, test_build_mode_build_step_with_function_node2) {
  std::string func_node_name = "Case_0";
  const auto &root_graph = MakeFunctionGraph2(func_node_name, CASE);
  const auto &sub_graph = MakeSubGraph("sub_graph_0/");
  AddPartitionedCall(root_graph, func_node_name, sub_graph);
  const auto &sub_graph_1 = MakeSubGraph("sub_graph_1/");
  AddPartitionedCall(root_graph, func_node_name, sub_graph_1);

  map<std::string, std::string> options;
  options.emplace(BUILD_MODE, BUILD_MODE_TUNING);
  options.emplace(BUILD_STEP, BUILD_STEP_AFTER_BUILDER);
  Status ret = ge::GELib::Initialize(options);
  EXPECT_EQ(ret, SUCCESS);
  Session session(options);
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(root_graph);
  session.AddGraph(1, graph, options);
  std::vector<InputTensorInfo> inputs;
  DUMP_GRAPH_WHEN("PrepareAfterInsertAipp", "PrepareAfterInferFormatAndShape");
  ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PrepareAfterInsertAipp) {
    EXPECT_EQ(graph->GetAllSubgraphs().size(), 2);  // TODO: other check
  };

  CHECK_GRAPH(PrepareAfterInferFormatAndShape) {
    EXPECT_EQ(graph->GetAllSubgraphs().size(), 2);  // TODO: other check
  };
};

TEST_F(TestAoe, test_build_mode_build_step_normal) {
  const auto &root_graph = MakeGraph();
  map<std::string, std::string> options;
  options.emplace(BUILD_MODE, BUILD_MODE_TUNING);
  options.emplace(BUILD_STEP, BUILD_STEP_AFTER_BUILDER);
  Status ret = ge::GELib::Initialize(options);
  EXPECT_EQ(ret, SUCCESS);
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(root_graph);
  Session session(options);
  session.AddGraphWithCopy(1, graph);
  std::vector<InputTensorInfo> inputs;
  ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);
};
}  // namespace ge

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

#include "framework/common/ge_types.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "ge_graph_dsl/assert/graph_assert.h"
#include "ge/ge_api.h"
#include "graph/ge_local_context.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/op_desc_utils.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "ge_running_env/fake_graph_optimizer.h"
#include "ge_running_env/fake_op.h"
#include "common/share_graph.h"
#include "ge_running_env/tensor_utils.h"
#include "register/optimization_option_registry.h"
#include "init_ge.h"

namespace ge {
/** 当前测试框架尚未完全支持通过 BuildGraph 接口编译、加载图, 部分算子走动态 shape 会找不到 converter
 *  因此部分用例暂时校验为失败，只校验编译流程
 */
class OptimizationOptionSt : public testing::Test {
 protected:
  void SetUp() override {
    ge::OmgContext ctx;
    domi::GetContext() = ctx;
    ReInitGe();
    global_options_ = ge::GetThreadLocalContext().GetAllGlobalOptions();
    graph_options_ = ge::GetThreadLocalContext().GetAllGraphOptions();
    session_options_ = ge::GetThreadLocalContext().GetAllSessionOptions();
    ge::GetThreadLocalContext().SetGlobalOption({});
    ge::GetThreadLocalContext().SetGraphOption({});
    ge::GetThreadLocalContext().SetSessionOption({});
    GetThreadLocalContext().GetOo().Initialize({}, {});
    ge_env.InstallDefault();
  }
  void TearDown() override {
    ge_env.Reset();
    ge_env.InstallDefault();
    ge::GetThreadLocalContext().SetGlobalOption(global_options_);
    ge::GetThreadLocalContext().SetGraphOption(graph_options_);
    ge::GetThreadLocalContext().SetSessionOption(session_options_);
  }

  GeRunningEnvFaker ge_env;
  std::map<std::string, std::string> graph_options_;
  std::map<std::string, std::string> session_options_;
  std::map<std::string, std::string> global_options_;
};

namespace {
Status DoCompileGraph(const Graph &graph, const uint32_t graph_id, const std::map<AscendString, AscendString> &options) {
  Session session(options);
  auto ret = session.AddGraph(graph_id, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  std::vector<InputTensorInfo> inputs;
  ret = session.BuildGraph(graph_id, inputs);
  return ret;
}

graphStatus InferShapeDefault(Operator &op) {
  (void)op;
  return GRAPH_SUCCESS;
}

graphStatus InferShapeForElementwiseOp(Operator &op) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  if (op.GetOutputsSize() > 0 && op.GetInputsSize() > 0) {
    op_desc->UpdateOutputDesc(0, op_desc->GetInputDesc(0));
  }
  return GRAPH_SUCCESS;
}

graphStatus InferShapeFuncForShape(Operator &op) {
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  auto x_desc = op_desc->MutableInputDesc("x");
  const auto x_dim_num = x_desc->GetShape().GetDimNum();

  auto td = op_desc->MutableOutputDesc("y");
  td->SetShape(ge::GeShape({static_cast<int64_t>(x_dim_num)}));
  td->SetOriginShape(ge::GeShape({static_cast<int64_t>(x_dim_num)}));
  td->SetDataType(DT_INT64);
  td->SetOriginDataType(DT_INT64);
  return GRAPH_SUCCESS;
}

graphStatus InferShapeFuncForReshape(Operator &op) {
  auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  op_desc->SetOpInferDepends({"shape"});
  auto x_desc = op_desc->MutableInputDesc("x");
  auto y_desc = op_desc->MutableOutputDesc("y");
  auto x_shape = vector<int64_t>(x_desc->GetShape().GetDims());
  std::string shape_input_name = "shape";

  ge::Tensor shape_tensor;
  op.GetInputConstData(shape_input_name.c_str(), shape_tensor);

  ge::GeShape output_shape;
  auto shape_tensor_desc = op_desc->MutableInputDesc("shape");
  auto &shape_ref = shape_tensor_desc->MutableShape();
  auto shape_dims = shape_ref.GetDims();

  int64_t dim_num = shape_dims[0];
  const int64_t *shape_data = const_cast<int64_t *>(reinterpret_cast<const int64_t *>(shape_tensor.GetData()));
  std::vector<int64_t> out_dims;
  int64_t product = 1;
  for (int64_t i = 0; i < dim_num; i++) {
    auto dim = shape_data[i];
    if (dim != 0 && product > (INT64_MAX / dim)) {
      return ge::GRAPH_PARAM_INVALID;
    }
    out_dims.push_back(dim);
    product *= dim;
  }

  auto td = op_desc->MutableOutputDesc("y");
  td->SetShape(ge::GeShape(out_dims));
  td->SetOriginShape(ge::GeShape(out_dims));
  td->SetDataType(op_desc->MutableInputDesc("x")->GetDataType());
  td->SetOriginDataType(op_desc->MutableInputDesc("x")->GetDataType());
  return ge::GRAPH_SUCCESS;
}

Graph CreateDynamicDimsGraphWithScalarInput() {
  DEF_GRAPH(test_graph) {
    const auto data0 =
        OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 0).OutCnt(1).TensorDesc(FORMAT_ND, DT_FLOAT, {10}).Build("data0");
    const auto data1 =
        OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 1).OutCnt(1).TensorDesc(FORMAT_ND, DT_FLOAT, {10}).Build("data1");
    const auto data2 =
        OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 2).OutCnt(1).TensorDesc(FORMAT_ND, DT_FLOAT, {}).Build("data2");
    auto add0 = OP_CFG(ADD).InCnt(2).OutCnt(1).TensorDesc(FORMAT_ND, DT_FLOAT, {10}).Build("add0");
    auto mul1 = OP_CFG(MUL).TensorDesc(FORMAT_ND, DT_FLOAT, {10}).InCnt(2).OutCnt(1).Build("mul1");

    auto net_output = OP_CFG(NETOUTPUT).InCnt(2).OutCnt(1).Build("net_output");

    CHAIN(NODE(data0)->EDGE(0, 0)->NODE(add0));
    CHAIN(NODE(data1)->EDGE(0, 1)->NODE(add0));
    CHAIN(NODE(add0)->EDGE(0, 0)->NODE(mul1));
    CHAIN(NODE(data0)->EDGE(0, 1)->NODE(mul1));
    CHAIN(NODE(mul1)->EDGE(0, 0)->NODE(net_output));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(net_output));
    CHAIN(NODE(add0)->CTRL_EDGE()->NODE(net_output));
  };
  return ToGeGraph(test_graph);
}

Graph CreateGraphWithReshape(const std::vector<int64_t> &shape0 = {2, 3, 4, 5},
                             const std::vector<int64_t> &shape1 = {2, 3, 20}) {
  DEF_GRAPH(test_graph) {
    const auto data0 =
        OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 0).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, shape0).Build("data0");
    const auto data1 =
        OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 1).OutCnt(1).TensorDesc(FORMAT_ND, DT_FLOAT, shape1).Build("data1");
    const auto shape_op = OP_CFG(SHAPE).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT64, {}).Build("shape_op");
    const auto reshape_op =
        OP_CFG(RESHAPE).InCnt(2).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {}).Build("reshape_op");
    const auto relu_op = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {}).Build("relu_op");
    const auto net_output =
        OP_CFG(NETOUTPUT).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {}).Build("net_output");

    relu_op->SetOpEngineName(kEngineNameAiCore);
    relu_op->SetOpKernelLibName(kEngineNameAiCore);

    CHAIN(NODE(data0)->EDGE(0, 0)->NODE(reshape_op)->EDGE(0, 0)->NODE(relu_op)->EDGE(0, 0)->NODE(net_output));
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(shape_op)->EDGE(0, 1)->NODE(reshape_op));
  };
  return ToGeGraph(test_graph);
}

Graph CreateGraphWithStaticAndDynamicShape() {
  DEF_GRAPH(test_graph) {
    const auto data0 =
        OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 0).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {-1, 3, 4, 5}).Build("data0");
    const auto data1 =
        OP_CFG(DATA).Attr(ATTR_NAME_INDEX, 1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, 3, 4, 5}).Build("data1");
    const auto relu0 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {}).Build("relu0");
    const auto relu1 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {}).Build("relu1");
    const auto relu2 = OP_CFG(RELU).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {}).Build("relu2");
    const auto cast_op = OP_CFG(CAST).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {}).Build("cast_op");
    const auto abs_op = OP_CFG(ABSVAL).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {}).Build("abs_op");
    const auto cast_op2 = OP_CFG(CAST).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {}).Build("cast_op2");
    const auto net_output = OP_CFG(NETOUTPUT).InCnt(2).TensorDesc(FORMAT_NCHW, DT_FLOAT, {}).Build("net_output");

    CHAIN(NODE(data0)->NODE(relu0)->NODE(relu2)->NODE(net_output));
    CHAIN(NODE(data1)->NODE(relu1)->NODE(cast_op)->NODE(abs_op)->NODE(cast_op2)->EDGE(0, 1)->NODE(net_output));
  };
  return ToGeGraph(test_graph);
}

Graph CreateIfGraph() {
  DEF_GRAPH(then_branch) {
    const auto data_1 = OP_CFG(DATA)
                            .Attr(ATTR_NAME_INDEX, 0)
                            .Attr(ATTR_NAME_PARENT_NODE_INDEX, 1)
                            .InCnt(1)
                            .OutCnt(1)
                            .TensorDesc(FORMAT_NCHW, DT_FLOAT, {})
                            .Build("sub_data1");
    const auto net_output = OP_CFG(NETOUTPUT)
                                .InputAttr(0, ATTR_NAME_PARENT_NODE_INDEX, 0)
                                .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
                                .InCnt(1)
                                .OutCnt(1)
                                .TensorDesc(FORMAT_NCHW, DT_FLOAT, {})
                                .Build("then_output");
    CHAIN(NODE(data_1)->NODE("cast", CAST)->NODE(net_output));
  };
  DEF_GRAPH(else_branch) {
    const auto data_1 = OP_CFG(DATA)
                            .Attr(ATTR_NAME_INDEX, 1)
                            .Attr(ATTR_NAME_PARENT_NODE_INDEX, 1)
                            .InCnt(1)
                            .OutCnt(1)
                            .TensorDesc(FORMAT_NCHW, DT_FLOAT, {})
                            .Build("sub_data1");
    const auto net_output = OP_CFG(NETOUTPUT)
                                .InputAttr(0, ATTR_NAME_PARENT_NODE_INDEX, 0)
                                .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
                                .InCnt(1)
                                .OutCnt(1)
                                .TensorDesc(FORMAT_NCHW, DT_FLOAT, {})
                                .Build("else_output");
    CHAIN(NODE(data_1)->NODE("abs", ABSVAL)->NODE(net_output));
  };
  DEF_GRAPH(if_graph) {
    std::vector<int64_t> shape = {2, 2, 3, 2};
    auto data_tensor = GenerateTensor(shape);
    const auto const_input0 =
        OP_CFG(CONSTANT).OutCnt(1).Weight(data_tensor).TensorDesc(FORMAT_NCHW, DT_FLOAT, shape).Build("const_input0");
    const auto data_input1 = OP_CFG(DATA)
                                 .Attr(ATTR_NAME_INDEX, 0)
                                 .OutCnt(1)
                                 .TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, 3, 4, 5})
                                 .Build("data_input1");
    const auto if_op = OP_CFG(IF).InCnt(2).OutCnt(1);
    CHAIN(NODE(const_input0)->NODE("if_op", if_op, then_branch, else_branch)->NODE("net_output", NETOUTPUT));
    CHAIN(NODE(data_input1)->EDGE(0, 1)->NODE("if_op"));
  };

  return ToGeGraph(if_graph);
}
}  // namespace

/** O1 + 纯静态图 (包含常量折叠)
 *  1. 构造纯静态图并且满足常量折叠条件
 *  2. 构造 Session
 *  3. AddGraph 添加图
 *  4. BuildGraph 编译图
 *  预期结果: 开启 O1 的场景下, 图中 Shape 为静态, 常量折叠生效 (shape 变成 const)
 */
TEST_F(OptimizationOptionSt, CompileWithO1_Ok_StaticGraphConstantFolding) {
  ge_env.Reset()
      .Install(FakeEngine("AIcoreEngine").KernelInfoStore("AIcoreEngine"))
      .Install(FakeOp(DATA).InfoStoreAndBuilder("AIcoreEngine").InferShape(InferShapeForElementwiseOp))
      .Install(FakeOp(CONSTANT).InfoStoreAndBuilder("AIcoreEngine").InferShape(InferShapeForElementwiseOp))
      .Install(FakeOp(RESHAPE).InfoStoreAndBuilder("AIcoreEngine").InferShape(InferShapeFuncForReshape))
      .Install(FakeOp(SHAPE).InfoStoreAndBuilder("AIcoreEngine").InferShape(InferShapeFuncForShape))
      .Install(FakeOp(RELU).InfoStoreAndBuilder("AIcoreEngine").InferShape(InferShapeForElementwiseOp))
      .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("AIcoreEngine").InferShape(InferShapeForElementwiseOp));

  auto graph = CreateGraphWithReshape();
  DUMP_GRAPH_WHEN("Build");
  std::map<AscendString, AscendString> options = {{"ge.oo.level", "O1"}};
  EXPECT_EQ(DoCompileGraph(graph, 1, options), SUCCESS);

  CHECK_GRAPH(Build) {
    // 常量折叠生效
    EXPECT_EQ(graph->FindFirstNodeMatchType(SHAPE), nullptr);
    EXPECT_EQ(graph->FindFirstNodeMatchType(RESHAPE), nullptr);
    EXPECT_NE(graph->FindFirstNodeMatchType(CONSTANT), nullptr);
    EXPECT_EQ(graph->GetDirectNodesSize(), 5UL);
    // 纯静态图
    EXPECT_EQ(GraphUtils::IsUnknownShapeGraph(graph), false);
  };
}

/** O1 + 纯静态图 (关闭常量折叠)
 *  1. 构造纯静态图并且满足常量折叠条件
 *  2. 构造 Session
 *  3. AddGraph 添加图
 *  4. BuildGraph 编译图
 *  预期结果: 开启 O1 的场景下, 图中 Shape 为静态, 常量折叠生效 (shape, reshape还在)
 */
TEST_F(OptimizationOptionSt, CompileWithO1_Ok_StaticGraphDisableConstantFolding) {
  ge_env.Reset()
      .Install(FakeEngine("AIcoreEngine").KernelInfoStore("AIcoreEngine"))
      .Install(FakeOp(DATA).InfoStoreAndBuilder("AIcoreEngine").InferShape(InferShapeForElementwiseOp))
      .Install(FakeOp(CONSTANT).InfoStoreAndBuilder("AIcoreEngine").InferShape(InferShapeForElementwiseOp))
      .Install(FakeOp(SHAPE).InfoStoreAndBuilder("AIcoreEngine").InferShape(InferShapeFuncForShape))
      .Install(FakeOp(RESHAPE).InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(RELU).InfoStoreAndBuilder("AIcoreEngine").InferShape(InferShapeForElementwiseOp))
      .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("AIcoreEngine").InferShape(InferShapeForElementwiseOp));

  auto graph = CreateGraphWithReshape();
  DUMP_GRAPH_WHEN("Build");
  std::map<AscendString, AscendString> options = {{"ge.oo.level", "O1"}, {ge::OO_CONSTANT_FOLDING, "false"}};
  EXPECT_EQ(DoCompileGraph(graph, 1, options), SUCCESS);

  CHECK_GRAPH(Build) {
    // 常量折叠未生效
    EXPECT_NE(graph->FindFirstNodeMatchType(SHAPE), nullptr);
    // EXPECT_NE(graph->FindFirstNodeMatchType(RESHAPE), nullptr);
    EXPECT_EQ(graph->FindFirstNodeMatchType(CONSTANT), nullptr);
    EXPECT_EQ(graph->GetDirectNodesSize(), 5UL);
    // 纯静态图
    EXPECT_EQ(GraphUtils::IsUnknownShapeGraph(graph), false);
  };
}

/** O1 + 纯动态图
 *  1. 构造纯动态图但是不满足常量折叠条件
 *  2. 构造 Session
 *  3. AddGraph 添加图
 *  4. BuildGraph 编译图
 *  预期结果: 开启 O1 的场景下, 图中 Shape 为动态, 常量折叠未生效 (图中无 const)
 */
TEST_F(OptimizationOptionSt, CompileWithO1_Ok_DynamicGraph) {
  ge_env.Reset()
      .Install(FakeEngine("AIcoreEngine").KernelInfoStore("AIcoreEngine"))
      .Install(FakeOp(DATA).InfoStoreAndBuilder("AIcoreEngine").InferShape(InferShapeForElementwiseOp))
      .Install(FakeOp(CONSTANT).InfoStoreAndBuilder("AIcoreEngine").InferShape(InferShapeForElementwiseOp))
      .Install(FakeOp(RESHAPE).InfoStoreAndBuilder("AIcoreEngine").InferShape(InferShapeDefault))
      .Install(FakeOp(SHAPE).InfoStoreAndBuilder("AIcoreEngine").InferShape(InferShapeFuncForShape))
      .Install(FakeOp(RELU).InfoStoreAndBuilder("AIcoreEngine").InferShape(InferShapeForElementwiseOp))
      .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("AIcoreEngine").InferShape(InferShapeForElementwiseOp));

  auto graph = CreateGraphWithReshape({-1, 2, 3, 4}, {-1, 3, 4});
  DUMP_GRAPH_WHEN("PreRunAfterOptimize1");
  std::map<AscendString, AscendString> options = {{"ge.oo.level", "O1"}};
  // 当前测试框架还不支持 BuildGraph 接口端到端编译、加载
  EXPECT_NE(DoCompileGraph(graph, 1, options), SUCCESS);

  CHECK_GRAPH(PreRunAfterOptimize1) {
    // 常量折叠未生效
    EXPECT_NE(graph->FindFirstNodeMatchType(SHAPE), nullptr);
    EXPECT_EQ(graph->FindFirstNodeMatchType(CONSTANT), nullptr);
    EXPECT_EQ(graph->GetDirectNodesSize(), 6UL);
    // 纯动态图
    EXPECT_EQ(GraphUtils::IsUnknownShapeGraph(graph), true);
  };
}

/** O1 + 动静态混合图
 *  1. 构造包含动态 Shape、静态 Shape 算子的图
 *  2. 构造 Session
 *  3. AddGraph 添加图
 *  4. BuildGraph 编译图
 *  预期结果: 开启 O1 的场景下, 图中包含静态子图
 */
TEST_F(OptimizationOptionSt, CompileWithO1_Ok_GraphWithStaticAndDynamicShape) {
  ge_env.Reset()
      .Install(FakeEngine("AIcoreEngine").KernelInfoStore("AIcoreEngine"))
      .Install(FakeOp(DATA).InfoStoreAndBuilder("AIcoreEngine").InferShape(InferShapeForElementwiseOp))
      .Install(FakeOp(CONSTANT).InfoStoreAndBuilder("AIcoreEngine").InferShape(InferShapeForElementwiseOp))
      .Install(FakeOp(RELU).InfoStoreAndBuilder("AIcoreEngine").InferShape(InferShapeForElementwiseOp))
      .Install(FakeOp(CAST).InfoStoreAndBuilder("AIcoreEngine").InferShape(InferShapeForElementwiseOp))
      .Install(FakeOp(ABSVAL).InfoStoreAndBuilder("AIcoreEngine").InferShape(InferShapeForElementwiseOp))
      .Install(FakeOp(PARTITIONEDCALL).InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("AIcoreEngine").InferShape(InferShapeForElementwiseOp));

  auto graph = CreateGraphWithStaticAndDynamicShape();
  DUMP_GRAPH_WHEN("Build");
  std::map<AscendString, AscendString> options = {{"ge.oo.level", "O1"}};
  // 当前测试框架还不支持 BuildGraph 接口端到端编译、加载
  EXPECT_NE(DoCompileGraph(graph, 1, options), SUCCESS);

  CHECK_GRAPH(Build) {
    EXPECT_EQ(graph->GetDirectNodesSize(), 5UL);
    const auto subgraphs = graph->GetAllSubgraphs();
    EXPECT_EQ(subgraphs.size(), 2UL);
    size_t static_subg_num = 0UL;
    size_t dynamic_subg_num = 0UL;
    for (const auto &subg : subgraphs) {
      if (subg->GetGraphUnknownFlag()) {
        // known shape subgraph
        ++dynamic_subg_num;
      } else {
        ++static_subg_num;
      }
    }
    EXPECT_GT(static_subg_num, 0UL);
    EXPECT_GT(dynamic_subg_num, 0UL);
  };
}

/** O1 + 死边消除场景
 *  1. 构造包含 if 算子的图
 *  2. 构造 Session
 *  3. AddGraph 添加图
 *  4. BuildGraph 编译图
 *  预期结果: 开启 O1 的场景下, 图中无 if, 仅有 partitioned_call
 */
TEST_F(OptimizationOptionSt, CompileWithO1_Ok_IfGraphDeadCodeElimination) {
  auto graph = CreateIfGraph();
  DUMP_GRAPH_WHEN("Build");
  std::map<AscendString, AscendString> options = {{"ge.oo.level", "O1"}};
  // 当前测试框架还不支持 BuildGraph 接口端到端编译、加载
  EXPECT_EQ(DoCompileGraph(graph, 1, options), SUCCESS);

  CHECK_GRAPH(Build) {
    EXPECT_EQ(graph->FindFirstNodeMatchType(IF), nullptr);
    EXPECT_NE(graph->FindFirstNodeMatchType(PARTITIONEDCALL), nullptr);
  };
}

/** O1 + 关闭死边消除场景
 *  1. 构造包含 if 算子的图
 *  2. 构造 Session
 *  3. AddGraph 添加图
 *  4. BuildGraph 编译图
 *  预期结果: 开启 O1、"dead_code_elimination = false" 的场景下, 图中有 if
 */
TEST_F(OptimizationOptionSt, CompileWithO1_Ok_IfGraphDisableDeadCodeElimination) {
  auto graph = CreateIfGraph();
  DUMP_GRAPH_WHEN("PreRunAfterOptimize1");
  std::map<AscendString, AscendString> options = {{"ge.oo.level", "O1"}, {ge::OO_DEAD_CODE_ELIMINATION, "false"}};
  // 当前测试框架还不支持 BuildGraph 接口端到端编译、加载
  EXPECT_NE(DoCompileGraph(graph, 1, options), SUCCESS);

  CHECK_GRAPH(PreRunAfterOptimize1) {
    EXPECT_NE(graph->FindFirstNodeMatchType(IF), nullptr);
    EXPECT_EQ(graph->FindFirstNodeMatchType(PARTITIONEDCALL), nullptr);
  };
}

/** O1 + 关闭死边消除场景
 *  1. 构造包含 switch 算子的图
 *  2. 构造 Session
 *  3. AddGraph 添加图
 *  4. BuildGraph 编译图
 *  预期结果: 开启 O1、"dead_code_elimination = false" 的场景下, 图中有 switch, merge
 */
TEST_F(OptimizationOptionSt, CompileWithO1_Ok_SwitchMergeGraphDisableDeadCodeElimination) {
  auto graph = gert::ShareGraph::BuildSwitchMergeGraph();
  DUMP_GRAPH_WHEN("Build");
  std::map<AscendString, AscendString> options = {{"ge.oo.level", "O1"}, {ge::OO_DEAD_CODE_ELIMINATION, "false"}};
  EXPECT_EQ(DoCompileGraph(graph, 1, options), SUCCESS);

  CHECK_GRAPH(Build) {
    EXPECT_NE(graph->FindFirstNodeMatchType(STREAMSWITCH), nullptr);
    EXPECT_NE(graph->FindFirstNodeMatchType(STREAMMERGE), nullptr);
    EXPECT_EQ(graph->FindFirstNodeMatchType(PARTITIONEDCALL), nullptr);
  };
}

/** O1 + DFS拓扑排序 兼容性测试
 *  1. 构造 Session 和 Compute Graph
 *  2. AddGraph 添加图, 并传入 O1 优化选项, 配置 DFS 拓扑序, 关闭常量折叠
 *  3. BuildGraph 编译图
 *  预期结果: 符合 DFS 拓扑序
 */
TEST_F(OptimizationOptionSt, CompileWithO1_Ok_TopoSortDisableConstantFoldingCompatibility) {
  ge_env.Reset()
      .Install(FakeEngine("AIcoreEngine").KernelInfoStore("AIcoreEngine"))
      .Install(FakeOp(DATA).InfoStoreAndBuilder("AIcoreEngine").InferShape(InferShapeForElementwiseOp))
      .Install(FakeOp(CONSTANT).InfoStoreAndBuilder("AIcoreEngine").InferShape(InferShapeForElementwiseOp))
      .Install(FakeOp(RELU).InfoStoreAndBuilder("AIcoreEngine").InferShape(InferShapeForElementwiseOp))
      .Install(FakeOp(CAST).InfoStoreAndBuilder("AIcoreEngine").InferShape(InferShapeForElementwiseOp))
      .Install(FakeOp(ABSVAL).InfoStoreAndBuilder("AIcoreEngine").InferShape(InferShapeForElementwiseOp))
      .Install(FakeOp(PARTITIONEDCALL).InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("AIcoreEngine").InferShape(InferShapeForElementwiseOp));

  DUMP_GRAPH_WHEN("Build");
  auto graph = CreateGraphWithStaticAndDynamicShape();
  std::map<AscendString, AscendString> options = {
      {"ge.oo.level", "O1"}, {ge::OO_CONSTANT_FOLDING, "false"}, {"ge.topoSortingMode", "1"}};
  EXPECT_NE(DoCompileGraph(graph, 1, options), SUCCESS);

  CHECK_GRAPH(Build) {
    std::vector<std::string> expect_node_names = {"data0",   "relu0",  "relu2",    "data1",    "relu1",
                                                  "cast_op", "abs_op", "cast_op2", "net_output"};
    std::vector<std::string> node_names;
    for (const auto &node : graph->GetAllNodes()) {
      const auto &name = node->GetName();
      if (name.find("test_graph") != 0) {
        node_names.emplace_back(name);
      }
    }
    EXPECT_EQ(node_names, expect_node_names);
  };
}

/** O1 + 默认拓扑排序 兼容性测试
 *  1. 构造 Session 和 Compute Graph
 *  2. AddGraph 添加图, 并传入 O1 优化选项
 *  3. BuildGraph 编译图
 *  预期结果: 符合 DFS 拓扑序
 */
TEST_F(OptimizationOptionSt, CompileWithO1_Ok_DefaultTopoSortCompatibility) {
  ge_env.Reset()
      .Install(FakeEngine("AIcoreEngine").KernelInfoStore("AIcoreEngine"))
      .Install(FakeOp(DATA).InfoStoreAndBuilder("AIcoreEngine").InferShape(InferShapeForElementwiseOp))
      .Install(FakeOp(CONSTANT).InfoStoreAndBuilder("AIcoreEngine").InferShape(InferShapeForElementwiseOp))
      .Install(FakeOp(RELU).InfoStoreAndBuilder("AIcoreEngine").InferShape(InferShapeForElementwiseOp))
      .Install(FakeOp(CAST).InfoStoreAndBuilder("AIcoreEngine").InferShape(InferShapeForElementwiseOp))
      .Install(FakeOp(ABSVAL).InfoStoreAndBuilder("AIcoreEngine").InferShape(InferShapeForElementwiseOp))
      .Install(FakeOp(PARTITIONEDCALL).InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("AIcoreEngine").InferShape(InferShapeForElementwiseOp));

  DUMP_GRAPH_WHEN("Build");
  auto graph = CreateGraphWithStaticAndDynamicShape();
  std::map<AscendString, AscendString> options = {
      {"ge.oo.level", "O1"}};
  EXPECT_NE(DoCompileGraph(graph, 1, options), SUCCESS);

  CHECK_GRAPH(Build) {
    std::vector<std::string> expect_node_names = {"data1", "data0", "test_graph_sub_0_know_PartitionedCall_0",
                                                  "test_graph_sub_1_unknow_PartitionedCall_1", "net_output"};
    std::vector<std::string> node_names;
    for (const auto &node : graph->GetDirectNode()) {
      node_names.emplace_back(node->GetName());
    }
    EXPECT_EQ(node_names, expect_node_names);
  };
}

/** O1 + 动态分档 兼容性测试
 *  1. 构造 Session 和 Compute Graph
 *  2. AddGraph 添加图, 并传入 O1 优化选项和分档配置
 *  3. BuildGraph 编译图
 *  预期结果: 分档构图正常
 */
TEST_F(OptimizationOptionSt, CompileWithO1_Ok_DynamicDimensionCompatibility) {
  auto multi_dims = MakeShared<FakeMultiDimsOptimizer>();
  ge_env.Install(FakeEngine("AIcoreEngine").GraphOptimizer("MultiDims", multi_dims));

  auto graph = CreateDynamicDimsGraphWithScalarInput();
  DUMP_GRAPH_WHEN("PreRunAfterOptimize1");
  std::map<AscendString, AscendString> options = {{"ge.inputShape", "data0:-1;data1:-1;data2:"},
                                             {"ge.dynamicDims", "10,10;2,2;4,4"},
                                             {"ge.dynamicNodeType", "1"},
                                             {"ge.runFlag", "0"},
                                             {"ge.oo.level", "O1"}};
  EXPECT_EQ(DoCompileGraph(graph, 100, options), SUCCESS);
  // check result
  CHECK_GRAPH(PreRunAfterOptimize1) {
    EXPECT_EQ(graph->GetAllSubgraphs().size(), 3);
    EXPECT_EQ(graph->GetDirectNodesSize(), 8);
  };
}

/*
 * 用例描述: 使用graph级别接口和ge.outputDatatype指定整网输出类型成功编译图

 *
 * 测试步骤：
 * 1. ir构造计算图
 * 2. 设option参数:
 *    OPTION_GRAPH_RUN_MODE=0
 *    OUTPUT_TYPE="INT8"
 * 3. 模型编译
 *
 * 预期结果：
 * 1. 模型执行成功
 * 2. dump的buid图中NetOutput节点类型为DT_INT8
 */
TEST_F(OptimizationOptionSt, CompileWithO1_Ok_SetDataTypeOption) {
  ge_env.Reset()
      .Install(FakeEngine("AIcoreEngine").KernelInfoStore("AIcoreEngine"))
      .Install(FakeOp(DATA).InfoStoreAndBuilder("AIcoreEngine").InferShape(InferShapeForElementwiseOp))
      .Install(FakeOp(CONSTANT).InfoStoreAndBuilder("AIcoreEngine").InferShape(InferShapeForElementwiseOp))
      .Install(FakeOp(SHAPE).InfoStoreAndBuilder("AIcoreEngine").InferShape(InferShapeFuncForShape))
      .Install(FakeOp(RESHAPE).InfoStoreAndBuilder("AIcoreEngine"))
      .Install(FakeOp(RELU).InfoStoreAndBuilder("AIcoreEngine").InferShape(InferShapeForElementwiseOp))
      .Install(FakeOp(NETOUTPUT).InfoStoreAndBuilder("AIcoreEngine").InferShape(InferShapeForElementwiseOp));

  auto graph = CreateGraphWithReshape();
  DUMP_GRAPH_WHEN("Build");
  std::map<AscendString, AscendString> options;
  options.emplace(ge::OPTION_GRAPH_RUN_MODE, "0");
  Session session(options);
  std::map<ge::AscendString, ge::AscendString> build_options = {
    {ge::AscendString(ge::ir_option::OUTPUT_TYPE), ge::AscendString("INT8")}
  };
  auto ret = session.AddGraph(1, graph, build_options);
  EXPECT_EQ(ret, SUCCESS);
  std::vector<InputTensorInfo> inputs;
  ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);

  CHECK_GRAPH(Build) {
    auto net_out_node = graph->FindNode("net_output");
    auto output_desc = net_out_node->GetOpDesc();
    EXPECT_EQ(output_desc->MutableInputDesc(0)->GetDataType(), DT_INT8);
  };
}
}  // namespace ge
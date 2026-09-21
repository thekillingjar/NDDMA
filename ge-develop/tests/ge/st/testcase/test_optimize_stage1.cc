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
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/node_adapter.h"
#include "framework/common/types.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/operator.h"
#include "graph/operator_factory.h"
#include "graph/graph.h"
#include "graph/tuning_utils.h"
#include "api/gelib/gelib.h"

#include "common/summary_checker.h"
#include "common/topo_checker.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "ge_graph_dsl/assert/check_utils.h"
#include "ge_graph_dsl/assert/graph_assert.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "ge_running_env/tensor_utils.h"

using namespace std;
namespace ge {
namespace {
// only support shape_data_type int64
const auto StubInferShapeV1ForReshape = [](Operator &op) {
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
  // stub shape data type int64
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
};
/**
┌────────────┐  (0,1)   ┌───────────┐  (0,0)   ┌────────┐  (0,0)   ┌────────────┐  (0,0)   ┌────────┐  (0,0)   ┌────────────┐  (0,0)   ┌───────────┐
│ constant_0 │ ───────> │ reshape_0 │ ───────> │ add_1  │ ───────> │ reshape_1  │ ───────> │ add_2  │ ───────> │ reshape_2  │ ───────> │ netoutput │
└────────────┘          └───────────┘          └────────┘          └────────────┘          └────────┘          └────────────┘          └───────────┘
                          ∧                      ∧                   ∧                       ∧                   ∧
                          │ (0,0)                │ (0,1)             │ (0,1)                 │ (0,1)             │ (0,1)
                          │                      │                   │                       │                   │
┌────────────┐  (0,1)   ┌───────────┐          ┌────────┐          ┌────────────┐          ┌────────┐          ┌────────────┐
│   data_1   │ ───────> │    add    │          │ data_2 │          │ constant_1 │          │ data_3 │          │ constant_2 │
└────────────┘          └───────────┘          └────────┘          └────────────┘          └────────┘          └────────────┘
                          ∧
                          │ (0,0)
                          │
                        ┌───────────┐
                        │  data_0   │
                        └───────────┘
 */
Graph BuildReshapeRecoveryGraph() {
  auto data_0 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 4, 4});
  auto data_1 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 5, 4, 4});
  auto add_0 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 5, 4, 4});

  std::vector<int64_t> const_data_0{1, 20, 4};
  GeTensorDesc const_tensor_desc_0(GeShape(vector<int64_t>{3}), FORMAT_NCHW, DT_INT64);
  GeTensorPtr const_tensor_0 = std::make_shared<GeTensor>(
      const_tensor_desc_0, reinterpret_cast<uint8_t *>(const_data_0.data()), sizeof(int64_t) * const_data_0.size());
  auto constant_0 = OP_CFG(CONSTANT).TensorDesc(FORMAT_NCHW, DT_INT64, {4}).Weight(const_tensor_0);
  auto reshape_0 = OP_CFG(RESHAPE).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 20, 4});

  auto data_2 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 20, 4});
  auto add_1 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 20, 4});

  std::vector<int64_t> const_data_1{1, 5, 4, 4};
  GeTensorDesc const_tensor_desc_1(GeShape(vector<int64_t>{4}), FORMAT_NCHW, DT_INT64);
  GeTensorPtr const_tensor_1 = std::make_shared<GeTensor>(
      const_tensor_desc_1, reinterpret_cast<uint8_t *>(const_data_1.data()), sizeof(int64_t) * const_data_1.size());
  auto constant_1 = OP_CFG(CONSTANT).TensorDesc(FORMAT_NCHW, DT_INT64, {4}).Weight(const_tensor_1);
  auto reshape_1 = OP_CFG(RESHAPE).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 5, 4, 4});

  auto data_3 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 5, 4, 4});
  auto add_2 = OP_CFG(ADD).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 5, 4, 4});

  GeTensorDesc const_tensor_desc_2(GeShape(vector<int64_t>{3}), FORMAT_NCHW, DT_INT64);
  GeTensorPtr const_tensor_2 = std::make_shared<GeTensor>(
      const_tensor_desc_2, reinterpret_cast<uint8_t *>(const_data_0.data()), sizeof(int64_t) * const_data_0.size());
  auto constant_2 = OP_CFG(CONSTANT).TensorDesc(FORMAT_NCHW, DT_INT64, {3}).Weight(const_tensor_2);
  auto reshape_2 = OP_CFG(RESHAPE).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 20, 4});

  DEF_GRAPH(g) {
                 CHAIN(NODE("data_0", data_0)->EDGE(0, 0)->NODE("add", add_0));
                 CHAIN(NODE("data_1", data_1)->EDGE(0, 1)->NODE("add", add_0));

                 CHAIN(NODE("add", add_0)->EDGE(0, 0)->NODE("reshape_0", reshape_0));
                 CHAIN(NODE("constant_0", constant_0)->EDGE(0, 1)->NODE("reshape_0", reshape_0));

                 CHAIN(NODE("reshape_0", reshape_0)->EDGE(0, 0)->NODE("add_1", add_1));
                 CHAIN(NODE("data_2", data_2)->EDGE(0, 1)->NODE("add_1", add_1));

                 CHAIN(NODE("add_1", add_1)->EDGE(0, 0)->NODE("reshape_1", reshape_1));
                 CHAIN(NODE("constant_1", constant_1)->EDGE(0, 1)->NODE("reshape_1", reshape_1));

                 CHAIN(NODE("reshape_1", reshape_1)->EDGE(0, 0)->NODE("add_2", add_2));
                 CHAIN(NODE("data_3", data_3)->EDGE(0, 1)->NODE("add_2", add_2));

                 CHAIN(NODE("add_2", add_2)->EDGE(0, 0)->NODE("reshape_2", reshape_2));
                 CHAIN(NODE("constant_2", constant_2)->EDGE(0, 1)->NODE("reshape_2", reshape_2));

                 CHAIN(NODE("reshape_2", reshape_2)->EDGE(0, 0)->NODE("netoutput", NETOUTPUT));
               };

  auto graph = ToGeGraph(g);
  const auto &compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  for (auto &node : compute_graph->GetDirectNode()) {
    if (node->GetType() == RESHAPE) {
      node->GetOpDesc()->AddInferFunc(StubInferShapeV1ForReshape);
    }
  }
  return graph;
}
}  // namespace
class OptimizeStage1SystemTest : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
 public:
  static void EXPECT_TestReshapeRecoveryPass(const map<AscendString, AscendString> &options = {}) {
    // build graph
    Graph graph = BuildReshapeRecoveryGraph();
    auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
    EXPECT_EQ(gert::SummaryChecker(compute_graph).StrictDirectNodeTypes(
              {{"Reshape", 3},
              {"Const", 3},
              {"Data", 4},
              {"Add", 3},
              {"NetOutput", 1}}),
              "success");
    // new session & add graph
    Session session(options);
    uint32_t graph_id = 1;
    auto ret = session.AddGraph(graph_id, graph, options);
    EXPECT_EQ(ret, SUCCESS);
    // build input tensor
    std::vector<InputTensorInfo> inputs;
    // build_graph through session
    ret = session.BuildGraph(graph_id, inputs); // we only care about compile stage
    EXPECT_EQ(ret, SUCCESS);
    CHECK_GRAPH(PreRunAfterOptimize1) {
      EXPECT_EQ(gert::SummaryChecker(graph).StrictDirectNodeTypes(
                {{"Reshape", 3},
                {"Const", 2},
                {"Data", 4},
                {"Add", 3},
                {"NetOutput", 1}}),
                "success");
      for (const auto &node : graph->GetDirectNode()) {
        if (node->GetType() == "Reshape") {
          EXPECT_EQ(gert::NodeTopoChecker(node).StrictConnectFrom({{"Add"}, {"Const"}}), "success");
        }
      }
    };
  }
};

/**预置条件: DUMP_GRAPH_WHEN("PreRunAfterOptimize1");
 * 输入: 默认空option
 * 执行步骤：构图
 *          AddGraph
 *          BuildGraph
 * 预期结果：编译成功
 *          Reshape和Const节点数量分别为3、2
 *          Reshape节点均有const连边
 * */
TEST_F(OptimizeStage1SystemTest, TestReshapeRecoveryPass_CheckReshapeAndConstTopoSuccess) {
  DUMP_GRAPH_WHEN("PreRunAfterOptimize1");
  EXPECT_TestReshapeRecoveryPass();
}

/**预置条件: DUMP_GRAPH_WHEN("PreRunAfterOptimize1");
 * 输入: 默认空option
 * 执行步骤：构图
 *          AddGraph
 *          BuildGraph
 *          重复执行上述步骤1次
 * 预期结果：编译成功
 *          Reshape和Const节点数量分别为3、2
 *          Reshape节点均有const连边
 *          2次编译出来的计算图一致
 * */
TEST_F(OptimizeStage1SystemTest, TestReshapeRecoveryPass_GetTheSameOM_Build2Times) {
  DUMP_GRAPH_WHEN("PreRunAfterOptimize1");
  EXPECT_TestReshapeRecoveryPass();
  EXPECT_TestReshapeRecoveryPass();
}

/**预置条件: DUMP_GRAPH_WHEN("PreRunAfterOptimize1");
 * 输入: session级别option: ge.disableOptimizations="RemoveSameConstPass";
 * 执行步骤：构图
 *          AddGraph
 *          BuildGraph
 * 预期结果：编译成功
 *          Reshape和Const节点数量分别为3、2
 *          Reshape节点均有const连边
 * */
TEST_F(OptimizeStage1SystemTest, TestReshapeRecoveryPass_CloseRemoveSameConstPass_CheckReshapeAndConstTopoSuccess) {
  DUMP_GRAPH_WHEN("PreRunAfterOptimize1");
  map<AscendString, AscendString> options;
  options[OPTION_DISABLE_OPTIMIZATIONS] = "RemoveSameConstPass";
  EXPECT_TestReshapeRecoveryPass(options);
}
}  // namespace ge

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
#include <iostream>

#include "macro_utils/dt_public_scope.h"
#include "graph/graph.h"
#include "graph/model.h"
#include "graph/utils/tensor_utils.h"
#include "ops_stub.h"
#include "macro_utils/dt_public_unscope.h"

using namespace std;
using namespace ge;

class UtestGeOperatorReg : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestGeOperatorReg, ge_test_operator_reg_test) {
  TensorDesc desc(ge::Shape({1, 3, 224, 224}));
  uint32_t size = desc.GetShape().GetShapeSize();
  desc.SetSize(size);
  auto data = op::Data("Data").set_attr_index(0);
  data.update_input_desc_data(desc);
  data.update_output_desc_out(desc);

  auto flatten = op::Flatten("Flatten").set_input_x(data, data.name_out_out());

  std::vector<Operator> inputs{data};
  std::vector<Operator> outputs{flatten};
  std::vector<Operator> targets{flatten};

  Graph graph("test_graph");
  graph.SetInputs(inputs).SetOutputs(outputs).SetTargets(targets);
  EXPECT_EQ(true, graph.IsValid());

  Graph graph1("test_graph1");
  auto data1 = op::Data("Data1").set_attr_index(0);
  data1.update_input_desc_data(desc);
  data1.update_output_desc_out(desc);
  std::vector<Operator> targets1{data1};
  graph1.SetInputs(inputs).SetTargets(targets1);
}

TEST_F(UtestGeOperatorReg, test_set_outputs) {
  TensorDesc desc(ge::Shape({1, 3, 224, 224}));
  uint32_t size = desc.GetShape().GetShapeSize();
  desc.SetSize(size);
  auto data = op::Data("Data").set_attr_index(0);
  data.update_input_desc_data(desc);
  data.update_output_desc_out(desc);

  auto flatten = op::Flatten("Flatten").set_input_x(data, data.name_out_out());
  std::vector<Operator> inputs{data};
  std::vector<Operator> targets{flatten};
  std::vector<pair<Operator, string>> outputs{{flatten, "Flattern"}};

  Graph graph("test_graph");
  graph.SetInputs(inputs).SetOutputs(outputs).SetTargets(targets);
  EXPECT_EQ(true, graph.IsValid());
}

TEST_F(UtestGeOperatorReg, test_setoutputs_node_not_exist) {
  TensorDesc desc(ge::Shape({1, 3, 224, 224}));
  uint32_t size = desc.GetShape().GetShapeSize();
  desc.SetSize(size);
  auto data0 = op::Data("Data0").set_attr_index(0);
  data0.update_input_desc_data(desc);
  data0.update_output_desc_out(desc);

  auto data1 = op::Data("Data1").set_attr_index(0);
  data1.update_input_desc_data(desc);
  data1.update_output_desc_out(desc);

  std::vector<Operator> inputs{data0};
  std::vector<Operator> outputs{data1};

  Graph graph("test_graph");
  EXPECT_NO_THROW(graph.SetInputs(inputs).SetOutputs(outputs));
}

bool buildGraph1(Graph &graph) {
  auto data = op::Data("data").set_attr_index(0);
  graphStatus ret = graph.AddOp(data);
  EXPECT_EQ(GRAPH_SUCCESS, ret);

  auto flatten = op::Flatten("flatten").set_input_x(data);
  ret = graph.AddOp(flatten);
  EXPECT_EQ(GRAPH_SUCCESS, ret);

  return true;
}

TEST_F(UtestGeOperatorReg, test_add_op) {
  Graph graph("simpleGraph");
  bool ret_graph = buildGraph1(graph);
  EXPECT_EQ(ret_graph, true);

  std::vector<string> op_name;
  graphStatus ret = graph.GetAllOpName(op_name);
  EXPECT_EQ(GRAPH_SUCCESS, ret);
  for (unsigned int i = 0; i < op_name.size(); i++) {
    std::cout << "opname: " << op_name[i] << std::endl;
  }
  EXPECT_EQ(op_name.size(), 2);

  Operator op;
  ret = graph.FindOpByName("dat", op);
  EXPECT_EQ(ret, GRAPH_FAILED);
  ret = graph.FindOpByName("data", op);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  ret = graph.FindOpByName("flatten", op);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  Operator data_op;
  (void)graph.FindOpByName("data", data_op);
  Operator f_op;
  (void)graph.FindOpByName("flatten", f_op);
  data_op.GetOutputsSize();
  std::vector<Operator> inputs{data_op};
  std::vector<Operator> outputs{f_op};
  graph.SetInputs(inputs).SetOutputs(outputs);
}

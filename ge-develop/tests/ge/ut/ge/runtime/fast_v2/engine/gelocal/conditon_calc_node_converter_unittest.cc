/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "engine/gelocal/condition_calc_converter.h"

#include <gtest/gtest.h>

#include "ge_graph_dsl/graph_dsl.h"
#include "engine/gelocal/inputs_converter.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "faker/global_data_faker.h"
#include "common/bg_test.h"
#include "common/topo_checker.h"
#include "lowering/graph_converter.h"

#include "register/op_impl_registry.h"
#include "exe_graph/lowering/lowering_definitions.h"
#include "framework/common/types.h"
#include "graph/operator_reg.h"
#include "graph/operator.h"
#include "array_ops.h"
#include "elewise_calculation_ops.h"
#include "framework/common/ge_types.h"
#include "common/share_graph.h"

using namespace ge;
namespace ge {
REG_OP(ConditionCalc)
    .DYNAMIC_INPUT(input, TensorType::ALL())
    .OUTPUT(cond, TensorType({DT_INT32}))
    .REQUIRED_ATTR(cond_func, String)
    .REQUIRED_ATTR(x_dependency, ListInt)
    .OP_END_FACTORY_REG(ConditionCalc)
}

namespace gert {
IMPL_OP(NetOutput);

class ConditionCalcNodeConverterUT : public bg::BgTestAutoCreateFrame {};
TEST_F(ConditionCalcNodeConverterUT, ConvertConditionCalcNode) {
  Create3StageFrames();
  auto dynamic_input0 = ge::op::Data("dynamic_input0").set_attr_index(0);
  auto dynamic_input1 = ge::op::Data("dynamic_input1").set_attr_index(1);
  auto condition_calc = ge::op::ConditionCalc("condition_calc")
                            .create_dynamic_input_input(2)
                            .set_dynamic_input_input(0, dynamic_input0)
                            .set_dynamic_input_input(1, dynamic_input1)
                            .set_attr_cond_func("cond_func")
                            .set_attr_x_dependency({0, 0});
  std::vector<std::pair<ge::Operator, std::vector<size_t>>> outputs;
  outputs.push_back(std::pair<ge::Operator, std::vector<size_t>>(condition_calc, {0}));
  auto graph = ge::Graph("main").SetInputs({dynamic_input0, dynamic_input1}).SetOutputs(outputs);
  auto main_graph = ge::GraphUtilsEx::GetComputeGraph(graph);

  auto root_model = GeModelBuilder(main_graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  LowerInput data_input = {{}, {}, &global_data};
  LowerInput condition_calc_input = {{}, {}, &global_data};
  for (auto &node : main_graph->GetDirectNode()) {
    if (node->GetType() == ge::DATA) {
      auto lr = LoweringDataNode(node, data_input);
      EXPECT_TRUE(lr.result.IsSuccess());
      condition_calc_input.input_shapes.push_back(lr.out_shapes[0]);
      condition_calc_input.input_addrs.push_back(lr.out_addrs[0]);
    }
  }

  auto condition_calc_node = main_graph->FindFirstNodeMatchType("ConditionCalc");
  auto condition_calc_lower_result = LoweringConditionCalc(condition_calc_node, condition_calc_input);
  EXPECT_TRUE(condition_calc_lower_result.result.IsSuccess());
  EXPECT_EQ(condition_calc_lower_result.out_addrs.size(), 1);
  EXPECT_EQ(condition_calc_lower_result.out_addrs[0]->GetPlacement(), kOnHost);
  EXPECT_EQ(condition_calc_lower_result.out_addrs[0]->GetFastNode()->GetType(), "BuildUnmanagedTensorData");

  EXPECT_EQ(condition_calc_lower_result.out_shapes.size(), 1);
  EXPECT_EQ(condition_calc_lower_result.out_shapes[0]->GetFastNode()->GetType(), "Const");

  EXPECT_TRUE(condition_calc_lower_result.order_holders.empty());
}

TEST_F(ConditionCalcNodeConverterUT, DataDependentHbmOutput) {
  bg::ValueHolder::PopGraphFrame();
  auto main_graph = ShareGraph::BuildAddConditionCalcGraph();
  main_graph->TopologicalSorting();
  auto root_model = GeModelBuilder(main_graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  auto add_node = main_graph->FindNode("add1");
  SetGraphOutShapeRange(main_graph);
  main_graph->SetGraphUnknownFlag(true);
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(SpaceRegistryFaker().Build());
  auto exe_graph = GraphConverter()
      .SetModelDescHolder(&model_desc_holder)
      .ConvertComputeGraphToExecuteGraph(main_graph, global_data);
  ASSERT_NE(exe_graph, nullptr);

  auto tds = ExecuteGraphUtils::FindNodesByTypeFromAllNodes(exe_graph.get(), "BuildUnmanagedTensorData");
  ASSERT_EQ(tds.size(), 1);
  auto td = tds[0];
  ASSERT_NE(td, nullptr);

  FastNodeTopoChecker checker(td);
  EXPECT_EQ(checker.StrictConnectFrom(std::vector<FastSrcNode>({{"ConditionCalc"}, {"InnerData"}})), "success");
  EXPECT_EQ(checker.InChecker()
                .DataFromByType("ConditionCalc")
                .DataFromByType("BuildTensor")
                .DataFromByType("CopyD2H")
                .Result(),
            "success");

  auto ccs = ExecuteGraphUtils::FindNodesByTypeFromAllNodes(exe_graph.get(), "ConditionCalc");
  ASSERT_EQ(ccs.size(), 1);
  auto cc = ccs[0];
  ASSERT_NE(cc, nullptr);
  EXPECT_EQ(
      FastNodeTopoChecker(cc).StrictConnectFrom({
          {"BuildTensor"},
          {"AllocMemHbm"},
          {"InnerData"}  // FindKernelFunc被折叠到init图了，这里体现为InnerData，下一步checker考虑添加跨越子图查找的方法
      }),
      "success");
}
}  // namespace gert
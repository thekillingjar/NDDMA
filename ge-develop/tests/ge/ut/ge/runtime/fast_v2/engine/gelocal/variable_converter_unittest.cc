/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "engine/gelocal/variable_converter.h"
#include <gtest/gtest.h>
#include <memory>
#include "graph/node.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "kernel/tensor_attr.h"
#include "faker/global_data_faker.h"
#include "common/const_data_helper.h"
#include "common/bg_test.h"
#include "common/topo_checker.h"
#include "common/share_graph.h"
#include "graph_builder/converter_checker.h"
#include "graph_builder/bg_memory.h"
#include "graph_builder/bg_variable.h"
#include "exe_graph/lowering/graph_frame.h"
#include "exe_graph/lowering/value_holder.h"

namespace gert {
using namespace ge;
class VariableConverterUt : public bg::BgTestAutoCreate3StageFrame {
};

#define CHECK_VARIABLE_LOWER_RESULT(lower_result)          \
  do {                                                     \
    ASSERT_TRUE((lower_result).result.IsSuccess());        \
    ASSERT_EQ((lower_result).out_shapes.size(), 1U);       \
    ASSERT_EQ((lower_result).out_addrs.size(), 1U);        \
    ASSERT_TRUE((lower_result).order_holders.empty());     \
    ASSERT_TRUE(IsValidHolder((lower_result).out_shapes)); \
    ASSERT_TRUE(IsValidHolder((lower_result).out_addrs));  \
  } while (false)

TEST_F(VariableConverterUt, LowerSameNameVariableTwice) {
  auto graph = SingleNodeGraphBuilder("g", "Variable").NumInputs(0).NumOutputs(1).Build();
  auto variable = graph->FindFirstNodeMatchType("Variable");
auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  bg::LowerConstDataNode(global_data);

  LowerInput lower_input;
  lower_input.global_data = &global_data;
  auto lower_var1 = LoweringVariable(variable, lower_input);
  CHECK_VARIABLE_LOWER_RESULT(lower_var1);

  ASSERT_EQ(bg::ValueHolder::GetCurrentExecuteGraph()->GetAllNodes().size(), 7U);  // stream + l1_allocator + l2_allocator + id + session + split
  ASSERT_EQ(lower_var1.out_shapes.front()->GetFastNode()->GetType(), "Init");

  // Lower same name variable
  auto lower_var2 = LoweringVariable(variable, lower_input);
  CHECK_VARIABLE_LOWER_RESULT(lower_var2);

  ASSERT_EQ(bg::ValueHolder::GetCurrentExecuteGraph()->GetAllNodes().size(), 7U);  // nodes unchanged as unique var resource
  ASSERT_EQ(lower_var2.out_shapes.front()->GetFastNode()->GetType(), "Init");

  ASSERT_EQ(lower_var1.out_shapes.front()->GetFastNode(), lower_var2.out_shapes.front()->GetFastNode());
}

TEST_F(VariableConverterUt, AllocOutputMemoryForRefVariableNode) {
  auto graph = SingleNodeGraphBuilder("g", "Foo").NumInputs(0).NumOutputs(3).Build();
  auto foo = graph->FindFirstNodeMatchType("Foo");
  auto desc1 = foo->GetOpDesc()->MutableOutputDesc(0);
  ge::AttrUtils::SetStr(desc1, ge::ASSIGN_VAR_NAME, "var_1");
  auto desc2 = foo->GetOpDesc()->MutableOutputDesc(1);
  ge::AttrUtils::SetStr(desc2, ge::REF_VAR_SRC_VAR_NAME, "var_2");

auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  bg::LowerConstDataNode(global_data);

  LowerInput lower_input;
  lower_input.global_data = &global_data;

  size_t output_size = 64;
  auto output_size_holder = bg::ValueHolder::CreateConst(&output_size, sizeof(output_size));
  std::vector<bg::ValueHolderPtr> output_sizes(3, output_size_holder);

  auto allocated_memories = bg::AllocOutputMemory(TensorPlacement::kOnDeviceHbm, foo, output_sizes, global_data);

  ASSERT_EQ(allocated_memories.size(), foo->GetAllOutDataAnchorsSize());
  ASSERT_EQ(allocated_memories[0], bg::GetVariableAddr("var_1", global_data, 0));
  ASSERT_EQ(allocated_memories[1], bg::GetVariableAddr("var_2", global_data, 0));
}

TEST_F(VariableConverterUt, AllocOutputMemoryForRefNode) {
  auto graph = SingleNodeGraphBuilder("g", "Foo").NumInputs(3).NumOutputs(2).Build();
  auto foo = graph->FindFirstNodeMatchType("Foo");
  auto desc1 = foo->GetOpDesc()->MutableOutputDesc(0);
  ge::AttrUtils::SetStr(desc1, ge::ASSIGN_VAR_NAME, "var_1");
  auto desc2 = foo->GetOpDesc()->MutableOutputDesc(1);

  ge::AttrUtils::SetBool(foo->GetOpDesc(), ge::ATTR_NAME_REFERENCE, true);
  foo->GetOpDesc()->MutableInputDesc(1)->SetDataType(ge::DT_UNDEFINED); // Unfed optional input
  foo->GetOpDesc()->MutableInputDesc(1)->SetFormat(FORMAT_RESERVED);
  ASSERT_NE(foo->GetOpDesc()->GetAllInputsSize(), foo->GetOpDesc()->GetInputsSize());

  auto &input_names = foo->GetOpDesc()->MutableAllInputName();
  input_names.clear();
  input_names["x"] = 0;
  input_names["y"] = 1;
  input_names["z"] = 2;
  auto &output_names = foo->GetOpDesc()->MutableAllOutputName();
  output_names.clear();
  output_names["x"] = 0; // Ref from input 0 but also assigned to variable
  output_names["z"] = 1; // Ref from input 2

auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  bg::LowerConstDataNode(global_data);

  LowerInput lower_input;
  lower_input.global_data = &global_data;

  size_t output_size = 64;
  auto output_size_holder = bg::ValueHolder::CreateConst(&output_size, sizeof(output_size));
  std::vector<bg::ValueHolderPtr> output_sizes(2, output_size_holder);

  auto input_addr_0 = bg::DevMemValueHolder::CreateConst(&output_size, sizeof(output_size), 0);
  auto input_addr_1 = bg::DevMemValueHolder::CreateConst(&output_size, sizeof(output_size), 0);
  std::vector<bg::DevMemValueHolderPtr> input_addrs{input_addr_0, input_addr_1};  // Unfed optional input "y"

  auto allocated_memories =
      bg::AllocOutputMemory(TensorPlacement::kOnDeviceHbm, foo, output_sizes, input_addrs, global_data);

  ASSERT_EQ(allocated_memories.size(), foo->GetAllOutDataAnchorsSize());
  ASSERT_EQ(allocated_memories[0], input_addr_0);
  ASSERT_EQ(allocated_memories[1], input_addr_1);
  ASSERT_NE(bg::GetVariableShape("var_1", global_data, 0), nullptr);
}
}  // namespace gert
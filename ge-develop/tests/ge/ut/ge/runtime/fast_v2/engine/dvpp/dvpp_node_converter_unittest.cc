/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "engine/dvpp/converter/dvpp_node_converter.h"
#include <gtest/gtest.h>
#include "ge_graph_dsl/graph_dsl.h"
#include "engine/gelocal/inputs_converter.h"
#include "graph/utils/graph_utils.h"
#include "graph_builder/exe_graph_comparer.h"
#include "graph_builder/value_holder_generator.h"
#include "common/share_graph.h"
#include "faker/global_data_faker.h"
#include "common/bg_test.h"
#include "faker/dvpp_taskdef_faker.h"
#include "framework/common/ge_types.h"
#include "mmpa/mmpa_api.h"
#include "stub/hostcpu_mmpa_stub.h"
#include "common/const_data_helper.h"
#include "graph/utils/graph_dump_utils.h"

using namespace ge;
namespace gert {
class DvppNodeConverterUT : public bg::BgTestAutoCreate3StageFrame {};
TEST_F(DvppNodeConverterUT, ConvertDvppNode) {
  auto graph = ShareGraph::BuildSingleNodeGraph();
  auto add_op_desc = graph->FindNode("add1")->GetOpDesc();
  AttrUtils::SetInt(add_op_desc, ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 4);
  add_op_desc->SetOpKernelLibName(ge::kEngineNameDvpp.c_str());
  DvppTaskDefFaker dvpp_task_def_faker;
auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).AddTaskDef("Add",dvpp_task_def_faker).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};

  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);
  ASSERT_TRUE(data1_ret.result.IsSuccess());
  ASSERT_TRUE(data2_ret.result.IsSuccess());

  LowerInput add_input = {{data1_ret.out_shapes[0], data2_ret.out_shapes[0]},
                          {data1_ret.out_addrs[0], data2_ret.out_addrs[0]},
                          &global_data};
  auto add_ret = LoweringDvppNode(
      graph->FindNode("add1"), add_input);
  ASSERT_TRUE(add_ret.result.IsSuccess());
  ASSERT_EQ(add_ret.out_addrs.size(), 1);
  ASSERT_EQ(add_ret.out_shapes.size(), 1);
  ASSERT_EQ(add_ret.order_holders.size(), 1);

  auto exe_graph = add_ret.out_addrs[0]->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr();
  ASSERT_NE(exe_graph, nullptr);
  // graph compare
  auto execute_graph = bg::ValueHolder::PopGraphFrame(ConvertDevMemValueHoldersToValueHolders(
      add_ret.out_addrs), add_ret.order_holders)->GetExecuteGraph();
  ASSERT_NE(execute_graph, nullptr);
  DumpGraph(execute_graph.get(), "GeneralDvppExe");
}
}  // namespace gert
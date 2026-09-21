/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "engine/aicpu/converter/sequence/split_to_sequence.h"

#include <gtest/gtest.h>
#include "ge_graph_dsl/graph_dsl.h"
#include "engine/gelocal/inputs_converter.h"
#include "common/bg_test.h"
#include "graph/utils/graph_utils.h"
#include "graph_builder/exe_graph_comparer.h"
#include "graph_builder/value_holder_generator.h"
#include "common/share_graph.h"
#include "faker/global_data_faker.h"
#include "common/bg_test.h"
#include "framework/common/ge_types.h"
#include "mmpa/mmpa_api.h"
#include "common/share_graph.h"
#include "register/node_converter_registry.h"
#include "engine/aicpu/graph_builder/bg_launch.h"
#include "faker/global_data_faker.h"
#include "faker/aicpu_taskdef_faker.h"
#include "exe_graph/lowering/frame_selector.h"
#include "common/const_data_helper.h"
#include "graph/utils/graph_dump_utils.h"

using namespace ge;
namespace gert {
class AicpuSequenceConverterUT : public bg::BgTestAutoCreate3StageFrame {};
/*
 *       NetOutput     
 *          |
 *   splitToSequence
 *    /            \
 * data1(x)  data2(split)
 */
TEST_F(AicpuSequenceConverterUT, ConvertSplitToSequenceNode) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("x", "Data")->NODE("splitToSequence", "splitToSequence")->NODE("NetOutput", "NetOutput"));
    CHAIN(NODE("split", "Data")->EDGE(0, 1)->NODE("splitToSequence", "splitToSequence"));
  };

  auto graph = ToComputeGraph(g1);
  auto split_to_sequence_op_desc = graph->FindNode("splitToSequence")->GetOpDesc();
  AttrUtils::SetInt(split_to_sequence_op_desc, "axis", 1);
  AttrUtils::SetInt(split_to_sequence_op_desc, "keep_dims", 1);
auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};

  auto data1 = graph->FindNode("x");
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  auto data1_ret = LoweringDataNode(data1, data_input);

  auto data2 = graph->FindNode("split");
  AttrUtils::SetInt(data2->GetOpDesc(), "index", 1);
  auto data2_ret = LoweringDataNode(data2, data_input);

  ASSERT_TRUE(data1_ret.result.IsSuccess());
  ASSERT_TRUE(data2_ret.result.IsSuccess());

  LowerInput input = {{data1_ret.out_shapes[0], data2_ret.out_shapes[0]},
                          {data1_ret.out_addrs[0], data2_ret.out_addrs[0]},
                          &global_data};
  auto sequence_ret = LoweringSplitToSeuqnece(graph->FindNode("splitToSequence"), input);
  ASSERT_TRUE(sequence_ret.result.IsSuccess());
  ASSERT_EQ(sequence_ret.out_addrs.size(), 1);
  ASSERT_EQ(sequence_ret.out_shapes.size(), 1);
  ASSERT_EQ(sequence_ret.order_holders.size(), 2);

  auto exe_graph = sequence_ret.out_addrs[0]->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr();
  ASSERT_NE(exe_graph, nullptr);
  // graph compare

  auto execute_graph = bg::ValueHolder::PopGraphFrame(ConvertDevMemValueHoldersToValueHolders(sequence_ret.out_addrs),
                                                      sequence_ret.order_holders)
                           ->GetExecuteGraph();
  ASSERT_NE(execute_graph, nullptr);
  auto split_sequence = ge::ExecuteGraphUtils::FindFirstNodeMatchType(execute_graph.get(), "StoreSplitTensorToSequence");
  ASSERT_NE(split_sequence, nullptr);
  auto session_input_node = split_sequence->GetInDataNodes().at(0);
  ASSERT_NE(session_input_node, nullptr);
  ASSERT_EQ(session_input_node->GetType(), "InnerData"); // GetSessionId在init图

  ge::DumpGraph(execute_graph.get(), "GeneralSplitToSequenceExe");
}

TEST_F(AicpuSequenceConverterUT, ConvertAicpTfNode) {
  auto graph = ShareGraph::BuildSingleNodeGraph();
  auto add_op_desc = graph->FindNode("add1")->GetOpDesc();
  add_op_desc->SetOpKernelLibName(ge::kEngineNameAiCpuTf.c_str());
  AiCpuTfTaskDefFaker aicpu_task_def_faker;
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).AddTaskDef("Add", aicpu_task_def_faker).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};
  EXPECT_NE(bg::GetContainerIdHolder(data_input), nullptr);
}
}  // namespace gert

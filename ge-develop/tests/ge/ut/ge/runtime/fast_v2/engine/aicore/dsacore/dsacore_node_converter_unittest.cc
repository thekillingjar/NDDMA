/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "engine/dsacore/converter/dsacore_node_converter.h"
#include <gtest/gtest.h>
#include "ge_graph_dsl/graph_dsl.h"
#include "engine/gelocal/inputs_converter.h"
#include "graph/utils/graph_utils.h"
#include "graph_builder/exe_graph_comparer.h"
#include "common/share_graph.h"
#include "faker/global_data_faker.h"
#include "common/bg_test.h"
#include "faker/dsacore_taskdef_faker.h"
#include "framework/common/ge_types.h"

using namespace ge;
namespace gert {
class DsaNodeConverterUT : public bg::BgTestAutoCreateFrame {};
TEST_F(DsaNodeConverterUT, ConvertDsaNode) {
  Create3StageFrames();
  auto graph = ShareGraph::BuildDsaRandomNormalGraph();
  auto randomnormal_op_desc = graph->FindNode("randomnormal")->GetOpDesc();
  AttrUtils::SetInt(randomnormal_op_desc, ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 4);
  randomnormal_op_desc->SetWorkspaceBytes({32, 32});
  randomnormal_op_desc->SetOpKernelLibName(ge::kEngineNameDsa.c_str());
  DsaCoreTaskDefFaker dsa_task_def_faker;
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).AddTaskDef("DSARandomNormal",
                                                       dsa_task_def_faker).Build();
  LowerInput data_input = {{}, {}, &global_data};

  auto data1_ret = LoweringDataNode(graph->FindNode("count"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("seed"), data_input);
  auto data3_ret = LoweringDataNode(graph->FindNode("mean"), data_input);
  auto data4_ret = LoweringDataNode(graph->FindNode("stddev"), data_input);
  ASSERT_TRUE(data1_ret.result.IsSuccess());
  ASSERT_TRUE(data2_ret.result.IsSuccess());
  ASSERT_TRUE(data3_ret.result.IsSuccess());
  ASSERT_TRUE(data4_ret.result.IsSuccess());

  LowerInput random_input = {{data1_ret.out_shapes[0], data2_ret.out_shapes[0],
                              data3_ret.out_shapes[0], data4_ret.out_shapes[0]},
                             {data1_ret.out_addrs[0], data2_ret.out_addrs[0],
                              data3_ret.out_addrs[0], data4_ret.out_addrs[0]},
                             &global_data};

  auto init_node = bg::ValueHolder::CreateVoid<bg::ValueHolder>("Init", {});
  bg::ValueHolder::PushGraphFrame(init_node, "Init");
  std::unique_ptr<bg::GraphFrame> init_frame = bg::ValueHolder::PopGraphFrame();

  auto de_init_node = bg::ValueHolder::CreateVoid<bg::ValueHolder>("DeInit", {});
  bg::ValueHolder::PushGraphFrame(de_init_node, "DeInit");
  std::unique_ptr<bg::GraphFrame> de_init_frame = bg::ValueHolder::PopGraphFrame();

  auto main_node = bg::ValueHolder::CreateVoid<bg::ValueHolder>(GetExecuteGraphTypeStr(ExecuteGraphType::kMain), {});
  bg::ValueHolder::PushGraphFrame(main_node, "Main");

  auto random_ret = LoweringDsaCoreNode(graph->FindNode("randomnormal"), random_input);
  ASSERT_TRUE(random_ret.result.IsSuccess());
  ASSERT_EQ(random_ret.out_addrs.size(), 1);
  ASSERT_EQ(random_ret.out_shapes.size(), 1);
  ASSERT_EQ(random_ret.order_holders.size(), 1);
}
}  // namespace gert
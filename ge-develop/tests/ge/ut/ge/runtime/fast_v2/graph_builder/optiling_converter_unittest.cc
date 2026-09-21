/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "engine/aicore/converter/aicore_node_converter.h"

#include <gtest/gtest.h>

#include "ge_graph_dsl/graph_dsl.h"
#include "engine/gelocal/inputs_converter.h"
#include "graph/utils/graph_utils.h"
#include "graph_builder/exe_graph_comparer.h"
#include "common/share_graph.h"
#include "faker/global_data_faker.h"
#include "common/bg_test.h"
#include "common/const_data_helper.h"

using namespace ge;
namespace gert {
LowerResult LoweringOpTiling(const ge::NodePtr &node, const LowerInput &lower_input);

class OpTilingConvertUT : public bg::BgTestAutoCreateFrame {
 public:
  bg::GraphFrame *root_frame = nullptr;
  std::unique_ptr<bg::GraphFrame> init_frame;
  std::unique_ptr<bg::GraphFrame> de_init_frame;
  void InitTestFrames() {
    root_frame = bg::ValueHolder::GetCurrentFrame();
    auto init_node = bg::ValueHolder::CreateVoid<bg::ValueHolder>("Init", {});
    bg::ValueHolder::PushGraphFrame(init_node, "Init");
    init_frame = bg::ValueHolder::PopGraphFrame();

    auto de_init_node = bg::ValueHolder::CreateVoid<bg::ValueHolder>("DeInit", {});
    bg::ValueHolder::PushGraphFrame(de_init_node, "DeInit");
    de_init_frame = bg::ValueHolder::PopGraphFrame();

    auto main_node = bg::ValueHolder::CreateVoid<bg::ValueHolder>(GetExecuteGraphTypeStr(ExecuteGraphType::kMain), {});
    bg::ValueHolder::PushGraphFrame(main_node, "Main");
  }
};
TEST_F(OpTilingConvertUT, ConvertGeneralOpTilingNodeNormal) {
  InitTestFrames();
  auto graph = ShareGraph::BuildOpTilingGraph("Add");
auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};

  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);
  ASSERT_TRUE(data1_ret.result.IsSuccess());
  ASSERT_TRUE(data2_ret.result.IsSuccess());

  LowerInput add_input = {{data1_ret.out_shapes[0], data2_ret.out_shapes[0]},
                          {data1_ret.out_addrs[0], data2_ret.out_addrs[0]},
                          &global_data};
  auto ret = LoweringOpTiling(graph->FindNode("OpTiling_Test"), add_input);
  ASSERT_TRUE(ret.result.IsSuccess());
}

TEST_F(OpTilingConvertUT, ConvertGeneralOpTilingNodeAttrError) {
  InitTestFrames();
  auto graph = ShareGraph::BuildOpTilingGraph("Add");
auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};

  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);
  ASSERT_TRUE(data1_ret.result.IsSuccess());
  ASSERT_TRUE(data2_ret.result.IsSuccess());

  LowerInput add_input = {{data1_ret.out_shapes[0], data2_ret.out_shapes[0]},
                          {data1_ret.out_addrs[0], data2_ret.out_addrs[0]},
                          &global_data};
  // sub_data1 owns no attribute named op_tilling_name
  auto ret = LoweringOpTiling(graph->FindNode("real_tiling_node"), add_input);
  ASSERT_FALSE(ret.result.IsSuccess());
}

TEST_F(OpTilingConvertUT, ConvertGeneralOpTilingNodeWithoutNode) {
  InitTestFrames();
  auto graph = ShareGraph::BuildOpTilingGraph("Add");
auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};

  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);
  ASSERT_TRUE(data1_ret.result.IsSuccess());
  ASSERT_TRUE(data2_ret.result.IsSuccess());
  // rename the node name to make sure lowering cant find node by name attribute
  graph->FindNode("real_tiling_node")->GetOpDesc()->SetName("add1");

  LowerInput add_input = {{data1_ret.out_shapes[0], data2_ret.out_shapes[0]},
                          {data1_ret.out_addrs[0], data2_ret.out_addrs[0]},
                          &global_data};
  auto ret = LoweringOpTiling(graph->FindNode("OpTiling_Test"), add_input);
  ASSERT_FALSE(ret.result.IsSuccess());
}
}  // namespace gert

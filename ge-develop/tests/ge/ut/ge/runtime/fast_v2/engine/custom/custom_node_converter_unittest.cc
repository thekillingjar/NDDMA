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
#include "engine/custom/converter/custom_node_converter.h"
#include "engine/node_converter_utils.h"
#include "common/share_graph.h"
#include "common/summary_checker.h"
#include "ge/ut/ge/runtime/fast_v2/common/const_data_helper.h"
#include "engine/gelocal/inputs_converter.h"
#include "common/bg_test.h"
#include "lowering/placement/placed_lowering_result.h"

namespace gert {
using namespace ge;
using namespace bg;
class CustomNodeConverterUT : public bg::BgTestAutoCreate3StageFrame {
protected:
  void SetUp() override {
    //BgTest::SetUp();
    BgTestAutoCreate3StageFrame::SetUp();
  }
  void TearDown() override {
    BgTestAutoCreate3StageFrame::TearDown();
  }
};
TEST_F(CustomNodeConverterUT, custom_op_convert_test) {
  auto graph = ShareGraph::BuildCustomOpGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  global_data.SetExternalAllocator(nullptr, ExecuteGraphType::kInit);
  global_data.SetExternalAllocator(nullptr, ExecuteGraphType::kMain);
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};
  auto data0_ret = LoweringDataNode(graph->FindNode("data0"), data_input);
  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);
  ASSERT_TRUE(data0_ret.result.IsSuccess());
  ASSERT_TRUE(data1_ret.result.IsSuccess());
  ASSERT_TRUE(data2_ret.result.IsSuccess());
  LowerInput add_input = {
      {data0_ret.out_shapes[0], data1_ret.out_shapes[0], data2_ret.out_shapes[0]},
      {data0_ret.out_addrs[0], data1_ret.out_addrs[0], data2_ret.out_addrs[0]},
      &global_data};
  graph->FindNode("data0")->GetOpDesc()->SetExtAttr("_lowering_result",
      gert::PlacedLoweringResult(graph->FindNode("data0"), std::move(data0_ret)));
  graph->FindNode("data1")->GetOpDesc()->SetExtAttr("_lowering_result",
      gert::PlacedLoweringResult(graph->FindNode("data1"), std::move(data1_ret)));
  graph->FindNode("data2")->GetOpDesc()->SetExtAttr("_lowering_result",
      gert::PlacedLoweringResult(graph->FindNode("data2"), std::move(data2_ret)));
  auto custom_op = graph->FindNode("custom_op");
  auto ret = LoweringCustomNode(custom_op, add_input);
  ASSERT_TRUE(ret.result.IsSuccess());
  ASSERT_EQ(ret.out_addrs.size(), 1);
  ASSERT_EQ(ret.out_shapes.size(), 1);
  ASSERT_EQ(ret.order_holders.size(), 0);
  auto frame = bg::ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto exe_graph = frame->GetExecuteGraph().get();
  ASSERT_NE(exe_graph, nullptr);
  // check main 图中节点数量, 由于没走CEM, main图上有俩InnerData连给PrepareCacheableTilingFwkData
  ASSERT_EQ(ExeGraphSummaryChecker(exe_graph).StrictAllNodeTypes({{"Data", 5},
                                                                  {"Const", 8},
                                                                  {"BuildRefTensor", 3},
                                                                  {"CalcTensorSizeFromStorage", 3},
                                                                  {"ExecuteCustomOp", 1},
                                                                  {"FreeCustomOpWorkspaces", 1},
                                                                  {"FreeMemory", 4},
                                                                  {"SelectL2Allocator", 1},
                                                                  {"SplitRtStreams", 1},
                                                                  {"InnerData", 4},
                                                                  {"MakeSureTensorAtDevice", 3},
                                                                  {"SplitDataTensor", 4}}), "success");
}
}
   
 
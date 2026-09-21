/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "checker/bg_test.h"
#include "checker/summary_checker.h"
#include "checker/topo_checker.h"
#include "exe_graph/lowering/frame_selector.h"
#include "exe_graph/runtime/execute_graph_types.h"
#include "graph/utils/execute_graph_utils.h"
#include <gtest/gtest.h>
#include <graph/utils/graph_utils.h>
namespace gert {
namespace bg {
class FastFrameSelectorUT : public BgTest {
 public:
  GraphFrame *root_frame = nullptr;
  std::unique_ptr<GraphFrame> init_frame;
  std::unique_ptr<GraphFrame> de_init_frame;
  void InitTestFrames() {
    root_frame = ValueHolder::GetCurrentFrame();
    auto init_node = ValueHolder::CreateVoid<bg::ValueHolder>("Init", {});
    ValueHolder::PushGraphFrame(init_node, "Init");
    init_frame = ValueHolder::PopGraphFrame();

    auto de_init_node = ValueHolder::CreateVoid<bg::ValueHolder>("DeInit", {});
    ValueHolder::PushGraphFrame(de_init_node, "DeInit");
    de_init_frame = ValueHolder::PopGraphFrame();

    auto main_node = ValueHolder::CreateVoid<bg::ValueHolder>(GetExecuteGraphTypeStr(ExecuteGraphType::kMain), {});
    ValueHolder::PushGraphFrame(main_node, "Main");
  }
};

/*
 * +-----------------------+
 * |FooGraph               |
 * |                       |
 * |   InnerNetOutput      |
 * |      |                |
 * |     foo2 <--+         |
 * |    /         \        |
 * |  c1           \       |
 * +---+-----+------+------+
 *     |     |      |
 *   data0 data1   bar1
 *                  |
 *                 c0
 */
TEST_F(FastFrameSelectorUT, SelectMainRoot_CreateOnRoot_NoMainGraph) {
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);

  auto foo1 = ValueHolder::CreateSingleDataOutput("Foo1", {data0, data1});
  EXPECT_NE(ValueHolder::PushGraphFrame(foo1, "FooGraph"), nullptr);

  // on FooGraph
  auto c1 = ValueHolder::CreateConst("ConstData", 10, true);

  // on RootGraph
  auto bars = FrameSelector::OnMainRoot([&]() -> std::vector<ValueHolderPtr> {
    auto c0 = ValueHolder::CreateConst("ConstData", 10, true);
    auto bar1 = ValueHolder::CreateSingleDataOutput("Bar1", {c0});
    return {bar1};
  });
  ASSERT_EQ(bars.size(), 1);
  ASSERT_NE(bars[0], nullptr);

  // on FooGraph
  auto foo2 = ValueHolder::CreateSingleDataOutput("Foo2", {c1, bars[0]});
  auto foo2_graph = ValueHolder::PopGraphFrame({foo2}, {});
  ASSERT_NE(foo2_graph, nullptr);

  auto frame = ValueHolder::PopGraphFrame();

  ASSERT_EQ(
      ExeGraphSummaryChecker(frame->GetExecuteGraph().get()).StrictDirectNodeTypes(
        {{"Data", 2}, {"Const", 1}, {"Foo1", 1}, {"Bar1", 1}}), "success");
  ASSERT_EQ(ExeGraphSummaryChecker(foo2_graph->GetExecuteGraph().get())
                .StrictDirectNodeTypes({{"InnerData", 1}, {"Const", 1}, {"Foo2", 1}, {"InnerNetOutput", 1}}),
            "success");

  ASSERT_EQ(FastNodeTopoChecker(bars[0]).StrictConnectTo(0, {{"Foo1", 2}}), "success");
  ASSERT_EQ(FastNodeTopoChecker(bars[0]).StrictConnectFrom({{"Const"}}), "success");
}
TEST_F(FastFrameSelectorUT, SelectMainRoot_CreateOnMainRoot_CurrentFrameIsMainRoot) {
  InitTestFrames();

  auto data0 = ValueHolder::CreateFeed(0);

  // on RootGraph
  auto bars = FrameSelector::OnMainRoot([&]() -> std::vector<ValueHolderPtr> {
    auto c0 = ValueHolder::CreateConst("ConstData", 10, true);
    auto bar1 = ValueHolder::CreateSingleDataOutput("Bar1", {c0, data0});
    return {bar1};
  });
  ASSERT_EQ(bars.size(), 1);
  ASSERT_NE(bars[0], nullptr);

  auto main_frame = ValueHolder::PopGraphFrame();
  auto root_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(main_frame, nullptr);
  ASSERT_NE(root_frame, nullptr);

  ASSERT_EQ(ExeGraphSummaryChecker(root_frame->GetExecuteGraph().get())
            .StrictDirectNodeTypes({{"Init", 1}, {"Main", 1}, {"DeInit", 1}}),
            "success");
  ASSERT_EQ(ExeGraphSummaryChecker(main_frame->GetExecuteGraph().get())
            .StrictDirectNodeTypes({{"Data", 1}, {"Const", 1}, {"Bar1", 1}}),
            "success");

  ASSERT_EQ(FastNodeTopoChecker(bars[0]).StrictConnectFrom({{"Const"}, {"Data"}}), "success");
  ASSERT_TRUE(bars[0]->GetFastNode()->GetAllOutNodes().empty());
}

/*
 * +-----------------------+
 * |FooGraph               |
 * |                       |
 * |   InnerNetOutput      |
 * |      |                |
 * |     foo2 <--+         |
 * |    /         \        |
 * |  c1           \       |
 * +---+-----+------+------+
 *     |     |      |
 *   data0 data1   bar1
 *                  |
 *                 c0
 */
TEST_F(FastFrameSelectorUT, SelectMainRoot_CreateOnMainRoot_CurrentFrameIsMainSubgraphs) {
  InitTestFrames();

  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);

  auto foo1 = ValueHolder::CreateSingleDataOutput("Foo1", {data0, data1});
  ValueHolder::PushGraphFrame(foo1, "FooGraph");

  // on FooGraph
  auto c1 = ValueHolder::CreateConst("ConstData", 10, true);

  // on MainRootGraph
  auto bars = FrameSelector::OnMainRoot([&]() -> std::vector<ValueHolderPtr> {
    auto c0 = ValueHolder::CreateConst("ConstData", 10, true);
    auto bar1 = ValueHolder::CreateSingleDataOutput("Bar1", {c0});
    return {bar1};
  });
  ASSERT_EQ(bars.size(), 1);
  ASSERT_NE(bars[0], nullptr);

  // on FooGraph
  auto foo2 = ValueHolder::CreateSingleDataOutput("Foo2", {c1, bars[0]});
  auto foo2_graph = ValueHolder::PopGraphFrame({foo2}, {});
  ASSERT_NE(foo2_graph, nullptr);

  auto frame = ValueHolder::PopGraphFrame();

  ASSERT_EQ(
      ExeGraphSummaryChecker(frame->GetExecuteGraph().get())
      .StrictDirectNodeTypes({{"Data", 2}, {"Const", 1}, {"Foo1", 1}, {"Bar1", 1}}),
      "success");
  ASSERT_EQ(ExeGraphSummaryChecker(foo2_graph->GetExecuteGraph().get())
            .StrictDirectNodeTypes({{"InnerData", 1}, {"Const", 1}, {"Foo2", 1}, {"InnerNetOutput", 1}}),
            "success");

  ASSERT_EQ(FastNodeTopoChecker(bars[0]).StrictConnectTo(0, {{"Foo1", 2}}), "success");
  ASSERT_EQ(FastNodeTopoChecker(bars[0]).StrictConnectFrom({{"Const"}}), "success");
}
TEST_F(FastFrameSelectorUT, SelectMainRoot_Failed_BuilderIsNullptr) {
  ASSERT_EQ(FrameSelector::OnMainRoot(nullptr).size(), 0);
}
TEST_F(FastFrameSelectorUT, SelectMainRoot_Failed_ConnectFromSubgraph) {
  InitTestFrames();

  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto foo1 = ValueHolder::CreateSingleDataOutput("Foo1", {data0, data1});
  ValueHolder::PushGraphFrame(foo1, "FooGraph");

  // on FooGraph
  auto c1 = ValueHolder::CreateConst("ConstData", 10, true);

  // on RootGraph
  auto bars = FrameSelector::OnMainRoot([&]() -> std::vector<ValueHolderPtr> {
    auto c0 = ValueHolder::CreateConst("ConstData", 10, true);
    auto bar1 = ValueHolder::CreateSingleDataOutput("Bar1", {c1});
    return {bar1};
  });
  ASSERT_EQ(bars.size(), 1);
  ASSERT_EQ(bars[0], nullptr);
}
TEST_F(FastFrameSelectorUT, SelectInitRoot_Success_OnlyInitNodeOut) {
  InitTestFrames();
  auto ret = FrameSelector::OnInitRoot([]() -> std::vector<ValueHolderPtr> {
    auto c1 = ValueHolder::CreateConst("Hello", 5);
    auto c2 = ValueHolder::CreateConst("Hello", 5);
    auto foo1 = ValueHolder::CreateSingleDataOutput("Foo1", {c1, c2});
    auto foo2 = ValueHolder::CreateDataOutput("Foo2", {foo1, c1}, 2);
    return {foo1, foo2[0], foo2[1]};
  });

  ASSERT_EQ(ret.size(), 3);
  ASSERT_EQ(ret[0]->GetFastNode()->GetType(), "Init");
  ASSERT_EQ(ret[1]->GetFastNode()->GetType(), "Init");
  ASSERT_EQ(ret[2]->GetFastNode()->GetType(), "Init");
  ASSERT_EQ(ret[0]->GetOutIndex(), 0);
  ASSERT_EQ(ret[1]->GetOutIndex(), 1);
  ASSERT_EQ(ret[2]->GetOutIndex(), 2);
}
TEST_F(FastFrameSelectorUT, SelectInitRoot_Success_ReEnter) {
  InitTestFrames();
  auto ret = FrameSelector::OnInitRoot([]() -> std::vector<ValueHolderPtr> {
    auto c1 = ValueHolder::CreateConst("Hello", 5);
    auto c2 = ValueHolder::CreateConst("Hello", 5);
    auto foo1 = ValueHolder::CreateSingleDataOutput("Foo1", {c1, c2});

    auto out_graph_outs = FrameSelector::OnInitRoot([]() -> std::vector<ValueHolderPtr> {
      auto c1 = ValueHolder::CreateConst("Hello", 5);
      auto c2 = ValueHolder::CreateConst("Hello", 5);
      return {ValueHolder::CreateSingleDataOutput("Bar1", {c1, c2})};
    });

    auto foo2 = ValueHolder::CreateDataOutput("Foo2", {HolderOnInit(out_graph_outs[0]), foo1}, 2);
    return {foo1, foo2[0], foo2[1]};
  });
  ASSERT_EQ(ret.size(), 3);

  EXPECT_EQ(ExeGraphSummaryChecker(init_frame->GetExecuteGraph().get())
            .StrictDirectNodeTypes({{"Const", 4}, {"Foo1", 1}, {"Bar1", 1}, {"Foo2", 1}, {"InnerNetOutput", 1}}),
            "success");

  auto inner_netoutput_node =
    ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_frame->GetExecuteGraph().get(), "InnerNetOutput");
  ASSERT_NE(inner_netoutput_node, nullptr);
  ASSERT_EQ(FastNodeTopoChecker(inner_netoutput_node).StrictConnectFrom({{"Bar1", 0}, {"Foo1", 0}, {"Foo2", 0}, {"Foo2", 1}}),
            "success");

  auto foo2_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_frame->GetExecuteGraph().get(), "Foo2");
  ASSERT_NE(foo2_node, nullptr);
  ASSERT_EQ(FastNodeTopoChecker(foo2_node).StrictConnectFrom({{"Bar1", 0}, {"Foo1", 0}}),
            "success");
}
TEST_F(FastFrameSelectorUT, SelectInitRoot_TheSameWithInput_ReturnInput) {
  InitTestFrames();
  auto ret = FrameSelector::OnInitRoot([]() -> std::vector<ValueHolderPtr> {
    auto c1 = ValueHolder::CreateConst("Hello", 5);
    auto c2 = ValueHolder::CreateConst("Hello", 5);
    auto foo1 = ValueHolder::CreateSingleDataOutput("Foo1", {c1, c2});
    auto foo2 = ValueHolder::CreateDataOutput("Foo2", {foo1, c1}, 2);
    return {foo1, foo2[0], foo2[1]};
  });

  ASSERT_EQ(ret.size(), 3);

  auto ret2 = FrameSelector::OnInitRoot([&]() -> std::vector<ValueHolderPtr> {
    return {HolderOnInit(ret[0]),
            ValueHolder::CreateSingleDataOutput("Foo3", {HolderOnInit(ret[0]), ValueHolder::CreateConst("Hello", 5)}),
            HolderOnInit(ret[1])};
  });
  ASSERT_EQ(ret2.size(), 3);

  ASSERT_EQ(ret2[0]->GetFastNode()->GetType(), "Init");
  ASSERT_EQ(ret2[1]->GetFastNode()->GetType(), "Init");
  ASSERT_EQ(ret2[2]->GetFastNode()->GetType(), "Init");
  ASSERT_EQ(ret2[0]->GetOutIndex(), 0);
  ASSERT_EQ(ret2[1]->GetOutIndex(), 3);
  ASSERT_EQ(ret2[2]->GetOutIndex(), 1);

  auto netoutput =
    ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_frame->GetExecuteGraph().get(), "InnerNetOutput");
  ASSERT_NE(netoutput, nullptr);
  EXPECT_EQ(FastNodeTopoChecker(netoutput).StrictConnectFrom({{"Foo1", 0}, {"Foo2", 0}, {"Foo2", 1}, {"Foo3", 0}}),
            "success");
}
TEST_F(FastFrameSelectorUT, SelectInitRoot_Success_InitNodeAndInitGraphOut) {
  InitTestFrames();
  std::vector<bg::ValueHolderPtr> init_node_outputs;
  std::vector<bg::ValueHolderPtr> init_graph_outputs;
  auto ret = FrameSelector::OnInitRoot(
      []() -> std::vector<ValueHolderPtr> {
        auto c1 = ValueHolder::CreateConst("Hello", 5);
        auto c2 = ValueHolder::CreateConst("Hello", 5);
        auto foo1 = ValueHolder::CreateSingleDataOutput("Foo1", {c1, c2});
        auto foo2 = ValueHolder::CreateDataOutput("Foo2", {foo1, c1}, 2);
        return {foo1, foo2[0], foo2[1]};
      },
      init_graph_outputs, init_node_outputs);
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);

  ASSERT_EQ(init_node_outputs.size(), 3);
  ASSERT_EQ(init_node_outputs[0]->GetFastNode()->GetType(), "Init");
  ASSERT_EQ(init_node_outputs[1]->GetFastNode()->GetType(), "Init");
  ASSERT_EQ(init_node_outputs[2]->GetFastNode()->GetType(), "Init");
  ASSERT_EQ(init_node_outputs[0]->GetOutIndex(), 0);
  ASSERT_EQ(init_node_outputs[1]->GetOutIndex(), 1);
  ASSERT_EQ(init_node_outputs[2]->GetOutIndex(), 2);

  ASSERT_EQ(init_graph_outputs.size(), 3);
  ASSERT_EQ(init_graph_outputs[0]->GetFastNode()->GetType(), "Foo1");
  ASSERT_EQ(init_graph_outputs[1]->GetFastNode()->GetType(), "Foo2");
  ASSERT_EQ(init_graph_outputs[2]->GetFastNode()->GetType(), "Foo2");
  ASSERT_EQ(init_graph_outputs[0]->GetOutIndex(), 0);
  ASSERT_EQ(init_graph_outputs[1]->GetOutIndex(), 0);
  ASSERT_EQ(init_graph_outputs[2]->GetOutIndex(), 1);
  ASSERT_EQ(init_graph_outputs[1]->GetFastNode(), init_graph_outputs[2]->GetFastNode());

  ASSERT_EQ(FastNodeTopoChecker(init_graph_outputs[0]).StrictConnectTo(0, {{"InnerNetOutput", 0}, {"Foo2", 0}}), "success");
  ASSERT_EQ(FastNodeTopoChecker(init_graph_outputs[1]).StrictConnectTo(0, {{"InnerNetOutput", 1}}), "success");
  ASSERT_EQ(FastNodeTopoChecker(init_graph_outputs[1]).StrictConnectTo(1, {{"InnerNetOutput", 2}}), "success");
}
TEST_F(FastFrameSelectorUT, SelectInitRoot_AllConnectToInnerNetOutput_CallTwoTimes) {
  InitTestFrames();
  auto outputs1 =
      FrameSelector::OnInitRoot([]() -> std::vector<ValueHolderPtr> { return {ValueHolder::CreateConst("Hello", 5)}; });
  ASSERT_EQ(outputs1.size(), 1);

  std::vector<bg::ValueHolderPtr> init_node_outputs;
  std::vector<bg::ValueHolderPtr> init_graph_outputs;
  auto ret = FrameSelector::OnInitRoot(
      []() -> std::vector<ValueHolderPtr> {
        auto c1 = ValueHolder::CreateConst("Hello", 5);
        auto c2 = ValueHolder::CreateConst("Hello", 5);
        return ValueHolder::CreateDataOutput("Foo2", {c1, c2}, 2);
      },
      init_graph_outputs, init_node_outputs);
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);

  ASSERT_EQ(init_graph_outputs.size(), 2);
  ASSERT_EQ(ExeGraphSummaryChecker(init_graph_outputs[0]->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr())
                .StrictAllNodeTypes({{"Const", 3}, {"Foo2", 1}, {"InnerNetOutput", 1}}),
            "success");

  ASSERT_EQ(init_node_outputs[1]->GetFastNode(), init_node_outputs[1]->GetFastNode());
  ASSERT_EQ(init_node_outputs[0]->GetFastNode()->GetType(), "Init");
  ASSERT_EQ(init_node_outputs[0]->GetOutIndex(), 1);
  ASSERT_EQ(init_node_outputs[1]->GetOutIndex(), 2);

  ASSERT_EQ(init_graph_outputs.size(), 2);
  ASSERT_EQ(init_graph_outputs[0]->GetFastNode()->GetType(), "Foo2");
  ASSERT_EQ(init_graph_outputs[1]->GetFastNode()->GetType(), "Foo2");
  ASSERT_EQ(init_graph_outputs[0]->GetOutIndex(), 0);
  ASSERT_EQ(init_graph_outputs[1]->GetOutIndex(), 1);
  ASSERT_EQ(init_graph_outputs[1]->GetFastNode(), init_graph_outputs[1]->GetFastNode());

  ASSERT_EQ(FastNodeTopoChecker(init_graph_outputs[0]).StrictConnectTo(0, {{"InnerNetOutput", 1}}), "success");
  ASSERT_EQ(FastNodeTopoChecker(init_graph_outputs[1]).StrictConnectTo(1, {{"InnerNetOutput", 2}}), "success");
}
TEST_F(FastFrameSelectorUT, SelectInitRoot_FrameCorrect_AfterSelection) {
  InitTestFrames();

  auto d0 = ValueHolder::CreateFeed(0);
  auto d1 = ValueHolder::CreateFeed(1);
  auto bar1 = ValueHolder::CreateSingleDataOutput("Bar1", {d0, d1});
  ValueHolder::PushGraphFrame(bar1, "Graph");

  auto init_outputs = FrameSelector::OnInitRoot([]() -> std::vector<ValueHolderPtr> {
    auto c1 = ValueHolder::CreateConst("Hello", 5);
    auto c2 = ValueHolder::CreateConst("Hello", 5);
    return ValueHolder::CreateDataOutput("Foo2", {c1, c2}, 2);
  });
  ASSERT_EQ(init_outputs.size(), 2);

  auto foo1 = ValueHolder::CreateSingleDataOutput("Foo1", {d0, d1});
  ValueHolder::PopGraphFrame({foo1}, {});

  ASSERT_NE(bar1, nullptr);
  auto bar1_graph = ge::FastNodeUtils::GetSubgraphFromNode(bar1->GetFastNode(), 0);
  ASSERT_NE(bar1_graph, nullptr);
  ASSERT_EQ(ExeGraphSummaryChecker(bar1_graph).StrictAllNodeTypes({{"InnerData", 2}, {"Foo1", 1}, {"InnerNetOutput", 1}}),
            "success");
}
TEST_F(FastFrameSelectorUT, SelectInitRoot_ConnectFromResultOk) {
  InitTestFrames();

  auto d0 = ValueHolder::CreateFeed(0);
  auto d1 = ValueHolder::CreateFeed(1);
  auto bar1 = ValueHolder::CreateSingleDataOutput("Bar1", {d0, d1});

  const ge::FastNode *foo2_node;
  auto init_outputs = FrameSelector::OnInitRoot([&]() -> std::vector<ValueHolderPtr> {
    auto c1 = ValueHolder::CreateConst("Hello", 5);
    auto c2 = ValueHolder::CreateConst("Hello", 5);
    auto foo2 = ValueHolder::CreateDataOutput("Foo2", {c1, c2}, 2);
    foo2_node = foo2[0]->GetFastNode();
    return foo2;
  });
  ASSERT_EQ(init_outputs.size(), 2);

  auto foo1 = ValueHolder::CreateSingleDataOutput("Foo1", {bar1, init_outputs[0]});

  ASSERT_NE(foo1, nullptr);
  ASSERT_EQ(foo1->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr()->GetParentNodeBarePtr()->GetType(), "Main");
  ASSERT_EQ(FastNodeTopoChecker(foo1).StrictConnectFrom({{"Bar1"}, {"InnerData"}}), "success");
  ConnectFromInitToMain(foo2_node, 0, foo1->GetFastNode(), 1);
}
TEST_F(FastFrameSelectorUT, SelectInitRoot_PlacementCorrect) {
  InitTestFrames();

  auto init_outputs = FrameSelector::OnInitRoot([&]() -> std::vector<ValueHolderPtr> {
    auto c1 = ValueHolder::CreateConst("Hello", 5);
    auto c2 = ValueHolder::CreateConst("Hello", 5);
    auto foo2 = ValueHolder::CreateDataOutput("Foo2", {c1, c2}, 2);
    foo2[0]->SetPlacement(kOnDeviceHbm);
    foo2[1]->SetPlacement(kOnDeviceHbm);
    auto foo3 = ValueHolder::CreateSingleDataOutput("Foo3", {c1, c2});
    foo3->SetPlacement(kOnHost);
    return {foo2[0], foo2[1], foo3};
  });
  ASSERT_EQ(init_outputs.size(), 3);
  EXPECT_EQ(init_outputs[0]->GetPlacement(), kOnDeviceHbm);
  EXPECT_EQ(init_outputs[1]->GetPlacement(), kOnDeviceHbm);
  EXPECT_EQ(init_outputs[2]->GetPlacement(), kOnHost);
}
TEST_F(FastFrameSelectorUT, SelectInitRoot_GuarderInDeInit_OutputHasGuarder) {
  InitTestFrames();

  auto init_outputs = FrameSelector::OnInitRoot([&]() -> std::vector<ValueHolderPtr> {
    auto c1 = ValueHolder::CreateConst("Hello", 5);
    auto c2 = ValueHolder::CreateConst("Hello", 5);
    auto foo1 = ValueHolder::CreateSingleDataOutput("Foo1", {c1});
    auto foo1_guarder = ValueHolder::CreateVoidGuarder("FreeFoo1", foo1, {});
    auto foo2 = ValueHolder::CreateDataOutput("Foo2", {c1, c2, foo1}, 2);
    auto foo2_guarder = ValueHolder::CreateVoidGuarder("FreeFoo2", foo2[0], {});
    return foo2;
  });
  ASSERT_EQ(init_outputs.size(), 2);

  ASSERT_EQ(ExeGraphSummaryChecker(init_frame->GetExecuteGraph().get())
            .StrictAllNodeTypes({{"Const", 2}, {"Foo1", 1}, {"FreeFoo1", 1}, {"Foo2", 1}, {"InnerNetOutput", 1}}),
            "success");
  auto netoutput = ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_frame->GetExecuteGraph().get(), "InnerNetOutput");
  ASSERT_NE(netoutput, nullptr);
  ASSERT_EQ(FastNodeTopoChecker(netoutput).StrictConnectFrom({{"Foo2", 0}, {"Foo2", 1}}), "success");

  ASSERT_EQ(ExeGraphSummaryChecker(de_init_frame->GetExecuteGraph().get())
            .StrictAllNodeTypes({{"InnerData", 1}, {"FreeFoo2", 1}}),
            "success");

  auto foo2 = ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_frame->GetExecuteGraph().get(), "Foo2");
  ASSERT_NE(foo2, nullptr);
  auto free_foo2 = ge::ExecuteGraphUtils::FindFirstNodeMatchType(de_init_frame->GetExecuteGraph().get(), "FreeFoo2");
  ASSERT_NE(free_foo2, nullptr);
  ConnectFromInitToDeInit(foo2, 0, free_foo2, 0);
}
TEST_F(FastFrameSelectorUT, SelectInitRoot_GuarderInDeInit_MultipleOutputHasGuarder) {
  InitTestFrames();

  auto init_outputs = FrameSelector::OnInitRoot([&]() -> std::vector<ValueHolderPtr> {
    auto c1 = ValueHolder::CreateConst("Hello", 5);
    auto c2 = ValueHolder::CreateConst("Hello", 5);
    auto foo1 = ValueHolder::CreateSingleDataOutput("Foo1", {c1});
    auto foo1_guarder = ValueHolder::CreateVoidGuarder("FreeFoo1", foo1, {});
    auto foo2 = ValueHolder::CreateDataOutput("Foo2", {c1, c2, foo1}, 2);
    auto foo2_guarder = ValueHolder::CreateVoidGuarder("FreeFoo2", foo2[1], {});
    return {foo1, foo2[0], foo2[1]};
  });
  ASSERT_EQ(init_outputs.size(), 3);

  ASSERT_EQ(ExeGraphSummaryChecker(init_frame->GetExecuteGraph().get())
                .StrictAllNodeTypes({{"Const", 2}, {"Foo1", 1}, {"Foo2", 1}, {"InnerNetOutput", 1}}),
            "success");
  auto netoutput =
    ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_frame->GetExecuteGraph().get(), "InnerNetOutput");
  ASSERT_NE(netoutput, nullptr);
  ASSERT_EQ(FastNodeTopoChecker(netoutput).StrictConnectFrom({{"Foo1", 0}, {"Foo2", 0}, {"Foo2", 1}}), "success");

  ASSERT_EQ(ExeGraphSummaryChecker(de_init_frame->GetExecuteGraph().get())
                .StrictAllNodeTypes({{"InnerData", 2}, {"FreeFoo2", 1}, {"FreeFoo1", 1}}),
            "success");

  auto foo2 = ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_frame->GetExecuteGraph().get(), "Foo2");
  ASSERT_NE(foo2, nullptr);
  auto free_foo2 = ge::ExecuteGraphUtils::FindFirstNodeMatchType(de_init_frame->GetExecuteGraph().get(), "FreeFoo2");
  ASSERT_NE(free_foo2, nullptr);
  ConnectFromInitToDeInit(foo2, 1, free_foo2, 0);

  auto foo1 = ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_frame->GetExecuteGraph().get(), "Foo1");
  ASSERT_NE(foo1, nullptr);
  auto free_foo1 = ge::ExecuteGraphUtils::FindFirstNodeMatchType(de_init_frame->GetExecuteGraph().get(), "FreeFoo1");
  ASSERT_NE(free_foo1, nullptr);
  ConnectFromInitToDeInit(foo1, 0, free_foo1, 0);
}
TEST_F(FastFrameSelectorUT, SelectInitRoot_AllGuardersInDeInit_MultipleTimes) {
  InitTestFrames();

  ge::FastNode *g1_foo2_node;
  ge::FastNode *g1_free_foo2_node;
  ge::FastNode *g1_foo1_node;
  ge::FastNode *g1_free_foo1_node;
  auto init_outputs_1 = FrameSelector::OnInitRoot([&]() -> std::vector<ValueHolderPtr> {
    auto c1 = ValueHolder::CreateConst("Hello", 5);
    auto c2 = ValueHolder::CreateConst("Hello", 5);
    auto foo1 = ValueHolder::CreateSingleDataOutput("Foo1", {c1});
    auto foo1_guarder = ValueHolder::CreateVoidGuarder("FreeFoo1", foo1, {});
    auto foo2 = ValueHolder::CreateDataOutput("Foo2", {c1, c2, foo1}, 2);
    auto foo2_guarder = ValueHolder::CreateVoidGuarder("FreeFoo2", foo2[1], {});
    g1_foo1_node = foo1->GetFastNode();
    g1_free_foo1_node = foo1_guarder->GetFastNode();
    g1_foo2_node = foo2[0]->GetFastNode();
    g1_free_foo2_node = foo2_guarder->GetFastNode();
    return {foo1, foo2[0], foo2[1]};
  });
  ASSERT_EQ(init_outputs_1.size(), 3);

  ge::FastNode *g2_foo2_node;
  ge::FastNode *g2_free_foo2_node;
  ge::FastNode *g2_foo1_node;
  ge::FastNode *g2_free_foo1_node;
  auto init_outputs_2 = FrameSelector::OnInitRoot([&]() -> std::vector<ValueHolderPtr> {
    auto c1 = ValueHolder::CreateConst("Hello", 5);
    auto c2 = ValueHolder::CreateConst("Hello", 5);
    auto foo1 = ValueHolder::CreateSingleDataOutput("Foo1", {c1});
    auto foo1_guarder = ValueHolder::CreateVoidGuarder("FreeFoo1", foo1, {});
    auto foo2 = ValueHolder::CreateDataOutput("Foo2", {c1, c2, foo1}, 2);
    auto foo2_guarder = ValueHolder::CreateVoidGuarder("FreeFoo2", foo2[1], {});
    g2_foo1_node = foo1->GetFastNode();
    g2_free_foo1_node = foo1_guarder->GetFastNode();
    g2_foo2_node = foo2[0]->GetFastNode();
    g2_free_foo2_node = foo2_guarder->GetFastNode();
    return {foo1, foo2[0], foo2[1]};
  });
  ASSERT_EQ(init_outputs_2.size(), 3);

  ASSERT_EQ(ExeGraphSummaryChecker(init_frame->GetExecuteGraph().get())
                .StrictAllNodeTypes({{"Const", 4}, {"Foo1", 2}, {"Foo2", 2}, {"InnerNetOutput", 1}}),
            "success");
  auto netoutput = ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_frame->GetExecuteGraph().get(), "InnerNetOutput");
  ASSERT_NE(netoutput, nullptr);
  ASSERT_EQ(FastNodeTopoChecker(netoutput).StrictConnectFrom(
                {{"Foo1", 0}, {"Foo2", 0}, {"Foo2", 1}, {"Foo1", 0}, {"Foo2", 0}, {"Foo2", 1}}),
            "success");

  ASSERT_EQ(ExeGraphSummaryChecker(de_init_frame->GetExecuteGraph().get())
                .StrictAllNodeTypes({{"InnerData", 4}, {"FreeFoo2", 2}, {"FreeFoo1", 2}}),
            "success");

  // check 1
  auto foo2 = init_frame->GetExecuteGraph()->FindNode(g1_foo2_node->GetNodeToken());
  ASSERT_NE(foo2, nullptr);
  auto free_foo2 = de_init_frame->GetExecuteGraph()->FindNode(g1_free_foo2_node->GetNodeToken());
  ASSERT_NE(free_foo2, nullptr);
  ConnectFromInitToDeInit(foo2, 1, free_foo2, 0);

  auto foo1 = init_frame->GetExecuteGraph()->FindNode(g1_foo1_node->GetNodeToken());
  ASSERT_NE(foo1, nullptr);
  auto free_foo1 = de_init_frame->GetExecuteGraph()->FindNode(g1_free_foo1_node->GetNodeToken());
  ASSERT_NE(free_foo1, nullptr);
  ConnectFromInitToDeInit(foo1, 0, free_foo1, 0);

  // check 2
  foo2 = init_frame->GetExecuteGraph()->FindNode(g2_foo2_node->GetNodeToken());
  ASSERT_NE(foo2, nullptr);
  free_foo2 = de_init_frame->GetExecuteGraph()->FindNode(g2_free_foo2_node->GetNodeToken());
  ASSERT_NE(free_foo2, nullptr);
  ConnectFromInitToDeInit(foo2, 1, free_foo2, 0);

  foo1 = init_frame->GetExecuteGraph()->FindNode(g2_foo1_node->GetNodeToken());
  ASSERT_NE(foo1, nullptr);
  free_foo1 = de_init_frame->GetExecuteGraph()->FindNode(g2_free_foo1_node->GetNodeToken());
  ASSERT_NE(free_foo1, nullptr);
  ConnectFromInitToDeInit(foo1, 0, free_foo1, 0);
}
TEST_F(FastFrameSelectorUT, HolderOnInit_GetInput_WhenInputInInit) {
  InitTestFrames();

  std::vector<ValueHolderPtr> graph_outputs;
  std::vector<ValueHolderPtr> node_outputs;
  ValueHolderPtr foo1;
  auto ret = FrameSelector::OnInitRoot(
      [&]() -> std::vector<ValueHolderPtr> {
        auto c1 = ValueHolder::CreateConst("Hello", 5);
        auto c2 = ValueHolder::CreateConst("Hello", 5);
        foo1 = ValueHolder::CreateSingleDataOutput("Foo1", {c1});
        auto foo1_guarder = ValueHolder::CreateVoidGuarder("FreeFoo1", foo1, {});
        auto foo2 = ValueHolder::CreateDataOutput("Foo2", {c1, c2, foo1}, 3);
        auto foo2_guarder = ValueHolder::CreateVoidGuarder("FreeFoo2", foo2[1], {});
        return foo2;
      },
      graph_outputs, node_outputs);
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);

  ASSERT_NE(foo1, nullptr);
  ASSERT_EQ(HolderOnInit(foo1), foo1);
  ASSERT_EQ(HolderOnInit(graph_outputs[0]), graph_outputs[0]);
  ASSERT_EQ(HolderOnInit(graph_outputs[1]), graph_outputs[1]);
  ASSERT_EQ(HolderOnInit(graph_outputs[2]), graph_outputs[2]);
}
TEST_F(FastFrameSelectorUT, HolderOnInit_GetInitOutput_WhenInputIsInitNode) {
  InitTestFrames();

  std::vector<ValueHolderPtr> graph_outputs;
  std::vector<ValueHolderPtr> node_outputs;
  ValueHolderPtr foo1;
  auto ret = FrameSelector::OnInitRoot(
      [&]() -> std::vector<ValueHolderPtr> {
        auto c1 = ValueHolder::CreateConst("Hello", 5);
        auto c2 = ValueHolder::CreateConst("Hello", 5);
        foo1 = ValueHolder::CreateSingleDataOutput("Foo1", {c1});
        auto foo1_guarder = ValueHolder::CreateVoidGuarder("FreeFoo1", foo1, {});
        auto foo2 = ValueHolder::CreateDataOutput("Foo2", {c1, c2, foo1}, 3);
        auto foo2_guarder = ValueHolder::CreateVoidGuarder("FreeFoo2", foo2[1], {});
        return foo2;
      },
      graph_outputs, node_outputs);
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);

  ASSERT_NE(foo1, nullptr);
  ASSERT_EQ(HolderOnInit(foo1), foo1);
  ASSERT_EQ(HolderOnInit(node_outputs[0])->GetFastNode(), graph_outputs[0]->GetFastNode());
  ASSERT_EQ(HolderOnInit(node_outputs[0])->GetOutIndex(), graph_outputs[0]->GetOutIndex());
  ASSERT_EQ(HolderOnInit(node_outputs[1])->GetFastNode(), graph_outputs[1]->GetFastNode());
  ASSERT_EQ(HolderOnInit(node_outputs[1])->GetOutIndex(), graph_outputs[1]->GetOutIndex());
  ASSERT_EQ(HolderOnInit(node_outputs[2])->GetFastNode(), graph_outputs[2]->GetFastNode());
  ASSERT_EQ(HolderOnInit(node_outputs[2])->GetOutIndex(), graph_outputs[2]->GetOutIndex());
}

TEST_F(FastFrameSelectorUT, OnMainRootLast_GetInitOutput_WhenInputIsInitNode) {
  InitTestFrames();

  std::vector<ValueHolderPtr> graph_outputs;
  std::vector<ValueHolderPtr> node_outputs;
  ValueHolderPtr foo1;
  auto ret = FrameSelector::OnInitRoot(
      [&]() -> std::vector<ValueHolderPtr> {
        auto c1 = ValueHolder::CreateConst("Hello", 5);
        auto c2 = ValueHolder::CreateConst("Hello", 5);
        foo1 = ValueHolder::CreateSingleDataOutput("Foo1", {c1});
        auto foo1_guarder = ValueHolder::CreateVoidGuarder("FreeFoo1", foo1, {});
        auto foo2 = ValueHolder::CreateDataOutput("Foo2", {c1, c2, foo1}, 3);
        auto foo2_guarder = ValueHolder::CreateVoidGuarder("FreeFoo2", foo2[1], {});
        return foo2;
      },
      graph_outputs, node_outputs);
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);

  ASSERT_NE(foo1, nullptr);
  ASSERT_EQ(HolderOnInit(foo1), foo1);
  ASSERT_EQ(HolderOnInit(node_outputs[0])->GetFastNode(), graph_outputs[0]->GetFastNode());
  ASSERT_EQ(HolderOnInit(node_outputs[0])->GetOutIndex(), graph_outputs[0]->GetOutIndex());
  ASSERT_EQ(HolderOnInit(node_outputs[1])->GetFastNode(), graph_outputs[1]->GetFastNode());
  ASSERT_EQ(HolderOnInit(node_outputs[1])->GetOutIndex(), graph_outputs[1]->GetOutIndex());
  ASSERT_EQ(HolderOnInit(node_outputs[2])->GetFastNode(), graph_outputs[2]->GetFastNode());
  ASSERT_EQ(HolderOnInit(node_outputs[2])->GetOutIndex(), graph_outputs[2]->GetOutIndex());
}
/*
 *      data0    c0
 *        \      /
 *          bar1
 */
TEST_F(FastFrameSelectorUT, OnMainRootLast_SetLastExecNode_NoNetoutput_Fail) {
  InitTestFrames();
  // build exe graph on main frame
  auto data0 = ValueHolder::CreateFeed(0);
  auto c0 = ValueHolder::CreateConst("ConstData", 10, true);
  auto bar1 = ValueHolder::CreateSingleDataOutput("Bar1", {c0, data0});

  auto last_node_builder = [&]() -> bg::ValueHolderPtr {
    return bg::ValueHolder::CreateVoid<ValueHolder>("LastExecNode", {c0});
  };
  auto last_holder = bg::FrameSelector::OnMainRootLast(last_node_builder);
  EXPECT_NE(last_holder, nullptr);
  auto last_exec_nodes = bg::ValueHolder::GetLastExecNodes();
  EXPECT_FALSE(last_exec_nodes.empty());

  auto main_frame = ValueHolder::PopGraphFrame();
  ASSERT_EQ(ExeGraphSummaryChecker(main_frame->GetExecuteGraph().get())
                .StrictDirectNodeTypes({{"Data", 1}, {"Const", 1}, {"Bar1", 1}, {"LastExecNode", 1}}),
            "success");
}

/*
 *      data0    c0                c0
 *        \      /                  |
 *          bar1     last_node: LastExecNode
 *           |
 *       netoutput
 *
 */
TEST_F(FastFrameSelectorUT, OnMainRootLast_SetLastExecNode_OneNodeInsideOneLayer_DefaultPriority) {
  InitTestFrames();
  // build exe graph on main frame
  auto data0 = ValueHolder::CreateFeed(0);
  auto c0 = ValueHolder::CreateConst("ConstData", 10, true);
  auto bar1 = ValueHolder::CreateSingleDataOutput("Bar1", {c0, data0});
  auto output = ValueHolder::CreateSingleDataOutput("NetOutput", {bar1});

  auto last_node_builder = [&]() -> bg::ValueHolderPtr {
    return bg::ValueHolder::CreateVoid<ValueHolder>("LastExecNode", {c0});
  };
  auto last_holder = bg::FrameSelector::OnMainRootLast(last_node_builder);
  EXPECT_NE(last_holder, nullptr);
  auto last_exec_nodes = bg::ValueHolder::GetLastExecNodes();
  EXPECT_FALSE(last_exec_nodes.empty());

  auto main_frame = ValueHolder::PopGraphFrame();
  ASSERT_EQ(ExeGraphSummaryChecker(main_frame->GetExecuteGraph().get())
                .StrictDirectNodeTypes({{"Data", 1}, {"Const", 1}, {"Bar1", 1}, {"LastExecNode", 1}, {"NetOutput", 1}}),
            "success");
  ASSERT_EQ(FastNodeTopoChecker(output).StrictConnectFrom({{"Bar1"}}), "success");
}
/*
 *      data0    c0                       c0                          data0   c0
 *        \      /                      /     \                         \    /
 *          bar1     last_node:        L0    L1               ====>       bar1
 *           |                                                           /    \c
 *       netoutput                                                      |       noop
 *                                                                      |        /c \c
 *                                                                      |      L0    L1
 *                                                                      |        \c  /c
 *                                                                      |        noop
 *                                                                       \       /c
 *                                                                       netoutput
 *                                                                       这里L0和L1分别有来自c0的数据输入。示意图中忽略
 */
TEST_F(FastFrameSelectorUT, OnMainRootLast_SetLastExecNode_TwoNodeInsideOneLayer_DefaultPriority) {
  InitTestFrames();
  // build exe graph on main frame
  auto data0 = ValueHolder::CreateFeed(0);
  auto c0 = ValueHolder::CreateConst("ConstData", 10, true);
  auto bar1 = ValueHolder::CreateSingleDataOutput("Bar1", {c0, data0});
  auto output = ValueHolder::CreateSingleDataOutput("NetOutput", {bar1});

  auto last0_node_builder = [&]() -> bg::ValueHolderPtr {
    return bg::ValueHolder::CreateVoid<ValueHolder>("L0", {c0});
  };
  auto last0_holder = bg::FrameSelector::OnMainRootLast(last0_node_builder);
  EXPECT_NE(last0_holder, nullptr);
  auto last1_node_builder = [&]() -> bg::ValueHolderPtr {
    return bg::ValueHolder::CreateVoid<ValueHolder>("L1", {c0});
  };
  auto last1_holder = bg::FrameSelector::OnMainRootLast(last1_node_builder);
  EXPECT_NE(last1_holder, nullptr);
  auto last_exec_nodes = bg::ValueHolder::GetLastExecNodes();
  EXPECT_EQ(last_exec_nodes.size(), 2U);

  auto main_frame = ValueHolder::PopGraphFrame();
  ASSERT_EQ(
      ExeGraphSummaryChecker(main_frame->GetExecuteGraph().get())
          .StrictDirectNodeTypes({{"Data", 1}, {"Const", 1}, {"Bar1", 1}, {"L0", 1}, {"L1", 1}, {"NetOutput", 1}}),
      "success");
}
/*
 *      data0    c0                       c0                          c0
 *        \      /                      /     \                     /   \
 *          bar1       last_priority0:  L0    L1    last_priority1: L2  L3
 *           |
 *       netoutput
 *
 *                                   |
 *                                   V      subgraph of PartitionCall
 *  +----------------------------------+    +-------------------------+
 *  |    data0    c0                   |    |        InnerData        |
 *  |      \      / \                  |    |      /     \  \  \      |
 *  |        bar1    PartitionedCall   |    |     L0    L1    L2  L3  |
 *  |         |                        |    |     \     /     /    /  |
 *  |     netoutput                    |    |       InnerNetoutput    |
 *  +----------------------------------+    +-------------------------+
 * 这里L0和L1分别有来自c0的数据输入。示意图中忽略
 */
TEST_F(FastFrameSelectorUT, OnMainRootLast_SetLastExecNode_TwoNodeInsideEachLayer_TwoPriority) {
  InitTestFrames();
  // build exe graph on main frame
  auto data0 = ValueHolder::CreateFeed(0);
  auto c0 = ValueHolder::CreateConst("ConstData", 10, true);
  auto bar1 = ValueHolder::CreateSingleDataOutput("Bar1", {c0, data0});
  auto bar2 = ValueHolder::CreateSingleDataOutput("Bar2", {c0});
  auto output = ValueHolder::CreateSingleDataOutput("NetOutput", {bar1, bar2});

  for (size_t node_size = 0U; node_size < 2; ++node_size) {
    auto last_node_builder = [&]() -> bg::ValueHolderPtr { return bg::ValueHolder::CreateVoid<ValueHolder>("LastExec", {c0}); };
    auto last_holder =
        bg::FrameSelector::OnMainRootLast(last_node_builder);
    EXPECT_NE(last_holder, nullptr);
  }
  for (size_t node_size = 0U; node_size < 2; ++node_size) {
    auto last_node_builder = [&]() -> std::vector<bg::ValueHolderPtr> { return {bg::ValueHolder::CreateVoid<ValueHolder>("LastEventExec", {data0})}; };
    auto last_holders =
        bg::FrameSelector::OnMainRootLastEventSync(last_node_builder);
    EXPECT_FALSE(last_holders.empty());
  }

  auto main_frame = ValueHolder::PopGraphFrame();
  std::vector<ValueHolderPtr> last_exe_nodes = main_frame->GetLastExecNodes();
  EXPECT_EQ(last_exe_nodes.size(), 2);

  auto stage_ids_2_pcalls = main_frame->GetExecuteGraph()->GetExtAttr<std::vector<ValueHolderPtr>>(kStageIdsToLastPartitionedCall);
  EXPECT_NE(stage_ids_2_pcalls, nullptr);
  auto last_event_sync_exe_node = stage_ids_2_pcalls->at(static_cast<size_t>(OnMainRootLastExecStage::kLastEventSyncStage));
  EXPECT_NE(last_event_sync_exe_node, nullptr);

  auto sub_exe_graph = ge::FastNodeUtils::GetSubgraphFromNode(last_event_sync_exe_node->GetFastNode(), 0U);
  EXPECT_NE(sub_exe_graph, nullptr);
  auto last_exec_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(sub_exe_graph, "LastEventExec");
  EXPECT_NE(last_exec_node, nullptr);
  ASSERT_EQ(FastNodeTopoChecker(last_exec_node).StrictConnectFrom({{"InnerData", 0}}), "success");
  ASSERT_NE(FastNodeTopoChecker(last_exec_node).StrictConnectTo(-1, {{"InnerNetoutput", 0}}), "success");

  ASSERT_EQ(ExeGraphSummaryChecker(sub_exe_graph)
                .StrictDirectNodeTypes({{"LastEventExec", 2}, {"InnerData", 1}, {"InnerNetOutput", 1}}),
            "success");

  ASSERT_EQ(ExeGraphSummaryChecker(main_frame->GetExecuteGraph().get())
                .StrictDirectNodeTypes({{"Data", 1},
                                        {"Const", 1},
                                        {"Bar1", 1},
                                        {"Bar2", 1},
                                        {"LastExec", 2},
                                        {"PartitionedCall", 1},
                                        {"NetOutput", 1}}),
            "success");
  ASSERT_EQ(FastNodeTopoChecker(output).StrictConnectFrom({{"Bar1"}, {"Bar2"}}), "success");
}

/*
 *      data0    c0                       c0                          data0   c0
 *        \      /                      /     \                         \    /
 *          bar1     last_node:        L0    L1               ====>       bar1
 *           |                                                           /    \c
 *       netoutput                                                      |       noop
 *                                                                      |        /c \c
 *                                                                      |      L0    L1
 *                                                                      |        \c  /c
 *                                                                      |        noop
 *                                                                       \       /c
 *                                                                       netoutput
 *                                                                       这里L0和L1分别有来自c0的数据输入。示意图中忽略
 */
TEST_F(FastFrameSelectorUT, OnMainRootLast_SetLastExecNode_TwoNodeInsideOneLayer_LastResourceCleanStage) {
  InitTestFrames();
  // build exe graph on main frame
  auto data0 = ValueHolder::CreateFeed(0);
  auto c0 = ValueHolder::CreateConst("ConstData", 10, true);
  auto bar1 = ValueHolder::CreateSingleDataOutput("Bar1", {c0, data0});
  auto output = ValueHolder::CreateSingleDataOutput("NetOutput", {bar1});

  auto last0_node_builder = [&]() -> std::vector<bg::ValueHolderPtr> {
    return {bg::ValueHolder::CreateVoid<ValueHolder>("L0", {c0})};
  };
  auto last0_holder = bg::FrameSelector::OnMainRootLastResourceClean(last0_node_builder);
  EXPECT_FALSE(last0_holder.empty());
  auto last1_node_builder = [&]() -> std::vector<bg::ValueHolderPtr> {
    return {bg::ValueHolder::CreateVoid<ValueHolder>("L1", {c0})};
  };
  auto last1_holder = bg::FrameSelector::OnMainRootLastResourceClean(last1_node_builder);
  EXPECT_FALSE(last1_holder.empty());
  auto last_exec_nodes = bg::ValueHolder::GetLastExecNodes();
  EXPECT_EQ(last_exec_nodes.size(), 0U);

  auto main_frame = ValueHolder::PopGraphFrame();
  ASSERT_EQ(
      ExeGraphSummaryChecker(main_frame->GetExecuteGraph().get())
          .StrictDirectNodeTypes({{"Data", 1}, {"Const", 1}, {"Bar1", 1}, {"PartitionedCall", 1}, {"NetOutput", 1}}),
      "success");

  auto stage_ids_2_pcalls = main_frame->GetExecuteGraph()->GetExtAttr<std::vector<ValueHolderPtr>>(kStageIdsToLastPartitionedCall);
  EXPECT_NE(stage_ids_2_pcalls, nullptr);
  auto last_resource_clean_exe_node = stage_ids_2_pcalls->at(static_cast<size_t>(OnMainRootLastExecStage::kLastResourceClean));
  EXPECT_NE(last_resource_clean_exe_node, nullptr);

  auto sub_exe_graph = ge::FastNodeUtils::GetSubgraphFromNode(last_resource_clean_exe_node->GetFastNode(), 0U);
  EXPECT_NE(sub_exe_graph, nullptr);
  auto last_exec_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(sub_exe_graph, "L0");
  EXPECT_NE(last_exec_node, nullptr);
  ASSERT_EQ(FastNodeTopoChecker(last_exec_node).StrictConnectFrom({{"InnerData", 0}}), "success");
  ASSERT_NE(FastNodeTopoChecker(last_exec_node).StrictConnectTo(-1, {{"InnerNetoutput", 0}}), "success");

  ASSERT_EQ(ExeGraphSummaryChecker(sub_exe_graph)
                .StrictDirectNodeTypes({{"L0", 1}, {"L1", 1}, {"InnerData", 1}, {"InnerNetOutput", 1}}),
            "success");
  ASSERT_EQ(FastNodeTopoChecker(output).StrictConnectFrom({{"Bar1"}}), "success");
}

/*
 *      data0    c0                c0
 *        \      /                  |
 *          bar1     first_node: FirstExecNode
 *           |
 *       netoutput
 *                    |
 *                    V
 *
 *  +----------------------------------+    +-------------------------+
 *  |    data0    c0                   |    |        InnerData        |
 *  |      \      / \                  |    |          |              |
 *  |        bar1    PartitionedCall   |    |     FirstExecNode       |
 *  |         |                        |    |           |             |
 *  |     netoutput                    |    |       InnerNetoutput    |
 *  +----------------------------------+    +-------------------------+
 */
TEST_F(FastFrameSelectorUT, OnMainRootLast_SetFirstExecNode_OneNodeInsideOneLayer_OnePriority) {
  dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 1);
  InitTestFrames();
  // build exe graph on main frame
  auto data0 = ValueHolder::CreateFeed(0);
  auto c0 = ValueHolder::CreateConst("ConstData", 10, true);
  auto bar1 = ValueHolder::CreateSingleDataOutput("Bar1", {c0, data0});
  auto output = ValueHolder::CreateSingleDataOutput("NetOutput", {bar1});

  auto first_node_builder = [&]() -> std::vector<bg::ValueHolderPtr> {
    return {bg::ValueHolder::CreateVoid<ValueHolder>("FirstExecNode", {c0})};
  };
  auto first_holders = bg::FrameSelector::OnMainRootFirst(first_node_builder);
  EXPECT_FALSE(first_holders.empty());

  auto main_frame = ValueHolder::PopGraphFrame();
  auto stage_ids_2_pcalls = main_frame->GetExecuteGraph()->GetExtAttr<std::vector<ValueHolderPtr>>(kStageIdsToFirstPartitionedCall);
  EXPECT_NE(stage_ids_2_pcalls, nullptr);
  auto first_exec_partitioncall = stage_ids_2_pcalls->at(static_cast<size_t>(OnMainRootFirstExecStage::kFirstEventSyncStage));
  EXPECT_NE(first_exec_partitioncall, nullptr);

  auto sub_exe_graph = ge::FastNodeUtils::GetSubgraphFromNode(first_exec_partitioncall->GetFastNode(), 0U);
  EXPECT_NE(sub_exe_graph, nullptr);
  auto first_exec_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(sub_exe_graph, "FirstExecNode");
  EXPECT_NE(first_exec_node, nullptr);
  ASSERT_EQ(FastNodeTopoChecker(first_exec_node).StrictConnectFrom({{"InnerData", 0}}), "success");
  ASSERT_NE(FastNodeTopoChecker(first_exec_node).StrictConnectTo(-1, {{"InnerNetoutput", 0}}), "success");

  ASSERT_EQ(ExeGraphSummaryChecker(sub_exe_graph)
                .StrictDirectNodeTypes({{"FirstExecNode", 1}, {"InnerData", 1}, {"InnerNetOutput", 1}}),
            "success");

  ASSERT_EQ(ExeGraphSummaryChecker(main_frame->GetExecuteGraph().get())
                .StrictDirectNodeTypes(
                    {{"Data", 1}, {"Const", 1}, {"Bar1", 1}, {"PartitionedCall", 1}, {"NetOutput", 1}}),
            "success");
  ASSERT_EQ(FastNodeTopoChecker(data0).StrictConnectTo(0, {{"Bar1"}}), "success");
  ASSERT_NE(FastNodeTopoChecker(data0).StrictConnectTo(-1, {{"NoOp", -1}}), "success");
}

/*
 *      data0     c0                       c0
 *       / \      /                      /     \
 *    bar1   bar2       first_priority0:  L0    L1
 *        \   |
 *       netoutput
 *
 *                                   |
 *                                   V
 *  +----------------------------------+    +-------------------------+
 *  |    data0     c0                  |    |        InnerData        |
 *  |      / \     /\                  |    |          / \            |
 *  |   bar1   bar2  PartitionedCall   |    |       L0    L1          |
 *  |      \   |                       |    |         \   /           |
 *  |     netoutput                    |    |       InnerNetoutput    |
 *  +----------------------------------+    +-------------------------+
 */
TEST_F(FastFrameSelectorUT, OnMainRootLast_SetFirstExecNode_TwoNodeInsideEachLayer_TwoPriority) {
  InitTestFrames();
  // build exe graph on main frame
  auto data0 = ValueHolder::CreateFeed(0);
  auto c0 = ValueHolder::CreateConst("ConstData", 10, true);
  auto bar1 = ValueHolder::CreateSingleDataOutput("Bar1", {data0});
  auto bar2 = ValueHolder::CreateSingleDataOutput("Bar2", {data0, c0});
  auto output = ValueHolder::CreateSingleDataOutput("NetOutput", {bar1, bar2});

  for (size_t node_size=0U; node_size < 2; ++node_size) {
    auto first_node_builder = [&]() -> std::vector<bg::ValueHolderPtr> {
      return {bg::ValueHolder::CreateVoid<ValueHolder>("FirstExec", {c0})};
    };
    auto first_holders = bg::FrameSelector::OnMainRootFirst(first_node_builder);
    EXPECT_FALSE(first_holders.empty());
  }

  auto main_frame = ValueHolder::PopGraphFrame();

  auto stage_ids_2_pcalls = main_frame->GetExecuteGraph()->GetExtAttr<std::vector<ValueHolderPtr>>(kStageIdsToFirstPartitionedCall);
  EXPECT_NE(stage_ids_2_pcalls, nullptr);
  auto first_exec_partitioncall = stage_ids_2_pcalls->at(static_cast<size_t>(OnMainRootFirstExecStage::kFirstEventSyncStage));
  EXPECT_NE(first_exec_partitioncall, nullptr);

  auto sub_exe_graph = ge::FastNodeUtils::GetSubgraphFromNode(first_exec_partitioncall->GetFastNode(), 0U);
  EXPECT_NE(sub_exe_graph, nullptr);
  ASSERT_EQ(ExeGraphSummaryChecker(sub_exe_graph)
                .StrictDirectNodeTypes({{"InnerData", 1},
                                        {"FirstExec", 2},
                                        {"InnerNetOutput", 1}}),
            "success");
  auto first_exec_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(sub_exe_graph, "FirstExec");
  EXPECT_NE(first_exec_node, nullptr);
  ASSERT_EQ(FastNodeTopoChecker(first_exec_node).StrictConnectFrom({{"InnerData", 0}}), "success");
  ASSERT_NE(FastNodeTopoChecker(first_exec_node).StrictConnectTo(-1, {{"InnerNetoutput", 0}}), "success");


  ASSERT_EQ(ExeGraphSummaryChecker(main_frame->GetExecuteGraph().get())
                .StrictDirectNodeTypes({{"Data", 1},
                                        {"Const", 1},
                                        {"Bar1", 1},
                                        {"Bar2", 1},
                                        {"PartitionedCall", 1},
                                        {"NetOutput", 1}}),
            "success");
  ASSERT_EQ(FastNodeTopoChecker(output).StrictConnectFrom({{"Bar1"}, {"Bar2"}}), "success");
}

TEST_F(FastFrameSelectorUT, SelectDeInitRoot_CreateOnDeInitRoot_CurrentFrameIsRoot) {
  InitTestFrames();
  auto frame = ValueHolder::PopGraphFrame();

  // on RootGraph
  const auto &bars = FrameSelector::OnDeInitRoot([]() -> std::vector<ValueHolderPtr> {
    auto bar1 = ValueHolder::CreateVoid<ValueHolder>("DavinviModelFinalize", {});
    return {bar1};
  });
  ASSERT_EQ(bars.size(), 1);
  ASSERT_NE(bars[0], nullptr);

  ASSERT_EQ(ExeGraphSummaryChecker(de_init_frame->GetExecuteGraph().get())
            .StrictDirectNodeTypes({{"DavinviModelFinalize", 1}}),
            "success");

  ASSERT_TRUE(bars[0]->GetFastNode()->GetAllOutNodes().empty());

  const auto &c1 = ValueHolder::CreateConst("ConstData", 10, true);
  const auto &c2 = ValueHolder::CreateConst("ConstData", 10, true);
  ValueHolder::PushGraphFrame(frame.release());
  ASSERT_EQ(bars[0]->AppendInputs({c1, c2}), 0);
  ASSERT_EQ(ExeGraphSummaryChecker(de_init_frame->GetExecuteGraph().get())
                .StrictDirectNodeTypes({{"DavinviModelFinalize", 1}, {"InnerData", 2}}),
            "success");
  const auto &c3 = ValueHolder::CreateConst("ConstData", 10, true);
  ASSERT_EQ(c3->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr()->GetParentNodeBarePtr()->GetType(), "Main");
}

TEST_F(FastFrameSelectorUT, SelectDeInitRoot_CreateOnDeInitRoot_CurrentFrameIsMain) {
  InitTestFrames();

  // on RootGraph
  const auto &bars = FrameSelector::OnDeInitRoot([]() -> std::vector<ValueHolderPtr> {
    auto bar1 = ValueHolder::CreateVoid<ValueHolder>("DavinviModelFinalize", {});
    return {bar1};
  });
  ASSERT_EQ(bars.size(), 1);
  ASSERT_NE(bars[0], nullptr);

  ASSERT_EQ(ExeGraphSummaryChecker(de_init_frame->GetExecuteGraph().get())
            .StrictDirectNodeTypes({{"DavinviModelFinalize", 1}}),
            "success");

  ASSERT_TRUE(bars[0]->GetFastNode()->GetAllOutNodes().empty());

  const auto &c1 = ValueHolder::CreateConst("ConstData", 10, true);
  ASSERT_EQ(c1->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr()->GetParentNodeBarePtr()->GetType(), "Main");

  ASSERT_NE(bars[0]->AppendInputs({c1}), 0);
}

TEST_F(FastFrameSelectorUT, SelectDeInitRoot_Failed_BuilderIsNullptr) {
  ASSERT_EQ(FrameSelector::OnDeInitRoot(nullptr).size(), 0);
}
}  // namespace bg
}  // namespace gert

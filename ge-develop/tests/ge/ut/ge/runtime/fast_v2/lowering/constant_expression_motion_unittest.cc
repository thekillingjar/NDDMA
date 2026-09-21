/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "lowering/pass/constant_expression_motion.h"

#include <gtest/gtest.h>

#include "exe_graph/lowering/value_holder.h"
#include "exe_graph/lowering/frame_selector.h"
#include "graph/utils/fast_node_utils.h"
#include "graph/utils/execute_graph_utils.h"
#include "graph/utils/execute_graph_adapter.h"
#include "graph/utils/graph_dump_utils.h"
#include "register/node_converter_registry.h"

#include "graph_builder/bg_condition.h"
#include "lowering/graph_converter.h"

#include "common/bg_test.h"
#include "common/summary_checker.h"
#include "common/topo_checker.h"
#include "common/share_graph.h"
#include "faker/global_data_faker.h"
#include "depends/op_stub/op_impl/less_important_op_impl.h"
#include "stub/gert_runtime_stub.h"
#include "framework/runtime/gert_const_types.h"
#include "faker/model_desc_holder_faker.h"

namespace gert {
namespace bg {
namespace {
void UnlinkInEdgeByIndex(ge::FastNode *const node, int32_t index) {
  auto owner_graph = node->GetExtendInfo()->GetOwnerGraphBarePtr();
  ASSERT_NE(owner_graph, nullptr);
  auto edge = node->GetInDataEdgeByIndex(index);
  if (edge != nullptr) {
    ASSERT_EQ(owner_graph->RemoveEdge(edge), ge::GRAPH_SUCCESS);
  }
}
/*
 *          Launch   <----- addresses
 *       /         \
 *    Bar          AllocMemory --> [addresses]
 *   /  \            |
 * Foo  Const2    InferShape   --> [shapes]
 *  |                |
 * Const1          Shapes
 */
LowerResult Converter1(const ge::NodePtr &node, const LowerInput &lower_input) {
  auto c1 = ValueHolder::CreateConst("Hello", 5, false);
  auto foo = ValueHolder::CreateSingleDataOutput("Foo", {c1});
  auto c2 = ValueHolder::CreateConst("World", 5, false);
  auto bar = ValueHolder::CreateSingleDataOutput("BarCond", {foo, c2});
  auto shapes = ValueHolder::CreateDataOutput("InferShape", lower_input.input_shapes, node->GetAllOutDataAnchorsSize());
  auto addresses =
      DevMemValueHolder::CreateDataOutput("AllocMemory", shapes, shapes.size(), node->GetOpDesc()->GetStreamId());
  for (const auto &addr : addresses) {
    addr->SetPlacement(kOnHost);
  }
  std::vector<ValueHolderPtr> inputs;
  inputs.push_back(bar);
  inputs.insert(inputs.end(), addresses.begin(), addresses.end());
  inputs.insert(inputs.end(), lower_input.input_addrs.begin(), lower_input.input_addrs.end());
  auto launch = ValueHolder::CreateVoid<bg::ValueHolder>("Launch", inputs);
  return {HyperStatus::Success(), {launch}, shapes, addresses};
}
/*
 *                        Launch   <----- addresses
 *                     /         \
 * [addresses] <--  Bar          AllocMemory
 *                 /  \            |
 *               Foo  Const2    InferShape   --> [shapes]
 *                |                |
 *               Const1          Shapes
 */
LowerResult Converter2(const ge::NodePtr &node, const LowerInput &lower_input) {
  auto c1 = ValueHolder::CreateConst("Hello", 5, false);
  auto foo = ValueHolder::CreateSingleDataOutput("Foo", {c1});
  auto c2 = ValueHolder::CreateConst("World", 5, false);
  auto bar = DevMemValueHolder::CreateSingleDataOutput("BarBody", {foo, c2}, node->GetOpDesc()->GetStreamId());
  auto shapes = ValueHolder::CreateDataOutput("InferShape", lower_input.input_shapes, node->GetAllOutDataAnchorsSize());
  auto addresses = ValueHolder::CreateDataOutput("AllocMemory", shapes, shapes.size());
  std::vector<ValueHolderPtr> inputs;
  inputs.push_back(bar);
  inputs.insert(inputs.end(), addresses.begin(), addresses.end());
  inputs.insert(inputs.end(), lower_input.input_addrs.begin(), lower_input.input_addrs.end());
  auto launch = ValueHolder::CreateVoid<bg::ValueHolder>("Launch", inputs);
  return {HyperStatus::Success(), {launch}, shapes, {bar}};
}
LowerResult Converter3(const ge::NodePtr &node, const LowerInput &lower_input) {
  std::vector<ValueHolderPtr> inputs;
  inputs.push_back(lower_input.global_data->GetStreamById(0));
  inputs.insert(inputs.cend(), lower_input.input_shapes.begin(), lower_input.input_shapes.end());
  inputs.insert(inputs.cend(), lower_input.input_addrs.begin(), lower_input.input_addrs.end());
  std::string name = "Execute_";
  auto out_count = node->GetAllOutDataAnchorsSize();
  auto outputs =
      DevMemValueHolder::CreateDataOutput("Execute_", inputs, out_count * 2, node->GetOpDesc()->GetStreamId());
  return {HyperStatus::Success(),
          {*outputs.begin()},
          {outputs.begin(), outputs.begin() + out_count},
          {outputs.begin() + out_count, outputs.end()}};
}
}  // namespace
class ConstantExpressionsMotionUT : public BgTest {
 protected:
  void SetUp() override {
    ValueHolder::PushGraphFrame();
    auto init = ValueHolder::CreateVoid<bg::ValueHolder>("Init", {});
    auto main = ValueHolder::CreateVoid<bg::ValueHolder>("Main", {});
    auto de_init = ValueHolder::CreateVoid<bg::ValueHolder>("DeInit", {});

    ValueHolder::PushGraphFrame(init, "Init");
    init_frame_ = ValueHolder::PopGraphFrame({}, {});

    ValueHolder::PushGraphFrame(de_init, "DeInit");
    de_init_frame_ = ValueHolder::PopGraphFrame();

    ValueHolder::PushGraphFrame(main, "Main");
  }

  void TearDown() override {
    BgTest::TearDown();
    init_frame_.reset();
    de_init_frame_.reset();
  }

  std::unique_ptr<GraphFrame> init_frame_;
  std::unique_ptr<GraphFrame> de_init_frame_;

  void InitTestFramesWithStream(LoweringGlobalData &global_data, int64_t stream_num = 1) {
    auto init_out = bg::FrameSelector::OnInitRoot([&stream_num, &global_data]() -> std::vector<bg::ValueHolderPtr> {
      global_data.LoweringAndSplitRtStreams(1);
      auto stream_num_holder = bg::ValueHolder::CreateConst(&stream_num, sizeof(stream_num));
      global_data.SetUniqueValueHolder(kGlobalDataModelStreamNum, stream_num_holder);
      return {};
    });
    global_data.LoweringAndSplitRtStreams(stream_num);
  }
};

/*
 * main graph:
 *
 *          NetOutput
 *             |
 *           Foo2
 *         /   \
 *      Foo1    data
 *     /   \
 * const   const
 */
TEST_F(ConstantExpressionsMotionUT, MoveToInit_Success_CeOnMain) {
  auto c1 = ValueHolder::CreateConst("Hello", 5, true);
  auto c2 = ValueHolder::CreateConst("World", 5, true);
  auto data = ValueHolder::CreateFeed(0);
  auto foo1 = ValueHolder::CreateSingleDataOutput("Foo", {c1, c2});
  auto foo2 = ValueHolder::CreateSingleDataOutput("Foo", {foo1, data});
  auto main_frame = ValueHolder::PopGraphFrame({foo2}, {}, "NetOutput");
  auto root_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(root_frame, nullptr);
  ASSERT_NE(main_frame, nullptr);

  bool changed = false;
  LoweringGlobalData global_data;
  ASSERT_EQ(ConstantExpressionMotion(&global_data).Run(root_frame->GetExecuteGraph().get(), changed),
            ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);

  EXPECT_EQ(ExeGraphSummaryChecker(main_frame->GetExecuteGraph().get())
                .StrictDirectNodeTypes(
                    std::map<std::string, size_t>{{"Data", 1}, {"Foo", 1}, {"InnerData", 1}, {"NetOutput", 1}}),
            "success");

  EXPECT_EQ(ExeGraphSummaryChecker(init_frame_->GetExecuteGraph().get())
                .StrictDirectNodeTypes(std::map<std::string, size_t>{{"Const", 2}, {"Foo", 1}, {"InnerNetOutput", 1}}),
            "success");
  EXPECT_EQ(de_init_frame_->GetExecuteGraph()->GetDirectNodesSize(), 0);

  ConnectFromInitToMain(foo1->GetFastNode(), 0, foo2->GetFastNode(), 0);

  changed = false;
  ASSERT_EQ(ConstantExpressionMotion(&global_data).Run(root_frame->GetExecuteGraph().get(), changed),
            ge::GRAPH_SUCCESS);
  ASSERT_FALSE(changed);
}
/*
 * main graph:
 *
 *               NetOutput
 *                  |
 *  +-----------> Bar2
 *  |            /   \
 *  +-----> Bar1     data
 *  |      /   \
 *  |   Foo1    const
 *  |  /   \
 * const   const
 */
TEST_F(ConstantExpressionsMotionUT, MoveToInit_PropagationSuccess_CeOnMain) {
  auto c1 = ValueHolder::CreateConst("Hello", 5, true);
  auto c2 = ValueHolder::CreateConst("World", 5, true);
  auto c3 = ValueHolder::CreateConst("World", 5, true);
  auto data = ValueHolder::CreateFeed(0);
  auto foo1 = ValueHolder::CreateSingleDataOutput("Foo", {c1, c2});
  auto bar1 = ValueHolder::CreateSingleDataOutput("Bar", {c1, foo1, c3});
  auto bar2 = ValueHolder::CreateSingleDataOutput("Bar", {c1, bar1, data});
  auto main_frame = ValueHolder::PopGraphFrame({bar2}, {}, "NetOutput");
  auto root_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(root_frame, nullptr);
  ASSERT_NE(main_frame, nullptr);

  bool changed = false;
  LoweringGlobalData global_data;
  ASSERT_EQ(ConstantExpressionMotion(&global_data).Run(root_frame->GetExecuteGraph().get(), changed),
            ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);

  EXPECT_EQ(ExeGraphSummaryChecker(main_frame->GetExecuteGraph().get())
                .StrictDirectNodeTypes(
                    std::map<std::string, size_t>{{"Data", 1}, {"Bar", 1}, {"InnerData", 2}, {"NetOutput", 1}}),
            "success");

  EXPECT_EQ(ExeGraphSummaryChecker(init_frame_->GetExecuteGraph().get())
                .StrictDirectNodeTypes(
                    std::map<std::string, size_t>{{"Const", 3}, {"Foo", 1}, {"Bar", 1}, {"InnerNetOutput", 1}}),
            "success");

  ConnectFromInitToMain(c1->GetFastNode(), 0, bar2->GetFastNode(), 0);
  ConnectFromInitToMain(bar1->GetFastNode(), 0, bar2->GetFastNode(), 1);
  EXPECT_EQ(de_init_frame_->GetExecuteGraph()->GetDirectNodesSize(), 0);

  changed = false;
  ASSERT_EQ(ConstantExpressionMotion(&global_data).Run(root_frame->GetExecuteGraph().get(), changed),
            ge::GRAPH_SUCCESS);
  ASSERT_FALSE(changed);
}
/*
 * main graph:
 *
 *                 NetOutput
 *               c    |
 *  Foo1Guarder <-- Foo2
 *            \   /   \
 *             Foo1    data
 *            /   \
 *        const   const
 */
TEST_F(ConstantExpressionsMotionUT, MoveToDeInit_Success_CeGuarderOnMain) {
  auto c1 = ValueHolder::CreateConst("Hello", 5, true);
  auto c2 = ValueHolder::CreateConst("World", 5, true);
  auto data = ValueHolder::CreateFeed(0);
  auto foo1 = ValueHolder::CreateSingleDataOutput("Foo", {c1, c2});
  auto foo1_guarder = ValueHolder::CreateVoidGuarder("FreeFoo", foo1, {});
  ASSERT_NE(foo1_guarder, nullptr);
  auto foo2 = ValueHolder::CreateSingleDataOutput("Foo", {foo1, data});
  auto main_frame = ValueHolder::PopGraphFrame({foo2}, {}, "NetOutput");
  auto root_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(root_frame, nullptr);
  ASSERT_NE(main_frame, nullptr);

  bool changed = false;
  LoweringGlobalData global_data;
  ASSERT_EQ(ConstantExpressionMotion(&global_data).Run(root_frame->GetExecuteGraph().get(), changed),
            ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);

  EXPECT_EQ(ExeGraphSummaryChecker(main_frame->GetExecuteGraph().get())
                .StrictDirectNodeTypes(
                    std::map<std::string, size_t>{{"Data", 1}, {"Foo", 1}, {"InnerData", 1}, {"NetOutput", 1}}),
            "success");

  EXPECT_EQ(ExeGraphSummaryChecker(init_frame_->GetExecuteGraph().get())
                .StrictDirectNodeTypes(std::map<std::string, size_t>{{"Const", 2}, {"Foo", 1}, {"InnerNetOutput", 1}}),
            "success");

  EXPECT_EQ(ExeGraphSummaryChecker(de_init_frame_->GetExecuteGraph().get())
                .StrictDirectNodeTypes(std::map<std::string, size_t>{{"InnerData", 1}, {"FreeFoo", 1}}),
            "success");

  ConnectFromInitToMain(foo1->GetFastNode(), 0, foo2->GetFastNode(), 0);
  ConnectFromInitToDeInit(foo1->GetFastNode(), 0, foo1_guarder->GetFastNode(), 0);

  changed = false;
  ASSERT_EQ(ConstantExpressionMotion(&global_data).Run(root_frame->GetExecuteGraph().get(), changed),
            ge::GRAPH_SUCCESS);
  ASSERT_FALSE(changed);
}
/*
 * main graph:
 *
 *    NetOutput
 *       |
 *      Foo1
 *     /   \
 * const   const
 */
TEST_F(ConstantExpressionsMotionUT, MoveToInit_NotMove_WhenNodeConnectToNetOutput) {
  auto c1 = ValueHolder::CreateConst("Hello", 5, true);
  auto c2 = ValueHolder::CreateConst("World", 5, true);
  auto foo1 = ValueHolder::CreateSingleDataOutput("Foo", {c1, c2});
  auto main_frame = ValueHolder::PopGraphFrame({foo1}, {}, "NetOutput");
  auto root_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(root_frame, nullptr);
  ASSERT_NE(main_frame, nullptr);

  bool changed = false;
  LoweringGlobalData global_data;
  ASSERT_EQ(ConstantExpressionMotion(&global_data).Run(root_frame->GetExecuteGraph().get(), changed),
            ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);

  EXPECT_EQ(ExeGraphSummaryChecker(main_frame->GetExecuteGraph().get())
                .StrictDirectNodeTypes(std::map<std::string, size_t>{{"Foo", 1}, {"InnerData", 2}, {"NetOutput", 1}}),
            "success");

  EXPECT_EQ(ExeGraphSummaryChecker(init_frame_->GetExecuteGraph().get())
                .StrictDirectNodeTypes(std::map<std::string, size_t>{{"Const", 2}, {"InnerNetOutput", 1}}),
            "success");
  EXPECT_EQ(de_init_frame_->GetExecuteGraph()->GetDirectNodesSize(), 0);

  ConnectFromInitToMain(c1->GetFastNode(), 0, foo1->GetFastNode(), 0);
  ConnectFromInitToMain(c2->GetFastNode(), 0, foo1->GetFastNode(), 1);

  changed = false;
  ASSERT_EQ(ConstantExpressionMotion(&global_data).Run(root_frame->GetExecuteGraph().get(), changed),
            ge::GRAPH_SUCCESS);
  ASSERT_FALSE(changed);
}
/*
 * main graph:
 *
 *               NetOutput
 *                  |
 *  +-----------> Bar2
 *  |            /   \
 *  +-----> Bar1 <-- data
 *  |      /   \   c
 *  |   Foo1    const
 *  |  /   \
 * const   const
 */
TEST_F(ConstantExpressionsMotionUT, MoveToInit_NotMove_WhenCeCtrlConnectFromNoCeNetOutput) {
  auto c1 = ValueHolder::CreateConst("Hello", 5, true);
  auto c2 = ValueHolder::CreateConst("World", 5, true);
  auto c3 = ValueHolder::CreateConst("World", 5, true);
  auto data = ValueHolder::CreateFeed(0);
  auto foo1 = ValueHolder::CreateSingleDataOutput("Foo", {c1, c2});
  auto bar1 = ValueHolder::CreateSingleDataOutput("Bar", {c1, foo1, c3});
  auto bar2 = ValueHolder::CreateSingleDataOutput("Bar", {c1, bar1, data});
  ValueHolder::AddDependency(data, bar1);
  auto main_frame = ValueHolder::PopGraphFrame({bar2}, {}, "NetOutput");
  auto root_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(root_frame, nullptr);
  ASSERT_NE(main_frame, nullptr);

  bool changed = false;
  LoweringGlobalData global_data;
  ASSERT_EQ(ConstantExpressionMotion(&global_data).Run(root_frame->GetExecuteGraph().get(), changed),
            ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);

  EXPECT_EQ(ExeGraphSummaryChecker(init_frame_->GetExecuteGraph().get())
                .StrictDirectNodeTypes(std::map<std::string, size_t>{{"Const", 3}, {"Foo", 1}, {"InnerNetOutput", 1}}),
            "success");

  EXPECT_EQ(ExeGraphSummaryChecker(main_frame->GetExecuteGraph().get())
                .StrictDirectNodeTypes(
                    std::map<std::string, size_t>{{"Data", 1}, {"Bar", 2}, {"InnerData", 3}, {"NetOutput", 1}}),
            "success");

  ConnectFromInitToMain(c1->GetFastNode(), 0, bar2->GetFastNode(), 0);
  ConnectFromInitToMain(c1->GetFastNode(), 0, bar1->GetFastNode(), 0);
  ConnectFromInitToMain(foo1->GetFastNode(), 0, bar1->GetFastNode(), 1);
  ConnectFromInitToMain(c3->GetFastNode(), 0, bar1->GetFastNode(), 2);
  EXPECT_EQ(de_init_frame_->GetExecuteGraph()->GetDirectNodesSize(), 0);

  changed = false;
  ASSERT_EQ(ConstantExpressionMotion(&global_data).Run(root_frame->GetExecuteGraph().get(), changed),
            ge::GRAPH_SUCCESS);
  ASSERT_FALSE(changed);
}
/*
 * main graph:
 *
 *                 NetOutput
 *               c    |
 *  Foo1Guarder <-- Foo2
 *   /        \   /   \
 * data       Foo1    data
 *            /   \
 *        const   const
 * todo: not support yet
 */
TEST_F(ConstantExpressionsMotionUT, MoveToInit_Failed_WhenCeGuarderConnectFromMain) {
  auto c1 = ValueHolder::CreateConst("Hello", 5, true);
  auto c2 = ValueHolder::CreateConst("World", 5, true);
  auto data0 = ValueHolder::CreateFeed(0);
  auto data1 = ValueHolder::CreateFeed(1);
  auto foo1 = ValueHolder::CreateSingleDataOutput("Foo", {c1, c2});
  auto foo1_guarder = ValueHolder::CreateVoidGuarder("FreeFoo", foo1, {data1});
  ASSERT_NE(foo1_guarder, nullptr);
  auto foo2 = ValueHolder::CreateSingleDataOutput("Foo", {foo1, data0});
  auto main_frame = ValueHolder::PopGraphFrame({foo2}, {}, "NetOutput");
  auto root_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(root_frame, nullptr);
  ASSERT_NE(main_frame, nullptr);

  // todo: 这个用例校验的时topo排序失败，是一个老用例，之前不支持，现在支持了但是还存在bug
  //       bug是存在跨图连边，从deinit图里面连接到了main图，导致topo排序报错。
  //       为了性能优化删除topo排序后，此处就校验不住了，暂时删除该用例，后续定位问题。
  // bool changed = false;
  // ASSERT_NE(ConstantExpressionMotion().Run(root_frame->GetExecuteGraph().get(), changed), ge::GRAPH_SUCCESS);
}
ge::FastNode *AddNode(ge::ExecuteGraph *const graph, const char *node_type, int32_t in_count, int32_t out_count) {
  static int32_t count = 0;
  auto op_desc =
      std::make_shared<ge::OpDesc>(std::string("NodeInUT_") + node_type + "_" + std::to_string(count++), node_type);
  for (int32_t i = 0; i < in_count; ++i) {
    op_desc->AddInputDesc(ge::GeTensorDesc());
  }
  for (int32_t i = 0; i < out_count; ++i) {
    op_desc->AddOutputDesc(ge::GeTensorDesc());
  }
  return graph->AddNode(op_desc);
}
ge::FastNode *AddConstNode(ge::ExecuteGraph *const graph, const void *const_data, size_t const_data_len) {
  static int32_t count = 0;
  auto op_desc = std::make_shared<ge::OpDesc>("ConstInUT_" + std::to_string(count++), "Const");
  op_desc->AddOutputDesc(ge::GeTensorDesc());
  auto buf = ge::Buffer::CopyFrom(static_cast<const uint8_t *>(const_data), const_data_len);
  ge::AttrUtils::SetZeroCopyBytes(op_desc, "value", std::move(buf));
  return graph->AddNode(op_desc);
}
/*
 * main graph first time:
 *          NetOutput
 *             |
 *           Foo2
 *         /   \
 *      Foo1    data
 *     /   \
 * const   const
 *
 * after CEM first time:
 *       NetOutput
 *          |
 *         Foo2
 *         / \
 * InnerData  data
 *
 * new CE:
 *       NetOutput
 *          |
 *         Foo2
 *         / \
 * InnerData  Foo3
 *           /   \
 *      const    const
 *
 *
 * after CEM second Time:
 *       NetOutput
 *          |
 *         Foo2
 *         / \
 * InnerData  InnerData
 *
 */
TEST_F(ConstantExpressionsMotionUT, MoveToInit_Succeess_PassSecondTimeHaveNewCE1) {
  auto c1 = ValueHolder::CreateConst("Hello", 5, true);
  auto c2 = ValueHolder::CreateConst("World", 5, true);
  auto data = ValueHolder::CreateFeed(0);
  auto foo1 = ValueHolder::CreateSingleDataOutput("Foo", {c1, c2});
  auto foo2 = ValueHolder::CreateSingleDataOutput("Foo", {foo1, data});
  auto main_frame = ValueHolder::PopGraphFrame({foo2}, {}, "NetOutput");
  auto root_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(root_frame, nullptr);
  ASSERT_NE(main_frame, nullptr);

  // CEM the first time
  bool changed = false;
  LoweringGlobalData global_data;
  auto pass = ConstantExpressionMotion(&global_data);
  ASSERT_EQ(pass.Run(root_frame->GetExecuteGraph().get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);

  // New CE
  auto main_graph = main_frame->GetExecuteGraph();
  auto c3 = AddConstNode(main_graph.get(), "HELLO", 5);
  auto c4 = AddConstNode(main_graph.get(), "HELLO", 5);
  auto foo3 = AddNode(main_graph.get(), "Foo", 2, 1);
  ASSERT_NE(main_graph->AddEdge(c3, 0, foo3, 0), nullptr);
  ASSERT_NE(main_graph->AddEdge(c4, 0, foo3, 1), nullptr);
  UnlinkInEdgeByIndex(foo2->GetFastNode(), 1);
  ASSERT_NE(main_graph->AddEdge(foo3, 0, foo2->GetFastNode(), 1), nullptr);

  // CEM the second time
  changed = false;
  ASSERT_EQ(pass.Run(root_frame->GetExecuteGraph().get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);

  // check summary
  EXPECT_EQ(ExeGraphSummaryChecker(init_frame_->GetExecuteGraph().get())
                .StrictDirectNodeTypes(
                    std::map<std::string, size_t>{{"Const", 4}, {"Foo", 2}, {"InnerNetOutput", 1}, {"NoOp", 1}}),
            "success");

  EXPECT_EQ(ExeGraphSummaryChecker(main_frame->GetExecuteGraph().get())
                .StrictDirectNodeTypes(
                    std::map<std::string, size_t>{{"Data", 1}, {"Foo", 1}, {"InnerData", 2}, {"NetOutput", 1}}),
            "success");
  EXPECT_EQ(de_init_frame_->GetExecuteGraph()->GetDirectNodesSize(), 0);

  ConnectFromInitToMain(foo1->GetFastNode(), 0, foo2->GetFastNode(), 0);
  ConnectFromInitToMain(foo3, 0, foo2->GetFastNode(), 1);

  changed = false;
  ASSERT_EQ(pass.Run(root_frame->GetExecuteGraph().get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_FALSE(changed);
}
/*
 * main graph first time:
 *          NetOutput
 *             |
 *           Foo2
 *         /   \
 *      Foo1    data
 *     /   \
 * const   const
 *
 * after CEM first time:
 *       NetOutput
 *          |
 *         Foo2
 *         / \
 * InnerData  data
 *
 * new CE:
 *          NetOutput
 *             |
 *            Foo4
 *           /     \
 *        Foo3     Foo2
 *        /  \     / \
 *   const InnerData  data
 *
 * after CEM second Time:
 *          NetOutput
 *             |
 *       +--------> Foo4
 *       |            \
 *       |             Foo2
 *       |            /    \
 *   InnerData  InnerData   data
 *
 */
// TEST_F(ConstantExpressionsMotionUT, MoveToInit_Succeess_PassSecondTimeHaveNewCE2) {
//   // todo not support yet
// }
/*
 * main graph:
 *  NetOutput
 *    |
 *  Const
 */
TEST_F(ConstantExpressionsMotionUT, MoveToInit_NotMove_ConstConnectToNetOutput) {
  auto c1 = ValueHolder::CreateConst("Hello", 5, true);
  auto main_frame = ValueHolder::PopGraphFrame({c1}, {}, "NetOutput");
  auto root_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(root_frame, nullptr);
  ASSERT_NE(main_frame, nullptr);

  bool changed = false;
  LoweringGlobalData global_data;
  ASSERT_EQ(ConstantExpressionMotion(&global_data).Run(root_frame->GetExecuteGraph().get(), changed),
            ge::GRAPH_SUCCESS);
  ASSERT_FALSE(changed);

  EXPECT_EQ(ExeGraphSummaryChecker(main_frame->GetExecuteGraph().get())
                .StrictDirectNodeTypes(std::map<std::string, size_t>{{"Const", 1}, {"NetOutput", 1}}),
            "success");

  EXPECT_EQ(ExeGraphSummaryChecker(init_frame_->GetExecuteGraph().get())
                .StrictDirectNodeTypes(std::map<std::string, size_t>{{"InnerNetOutput", 1}}),
            "success");
  EXPECT_EQ(de_init_frame_->GetExecuteGraph()->GetAllNodes().size(), 0);
}
/*
 * main graph:
 * NetOutput                          NetOutput
 *     |                                   |
 *    if      =>                          if
 *    |                                 /   \
 *  data             InnerData(from init)  data
 *
 * then graph:
 *        InnerNetOutput            InnerNetOutput
 *             |                         |
 *           Foo2           =>         Foo2
 *         /   \                       /  \
 *      Foo1    InnerData      InnerData  InnerData
 *     /   \
 * const   const
 *
 */
TEST_F(ConstantExpressionsMotionUT, If_Success_CeOnThen) {
  auto data = ValueHolder::CreateFeed(0);
  auto if_outputs = bg::If<bg::ValueHolder>(
      data,
      [&]() -> std::vector<ValueHolderPtr> {
        auto c1 = ValueHolder::CreateConst("Hello", 5, true);
        auto c2 = ValueHolder::CreateConst("Hello", 5, true);
        auto foo1 = ValueHolder::CreateSingleDataOutput("Foo", {c1, c2});
        auto foo2 = ValueHolder::CreateSingleDataOutput("Foo", {foo1, data});
        return {foo2};
      },
      [&]() -> std::vector<ValueHolderPtr> { return {ValueHolder::CreateSingleDataOutput("Foo", {data})}; });
  auto main_frame = ValueHolder::PopGraphFrame(if_outputs, {}, "NetOutput");
  auto root_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(root_frame, nullptr);
  ASSERT_NE(main_frame, nullptr);

  bool changed = false;
  LoweringGlobalData global_data;
  ASSERT_EQ(ConstantExpressionMotion(&global_data).Run(root_frame->GetExecuteGraph().get(), changed),
            ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);

  EXPECT_EQ(ExeGraphSummaryChecker(main_frame->GetExecuteGraph().get())
                .StrictDirectNodeTypes(
                    std::map<std::string, size_t>{{"Data", 1}, {"If", 1}, {"InnerData", 1}, {"NetOutput", 1}}),
            "success");
  EXPECT_EQ(ExeGraphSummaryChecker(init_frame_->GetExecuteGraph().get())
                .StrictDirectNodeTypes(std::map<std::string, size_t>{{"Const", 2}, {"Foo", 1}, {"InnerNetOutput", 1}}),
            "success");
  EXPECT_EQ(de_init_frame_->GetExecuteGraph()->GetDirectNodesSize(), 0);

  auto if_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(main_frame->GetExecuteGraph().get(), "If");
  ASSERT_NE(if_node, nullptr);
  auto then_graph = ge::FastNodeUtils::GetSubgraphFromNode(if_node, 1U);
  ASSERT_NE(then_graph, nullptr);
  auto else_graph = ge::FastNodeUtils::GetSubgraphFromNode(if_node, 2U);
  ASSERT_NE(else_graph, nullptr);

  EXPECT_EQ(
      ExeGraphSummaryChecker(then_graph)
          .StrictDirectNodeTypes(std::map<std::string, size_t>{{"Foo", 1}, {"InnerData", 2}, {"InnerNetOutput", 1}}),
      "success");
  EXPECT_EQ(
      ExeGraphSummaryChecker(else_graph)
          .StrictDirectNodeTypes(std::map<std::string, size_t>{{"InnerData", 1}, {"Foo", 1}, {"InnerNetOutput", 1}}),
      "success");

  auto foo2_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(then_graph, "Foo");
  auto foo1_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_frame_->GetExecuteGraph().get(), "Foo");
  ASSERT_NE(foo2_node, nullptr);
  ASSERT_NE(foo1_node, nullptr);
  ConnectFromInitToMain(foo1_node, 0, foo2_node, 0);

  changed = false;
  ASSERT_EQ(ConstantExpressionMotion(&global_data).Run(root_frame->GetExecuteGraph().get(), changed),
            ge::GRAPH_SUCCESS);
  ASSERT_FALSE(changed);
}

TEST_F(ConstantExpressionsMotionUT, If_Success_OnlyConstConnectToIf) {
  auto c0 = ValueHolder::CreateConst("Hello", 5, true);;
  auto if_outputs = bg::If<bg::ValueHolder>(
      c0,
      [&]() -> std::vector<ValueHolderPtr> {
        auto c1 = ValueHolder::CreateConst("Hello", 5, true);
        auto foo1 = ValueHolder::CreateSingleDataOutput("Foo1", {c1});
        return {foo1};
      },
      [&]() -> std::vector<ValueHolderPtr> {
        auto c2 = ValueHolder::CreateConst("Hello", 5, true);
        return {ValueHolder::CreateSingleDataOutput("Foo2", {c2})};
      });
  auto foo3 = ValueHolder::CreateSingleDataOutput("Foo3", {if_outputs});
  auto main_frame = ValueHolder::PopGraphFrame({foo3}, {}, "NetOutput");
  auto root_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(root_frame, nullptr);
  ASSERT_NE(main_frame, nullptr);

  bool changed = false;
  LoweringGlobalData global_data;
  ASSERT_EQ(ConstantExpressionMotion(&global_data).Run(root_frame->GetExecuteGraph().get(), changed),
            ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);

  EXPECT_EQ(ExeGraphSummaryChecker(main_frame->GetExecuteGraph().get())
                .StrictDirectNodeTypes(
                    std::map<std::string, size_t>{{"Foo3", 1}, {"If", 1}, {"InnerData", 3}, {"NetOutput", 1}}),
            "success");
  EXPECT_EQ(ExeGraphSummaryChecker(init_frame_->GetExecuteGraph().get())
                .StrictDirectNodeTypes(std::map<std::string, size_t>{{"Const", 3}, {"InnerNetOutput", 1}}),
            "success");
  EXPECT_EQ(de_init_frame_->GetExecuteGraph()->GetDirectNodesSize(), 0);

  auto if_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(main_frame->GetExecuteGraph().get(), "If");
  ASSERT_NE(if_node, nullptr);
  auto then_graph = ge::FastNodeUtils::GetSubgraphFromNode(if_node, 1U);
  ASSERT_NE(then_graph, nullptr);
  auto else_graph = ge::FastNodeUtils::GetSubgraphFromNode(if_node, 2U);
  ASSERT_NE(else_graph, nullptr);

  EXPECT_EQ(
      ExeGraphSummaryChecker(then_graph)
          .StrictDirectNodeTypes(std::map<std::string, size_t>{{"Foo1", 1}, {"InnerData", 1}, {"InnerNetOutput", 1}}),
      "success");
  EXPECT_EQ(
      ExeGraphSummaryChecker(else_graph)
          .StrictDirectNodeTypes(std::map<std::string, size_t>{{"InnerData", 1}, {"Foo2", 1}, {"InnerNetOutput", 1}}),
      "success");

  changed = false;
  ASSERT_EQ(ConstantExpressionMotion(&global_data).Run(root_frame->GetExecuteGraph().get(), changed),
            ge::GRAPH_SUCCESS);
  ASSERT_FALSE(changed);

}
/*
 * main graph:
 * NetOutput                          NetOutput
 *     |                                   |
 *    if      =>                           if
 *    |                                  /   \
 *  data             InnerData(from init)x2  data
 *
 * then/se graph:
 *        InnerNetOutput            InnerNetOutput
 *             |                         |
 *      FooThen/Else        =>      FooThen/Else
 *         /   \                       /  \
 *      Foo1    InnerData      InnerData  InnerData
 *     /   \
 * const   const
 *
 */
TEST_F(ConstantExpressionsMotionUT, If_Success_CeOnThenAndElse) {
  auto data = ValueHolder::CreateFeed(0);
  auto if_outputs = bg::If<bg::ValueHolder>(
      data,
      [&]() -> std::vector<ValueHolderPtr> {
        auto c1 = ValueHolder::CreateConst("Hello", 5, true);
        auto c2 = ValueHolder::CreateConst("Hello", 5, true);
        auto foo1 = ValueHolder::CreateSingleDataOutput("FooThen", {c1, c2});
        auto foo2 = ValueHolder::CreateSingleDataOutput("FooThen", {foo1, data});
        return {foo2};
      },
      [&]() -> std::vector<ValueHolderPtr> {
        auto c1 = ValueHolder::CreateConst("Hello", 5, true);
        auto c2 = ValueHolder::CreateConst("Hello", 5, true);
        auto foo1 = ValueHolder::CreateSingleDataOutput("FooElse", {c1, c2});
        auto foo2 = ValueHolder::CreateSingleDataOutput("FooElse", {foo1, data});
        return {foo2};
      });
  auto main_frame = ValueHolder::PopGraphFrame(if_outputs, {}, "NetOutput");
  auto root_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(root_frame, nullptr);
  ASSERT_NE(main_frame, nullptr);

  bool changed = false;
  LoweringGlobalData global_data;
  ASSERT_EQ(ConstantExpressionMotion(&global_data).Run(root_frame->GetExecuteGraph().get(), changed),
            ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);

  // check init/main/deinit graph summary
  EXPECT_EQ(ExeGraphSummaryChecker(main_frame->GetExecuteGraph().get())
                .StrictDirectNodeTypes(
                    std::map<std::string, size_t>{{"Data", 1}, {"If", 1}, {"InnerData", 2}, {"NetOutput", 1}}),
            "success");
  EXPECT_EQ(ExeGraphSummaryChecker(init_frame_->GetExecuteGraph().get())
                .StrictDirectNodeTypes(
                    std::map<std::string, size_t>{{"Const", 4}, {"FooThen", 1}, {"FooElse", 1}, {"InnerNetOutput", 1}}),
            "success");
  EXPECT_EQ(de_init_frame_->GetExecuteGraph()->GetDirectNodesSize(), 0);

  auto if_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(main_frame->GetExecuteGraph().get(), "If");
  ASSERT_NE(if_node, nullptr);
  auto then_graph = ge::FastNodeUtils::GetSubgraphFromNode(if_node, 1U);
  ASSERT_NE(then_graph, nullptr);
  auto else_graph = ge::FastNodeUtils::GetSubgraphFromNode(if_node, 2U);
  ASSERT_NE(else_graph, nullptr);

  // check then/else graph summary
  EXPECT_EQ(ExeGraphSummaryChecker(then_graph)
                .StrictDirectNodeTypes(
                    std::map<std::string, size_t>{{"FooThen", 1}, {"InnerData", 2}, {"InnerNetOutput", 1}}),
            "success");
  EXPECT_EQ(ExeGraphSummaryChecker(else_graph)
                .StrictDirectNodeTypes(
                    std::map<std::string, size_t>{{"FooElse", 1}, {"InnerData", 2}, {"InnerNetOutput", 1}}),
            "success");

  // check connection from init to then/else graph nodes
  auto foo2_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(then_graph, "FooThen");
  auto foo1_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_frame_->GetExecuteGraph().get(), "FooThen");
  ASSERT_NE(foo2_node, nullptr);
  ASSERT_NE(foo1_node, nullptr);
  ConnectFromInitToMain(foo1_node, 0, foo2_node, 0);

  foo2_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(else_graph, "FooElse");
  foo1_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_frame_->GetExecuteGraph().get(), "FooElse");
  ASSERT_NE(foo2_node, nullptr);
  ASSERT_NE(foo1_node, nullptr);
  ConnectFromInitToMain(foo1_node, 0, foo2_node, 0);

  changed = false;
  ASSERT_EQ(ConstantExpressionMotion(&global_data).Run(root_frame->GetExecuteGraph().get(), changed),
            ge::GRAPH_SUCCESS);
  ASSERT_FALSE(changed);
}
/*
 * main graph:
 * NetOutput                          NetOutput
 *     |                                   |
 *    if      =>                          if
 *    |                                  /   \
 *  data             InnerData(from init)    data
 *
 * then graph:
 *               InnerNetOutput              InnerNetOutput
 *               c    |                            |
 *  Foo1Guarder <-- Foo2                          Foo2
 *            \   /   \              =>           /  \
 *             Foo1    InnerData          InnerData  InnerData
 *            /   \
 *        const   const
 */
TEST_F(ConstantExpressionsMotionUT, If_Success_CeGuarderOnThen) {
  auto data = ValueHolder::CreateFeed(0);
  auto if_outputs = bg::If<bg::ValueHolder>(
      data,
      [&]() -> std::vector<ValueHolderPtr> {
        auto c1 = ValueHolder::CreateConst("Hello", 5, true);
        auto c2 = ValueHolder::CreateConst("World", 5, true);
        auto foo1 = ValueHolder::CreateSingleDataOutput("Foo", {c1, c2});
        auto foo1_guarder = ValueHolder::CreateVoidGuarder("FreeFoo", foo1, {});
        auto foo2 = ValueHolder::CreateSingleDataOutput("Foo", {foo1, data});
        return {foo2};
      },
      [&]() -> std::vector<ValueHolderPtr> { return {ValueHolder::CreateSingleDataOutput("Bar", {data})}; });
  auto main_frame = ValueHolder::PopGraphFrame(if_outputs, {}, "NetOutput");
  auto root_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(root_frame, nullptr);
  ASSERT_NE(main_frame, nullptr);

  bool changed = false;
  LoweringGlobalData global_data;
  ASSERT_EQ(ConstantExpressionMotion(&global_data).Run(root_frame->GetExecuteGraph().get(), changed),
            ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);

  EXPECT_EQ(ExeGraphSummaryChecker(main_frame->GetExecuteGraph().get())
                .StrictDirectNodeTypes(
                    std::map<std::string, size_t>{{"Data", 1}, {"If", 1}, {"InnerData", 1}, {"NetOutput", 1}}),
            "success");
  EXPECT_EQ(ExeGraphSummaryChecker(init_frame_->GetExecuteGraph().get())
                .StrictDirectNodeTypes(std::map<std::string, size_t>{{"Const", 2}, {"Foo", 1}, {"InnerNetOutput", 1}}),
            "success");
  EXPECT_EQ(ExeGraphSummaryChecker(de_init_frame_->GetExecuteGraph().get())
                .StrictDirectNodeTypes(std::map<std::string, size_t>{{"InnerData", 1}, {"FreeFoo", 1}}),
            "success");

  auto if_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(main_frame->GetExecuteGraph().get(), "If");
  ASSERT_NE(if_node, nullptr);
  auto then_graph = ge::FastNodeUtils::GetSubgraphFromNode(if_node, 1U);
  ASSERT_NE(then_graph, nullptr);
  auto else_graph = ge::FastNodeUtils::GetSubgraphFromNode(if_node, 2U);
  ASSERT_NE(else_graph, nullptr);

  EXPECT_EQ(
      ExeGraphSummaryChecker(then_graph)
          .StrictDirectNodeTypes(std::map<std::string, size_t>{{"Foo", 1}, {"InnerData", 2}, {"InnerNetOutput", 1}}),
      "success");
  EXPECT_EQ(
      ExeGraphSummaryChecker(else_graph)
          .StrictDirectNodeTypes(std::map<std::string, size_t>{{"InnerData", 1}, {"Bar", 1}, {"InnerNetOutput", 1}}),
      "success");

  auto foo2_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(then_graph, "Foo");
  auto foo1_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_frame_->GetExecuteGraph().get(), "Foo");
  ASSERT_NE(foo2_node, nullptr);
  ASSERT_NE(foo1_node, nullptr);
  ConnectFromInitToMain(foo1_node, 0, foo2_node, 0);

  auto free_foo = ge::ExecuteGraphUtils::FindFirstNodeMatchType(de_init_frame_->GetExecuteGraph().get(), "FreeFoo");
  ASSERT_NE(free_foo, nullptr);
  ConnectFromInitToDeInit(foo1_node, 0, free_foo, 0);

  changed = false;
  ASSERT_EQ(ConstantExpressionMotion(&global_data).Run(root_frame->GetExecuteGraph().get(), changed),
            ge::GRAPH_SUCCESS);
  ASSERT_FALSE(changed);
}
/*
 * main graph:
 * NetOutput                          NetOutput
 *     |                                   |
 *    if      =>                          if
 *    |                                  /   \
 *  data             InnerData(from init)    data
 *
 * then graph:
 *
 *               InnerNetOutput
 *               c    |
 *  Foo1Guarder <-- Foo2
 *    |       \   /    \
 *    |        Foo1     \
 *    |       /   \      \
 *    |   const   const   \
 *  InnerData -------------+
 *
 *  todo: not support yet
 */
TEST_F(ConstantExpressionsMotionUT, If_Success_CeGuarderOnThenAndInputFromMain) {
  auto data = ValueHolder::CreateFeed(0);
  auto if_outputs = bg::If<bg::ValueHolder>(
      data,
      [&]() -> std::vector<ValueHolderPtr> {
        auto c1 = ValueHolder::CreateConst("Hello", 5, true);
        auto c2 = ValueHolder::CreateConst("World", 5, true);
        auto foo1 = ValueHolder::CreateSingleDataOutput("Foo", {c1, c2});
        auto foo1_guarder = ValueHolder::CreateVoidGuarder("FreeFoo", foo1, {data});
        auto foo2 = ValueHolder::CreateSingleDataOutput("Foo", {foo1, data});
        return {foo2};
      },
      [&]() -> std::vector<ValueHolderPtr> { return {data}; });
  auto main_frame = ValueHolder::PopGraphFrame(if_outputs, {}, "NetOutput");
  auto root_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(root_frame, nullptr);
  ASSERT_NE(main_frame, nullptr);

  // todo: 这个用例校验的时topo排序失败，是一个老用例，之前不支持，现在支持了但是还存在bug
  //       bug是存在跨图连边，从deinit图里面连接到了main图，导致topo排序报错。
  //       为了性能优化删除topo排序后，此处就校验不住了，暂时删除该用例，后续定位问题。
  // bool changed = false;
  // ASSERT_NE(ConstantExpressionMotion().Run(root_frame->GetExecuteGraph().get(), changed), ge::GRAPH_SUCCESS);
}
/*
 * main graph:
 * NetOutput                          NetOutput
 *     |                                   |
 *    if      =>                          if
 *    |                                  /   \
 *  data             InnerData(from init)    data
 *
 * then graph:
 *                       InnerNetOutput
 *                        /       \
 *                     Foo2    InnerData                        InnerNetOutput
 *                    /   \              =>                      /         \
 *                Foo1    const3                              Foo2      InnerData
 *                /   \                                     /    \
 *          const1   const2              InnerData(From Init)   InnerData(From Init const)
 *
 * else graph:
 *    InnerNetOutput
 *     /       |
 *  Foo        |
 *     \       |
 *      InnerData
 */
TEST_F(ConstantExpressionsMotionUT, If_RemainNodesConnectToNetOutput_AllNodesCe) {
  std::string foo2_name, foo1_name, const3_name;
  auto data = ValueHolder::CreateFeed(0);
  auto if_outputs = bg::If<bg::ValueHolder>(
      data,
      [&]() -> std::vector<ValueHolderPtr> {
        auto c1 = ValueHolder::CreateConst("Hello", 5);
        auto c2 = ValueHolder::CreateConst("Hello", 5);
        auto c3 = ValueHolder::CreateConst("Hello", 5);
        auto foo1 = ValueHolder::CreateSingleDataOutput("Foo", {c1, c2});
        auto foo2 = ValueHolder::CreateSingleDataOutput("Foo", {foo1, c3});
        foo2_name = foo2->GetFastNode()->GetName();
        foo1_name = foo1->GetFastNode()->GetName();
        const3_name = c3->GetFastNode()->GetName();
        return {foo2, data};
      },
      [&]() -> std::vector<ValueHolderPtr> {
        return {ValueHolder::CreateSingleDataOutput("Foo", {data}), data};
      });
  auto main_frame = ValueHolder::PopGraphFrame(if_outputs, {}, "NetOutput");
  auto root_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(root_frame, nullptr);
  ASSERT_NE(main_frame, nullptr);

  bool changed = false;
  LoweringGlobalData global_data;
  ASSERT_EQ(ConstantExpressionMotion(&global_data).Run(root_frame->GetExecuteGraph().get(), changed),
            ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);

  EXPECT_EQ(ExeGraphSummaryChecker(main_frame->GetExecuteGraph().get())
                .StrictDirectNodeTypes(
                    std::map<std::string, size_t>{{"Data", 1}, {"If", 1}, {"InnerData", 2}, {"NetOutput", 1}}),
            "success");
  EXPECT_EQ(ExeGraphSummaryChecker(init_frame_->GetExecuteGraph().get())
                .StrictDirectNodeTypes(std::map<std::string, size_t>{{"Const", 3}, {"Foo", 1}, {"InnerNetOutput", 1}}),
            "success");
  EXPECT_EQ(de_init_frame_->GetExecuteGraph()->GetDirectNodesSize(), 0);

  auto if_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(main_frame->GetExecuteGraph().get(), "If");
  ASSERT_NE(if_node, nullptr);
  auto then_graph = ge::FastNodeUtils::GetSubgraphFromNode(if_node, 1U);
  ASSERT_NE(then_graph, nullptr);
  auto else_graph = ge::FastNodeUtils::GetSubgraphFromNode(if_node, 2U);
  ASSERT_NE(else_graph, nullptr);

  EXPECT_EQ(
      ExeGraphSummaryChecker(then_graph)
          .StrictDirectNodeTypes(std::map<std::string, size_t>{{"InnerData", 3}, {"Foo", 1}, {"InnerNetOutput", 1}}),
      "success");
  EXPECT_EQ(
      ExeGraphSummaryChecker(else_graph)
          .StrictDirectNodeTypes(std::map<std::string, size_t>{{"InnerData", 1}, {"Foo", 1}, {"InnerNetOutput", 1}}),
      "success");

  auto output_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(then_graph, "InnerNetOutput");
  ASSERT_NE(output_node, nullptr);
  EXPECT_EQ(FastNodeTopoChecker(output_node).StrictConnectFrom({{"Foo", 0}, {"InnerData", 0}}), "success");

  auto foo2_node = ge::ExecuteGraphUtils::FindNodeFromAllNodes(then_graph, foo2_name.c_str());
  ASSERT_NE(foo2_node, nullptr);
  auto foo1_node = ge::ExecuteGraphUtils::FindNodeFromAllNodes(init_frame_->GetExecuteGraph().get(), foo1_name.c_str());
  ASSERT_NE(foo1_node, nullptr);
  auto c3_node = ge::ExecuteGraphUtils::FindNodeFromAllNodes(init_frame_->GetExecuteGraph().get(), const3_name.c_str());
  ASSERT_NE(c3_node, nullptr);
  ConnectFromInitToMain(foo1_node, 0, foo2_node, 0);
  ConnectFromInitToMain(c3_node, 0, foo2_node, 1);

  changed = false;
  ASSERT_EQ(ConstantExpressionMotion(&global_data).Run(root_frame->GetExecuteGraph().get(), changed),
            ge::GRAPH_SUCCESS);
  ASSERT_FALSE(changed);
}
/*
 *
 * +------------------+
 * |                  |  Init Graph
 * |  InnetNetOutput  |
 * |   1|  \0         |
 * |    |  Foo        |
 * |    |  / \        |
 * |    c1     c2     |
 * +------------------+
 */
TEST_F(ConstantExpressionsMotionUT, CemSuccess_ConstFromInit) {
  auto outputs = bg::FrameSelector::OnInitRoot([&]() -> std::vector<bg::ValueHolderPtr> {
    auto c1 = bg::ValueHolder::CreateConst("Hello", 5);
    auto c2 = bg::ValueHolder::CreateConst("Hello", 5);
    auto foo1 = bg::ValueHolder::CreateSingleDataOutput("Foo", {c1, c2});
    return {foo1, c1};
  });
  ASSERT_EQ(outputs.size(), 2);

  auto data0 = ValueHolder::CreateFeed(0);

  auto bar1 = ValueHolder::CreateSingleDataOutput("Bar1", {outputs[1]});
  auto bar2 = ValueHolder::CreateSingleDataOutput("Bar2", {outputs[0]});
  auto foo1 = ValueHolder::CreateSingleDataOutput("Foo1", {data0, outputs[0], bar1});
  auto foo2 = ValueHolder::CreateSingleDataOutput("Foo2", {outputs[1], bar2});
  auto main_frame = ValueHolder::PopGraphFrame({foo1, foo2}, {}, "NetOutput");
  auto root_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(main_frame, nullptr);
  ASSERT_NE(root_frame, nullptr);

  bool changed = false;
  LoweringGlobalData global_data;
  ASSERT_EQ(ConstantExpressionMotion(&global_data).Run(root_frame->GetExecuteGraph().get(), changed),
            ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);

  EXPECT_EQ(ExeGraphSummaryChecker(main_frame->GetExecuteGraph().get())
                .StrictDirectNodeTypes({{"Foo1", 1}, {"Foo2", 1}, {"NetOutput", 1}, {"Data", 1}, {"InnerData", 4}}),
            "success");

  EXPECT_EQ(ExeGraphSummaryChecker(init_frame_->GetExecuteGraph().get())
                .StrictDirectNodeTypes(
                    {{"Bar1", 1}, {"Bar2", 1}, {"InnerNetOutput", 1}, {"Const", 2}, {"Foo", 1}, {"NoOp", 1}}),
            "success");

  // init grpah connection ok
  auto bar1_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_frame_->GetExecuteGraph().get(), "Bar1");
  auto bar2_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_frame_->GetExecuteGraph().get(), "Bar2");
  ASSERT_NE(bar1_node, nullptr);
  ASSERT_NE(bar2_node, nullptr);
  EXPECT_EQ(FastNodeTopoChecker(bar1_node).StrictConnectFrom({{"Const"}, {"NoOp"}}), "success");
  EXPECT_EQ(FastNodeTopoChecker(bar2_node).StrictConnectFrom({{"Foo"}, {"NoOp"}}), "success");

  auto foo1_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(main_frame->GetExecuteGraph().get(), "Foo1");
  auto foo2_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(main_frame->GetExecuteGraph().get(), "Foo2");
  ASSERT_NE(foo1_node, nullptr);
  ASSERT_NE(foo2_node, nullptr);

  ConnectFromInitToMain(bar1_node, 0, foo1_node, 2);
  ConnectFromInitToMain(bar2_node, 0, foo2_node, 1);
}
/*
 * init graph:
 * +------------------+
 * |                  |
 * |  InnetNetOutput  |
 * |   1|  \0         |
 * |    |  Foo  Func  |
 * |    |  / \  |     |
 * |    c1     c2     |
 * +------------------+
 * main graph:
 *
 *                          netoutput
 *                             |
 *    + --------------------  Foo1
 *    |                       /  \
 *   bar2         innerdata(Foo)  innerdata(c1)
 *    |
 *   bar1
 *    |
 *  const
 */
TEST_F(ConstantExpressionsMotionUT, CemSuccess_CheckNoOpConnect) {
  auto outputs = bg::FrameSelector::OnInitRoot([&]() -> std::vector<bg::ValueHolderPtr> {
    auto c1 = bg::ValueHolder::CreateConst("Hello", 5);
    auto c2 = bg::ValueHolder::CreateConst("Hello", 5);
    auto foo1 = bg::ValueHolder::CreateSingleDataOutput("Foo", {c1, c2});
    auto func = bg::ValueHolder::CreateVoid<bg::ValueHolder>("Func", {c2});
    return {foo1, c1};
  });
  ASSERT_EQ(outputs.size(), 2);

  auto c3 = bg::ValueHolder::CreateConst("Hello", 5);
  auto bar1 = ValueHolder::CreateSingleDataOutput("Bar1", {c3});
  auto bar2 = ValueHolder::CreateSingleDataOutput("Bar2", {bar1});
  auto foo1 = ValueHolder::CreateSingleDataOutput("Foo1", {bar2, outputs[0], outputs[1]});
  auto main_frame = ValueHolder::PopGraphFrame({foo1}, {}, "NetOutput");
  auto root_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(main_frame, nullptr);
  ASSERT_NE(root_frame, nullptr);

  bool changed = false;
  LoweringGlobalData global_data;
  ASSERT_EQ(ConstantExpressionMotion(&global_data).Run(root_frame->GetExecuteGraph().get(), changed),
            ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);

  EXPECT_EQ(ExeGraphSummaryChecker(main_frame->GetExecuteGraph().get())
                .StrictDirectNodeTypes({{"Foo1", 1}, {"NetOutput", 1}, {"InnerData", 3}}),
            "success");

  EXPECT_EQ(
      ExeGraphSummaryChecker(init_frame_->GetExecuteGraph().get())
          .StrictDirectNodeTypes(
              {{"Bar1", 1}, {"Bar2", 1}, {"InnerNetOutput", 1}, {"Const", 3}, {"Foo", 1}, {"NoOp", 1}, {"Func", 1}}),
      "success");

  // init grpah connection ok
  auto bar1_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_frame_->GetExecuteGraph().get(), "Bar1");
  auto bar2_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_frame_->GetExecuteGraph().get(), "Bar2");
  auto no_op_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_frame_->GetExecuteGraph().get(), "NoOp");
  ASSERT_NE(bar1_node, nullptr);
  ASSERT_NE(bar2_node, nullptr);
  ASSERT_NE(no_op_node, nullptr);
  EXPECT_EQ(FastNodeTopoChecker(bar1_node).StrictConnectFrom({{"Const"}, {"NoOp"}}), "success");
  EXPECT_EQ(FastNodeTopoChecker(bar2_node).StrictConnectFrom({{"Bar1"}}), "success");
  EXPECT_EQ(FastNodeTopoChecker(no_op_node).StrictConnectFrom({{"Foo"}, {"Func"}}), "success");

  auto foo1_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(main_frame->GetExecuteGraph().get(), "Foo1");
  ASSERT_NE(foo1_node, nullptr);
  ConnectFromInitToMain(bar2_node, 0, foo1_node, 0);
}
/*
 * init graph:
 * +------------------+
 * |                  |
 * |  InnetNetOutput  |
 * |   1|  \0         |
 * |    |  Foo        |
 * |    |  / \        |
 * |    c1     c2     |
 * +------------------+
 * main graph:
 *
 *                      netoutput
 *                         |
 *    + ----------------  Foo1
 *    |                   /  \
 *   c3   innerdata(Foo)  innerdata(c1)
 */
TEST_F(ConstantExpressionsMotionUT, CemSuccess_CheckNotCreateNoOp) {
  auto outputs = bg::FrameSelector::OnInitRoot([&]() -> std::vector<bg::ValueHolderPtr> {
    auto c1 = bg::ValueHolder::CreateConst("Hello", 5);
    auto c2 = bg::ValueHolder::CreateConst("Hello", 5);
    auto foo1 = bg::ValueHolder::CreateSingleDataOutput("Foo", {c1, c2});
    return {foo1, c1};
  });
  ASSERT_EQ(outputs.size(), 2);

  auto c3 = bg::ValueHolder::CreateConst("Hello", 5);
  auto foo1 = ValueHolder::CreateSingleDataOutput("Foo1", {c3, outputs[0], outputs[1]});
  auto main_frame = ValueHolder::PopGraphFrame({foo1}, {}, "NetOutput");
  auto root_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(main_frame, nullptr);
  ASSERT_NE(root_frame, nullptr);

  bool changed = false;
  LoweringGlobalData global_data;
  ASSERT_EQ(ConstantExpressionMotion(&global_data).Run(root_frame->GetExecuteGraph().get(), changed),
            ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);

  EXPECT_EQ(ExeGraphSummaryChecker(main_frame->GetExecuteGraph().get())
                .StrictDirectNodeTypes({{"Foo1", 1}, {"NetOutput", 1}, {"InnerData", 3}}),
            "success");

  EXPECT_EQ(ExeGraphSummaryChecker(init_frame_->GetExecuteGraph().get())
                .StrictDirectNodeTypes({{"InnerNetOutput", 1}, {"Const", 3}, {"Foo", 1}}),
            "success");

  ConnectFromInitToMain(c3->GetFastNode(), 0, foo1->GetFastNode(), 0);
}
/*
 * init graph:
 * +------------------+
 * |  InnetNetOutput  |
 * |        |         |
 * |       Foo        |
 * |       / \        |
 * |     c1    c2     |
 * +------------------+
 * main graph first time:
 *
 *                        NetOutput
 *                            |
 *      + -----------------  Foo2
 *      |                  /   \
 *      |                Foo1    data
 *      |               /   \
 *  Innerdata       const   const
 *
 * after CEM first time:
 *                        NetOutput
 *                            |
 *       +---------------   Foo2
 *       |                   / \
 *   InnerData       InnerData  data
 *
 * new CE:
 *                        NetOutput
 *                           |
 *        + --------------  Foo2
 *        |                 / \
 *   InnerData     InnerData  Foo3
 *                           /   \
 *                      const    const
 *
 * after CEM second Time:
 *                         NetOutput
 *                             |
 *        + --------------   Foo2
 *        |                   / \
 *   InnerData        InnerData  InnerData
 */
TEST_F(ConstantExpressionsMotionUT, CemSuccess_CheckNoOpCreateTwice) {
  auto outputs = bg::FrameSelector::OnInitRoot([&]() -> std::vector<bg::ValueHolderPtr> {
    auto c1 = bg::ValueHolder::CreateConst("Hello", 5);
    auto c2 = bg::ValueHolder::CreateConst("Hello", 5);
    auto foo1 = bg::ValueHolder::CreateSingleDataOutput("Foo", {c1, c2});
    return {foo1};
  });
  ASSERT_EQ(outputs.size(), 1);

  auto c1 = ValueHolder::CreateConst("Hello", 5, true);
  auto c2 = ValueHolder::CreateConst("World", 5, true);
  auto data = ValueHolder::CreateFeed(0);
  auto foo1 = ValueHolder::CreateSingleDataOutput("Foo", {c1, c2});
  auto foo2 = ValueHolder::CreateSingleDataOutput("Foo", {outputs[0], foo1, data});
  auto main_frame = ValueHolder::PopGraphFrame({foo2}, {}, "NetOutput");
  auto root_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(root_frame, nullptr);
  ASSERT_NE(main_frame, nullptr);

  // CEM the first time
  bool changed = false;
  LoweringGlobalData global_data;
  auto pass = ConstantExpressionMotion(&global_data);
  ASSERT_EQ(pass.Run(root_frame->GetExecuteGraph().get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);

  // check summary
  EXPECT_EQ(ExeGraphSummaryChecker(init_frame_->GetExecuteGraph().get())
                .StrictDirectNodeTypes(
                    std::map<std::string, size_t>{{"Const", 4}, {"Foo", 2}, {"InnerNetOutput", 1}, {"NoOp", 1}}),
            "success");

  EXPECT_EQ(ExeGraphSummaryChecker(main_frame->GetExecuteGraph().get())
                .StrictDirectNodeTypes(
                    std::map<std::string, size_t>{{"Data", 1}, {"Foo", 1}, {"InnerData", 2}, {"NetOutput", 1}}),
            "success");
  EXPECT_EQ(de_init_frame_->GetExecuteGraph()->GetDirectNodesSize(), 0);

  // New CE
  auto main_graph = main_frame->GetExecuteGraph();
  auto c3 = AddConstNode(main_graph.get(), "HELLO", 5);
  auto c4 = AddConstNode(main_graph.get(), "HELLO", 5);
  auto foo3 = AddNode(main_graph.get(), "Foo", 2, 1);
  EXPECT_NE(main_graph->AddEdge(c3, 0, foo3, 0), nullptr);
  EXPECT_NE(main_graph->AddEdge(c4, 0, foo3, 1), nullptr);
  UnlinkInEdgeByIndex(foo2->GetFastNode(), 2);
  EXPECT_NE(main_graph->AddEdge(foo3, 0, foo2->GetFastNode(), 2), nullptr);

  // CEM the second time
  changed = false;
  ASSERT_EQ(pass.Run(root_frame->GetExecuteGraph().get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);

  // check summary
  EXPECT_EQ(ExeGraphSummaryChecker(init_frame_->GetExecuteGraph().get())
                .StrictDirectNodeTypes(
                    std::map<std::string, size_t>{{"Const", 6}, {"Foo", 3}, {"InnerNetOutput", 1}, {"NoOp", 2}}),
            "success");

  EXPECT_EQ(ExeGraphSummaryChecker(main_frame->GetExecuteGraph().get())
                .StrictDirectNodeTypes(
                    std::map<std::string, size_t>{{"Data", 1}, {"Foo", 1}, {"InnerData", 3}, {"NetOutput", 1}}),
            "success");
  EXPECT_EQ(de_init_frame_->GetExecuteGraph()->GetDirectNodesSize(), 0);

  ConnectFromInitToMain(foo1->GetFastNode(), 0, foo2->GetFastNode(), 1);
  ConnectFromInitToMain(foo3, 0, foo2->GetFastNode(), 2);

  changed = false;
  ASSERT_EQ(pass.Run(root_frame->GetExecuteGraph().get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_FALSE(changed);
}
/*
 * init graph:
 * +------------------+
 * |  InnetNetOutput  |
 * |        |         |
 * |       Foo        |
 * |       / \        |
 * |     c1    c2     |
 * +------------------+
 * main graph first time:
 *
 *                        NetOutput
 *                            |
 *      + -----------------  Foo2
 *      |                  /   \
 *      |                c3    data
 *      |
 *  Innerdata(Foo)
 *
 *  after CEM first time:
 *                        NetOutput
 *                            |
 *       +---------------   Foo2
 *       |                   / \
 *   InnerData       InnerData  data
 *
 *  * new CE:
 *                        NetOutput
 *                           |
 *        + --------------  Foo2
 *        |                 / \
 *   InnerData     InnerData  Foo3
 *                           /   \
 *                          c4    c5
 *
 *  * after CEM second Time:
 *                         NetOutput
 *                             |
 *        + --------------   Foo2
 *        |                   / \
 *   InnerData        InnerData  InnerData
 */
// 第一次cem不需要构造NoOp,第二次需要构造NoOp
TEST_F(ConstantExpressionsMotionUT, CemSuccess_CheckNoOpCreateOnlySecondTime) {
  auto outputs = bg::FrameSelector::OnInitRoot([&]() -> std::vector<bg::ValueHolderPtr> {
    auto c1 = bg::ValueHolder::CreateConst("Hello", 5);
    auto c2 = bg::ValueHolder::CreateConst("Hello", 5);
    auto foo1 = bg::ValueHolder::CreateSingleDataOutput("Foo", {c1, c2});
    return {foo1};
  });
  ASSERT_EQ(outputs.size(), 1);

  auto foo1 = ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_frame_->GetExecuteGraph().get(), "Foo");
  ASSERT_NE(foo1, nullptr);

  auto c3 = ValueHolder::CreateConst("Hello", 5, true);
  auto data = ValueHolder::CreateFeed(0);
  auto foo2 = ValueHolder::CreateSingleDataOutput("Foo", {outputs[0], c3, data});
  auto main_frame = ValueHolder::PopGraphFrame({foo2}, {}, "NetOutput");
  auto root_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(root_frame, nullptr);
  ASSERT_NE(main_frame, nullptr);

  // CEM the first time
  bool changed = false;
  LoweringGlobalData global_data;
  auto pass = ConstantExpressionMotion(&global_data);
  ASSERT_EQ(pass.Run(root_frame->GetExecuteGraph().get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);

  // check summary
  EXPECT_EQ(ExeGraphSummaryChecker(init_frame_->GetExecuteGraph().get())
                .StrictDirectNodeTypes(std::map<std::string, size_t>{{"Const", 3}, {"Foo", 1}, {"InnerNetOutput", 1}}),
            "success");

  EXPECT_EQ(ExeGraphSummaryChecker(main_frame->GetExecuteGraph().get())
                .StrictDirectNodeTypes(
                    std::map<std::string, size_t>{{"Data", 1}, {"Foo", 1}, {"InnerData", 2}, {"NetOutput", 1}}),
            "success");
  EXPECT_EQ(de_init_frame_->GetExecuteGraph()->GetDirectNodesSize(), 0);

  // New CE
  auto main_graph = main_frame->GetExecuteGraph();
  auto c4 = AddConstNode(main_graph.get(), "HELLO", 5);
  auto c5 = AddConstNode(main_graph.get(), "HELLO", 5);
  auto foo3 = AddNode(main_graph.get(), "Foo", 2, 1);
  EXPECT_NE(main_graph->AddEdge(c4, 0, foo3, 0), nullptr);
  EXPECT_NE(main_graph->AddEdge(c5, 0, foo3, 1), nullptr);
  UnlinkInEdgeByIndex(foo2->GetFastNode(), 2);
  EXPECT_NE(main_graph->AddEdge(foo3, 0, foo2->GetFastNode(), 2), nullptr);

  // CEM the second time
  changed = false;
  ASSERT_EQ(pass.Run(root_frame->GetExecuteGraph().get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);

  // check summary
  EXPECT_EQ(ExeGraphSummaryChecker(init_frame_->GetExecuteGraph().get())
                .StrictDirectNodeTypes(
                    std::map<std::string, size_t>{{"Const", 5}, {"Foo", 2}, {"InnerNetOutput", 1}, {"NoOp", 1}}),
            "success");

  EXPECT_EQ(ExeGraphSummaryChecker(main_frame->GetExecuteGraph().get())
                .StrictDirectNodeTypes(
                    std::map<std::string, size_t>{{"Data", 1}, {"Foo", 1}, {"InnerData", 3}, {"NetOutput", 1}}),
            "success");
  EXPECT_EQ(de_init_frame_->GetExecuteGraph()->GetDirectNodesSize(), 0);

  ConnectFromInitToMain(c3->GetFastNode(), 0, foo2->GetFastNode(), 1);
  ConnectFromInitToMain(foo1, 0, foo2->GetFastNode(), 0);
  ConnectFromInitToMain(foo3, 0, foo2->GetFastNode(), 2);

  changed = false;
  ASSERT_EQ(pass.Run(root_frame->GetExecuteGraph().get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_FALSE(changed);
}

TEST_F(ConstantExpressionsMotionUT, While_CemSuccess_WhenCondGraphHasCe) {
  while (ValueHolder::PopGraphFrame() != nullptr) {
  }
  auto compute_graph = ShareGraph::WhileGraph2();
  ASSERT_NE(compute_graph, nullptr);
  compute_graph->TopologicalSorting();

  GertRuntimeStub stub;
  stub.GetConverterStub().Register("Add_1", Converter3).Register("LessThan_5", Converter1);

  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  auto model_desc_holder = ModelDescHolderFaker().Build();
  auto exe_graph = GraphConverter()
                       .SetModelDescHolder(&model_desc_holder)
                       .ConvertComputeGraphToExecuteGraph(compute_graph, global_data);
  ASSERT_NE(exe_graph, nullptr);

  auto main_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Main");
  ASSERT_NE(main_node, nullptr);
  auto main_graph = ge::FastNodeUtils::GetSubgraphFromNode(main_node, 0);
  ASSERT_NE(main_graph, nullptr);
  auto init_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Init");
  ASSERT_NE(init_node, nullptr);
  auto init_graph = ge::FastNodeUtils::GetSubgraphFromNode(init_node, 0);
  ASSERT_NE(init_graph, nullptr);

  auto while_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(main_graph, "While");
  ASSERT_NE(while_node, nullptr);
  auto ctrl_graph = ge::FastNodeUtils::GetSubgraphFromNode(while_node, 0U);
  ASSERT_NE(ctrl_graph, nullptr);
  auto subgraph_call = ge::ExecuteGraphUtils::FindFirstNodeMatchType(ctrl_graph, "SubgraphCall");
  ASSERT_NE(subgraph_call, nullptr);
  auto cond_graph = ge::FastNodeUtils::GetSubgraphFromNode(subgraph_call, 1U);
  ASSERT_NE(cond_graph, nullptr);

  auto netoutput = ge::ExecuteGraphUtils::FindFirstNodeMatchType(cond_graph, "InnerNetOutput");
  EXPECT_EQ(FastNodeTopoChecker(netoutput).StrictConnectFrom({{"GenCondForWhile"}}), "success");
  auto launch_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(cond_graph, "Launch");
  ASSERT_NE(launch_node, nullptr);
  auto bar_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_graph, "BarCond");
  ASSERT_NE(bar_node, nullptr);

  ASSERT_EQ(ge::ExecuteGraphUtils::FindFirstNodeMatchType(cond_graph, "BarCond"), nullptr);
  ASSERT_EQ(ge::ExecuteGraphUtils::FindFirstNodeMatchType(cond_graph, "Foo"), nullptr);
  ASSERT_EQ(ge::ExecuteGraphUtils::FindFirstNodeMatchType(cond_graph, "Const"), nullptr);

  ConnectFromInitToMain(bar_node, 0, launch_node, 0);
}
TEST_F(ConstantExpressionsMotionUT, While_NoEffectToCondNetOutput_OnlyBodyCem) {
  while (ValueHolder::PopGraphFrame() != nullptr)
    ;
  auto compute_graph = ShareGraph::WhileGraph2();
  ASSERT_NE(compute_graph, nullptr);
  compute_graph->TopologicalSorting();

  GertRuntimeStub stub;
  stub.GetConverterStub().Register("Add_1", Converter2).Register("LessThan_5", Converter3);

  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  auto model_desc_holder = ModelDescHolderFaker().Build();
  auto exe_graph = GraphConverter()
                       .SetModelDescHolder(&model_desc_holder)
                       .ConvertComputeGraphToExecuteGraph(compute_graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  auto main_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Main");
  ASSERT_NE(main_node, nullptr);
  auto main_graph = ge::FastNodeUtils::GetSubgraphFromNode(main_node, 0);
  ASSERT_NE(main_graph, nullptr);
  auto init_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Init");
  ASSERT_NE(init_node, nullptr);
  auto init_graph = ge::FastNodeUtils::GetSubgraphFromNode(init_node, 0);
  ASSERT_NE(init_graph, nullptr);

  auto while_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(main_graph, "While");
  ASSERT_NE(while_node, nullptr);
  auto ctrl_graph = ge::FastNodeUtils::GetSubgraphFromNode(while_node, 0U);
  ASSERT_NE(ctrl_graph, nullptr);
  auto subgraph_call = ge::ExecuteGraphUtils::FindFirstNodeMatchType(ctrl_graph, "SubgraphCall");
  ASSERT_NE(subgraph_call, nullptr);
  auto cond_graph = ge::FastNodeUtils::GetSubgraphFromNode(subgraph_call, 1U);
  ASSERT_NE(cond_graph, nullptr);

  auto netoutput = ge::ExecuteGraphUtils::FindFirstNodeMatchType(cond_graph, "InnerNetOutput");
  EXPECT_EQ(FastNodeTopoChecker(netoutput).StrictConnectFrom({{"GenCondForWhile"}}), "success");
}
TEST_F(ConstantExpressionsMotionUT, While_AllCem_CondAndBodyContainsCE) {
  while (ValueHolder::PopGraphFrame() != nullptr)
    ;
  auto compute_graph = ShareGraph::WhileGraph2();
  ASSERT_NE(compute_graph, nullptr);
  compute_graph->TopologicalSorting();

  GertRuntimeStub stub;
  stub.GetConverterStub().Register("Add_1", Converter2).Register("LessThan_5", Converter1);

  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  auto model_desc_holder = ModelDescHolderFaker().Build();
  auto exe_graph = GraphConverter()
                       .SetModelDescHolder(&model_desc_holder)
                       .ConvertComputeGraphToExecuteGraph(compute_graph, global_data);
  ASSERT_NE(exe_graph, nullptr);

  auto main_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Main");
  ASSERT_NE(main_node, nullptr);
  auto main_graph = ge::FastNodeUtils::GetSubgraphFromNode(main_node, 0);
  ASSERT_NE(main_graph, nullptr);
  auto init_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Init");
  ASSERT_NE(init_node, nullptr);
  auto init_graph = ge::FastNodeUtils::GetSubgraphFromNode(init_node, 0);
  ASSERT_NE(init_graph, nullptr);

  EXPECT_EQ(ExeGraphSummaryChecker(main_graph)
                .StrictDirectNodeTypes({{"Data", 5},
                                        {"SplitDataTensor", 1},
                                        {"OutputData", 1},
                                        {"NetOutput", 1},
                                        {"EnsureTensorAtOutMemory", 1},
                                        {"InnerData", 11},
                                        {"IdentityAddr", 1},
                                        {"IdentityShape", 1},
                                        {"SelectL1Allocator", 1},
                                        {"SelectL2Allocator", 1},
                                        {"CalcTensorSizeFromStorage", 1},
                                        {"SplitRtStreams", 1},
                                        {"MakeSureTensorAtDevice", 1},
                                        {"While", 1},
                                        {"FreeMemory", 2}}),
            "success");

  auto while_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(main_graph, "While");
  ASSERT_NE(while_node, nullptr);
  EXPECT_EQ(FastNodeTopoChecker(while_node)
                .StrictConnectFrom(
                    {{"IdentityShape", 0}, {"IdentityAddr", 0}, {"InnerData", 0}, {"InnerData", 0}, {"InnerData", 0}}),
            "success");

  auto body_graph = ge::FastNodeUtils::GetSubgraphFromNode(while_node, 1U);
  ASSERT_NE(body_graph, nullptr);
  auto body_netoutput = ge::ExecuteGraphUtils::FindFirstNodeMatchType(body_graph, "InnerNetOutput");
  ASSERT_NE(body_netoutput, nullptr);
  // the body subgraph should add inputs/outputs sync when cem on the cond subgraph
  EXPECT_EQ(FastNodeTopoChecker(body_netoutput).StrictConnectFrom({{"IdentityShapeAndAddr"}, {"IdentityShapeAndAddr"}}),
            "success");
  auto body_launch_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(body_graph, "Launch");
  ASSERT_NE(body_launch_node, nullptr);
  EXPECT_EQ(FastNodeTopoChecker(body_launch_node).StrictConnectFrom({{"InnerData", 0}, {"AllocMemory"}, {"InnerData"}}),
            "success");
  auto bar_body_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_graph, "BarBody");
  ASSERT_NE(bar_body_node, nullptr);
  ConnectFromInitToMain(bar_body_node, 0, body_launch_node, 0);
}

/*
 * main graph:
 *           NetOutput
 *           /       \
 *      Foo3          |
 *       |        c   |
 *  Foo1Guarder <-- Foo2
 *            \   /   \
 *             Foo1    data
 *            /   \
 *        const   const
 */
TEST_F(ConstantExpressionsMotionUT, MoveToDeInit_SkipGuaderHasOutput_CeGuarderOnMain) {
  auto c1 = ValueHolder::CreateConst("Hello", 5, true);
  auto c2 = ValueHolder::CreateConst("World", 5, true);
  auto data = ValueHolder::CreateFeed(0);
  auto foo1 = ValueHolder::CreateSingleDataOutput("Foo", {c1, c2});
  auto foo1_guarder = DevMemValueHolder::CreateSingleDataOutput("FreeFoo", {foo1}, 0);
  foo1->SetGuarder(foo1_guarder);
  ASSERT_NE(foo1_guarder, nullptr);
  auto foo2 = ValueHolder::CreateSingleDataOutput("Foo", {foo1, data});
  auto foo3 = ValueHolder::CreateSingleDataOutput("Foo", {foo1_guarder});
  auto main_frame = ValueHolder::PopGraphFrame({foo2, foo3}, {}, "NetOutput");
  auto root_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(root_frame, nullptr);
  ASSERT_NE(main_frame, nullptr);

  bool changed = false;
  LoweringGlobalData global_data;
  ASSERT_EQ(ConstantExpressionMotion(&global_data).Run(root_frame->GetExecuteGraph().get(), changed),
            ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);
  ge::DumpGraph(root_frame->GetExecuteGraph().get(), "guarder1");
  ASSERT_EQ(root_frame->GetExecuteGraph()->TopologicalSorting(), ge::SUCCESS);

  EXPECT_EQ(ExeGraphSummaryChecker(main_frame->GetExecuteGraph().get())
                .StrictDirectNodeTypes(std::map<std::string, size_t>{
                    {"Data", 1}, {"Foo", 2}, {"FreeFoo", 1}, {"InnerData", 1}, {"NetOutput", 1}}),
            "success");

  EXPECT_EQ(ExeGraphSummaryChecker(init_frame_->GetExecuteGraph().get())
                .StrictDirectNodeTypes(std::map<std::string, size_t>{{"Const", 2}, {"Foo", 1}, {"InnerNetOutput", 1}}),
            "success");
  EXPECT_EQ(de_init_frame_->GetExecuteGraph()->GetDirectNodesSize(), 0U);

  ConnectFromInitToMain(foo1->GetFastNode(), 0, foo2->GetFastNode(), 0);
  ConnectFromInitToMain(foo1->GetFastNode(), 0, foo1_guarder->GetFastNode(), 0);

  changed = false;
  ASSERT_EQ(ConstantExpressionMotion(&global_data).Run(root_frame->GetExecuteGraph().get(), changed),
            ge::GRAPH_SUCCESS);
  ASSERT_FALSE(changed);
}

/*
 * main graph:
 *                NetOutput                      NetOutput
 *                    |                              |
 *  Foo1Guarder <-- Foo2       =====>              Foo2
 *            \   /    \                           /    \
 *             Foo1    data                 Innerdata   data
 *            /    \
 *          c1   CreateHostL2Allocator
 */
TEST_F(ConstantExpressionsMotionUT, MoveToInit_SUCCESS_HasEquilvantNodesInInit) {
  LoweringGlobalData global_data;
  InitTestFramesWithStream(global_data, 2);
  auto host_allocator = global_data.GetOrCreateAllocator({TensorPlacement::kOnHost, AllocatorUsage::kAllocNodeOutput});
  ASSERT_NE(host_allocator, nullptr);

  size_t size = 1U;
  auto c1 = ValueHolder::CreateConst(&size, sizeof(size_t));
  auto foo1 = ValueHolder::CreateSingleDataOutput("Foo1", {host_allocator, c1});
  auto foo1_guarder = ValueHolder::CreateVoidGuarder("FreeFoo1", foo1, {});
  auto data = ValueHolder::CreateFeed(0);
  auto foo2 = ValueHolder::CreateSingleDataOutput("Foo2", {data, foo1});
  auto main_frame = ValueHolder::PopGraphFrame({foo2}, {}, "NetOutput");
  auto root_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(root_frame, nullptr);
  ASSERT_NE(main_frame, nullptr);

  bool changed = false;
  auto pass = ConstantExpressionMotion(&global_data);
  ASSERT_EQ(pass.Run(root_frame->GetExecuteGraph().get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);

  EXPECT_EQ(ExeGraphSummaryChecker(main_frame->GetExecuteGraph().get())
                .StrictDirectNodeTypes(std::map<std::string, size_t>{{"Data", 2},
                                                                     {"InnerData", 3},
                                                                     {"Foo2", 1},
                                                                     {"SplitRtStreams", 1},
                                                                     {"CreateHostL2Allocator", 1},
                                                                     {"NetOutput", 1}}),
            "success");

  EXPECT_EQ(ExeGraphSummaryChecker(init_frame_->GetExecuteGraph().get())
                .StrictDirectNodeTypes(std::map<std::string, size_t>{{"Const", 5},
                                                                     {"Data", 1},
                                                                     {"SplitRtStreams", 1},
                                                                     {"Foo1", 1},
                                                                     {"CreateL1Allocator", 1},
                                                                     {"CreateHostL2Allocator", 1},
                                                                     {"NoOp", 1},
                                                                     {"InnerNetOutput", 1}}),
            "success");
  EXPECT_EQ(ExeGraphSummaryChecker(de_init_frame_->GetExecuteGraph().get())
                .StrictDirectNodeTypes(std::map<std::string, size_t>{{"InnerData", 1}, {"FreeFoo1", 1}}),
            "success");

  ConnectFromInitToMain(foo1->GetFastNode(), 0, foo2->GetFastNode(), 1);
  ConnectFromInitToDeInit(foo1->GetFastNode(), 0, foo1_guarder->GetFastNode(), 0);

  changed = false;
  ASSERT_EQ(pass.Run(root_frame->GetExecuteGraph().get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_FALSE(changed);
}

/*
 * main graph:
 *                NetOutput
 *                    |
 *   FreeMemHost <-- Foo
 *            \   /    \
 *      AllocMemHost    data
 *            /    \
 *          c1   CreateHostL2Allocator
 */
TEST_F(ConstantExpressionsMotionUT, MoveToInit_NotMove_EquilvantNodesConnectToAllocNode) {
  LoweringGlobalData global_data;
  InitTestFramesWithStream(global_data, 2);
  auto host_allocator = global_data.GetOrCreateAllocator({TensorPlacement::kOnHost, AllocatorUsage::kAllocNodeOutput});
  ASSERT_NE(host_allocator, nullptr);

  size_t size = 1U;
  auto c1 = ValueHolder::CreateConst(&size, sizeof(size_t));
  auto alloc_node = ValueHolder::CreateSingleDataOutput("AllocMemHost", {host_allocator, c1});
  auto guarder = ValueHolder::CreateVoidGuarder("FreeMemHost", alloc_node, {});
  auto data = ValueHolder::CreateFeed(0);
  auto foo = ValueHolder::CreateSingleDataOutput("Foo", {data, alloc_node});
  auto main_frame = ValueHolder::PopGraphFrame({foo}, {}, "NetOutput");
  auto root_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(root_frame, nullptr);
  ASSERT_NE(main_frame, nullptr);

  bool changed = false;
  auto pass = ConstantExpressionMotion(&global_data);
  ASSERT_EQ(pass.Run(root_frame->GetExecuteGraph().get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);

  EXPECT_EQ(ExeGraphSummaryChecker(main_frame->GetExecuteGraph().get())
                .StrictDirectNodeTypes(std::map<std::string, size_t>{{"Data", 2},
                                                                     {"InnerData", 3},
                                                                     {"AllocMemHost", 1},
                                                                     {"FreeMemHost", 1},
                                                                     {"Foo", 1},
                                                                     {"SplitRtStreams", 1},
                                                                     {"CreateHostL2Allocator", 1},
                                                                     {"NetOutput", 1}}),
            "success");

  EXPECT_EQ(ExeGraphSummaryChecker(init_frame_->GetExecuteGraph().get())
                .StrictDirectNodeTypes(std::map<std::string, size_t>{{"Const", 5},
                                                                     {"Data", 1},
                                                                     {"SplitRtStreams", 1},
                                                                     {"CreateL1Allocator", 1},
                                                                     {"CreateHostL2Allocator", 1},
                                                                     {"InnerNetOutput", 1}}),
            "success");
  EXPECT_EQ(de_init_frame_->GetExecuteGraph().get()->GetDirectNodesSize(), 0U);
  EXPECT_EQ(FastNodeTopoChecker(alloc_node).StrictConnectFrom({{"CreateHostL2Allocator", 0}, {"InnerData", 0}}),
            "success");

  changed = false;
  ASSERT_EQ(pass.Run(root_frame->GetExecuteGraph().get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_FALSE(changed);
}

/*
 * main graph:
 *                NetOutput
 *                    |
 *  Foo1Guarder <-- Foo2
 *            \   /    \
 *             Foo1    data
 *            /    \
 *          c1   SelectL2Allocator
 */
TEST_F(ConstantExpressionsMotionUT, MoveToInit_NotMove_NoEquilvantNodesInInit) {
  LoweringGlobalData global_data;
  InitTestFramesWithStream(global_data, 2);
  auto device_allocator =
      global_data.GetOrCreateAllocator({TensorPlacement::kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
  ASSERT_NE(device_allocator, nullptr);

  size_t size = 1U;
  auto c1 = ValueHolder::CreateConst(&size, sizeof(size_t));
  auto foo1 = ValueHolder::CreateSingleDataOutput("Foo1", {device_allocator, c1});
  auto foo_guarder = ValueHolder::CreateVoidGuarder("FreeFoo1", foo1, {});
  auto data = ValueHolder::CreateFeed(0);
  auto foo2 = ValueHolder::CreateSingleDataOutput("Foo2", {data, foo1});
  auto main_frame = ValueHolder::PopGraphFrame({foo2}, {}, "NetOutput");
  auto root_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(root_frame, nullptr);
  ASSERT_NE(main_frame, nullptr);

  bool changed = false;
  auto pass = ConstantExpressionMotion(&global_data);
  ASSERT_EQ(pass.Run(root_frame->GetExecuteGraph().get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);

  EXPECT_EQ(ExeGraphSummaryChecker(main_frame->GetExecuteGraph().get())
                .StrictDirectNodeTypes(std::map<std::string, size_t>{{"Data", 2},
                                                                     {"InnerData", 5},
                                                                     {"Foo1", 1},
                                                                     {"FreeFoo1", 1},
                                                                     {"Foo2", 1},
                                                                     {"SplitRtStreams", 1},
                                                                     {"SelectL2Allocator", 1},
                                                                     {"NetOutput", 1}}),
            "success");

  EXPECT_EQ(ExeGraphSummaryChecker(init_frame_->GetExecuteGraph().get())
                .StrictDirectNodeTypes(std::map<std::string, size_t>{{"Const", 7},
                                                                     {"Data", 1},
                                                                     {"SplitRtStreams", 1},
                                                                     {"CreateL1Allocator", 1},
                                                                     {"CreateL2Allocators", 1},
                                                                     {"InnerNetOutput", 1}}),
            "success");
  EXPECT_EQ(de_init_frame_->GetExecuteGraph().get()->GetDirectNodesSize(), 0U);
  EXPECT_EQ(FastNodeTopoChecker(foo1).StrictConnectFrom({{"SelectL2Allocator", 0}, {"InnerData", 0}}), "success");

  changed = false;
  ASSERT_EQ(pass.Run(root_frame->GetExecuteGraph().get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_FALSE(changed);
}

/*
 * main graph:
 * NetOutput                                   NetOutput
 *     |                                          |
 *    if      =>                                  if
 *   /  \                                        /  \
 * data SelectL2Allocator     InnerData(from init)  data    SelectL2Allocator
 *
 * then graph:
 *               InnerNetOutput              InnerNetOutput
 *               c    |                            |
 *  Foo1Guarder <-- Foo2                          Foo2
 *            \   /   \              =>           /  \
 *             Foo1   const               InnerData  const
 *            /   \
 *     InnerData   const
 */
TEST_F(ConstantExpressionsMotionUT, MoveToInit_SUCCESS_EquilvantNodesConnectToSubgraph) {
  LoweringGlobalData global_data;
  InitTestFramesWithStream(global_data, 2);
  auto device_allocator_on_main =
      global_data.GetOrCreateAllocator({TensorPlacement::kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
  ASSERT_NE(device_allocator_on_main, nullptr);

  auto main_device_allocator_0 =
      global_data.GetOrCreateAllocator({TensorPlacement::kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
  ASSERT_NE(main_device_allocator_0, nullptr);

  auto init_device_allocator_0 = bg::FrameSelector::OnInitRoot([&global_data]() -> std::vector<bg::ValueHolderPtr> {
    auto allocator =
        global_data.GetOrCreateAllocator({TensorPlacement::kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
    return {allocator};
  });

  auto data = ValueHolder::CreateFeed(0);
  auto if_outputs = bg::If<bg::ValueHolder>(
      data,
      [&]() -> std::vector<ValueHolderPtr> {
        auto c1 = ValueHolder::CreateConst("Hello", 5, true);
        auto c2 = ValueHolder::CreateConst("World", 5, true);
        auto foo1 = ValueHolder::CreateSingleDataOutput("Foo1", {main_device_allocator_0, c1});
        auto foo1_guarder = ValueHolder::CreateVoidGuarder("FreeFoo1", foo1, {});
        auto foo2 = ValueHolder::CreateSingleDataOutput("Foo2", {foo1, c2});
        return {foo2};
      },
      [&]() -> std::vector<ValueHolderPtr> { return {ValueHolder::CreateSingleDataOutput("Bar", {data})}; });
  auto main_frame = ValueHolder::PopGraphFrame(if_outputs, {}, "NetOutput");
  auto root_frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(root_frame, nullptr);
  ASSERT_NE(main_frame, nullptr);

  bool changed = false;
  auto pass = ConstantExpressionMotion(&global_data);
  ASSERT_EQ(pass.Run(root_frame->GetExecuteGraph().get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);
  ASSERT_EQ(root_frame->GetExecuteGraph()->TopologicalSorting(), ge::SUCCESS);

  EXPECT_EQ(ExeGraphSummaryChecker(main_frame->GetExecuteGraph().get())
                .StrictDirectNodeTypes(std::map<std::string, size_t>{{"Data", 2},
                                                                     {"InnerData", 6},
                                                                     {"If", 1},
                                                                     {"SplitRtStreams", 1},
                                                                     {"SelectL2Allocator", 1},
                                                                     {"NetOutput", 1}}),
            "success");

  EXPECT_EQ(ExeGraphSummaryChecker(init_frame_->GetExecuteGraph().get())
                .StrictDirectNodeTypes(std::map<std::string, size_t>{{"Const", 8},
                                                                     {"Data", 1},
                                                                     {"SplitRtStreams", 1},
                                                                     {"Foo1", 1},
                                                                     {"CreateL1Allocator", 1},
                                                                     {"CreateL2Allocators", 1},
                                                                     {"NoOp", 1},
                                                                     {"CreateInitL2Allocator", 1},
                                                                     {"InnerNetOutput", 1}}),
            "success");
  EXPECT_EQ(ExeGraphSummaryChecker(de_init_frame_->GetExecuteGraph().get())
                .StrictDirectNodeTypes(std::map<std::string, size_t>{{"InnerData", 1}, {"FreeFoo1", 1}}),
            "success");

  auto if_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(main_frame->GetExecuteGraph().get(), "If");
  ASSERT_NE(if_node, nullptr);
  auto then_graph = ge::FastNodeUtils::GetSubgraphFromNode(if_node, 1U);
  ASSERT_NE(then_graph, nullptr);
  auto else_graph = ge::FastNodeUtils::GetSubgraphFromNode(if_node, 2U);
  ASSERT_NE(else_graph, nullptr);

  EXPECT_EQ(
      ExeGraphSummaryChecker(then_graph)
          .StrictDirectNodeTypes(std::map<std::string, size_t>{{"InnerData", 2}, {"Foo2", 1}, {"InnerNetOutput", 1}}),
      "success");
  EXPECT_EQ(
      ExeGraphSummaryChecker(else_graph)
          .StrictDirectNodeTypes(std::map<std::string, size_t>{{"InnerData", 1}, {"Bar", 1}, {"InnerNetOutput", 1}}),
      "success");

  auto foo1_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_frame_->GetExecuteGraph().get(), "Foo1");
  ASSERT_NE(foo1_node, nullptr);
  auto foo1_guarder_node =
      ge::ExecuteGraphUtils::FindFirstNodeMatchType(de_init_frame_->GetExecuteGraph().get(), "FreeFoo1");
  ASSERT_NE(foo1_guarder_node, nullptr);
  auto foo2_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(then_graph, "Foo2");
  ASSERT_NE(foo2_node, nullptr);
  EXPECT_EQ(
      FastNodeTopoChecker(foo1_node).StrictConnectFrom({{"CreateInitL2Allocator", 0}, {"Const", 0}, {"NoOp", -1}}),
      "success");

  ConnectFromInitToMain(foo1_node, 0, foo2_node, 0);
  ConnectFromInitToDeInit(foo1_node, 0, foo1_guarder_node, 0);

  changed = false;
  ASSERT_EQ(pass.Run(root_frame->GetExecuteGraph().get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_FALSE(changed);
}
}  // namespace bg
}  // namespace gert

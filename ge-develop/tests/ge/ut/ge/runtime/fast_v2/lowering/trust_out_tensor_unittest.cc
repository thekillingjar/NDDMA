/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "lowering/pass/trust_out_tensor.h"
#include <gtest/gtest.h>
#include "common/bg_test.h"
#include "common/topo_checker.h"
#include "common/summary_checker.h"
#include "common/exe_graph.h"
#include "graph/utils/execute_graph_utils.h"
#include "exe_graph/lowering/frame_selector.h"
#include "exe_graph/lowering/lowering_global_data.h"
#include "graph_builder/bg_model_desc.h"
#include "graph/utils/graph_dump_utils.h"
#include "framework/runtime/gert_const_types.h"
namespace gert {
namespace bg {
namespace {
ExeGraph BuildGraph1() {
  auto in_shape = ValueHolder::CreateFeed(0);
  auto tensor_data = ValueHolder::CreateFeed(1);
  auto tensor_attr = ValueHolder::CreateConst("Hello", 5);
  auto stream = ValueHolder::CreateFeed(-1);

  auto output_data = ValueHolder::CreateSingleDataOutput("OutputData", {});
  auto out_shape = ValueHolder::CreateSingleDataOutput("InferShape", {in_shape});
  auto tiling_result = ValueHolder::CreateSingleDataOutput("Tiling", {in_shape, out_shape});
  auto copy_holder = ValueHolder::CreateVoid<bg::ValueHolder>(
      "EnsureTensorAtOutMemory", {out_shape, tensor_data, tensor_attr, stream, output_data});

  auto main_frame = ValueHolder::PopGraphFrame({output_data}, {}, "NetOutput");
  auto frame = ValueHolder::PopGraphFrame();
  return ExeGraph(frame->GetExecuteGraph());
}
ExeGraph BuildGraph2() {
  auto in_shape = ValueHolder::CreateFeed(0);
  auto tensor_attr = ValueHolder::CreateConst("Hello", 5);
  auto stream = ValueHolder::CreateFeed(-1);

  auto output_datas = ValueHolder::CreateDataOutput("OutputData", {}, 3);
  auto out_shapes = ValueHolder::CreateDataOutput("InferShape", {in_shape}, 3);
  auto tiling_result =
      ValueHolder::CreateSingleDataOutput("Tiling", {in_shape, out_shapes[0], out_shapes[1], out_shapes[2]});
  for (int32_t i = 0; i < 3; ++i) {
    auto copy_holder = ValueHolder::CreateVoid<bg::ValueHolder>(
        "EnsureTensorAtOutMemory",
        {out_shapes[i], ValueHolder::CreateFeed(i + 1), tensor_attr, stream, output_datas[i]});
  }

  auto main_frame = ValueHolder::PopGraphFrame(output_datas, {}, "NetOutput");
  auto frame = ValueHolder::PopGraphFrame();
  return ExeGraph(frame->GetExecuteGraph());
}
ExeGraph BuildGraph3() {
  auto in_shape = ValueHolder::CreateFeed(0);
  auto tensor_attr = ValueHolder::CreateConst("Hello", 5);
  auto stream = ValueHolder::CreateFeed(-1);

  auto output_datas = ValueHolder::CreateDataOutput("OutputData", {}, 3);
  auto out_shapes = ValueHolder::CreateDataOutput("InferShape", {in_shape}, 3);
  auto tiling_result = ValueHolder::CreateSingleDataOutput("Tiling", {in_shape, out_shapes[0], out_shapes[1]});
  for (int32_t i = 0; i < 2; ++i) {
    auto copy_holder = ValueHolder::CreateVoid<bg::ValueHolder>(
        "EnsureTensorAtOutMemory",
        {out_shapes[i], ValueHolder::CreateFeed(i + 1), tensor_attr, stream, output_datas[i]});
  }

  auto main_frame = ValueHolder::PopGraphFrame(output_datas, {}, "NetOutput");
  auto frame = ValueHolder::PopGraphFrame();
  return ExeGraph(frame->GetExecuteGraph());
}
ExeGraph BuildGraph4() {
  auto in_shape = ValueHolder::CreateFeed(0);
  auto tensor_attr = ValueHolder::CreateConst("Hello", 5);
  auto stream = ValueHolder::CreateFeed(-1);

  auto output_datas = ValueHolder::CreateDataOutput("OutputData", {}, 3);
  auto out_shapes = ValueHolder::CreateDataOutput("InferShape", {in_shape}, 3);
  auto tiling_result =
      ValueHolder::CreateSingleDataOutput("Tiling", {in_shape, out_shapes[0], out_shapes[1], out_shapes[2]});
  for (int32_t i = 0; i < 2; ++i) {
    auto copy_holder = ValueHolder::CreateVoid<bg::ValueHolder>(
        "EnsureTensorAtOutMemory",
        {out_shapes[i], ValueHolder::CreateFeed(i + 1), tensor_attr, stream, output_datas[i]});
  }

  auto main_frame = ValueHolder::PopGraphFrame(output_datas, {}, "NetOutput");
  auto frame = ValueHolder::PopGraphFrame();
  return ExeGraph(frame->GetExecuteGraph());
}

/**
 *
 *                ---> EnsureTensorAtOutMemory(x2) <---
 *              /          |                |          \
 *   AllocMemory   tensor_attr(const)  stream(Data)  OutputData
 *
 *                                                                       EnsureTensorAtOutMemory  <-------+
 *                                                                     /  |           |        \          |
 *                                                                CopyD2H |           |         \         |
 *                                                                 /    \/            |          \        |
 *                                                       SyncStream     +             |           \       |
 *                                                           |         /              |            \      |
 *       +---------------------+-----------+-------->   KernelLaunch  /               |             |     |
 *       |                     |           |                |||      /                |             |     |
 *       |                     |   in_tensor_data(Data)  AllocMemory                  |             |     |
 *       |                     \                        /       \\\                   |             |     |
 *    Tiling                    \             allocator(Data)   CalcTensorSize   tensor_attr(const) |  OutputData
 *   |   |||                     \              |||                                                 |
 *   | InferShape ================+==============+                                                  |
 *   \    |                       |                                                                 |
 *     in_shape(Data)        stream(Data)-----------------------------------------------------------+
 */
ExeGraph BuildGraph5() {
  auto in_shape = ValueHolder::CreateFeed(0);
  auto in_tensor_data = ValueHolder::CreateFeed(1);
  auto tensor_attr = ValueHolder::CreateConst("Hello", 5);
  auto allocator = ValueHolder::CreateSingleDataOutput("InnerData", {});
  auto stream = ValueHolder::CreateFeed(-1);

  auto output_datas = ValueHolder::CreateDataOutput("OutputData", {}, 3);
  auto out_max_shapes = ValueHolder::CreateDataOutput("InferShape", {in_shape}, 3);
  auto out_sizes = ValueHolder::CreateDataOutput("CalcTensorSize", out_max_shapes, out_max_shapes.size());
  std::vector<ValueHolderPtr> out_addrs;
  out_addrs.reserve(out_sizes.size());
  for (size_t i = 0; i < 3; ++i) {
    out_addrs.emplace_back(ValueHolder::CreateSingleDataOutput("AllocMemory", {allocator, out_sizes[i]}));
  }
  auto tiling_result = ValueHolder::CreateSingleDataOutput(
      "Tiling", {in_shape, out_max_shapes[0], out_max_shapes[1], out_max_shapes[2]});
  auto launch_holder = ValueHolder::CreateVoid<bg::ValueHolder>(
      "KernelLaunch", {stream, in_tensor_data, out_addrs[0], out_addrs[1], out_addrs[2], tiling_result});
  auto sync_stream_holder = ValueHolder::CreateVoid<bg::ValueHolder>("SyncStream", {stream});
  ValueHolder::AddDependency(launch_holder, sync_stream_holder);
  auto host_addr = ValueHolder::CreateSingleDataOutput("CopyD2H", {out_addrs[1]});
  ValueHolder::AddDependency(sync_stream_holder, host_addr);

  for (int32_t i = 0; i < 3; ++i) {
    if (i == 1) {
      ValueHolder::CreateVoid<bg::ValueHolder>("EnsureTensorAtOutMemory",
                                               {host_addr, out_addrs[i], tensor_attr, stream, output_datas[i]});
    } else {
      ValueHolder::CreateVoid<bg::ValueHolder>("EnsureTensorAtOutMemory",
                                               {out_max_shapes[i], out_addrs[i], tensor_attr, stream, output_datas[i]});
    }
  }

  auto main_frame = ValueHolder::PopGraphFrame(output_datas, {}, "NetOutput");
  auto frame = ValueHolder::PopGraphFrame();
  return ExeGraph(frame->GetExecuteGraph());
}

/*
 *                            NetOutput
 *                             /c    ^
 *    + ---> EnsureTensorAtOutMemory |
 *    |             |   |     |    \ |
 *    |             |  attrs stream \|
 *    |             |                +
 *    |             |      SomeNodes |
 *    |             |          |     |
 *    |    SplitTensorForOutputData  |
 *    |      |     |                 |
 *    |      | CalcSize              |
 *    |      |     |                 |
 *    +------+-----+-- OutputData ---+
 */
ExeGraph BuildGraph7() {
  auto output_data = bg::ValueHolder::CreateSingleDataOutput("OutputData", {});
  auto tensor_attrs = bg::ValueHolder::CreateConst("Hello", 5);
  auto stream = bg::ValueHolder::CreateFeed(-1);
  ge::DataType dt = ge::DT_FLOAT;
  auto calc_size = bg::ValueHolder::CreateSingleDataOutput("CalcTensorSizeFromShape",
                                                           {ValueHolder::CreateConst(&dt, sizeof(dt)), output_data});
  auto allocator = ValueHolder::CreateSingleDataOutput("InnerData", {});
  auto shape_and_tensor_data =
      ValueHolder::CreateDataOutput("SplitTensorForOutputData", {output_data, allocator, calc_size}, 2);
  auto launch =
      ValueHolder::CreateVoid<bg::ValueHolder>("Launch", {ValueHolder::CreateFeed(0), shape_and_tensor_data.at(1)});
  auto ensure_holder = ValueHolder::CreateVoid<bg::ValueHolder>(
      "EnsureTensorAtOutMemory", {output_data, shape_and_tensor_data.at(1), tensor_attrs, stream, output_data});

  auto main_frame = ValueHolder::PopGraphFrame({output_data}, {}, "NetOutput");
  auto frame = ValueHolder::PopGraphFrame();
  return ExeGraph(frame->GetExecuteGraph());
}
/*
 *                            NetOutput
 *                             /c    ^
 *    + ---> EnsureTensorAtOutMemory |
 *    |             |   |     |    \ |
 *    |             |  attrs stream \|
 *    |             |                +
 *    |             |      SomeNodes |
 *    |             |          |     |
 *    |    SplitTensorForOutputData  |
 *    |      |     |                 |
 *    |      | CalcSize              |
 *    |      |     |                 |
 * shape     +-----+-- OutputData ---+
 */
ExeGraph BuildGraph8() {
  auto output_data = bg::ValueHolder::CreateSingleDataOutput("OutputData", {});
  auto tensor_attrs = bg::ValueHolder::CreateConst("Hello", 5);
  auto stream = bg::ValueHolder::CreateFeed(-1);
  auto input_addr = ValueHolder::CreateFeed(0);
  auto input_shape = ValueHolder::CreateFeed(1);
  ge::DataType dt = ge::DT_FLOAT;
  auto calc_size = bg::ValueHolder::CreateSingleDataOutput("CalcTensorSizeFromShape",
                                                           {ValueHolder::CreateConst(&dt, sizeof(dt)), output_data});
  auto allocator = ValueHolder::CreateSingleDataOutput("InnerData", {});
  auto shape_and_tensor_data =
      ValueHolder::CreateDataOutput("SplitTensorForOutputData", {output_data, allocator, calc_size}, 2);
  auto launch = ValueHolder::CreateVoid<bg::ValueHolder>("Launch", {input_addr, shape_and_tensor_data.at(1)});
  auto ensure_holder = ValueHolder::CreateVoid<bg::ValueHolder>(
      "EnsureTensorAtOutMemory", {input_shape, shape_and_tensor_data.at(1), tensor_attrs, stream, output_data});

  auto main_frame = ValueHolder::PopGraphFrame({output_data}, {}, "NetOutput");
  auto frame = ValueHolder::PopGraphFrame();
  return ExeGraph(frame->GetExecuteGraph());
}
/*
 *                            NetOutput
 *                             /c    ^
 *    + ---> EnsureTensorAtOutMemory |
 *    |             |   |     |    \ |
 *    |             |  attrs stream \|
 *    |             |                +
 *    |             |      SomeNodes |
 *    |             |          |     |
 *    |    SplitTensorForOutputData  |
 *    |      |     |                 |
 *    |      | CalcSize              |
 *    |      |     |                 |
 *    |      |   shape               |
 *    |      |                       |
 *    +------+-------- OutputData ---+
 */
ExeGraph BuildGraph9() {
  auto output_data = bg::ValueHolder::CreateSingleDataOutput("OutputData", {});
  auto tensor_attrs = bg::ValueHolder::CreateConst("Hello", 5);
  auto stream = bg::ValueHolder::CreateFeed(-1);
  auto input_addr = ValueHolder::CreateFeed(0);
  auto input_shape = ValueHolder::CreateFeed(1);
  ge::DataType dt = ge::DT_FLOAT;
  auto c_dtype = ValueHolder::CreateConst(&dt, sizeof(dt));
  auto calc_size = bg::ValueHolder::CreateSingleDataOutput("CalcTensorSizeFromShape", {c_dtype, input_shape});
  auto shape_and_tensor_data = ValueHolder::CreateDataOutput("SplitTensorForOutputData", {output_data, calc_size}, 2);
  auto launch = ValueHolder::CreateVoid<bg::ValueHolder>("Launch", {input_addr, shape_and_tensor_data.at(1)});
  auto ensure_holder = ValueHolder::CreateVoid<bg::ValueHolder>(
      "EnsureTensorAtOutMemory", {output_data, shape_and_tensor_data.at(1), tensor_attrs, stream, output_data});

  auto main_frame = ValueHolder::PopGraphFrame({output_data}, {}, "NetOutput");
  auto frame = ValueHolder::PopGraphFrame();
  return ExeGraph(frame->GetExecuteGraph());
}
/*
 *                            NetOutput
 *                             /c    ^
 *    + ---> EnsureTensorAtOutMemory |
 *    |             |   |     |    \ |
 *    |             |  attrs stream \|
 *    |             |                +
 *    |             |      SomeNodes |
 *    |             |          |     |
 *    |    SplitTensorForOutputData  |
 *    |      |   |                   |
 *    |      |   | SomeNodes         |
 *    |      |   |    |              |
 *    |      |  CalcSize             |
 *    |      |     |                 |
 *    +------+-----+-- OutputData ---+
 */
ExeGraph BuildGraph10() {
  auto output_data = bg::ValueHolder::CreateSingleDataOutput("OutputData", {});
  auto tensor_attrs = bg::ValueHolder::CreateConst("Hello", 5);
  auto stream = bg::ValueHolder::CreateFeed(-1);
  ge::DataType dt = ge::DT_FLOAT;
  auto calc_size = bg::ValueHolder::CreateSingleDataOutput("CalcTensorSizeFromShape",
                                                           {ValueHolder::CreateConst(&dt, sizeof(dt)), output_data});
  auto foo = ValueHolder::CreateVoid<bg::ValueHolder>("Foo", {calc_size});
  auto shape_and_tensor_data = ValueHolder::CreateDataOutput("SplitTensorForOutputData", {output_data, calc_size}, 2);
  auto launch =
      ValueHolder::CreateVoid<bg::ValueHolder>("Launch", {ValueHolder::CreateFeed(0), shape_and_tensor_data.at(1)});
  auto ensure_holder = ValueHolder::CreateVoid<bg::ValueHolder>(
      "EnsureTensorAtOutMemory", {output_data, shape_and_tensor_data.at(1), tensor_attrs, stream, output_data});

  auto main_frame = ValueHolder::PopGraphFrame({output_data}, {}, "NetOutput");
  auto frame = ValueHolder::PopGraphFrame();
  return ExeGraph(frame->GetExecuteGraph());
}
/*
 *                            NetOutput
 *                             /c    ^
 *    + ---> EnsureTensorAtOutMemory |
 *    |             |   |     |    \ |
 *    |             |  attrs stream \|
 *    |             |                +
 *    |             |      SomeNodes |
 *    |             |          |     |
 *    |    SplitTensorForOutputData  |
 *    |      |     |                 |
 *    |      | SpecCalcSize          |
 *    |      |     |                 |
 *    +------+-----+-- OutputData ---+
 */
ExeGraph BuildGraph11() {
  auto output_data = bg::ValueHolder::CreateSingleDataOutput("OutputData", {});
  auto tensor_attrs = bg::ValueHolder::CreateConst("Hello", 5);
  auto stream = bg::ValueHolder::CreateFeed(-1);
  ge::DataType dt = ge::DT_FLOAT;
  auto calc_size = bg::ValueHolder::CreateSingleDataOutput("SpecCalcTensorSizeFromShape",
                                                           {ValueHolder::CreateConst(&dt, sizeof(dt)), output_data});
  auto shape_and_tensor_data = ValueHolder::CreateDataOutput("SplitTensorForOutputData", {output_data, calc_size}, 2);
  auto launch =
      ValueHolder::CreateVoid<bg::ValueHolder>("Launch", {ValueHolder::CreateFeed(0), shape_and_tensor_data.at(1)});
  auto ensure_holder = ValueHolder::CreateVoid<bg::ValueHolder>(
      "EnsureTensorAtOutMemory", {output_data, shape_and_tensor_data.at(1), tensor_attrs, stream, output_data});

  auto main_frame = ValueHolder::PopGraphFrame({output_data}, {}, "NetOutput");
  auto frame = ValueHolder::PopGraphFrame();
  return ExeGraph(frame->GetExecuteGraph());
}
ExeGraph BuildGraph12() {
  LoweringGlobalData global_data;
  size_t const_data_num = static_cast<size_t>(ConstDataType::kTypeEnd);
  auto const_data_outputs = bg::FrameSelector::OnInitRoot([&]() -> std::vector<bg::ValueHolderPtr> {
    std::vector<bg::ValueHolderPtr> const_datas;
    for (size_t i = 0U; i < const_data_num; ++i) {
      auto const_data_holder = bg::ValueHolder::CreateConstData(static_cast<int64_t>(i));
      const_datas.emplace_back(const_data_holder);
    }
    return const_datas;
  });
  for (size_t i = 0U; i < const_data_num; ++i) {
    const auto &const_data_name = GetConstDataTypeStr(static_cast<ConstDataType>(i));
    global_data.SetUniqueValueHolder(const_data_name, const_data_outputs[i]);
  }

  std::string type = "Add";
  auto infer_func = bg::FrameSelector::OnInitRoot([&type, &global_data]() -> std::vector<bg::ValueHolderPtr> {
    auto node_type = ValueHolder::CreateConst(type.c_str(), type.size() + 1, true);
    auto space_registry = bg::HolderOnInit(bg::GetSpaceRegistry(global_data));
    return {ValueHolder::CreateSingleDataOutput("FindInferShapeFunc", {node_type, space_registry})};
  });

  auto in_shape = ValueHolder::CreateFeed(0);
  auto tensor_attr = ValueHolder::CreateConst("Hello", 5);
  auto stream = ValueHolder::CreateFeed(-1);

  std::vector<ValueHolderPtr> inputs{in_shape, infer_func[0]};

  auto output_datas = ValueHolder::CreateDataOutput("OutputData", {}, 3);
  auto out_shapes = ValueHolder::CreateDataOutput("InferShape", inputs, 3);
  auto tiling_result = ValueHolder::CreateSingleDataOutput("Tiling", {in_shape, out_shapes[0], out_shapes[1]});
  for (int32_t i = 0; i < 2; ++i) {
    auto copy_holder = ValueHolder::CreateVoid<bg::ValueHolder>(
        "EnsureTensorAtOutMemory",
        {out_shapes[i], ValueHolder::CreateFeed(i + 1), tensor_attr, stream, output_datas[i]});
  }

  auto main_frame = ValueHolder::PopGraphFrame(output_datas, {}, "NetOutput");
  auto frame = ValueHolder::PopGraphFrame();
  return ExeGraph(frame->GetExecuteGraph());
}
}  // namespace
class TrustOutTensorUT : public BgTestAutoCreate3StageFrame {};

TEST_F(TrustOutTensorUT, TrustOutShape_DoNothing_NotEnable) {
  auto graph = BuildGraph1();
  bool changed = false;
  ASSERT_EQ(TrustOutTensor().Run(graph.GetMainGraph().get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_FALSE(changed);
}
TEST_F(TrustOutTensorUT, TrustOutShape_InferShapeDeleted_SingleOutShape) {
  auto graph = BuildGraph1();
  bool changed = false;
  auto pass = TrustOutTensor();
  pass.SetLoweringOption({.trust_shape_on_out_tensor = true});
  ASSERT_EQ(pass.Run(graph.GetMainGraph().get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);

  auto main_graph = graph.GetMainGraph().get();
  ASSERT_NE(main_graph, nullptr);
  EXPECT_EQ(ge::ExecuteGraphUtils::FindFirstNodeMatchType(main_graph, "InferShape"), nullptr);

  auto output_data_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(main_graph, "OutputData");
  ASSERT_NE(output_data_node, nullptr);
  EXPECT_EQ(
      FastNodeTopoChecker(output_data_node)
          .StrictConnectTo(
              0, {{"Tiling", 1}, {"EnsureTensorAtOutMemory", 0}, {"EnsureTensorAtOutMemory", 4}, {"NetOutput", 0}}),
      "success");

  auto copy_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(main_graph, "EnsureTensorAtOutMemory");
  ASSERT_NE(copy_node, nullptr);
  EXPECT_EQ(
      FastNodeTopoChecker(copy_node).StrictConnectFrom({{"OutputData"}, {"Data"}, {"Const"}, {"Data"}, {"OutputData"}}),
      "success");

  auto tiling_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(main_graph, "Tiling");
  ASSERT_NE(tiling_node, nullptr);
  EXPECT_EQ(FastNodeTopoChecker(tiling_node).StrictConnectFrom({{"Data"}, {"OutputData"}}), "success");
}
TEST_F(TrustOutTensorUT, TrustOutShape_InferShapeDeleted_ThreeOutShape) {
  auto graph = BuildGraph2();
  bool changed = false;
  auto pass = TrustOutTensor();
  pass.SetLoweringOption({.trust_shape_on_out_tensor = true});
  ASSERT_EQ(pass.Run(graph.GetMainGraph().get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);
  auto main_graph = graph.GetMainGraph().get();
  ASSERT_NE(main_graph, nullptr);

  EXPECT_EQ(ExeGraphSummaryChecker(main_graph)
                .StrictDirectNodeTypes({{"Data", 5},
                                        {"Const", 1},
                                        {"OutputData", 1},
                                        {"NetOutput", 1},
                                        {"Tiling", 1},
                                        {"EnsureTensorAtOutMemory", 3}}),
            "success");

  auto output_data = ge::ExecuteGraphUtils::FindFirstNodeMatchType(main_graph, "OutputData");
  ASSERT_NE(output_data, nullptr);
  EXPECT_EQ(
      FastNodeTopoChecker(output_data)
          .StrictConnectTo(
              0, {{"Tiling", 1}, {"EnsureTensorAtOutMemory", 0}, {"EnsureTensorAtOutMemory", 4}, {"NetOutput", 0}}),
      "success");
  EXPECT_EQ(
      FastNodeTopoChecker(output_data)
          .StrictConnectTo(
              1, {{"Tiling", 2}, {"EnsureTensorAtOutMemory", 0}, {"EnsureTensorAtOutMemory", 4}, {"NetOutput", 1}}),
      "success");
  EXPECT_EQ(
      FastNodeTopoChecker(output_data)
          .StrictConnectTo(
              2, {{"Tiling", 3}, {"EnsureTensorAtOutMemory", 0}, {"EnsureTensorAtOutMemory", 4}, {"NetOutput", 2}}),
      "success");
}
TEST_F(TrustOutTensorUT, TrustOutShape_InferShapeDeleted_ThreeOutShapeOneNotConnected) {
  auto graph = BuildGraph3();
  bool changed = false;
  auto pass = TrustOutTensor();
  pass.SetLoweringOption({.trust_shape_on_out_tensor = true});
  ASSERT_EQ(pass.Run(graph.GetMainGraph().get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);
  auto main_graph = graph.GetMainGraph().get();
  ASSERT_NE(main_graph, nullptr);

  EXPECT_EQ(ExeGraphSummaryChecker(main_graph)
                .StrictDirectNodeTypes({{"Data", 4},
                                        {"Const", 1},
                                        {"OutputData", 1},
                                        {"NetOutput", 1},
                                        {"Tiling", 1},
                                        {"EnsureTensorAtOutMemory", 2}}),
            "success");
}
TEST_F(TrustOutTensorUT, TrustOutShape_InferShapeReserved_ThreeOutShapeTwoModelOut) {
  auto graph = BuildGraph4();
  bool changed = false;
  auto pass = TrustOutTensor();
  pass.SetLoweringOption({.trust_shape_on_out_tensor = true});
  ASSERT_EQ(pass.Run(graph.GetMainGraph().get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_FALSE(changed);
  auto main_graph = graph.GetMainGraph().get();
  ASSERT_NE(main_graph, nullptr);

  EXPECT_EQ(ExeGraphSummaryChecker(main_graph)
                .StrictDirectNodeTypes({{"Data", 4},
                                        {"Const", 1},
                                        {"OutputData", 1},
                                        {"NetOutput", 1},
                                        {"InferShape", 1},
                                        {"Tiling", 1},
                                        {"EnsureTensorAtOutMemory", 2}}),
            "success");
}
// 第三类算子
TEST_F(TrustOutTensorUT, TrustOutShape_DoNothing_ShapeNotGeneratedByInferShape) {
  auto graph = BuildGraph5();
  bool changed = false;
  auto pass = TrustOutTensor();
  pass.SetLoweringOption({.trust_shape_on_out_tensor = true});
  ASSERT_EQ(pass.Run(graph.GetMainGraph().get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_FALSE(changed);
  auto main_graph = graph.GetMainGraph().get();
  ASSERT_NE(main_graph, nullptr);

  EXPECT_EQ(ExeGraphSummaryChecker(main_graph)
                .StrictDirectNodeTypes({{"Data", 3},
                                        {"InnerData", 1},
                                        {"Const", 1},
                                        {"AllocMemory", 3},
                                        {"CalcTensorSize", 1},
                                        {"CopyD2H", 1},
                                        {"KernelLaunch", 1},
                                        {"SyncStream", 1},
                                        {"OutputData", 1},
                                        {"NetOutput", 1},
                                        {"InferShape", 1},
                                        {"Tiling", 1},
                                        {"EnsureTensorAtOutMemory", 3}}),
            "success");
}
TEST_F(TrustOutTensorUT, TrustOutShape_DeleteEnsureAndCalcSizeNode) {
  auto graph = BuildGraph7();
  bool changed = false;
  LoweringOption oo;
  oo.trust_shape_on_out_tensor = true;
  TrustOutTensor pass;
  pass.SetLoweringOption(oo);
  ASSERT_EQ(pass.Run(graph.GetMainGraph().get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);

  auto main_graph = graph.GetMainGraph().get();
  ASSERT_NE(main_graph, nullptr);
  EXPECT_EQ(ExeGraphSummaryChecker(main_graph)
                .StrictDirectNodeTypes({{"NetOutput", 1},
                                        {"Data", 2},
                                        {"InnerData", 1},
                                        {"OutputData", 1},
                                        {"SplitTensorForOutputData", 1},
                                        {"Launch", 1}}),
            "success");

  auto split_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(main_graph, "SplitTensorForOutputData");
  ASSERT_NE(split_node, nullptr);
  EXPECT_EQ(split_node->GetDataInNum(), 2U);
  EXPECT_EQ(split_node->GetDataOutNum(), 2U);
  EXPECT_EQ(
      FastNodeTopoChecker(split_node).StrictConnectFrom({{"OutputData", 0}, {"InnerData", 0}, {"OutputData", -1}}),
      "success");
  EXPECT_EQ(FastNodeTopoChecker(split_node).StrictConnectTo(1, {{"Launch"}}), "success");
}
TEST_F(TrustOutTensorUT, TrustOutShape_DeleteSizeCalc_WhenOutputDataNotConnectedAsShape) {
  auto graph = BuildGraph8();
  bool changed = false;
  LoweringOption oo;
  oo.trust_shape_on_out_tensor = true;
  TrustOutTensor pass;
  pass.SetLoweringOption(oo);
  ge::DumpGraph(graph.GetMainGraph().get(), "BeforeFastTrustOutShape");
  ASSERT_EQ(pass.Run(graph.GetMainGraph().get(), changed), ge::GRAPH_SUCCESS);
  ge::DumpGraph(graph.GetMainGraph().get(), "AfterFastTrustOutShape");
  ASSERT_TRUE(changed);

  auto main_graph = graph.GetMainGraph().get();
  ASSERT_NE(main_graph, nullptr);
  EXPECT_EQ(ExeGraphSummaryChecker(main_graph)
                .StrictDirectNodeTypes({{"NetOutput", 1},
                                        {"Data", 3},
                                        {"OutputData", 1},
                                        {"EnsureTensorAtOutMemory", 1},
                                        {"InnerData", 1},
                                        {"Const", 1},
                                        {"SplitTensorForOutputData", 1},
                                        {"Launch", 1}}),
            "success");

  auto split_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(main_graph, "SplitTensorForOutputData");
  ASSERT_NE(split_node, nullptr);
  EXPECT_EQ(split_node->GetDataInNum(), 2U);
  EXPECT_EQ(split_node->GetDataOutNum(), 2U);
  EXPECT_EQ(
      FastNodeTopoChecker(split_node).StrictConnectFrom({{"OutputData", 0}, {"InnerData", 0}, {"OutputData", -1}}),
      "success");
  EXPECT_EQ(FastNodeTopoChecker(split_node).StrictConnectTo(1, {{"Launch"}, {"EnsureTensorAtOutMemory"}}), "success");

  auto copy_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(main_graph, "EnsureTensorAtOutMemory");
  ASSERT_NE(copy_node, nullptr);
  EXPECT_EQ(FastNodeTopoChecker(copy_node).StrictConnectFrom(
                {{"Data"}, {"SplitTensorForOutputData"}, {"Const"}, {"Data"}, {"OutputData"}}),
            "success");
}
TEST_F(TrustOutTensorUT, TrustOutShape_DeleteEnsureNode_WhenCalcSizeNotFromOutputData) {
  auto graph = BuildGraph9();
  bool changed = false;
  LoweringOption oo;
  oo.trust_shape_on_out_tensor = true;
  TrustOutTensor pass;
  pass.SetLoweringOption(oo);
  ASSERT_EQ(pass.Run(graph.GetMainGraph().get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);

  auto main_graph = graph.GetMainGraph().get();
  ASSERT_NE(main_graph, nullptr);
  EXPECT_EQ(ExeGraphSummaryChecker(main_graph)
                .StrictDirectNodeTypes({{"NetOutput", 1},
                                        {"Data", 3},
                                        {"Const", 1},
                                        {"OutputData", 1},
                                        {"CalcTensorSizeFromShape", 1},
                                        {"SplitTensorForOutputData", 1},
                                        {"Launch", 1}}),
            "success");

  auto split_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(main_graph, "SplitTensorForOutputData");
  ASSERT_NE(split_node, nullptr);
  EXPECT_EQ(split_node->GetDataInNum(), 2U);
  EXPECT_EQ(split_node->GetDataOutNum(), 2U);
  EXPECT_EQ(FastNodeTopoChecker(split_node).StrictConnectFrom({{"OutputData", 0}, {"CalcTensorSizeFromShape", 0}}),
            "success");
  EXPECT_EQ(FastNodeTopoChecker(split_node).StrictConnectTo(1, {{"Launch"}}), "success");
}
TEST_F(TrustOutTensorUT, TrustOutShape_DeleteEnsureNode_WhenCalcSizeHasMultipleOutputs) {
  auto graph = BuildGraph10();
  bool changed = false;
  LoweringOption oo;
  oo.trust_shape_on_out_tensor = true;
  TrustOutTensor pass;
  pass.SetLoweringOption(oo);
  ASSERT_EQ(pass.Run(graph.GetMainGraph().get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);

  auto main_graph = graph.GetMainGraph().get();
  ASSERT_NE(main_graph, nullptr);
  EXPECT_EQ(ExeGraphSummaryChecker(main_graph)
                .StrictDirectNodeTypes({{"NetOutput", 1},
                                        {"Data", 2},
                                        {"Foo", 1},
                                        {"Const", 1},
                                        {"OutputData", 1},
                                        {"CalcTensorSizeFromShape", 1},
                                        {"SplitTensorForOutputData", 1},
                                        {"Launch", 1}}),
            "success");
}
TEST_F(TrustOutTensorUT, TrustOutShape_DeleteEnsureNode_WhenCalcSizeNotFound) {
  auto graph = BuildGraph11();
  bool changed = false;
  LoweringOption oo;
  oo.trust_shape_on_out_tensor = true;
  TrustOutTensor pass;
  pass.SetLoweringOption(oo);
  ASSERT_EQ(pass.Run(graph.GetMainGraph().get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);

  auto main_graph = graph.GetMainGraph().get();
  ASSERT_NE(main_graph, nullptr);
  EXPECT_EQ(ExeGraphSummaryChecker(main_graph)
                .StrictDirectNodeTypes({{"NetOutput", 1},
                                        {"Data", 2},
                                        {"Const", 1},
                                        {"OutputData", 1},
                                        {"SpecCalcTensorSizeFromShape", 1},
                                        {"SplitTensorForOutputData", 1},
                                        {"Launch", 1}}),
            "success");
}
TEST_F(TrustOutTensorUT, TrustOutShape_DoNothing_SecondTime) {
  auto graph = BuildGraph7();
  bool changed = false;
  LoweringOption oo;
  oo.trust_shape_on_out_tensor = true;
  TrustOutTensor pass;
  pass.SetLoweringOption(oo);
  ASSERT_EQ(pass.Run(graph.GetMainGraph().get(), changed), ge::GRAPH_SUCCESS);
  changed = false;
  ASSERT_EQ(pass.Run(graph.GetMainGraph().get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_FALSE(changed);

  auto main_graph = graph.GetMainGraph().get();
  ASSERT_NE(main_graph, nullptr);
  EXPECT_EQ(ExeGraphSummaryChecker(main_graph)
                .StrictDirectNodeTypes({{"NetOutput", 1},
                                        {"Data", 2},
                                        {"InnerData", 1},
                                        {"OutputData", 1},
                                        {"SplitTensorForOutputData", 1},
                                        {"Launch", 1}}),
            "success");

  auto split_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(main_graph, "SplitTensorForOutputData");
  ASSERT_NE(split_node, nullptr);
  EXPECT_EQ(split_node->GetDataInNum(), 2U);
  EXPECT_EQ(split_node->GetDataOutNum(), 2U);
  EXPECT_EQ(
      FastNodeTopoChecker(split_node).StrictConnectFrom({{"OutputData", 0}, {"InnerData", 0}, {"OutputData", -1}}),
      "success");
  EXPECT_EQ(FastNodeTopoChecker(split_node).StrictConnectTo(1, {{"Launch"}}), "success");
}

TEST_F(TrustOutTensorUT, TrustOutShape_DeleteFindInferSuccess) {
  auto graph = BuildGraph12();

  bool changed = false;
  LoweringOption oo;
  oo.trust_shape_on_out_tensor = true;
  TrustOutTensor pass;
  pass.SetLoweringOption(oo);
  ASSERT_EQ(pass.Run(graph.GetMainGraph().get(), changed), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(changed);
  // ge::DumpGraph(graph.GetMainGraph().get(), "DeleteFindInferSuccessAfter");

  auto main_graph = graph.GetMainGraph().get();
  ASSERT_NE(main_graph, nullptr);
  EXPECT_EQ(ExeGraphSummaryChecker(main_graph)
                .StrictDirectNodeTypes({{"Data", 4},
                                        {"Const", 1},
                                        {"OutputData", 1},
                                        {"NetOutput", 1},
                                        {"Tiling", 1},
                                        {"EnsureTensorAtOutMemory", 2}}),
            "success");
  // GE_DUMP(init_frame_->GetExecuteGraph(), "DeleteFindInferSuccessAfterInit");
  auto find_infer_shape_func =
      ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_frame_->GetExecuteGraph().get(), "FindInferShapeFunc");
  ASSERT_EQ(find_infer_shape_func, nullptr);
  EXPECT_EQ(ExeGraphSummaryChecker(init_frame_->GetExecuteGraph().get())
                .StrictDirectNodeTypes(
                    {{"ConstData", 3}, {"Const", 2}, {"GetSpaceRegistry", 1}, {"NoOp", 1}, {"InnerNetOutput", 1}}),
            "success");
}
}  // namespace bg
}  // namespace gert
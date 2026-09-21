/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/lowering/generate_exe_graph.h"
#include <gtest/gtest.h>
#include "exe_graph/lowering/dev_mem_value_holder.h"
#include "checker/bg_test.h"
#include "checker/topo_checker.h"
namespace gert {
using namespace bg;
namespace {
std::vector<ValueHolderPtr> StubInferShape(const ge::NodePtr &node,
                                                         const std::vector<ValueHolderPtr> &shapes,
                                                         LoweringGlobalData &globalData) {
  return ValueHolder::CreateDataOutput("InferShape", shapes, 10);
}
std::vector<DevMemValueHolderPtr> StubAllocOutputMemory(TensorPlacement placement, const ge::NodePtr &node,
                                                        const std::vector<ValueHolderPtr> &output_sizes,
                                                        LoweringGlobalData &global_data) {
  return DevMemValueHolder::CreateDataOutput("AllocOutputMemory", output_sizes, output_sizes.size(), 0);
}
std::vector<ValueHolderPtr> StubCalcTensorSize(const ge::NodePtr &node,
                                               const std::vector<ValueHolderPtr> &output_shapes) {
  return ValueHolder::CreateDataOutput("CalcTensorSize", output_shapes, output_shapes.size());
}
ge::NodePtr FakeNode() {
  static size_t counter = 0;
  static ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  auto op_desc = std::make_shared<ge::OpDesc>("FakeNode_" + std::to_string(counter++), "FakeNode");
  return graph->AddNode(op_desc);
}
}  // namespace
class FastGenerateExeGraphUT : public BgTest {
 protected:
  void SetUp() override {
    BgTest::SetUp();
    bg::GenerateExeGraph::AddBuilderImplement({nullptr, nullptr, nullptr});
  }
  void InitTestFramesWithStream(LoweringGlobalData &global_data) {
    root_frame = bg::ValueHolder::GetCurrentFrame();
    auto init_node = bg::ValueHolder::CreateVoid<bg::ValueHolder>("Init", {});
    bg::ValueHolder::PushGraphFrame(init_node, "Init");
    global_data.LoweringAndSplitRtStreams(1);
    init_frame = bg::ValueHolder::PopGraphFrame();

    auto de_init_node = bg::ValueHolder::CreateVoid<bg::ValueHolder>("DeInit", {});
    bg::ValueHolder::PushGraphFrame(de_init_node, "DeInit");
    de_init_frame = bg::ValueHolder::PopGraphFrame();

    auto main_node = bg::ValueHolder::CreateVoid<bg::ValueHolder>(GetExecuteGraphTypeStr(ExecuteGraphType::kMain), {});
    bg::ValueHolder::PushGraphFrame(main_node, "Main");
    global_data.LoweringAndSplitRtStreams(1);
  }
  bg::GraphFrame *root_frame;
  std::unique_ptr<bg::GraphFrame> init_frame;
  std::unique_ptr<bg::GraphFrame> de_init_frame;
};

TEST_F(FastGenerateExeGraphUT, NoImpl_Failed_InferShape) {
  LoweringGlobalData gd;
  ASSERT_TRUE(bg::GenerateExeGraph::InferShape(nullptr, {bg::ValueHolder::CreateFeed(0)}, gd).empty());
}
TEST_F(FastGenerateExeGraphUT, NoImpl_Failed_AllocOutputMemory) {
  LoweringGlobalData gd;
  ASSERT_TRUE(
      bg::GenerateExeGraph::AllocOutputMemory(kOnDeviceHbm, nullptr, {bg::ValueHolder::CreateFeed(0)}, gd).empty());
}
TEST_F(FastGenerateExeGraphUT, NoImpl_Failed_CalcTensorSize) {
  LoweringGlobalData gd;
  ASSERT_TRUE(bg::GenerateExeGraph::CalcTensorSize(nullptr, {bg::ValueHolder::CreateFeed(0)}).empty());
}

TEST_F(FastGenerateExeGraphUT, StubImpl_GraphCorrect_InferShape) {
  bg::GenerateExeGraph::AddBuilderImplement({StubInferShape, nullptr, nullptr});
  auto input_shape = bg::ValueHolder::CreateFeed(0);
  LoweringGlobalData gd;
  auto shapes = bg::GenerateExeGraph::InferShape(nullptr, {input_shape}, gd);
  ASSERT_EQ(shapes.size(), 10);
  ASSERT_EQ(shapes[0]->GetFastNode()->GetType(), "InferShape");
  ASSERT_EQ(FastNodeTopoChecker(shapes[0]).StrictConnectFrom({{input_shape}}), "success");
}
TEST_F(FastGenerateExeGraphUT, StubImpl_GraphCorrect_AllocOutputMemory) {
  bg::GenerateExeGraph::AddBuilderImplement({nullptr, StubAllocOutputMemory, nullptr});
  auto input_shape0 = bg::ValueHolder::CreateFeed(0);
  auto input_shape1 = bg::ValueHolder::CreateFeed(1);

  LoweringGlobalData gd;
  auto shapes = bg::GenerateExeGraph::AllocOutputMemory(kOnDeviceHbm, nullptr, {input_shape0, input_shape1}, gd);

  ASSERT_EQ(shapes.size(), 2);
  ASSERT_EQ(shapes[0]->GetFastNode()->GetType(), "AllocOutputMemory");
  ASSERT_EQ(FastNodeTopoChecker(shapes[0]).StrictConnectFrom({{input_shape0, input_shape1}}), "success");
}
TEST_F(FastGenerateExeGraphUT, StubImpl_GraphCorrect_CalcTensorSize) {
  bg::GenerateExeGraph::AddBuilderImplement({nullptr, nullptr, StubCalcTensorSize});
  auto input_shape0 = bg::ValueHolder::CreateFeed(0);
  auto input_shape1 = bg::ValueHolder::CreateFeed(1);

  LoweringGlobalData gd;
  auto shapes = bg::GenerateExeGraph::CalcTensorSize(nullptr, {input_shape0, input_shape1});

  ASSERT_EQ(shapes.size(), 2);
  ASSERT_EQ(shapes[0]->GetFastNode()->GetType(), "CalcTensorSize");
  ASSERT_EQ(FastNodeTopoChecker(shapes[0]).StrictConnectFrom({{input_shape0, input_shape1}}), "success");
}
TEST_F(FastGenerateExeGraphUT, MakeSureTensorAtHost_Success) {
  LoweringGlobalData global_data;
  InitTestFramesWithStream(global_data);
  auto src_addr = (void *) (0x1);
  auto addr = bg::ValueHolder::CreateConst(&src_addr, sizeof(void *));
  size_t src_size = 8U;
  auto size = bg::ValueHolder::CreateConst(&src_size, sizeof(size_t));
  auto node = FakeNode().get();
  node->GetOpDesc()->SetStreamId(1);
  ASSERT_NE(bg::GenerateExeGraph::MakeSureTensorAtHost(node, global_data, addr, size), nullptr);
}
TEST_F(FastGenerateExeGraphUT, CalcTensorSizeFromShape_Success) {
  LoweringGlobalData global_data;
  InitTestFramesWithStream(global_data);
  auto shape = ge::Shape(std::vector<int64_t>({1, 2, 3, 4}));
  auto shape_holder = bg::ValueHolder::CreateConst(&shape, sizeof(ge::Shape));
  ASSERT_NE(bg::GenerateExeGraph::CalcTensorSizeFromShape(ge::DT_UINT8, shape_holder), nullptr);
}
TEST_F(FastGenerateExeGraphUT, FreeMemoryGuarder_Success) {
  auto data_addr = (void *) (0x1);
  auto addr = bg::ValueHolder::CreateConst(&data_addr, sizeof(void *));
  size_t data_size = 8;
  auto size = ValueHolder::CreateConst(&data_size, sizeof(size_t));
  auto addr_holder = ValueHolder::CreateSingleDataOutput("CopyD2H", {addr, size});
  ASSERT_NE(bg::GenerateExeGraph::FreeMemoryGuarder(addr_holder), nullptr);
}
}  // namespace gert

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "lowering/placement/placed_lowering_result.h"
#include <gtest/gtest.h>
#include "common/bg_test.h"
#include "common/topo_checker.h"
#include "faker/node_faker.h"
#include "faker/global_data_faker.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_dump_utils.h"

namespace gert {
using namespace bg;
using namespace ge;
namespace {
NodePtr StubNode(const char *node_type = "DefaultNodeType") {
  return ComputeNodeFaker().NameAndType(node_type, node_type).IoNum(0, 4).Build();
}
void InitTestFramesWithStream(LoweringGlobalData &global_data, int64_t stream_num = 1) {
  auto init_out = bg::FrameSelector::OnInitRoot([&stream_num, &global_data]()-> std::vector<bg::ValueHolderPtr> {
    global_data.LoweringAndSplitRtStreams(1);
    auto stream_num_holder = bg::ValueHolder::CreateConst(&stream_num, sizeof(stream_num));
    global_data.SetUniqueValueHolder(kGlobalDataModelStreamNum, stream_num_holder);
    return {};
  });
  global_data.LoweringAndSplitRtStreams(stream_num);
}
}  // namespace
class PlacedLowerResultUT : public BgTestAutoCreate3StageFrame {};
TEST_F(PlacedLowerResultUT, ResultSetAndGet) {
  auto node = StubNode();
  ASSERT_NE(node, nullptr);

  LowerResult lower_result = {HyperStatus::Success(), {}, {ValueHolder::CreateFeed(0)}, {DevMemValueHolder::CreateSingleDataOutput("Data", {}, 0)}};
  PlacedLoweringResult plr(node, std::move(lower_result));
  ASSERT_NE(plr.GetResult(), nullptr);
  EXPECT_TRUE(plr.GetResult()->result.IsSuccess());
  EXPECT_TRUE(plr.GetResult()->order_holders.empty());
  EXPECT_EQ(plr.GetResult()->out_addrs.size(), 1);
  EXPECT_EQ(plr.GetResult()->out_shapes.size(), 1);
}
TEST_F(PlacedLowerResultUT, ResultGet_SamePlacement) {
  auto node = StubNode();
  ASSERT_NE(node, nullptr);
  LoweringGlobalData global_data;
  

  LowerResult lower_result = {
      HyperStatus::Success(), {ValueHolder::CreateFeed(2)}, {ValueHolder::CreateFeed(0)}, {DevMemValueHolder::CreateSingleDataOutput("Data", {}, 0)}};
  lower_result.out_addrs[0]->SetPlacement(kOnDeviceHbm);
  auto order_holder = lower_result.order_holders[0].get();
  auto shape = lower_result.out_shapes[0].get();
  auto address = lower_result.out_addrs[0].get();

  PlacedLoweringResult plr(node, std::move(lower_result));
  auto result = plr.GetOutputResult(global_data, 0, {kOnDeviceHbm, bg::kMainStream}, false);
  EXPECT_TRUE(result->has_init);
  ASSERT_EQ(result->order_holders.size(), 1);
  EXPECT_EQ(result->order_holders[0].get(), order_holder);
  EXPECT_EQ(result->shape.get(), shape);
  EXPECT_EQ(result->address.get(), address);
}

TEST_F(PlacedLowerResultUT, ResultGet2_DevicePlacement) {
  auto node = StubNode();
  ASSERT_NE(node, nullptr);
  LoweringGlobalData global_data;

  LowerResult lower_result = {
      HyperStatus::Success(), {ValueHolder::CreateFeed(2)}, {ValueHolder::CreateFeed(0)}, {DevMemValueHolder::CreateSingleDataOutput("Data", {}, 0)}};
  lower_result.out_addrs[0]->SetPlacement(kOnDeviceHbm);
  auto order_holder = lower_result.order_holders[0].get();

  PlacedLoweringResult plr(node, std::move(lower_result));
  auto result = plr.GetOutputTensorResult(global_data, 0, {kOnDeviceHbm, bg::kMainStream});
  EXPECT_TRUE(result->has_init);
  ASSERT_EQ(result->order_holders.size(), 1);
  EXPECT_EQ(result->order_holders[0].get(), order_holder);
}

TEST_F(PlacedLowerResultUT, ResultGet2_PlacementEndToP2p) {
  auto node = StubNode();
  ASSERT_NE(node, nullptr);
  LoweringGlobalData global_data;
  InitTestFramesWithStream(global_data, 1);

  LowerResult lower_result = {
      HyperStatus::Success(), {ValueHolder::CreateFeed(2)}, {ValueHolder::CreateFeed(0)}, {DevMemValueHolder::CreateSingleDataOutput("Data", {}, 0)}};
  lower_result.out_addrs[0]->SetPlacement(kTensorPlacementEnd);
  auto order_holder = lower_result.order_holders[0].get();

  PlacedLoweringResult plr(node, std::move(lower_result));
  auto result = plr.GetOutputTensorResult(global_data, 0, {kOnDeviceP2p, bg::kMainStream});
  EXPECT_TRUE(result->has_init);
  ASSERT_EQ(result->order_holders.size(), 1);
  EXPECT_EQ(result->order_holders[0].get(), order_holder);
}

TEST_F(PlacedLowerResultUT, ResultGet2_HbmToP2p) {
  auto node = StubNode();
  ASSERT_NE(node, nullptr);
  LoweringGlobalData global_data;
  InitTestFramesWithStream(global_data, 1);
  LowerResult lower_result = {
      HyperStatus::Success(), {ValueHolder::CreateFeed(2)}, {ValueHolder::CreateFeed(0)}, {DevMemValueHolder::CreateSingleDataOutput("Data", {}, 0)}};
  lower_result.out_addrs[0]->SetPlacement(kOnDeviceHbm);
  auto order_holder = lower_result.order_holders[0].get();

  PlacedLoweringResult plr(node, std::move(lower_result));
  auto result = plr.GetOutputTensorResult(global_data, 0, {kOnDeviceP2p, bg::kMainStream});
  EXPECT_TRUE(result->has_init);
  ASSERT_EQ(result->order_holders.size(), 1);
  EXPECT_EQ(result->order_holders[0].get(), order_holder);
}

TEST_F(PlacedLowerResultUT, ResultGet2_HostToP2p) {
  auto node = StubNode();
  ASSERT_NE(node, nullptr);
  LoweringGlobalData global_data;
  InitTestFramesWithStream(global_data, 1);
  LowerResult lower_result = {
      HyperStatus::Success(), {ValueHolder::CreateFeed(2)}, {ValueHolder::CreateFeed(0)}, {DevMemValueHolder::CreateSingleDataOutput("Data", {}, 0)}};
  lower_result.out_addrs[0]->SetPlacement(kOnHost);

  PlacedLoweringResult plr(node, std::move(lower_result));
  auto result = plr.GetOutputTensorResult(global_data, 0, {kOnDeviceP2p, bg::kMainStream});
  ASSERT_NE(result, nullptr);
  EXPECT_TRUE(result->has_init);
}

TEST_F(PlacedLowerResultUT, ResultGet2_P2pToHost) {
  auto node = StubNode();
  ASSERT_NE(node, nullptr);
  LoweringGlobalData global_data;
  InitTestFramesWithStream(global_data, 1);
  LowerResult lower_result = {
      HyperStatus::Success(), {ValueHolder::CreateFeed(2)}, {ValueHolder::CreateFeed(0)}, {DevMemValueHolder::CreateSingleDataOutput("Data", {}, 0)}};
  lower_result.out_addrs[0]->SetPlacement(kOnDeviceP2p);

  PlacedLoweringResult plr(node, std::move(lower_result));
  auto result = plr.GetOutputTensorResult(global_data, 0, {kOnHost, bg::kMainStream});
  ASSERT_NE(result, nullptr);
  EXPECT_TRUE(result->has_init);
  ASSERT_EQ(result->order_holders.size(), 1);
}

TEST_F(PlacedLowerResultUT, ResultGet2_P2pToHbm) {
  auto node = StubNode();
  ASSERT_NE(node, nullptr);
  LoweringGlobalData global_data;
  InitTestFramesWithStream(global_data, 1);
  LowerResult lower_result = {
      HyperStatus::Success(), {ValueHolder::CreateFeed(2)}, {ValueHolder::CreateFeed(0)}, {DevMemValueHolder::CreateSingleDataOutput("Data", {}, 0)}};
  lower_result.out_addrs[0]->SetPlacement(kOnDeviceP2p);

  PlacedLoweringResult plr(node, std::move(lower_result));
  auto result = plr.GetOutputTensorResult(global_data, 0, {kOnDeviceHbm, bg::kMainStream});
  ASSERT_NE(result, nullptr);
  EXPECT_TRUE(result->has_init);
  ASSERT_EQ(result->order_holders.size(), 1);
}

TEST_F(PlacedLowerResultUT, ResultGet2_DevicePlacementWithGuard) {
  auto node = StubNode();
  ASSERT_NE(node, nullptr);
  auto compute_graph = std::make_shared<ge::ComputeGraph>("tmp-graph");
  compute_graph->AddNode(node);
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  LoweringGlobalData global_data = GlobalDataFaker(root_model).Build();

  LowerResult lower_result = {
      HyperStatus::Success(), {ValueHolder::CreateFeed(2)}, {ValueHolder::CreateFeed(0)}, {DevMemValueHolder::CreateSingleDataOutput("Data", {}, 0)}};
  lower_result.out_addrs[0]->SetPlacement(kOnHost);

  PlacedLoweringResult plr(node, std::move(lower_result));
  auto result = plr.GetOutputTensorResult(global_data, 0, {kOnDeviceHbm, bg::kMainStream});
  EXPECT_TRUE(result->has_init);
}

TEST_F(PlacedLowerResultUT, ResultGet_HostPlacement) {
  auto node = StubNode();
  ASSERT_NE(node, nullptr);
  LoweringGlobalData global_data;
  
  auto streams = global_data.LoweringAndSplitRtStreams(1);

  LowerResult lower_result = {
      HyperStatus::Success(), {ValueHolder::CreateFeed(2)}, {ValueHolder::CreateFeed(0)}, {DevMemValueHolder::CreateSingleDataOutput("Data", {}, 0)}};
  lower_result.out_addrs[0]->SetPlacement(kOnDeviceHbm);

  PlacedLoweringResult plr(node, std::move(lower_result));
  auto result = plr.GetOutputTensorResult(global_data, 0, {kOnHost, bg::kMainStream});
  EXPECT_TRUE(result->has_init);
  ASSERT_EQ(result->order_holders.size(), 1);
}

TEST_F(PlacedLowerResultUT, ResultGet2_DevicePlacementMulti) {
  auto node = StubNode();
  ASSERT_NE(node, nullptr);
  LoweringGlobalData global_data;
  
  auto streams = global_data.LoweringAndSplitRtStreams(1);

  LowerResult lower_result = {
      HyperStatus::Success(), {ValueHolder::CreateFeed(2)}, {ValueHolder::CreateFeed(0)}, {DevMemValueHolder::CreateSingleDataOutput("Data", {}, 0)}};
  lower_result.out_addrs[0]->SetPlacement(kOnDeviceHbm);
  auto order_holder = lower_result.order_holders[0].get();

  PlacedLoweringResult plr(node, std::move(lower_result));
  auto result = plr.GetOutputTensorResult(global_data, 0, {kOnDeviceHbm, bg::kMainStream});
  EXPECT_TRUE(result->has_init);
  ASSERT_EQ(result->order_holders.size(), 1);
  EXPECT_EQ(result->order_holders[0].get(), order_holder);
  result = plr.GetOutputTensorResult(global_data, 0, {kOnDeviceHbm, bg::kMainStream});
  EXPECT_TRUE(result->has_init);
  ASSERT_EQ(result->order_holders.size(), 1);
  EXPECT_EQ(result->order_holders[0].get(), order_holder);
}

TEST_F(PlacedLowerResultUT, ResultGet_InvalidIndex) {
  auto node = StubNode();
  ASSERT_NE(node, nullptr);
  LoweringGlobalData global_data;
  
  auto streams = global_data.LoweringAndSplitRtStreams(1);

  LowerResult lower_result = {
      HyperStatus::Success(), {ValueHolder::CreateFeed(2)}, {ValueHolder::CreateFeed(0)}, {DevMemValueHolder::CreateSingleDataOutput("Data", {}, 0)}};
  lower_result.out_addrs[0]->SetPlacement(kOnDeviceHbm);

  PlacedLoweringResult plr(node, std::move(lower_result));
  auto result = plr.GetOutputResult(global_data, -1, {kOnHost, bg::kMainStream}, true);
  EXPECT_EQ(result, nullptr);
}

/*
 *
 *                    result
 *                  /      |
 *        BuildTensor      |
 *        /  \             |
 *   shape   CopyD2H       |
 *          |c     \       |
 *     SyncStream   \      |
 *          |        \     |
 * order_holder      address
 */
TEST_F(PlacedLowerResultUT, ResultGet_DataDependentyFromHbm) {
  auto node = StubNode();
  ASSERT_NE(node, nullptr);
  LoweringGlobalData global_data;
  auto streams = global_data.LoweringAndSplitRtStreams(1);

  LowerResult lower_result = {
      HyperStatus::Success(), {ValueHolder::CreateFeed(2)}, {ValueHolder::CreateFeed(0)}, {DevMemValueHolder::CreateSingleDataOutput("Data", {}, 0)}};
  lower_result.out_addrs[0]->SetPlacement(kOnDeviceHbm);
  auto order_holders = lower_result.order_holders;
  auto shape = lower_result.out_shapes[0];
  auto address = lower_result.out_addrs[0];

  PlacedLoweringResult plr(node, std::move(lower_result));
  auto result = plr.GetOutputResult(global_data, 0, {kOnDeviceHbm, bg::kMainStream}, true);
  EXPECT_TRUE(result->has_init);
  EXPECT_EQ(result->address, address);
  EXPECT_EQ(order_holders[0], result->order_holders[1]);

  ASSERT_NE(result->shape, nullptr);
  EXPECT_EQ(result->shape->GetFastNode()->GetType(), "BuildTensor");
  FastNodeTopoChecker tensor_checker(result->shape);
  EXPECT_EQ(tensor_checker.StrictConnectFrom(std::vector<FastSrcNode>({
                shape,           // shape
                {"CopyD2H", 0},  // address
                {"Const", 0}     // tensor attr
            })),
            "success");
  EXPECT_EQ(tensor_checker.InChecker()
                .DataFromByType("CopyD2H")
                .CtrlFromByType("SyncStream")
                .CtrlFrom(order_holders[0])
                .Result(),
            "success");
  EXPECT_EQ(tensor_checker.InChecker().DataFromByType("CopyD2H").DataFrom(address).Result(), "success");
}

/*
 *
 *               result
 *             /      |
 *   BuildTensor      |
 *     /   \          |
 * shape   address ---+
 */
TEST_F(PlacedLowerResultUT, ResultGet_DataDependentFromHost) {
  auto node = StubNode();
  LoweringGlobalData global_data;
  auto streams = global_data.LoweringAndSplitRtStreams(1);

  LowerResult lower_result = {
      HyperStatus::Success(), {ValueHolder::CreateFeed(2)}, {ValueHolder::CreateFeed(0)}, {DevMemValueHolder::CreateSingleDataOutput("Data", {}, 0)}};
  lower_result.out_addrs[0]->SetPlacement(kOnHost);

  auto order_holders = lower_result.order_holders;
  auto shape = lower_result.out_shapes[0];
  auto address = lower_result.out_addrs[0];

  PlacedLoweringResult plr(node, std::move(lower_result));
  auto result = plr.GetOutputResult(global_data, 0, {kOnHost, bg::kMainStream}, true);
  ASSERT_NE(result, nullptr);
  ASSERT_TRUE(result->has_init);

  EXPECT_EQ(result->order_holders, order_holders);
  EXPECT_EQ(result->address.get(), address.get());

  ASSERT_NE(result->shape, nullptr);
  EXPECT_EQ(result->shape->GetFastNode()->GetType(), "BuildTensor");

  DumpGraph(ValueHolder::GetCurrentFrame()->GetExecuteGraph().get(), "tmp");
  std::vector<FastSrcNode> src_nodes{shape, address, {"Const", 0}};

  EXPECT_EQ(FastNodeTopoChecker(result->shape).StrictConnectFrom(src_nodes), "success");
  EXPECT_EQ(result->address, address);
}

/*
 *       result
 *       /   \
 *   shape   CopyD2H
 *          |c     \
 *     SyncStream   \
 *          |        \
 * order_holder      address
 */
TEST_F(PlacedLowerResultUT, ResultGet_HbmToHost) {
  auto node = StubNode();
  LoweringGlobalData global_data;
  auto streams = global_data.LoweringAndSplitRtStreams(1);

  LowerResult lower_result = {
      HyperStatus::Success(), {ValueHolder::CreateFeed(2)}, {ValueHolder::CreateFeed(0)}, {DevMemValueHolder::CreateSingleDataOutput("Data", {}, 0)}};
  lower_result.out_addrs[0]->SetPlacement(kOnDeviceHbm);

  auto order_holders = lower_result.order_holders;
  auto shape = lower_result.out_shapes[0];
  auto address = lower_result.out_addrs[0];

  PlacedLoweringResult plr(node, std::move(lower_result));
  auto result = plr.GetOutputResult(global_data, 0, {kOnHost, bg::kMainStream}, false);

  EXPECT_EQ(result->shape, shape);
  EXPECT_FALSE(result->order_holders.empty());
  EXPECT_EQ(result->address->GetFastNode()->GetType(), "CopyD2H");
  FastNodeTopoChecker checker(result->address);
  EXPECT_EQ(checker.StrictConnectFrom(std::vector<FastSrcNode>({
                address,                        // address
                {"CalcTensorSizeFromStorage"},  // tensor size
                {"CreateHostL2Allocator"},
                {"SyncStream", -1}  // 来自流同步的控制边
            })),
            "success");

  auto sync_stream = ge::ExecuteGraphUtils::FindFirstNodeMatchType(result->address->GetCurrentExecuteGraph(), "SyncStream");
  ASSERT_NE(sync_stream, nullptr);
  checker = FastNodeTopoChecker(sync_stream);
  std::vector<FastSrcNode> sync_stream_inputs;
  sync_stream_inputs.emplace_back(streams[0]);
  for (const auto &ctrl_in : order_holders) {
    sync_stream_inputs.emplace_back(FastSrcNode::Name(ctrl_in->GetFastNode()->GetName(), -1));
  }
  EXPECT_EQ(checker.StrictConnectFrom(sync_stream_inputs), "success");
}

TEST_F(PlacedLowerResultUT, ResultGet_HbmToHost_Twice) {
  auto node = StubNode();
  LoweringGlobalData global_data;
  auto streams = global_data.LoweringAndSplitRtStreams(1);
  LowerResult lower_result = {
      HyperStatus::Success(), {ValueHolder::CreateFeed(4)}, {ValueHolder::CreateFeed(0), ValueHolder::CreateFeed(1)},
      {DevMemValueHolder::CreateSingleDataOutput("data",{},0), DevMemValueHolder::CreateSingleDataOutput("data",{},0)}};
  lower_result.out_addrs[0]->SetPlacement(kOnDeviceHbm);
  lower_result.out_addrs[1]->SetPlacement(kOnDeviceHbm);

  auto order_holders = lower_result.order_holders;
  std::vector<ValueHolderPtr> address = {lower_result.out_addrs[0], lower_result.out_addrs[1]};
  std::vector<ValueHolderPtr> shape = {lower_result.out_shapes[0], lower_result.out_shapes[1]};
  PlacedLoweringResult plr(node, std::move(lower_result));
  for (int i = 0; i < 2; ++i) {
    auto result = plr.GetOutputResult(global_data, i, {kOnHost, bg::kMainStream}, false);
    EXPECT_EQ(result->shape, shape[i]);
    EXPECT_FALSE(result->order_holders.empty());
    EXPECT_EQ(result->address->GetFastNode()->GetType(), "CopyD2H");
    FastNodeTopoChecker checker(result->address);
    EXPECT_EQ(checker.StrictConnectFrom(std::vector<FastSrcNode>({
                  address[i],                        // address
                  {"CalcTensorSizeFromStorage"},  // tensor size
                  {"CreateHostL2Allocator"},
                  {"SyncStream", -1}  // 来自流同步的控制边
              })),
              "success");
  }
  auto sync_stream = ge::ExecuteGraphUtils::FindFirstNodeMatchType(address[0]->GetCurrentExecuteGraph(), "SyncStream");
  ASSERT_NE(sync_stream, nullptr);
  ASSERT_EQ(sync_stream->GetOutControlNodes().size(), 2);
  FastNodeTopoChecker checker = FastNodeTopoChecker(sync_stream);
  std::vector<FastSrcNode> sync_stream_inputs;
  sync_stream_inputs.emplace_back(streams[0]);
  for (const auto &ctrl_in : order_holders) {
    sync_stream_inputs.emplace_back(FastSrcNode::Name(ctrl_in->GetFastNode()->GetName(), -1));
    sync_stream_inputs.emplace_back(FastSrcNode::Name(ctrl_in->GetFastNode()->GetName(), -1));
  }
  EXPECT_EQ(checker.StrictConnectFrom(sync_stream_inputs), "success");
}
/*
 *                result
 *               /    |
 *     BuildTensor   /
 *       /   \     /
 *   shape   CopyD2H
 *          |c     \
 *     SyncStream   \
 *          |        \
 * order_holder      address
 */
TEST_F(PlacedLowerResultUT, ResultGet_HbmToHost_DataDependent) {
  auto node = StubNode();
  LoweringGlobalData global_data;
  auto streams = global_data.LoweringAndSplitRtStreams(1);

  LowerResult lower_result = {
      HyperStatus::Success(), {ValueHolder::CreateFeed(2)}, {ValueHolder::CreateFeed(0)}, {DevMemValueHolder::CreateSingleDataOutput("Data", {}, 0)}};
  lower_result.out_addrs[0]->SetPlacement(kOnDeviceHbm);

  auto order_holders = lower_result.order_holders;
  auto shape = lower_result.out_shapes[0];
  auto address = lower_result.out_addrs[0];

  PlacedLoweringResult plr(node, std::move(lower_result));
  auto result = plr.GetOutputResult(global_data, 0, {kOnHost, bg::kMainStream}, true);

  // 数据依赖+OnHost：address应该在host，同时shape输入是一个Tensor
  EXPECT_FALSE(result->order_holders.empty());
  auto copy_d2h = ge::ExecuteGraphUtils::FindFirstNodeMatchType(result->address->GetCurrentExecuteGraph(), "CopyD2H");
  auto sync_stream = ge::ExecuteGraphUtils::FindFirstNodeMatchType(result->address->GetCurrentExecuteGraph(), "SyncStream");
  ASSERT_NE(copy_d2h, nullptr);
  ASSERT_NE(sync_stream, nullptr);

  EXPECT_EQ(result->shape->GetFastNode()->GetType(), "BuildTensor");
  FastNodeTopoChecker checker(result->shape);
  EXPECT_EQ(checker.StrictConnectFrom(std::vector<FastSrcNode>({
                shape,          // shape
                {copy_d2h, 0},  // address
                {"Const", 0}    // tensor attr
            })),
            "success");

  EXPECT_EQ(result->address->GetFastNode(), copy_d2h);
  checker = FastNodeTopoChecker(result->address);
  EXPECT_EQ(checker.StrictConnectFrom(std::vector<FastSrcNode>({
                address,                        // address
                {"CalcTensorSizeFromStorage"},  // tensor size
                {"CreateHostL2Allocator"},
                {sync_stream, -1}  // 来自流同步的控制边
            })),
            "success");

  checker = FastNodeTopoChecker(sync_stream);
  std::vector<FastSrcNode> sync_stream_inputs;
  sync_stream_inputs.emplace_back(streams[0]);
  for (const auto &ctrl_in : order_holders) {
    sync_stream_inputs.emplace_back(FastSrcNode::Name(ctrl_in->GetFastNode()->GetName(), -1));
  }
  EXPECT_EQ(checker.StrictConnectFrom(sync_stream_inputs), "success");
}

/*
 *
 * +--------------------->  result <---------+
 * |                          |              |
 * |                       CopyH2D           |
 * |                      /      |           |
 * |          CalcTensorSize     |           |
 * |              /    |         |           |
 * |  const(dtype)     shape   address   ordered_holders
 * |                    |
 * +--------------------+
 */
TEST_F(PlacedLowerResultUT, ResultGet_HostToHbm) {
  auto node = StubNode();
  auto compute_graph = std::make_shared<ge::ComputeGraph>("tmp-graph");
  compute_graph->AddNode(node);
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  LoweringGlobalData global_data = GlobalDataFaker(root_model).Build();
  auto streams = global_data.LoweringAndSplitRtStreams(1);

  LowerResult lower_result = {HyperStatus::Success(),
                              {ValueHolder::CreateFeed(2)},
                              {ValueHolder::CreateFeed(0)},
                              {DevMemValueHolder::CreateSingleDataOutput("Data", {}, 0)}};
  lower_result.out_addrs[0]->SetPlacement(kOnHost);

  auto order_holders = lower_result.order_holders;
  auto shape = lower_result.out_shapes[0];
  auto address = lower_result.out_addrs[0];

  PlacedLoweringResult plr(node, std::move(lower_result));
  auto result = plr.GetOutputResult(global_data, 0, {kOnDeviceHbm, bg::kMainStream}, false);
  EXPECT_EQ(result->shape, shape);

  EXPECT_EQ(result->address->GetFastNode()->GetType(), "CopyH2D");
  FastNodeTopoChecker checker(result->address);
  std::vector<FastSrcNode> dependency_node;
  dependency_node.emplace_back(streams[0]);
  dependency_node.emplace_back("SelectL2Allocator", 0);
  dependency_node.emplace_back(address);
  dependency_node.emplace_back("CalcUnalignedTensorSizeFromStorage", 0);
  dependency_node.emplace_back(shape);
  dependency_node.emplace_back("Const", 0); // DATA TYPE
  for (auto holder : order_holders) {
    dependency_node.emplace_back(holder->GetFastNode()->GetType(), -1);
  }
  EXPECT_EQ(checker.StrictConnectFrom(dependency_node, true), "success");
}

TEST_F(PlacedLowerResultUT, ResultGet_HostToHbmOnInit_Const) {
  auto node = StubNode("Constant");
  auto graph = std::make_shared<ge::ComputeGraph>("tmp");
  graph->AddNode(node);
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  LoweringGlobalData global_data = GlobalDataFaker(root_model).Build();
  auto streams = global_data.LoweringAndSplitRtStreams(1);
  auto shape_and_addr = bg::FrameSelector::OnInitRoot([&]() -> std::vector<bg::ValueHolderPtr> {
    auto c1 = ValueHolder::CreateConst("Hello", 5);
    global_data.LoweringAndSplitRtStreams(1);
    auto ret = DevMemValueHolder::CreateDataOutput("SplitTensor", {c1}, 2, 0);
    ret[0]->SetPlacement(kOnHost);
    ret[1]->SetPlacement(kOnHost);
    std::vector<bg::ValueHolderPtr> result(ret.begin(), ret.end());
    return result;
  });
  ASSERT_EQ(shape_and_addr.size(), 2);
  ASSERT_EQ(shape_and_addr[0]->GetPlacement(), kOnHost);
  ASSERT_EQ(shape_and_addr[1]->GetPlacement(), kOnHost);

  LowerResult lower_result = {HyperStatus::Success(), {}, {shape_and_addr[0]}, {std::dynamic_pointer_cast<DevMemValueHolder>(shape_and_addr[1])}};

  auto shape = lower_result.out_shapes[0];
  auto address = lower_result.out_addrs[0];

  PlacedLoweringResult plr(node, std::move(lower_result));
  auto result = plr.GetOutputResult(global_data, 0, {kOnDeviceHbm, bg::kMainStream}, false);
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->shape->GetOutIndex(), shape->GetOutIndex());

  EXPECT_EQ(result->address->GetFastNode()->GetType(), "Init");
  EXPECT_EQ(result->shape->GetFastNode()->GetType(), "Init");

  auto netoutput_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_frame_->GetExecuteGraph().get(), "InnerNetOutput");
  ASSERT_NE(netoutput_node, nullptr);
  auto in_data_edge = netoutput_node->GetInDataEdgeByIndex(result->address->GetOutIndex());
  ASSERT_NE(in_data_edge->src, nullptr);
  EXPECT_EQ(in_data_edge->src->GetType(), "CopyH2D");
  EXPECT_EQ(in_data_edge->src_output, 0);

  in_data_edge = netoutput_node->GetInDataEdgeByIndex(result->shape->GetOutIndex());
  ASSERT_NE(in_data_edge->src, nullptr);
  EXPECT_EQ(in_data_edge->src->GetType(), "SplitTensor");
  EXPECT_EQ(in_data_edge->src_output, 0);

  auto copy_h2d_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_frame_->GetExecuteGraph().get(), "CopyH2D");
  ASSERT_NE(copy_h2d_node, nullptr);

  EXPECT_EQ(FastNodeTopoChecker(copy_h2d_node)
                .StrictConnectFrom(
                    {{"SplitRtStreams"}, {"CreateInitL2Allocator"}, HolderOnInit(shape_and_addr[1]), {"CalcUnalignedTensorSizeFromStorage"}, HolderOnInit(shape_and_addr[0]), {"Const"}}),
            "success");
}

TEST_F(PlacedLowerResultUT, ResultGet_PassThrowOrderedHolders_WhenConstHasCtrlIn) {
  auto node = StubNode("Constant");
  auto graph = std::make_shared<ge::ComputeGraph>("tmp");
  graph->AddNode(node);
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  LoweringGlobalData global_data = GlobalDataFaker(root_model).Build();
  
  auto shape_and_addr = bg::FrameSelector::OnInitRoot([&]() -> std::vector<bg::ValueHolderPtr> {
    auto c1 = ValueHolder::CreateConst("Hello", 5);
    auto ret = ValueHolder::CreateDataOutput("SplitTensor", {c1}, 2);
    ret[0]->SetPlacement(kOnHost);
    ret[1]->SetPlacement(kOnHost);
    return ret;
  });
  ASSERT_EQ(shape_and_addr.size(), 2);
  ASSERT_EQ(shape_and_addr[0]->GetPlacement(), kOnHost);
  ASSERT_EQ(shape_and_addr[1]->GetPlacement(), kOnHost);
  auto holder0 = bg::ValueHolder::CreateFeed(0);
  auto holder1 = bg::ValueHolder::CreateFeed(1);

  LowerResult lower_result = {HyperStatus::Success(), {holder0, holder1}, {shape_and_addr[0]}, {DevMemValueHolder::CreateSingleDataOutput("Data", {}, 0)}};

  PlacedLoweringResult plr(node, std::move(lower_result));
  auto result = plr.GetOutputResult(global_data, 0, {kOnDeviceHbm, bg::kMainStream}, false);
  ASSERT_NE(result, nullptr);
  ASSERT_EQ(result->order_holders, std::vector<bg::ValueHolderPtr>({holder0, holder1}));
}
TEST_F(PlacedLowerResultUT, ResultGet_GenOnInit_WhenCtrlInFromInitNode) {
  auto node = StubNode("Constant");
  auto graph = std::make_shared<ge::ComputeGraph>("tmp");
  graph->AddNode(node);
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  LoweringGlobalData global_data = GlobalDataFaker(root_model).Build();
  
  auto shape_and_addr = bg::FrameSelector::OnInitRoot([&]() -> std::vector<bg::ValueHolderPtr> {
    global_data.LoweringAndSplitRtStreams(1);
    auto c1 = ValueHolder::CreateConst("Hello", 5);
    auto ret = DevMemValueHolder::CreateDataOutput("SplitTensor", {c1}, 2, 0);
    ret[0]->SetPlacement(kOnHost);
    ret[1]->SetPlacement(kOnHost);
    std::vector<bg::ValueHolderPtr> result(ret.begin(), ret.end());
    return result;
  });
  ASSERT_EQ(shape_and_addr.size(), 2);
  ASSERT_EQ(shape_and_addr[0]->GetPlacement(), kOnHost);
  ASSERT_EQ(shape_and_addr[1]->GetPlacement(), kOnHost);

  LowerResult lower_result = {HyperStatus::Success(), {shape_and_addr[0]}, {shape_and_addr[0]}, {std::dynamic_pointer_cast<DevMemValueHolder>(shape_and_addr[1])}};

  auto shape = lower_result.out_shapes[0];

  PlacedLoweringResult plr(node, std::move(lower_result));
  auto result = plr.GetOutputResult(global_data, 0, {kOnDeviceHbm, bg::kMainStream}, false);
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(result->shape->GetOutIndex(), shape->GetOutIndex());

  EXPECT_EQ(result->address->GetFastNode()->GetType(), "Init");
  EXPECT_EQ(result->shape->GetFastNode()->GetType(), "Init");
}
/*
 *
 * 用例ResultGet_HbmToHost_DataDependent的多输出版本：
 *
 *                result
 *               /    |
 *     BuildTensor   /
 *       /   \     /
 *   shape   CopyD2H
 *          |c     \
 *     SyncStream   \
 *          |        \
 * order_holder      address
 */
TEST_F(PlacedLowerResultUT, MultipleOuptuts_ResultGet_HbmToHost_DataDependent) {
  auto node = StubNode();
  LoweringGlobalData global_data;
  
  auto streams = global_data.LoweringAndSplitRtStreams(1);

  LowerResult lower_result = {
      HyperStatus::Success(),
      {ValueHolder::CreateFeed(8), ValueHolder::CreateFeed(9)},
      {ValueHolder::CreateFeed(0), ValueHolder::CreateFeed(1), ValueHolder::CreateFeed(2), ValueHolder::CreateFeed(3)},
      {DevMemValueHolder::CreateSingleDataOutput("data",{},0), DevMemValueHolder::CreateSingleDataOutput("data",{},0),
       DevMemValueHolder::CreateSingleDataOutput("data",{},0), DevMemValueHolder::CreateSingleDataOutput("data",{},0)}};
  lower_result.out_addrs[0]->SetPlacement(kOnDeviceHbm);

  auto order_holders = lower_result.order_holders;
  auto shape = lower_result.out_shapes[2];
  auto address = lower_result.out_addrs[2];

  PlacedLoweringResult plr(node, std::move(lower_result));
  auto result = plr.GetOutputResult(global_data, 2, {kOnHost, bg::kMainStream}, true);

  // 数据依赖+OnHost：address应该在host，同时shape输入是一个Tensor
  EXPECT_FALSE(result->order_holders.empty());
  auto copy_d2h = ge::ExecuteGraphUtils::FindFirstNodeMatchType(result->address->GetCurrentExecuteGraph(), "CopyD2H");
  auto sync_stream = ge::ExecuteGraphUtils::FindFirstNodeMatchType(result->address->GetCurrentExecuteGraph(), "SyncStream");
  ASSERT_NE(copy_d2h, nullptr);
  ASSERT_NE(sync_stream, nullptr);

  EXPECT_EQ(result->shape->GetFastNode()->GetType(), "BuildTensor");
  FastNodeTopoChecker checker(result->shape);
  EXPECT_EQ(checker.StrictConnectFrom(std::vector<FastSrcNode>({
                shape,          // shape
                {copy_d2h, 0},  // address
                {"Const", 0}    // tensor attr
            })),
            "success");

  EXPECT_EQ(result->address->GetFastNode(), copy_d2h);
  checker = FastNodeTopoChecker(result->address);
  EXPECT_EQ(checker.StrictConnectFrom(std::vector<FastSrcNode>({
                address,                        // address
                {"CalcTensorSizeFromStorage"},  // tensor size
                {"CreateHostL2Allocator"},
                {sync_stream, -1}  // 来自流同步的控制边
            })),
            "success");

  checker = FastNodeTopoChecker(sync_stream);
  std::vector<FastSrcNode> sync_stream_inputs;
  sync_stream_inputs.emplace_back(streams[0]);
  for (const auto &ctrl_in : order_holders) {
    sync_stream_inputs.emplace_back(FastSrcNode::Name(ctrl_in->GetFastNode()->GetName(), -1));
  }
  EXPECT_EQ(checker.StrictConnectFrom(sync_stream_inputs), "success");
}

/*
 *
 * +--------------------->  result <---------+
 * |                          |              |
 * |                       CopyH2D           |
 * |                      /      |           |
 * |          CalcTensorSize     |           |
 * |              /    |         |           |
 * |  const(dtype)     shape   address   ordered_holders
 * |                    |
 * +--------------------+
 */
TEST_F(PlacedLowerResultUT, ResultGet_HostToHbm_DataDependent) {
  auto node = StubNode();
  auto compute_graph = std::make_shared<ge::ComputeGraph>("tmp-graph");
  compute_graph->AddNode(node);
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  LoweringGlobalData global_data = GlobalDataFaker(root_model).Build();
  auto streams = global_data.LoweringAndSplitRtStreams(1);

  LowerResult lower_result = {
      HyperStatus::Success(), {ValueHolder::CreateFeed(2)}, {ValueHolder::CreateFeed(0)}, {DevMemValueHolder::CreateSingleDataOutput("Data", {}, 0)}};
  lower_result.out_addrs[0]->SetPlacement(kOnHost);

  auto order_holders = lower_result.order_holders;
  auto shape = lower_result.out_shapes[0];
  auto address = lower_result.out_addrs[0];

  PlacedLoweringResult plr(node, std::move(lower_result));
  auto result = plr.GetOutputResult(global_data, 0, {kOnDeviceHbm, bg::kMainStream}, true);

  EXPECT_EQ(result->address->GetFastNode()->GetType(), "CopyH2D");
  FastNodeTopoChecker checker(result->address);
  std::vector<FastSrcNode> dependency_node;
  dependency_node.emplace_back(streams[0]);
  dependency_node.emplace_back("SelectL2Allocator");
  dependency_node.emplace_back(address);
  dependency_node.emplace_back("CalcUnalignedTensorSizeFromStorage");
  dependency_node.emplace_back(shape);
  dependency_node.emplace_back("Const");
  for (auto holder : order_holders) {
    dependency_node.emplace_back(holder->GetFastNode()->GetType(), -1);
  }
  EXPECT_EQ(checker.StrictConnectFrom(dependency_node, true), "success");
  checker = FastNodeTopoChecker(result->shape);
  std::vector<FastSrcNode> src_nodes{shape, address, {"Const", 0}};
  EXPECT_EQ(FastNodeTopoChecker(result->shape).StrictConnectFrom(src_nodes), "success");
}

/*
 *                        result
 *                      /      |
 *                    /        |
 *         BuildTensor         |
 *        /        \           |
 *   shape  MakeSureAtHostDDR  |
 *          |        \         |
 *          |         \        |
 *          |          \       |
 * order_holder          address
 */
TEST_F(PlacedLowerResultUT, ResultGet_DataDependentyFromPlacemenEnd) {
  auto node = StubNode();
  ASSERT_NE(node, nullptr);
  LoweringGlobalData global_data;
  
  auto streams = global_data.LoweringAndSplitRtStreams(1);

  LowerResult lower_result = {
      HyperStatus::Success(), {ValueHolder::CreateFeed(2)}, {ValueHolder::CreateFeed(0)}, {DevMemValueHolder::CreateSingleDataOutput("Data", {}, 0)}};
  lower_result.out_addrs[0]->SetPlacement(kTensorPlacementEnd);
  auto order_holders = lower_result.order_holders;
  auto shape = lower_result.out_shapes[0];
  auto address = lower_result.out_addrs[0];

  PlacedLoweringResult plr(node, std::move(lower_result));
  auto result = plr.GetOutputResult(global_data, 0, {kOnHost, bg::kMainStream}, true);
  EXPECT_TRUE(result->has_init);
  EXPECT_NE(result->address, address);

  ASSERT_NE(result->shape, nullptr);
  EXPECT_EQ(result->shape->GetFastNode()->GetType(), "BuildTensor");
  FastNodeTopoChecker tensor_checker(result->shape);
  EXPECT_EQ(tensor_checker.StrictConnectFrom(std::vector<FastSrcNode>({
                shape,                        // shape
                {"MakeSureTensorAtHost", 0},  // address
                {"Const", 0}                  // tensor attr
            })),
            "success");
  EXPECT_EQ(tensor_checker.InChecker().DataFromByType("MakeSureTensorAtHost").DataFrom(address).Result(), "success");
}

/*
 *                        result
 *                      /      |
 *                    /        |
 *         BuildTensor         |
 *        /        \           |
 *   shape  MakeSureAtHostDDR  |
 *          |        \         |
 *          |         \        |
 *          |          \       |
 * order_holder          address
 */
TEST_F(PlacedLowerResultUT, ResultGet_StringDataDependentyFromPlacemenEnd) {
  auto node = ComputeNodeFaker().IoNum(1, 1, ge::DT_STRING).Build();
  ASSERT_NE(node, nullptr);
  auto compute_graph = std::make_shared<ge::ComputeGraph>("tmp-graph");
  compute_graph->AddNode(node);
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  LoweringGlobalData global_data = GlobalDataFaker(root_model).Build();
  
  auto streams = global_data.LoweringAndSplitRtStreams(1);

  LowerResult lower_result = {
      HyperStatus::Success(), {ValueHolder::CreateFeed(2)}, {ValueHolder::CreateFeed(0)}, {DevMemValueHolder::CreateSingleDataOutput("Data", {}, 0)}};
  lower_result.out_addrs[0]->SetPlacement(kTensorPlacementEnd);
  auto order_holders = lower_result.order_holders;
  auto shape = lower_result.out_shapes[0];
  auto address = lower_result.out_addrs[0];

  PlacedLoweringResult plr(node, std::move(lower_result));
  auto result = plr.GetOutputResult(global_data, 0, {kOnDeviceHbm, bg::kMainStream}, false);
  EXPECT_TRUE(result->has_init);
  EXPECT_NE(result->address, address);

  ASSERT_NE(result->shape, nullptr);
  EXPECT_EQ(result->shape->GetFastNode()->GetType(), "Data");
  FastNodeTopoChecker tensor_checker(result->address);
  EXPECT_EQ(tensor_checker.StrictConnectFrom(std::vector<FastSrcNode>({
                {"SplitRtStreams", 0},
                {"SelectL2Allocator", 0},
                {"Data", 0},
                {"CalcStringTensorSize", 0}, 
                {"Data", 0},
                {"Const", 0}
            })),
            "success");
  EXPECT_EQ(tensor_checker.InChecker().DataFromByType("CalcStringTensorSize").DataFrom(address).Result(), "success");
}

TEST_F(PlacedLowerResultUT, ResultGet_StringDataHbmToHost) {
  auto node = ComputeNodeFaker().IoNum(1, 1, ge::DT_STRING).Build();
  LoweringGlobalData global_data;
  auto streams = global_data.LoweringAndSplitRtStreams(1);

  LowerResult lower_result = {
      HyperStatus::Success(), {ValueHolder::CreateFeed(2)}, {ValueHolder::CreateFeed(0)}, {DevMemValueHolder::CreateSingleDataOutput("Data", {}, 0)}};
  lower_result.out_addrs[0]->SetPlacement(kOnDeviceHbm);

  auto order_holders = lower_result.order_holders;
  auto shape = lower_result.out_shapes[0];
  auto address = lower_result.out_addrs[0];

  PlacedLoweringResult plr(node, std::move(lower_result));
  auto result = plr.GetOutputResult(global_data, 0, {kOnHost, bg::kMainStream}, false);

  EXPECT_EQ(result->shape, shape);
  EXPECT_FALSE(result->order_holders.empty());
  EXPECT_EQ(result->address->GetFastNode()->GetType(), "CopyD2H");
  FastNodeTopoChecker checker(result->address);
  EXPECT_EQ(checker.StrictConnectFrom(std::vector<FastSrcNode>({
                address,                        // address
                {"CalcStringTensorSize"},  // tensor size
                {"CreateHostL2Allocator"},
                {"SyncStream", -1}  // 来自流同步的控制边
            })),
            "success");

  auto sync_stream = ge::ExecuteGraphUtils::FindFirstNodeMatchType(result->address->GetCurrentExecuteGraph(), "SyncStream");
  ASSERT_NE(sync_stream, nullptr);
  checker = FastNodeTopoChecker(sync_stream);
  std::vector<FastSrcNode> sync_stream_inputs;
  sync_stream_inputs.emplace_back(streams[0]);
  for (const auto &ctrl_in : order_holders) {
    sync_stream_inputs.emplace_back(FastSrcNode::Name(ctrl_in->GetFastNode()->GetName(), -1));
  }
  EXPECT_EQ(checker.StrictConnectFrom(sync_stream_inputs), "success");
}

/*
 *
 * +--------------------->  result <---------+
 * |                          |              |
 * |                       CopyH2D           |
 * |                      /      |           |
 * |          CalcTensorSize     |           |
 * |              /    |         |           |
 * |  const(dtype)     shape   address   ordered_holders
 * |                    |
 * +--------------------+
 */
TEST_F(PlacedLowerResultUT, ResultGet_StringDataHostToHbm) {
  auto node = ComputeNodeFaker().IoNum(1, 1, ge::DT_STRING).Build();
  auto compute_graph = std::make_shared<ge::ComputeGraph>("tmp-graph");
  compute_graph->AddNode(node);
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  LoweringGlobalData global_data = GlobalDataFaker(root_model).Build();
  
  auto streams = global_data.LoweringAndSplitRtStreams(1);

  LowerResult lower_result = {
      HyperStatus::Success(), {ValueHolder::CreateFeed(2)}, {ValueHolder::CreateFeed(0)}, {DevMemValueHolder::CreateSingleDataOutput("Data", {}, 0)}};
  lower_result.out_addrs[0]->SetPlacement(kOnHost);

  auto order_holders = lower_result.order_holders;
  auto shape = lower_result.out_shapes[0];
  auto address = lower_result.out_addrs[0];

  PlacedLoweringResult plr(node, std::move(lower_result));
  auto result = plr.GetOutputResult(global_data, 0, {kOnDeviceHbm, bg::kMainStream}, false);
  EXPECT_EQ(result->shape, shape);

  EXPECT_EQ(result->address->GetFastNode()->GetType(), "CopyH2D");
  FastNodeTopoChecker checker(result->address);
  std::vector<FastSrcNode> dependency_node;
  dependency_node.emplace_back(streams[0]);
  dependency_node.emplace_back("SelectL2Allocator");
  dependency_node.emplace_back(address);
  dependency_node.emplace_back("CalcStringTensorSize");
  dependency_node.emplace_back(shape);
  dependency_node.emplace_back("Const");
  for (auto holder : order_holders) {
    dependency_node.emplace_back(holder->GetFastNode()->GetType(), -1);
  }
  EXPECT_EQ(checker.StrictConnectFrom(dependency_node, true), "success");
}

/*
 *
 * +--------------------->  result <---------+
 * |                          |              |
 * |                 MakeSureAtDeviceHbm     |
 * |                      /      |           |
 * |CalcTensorSizeFromStorage    |           |
 * |              /    |         |           |
 * |  const(dtype)     shape   address   ordered_holders
 * |                    |
 * +--------------------+
 */
TEST_F(PlacedLowerResultUT, ResultGet_PlacementEndToHbm) {
  auto node = StubNode();
  auto compute_graph = std::make_shared<ge::ComputeGraph>("tmp-graph");
  compute_graph->AddNode(node);
  auto root_model = GeModelBuilder(compute_graph).BuildGeRootModel();
  LoweringGlobalData global_data = GlobalDataFaker(root_model).Build();

  ge::DataType data_type = ge::DT_FLOAT;
  auto dt_holder = bg::ValueHolder::CreateConst(&data_type, sizeof(data_type));

  LowerResult lower_result = {
      HyperStatus::Success(), {ValueHolder::CreateFeed(2)}, {ValueHolder::CreateFeed(0)}, {DevMemValueHolder::CreateSingleDataOutput("Data", {}, 0)}};
  lower_result.out_addrs[0]->SetPlacement(kTensorPlacementEnd);

  auto order_holders = lower_result.order_holders;
  auto shape = lower_result.out_shapes[0];
  auto address = lower_result.out_addrs[0];

  PlacedLoweringResult plr(node, std::move(lower_result));
  auto result = plr.GetOutputResult(global_data, 0, {kOnDeviceHbm, bg::kMainStream}, false);

  EXPECT_EQ(result->address->GetFastNode()->GetType(), "MakeSureTensorAtDevice");
  FastNodeTopoChecker checker(result->address);
  EXPECT_EQ(checker.StrictConnectFrom(std::vector<FastSrcNode>({{"SplitRtStreams", 0},
                                                            {{"SelectL2Allocator"}, 0},
                                                            {"Data", 0},
                                                            {"CalcTensorSizeFromStorage", 0},
                                                            {"Data", 0},
                                                            {"Const", 0}}),
                                      true),
            "success");
}
}  // namespace gert

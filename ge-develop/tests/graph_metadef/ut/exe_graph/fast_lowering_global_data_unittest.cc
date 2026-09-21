/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/lowering/lowering_global_data.h"
#include "exe_graph/lowering/frame_selector.h"
#include <gtest/gtest.h>
#include "checker/bg_test.h"
#include "exe_graph/lowering/value_holder.h"
#include "exe_graph/runtime/execute_graph_types.h"
#include "checker/summary_checker.h"
#include "checker/topo_checker.h"
#include "exe_graph/lowering/lowering_opt.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_dump_utils.h"

namespace gert {
namespace {
ge::NodePtr BuildTestNode() {
  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  return graph->AddNode(op_desc);
}
}  // namespace
class FastLoweringGlobalDataUT : public BgTest {
 protected:
  void SetUp() override {
    BgTest::SetUp();
  }

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
  void InitTestFramesWithStream(LoweringGlobalData &global_data, int64_t stream_num = 1) {
    root_frame = bg::ValueHolder::GetCurrentFrame();
    auto init_node = bg::ValueHolder::CreateVoid<bg::ValueHolder>("Init", {});
    bg::ValueHolder::PushGraphFrame(init_node, "Init");
    global_data.LoweringAndSplitRtStreams(1);
    // prepare stream num in init
    auto init_out = bg::FrameSelector::OnInitRoot([&stream_num, &global_data]()-> std::vector<bg::ValueHolderPtr> {
      auto stream_num_holder = bg::ValueHolder::CreateConst(&stream_num, sizeof(stream_num));
      global_data.SetUniqueValueHolder(kGlobalDataModelStreamNum, stream_num_holder);
      return {};
    });
    init_frame = bg::ValueHolder::PopGraphFrame();

    auto de_init_node = bg::ValueHolder::CreateVoid<bg::ValueHolder>("DeInit", {});
    bg::ValueHolder::PushGraphFrame(de_init_node, "DeInit");
    de_init_frame = bg::ValueHolder::PopGraphFrame();

    auto main_node = bg::ValueHolder::CreateVoid<bg::ValueHolder>(GetExecuteGraphTypeStr(ExecuteGraphType::kMain), {});
    bg::ValueHolder::PushGraphFrame(main_node, "Main");
    global_data.LoweringAndSplitRtStreams(stream_num);
  }
  bg::GraphFrame *root_frame;
  std::unique_ptr<bg::GraphFrame> init_frame;
  std::unique_ptr<bg::GraphFrame> de_init_frame;
};

TEST_F(FastLoweringGlobalDataUT, SetGetCompileResultOk) {
  LoweringGlobalData gd;

  auto node = BuildTestNode();
  ASSERT_NE(node, nullptr);

  EXPECT_EQ(gd.FindCompiledResult(node), nullptr);

  gd.AddCompiledResult(node, {});
  ASSERT_NE(gd.FindCompiledResult(node), nullptr);
  EXPECT_TRUE(gd.FindCompiledResult(node)->GetTaskDefs().empty());
}

TEST_F(FastLoweringGlobalDataUT, SetGetKnownSubgraphModel) {
  LoweringGlobalData gd;

  std::string graph_name = "graph";

  EXPECT_EQ(gd.GetGraphStaticCompiledModel(graph_name), nullptr);

  gd.AddStaticCompiledGraphModel(graph_name, reinterpret_cast<void *>(0x123));
  EXPECT_EQ(gd.GetGraphStaticCompiledModel(graph_name), reinterpret_cast<void *>(0x123));
}

TEST_F(FastLoweringGlobalDataUT, GetOrCreateAllocatorOk) {
  LoweringGlobalData gd;
  InitTestFramesWithStream(gd);
  auto allocator1 = gd.GetOrCreateL1Allocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
  ASSERT_NE(allocator1, nullptr);
  EXPECT_EQ(allocator1, gd.GetOrCreateL1Allocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput}));
}

TEST_F(FastLoweringGlobalDataUT, GetOrCreateL1Allocator_InitRootCreateSync1) {
  InitTestFrames();
  LoweringGlobalData gd;
  auto holder = gd.GetOrCreateL1Allocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
  ASSERT_NE(holder, nullptr);

  ASSERT_EQ(gd.GetL1Allocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput}), holder);

  std::vector<bg::ValueHolderPtr> on_init;
  std::vector<bg::ValueHolderPtr> on_root;
  auto ret = bg::FrameSelector::OnInitRoot(
      [&]() -> std::vector<bg::ValueHolderPtr> {
        auto allocator = gd.GetL1Allocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
        return {allocator};
      },
      on_init, on_root);
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
  ASSERT_EQ(on_init.size(), 1U);
  ASSERT_EQ(on_root.size(), 1U);
  ASSERT_NE(on_init[0], nullptr);
  ASSERT_NE(on_root[0], nullptr);
}
TEST_F(FastLoweringGlobalDataUT, GetOrCreateL1Allocator_InitRootCreateSync2) {
  InitTestFrames();
  LoweringGlobalData gd;
  std::vector<bg::ValueHolderPtr> on_init;
  std::vector<bg::ValueHolderPtr> on_root;
  auto ret = bg::FrameSelector::OnInitRoot(
      [&]() -> std::vector<bg::ValueHolderPtr> {
        return {gd.GetOrCreateL1Allocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput})};
      },
      on_init, on_root);
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
  ASSERT_NE(on_init[0], nullptr);

  ASSERT_NE(gd.GetL1Allocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput}), nullptr);
}
TEST_F(FastLoweringGlobalDataUT, GetOrCreateL1Allocator_CreateSelectAllocator_MainExternalAllocatorSet) {
  LoweringGlobalData gd;
  InitTestFramesWithStream(gd);
  gd.SetExternalAllocator(bg::ValueHolder::CreateFeed(-2));

  auto allocator1 = gd.GetOrCreateL1Allocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
  ASSERT_NE(allocator1, nullptr);
  EXPECT_EQ(allocator1->GetFastNode()->GetType(), "SelectL1Allocator");
  EXPECT_EQ(FastNodeTopoChecker(allocator1).StrictConnectFrom(
              {{"InnerData"}, {"Data"}, {"InnerData"}, {"SplitRtStreams"}}),
            "success");
  auto create_allocator_node =
    ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_frame->GetExecuteGraph().get(), "CreateL1Allocator");
  ASSERT_NE(create_allocator_node, nullptr);
  ConnectFromInitToMain(create_allocator_node, 0, allocator1->GetFastNode(), 2);

  bg::ValueHolderPtr init_allocator = nullptr;
  bg::FrameSelector::OnInitRoot([&]() -> std::vector<bg::ValueHolderPtr> {
    init_allocator = gd.GetOrCreateL1Allocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
    return {};
  });
  ASSERT_NE(init_allocator, nullptr);
  EXPECT_EQ(init_allocator->GetFastNode()->GetType(), "CreateL1Allocator");
  EXPECT_EQ(FastNodeTopoChecker(init_allocator).StrictConnectFrom({{"Const"}}), "success");
}

TEST_F(FastLoweringGlobalDataUT, GetOrCreateL1Allocator_MainExternalAllocatorSet_P2pNotUseExternal) {
  LoweringGlobalData gd;
  InitTestFramesWithStream(gd);
  gd.SetExternalAllocator(bg::ValueHolder::CreateFeed(-2));

  auto allocator1 = gd.GetOrCreateL1Allocator({kOnDeviceP2p, AllocatorUsage::kAllocNodeOutput});
  ASSERT_NE(allocator1, nullptr);
  EXPECT_EQ(allocator1->GetFastNode()->GetType(), "Init");

  bg::ValueHolderPtr init_allocator = nullptr;
  bg::FrameSelector::OnInitRoot([&]() -> std::vector<bg::ValueHolderPtr> {
    init_allocator = gd.GetOrCreateL1Allocator({kOnDeviceP2p, AllocatorUsage::kAllocNodeOutput});
    return {};
  });
  ASSERT_NE(init_allocator, nullptr);
  EXPECT_EQ(init_allocator->GetFastNode()->GetType(), "CreateL1Allocator");
  EXPECT_EQ(FastNodeTopoChecker(init_allocator).StrictConnectFrom({{"Const"}}), "success");
}

TEST_F(FastLoweringGlobalDataUT, GetOrCreateL1Allocator_CreateSelectAllocator_ExternalAllocatorSet) {
  LoweringGlobalData gd;
  InitTestFramesWithStream(gd);
  gd.SetExternalAllocator(bg::ValueHolder::CreateFeed(-2));
  bg::FrameSelector::OnInitRoot([&]() -> std::vector<bg::ValueHolderPtr> {
    gd.SetExternalAllocator(bg::ValueHolder::CreateFeed(-2), ExecuteGraphType::kInit);
    return {};
  });

  auto allocator1 = gd.GetOrCreateL1Allocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
  ASSERT_NE(allocator1, nullptr);
  EXPECT_EQ(allocator1->GetFastNode()->GetType(), "SelectL1Allocator");
  EXPECT_EQ(FastNodeTopoChecker(allocator1).StrictConnectFrom(
              {{"InnerData"}, {"Data"}, {"InnerData"}, {"SplitRtStreams"}}),
            "success");
  auto create_allocator_node =
    ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_frame->GetExecuteGraph().get(), "CreateL1Allocator");
  ASSERT_NE(create_allocator_node, nullptr);
  ConnectFromInitToMain(create_allocator_node, 0, allocator1->GetFastNode(), 2);

  bg::ValueHolderPtr init_allocator = nullptr;
  bg::FrameSelector::OnInitRoot([&]() -> std::vector<bg::ValueHolderPtr> {
    init_allocator = gd.GetOrCreateL1Allocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
    return {};
  });
  ASSERT_NE(init_allocator, nullptr);
  EXPECT_EQ(init_allocator->GetFastNode()->GetType(), "SelectL1Allocator");
  EXPECT_EQ(FastNodeTopoChecker(init_allocator).StrictConnectFrom(
              {{"Const"}, {"Data"}, {"CreateL1Allocator"}, {"SplitRtStreams"}}),
            "success");
}

TEST_F(FastLoweringGlobalDataUT, GetOrCreateL1Allocator_ExternalAllocatorSet_P2pNotUseExternal) {
  LoweringGlobalData gd;
  InitTestFramesWithStream(gd);
  gd.SetExternalAllocator(bg::ValueHolder::CreateFeed(-2));
  bg::FrameSelector::OnInitRoot([&]() -> std::vector<bg::ValueHolderPtr> {
    gd.SetExternalAllocator(bg::ValueHolder::CreateFeed(-2), ExecuteGraphType::kInit);
    return {};
  });

  auto allocator1 = gd.GetOrCreateL1Allocator({kOnDeviceP2p, AllocatorUsage::kAllocNodeOutput});
  ASSERT_NE(allocator1, nullptr);
  EXPECT_EQ(allocator1->GetFastNode()->GetType(), "Init");

  bg::ValueHolderPtr init_allocator = nullptr;
  bg::FrameSelector::OnInitRoot([&]() -> std::vector<bg::ValueHolderPtr> {
    init_allocator = gd.GetOrCreateL1Allocator({kOnDeviceP2p, AllocatorUsage::kAllocNodeOutput});
    return {};
  });
  ASSERT_NE(init_allocator, nullptr);
  EXPECT_EQ(init_allocator->GetFastNode()->GetType(), "CreateL1Allocator");
  EXPECT_EQ(FastNodeTopoChecker(init_allocator).StrictConnectFrom(
      {{"Const"}}), "success");
}

TEST_F(FastLoweringGlobalDataUT, GetOrCreateL1Allocator_ExternalAllocatorSet_UseAlwaysExternalAllocatorOption) {
  LoweringGlobalData gd;
  InitTestFramesWithStream(gd);
  gd.SetExternalAllocator(bg::ValueHolder::CreateFeed(-2));
  LoweringOption opt;
  opt.always_external_allocator = true;
  gd.SetLoweringOption(opt);
  bg::FrameSelector::OnInitRoot([&]() -> std::vector<bg::ValueHolderPtr> {
    gd.SetExternalAllocator(bg::ValueHolder::CreateFeed(-2), ExecuteGraphType::kInit);
    return {};
  });

  auto allocator1 = gd.GetOrCreateL1Allocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
  ASSERT_NE(allocator1, nullptr);
  EXPECT_EQ(allocator1->GetFastNode()->GetType(), "SelectL1Allocator");

  auto create_allocator_node =
    ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_frame->GetExecuteGraph().get(), "CreateL1Allocator");
  // 外置allocator后，图中就不存在CreateAllocator节点了
  ASSERT_EQ(create_allocator_node, nullptr);

  auto get_allocator_node =
    ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_frame->GetExecuteGraph().get(), "GetExternalL1Allocator");
  // 外置allocator后，init
  ASSERT_NE(get_allocator_node, nullptr);

  bg::ValueHolderPtr init_allocator = nullptr;
  bg::FrameSelector::OnInitRoot([&]() -> std::vector<bg::ValueHolderPtr> {
    init_allocator = gd.GetOrCreateL1Allocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
    return {};
  });
  ASSERT_NE(init_allocator, nullptr);
  EXPECT_EQ(init_allocator->GetFastNode()->GetType(), "GetExternalL1Allocator");
  EXPECT_EQ(FastNodeTopoChecker(init_allocator).StrictConnectFrom(
            {{"Const"}, {"Data"}}),
            "success");
}

TEST_F(FastLoweringGlobalDataUT, GetOrCreateL1Allocator_ExternalAllocatorSet_UseAlwaysExternalAllocatorOption_P2pNotUseExternal) {
  LoweringGlobalData gd;
  InitTestFramesWithStream(gd);
  gd.SetExternalAllocator(bg::ValueHolder::CreateFeed(-2));
  LoweringOption opt;
  opt.always_external_allocator = true;
  gd.SetLoweringOption(opt);
  bg::FrameSelector::OnInitRoot([&]() -> std::vector<bg::ValueHolderPtr> {
    gd.SetExternalAllocator(bg::ValueHolder::CreateFeed(-2), ExecuteGraphType::kInit);
    return {};
  });

  auto allocator1 = gd.GetOrCreateL1Allocator({kOnDeviceP2p, AllocatorUsage::kAllocNodeOutput});
  ASSERT_NE(allocator1, nullptr);
  EXPECT_EQ(allocator1->GetFastNode()->GetType(), "Init");

  auto create_allocator_node =
    ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_frame->GetExecuteGraph().get(), "CreateL1Allocator");
  ASSERT_NE(create_allocator_node, nullptr);

  bg::ValueHolderPtr init_allocator = nullptr;
  bg::FrameSelector::OnInitRoot([&]() -> std::vector<bg::ValueHolderPtr> {
    init_allocator = gd.GetOrCreateL1Allocator({kOnDeviceP2p, AllocatorUsage::kAllocNodeOutput});
    return {};
  });
  ASSERT_NE(init_allocator, nullptr);
  EXPECT_EQ(init_allocator->GetFastNode()->GetType(), "CreateL1Allocator");
  EXPECT_EQ(FastNodeTopoChecker(init_allocator).StrictConnectFrom(
      {{"Const"}}), "success");
}

TEST_F(FastLoweringGlobalDataUT, GetOrCreateL1Allocator_AlwaysReturnOnRootFrame_CallInSubgraph) {
  InitTestFrames();
  LoweringGlobalData gd;

  auto data0 = bg::ValueHolder::CreateFeed(0);
  auto foo1 = bg::ValueHolder::CreateSingleDataOutput("Foo", {data0});

  bg::ValueHolder::PushGraphFrame(foo1, "FooGraph");
  auto allocator1 = gd.GetOrCreateL1Allocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
  ASSERT_NE(allocator1, nullptr);
  ASSERT_NE(bg::ValueHolder::PopGraphFrame(), nullptr);

  ASSERT_EQ(allocator1->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr(), root_frame->GetExecuteGraph().get());
}

TEST_F(FastLoweringGlobalDataUT, GetOrCreateL1Allocator_AlwaysCreateOnInitFrame_CallInSubgraph) {
  InitTestFrames();
  LoweringGlobalData gd;

  auto data0 = bg::ValueHolder::CreateFeed(0);
  auto foo1 = bg::ValueHolder::CreateSingleDataOutput("Foo", {data0});

  bg::ValueHolder::PushGraphFrame(foo1, "FooGraph");
  auto allocator1 = gd.GetOrCreateL1Allocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
  ASSERT_NE(allocator1, nullptr);

  ASSERT_NE(bg::ValueHolder::PopGraphFrame(), nullptr);

  ASSERT_EQ(ExeGraphSummaryChecker(init_frame->GetExecuteGraph().get())
                .StrictAllNodeTypes({{"CreateL1Allocator", 1}, {"Const", 1}, {"InnerNetOutput", 1}}),
            "success");
}
TEST_F(FastLoweringGlobalDataUT, GetOrCreateL1Allocator_ReturnOnInit_WhenGetOnInit) {
  InitTestFrames();
  LoweringGlobalData gd;

  auto data0 = bg::ValueHolder::CreateFeed(0);
  auto foo1 = bg::ValueHolder::CreateSingleDataOutput("Foo", {data0});

  bg::ValueHolder::PushGraphFrame(foo1, "FooGraph");
  auto allocator1 = gd.GetOrCreateL1Allocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
  ASSERT_NE(allocator1, nullptr);
  ASSERT_EQ(allocator1->GetFastNode()->GetType(), "Init");

  std::vector<bg::ValueHolderPtr> graph_out;
  std::vector<bg::ValueHolderPtr> node_out;
  auto ret = bg::FrameSelector::OnInitRoot(
      [&]() -> std::vector<bg::ValueHolderPtr> {
        return {gd.GetOrCreateL1Allocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput})};
      },
      graph_out, node_out);
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
  ASSERT_EQ(graph_out.size(), 1);
  auto init_node = graph_out[0]->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr()->GetParentNodeBarePtr();
  ASSERT_NE(init_node, nullptr);
  ASSERT_EQ(init_node->GetType(), "Init");
}
TEST_F(FastLoweringGlobalDataUT, GetOrCreateUniqueValueHolderOk) {
  LoweringGlobalData gd;
  auto builder = [&]() -> bg::ValueHolderPtr {
    auto resource_holder = bg::FrameSelector::OnMainRoot([&]() -> std::vector<bg::ValueHolderPtr> {
      std::string name = "aicpu_resource";
      auto name_holder = bg::ValueHolder::CreateConst(name.c_str(), name.size(), true);
      auto create_container_holder = bg::ValueHolder::CreateSingleDataOutput("CreateStepContainer", {name_holder});
      bg::ValueHolder::CreateVoidGuarder("DestroyStepContainer", create_container_holder, {});
      return {create_container_holder};
    });
    return resource_holder[0];
  };
  auto holder_0 = gd.GetOrCreateUniqueValueHolder("aicpu_container_0", builder);
  EXPECT_NE(holder_0, nullptr);

  auto clear_builder = [&]() -> bg::ValueHolderPtr {
    return bg::ValueHolder::CreateVoid<bg::ValueHolder>("ClearStepContainer", {holder_0});
  };
  auto clear_holder = bg::FrameSelector::OnMainRootLast(clear_builder);
  EXPECT_NE(clear_holder, nullptr);
  std::string create_resource_name = holder_0->GetFastNode()->GetOpDescBarePtr()->GetName();
  EXPECT_EQ(create_resource_name.find("CreateStepContainer"), 0);

  auto last_exec_nodes = bg::ValueHolder::GetLastExecNodes();
  EXPECT_EQ(last_exec_nodes.size(), 1);
  EXPECT_NE(last_exec_nodes[0], nullptr);
  std::string clear_resource_name = last_exec_nodes[0]->GetFastNode()->GetOpDescBarePtr()->GetName();
  EXPECT_EQ(clear_resource_name.find("ClearStepContainer"), 0);

  // use same key: aicpu_container_0, check unique
  auto holder_1 = gd.GetOrCreateUniqueValueHolder("aicpu_container_0", builder);
  EXPECT_EQ(last_exec_nodes.size(), 1);
  last_exec_nodes.clear();
}

TEST_F(FastLoweringGlobalDataUT, OnMainRootLastOk) {
  LoweringGlobalData gd;
  uint64_t global_container_id = 0;
  auto builder = [&]() -> bg::ValueHolderPtr {
    uint64_t container_id = global_container_id++;
    auto container_id_holder = bg::ValueHolder::CreateConst(&container_id, sizeof(uint64_t));
    uint64_t session_id = 0;
    auto session_id_holder = bg::ValueHolder::CreateConst(&session_id, sizeof(uint64_t));
    auto resource_holder = bg::FrameSelector::OnMainRoot([&]() -> std::vector<bg::ValueHolderPtr> {
      auto create_session_holder = bg::ValueHolder::CreateSingleDataOutput("CreateSession", {session_id_holder});
      bg::ValueHolder::CreateVoidGuarder("DestroySession", create_session_holder, {});
      auto clear_builder = [&]() -> bg::ValueHolderPtr {
        return bg::ValueHolder::CreateVoid<bg::ValueHolder>("ClearStepContainer", {session_id_holder, container_id_holder});
      };
      auto clear_holder = bg::FrameSelector::OnMainRootLast(clear_builder);
      EXPECT_NE(clear_holder, nullptr);
      return {container_id_holder};
    });
    return resource_holder[0];
  };
  auto holder_0 = gd.GetOrCreateUniqueValueHolder("aicpu_container_0", builder);
  EXPECT_NE(holder_0, nullptr);

  auto last_exec_nodes = bg::ValueHolder::GetLastExecNodes();
  EXPECT_EQ(last_exec_nodes.size(), 1);
  EXPECT_NE(last_exec_nodes[0], nullptr);
  std::string clear_resource_name = last_exec_nodes[0]->GetFastNode()->GetOpDescBarePtr()->GetName();
  EXPECT_EQ(clear_resource_name.find("ClearStepContainer"), 0);

  // use same key: aicpu_container_0, check unique
  auto holder_1 = gd.GetOrCreateUniqueValueHolder("aicpu_container_0", builder);
  EXPECT_EQ(last_exec_nodes.size(), 1);
  last_exec_nodes.clear();
}

TEST_F(FastLoweringGlobalDataUT, SinkWeightInfoTest) {
  LoweringGlobalData gd;
  size_t weight_info = 1;
  gd.SetModelWeightSize(weight_info);
  auto result = gd.GetModelWeightSize();
  EXPECT_EQ(result, weight_info);
}

TEST_F(FastLoweringGlobalDataUT, GetValueHolersSizeTest) {
  LoweringGlobalData gd;
  gd.SetValueHolders("test1", nullptr);
  EXPECT_EQ(gd.GetValueHoldersSize("test1"), 1);
  EXPECT_EQ(gd.GetValueHoldersSize("test2"), 0);
  gd.SetValueHolders("test1", nullptr);
  EXPECT_EQ(gd.GetValueHoldersSize("test1"), 2);

  gd.SetUniqueValueHolder("test3", nullptr);
  EXPECT_EQ(gd.GetValueHoldersSize("test3"), 1);
  gd.SetUniqueValueHolder("test3", nullptr);
  EXPECT_EQ(gd.GetValueHoldersSize("test3"), 1);
}

TEST_F(FastLoweringGlobalDataUT, SetGetUniqueValueHoler) {
  LoweringGlobalData gd;
  gd.SetUniqueValueHolder("test1", nullptr);
  EXPECT_EQ(gd.GetValueHoldersSize("test1"), 1);
  EXPECT_EQ(gd.GetValueHoldersSize("test2"), 0);
  EXPECT_EQ(gd.GetUniqueValueHolder("test1"), nullptr);

  gd.SetUniqueValueHolder("test1", bg::ValueHolder::CreateVoid<bg::ValueHolder>("TEST", {}));
  EXPECT_EQ(gd.GetValueHoldersSize("test1"), 1);
  EXPECT_NE(gd.GetUniqueValueHolder("test1"), nullptr);
}

TEST_F(FastLoweringGlobalDataUT, StaticModelWsSizeTest) {
  LoweringGlobalData gd;
  int64_t require_ws_size = 1;
  gd.SetStaicModelWsSize(require_ws_size);
  auto result = gd.GetStaticModelWsSize();
  EXPECT_EQ(result, require_ws_size);
}

TEST_F(FastLoweringGlobalDataUT, FixedFeatureMemoryBaseTest) {
  LoweringGlobalData gd;
  gd.SetFixedFeatureMemoryBase((void *)0x355, 4);
  auto fixed_feature_mem = gd.GetFixedFeatureMemoryBase();
  EXPECT_EQ((size_t)fixed_feature_mem.first, 0x355);
  EXPECT_EQ(fixed_feature_mem.second, 4);
}

/*
 * init_graph:
 *                                                +--------------------------+
 *                                                |          |               |
 *  Const(placement)  Const(stream_num)    Const(placement)  Const(usage)    |
 *          \          /                            \        /               |
 *             CreateL2Allocator                   CreateL1Allocator           |
 *                             \                     /                       |
 *                              \                   /                        |
 *                                 InnerNetOutput <--------------------------+
 *
 *
 *
 * main_graph:
 * *                                         InnerData(l1 allocators)
 *             Data(rt_streams)
 *                     |                      /
 *                 SplitRtStreams             /    InnerData(l2 allocators)
 *                              \           /     /
 *                              SelectL2Allocator
 *
 *  测试无外置allocator场景下的L2 allocator
 */
TEST_F(FastLoweringGlobalDataUT, GetOrCreateL2AllocatorInMain_Device_WithoutExternalAllocator) {
  LoweringGlobalData global_data;
  InitTestFramesWithStream(global_data, 3);

  auto l2_allocator =
      global_data.GetOrCreateL2Allocator(0, {TensorPlacement::kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
  EXPECT_NE(l2_allocator, nullptr);

  auto get_l2_allocator =
    global_data.GetMainL2Allocator(0, {TensorPlacement::kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
  EXPECT_NE(get_l2_allocator, nullptr);
  EXPECT_EQ(l2_allocator->GetFastNode(), get_l2_allocator->GetFastNode());

  auto init_exe_graph = init_frame->GetExecuteGraph().get();
  EXPECT_EQ(ExeGraphSummaryChecker(init_exe_graph)
                .StrictDirectNodeTypes(std::map<std::string, size_t>{{"Data", 1},
                                                                     {"SplitRtStreams", 1},
                                                                     {"Const",4},
                                                                     {"CreateL1Allocator", 1},
                                                                     {"CreateL2Allocators", 1},
                                                                     {"InnerNetOutput", 1}}),
            "success");

  auto main_frame = bg::ValueHolder::PopGraphFrame();
  auto main_exe_graph = main_frame->GetExecuteGraph().get();
  ge::DumpGraph(main_exe_graph->GetParentGraphBarePtr(), "TestL2Allocator");
  EXPECT_EQ(ExeGraphSummaryChecker(main_exe_graph)
                .StrictDirectNodeTypes(std::map<std::string, size_t>{
                    {"Const", 2}, {"Data", 1}, {"InnerData", 2}, {"SplitRtStreams", 1}, {"SelectL2Allocator", 1}}),
            "success");
  FastNodeTopoChecker checker(l2_allocator);
  // Const(logic_stream_id), GetStreamById, InnerData(L1 allocator), InnerData(L2 allocators)
  EXPECT_EQ(checker.StrictConnectFrom(std::vector<SrcFastNode>({{"Const"}, {"SplitRtStreams"}, {"InnerData"}, {"InnerData"}})), "success");
}

TEST_F(FastLoweringGlobalDataUT, GetOrCreateL2AllocatorInMain_FollowingPlacement_GetHostAllocator) {
  LoweringGlobalData global_data;
  InitTestFramesWithStream(global_data, 3);

  auto l2_allocator =
      global_data.GetOrCreateL2Allocator(0, {TensorPlacement::kFollowing, AllocatorUsage::kAllocNodeOutput});
  EXPECT_NE(l2_allocator, nullptr);

  auto get_l2_allocator =
    global_data.GetMainL2Allocator(0, {TensorPlacement::kFollowing, AllocatorUsage::kAllocNodeOutput});
  EXPECT_NE(get_l2_allocator, nullptr);
  EXPECT_EQ(l2_allocator->GetFastNode(), get_l2_allocator->GetFastNode());
  auto get_l2_allocator_init =
    global_data.GetInitL2Allocator({TensorPlacement::kFollowing, AllocatorUsage::kAllocNodeOutput});
  EXPECT_NE(get_l2_allocator_init, nullptr);

  auto init_out = bg::FrameSelector::OnInitRoot([&]()-> std::vector<bg::ValueHolderPtr> {
    return {global_data.GetOrCreateL2Allocator(0, {TensorPlacement::kFollowing, AllocatorUsage::kAllocNodeOutput})};
  });
  EXPECT_EQ(init_out.size(), 1);
  EXPECT_NE(init_out[0], nullptr);
  auto host_allocator_init = bg::HolderOnInit(init_out[0]);
  EXPECT_EQ(host_allocator_init->GetFastNode(), get_l2_allocator_init->GetFastNode());
  EXPECT_EQ(host_allocator_init->GetFastNode()->GetType(), "CreateHostL2Allocator");
}

TEST_F(FastLoweringGlobalDataUT, GetOrCreateL2AllocatorOnInit_UnsupportedPlacement_Failed) {
  LoweringGlobalData global_data;
  InitTestFramesWithStream(global_data, 3);

  auto init_out = bg::FrameSelector::OnInitRoot([&]()-> std::vector<bg::ValueHolderPtr> {
    return {global_data.GetOrCreateL2Allocator(0, {TensorPlacement::kTensorPlacementEnd, AllocatorUsage::kAllocNodeOutput})};
  });
  EXPECT_EQ(init_out.size(), 0);

  auto get_l2_allocator_init =
    global_data.GetInitL2Allocator({TensorPlacement::kTensorPlacementEnd, AllocatorUsage::kAllocNodeOutput});
  EXPECT_EQ(get_l2_allocator_init, nullptr);
}
/*
 * init_graph:
 *
 *
 *                                      Const(placement)
 *                                        /           |
 *                                        |   CreateL1Allocator
 *                                        |          /
 *                                         \        /
 *                                       InnerNetOutput
 *
 *
 *
 * main_graph:
 * *                                    InnerData(l1 allocators)
 *                                             |
 *                                      CreateHostL2Allocator
 *
 *
 *  测试无外置allocator场景下的host L2 allocator
 */
TEST_F(FastLoweringGlobalDataUT, GetOrCreateL2AllocatorInMain_Host_WithoutExternalAllocator) {
  LoweringGlobalData global_data;
  InitTestFramesWithStream(global_data, 3);

  auto l2_allocator =
      global_data.GetOrCreateL2Allocator(0, {TensorPlacement::kOnHost, AllocatorUsage::kAllocNodeOutput});
  EXPECT_NE(l2_allocator, nullptr);

  auto get_l2_allocator =
    global_data.GetMainL2Allocator(0, {TensorPlacement::kOnHost, AllocatorUsage::kAllocNodeOutput});
  EXPECT_NE(get_l2_allocator, nullptr);
  EXPECT_EQ(l2_allocator->GetFastNode(), get_l2_allocator->GetFastNode());
  auto get_l2_allocator_init =
    global_data.GetInitL2Allocator({TensorPlacement::kOnHost, AllocatorUsage::kAllocNodeOutput});
  EXPECT_NE(get_l2_allocator_init, nullptr);

  auto main_frame = bg::ValueHolder::PopGraphFrame();
  auto init_exe_graph = init_frame->GetExecuteGraph().get();
  EXPECT_EQ(ExeGraphSummaryChecker(init_exe_graph)
                .StrictDirectNodeTypes(std::map<std::string, size_t>{{"Data", 1},
                                                                     {"SplitRtStreams", 1},
                                                                     {"Const", 3},
                                                                     {"CreateL1Allocator", 1},
                                                                     {"InnerNetOutput", 1},
                                                                     {"CreateHostL2Allocator", 1}}),
            "success");

  auto main_exe_graph = main_frame->GetExecuteGraph().get();
  ge::DumpGraph(main_exe_graph->GetParentGraphBarePtr(), "TestHostL2Allocator");
  EXPECT_EQ(ExeGraphSummaryChecker(main_exe_graph)
                .StrictDirectNodeTypes(std::map<std::string, size_t>{{"Const", 1}, // stream num
                                                                     {"Data", 1}, // stream
                                                                     {"SplitRtStreams", 1},
                                                                     {"InnerData", 1},
                                                                     {"CreateHostL2Allocator", 1}}),
            "success");
  FastNodeTopoChecker checker(l2_allocator);
  // InnerData(L1 allocator)
  EXPECT_EQ(checker.StrictConnectFrom(std::vector<SrcFastNode>({{"InnerData"}})), "success");
}
/*
 * init_graph:
 *                                               +--------------------------+
 *                                                |          |               |
 *  Const(placement)  Const(stream_num)    Const(placement)  Const(usage)    |
 *          \          /                            \        /               |       Data(exteranl_allocator)
 *             CreateL2Allocator                   CreateL1Allocator ----------+-----+      |         Data(external_stream)
 *                             \                     /                       |      \     |       /
 *                              \                   /                        |    SelectAllocator
 *                                  InnerNetOutput <-------------------------+
 *
 *
 * main_graph:
 *                                         SelectL1Allocator
 *              Data(rt_streams)               /
 *                     |                      /
 *                 GetStreamById             /    InnerData(l2 allocators)
 *                              \           /     /
 *                              SelectL2Allocator
 *
 *   测试有外置allocator场景下的L2 allocator
 */
TEST_F(FastLoweringGlobalDataUT, GetOrCreateAllocatorInMain_Device_WithExternalAllocator) {
  LoweringGlobalData global_data;
  InitTestFramesWithStream(global_data, 3);

  // prepare external allocator on init and main
  bg::FrameSelector::OnInitRoot([&]()-> std::vector<bg::ValueHolderPtr> {
    auto external_allocator_init = bg::ValueHolder::CreateFeed(-2);
    global_data.SetExternalAllocator(static_cast<bg::ValueHolderPtr &&>(external_allocator_init), ExecuteGraphType::kInit);
    return {};
  });
  auto external_allocator = bg::ValueHolder::CreateFeed(-2);
  global_data.SetExternalAllocator(static_cast<bg::ValueHolderPtr &&>(external_allocator), ExecuteGraphType::kMain);

  auto l2_allocator =
      global_data.GetOrCreateL2Allocator(0, {TensorPlacement::kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
  EXPECT_NE(l2_allocator, nullptr);

  auto get_l2_allocator =
  global_data.GetMainL2Allocator(0, {TensorPlacement::kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
  EXPECT_NE(get_l2_allocator, nullptr);
  EXPECT_EQ(l2_allocator->GetFastNode(), get_l2_allocator->GetFastNode());

  auto main_frame = bg::ValueHolder::PopGraphFrame();
  auto init_exe_graph = init_frame->GetExecuteGraph().get();
  ge::DumpGraph(init_exe_graph, "l2_allocator_init");
  EXPECT_EQ(ExeGraphSummaryChecker(init_exe_graph)
                .StrictDirectNodeTypes(std::map<std::string, size_t>{{"Const", 4},
                                                                     {"Data", 2},
                                                                     {"SplitRtStreams", 1},
                                                                     {"CreateL1Allocator", 1},
                                                                     {"CreateL2Allocators", 1},
                                                                     {"SelectL1Allocator", 1},
                                                                     {"InnerNetOutput", 1}}),
            "success");

  auto main_exe_graph = main_frame->GetExecuteGraph().get();
  // ge::DumpGraph(main_exe_graph->GetParentGraphBarePtr(), "TestDeviceL2AllocatorWithExternalL1Allocator");
  EXPECT_EQ(ExeGraphSummaryChecker(main_exe_graph)
                .StrictDirectNodeTypes(std::map<std::string, size_t>{{"Const", 2},
                                                                     {"Data", 2},
                                                                     {"InnerData", 3},
                                                                     {"SplitRtStreams", 1},
                                                                     {"SelectL1Allocator", 1},
                                                                     {"SelectL2Allocator", 1}}),
            "success");
  FastNodeTopoChecker checker(l2_allocator);
  // Const(logic_stream_id), GetStreamById, SelectL1Allocator(L1 allocator), InnerData(L2 allocators)
  EXPECT_EQ(checker.StrictConnectFrom(std::vector<SrcFastNode>({{"Const"}, {"SplitRtStreams"}, {"SelectL1Allocator"}, {"InnerData"}})), "success");
}

/*
 * init_graph:
 *                                               +--------------------------+
 *                                                |          |               |
 *  Const(placement)  Const(stream_num)    Const(placement)  Const(usage)    |
 *          \          /                            \        /               |       Data(exteranl_allocator)
 *             CreateL2Allocator                   CreateL1Allocator ----------+-----+      |         Data(external_stream)
 *                             \                     /                       |      \     |       /
 *                              \                   /                        |    SelectAllocator
 *                                  InnerNetOutput <-------------------------+
 *
 *
 * main_graph:
 *                                         SelectL1Allocator
 *   Const(stream_id)   Data(rt_streams)       /
 *                 \      /                   /
 *                 GetStreamById             /    InnerData(l2 allocators)
 *                              \           /     /
 *                              SelectL2Allocator
 *                                  /    \
 *                          consumer00   consumer01
 *   测试多次调用有外置allocator场景下的L2 allocator
 *   预期结果：同一条流上多次调用 select l2 allocator只生成1个kerenel
 */
TEST_F(FastLoweringGlobalDataUT, GetOrCreateAllocatorInMain_WithExternalAllocator_CallMultiTimes) {
  LoweringGlobalData global_data;
  InitTestFramesWithStream(global_data, 3);

  // prepare external allocator on init and main
  bg::FrameSelector::OnInitRoot([&]()-> std::vector<bg::ValueHolderPtr> {
    auto external_allocator_init = bg::ValueHolder::CreateFeed(-2);
    global_data.SetExternalAllocator(static_cast<bg::ValueHolderPtr &&>(external_allocator_init), ExecuteGraphType::kInit);
    return {};
  });
  auto external_allocator = bg::ValueHolder::CreateFeed(-2);
  global_data.SetExternalAllocator(static_cast<bg::ValueHolderPtr &&>(external_allocator), ExecuteGraphType::kMain);

  // prepare rtStreams

  auto l2_allocator_00 =
      global_data.GetOrCreateL2Allocator(0, {TensorPlacement::kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
  EXPECT_NE(l2_allocator_00, nullptr);
  auto consumer00 = bg::ValueHolder::CreateVoid<bg::ValueHolder>("consumer00", {l2_allocator_00});

  auto l2_allocator_01 =
      global_data.GetOrCreateL2Allocator(0, {TensorPlacement::kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
  EXPECT_NE(l2_allocator_01, nullptr);
  EXPECT_EQ(l2_allocator_00, l2_allocator_01);
  auto consumer01 = bg::ValueHolder::CreateVoid<bg::ValueHolder>("consumer01", {l2_allocator_01});

  auto l2_allocator_10 =
      global_data.GetOrCreateL2Allocator(1, {TensorPlacement::kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
  EXPECT_NE(l2_allocator_01, nullptr);
  auto consumer10 = bg::ValueHolder::CreateVoid<bg::ValueHolder>("consumer10", {l2_allocator_10});

  auto main_frame = bg::ValueHolder::PopGraphFrame();
  auto init_exe_graph = init_frame->GetExecuteGraph().get();
  // ge::DumpGraph(init_exe_graph, "l2_allocator_init");
  EXPECT_EQ(ExeGraphSummaryChecker(init_exe_graph)
                .StrictDirectNodeTypes(std::map<std::string, size_t>{{"Const", 4},
                                                                     {"Data", 2},
                                                                     {"SplitRtStreams", 1},
                                                                     {"CreateL1Allocator", 1},
                                                                     {"CreateL2Allocators", 1},
                                                                     {"SelectL1Allocator", 1},
                                                                     {"InnerNetOutput", 1}}),
            "success");

  auto main_exe_graph = main_frame->GetExecuteGraph().get();
  // ge::DumpGraph(main_exe_graph, "l2_allocator_main");
  EXPECT_EQ(ExeGraphSummaryChecker(main_exe_graph)
                .StrictDirectNodeTypes(std::map<std::string, size_t>{{"Const", 3},
                                                                     {"Data", 2},
                                                                     {"InnerData", 3},
                                                                     {"SplitRtStreams", 1},
                                                                     {"SelectL1Allocator", 1},
                                                                     {"SelectL2Allocator", 2},
                                                                     {"consumer00", 1},
                                                                     {"consumer01", 1},
                                                                     {"consumer10", 1}}),
            "success");
  FastNodeTopoChecker checker(l2_allocator_00);
  // Const(logic_stream_id), GetStreamById, SelectL1Allocator(L1 allocator), InnerData(L2 allocators)
  EXPECT_EQ(checker.StrictConnectFrom(std::vector<SrcFastNode>({{"Const"}, {"SplitRtStreams"}, {"SelectL1Allocator"}, {"InnerData"}})), "success");
  EXPECT_EQ(checker.StrictConnectTo(0, std::vector<SrcFastNode>({{"consumer00"},{"consumer01"}})), "success");
}

/*
 * init_graph:
 *                                               +--------------------------+
 *                                                |          |               |
 *  Const(placement)  Const(stream_num)    Const(placement)  Const(usage)    |
 *          \          /                            \        /               |       Data(exteranl_allocator)
 *             CreateL2Allocator                   CreateL1Allocator ----------+-----+      |         Data(external_stream)
 *                             \                     /                       |      \     |       /
 *                              \                   /                        |    SelectAllocator
 *                                  InnerNetOutput <-------------------------+
 *
 *
 * main_graph:
 *                                         SelectL1Allocator
 *   Const(stream_id)   Data(rt_streams)       /
 *                 \      /                   /
 *                 GetStreamById             /    InnerData(l2 allocators)
 *                              \           /     /                        .........
 *                              SelectL2Allocator                         CreateHostL2Allocator
 *                                   /      \                                      /   \
 *                             consumer00  consumer02                     consumer01   consumer03
 *
 *   测试多次调用有外置allocator场景下的L2 allocator
 *   预期结果：同一条流上多次调用 select l2 allocator，但是placement不同，生成2个kerenel
 */
TEST_F(FastLoweringGlobalDataUT, GetOrCreateAllocatorInMain_WithExternalAllocator_CallMultiTimesWithDiffPlacement) {
  LoweringGlobalData global_data;
  InitTestFramesWithStream(global_data, 3);

  // prepare external allocator on init and main
  bg::FrameSelector::OnInitRoot([&global_data]()-> std::vector<bg::ValueHolderPtr> {
    auto external_allocator_init = bg::ValueHolder::CreateFeed(-2);
    global_data.SetExternalAllocator(static_cast<bg::ValueHolderPtr &&>(external_allocator_init), ExecuteGraphType::kInit);
    return {};
  });
  auto external_allocator = bg::ValueHolder::CreateFeed(-2);
  global_data.SetExternalAllocator(static_cast<bg::ValueHolderPtr &&>(external_allocator), ExecuteGraphType::kMain);

  auto l2_allocator_00 =
      global_data.GetOrCreateL2Allocator(0, {TensorPlacement::kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
  EXPECT_NE(l2_allocator_00, nullptr);
  EXPECT_EQ(l2_allocator_00->GetFastNode()->GetType(), "SelectL2Allocator");
  auto consumer00 = bg::ValueHolder::CreateVoid<bg::ValueHolder>("consumer00", {l2_allocator_00});

  auto l2_allocator_01 =
      global_data.GetOrCreateL2Allocator(0, {TensorPlacement::kOnHost, AllocatorUsage::kAllocNodeOutput});
  EXPECT_NE(l2_allocator_01, nullptr);
  EXPECT_EQ(l2_allocator_01->GetFastNode()->GetType(), "CreateHostL2Allocator");
  EXPECT_NE(l2_allocator_00, l2_allocator_01);
  auto consumer01 = bg::ValueHolder::CreateVoid<bg::ValueHolder>("consumer01", {l2_allocator_01});

  auto l2_allocator_02 =
      global_data.GetOrCreateL2Allocator(0, {TensorPlacement::kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
  EXPECT_NE(l2_allocator_02, nullptr);
  EXPECT_EQ(l2_allocator_02->GetFastNode()->GetType(), "SelectL2Allocator");
  auto consumer02 = bg::ValueHolder::CreateVoid<bg::ValueHolder>("consumer02", {l2_allocator_02});

  auto l2_allocator_03 =
      global_data.GetOrCreateL2Allocator(0, {TensorPlacement::kOnHost, AllocatorUsage::kAllocNodeOutput});
  EXPECT_NE(l2_allocator_03, nullptr);
  EXPECT_EQ(l2_allocator_03->GetFastNode()->GetType(), "CreateHostL2Allocator");
  auto consumer03 = bg::ValueHolder::CreateVoid<bg::ValueHolder>("consumer03", {l2_allocator_03});

  auto main_frame = bg::ValueHolder::PopGraphFrame();
  auto init_exe_graph = init_frame->GetExecuteGraph().get();
  // ge::DumpGraph(init_exe_graph, "l2_allocator_init");
  EXPECT_EQ(ExeGraphSummaryChecker(init_exe_graph)
                .StrictDirectNodeTypes(std::map<std::string, size_t>{{"Const", 5},
                                                                     {"Data", 2},
                                                                     {"SplitRtStreams", 1},
                                                                     {"CreateL1Allocator", 2},
                                                                     {"CreateL2Allocators", 1},
                                                                     {"SelectL1Allocator", 2},
                                                                     {"InnerNetOutput", 1},
                                                                     {"CreateHostL2Allocator", 1}}),
            "success");

  auto main_exe_graph = main_frame->GetExecuteGraph().get();
  ge::DumpGraph(main_exe_graph->GetParentGraphBarePtr(), "CallMultiTimesWithDiffPlacement");
  EXPECT_EQ(ExeGraphSummaryChecker(main_exe_graph)
                .StrictDirectNodeTypes(std::map<std::string, size_t>{{"Const", 2},
                                                                     {"Data", 2},
                                                                     {"InnerData", 5},
                                                                     {"SplitRtStreams", 1},
                                                                     {"SelectL1Allocator", 2},
                                                                     {"SelectL2Allocator", 1},
                                                                     {"CreateHostL2Allocator", 1},
                                                                     {"consumer00", 1},
                                                                     {"consumer01", 1},
                                                                     {"consumer02", 1},
                                                                     {"consumer03", 1}}),
            "success");
  FastNodeTopoChecker checker(l2_allocator_00);
  // Const(logic_stream_id), GetStreamById, SelectL1Allocator(L1 allocator), InnerData(L2 allocators)
  EXPECT_EQ(checker.StrictConnectFrom(std::vector<SrcFastNode>({{"Const"}, {"SplitRtStreams"}, {"SelectL1Allocator"}, {"InnerData"}})), "success");
}

/*
 *       Data  Const                     Const(out)
 *          \  /                          |
 *     SplitRtStreams              CreateL1Allocator(out)   Const
 *                 \              /                         /
 *                 SelectL1Allocator------> CreateL2Allocators(out)
 *                     (out)    \              /
 *                              CreateInitL2Allocator (out)
 *                                     /    \
 *                                    /      consumer
 *                                    \       /
 *                                 InnerNetOutput
 */
TEST_F(FastLoweringGlobalDataUT, GetOrCreateInitL2AllocatorOnInit_Device) {
  LoweringGlobalData global_data;
  InitTestFramesWithStream(global_data, 3);

  // prepare external allocator on init and main
  auto init_out1 = bg::FrameSelector::OnInitRoot([&global_data]()-> std::vector<bg::ValueHolderPtr> {
    auto external_allocator_init = bg::ValueHolder::CreateFeed(-2);
    global_data.SetExternalAllocator(static_cast<bg::ValueHolderPtr &&>(external_allocator_init), ExecuteGraphType::kInit);

    AllocatorDesc desc = {kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput};
    auto init_l2_allocator = global_data.GetOrCreateL2Allocator(0, desc);
    EXPECT_NE(init_l2_allocator, nullptr);
    auto consumer = bg::ValueHolder::CreateSingleDataOutput("consumer", {init_l2_allocator});
    return {init_l2_allocator, consumer};
  });
  EXPECT_EQ(init_out1.size(), 2);

  auto device_l2_allocator_init = bg::HolderOnInit(init_out1[0]);
  EXPECT_NE(device_l2_allocator_init, nullptr);
  auto get_device_l2_allocator_init =
    global_data.GetInitL2Allocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
  EXPECT_NE(device_l2_allocator_init, nullptr);
  EXPECT_EQ(device_l2_allocator_init->GetFastNode(), get_device_l2_allocator_init->GetFastNode());

  auto init_exe_graph = init_frame->GetExecuteGraph().get();
  ge::DumpGraph(init_exe_graph, "l2_allocator_init");
  EXPECT_EQ(ExeGraphSummaryChecker(init_exe_graph)
                .StrictDirectNodeTypes(std::map<std::string, size_t>{{"Const", 4},
                                                                     {"Data", 2},
                                                                     {"SplitRtStreams", 1},
                                                                     {"CreateL1Allocator", 1},
                                                                     {"CreateL2Allocators", 1},
                                                                     {"CreateInitL2Allocator", 1},
                                                                     {"SelectL1Allocator", 1},
                                                                     {"consumer", 1},
                                                                     {"InnerNetOutput", 1}}),
            "success");
  auto consumer = HolderOnInit(init_out1[1]);
  FastNodeTopoChecker checker(consumer);
  EXPECT_EQ(checker.StrictConnectFrom(std::vector<SrcFastNode>({{"CreateInitL2Allocator"}})), "success");
}

/*
 *       Data  Const                     Const(out)
 *          \  /                          |
 *     SplitRtStreams              CreateL1Allocator(out)   Const
 *                 \              /                         /
 *                 SelectL1Allocator------> CreateL2Allocators(out)
 *                     (out)    \              /
 *                              CreateHostL2Allocator (out)
 *                                     /
 *                                    /
 *                              InnerNetOutput
 */

TEST_F(FastLoweringGlobalDataUT, GetOrCreateInitL2AllocatorOnInit_Host) {
  LoweringGlobalData global_data;
  InitTestFramesWithStream(global_data, 3);

  // prepare external allocator on init and main
  auto init_out = bg::FrameSelector::OnInitRoot([&global_data]() -> std::vector<bg::ValueHolderPtr> {
    auto external_allocator_init = bg::ValueHolder::CreateFeed(-2);
    global_data.SetExternalAllocator(static_cast<bg::ValueHolderPtr &&>(external_allocator_init), ExecuteGraphType::kInit);

    AllocatorDesc desc = {kOnHost, AllocatorUsage::kAllocNodeOutput};
    auto init_l2_allocator =  global_data.GetOrCreateL2Allocator(0, desc);
    EXPECT_NE(init_l2_allocator, nullptr);
    auto consumer = bg::ValueHolder::CreateSingleDataOutput("consumer", {init_l2_allocator});
    return {init_l2_allocator, consumer};
  });
  EXPECT_EQ(init_out.size(), 2);

  auto host_l2_allocator_init = bg::HolderOnInit(init_out[0]);
  EXPECT_NE(host_l2_allocator_init, nullptr);
  auto get_host_l2_allocator_init =
    global_data.GetInitL2Allocator({kOnHost, AllocatorUsage::kAllocNodeOutput});
  EXPECT_NE(get_host_l2_allocator_init, nullptr);
  EXPECT_EQ(host_l2_allocator_init->GetFastNode(), get_host_l2_allocator_init->GetFastNode());

  auto init_exe_graph = init_frame->GetExecuteGraph().get();
  ge::DumpGraph(init_exe_graph, "l2_allocator_init");
  EXPECT_EQ(ExeGraphSummaryChecker(init_exe_graph)
                .StrictDirectNodeTypes(std::map<std::string, size_t>{{"Const", 3},
                                                                     {"Data", 2},
                                                                     {"SplitRtStreams", 1},
                                                                     {"CreateL1Allocator", 1},
                                                                     {"SelectL1Allocator", 1},
                                                                     {"CreateHostL2Allocator", 1},
                                                                     {"consumer", 1},
                                                                     {"InnerNetOutput", 1}}),
            "success");

  auto consumer = HolderOnInit(init_out[1]);
  FastNodeTopoChecker checker(consumer);
  EXPECT_EQ(checker.StrictConnectFrom(std::vector<SrcFastNode>({{"CreateHostL2Allocator"}})), "success");

  // get host l2 allocator in another init lowering
  auto init_out1 = bg::FrameSelector::OnInitRoot([&global_data]() -> std::vector<bg::ValueHolderPtr> {
    AllocatorDesc desc = {kOnHost, AllocatorUsage::kAllocNodeOutput};
    auto init_l2_allocator =  global_data.GetOrCreateL2Allocator(0, desc);
    EXPECT_NE(init_l2_allocator, nullptr);
    auto consumer1 = bg::ValueHolder::CreateSingleDataOutput("consumer1", {init_l2_allocator});
    return {init_l2_allocator, consumer1};
  });
  EXPECT_FALSE(init_out1.empty());

  auto consumer1 = HolderOnInit(init_out[1]);
  FastNodeTopoChecker checker1(consumer1);
  EXPECT_EQ(checker.StrictConnectFrom(std::vector<SrcFastNode>({{"CreateHostL2Allocator"}})), "success");
  EXPECT_EQ(ExeGraphSummaryChecker(init_exe_graph)
                .StrictDirectNodeTypes(std::map<std::string, size_t>{{"Const", 3},
                                                                     {"Data", 2},
                                                                     {"SplitRtStreams", 1},
                                                                     {"CreateL1Allocator", 1},
                                                                     {"SelectL1Allocator", 1},
                                                                     {"CreateHostL2Allocator", 1},
                                                                     {"consumer", 1},
                                                                     {"consumer1", 1},
                                                                     {"InnerNetOutput", 1}}),
            "success");
}

TEST_F(FastLoweringGlobalDataUT, GetOrCreateAllL2Allocators_success) {
  LoweringGlobalData global_data;
  InitTestFramesWithStream(global_data, 3);

  global_data.LoweringAndSplitRtStreams(3);
  auto l2_allocator = global_data.GetOrCreateL2Allocator(1, {kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
  EXPECT_NE(l2_allocator, nullptr);
  auto all_l2_allocators = global_data.GetOrCreateAllL2Allocators();
  EXPECT_NE(all_l2_allocators, nullptr);
  EXPECT_EQ(all_l2_allocators->GetFastNode()->GetType(), "Init");
  auto consumer = bg::ValueHolder::CreateSingleDataOutput("consumer", {all_l2_allocators});

  auto init_exe_graph = init_frame->GetExecuteGraph().get();
  ge::DumpGraph(init_exe_graph, "l2_allocator_init");
  EXPECT_EQ(ExeGraphSummaryChecker(init_exe_graph)
                .StrictDirectNodeTypes(std::map<std::string, size_t>{{"Const", 4},
                                                                     {"Data", 1},
                                                                     {"SplitRtStreams", 1},
                                                                     {"CreateL1Allocator", 1},
                                                                     {"CreateL2Allocators", 1},
                                                                     {"InnerNetOutput", 1}}),
            "success");

  auto all_l2_allocators_in_init = HolderOnInit(all_l2_allocators);
  EXPECT_EQ(all_l2_allocators_in_init->GetFastNode()->GetType(), "CreateL2Allocators");

  FastNodeTopoChecker checker(consumer);
  EXPECT_EQ(checker.StrictConnectFrom(std::vector<SrcFastNode>({{"InnerData", 0}})), "success");
}
/*
 * main_graph:
 *                  Data(rt_streams)
 *                          |
 *                     SplitRtStreams
 *                          |
 *                        consumer
 *
 */
TEST_F(FastLoweringGlobalDataUT, GetStreamById_Main_Once) {
  InitTestFrames();
  LoweringGlobalData global_data;

  // prepare rtStreams
  auto all_rt_streams = global_data.LoweringAndSplitRtStreams(1);
  EXPECT_EQ(all_rt_streams.size(), 1);

  auto rt_stream = global_data.GetStreamById(0);
  EXPECT_NE(rt_stream, nullptr);
  auto consumer = bg::ValueHolder::CreateVoid<bg::ValueHolder>("consumer", {rt_stream});

  auto main_frame = bg::ValueHolder::PopGraphFrame();
  auto main_exe_graph = main_frame->GetExecuteGraph().get();
  EXPECT_EQ(ExeGraphSummaryChecker(main_exe_graph)
                .StrictDirectNodeTypes(std::map<std::string, size_t>{
                    {"Data", 1}, {"Const", 1}, {"SplitRtStreams", 1}, {"consumer", 1}}),
            "success");
  FastNodeTopoChecker checker(rt_stream);
  // Const(logic_stream_id), Data(rt_streams)
  EXPECT_EQ(checker.StrictConnectFrom(std::vector<SrcFastNode>({{"Data"}, {"Const"}})), "success");
  FastNodeTopoChecker consumer_checker(consumer);
  EXPECT_EQ(consumer_checker.StrictConnectFrom(std::vector<SrcFastNode>({{"SplitRtStreams", 0}})), "success");
}
/*
 * main_graph:
 *                         Data(rt_streams)
 *                               |
 *                       SplitRtStreams
 *                             /  \
 *                      consumer0  consumer1
 *
 */
TEST_F(FastLoweringGlobalDataUT, GetStreamById_Main_SameStreamCallTwice) {
  InitTestFrames();
  LoweringGlobalData global_data;

  // prepare rtStreams
  auto all_rt_streams = global_data.LoweringAndSplitRtStreams(1);
  EXPECT_EQ(all_rt_streams.size(), 1);
  EXPECT_TRUE(global_data.IsSingleStreamScene());

  auto rt_stream = global_data.GetStreamById(0);
  EXPECT_NE(rt_stream, nullptr);
  auto consumer = bg::ValueHolder::CreateVoid<bg::ValueHolder>("consumer0", {rt_stream});

  auto rt_stream1 = global_data.GetStreamById(0);
  EXPECT_NE(rt_stream1, nullptr);
  auto consumer1 = bg::ValueHolder::CreateVoid<bg::ValueHolder>("consumer1", {rt_stream1});

  auto main_frame = bg::ValueHolder::PopGraphFrame();
  auto main_exe_graph = main_frame->GetExecuteGraph().get();
  EXPECT_EQ(ExeGraphSummaryChecker(main_exe_graph)
                .StrictDirectNodeTypes(std::map<std::string, size_t>{
                    {"Data", 1},{"Const", 1}, {"SplitRtStreams", 1}, {"consumer0", 1}, {"consumer1", 1}}),
            "success");
  FastNodeTopoChecker checker(rt_stream);
  // Const(logic_stream_id), Data(rt_streams)
  EXPECT_EQ(checker.StrictConnectFrom(std::vector<SrcFastNode>({{"Data"}, {"Const"}})), "success");
  EXPECT_EQ(checker.StrictConnectTo(0, std::vector<SrcFastNode>({{"consumer0"}, {"consumer1"}})), "success");
}

/*
 * main_graph:
 *                        Data(rt_streams)
 *                             |
 *                       SplitRtStreams
 *                             /     \
 *                     consumer0     consumer1
 *
 */
TEST_F(FastLoweringGlobalDataUT, GetStreamById_Main_DiffStreamCallTwice) {
  InitTestFrames();
  LoweringGlobalData global_data;

  // prepare rtStreams
  auto all_rt_streams = global_data.LoweringAndSplitRtStreams(2);
  EXPECT_EQ(all_rt_streams.size(), 2);
  EXPECT_TRUE(!global_data.IsSingleStreamScene());

  auto rt_stream = global_data.GetStreamById(0);
  EXPECT_NE(rt_stream, nullptr);
  auto consumer = bg::ValueHolder::CreateVoid<bg::ValueHolder>("consumer0", {rt_stream});

  auto rt_stream1 = global_data.GetStreamById(1);
  EXPECT_NE(rt_stream1, nullptr);
  auto consumer1 = bg::ValueHolder::CreateVoid<bg::ValueHolder>("consumer1", {rt_stream1});

  auto main_frame = bg::ValueHolder::PopGraphFrame();
  auto main_exe_graph = main_frame->GetExecuteGraph().get();
  EXPECT_EQ(ExeGraphSummaryChecker(main_exe_graph)
                .StrictDirectNodeTypes(std::map<std::string, size_t>{
                    {"Data", 1},{"Const", 1}, {"SplitRtStreams", 1}, {"consumer0", 1}, {"consumer1", 1}}),
            "success");
  FastNodeTopoChecker checker(rt_stream);
  // Const(logic_stream_id), Data(rt_streams)
  EXPECT_EQ(checker.StrictConnectFrom(std::vector<SrcFastNode>({{"Data"}, {"Const"}})), "success");
  EXPECT_EQ(checker.StrictConnectTo(0, std::vector<SrcFastNode>({{"consumer0"}})), "success");
  EXPECT_EQ(checker.StrictConnectTo(1, std::vector<SrcFastNode>({{"consumer1"}})), "success");
}

TEST_F(FastLoweringGlobalDataUT, GetStreamById_Main_WithOutSplitRtStreams) {
  InitTestFrames();
  LoweringGlobalData global_data;

  auto rt_stream = global_data.GetStreamById(0);
  EXPECT_EQ(rt_stream, nullptr);
}

/*
 * init_graph:
 *                  Data(rt_streams)
 *                          |
 *                     SplitRtStreams
 *                          |
 *                        consumer
 *
 */
TEST_F(FastLoweringGlobalDataUT, GetStreamById_Init_Once) {
  InitTestFrames();
  LoweringGlobalData global_data;
  auto consumer_and_stream = bg::FrameSelector::OnInitRoot([&global_data]() -> std::vector<bg::ValueHolderPtr> {
    // prepare rtStreams
    auto all_rt_streams = global_data.LoweringAndSplitRtStreams(1);
    EXPECT_EQ(all_rt_streams.size(), 1);

    auto rt_stream = global_data.GetStreamById(0);
    EXPECT_NE(rt_stream, nullptr);
    auto consumer = bg::ValueHolder::CreateSingleDataOutput("consumer", {rt_stream});
    return {consumer, rt_stream};
  });
  EXPECT_EQ(consumer_and_stream.size(), 2);

  auto init_exe_graph = init_frame->GetExecuteGraph().get();
  EXPECT_EQ(ExeGraphSummaryChecker(init_exe_graph)
                .StrictDirectNodeTypes(std::map<std::string, size_t>{
                    {"Data", 1}, {"Const", 1}, {"SplitRtStreams", 1}, {"consumer", 1}, {"InnerNetOutput", 1}}),
            "success");
  auto split_rt_streams = HolderOnInit(consumer_and_stream[1]);
  FastNodeTopoChecker checker(split_rt_streams);
  // Const(logic_stream_id), Data(rt_streams)
  EXPECT_EQ(checker.StrictConnectFrom(std::vector<SrcFastNode>({{"Data"}, {"Const"}})), "success");
  auto consumer = HolderOnInit(consumer_and_stream[0]);
  FastNodeTopoChecker consumer_checker(consumer);
  EXPECT_EQ(consumer_checker.StrictConnectFrom(std::vector<SrcFastNode>({{"SplitRtStreams", 0}})), "success");
}

/*
 * init_graph:
 *                        Data(rt_streams)
 *                               |
 *                       SplitRtStreams
 *                             /  \
 *                      consumer0  consumer1
 *
 */
TEST_F(FastLoweringGlobalDataUT, GetStreamById_Init_SameStreamCallTwice) {
  InitTestFrames();
  LoweringGlobalData global_data;

  auto consumer_and_stream = bg::FrameSelector::OnInitRoot([&global_data]() -> std::vector<bg::ValueHolderPtr> {
    // prepare rtStreams
    auto all_rt_streams = global_data.LoweringAndSplitRtStreams(1);
    EXPECT_EQ(all_rt_streams.size(), 1);

    auto rt_stream = global_data.GetStreamById(0);
    EXPECT_NE(rt_stream, nullptr);
    auto consumer = bg::ValueHolder::CreateSingleDataOutput("consumer0", {rt_stream});

    auto rt_stream1 = global_data.GetStreamById(0);
    EXPECT_NE(rt_stream1, nullptr);
    auto consumer1 = bg::ValueHolder::CreateSingleDataOutput("consumer1", {rt_stream1});
    return {rt_stream, consumer, consumer1};
  });
  EXPECT_EQ(consumer_and_stream.size(), 3);

  auto init_exe_graph = init_frame->GetExecuteGraph().get();
  EXPECT_EQ(ExeGraphSummaryChecker(init_exe_graph)
                .StrictDirectNodeTypes(std::map<std::string, size_t>{{"Data", 1},
                                                                     {"Const", 1},
                                                                     {"SplitRtStreams", 1},
                                                                     {"consumer0", 1},
                                                                     {"consumer1", 1},
                                                                     {"InnerNetOutput", 1}}),
            "success");
  auto rt_stream = HolderOnInit(consumer_and_stream[0]);
  FastNodeTopoChecker checker(rt_stream);
  // Const(logic_stream_id), Data(rt_streams)
  EXPECT_EQ(checker.StrictConnectFrom(std::vector<SrcFastNode>({{"Data"}, {"Const"}})), "success");
  EXPECT_EQ(checker.StrictConnectTo(0, std::vector<SrcFastNode>({{"consumer0"}, {"consumer1"}, {"InnerNetOutput"}})),
            "success");
}

TEST_F(FastLoweringGlobalDataUT, GetStreamById_Init_WithOutSplitRtStreams) {
  InitTestFrames();
  LoweringGlobalData global_data;
  auto consumer_and_stream = bg::FrameSelector::OnInitRoot([&global_data]() -> std::vector<bg::ValueHolderPtr> {
    auto rt_stream = global_data.GetStreamById(0);
    EXPECT_EQ(rt_stream, nullptr);
    return {rt_stream};
  });
  EXPECT_EQ(consumer_and_stream.size(), 0);
}

TEST_F(FastLoweringGlobalDataUT, GetStreamById_Init_StreamIdOutOfRange) {
  InitTestFrames();
  LoweringGlobalData global_data;
  auto consumer_and_stream = bg::FrameSelector::OnInitRoot([&global_data]() -> std::vector<bg::ValueHolderPtr> {
    // prepare rtStreams
    auto all_rt_streams = global_data.LoweringAndSplitRtStreams(1);
    EXPECT_EQ(all_rt_streams.size(), 1);

    auto rt_stream = global_data.GetStreamById(2); // stream id out of range
    EXPECT_EQ(rt_stream, nullptr);
    return {rt_stream};
  });
  EXPECT_EQ(consumer_and_stream.size(), 0);
}

TEST_F(FastLoweringGlobalDataUT, GetStreamById_Init_StreamNumOutOfRange) {
  InitTestFrames();
  LoweringGlobalData global_data;
  auto consumer_and_stream = bg::FrameSelector::OnInitRoot([&global_data]() -> std::vector<bg::ValueHolderPtr> {
    // prepare rtStreams
    auto all_rt_streams = global_data.LoweringAndSplitRtStreams(2); // stream num out of range
    EXPECT_EQ(all_rt_streams.size(), 0);

    auto rt_stream = global_data.GetStreamById(0);
    EXPECT_EQ(rt_stream, nullptr);
    return {rt_stream};
  });
  EXPECT_EQ(consumer_and_stream.size(), 0);
}

TEST_F(FastLoweringGlobalDataUT, GetNotifyById_Main) {
  InitTestFrames();
  LoweringGlobalData global_data;
  auto notify_0 = bg::ValueHolder::CreateFeed(0);
  auto notify_1 = bg::ValueHolder::CreateFeed(1);
  std::vector<bg::ValueHolderPtr> notifies{notify_0, notify_1};
  (void) bg::FrameSelector::OnMainRoot([&global_data, &notifies]() -> std::vector<bg::ValueHolderPtr> {
    global_data.SetRtNotifies(notifies);

    auto rt_notify0 = global_data.GetNotifyById(0);
    EXPECT_NE(rt_notify0, nullptr);
    auto rt_notify1 = global_data.GetNotifyById(1);
    EXPECT_NE(rt_notify1, nullptr);
    auto rt_notify2 = global_data.GetNotifyById(2);
    EXPECT_EQ(rt_notify2, nullptr);
    return {};
  });

  (void) bg::FrameSelector::OnInitRoot([&global_data]() -> std::vector<bg::ValueHolderPtr> {
    auto rt_notify0 = global_data.GetNotifyById(0);
    EXPECT_EQ(rt_notify0, nullptr);
    return {};
  });
}

TEST_F(FastLoweringGlobalDataUT, GetNotifyById_Init) {
  InitTestFrames();
  LoweringGlobalData global_data;
  auto notify_0 = bg::ValueHolder::CreateFeed(0);
  auto notify_1 = bg::ValueHolder::CreateFeed(1);
  std::vector<bg::ValueHolderPtr> notifies{notify_0, notify_1};
  (void) bg::FrameSelector::OnInitRoot([&global_data, &notifies]() -> std::vector<bg::ValueHolderPtr> {
    global_data.SetRtNotifies(notifies);

    auto rt_notify0 = global_data.GetNotifyById(0);
    EXPECT_NE(rt_notify0, nullptr);
    auto rt_notify1 = global_data.GetNotifyById(1);
    EXPECT_NE(rt_notify1, nullptr);
    auto rt_notify2 = global_data.GetNotifyById(2);
    EXPECT_EQ(rt_notify2, nullptr);
    return {};
  });
}
}  // namespace gert

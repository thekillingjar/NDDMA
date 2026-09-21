/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_builder/multi_stream/bg_event.h"
#include <gtest/gtest.h>
#include "ge_graph_dsl/graph_dsl.h"
#include "register/node_converter_registry.h"
#include "exe_graph_comparer.h"
#include "faker/global_data_faker.h"
#include "common/bg_test.h"
#include "register/op_impl_registry.h"
#include "common/share_graph.h"
#include "common/summary_checker.h"
#include "common/const_data_helper.h"

#include "common/topo_checker.h"
#include "graph/utils/graph_dump_utils.h"

namespace gert {
namespace bg {
namespace {
constexpr const ge::char_t *kGlobalDataGertEvents = "GertEvents";
using namespace ge;
/**
 *     A(0)
 *    /    \.
 *   B(1)   C(2)
 *   |      |
 *   D(1)   E(2)
 *   |     |
 *   |     /
 *    \  /
 *      F(0)
 * 
 */
void CreateMultiStreamComputeGraph(const ge::ComputeGraphPtr &graph, std::vector<EventInfo> &all_event_info) {
  const auto &a_desc = std::make_shared<ge::OpDesc>("A", DATA);
  a_desc->AddInputDesc(ge::GeTensorDesc());
  a_desc->AddOutputDesc(ge::GeTensorDesc());
  a_desc->SetStreamId(0);
  std::vector<int64_t> a_send = {0, 1};
  AttrUtils::SetListInt(a_desc, ge::ATTR_NAME_SEND_EVENT_IDS, a_send);
  const auto &a_node = graph->AddNode(a_desc);
  all_event_info.emplace_back(EventInfo(0, 1));
  all_event_info.emplace_back(EventInfo(0, 2));

  const auto &b_desc = std::make_shared<OpDesc>("B", "testa");
  b_desc->AddInputDesc(GeTensorDesc());
  b_desc->AddOutputDesc(GeTensorDesc());
  b_desc->SetStreamId(1);
  AttrUtils::SetListInt(b_desc, ge::ATTR_NAME_RECV_EVENT_IDS, {0});
  const auto &b_node = graph->AddNode(b_desc);

  const auto &c_desc = std::make_shared<OpDesc>("C", "testa");
  c_desc->AddInputDesc(GeTensorDesc());
  c_desc->AddOutputDesc(GeTensorDesc());
  c_desc->SetStreamId(2);
  AttrUtils::SetListInt(c_desc, ge::ATTR_NAME_RECV_EVENT_IDS, {1});
  const auto &c_node = graph->AddNode(c_desc);

  const auto &d_desc = std::make_shared<OpDesc>("D", "testa");
  d_desc->AddInputDesc(GeTensorDesc());
  d_desc->AddOutputDesc(GeTensorDesc());
  d_desc->SetStreamId(1);
  AttrUtils::SetListInt(d_desc, ge::ATTR_NAME_SEND_EVENT_IDS, {2});
  const auto &d_node = graph->AddNode(d_desc);
  all_event_info.emplace_back(EventInfo(1, 0));

  const auto &e_desc = std::make_shared<OpDesc>("E", "testa");
  e_desc->AddInputDesc(GeTensorDesc());
  e_desc->AddOutputDesc(GeTensorDesc());
  e_desc->SetStreamId(2);
  AttrUtils::SetListInt(e_desc, ge::ATTR_NAME_SEND_EVENT_IDS, {3});
  const auto &e_node = graph->AddNode(e_desc);
  all_event_info.emplace_back(EventInfo(2, 0));

  const auto &f_desc = std::make_shared<OpDesc>("F", "testa");
  f_desc->AddInputDesc(GeTensorDesc());
  f_desc->AddInputDesc(GeTensorDesc());
  f_desc->AddOutputDesc(GeTensorDesc());
  f_desc->SetStreamId(0);
  AttrUtils::SetListInt(f_desc, ge::ATTR_NAME_RECV_EVENT_IDS, {2,3});
  const auto &f_node = graph->AddNode(f_desc);

  GraphUtils::AddEdge(a_node->GetOutDataAnchor(0), b_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(a_node->GetOutDataAnchor(0), c_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(c_node->GetOutDataAnchor(0), e_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(b_node->GetOutDataAnchor(0), d_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(d_node->GetOutDataAnchor(0), f_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(e_node->GetOutDataAnchor(0), f_node->GetInDataAnchor(1));
  graph->SetGraphUnknownFlag(true);
}

/**
 * 
 *     A(0)   C(1)
 *    /          |
 *   B(0)    D(1)
 *   |     /
 *   E(0)
 * 
 */
void CreateFirstEventSyncComputeGraph(const ge::ComputeGraphPtr &graph) {
  const auto &a_desc = std::make_shared<ge::OpDesc>("A", DATA);
  a_desc->AddInputDesc(ge::GeTensorDesc());
  a_desc->AddOutputDesc(ge::GeTensorDesc());
  a_desc->SetStreamId(0);
  const auto &a_node = graph->AddNode(a_desc);

  const auto &b_desc = std::make_shared<OpDesc>("B", "testa");
  b_desc->AddInputDesc(GeTensorDesc());
  b_desc->AddOutputDesc(GeTensorDesc());
  b_desc->SetStreamId(0);
  const auto &b_node = graph->AddNode(b_desc);

  const auto &c_desc = std::make_shared<OpDesc>("C", "testa");
  c_desc->AddInputDesc(GeTensorDesc());
  c_desc->AddOutputDesc(GeTensorDesc());
  c_desc->SetStreamId(1);
  const auto &c_node = graph->AddNode(c_desc);

  const auto &d_desc = std::make_shared<OpDesc>("D", "testa");
  d_desc->AddInputDesc(GeTensorDesc());
  d_desc->AddOutputDesc(GeTensorDesc());
  d_desc->SetStreamId(1);
  AttrUtils::SetListInt(d_desc, ge::ATTR_NAME_SEND_EVENT_IDS, {0});
  const auto &d_node = graph->AddNode(d_desc);

  const auto &e_desc = std::make_shared<OpDesc>("E", "testa");
  e_desc->AddInputDesc(GeTensorDesc());
  e_desc->AddInputDesc(GeTensorDesc());
  e_desc->AddOutputDesc(GeTensorDesc());
  e_desc->SetStreamId(0);
  AttrUtils::SetListInt(e_desc, ge::ATTR_NAME_RECV_EVENT_IDS, {0});
  const auto &e_node = graph->AddNode(e_desc);

  GraphUtils::AddEdge(a_node->GetOutDataAnchor(0), b_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(b_node->GetOutDataAnchor(0), e_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(c_node->GetOutDataAnchor(0), d_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(d_node->GetOutDataAnchor(0), e_node->GetInDataAnchor(1));
  graph->SetGraphUnknownFlag(true);
}

/**
 * 
 *       A(0)
 *    /    \    \
 *   B(0)   C(1)  D(1)
 *   |
 *   E(0)
 * 
 */
void CreateLastEventSyncComputeGraph(const ge::ComputeGraphPtr &graph) {
  const auto &a_desc = std::make_shared<ge::OpDesc>("A", DATA);
  a_desc->AddInputDesc(ge::GeTensorDesc());
  a_desc->AddOutputDesc(ge::GeTensorDesc());
  a_desc->SetStreamId(0);
  a_desc->SetAttachedStreamId(3);
  AttrUtils::SetListInt(a_desc, ge::ATTR_NAME_SEND_EVENT_IDS, {0,1});
  const auto &a_node = graph->AddNode(a_desc);

  const auto &b_desc = std::make_shared<OpDesc>("B", "testb");
  b_desc->AddInputDesc(GeTensorDesc());
  b_desc->AddOutputDesc(GeTensorDesc());
  b_desc->SetStreamId(0);
  b_desc->SetAttachedStreamId(3);
  const auto &b_node = graph->AddNode(b_desc);

  const auto &c_desc = std::make_shared<OpDesc>("C", "testc");
  c_desc->AddInputDesc(GeTensorDesc());
  c_desc->AddOutputDesc(GeTensorDesc());
  c_desc->SetStreamId(1);
  c_desc->SetAttachedStreamId(3);
  AttrUtils::SetListInt(c_desc, ge::ATTR_NAME_RECV_EVENT_IDS, {0});
  const auto &c_node = graph->AddNode(c_desc);

  const auto &d_desc = std::make_shared<OpDesc>("D", "testd");
  d_desc->AddInputDesc(GeTensorDesc());
  d_desc->AddOutputDesc(GeTensorDesc());
  d_desc->SetStreamId(1);
  AttrUtils::SetListInt(d_desc, ge::ATTR_NAME_RECV_EVENT_IDS, {1});
  const auto &d_node = graph->AddNode(d_desc);

  const auto &e_desc = std::make_shared<OpDesc>("E", "testa");
  e_desc->AddInputDesc(GeTensorDesc());
  e_desc->AddOutputDesc(GeTensorDesc());
  e_desc->SetStreamId(0);
  const auto &e_node = graph->AddNode(e_desc);

  GraphUtils::AddEdge(a_node->GetOutDataAnchor(0), b_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(b_node->GetOutDataAnchor(0), e_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(a_node->GetOutDataAnchor(0), c_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(a_node->GetOutDataAnchor(0), d_node->GetInDataAnchor(0));
  graph->SetGraphUnknownFlag(true);
}
} // namespace
class BgEventUT : public BgTestAutoCreateFrame {
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
 * main_graph:
 *     Const(logic_stream_id)           Const(event_ids)  CreateGertEvents   Data(rt_events)   SelectL2Allocator
 *                |        \                   |                 |                  |                 |
 *                |        GetRtStreamById       |                 |                  |                 |
 *                |            |               |                 |                  |                 |
 *                +-------->SendEvents<--------+-----------------+------------------+-----------------+
 *
 *  测试无外置allocator场景下的SendEvent
 */
TEST_F(BgEventUT, SendEvents_OneEvent) {
  InitTestFrames();
  LoweringGlobalData global_data;

  // prepare stream num in init for l2 allocator
  int64_t stream_num = 3;
  auto init_out = FrameSelector::OnInitRoot([&stream_num, &global_data]()-> std::vector<ValueHolderPtr> {
    auto stream_num_holder = ValueHolder::CreateConst(&stream_num, sizeof(stream_num));
    global_data.SetUniqueValueHolder(kGlobalDataModelStreamNum, stream_num_holder);
    return {};
  });

  // prepare rtStreams
  global_data.LoweringAndSplitRtStreams(stream_num);

  // prepare rtEvents
  auto rt_events_holder = ValueHolder::CreateFeed(1);
  global_data.SetUniqueValueHolder(bg::kGlobalDataRtEvents, rt_events_holder);

  // prepare GertEvents
  auto gert_events_holder = ValueHolder::CreateFeed(2);
  global_data.SetUniqueValueHolder(kGlobalDataGertEvents, gert_events_holder);

  auto order_holder = bg::SendEvents(0, {1}, global_data);
  EXPECT_NE(order_holder, nullptr);

  auto main_frame = ValueHolder::PopGraphFrame();
  auto init_exe_graph = init_frame->GetExecuteGraph().get();
   // GE_DUMP(init_exe_graph, "send_event_init");
  EXPECT_EQ(ExeGraphSummaryChecker(init_exe_graph)
                .StrictDirectNodeTypes(std::map<std::string, size_t>{
                    {"Const", 3}, {"CreateL1Allocator", 1}, {"CreateL2Allocators", 1}, {"InnerNetOutput", 1}}),
            "success");

  auto main_exe_graph = main_frame->GetExecuteGraph().get();
   // GE_DUMP(main_exe_graph, "send_event_main");
  EXPECT_EQ(ExeGraphSummaryChecker(main_exe_graph)
                .DirectNodeTypesOnlyCareAbout(std::map<std::string, size_t>{
                 {"Data", 3}, {"InnerData", 2}, {"SplitRtStreams", 1}, {"SelectL2Allocator", 1}, {"SendEvents", 1}}),
            "success");
  FastNodeTopoChecker checker(order_holder);
  // Const(logic_stream_id), GetRtStreamById, Const(event_ids), Data(gert events), Data(rt_events),SelectL2Allocator
  EXPECT_EQ(checker.StrictConnectFrom(std::vector<FastSrcNode>({{"SplitRtStreams"}, {"Const"}, {"Data"}, {"Data"}, {"SelectL2Allocator"}}), true), "success");
}

TEST_F(BgEventUT, SendEvents_MultiEvents) {
  InitTestFrames();
  LoweringGlobalData global_data;

  // prepare stream num in init for l2 allocator
  int64_t stream_num = 3;
  auto init_out = FrameSelector::OnInitRoot([&stream_num, &global_data]()-> std::vector<ValueHolderPtr> {
    auto stream_num_holder = ValueHolder::CreateConst(&stream_num, sizeof(stream_num));
    global_data.SetUniqueValueHolder(kGlobalDataModelStreamNum, stream_num_holder);
    return {};
  });

  // prepare rtStreams
  global_data.LoweringAndSplitRtStreams(stream_num);

  // prepare rtEvents
  auto rt_events_holder = ValueHolder::CreateFeed(1);
  global_data.SetUniqueValueHolder(bg::kGlobalDataRtEvents, rt_events_holder);

  // prepare GertEvents
  auto gert_events_holder = ValueHolder::CreateFeed(2);
  global_data.SetUniqueValueHolder(bg::kGlobalDataGertEvents, gert_events_holder);

  std::vector<int64_t> event_id_list = {0, 1};
  auto order_holder = bg::SendEvents(0, event_id_list, global_data);
  EXPECT_NE(order_holder, nullptr);

  auto main_frame = ValueHolder::PopGraphFrame();
  auto init_exe_graph = init_frame->GetExecuteGraph().get();
  // GE_DUMP(init_exe_graph, "send_event_init");
  EXPECT_EQ(ExeGraphSummaryChecker(init_exe_graph)
                .StrictDirectNodeTypes(std::map<std::string, size_t>{
                    {"Const", 3}, {"CreateL1Allocator", 1}, {"CreateL2Allocators", 1}, {"InnerNetOutput", 1}}),
            "success");

  auto main_exe_graph = main_frame->GetExecuteGraph().get();
  // GE_DUMP(main_exe_graph, "send_event_main");
  EXPECT_EQ(ExeGraphSummaryChecker(main_exe_graph)
                .DirectNodeTypesOnlyCareAbout(std::map<std::string, size_t>{
                    {"Data", 3}, {"InnerData", 2}, {"SplitRtStreams", 1}, {"SelectL2Allocator", 1}, {"SendEvents", 1}}),
            "success");
  FastNodeTopoChecker checker(order_holder);
  // Const(logic_stream_id), GetRtStreamById, Const(event_ids), Data(gert events), Data(rt_events),SelectL2Allocator
  EXPECT_EQ(checker.StrictConnectFrom(std::vector<FastSrcNode>({{"SplitRtStreams"}, {"Const"}, {"Data"}, {"Data"}, {"SelectL2Allocator"}}), true), "success");

  // check event_id_list const value is ok
  auto event_ids = order_holder->GetFastNode()->GetInDataEdgeByIndex(1)->src;
  EXPECT_NE(event_ids, nullptr);
  ge::Buffer buffer;
  ASSERT_TRUE(ge::AttrUtils::GetZeroCopyBytes(event_ids->GetOpDescBarePtr(), "value", buffer));

  size_t total_size = 0U;
  auto event_id_vec_holder = ContinuousVector::Create<int64_t>(event_id_list.size(), total_size);
  ASSERT_EQ(buffer.GetSize(), total_size);
  auto event_id_vec = reinterpret_cast<ContinuousVector *>(buffer.GetData());
  auto event_id_ptr = reinterpret_cast<int64_t *>(event_id_vec->MutableData());
  for (size_t i = 0U; i < event_id_list.size(); ++i) {
    EXPECT_EQ(event_id_ptr[i], event_id_list[i]);
  }
}
TEST_F(BgEventUT, CollectAndCreateGertEvents_ok) {
  InitTestFrames();
  LoweringGlobalData global_data;
  ge::ComputeGraphPtr compute_graph = std::make_shared<ge::ComputeGraph>("test");
  std::vector<EventInfo> expect_events_info;
  CreateMultiStreamComputeGraph(compute_graph, expect_events_info);
  auto model_desc_holder = ModelDescHolderFaker().Build(1, 4);

  std::vector<std::vector<EventInfo>> stage_2_sync_events(static_cast<size_t>(SyncEventStage::kStageEnd));
  auto create_gert_events_holder =
      bg::CollectAndCreateGertEvents(compute_graph, model_desc_holder.GetModelDesc(), global_data, stage_2_sync_events);
  EXPECT_NE(create_gert_events_holder, nullptr);
  auto &first_sync_events = stage_2_sync_events[static_cast<size_t>(SyncEventStage::kFirstSyncStage)];
  auto &last_sync_events = stage_2_sync_events[static_cast<size_t>(SyncEventStage::kLastSyncStage)];
  EXPECT_TRUE(first_sync_events.empty());
  EXPECT_TRUE(last_sync_events.empty());

  auto init_exe_graph = init_frame->GetExecuteGraph().get();
  // GE_DUMP(init_exe_graph, "CollectAndCreateGertEvents_ok");
  EXPECT_EQ(ExeGraphSummaryChecker(init_exe_graph)
                .StrictDirectNodeTypes(std::map<std::string, size_t>{{"Const", 1}, {"CreateGertEvents", 1}, {"InnerNetOutput", 1}}),
            "success");
  // check event_infos const value is right
  auto create_gert_events_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_exe_graph, "CreateGertEvents");
  auto events_info = create_gert_events_node->GetInDataNodes().at(0);
  ASSERT_EQ(events_info->GetType(), "Const");
  ge::Buffer buffer;
  ASSERT_TRUE(ge::AttrUtils::GetZeroCopyBytes(events_info->GetOpDescBarePtr(), "value", buffer));
  auto const_value_vec = reinterpret_cast<ContinuousVector *>(buffer.GetData());
  ASSERT_EQ(const_value_vec->GetSize(), expect_events_info.size());
  for (size_t i = 0U; i < const_value_vec->GetSize(); ++i) {
    auto event_info = reinterpret_cast<const EventInfo *>(const_value_vec->GetData())[i];
    auto expect_event_info = expect_events_info[i];
    EXPECT_EQ(event_info.src_logic_stream_id, expect_event_info.src_logic_stream_id);
    EXPECT_EQ(event_info.dst_logic_stream_id, expect_event_info.dst_logic_stream_id);
  }
}

TEST_F(BgEventUT, LoweringFirstEventSync_ok) {
  InitTestFrames();
  LoweringGlobalData global_data;
  ge::ComputeGraphPtr compute_graph = std::make_shared<ge::ComputeGraph>("test");
  CreateFirstEventSyncComputeGraph(compute_graph);

  // prepare stream num in init for l2 allocator
  int64_t stream_num = 2;
  auto init_out = FrameSelector::OnInitRoot([&stream_num, &global_data]() -> std::vector<ValueHolderPtr> {
    auto stream_num_holder = ValueHolder::CreateConst(&stream_num, sizeof(stream_num));
    global_data.SetUniqueValueHolder(kGlobalDataModelStreamNum, stream_num_holder);
    return {};
  });

  // prepare rtStreams
  global_data.LoweringAndSplitRtStreams(stream_num);

  // prepare rtEvents
  auto rt_events_holder = ValueHolder::CreateFeed(1);
  global_data.SetUniqueValueHolder(bg::kGlobalDataRtEvents, rt_events_holder);

  // prepare GertEvents
  auto gert_events_holder = ValueHolder::CreateFeed(2);
  global_data.SetUniqueValueHolder(bg::kGlobalDataGertEvents, gert_events_holder);

  auto model_desc_holder = ModelDescHolderFaker().Build(1, 1);

  std::vector<std::vector<EventInfo>> stage_2_sync_events(static_cast<size_t>(SyncEventStage::kStageEnd));
  auto create_gert_events_holder =
      bg::CollectAndCreateGertEvents(compute_graph, model_desc_holder.GetModelDesc(), global_data, stage_2_sync_events);
  EXPECT_NE(create_gert_events_holder, nullptr);
  auto &first_sync_events = stage_2_sync_events[static_cast<size_t>(SyncEventStage::kFirstSyncStage)];
  auto &last_sync_events = stage_2_sync_events[static_cast<size_t>(SyncEventStage::kLastSyncStage)];
  EXPECT_EQ(first_sync_events.size(), 1);
  EXPECT_TRUE(last_sync_events.empty());
  bg::LoweringFirstSyncEvents(first_sync_events, model_desc_holder.GetModelDesc().GetReusableEventNum(), global_data);

  auto init_exe_graph = init_frame->GetExecuteGraph().get();
  // GE_DUMP(init_exe_graph, "LoweringFirstEventSync_init");
  EXPECT_EQ(ExeGraphSummaryChecker(init_exe_graph)
                .StrictDirectNodeTypes(std::map<std::string, size_t>{{"Const", 4},
                                                                     {"CreateGertEvents", 1},
                                                                     {"CreateL1Allocator", 1},
                                                                     {"CreateL2Allocators", 1},
                                                                     {"InnerNetOutput", 1}}),
            "success");

  auto main_frame = ValueHolder::PopGraphFrame();
  auto main_exe_graph = main_frame->GetExecuteGraph().get();
  // GE_DUMP(main_exe_graph, "LoweringFirstEventSync");
  EXPECT_EQ(ExeGraphSummaryChecker(main_exe_graph)
                .StrictDirectNodeTypes(std::map<std::string, size_t>{{"Const", 3},
                                                                     {"Data", 3},
                                                                     {"InnerData", 2},
                                                                     {"PartitionedCall", 1},
                                                                     {"SplitRtStreams", 1},
                                                                     {"SelectL2Allocator", 2}}),
            "success");

  auto stage_ids_to_pcall =
      main_frame->GetExecuteGraph()->GetExtAttr<std::vector<bg::ValueHolderPtr>>(kStageIdsToFirstPartitionedCall);
  EXPECT_EQ(stage_ids_to_pcall->size(), static_cast<size_t>(OnMainRootFirstExecStage::kStageSize));
  auto first_sync_pcall = stage_ids_to_pcall->at(static_cast<size_t>(OnMainRootFirstExecStage::kFirstEventSyncStage));

  FastNodeTopoChecker checker(first_sync_pcall);
  EXPECT_EQ(checker.StrictConnectFrom(std::vector<FastSrcNode>({{"SplitRtStreams"},       // rt stream of send
                                                            {"Data"},                 // gert events
                                                            {"Data"},                 // rt events
                                                            {"SelectL2Allocator"},    // l2 allocator of send
                                                            {"SplitRtStreams"},       // rt stream of wait
                                                            {"SelectL2Allocator"}}),  // l2 allocator of wait
                                      true),
            "success");

  auto pcall_subgraph = ge::FastNodeUtils::GetSubgraphFromNode(first_sync_pcall->GetFastNode(), 0);
  // GE_DUMP(pcall_subgraph, "LoweringFirstEventSync_pcall_subgraph");
  EXPECT_EQ(ExeGraphSummaryChecker(pcall_subgraph)
                .StrictDirectNodeTypes(std::map<std::string, size_t>{
                    {"Const", 2}, {"InnerData", 6}, {"SendEvents", 1}, {"WaitEvents", 1}, {"InnerNetOutput", 1}}),
            "success");
}

TEST_F(BgEventUT, LoweringLastEventSync_ok) {
  InitTestFrames();
  LoweringGlobalData global_data;
  ge::ComputeGraphPtr compute_graph = std::make_shared<ge::ComputeGraph>("test");
  CreateLastEventSyncComputeGraph(compute_graph);

  // prepare stream num in init for l2 allocator
  int64_t stream_num = 2;
  auto init_out = FrameSelector::OnInitRoot([&stream_num, &global_data]()-> std::vector<ValueHolderPtr> {
    auto stream_num_holder = ValueHolder::CreateConst(&stream_num, sizeof(stream_num));
    global_data.SetUniqueValueHolder(kGlobalDataModelStreamNum, stream_num_holder);
    return {};
  });

  // prepare rtStreams
  global_data.LoweringAndSplitRtStreams(stream_num);

  // prepare rtEvents
  auto rt_events_holder = ValueHolder::CreateFeed(1);
  global_data.SetUniqueValueHolder(bg::kGlobalDataRtEvents, rt_events_holder);

  // prepare GertEvents
  auto gert_events_holder = ValueHolder::CreateFeed(2);
  global_data.SetUniqueValueHolder(bg::kGlobalDataGertEvents, gert_events_holder);

  auto model_desc_holder = ModelDescHolderFaker().Build(2, 2);

  std::vector<std::vector<EventInfo>> stage_2_sync_events(static_cast<size_t>(SyncEventStage::kStageEnd));
  auto create_gert_events_holder =
      bg::CollectAndCreateGertEvents(compute_graph, model_desc_holder.GetModelDesc(), global_data, stage_2_sync_events);
  EXPECT_NE(create_gert_events_holder, nullptr);
  auto &first_sync_events = stage_2_sync_events[static_cast<size_t>(SyncEventStage::kFirstSyncStage)];
  auto &last_sync_events = stage_2_sync_events[static_cast<size_t>(SyncEventStage::kLastSyncStage)];
  auto &last_resource_claen_events = stage_2_sync_events[static_cast<size_t>(SyncEventStage::kLastResourceCleanStage)];
  EXPECT_EQ(last_sync_events.size(), 2);
  EXPECT_EQ(last_resource_claen_events.size(), 1);
  EXPECT_TRUE(first_sync_events.empty());
  bg::LoweringFirstSyncEvents(first_sync_events, model_desc_holder.GetModelDesc().GetReusableEventNum(), global_data);
  bg::LoweringLastSyncEvents(last_sync_events, model_desc_holder.GetModelDesc().GetReusableEventNum(), global_data);

  auto init_exe_graph = init_frame->GetExecuteGraph().get();
  // GE_DUMP(init_exe_graph, "LoweringLastEventSync_init");
  EXPECT_EQ(ExeGraphSummaryChecker(init_exe_graph)
                .StrictDirectNodeTypes(std::map<std::string, size_t>{{"Const", 4},
                                                                     {"CreateGertEvents", 1},
                                                                     {"CreateL1Allocator", 1},
                                                                     {"CreateL2Allocators", 1},
                                                                     {"InnerNetOutput", 1}}),
            "success");

  auto main_frame = ValueHolder::PopGraphFrame();
  auto main_exe_graph = main_frame->GetExecuteGraph().get();
  // GE_DUMP(main_exe_graph, "LoweringLastEventSync_main");
  EXPECT_EQ(ExeGraphSummaryChecker(main_exe_graph)
                .StrictDirectNodeTypes(std::map<std::string, size_t>{{"Const", 3},
                                                                     {"Data", 3},
                                                                     {"InnerData", 2},
                                                                     {"SplitRtStreams", 1},
                                                                     {"PartitionedCall", 1},
                                                                     {"SelectL2Allocator", 2}}),
            "success");

  auto stage_ids_to_first_pcall = main_frame->GetExecuteGraph()->GetExtAttr<std::vector<bg::ValueHolderPtr>>(kStageIdsToFirstPartitionedCall);
  EXPECT_EQ(stage_ids_to_first_pcall, nullptr);

  auto stage_ids_to_last_pcall = main_frame->GetExecuteGraph()->GetExtAttr<std::vector<bg::ValueHolderPtr>>(kStageIdsToLastPartitionedCall);
  EXPECT_EQ(stage_ids_to_last_pcall->size(), static_cast<size_t>(OnMainRootLastExecStage::kStageSize));
  auto last_sync_pcall = stage_ids_to_last_pcall->at(static_cast<size_t>(OnMainRootLastExecStage::kLastEventSyncStage));
  EXPECT_NE(last_sync_pcall, nullptr);
  FastNodeTopoChecker checker(last_sync_pcall);
  EXPECT_EQ(checker.StrictConnectFrom(std::vector<FastSrcNode>({{"SplitRtStreams"}, // rt stream of send
                                                            {"Data"}, // (gert events)
                                                            {"Data"}, // (rt_events)
                                                            {"SelectL2Allocator"}, // l2 allocator of send
                                                            {"SplitRtStreams"}, // rt stream of wait
                                                            {"SelectL2Allocator"}}), // l2 allocator of wait
                                      true),
            "success");

  auto pcall_subgraph = ge::FastNodeUtils::GetSubgraphFromNode(last_sync_pcall->GetFastNode(), 0);
  ge::DumpGraph(pcall_subgraph, "LoweringLastEventSync_pcall");
  EXPECT_EQ(ExeGraphSummaryChecker(pcall_subgraph)
                .StrictDirectNodeTypes(std::map<std::string, size_t>{
                    {"Const", 3}, {"InnerData", 6}, {"SendEvents", 2}, {"WaitEvents", 1}, {"InnerNetOutput", 1}}),
            "success");
}
}  // namespace bg
}  // namespace gert
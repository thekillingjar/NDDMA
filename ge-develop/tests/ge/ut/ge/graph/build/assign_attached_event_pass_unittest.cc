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
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/utils/graph_utils_ex.h"

#include "macro_utils/dt_public_scope.h"
#include "graph/build/stream/logical_stream_allocator.h"
#include "graph/build/stream/stream_allocator.h"
#include "graph/build/stream/assign_attached_event_pass.h"
#include "macro_utils/dt_public_unscope.h"
#include "graph/debug/ge_attr_define.h"

using namespace std;

namespace ge {
class UtestAssignAttachedEventPass : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

namespace {
ComputeGraphPtr MakeGraph() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", DATA)->NODE("relu1", RELU)->NODE("netoutput", NETOUTPUT));
    CHAIN(NODE("data2", DATA)->NODE("relu2", RELU)->NODE("netoutput", NETOUTPUT));
    CHAIN(NODE("data3", DATA)->NODE("relu3", RELU)->NODE("netoutput", NETOUTPUT));
    CHAIN(NODE("data4", DATA)->NODE("relu4", RELU)->NODE("netoutput", NETOUTPUT));
  };
  auto graph1 = ToGeGraph(g1);
  auto graph = ge::GraphUtilsEx::GetComputeGraph(graph1);
  graph->TopologicalSortingGraph();
  return graph;
}

/**
 * 图中节点设置了attached event，节点均未设置sync res key，不复用event
 * @return
 */
ComputeGraphPtr CreateGraph0() {
  const auto &graph = MakeGraph();
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);
  NamedAttrs named_attrs;
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, "_attached_sync_res_type", "event"));
  EXPECT_TRUE(AttrUtils::SetListNamedAttrs(relu1->GetOpDesc(), "_attached_sync_res_info", {named_attrs}));
  EXPECT_TRUE(AttrUtils::SetListNamedAttrs(relu2->GetOpDesc(), "_attached_sync_res_info", {named_attrs}));
  return graph;
}

/**
 * 图中节点设置了attached event，节点设置了相同的sync res key
 * @return
 */
ComputeGraphPtr CreateGraph1() {
  const auto &graph = MakeGraph();
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);
  NamedAttrs named_attrs;
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, "_attached_sync_res_type", "event"));
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, "_attached_sync_res_key", "res_key"));
  EXPECT_TRUE(AttrUtils::SetListNamedAttrs(relu1->GetOpDesc(), "_attached_sync_res_info", {named_attrs}));
  EXPECT_TRUE(AttrUtils::SetListNamedAttrs(relu2->GetOpDesc(), "_attached_sync_res_info", {named_attrs}));
  return graph;
}

/**
 * 包含子图，子图中节点设置了attached event，节点设置了相同的sync res key
 * @return
 */
ComputeGraphPtr CreateGraph1WithSubgraphs() {
  auto sub_graph = MakeGraph();
  auto relu1 = sub_graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = sub_graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);
  NamedAttrs named_attrs;
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, "_attached_sync_res_type", "event"));
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, "_attached_sync_res_key", "res_key"));
  EXPECT_TRUE(AttrUtils::SetListNamedAttrs(relu1->GetOpDesc(), "_attached_sync_res_info", {named_attrs}));
  EXPECT_TRUE(AttrUtils::SetListNamedAttrs(relu2->GetOpDesc(), "_attached_sync_res_info", {named_attrs}));
  sub_graph->SetName("g1");

  ComputeGraphPtr root_graph;
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", DATA)->NODE("partitioncall_0", PARTITIONEDCALL)->NODE("netoutput", NETOUTPUT));
  };
  auto graph1 = ToGeGraph(g1);
  root_graph = ge::GraphUtilsEx::GetComputeGraph(graph1);

  auto node_with_subgraph = root_graph->FindNode("partitioncall_0");
  EXPECT_NE(node_with_subgraph, nullptr);
  node_with_subgraph->GetOpDesc()->AddSubgraphName(sub_graph->GetName());
  node_with_subgraph->GetOpDesc()->SetSubgraphInstanceName(0, sub_graph->GetName());
  sub_graph->SetParentNode(node_with_subgraph);
  sub_graph->SetParentGraph(root_graph);
  root_graph->AddSubGraph(sub_graph);
  EXPECT_EQ(root_graph->TopologicalSorting(), GRAPH_SUCCESS);
  return root_graph;
}

/**
 * 图中节点设置了attached event，节点设置了不同的sync res key
 * @return
 */
ComputeGraphPtr CreateGraph2() {
  const auto &graph = MakeGraph();
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);
  NamedAttrs named_attrs;
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, "_attached_sync_res_type", "event"));
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, "_attached_sync_res_key", "res_key0"));
  EXPECT_TRUE(AttrUtils::SetListNamedAttrs(relu1->GetOpDesc(), "_attached_sync_res_info", {named_attrs}));

  NamedAttrs named_attrs1;
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs1, "_attached_sync_res_type", "event"));
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs1, "_attached_sync_res_key", "res_key1"));
  EXPECT_TRUE(AttrUtils::SetListNamedAttrs(relu2->GetOpDesc(), "_attached_sync_res_info", {named_attrs1}));
  return graph;
}

/**
 * 图中一个节点设置了attached res type，但不是event；一个节点未设置attach res info
 * @return
 */
ComputeGraphPtr CreateGraph3() {
  const auto &graph = MakeGraph();
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  NamedAttrs named_attrs;
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, "_attached_sync_res_type", "notify"));
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, "_attached_sync_res_key", "res_key0"));
  EXPECT_TRUE(AttrUtils::SetListNamedAttrs(relu1->GetOpDesc(), "_attached_sync_res_info", {named_attrs}));
  return graph;
}

/**
 * 图中节点设置了attached event，包含多个属性
 * @return
 */
ComputeGraphPtr CreateGraph4() {
  const auto &graph = MakeGraph();
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);
  NamedAttrs named_attrs;
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, "_attached_sync_res_type", "event"));
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, "_attached_sync_res_key", "res_key0"));
  NamedAttrs named_attrs1;
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs1, "_attached_sync_res_type", "event"));
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs1, "_attached_sync_res_key", "res_key1"));
  EXPECT_TRUE(AttrUtils::SetListNamedAttrs(relu1->GetOpDesc(), "_attached_sync_res_info", {named_attrs, named_attrs1}));

  NamedAttrs named_attrs2;
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs2, "_attached_sync_res_type", "event"));
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs2, "_attached_sync_res_key", "res_key2"));
  EXPECT_TRUE(AttrUtils::SetListNamedAttrs(relu2->GetOpDesc(), "_attached_sync_res_info", {named_attrs, named_attrs2}));
  return graph;
}

/**
 * 图中节点设置了attached event，节点均未设置sync res key，不复用event
 * @return
 */
ComputeGraphPtr CreateGraph0_V2() {
  const auto &graph = MakeGraph();
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);

  ge::GeAttrValue::NAMED_ATTRS attached_event;
  (void)ge::AttrUtils::SetStr(attached_event, ATTR_NAME_ATTACHED_RESOURCE_NAME, "group0");
  (void)ge::AttrUtils::SetInt(attached_event, ge::ATTR_NAME_ATTACHED_RESOURCE_TYPE,
                              static_cast<int64_t>(SyncResType::kSyncResEvent));
  (void)ge::AttrUtils::SetBool(attached_event, ge::ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG, true);
  std::vector<NamedAttrs> attached_sync_res_info_list_from_attr;
  attached_sync_res_info_list_from_attr.emplace_back(attached_event);

  (void)ge::AttrUtils::SetListNamedAttrs(relu1->GetOpDesc(), ge::ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST,
                                         attached_sync_res_info_list_from_attr);
  (void)ge::AttrUtils::SetListNamedAttrs(relu2->GetOpDesc(), ge::ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST,
                                         attached_sync_res_info_list_from_attr);
  return graph;
}

/**
 * 图中节点设置了attached event，节点设置了相同的sync res key
 * @return
 */
ComputeGraphPtr CreateGraph1_V2() {
  const auto &graph = MakeGraph();
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);
  NamedAttrs named_attrs;
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, ATTR_NAME_ATTACHED_RESOURCE_NAME, "event"));
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, "res_key"));
  EXPECT_TRUE(AttrUtils::SetInt(named_attrs, ATTR_NAME_ATTACHED_RESOURCE_TYPE,
                                static_cast<int64_t>(SyncResType::kSyncResEvent)));
  EXPECT_TRUE(AttrUtils::SetBool(named_attrs, ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG, true));

  EXPECT_TRUE(AttrUtils::SetListNamedAttrs(relu1->GetOpDesc(), ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST, {named_attrs}));
  EXPECT_TRUE(AttrUtils::SetListNamedAttrs(relu2->GetOpDesc(), ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST, {named_attrs}));
  return graph;
}

/**
 * 包含子图，子图中节点设置了attached event，节点设置了相同的sync res key
 * @return
 */
ComputeGraphPtr CreateGraph1WithSubgraphs_V2() {
  auto sub_graph = MakeGraph();
  auto relu1 = sub_graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = sub_graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);
  NamedAttrs named_attrs;
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, ATTR_NAME_ATTACHED_RESOURCE_NAME, "event"));
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, "res_key"));
  EXPECT_TRUE(AttrUtils::SetInt(named_attrs, ATTR_NAME_ATTACHED_RESOURCE_TYPE,
                                static_cast<int64_t>(SyncResType::kSyncResEvent)));
  EXPECT_TRUE(AttrUtils::SetBool(named_attrs, ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG, true));
  EXPECT_TRUE(AttrUtils::SetListNamedAttrs(relu1->GetOpDesc(), ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST, {named_attrs}));
  EXPECT_TRUE(AttrUtils::SetListNamedAttrs(relu2->GetOpDesc(), ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST, {named_attrs}));
  sub_graph->SetName("g1");

  ComputeGraphPtr root_graph;
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", DATA)->NODE("partitioncall_0", PARTITIONEDCALL)->NODE("netoutput", NETOUTPUT));
  };
  auto graph1 = ToGeGraph(g1);
  root_graph = ge::GraphUtilsEx::GetComputeGraph(graph1);

  auto node_with_subgraph = root_graph->FindNode("partitioncall_0");
  EXPECT_NE(node_with_subgraph, nullptr);
  node_with_subgraph->GetOpDesc()->AddSubgraphName(sub_graph->GetName());
  node_with_subgraph->GetOpDesc()->SetSubgraphInstanceName(0, sub_graph->GetName());
  sub_graph->SetParentNode(node_with_subgraph);
  sub_graph->SetParentGraph(root_graph);
  root_graph->AddSubGraph(sub_graph);
  EXPECT_EQ(root_graph->TopologicalSorting(), GRAPH_SUCCESS);
  return root_graph;
}

/**
 * 图中节点设置了attached event，节点设置了不同的sync res key
 * @return
 */
ComputeGraphPtr CreateGraph2_V2() {
  const auto &graph = MakeGraph();
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);

  NamedAttrs named_attrs;
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, ATTR_NAME_ATTACHED_RESOURCE_NAME, "event"));
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, "res_key0"));
  EXPECT_TRUE(AttrUtils::SetInt(named_attrs, ATTR_NAME_ATTACHED_RESOURCE_TYPE,
                                static_cast<int64_t>(SyncResType::kSyncResEvent)));
  EXPECT_TRUE(AttrUtils::SetBool(named_attrs, ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG, true));
  EXPECT_TRUE(AttrUtils::SetListNamedAttrs(relu1->GetOpDesc(), ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST, {named_attrs}));

  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, "res_key1"));
  EXPECT_TRUE(AttrUtils::SetListNamedAttrs(relu2->GetOpDesc(), ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST, {named_attrs}));
  return graph;
}

/**
 * 图中一个节点设置了attached res type，但不是event；一个节点未设置attach res info
 * @return
 */
ComputeGraphPtr CreateGraph3_V2() {
  const auto &graph = MakeGraph();
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);

  NamedAttrs named_attrs;
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, ATTR_NAME_ATTACHED_RESOURCE_NAME, "notify"));
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, "res_key0"));
  EXPECT_TRUE(AttrUtils::SetInt(named_attrs, ATTR_NAME_ATTACHED_RESOURCE_TYPE,
                                static_cast<int64_t>(SyncResType::kSyncResNotify)));
  EXPECT_TRUE(AttrUtils::SetBool(named_attrs, ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG, true));
  EXPECT_TRUE(AttrUtils::SetListNamedAttrs(relu1->GetOpDesc(), ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST, {named_attrs}));
  return graph;
}

/**
 * 图中节点设置了attached event，包含多个属性
 * @return
 */
ComputeGraphPtr CreateGraph4_V2() {
  const auto &graph = MakeGraph();
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);

  NamedAttrs named_attrs;
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, ATTR_NAME_ATTACHED_RESOURCE_NAME, "event"));
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, "res_key0"));
  EXPECT_TRUE(AttrUtils::SetInt(named_attrs, ATTR_NAME_ATTACHED_RESOURCE_TYPE,
                                static_cast<int64_t>(SyncResType::kSyncResEvent)));
  EXPECT_TRUE(AttrUtils::SetBool(named_attrs, ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG, true));
  EXPECT_TRUE(AttrUtils::SetListNamedAttrs(relu1->GetOpDesc(), ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST, {named_attrs}));

  NamedAttrs named_attrs1;
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs1, ATTR_NAME_ATTACHED_RESOURCE_NAME, "event"));
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs1, ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, "res_key1"));
  EXPECT_TRUE(AttrUtils::SetInt(named_attrs1, ATTR_NAME_ATTACHED_RESOURCE_TYPE,
                                static_cast<int64_t>(SyncResType::kSyncResEvent)));
  EXPECT_TRUE(AttrUtils::SetBool(named_attrs1, ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG, true));

  NamedAttrs named_attrs2;
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs2, ATTR_NAME_ATTACHED_RESOURCE_NAME, "event"));
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs2, ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, "res_key2"));
  EXPECT_TRUE(AttrUtils::SetInt(named_attrs2, ATTR_NAME_ATTACHED_RESOURCE_TYPE,
                                static_cast<int64_t>(SyncResType::kSyncResEvent)));
  EXPECT_TRUE(AttrUtils::SetBool(named_attrs2, ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG, true));

  EXPECT_TRUE(AttrUtils::SetListNamedAttrs(relu1->GetOpDesc(), ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST,
                                           {named_attrs, named_attrs1}));
  EXPECT_TRUE(AttrUtils::SetListNamedAttrs(relu2->GetOpDesc(), ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST,
                                           {named_attrs, named_attrs2}));
  return graph;
}


/**
 * 适配V2场景，属性名字发生变化，采用V2版本表示新的属性方式;
 * V1 V2场景均存在，需要兼容处理
 * @return
 */
ComputeGraphPtr CreateGraph5_V2() {
  const auto &graph = MakeGraph();
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);

  auto relu3 = graph->FindNode("relu3");
  EXPECT_NE(relu3, nullptr);
  auto relu4 = graph->FindNode("relu4");
  EXPECT_NE(relu4, nullptr);

  ge::GeAttrValue::NAMED_ATTRS attached_event;
  (void)ge::AttrUtils::SetStr(attached_event, ATTR_NAME_ATTACHED_RESOURCE_NAME, "group0");
  (void)ge::AttrUtils::SetStr(attached_event, ge::ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, "as0");
  (void)ge::AttrUtils::SetInt(attached_event, ge::ATTR_NAME_ATTACHED_RESOURCE_TYPE,
                              static_cast<int64_t>(SyncResType::kSyncResEvent));
  (void)ge::AttrUtils::SetBool(attached_event, ge::ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG, true);
  (void)ge::AttrUtils::SetListNamedAttrs(relu1->GetOpDesc(), ge::ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST,
                                         {attached_event});
  (void)ge::AttrUtils::SetListNamedAttrs(relu2->GetOpDesc(), ge::ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST,
                                         {attached_event});

  NamedAttrs named_attrs;
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, "_attached_sync_res_type", "event"));
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, "_attached_sync_res_key", "res_key0"));
  EXPECT_TRUE(AttrUtils::SetListNamedAttrs(relu3->GetOpDesc(), "_attached_sync_res_info", {named_attrs}));
  EXPECT_TRUE(AttrUtils::SetListNamedAttrs(relu4->GetOpDesc(), "_attached_sync_res_info", {named_attrs}));
  return graph;
}

// 提供一个从属性中获取event_id列表的接口
std::vector<int64_t> GetAttachedEventIds_V2(const NodePtr node) {
  std::vector<NamedAttrs> attached_sync_res_info_list_from_attr;
  AttrUtils::GetListNamedAttrs(node->GetOpDescBarePtr(), ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST,
                               attached_sync_res_info_list_from_attr);
  std::vector<int64_t> event_ids;
  for (const auto &attr : attached_sync_res_info_list_from_attr) {
    int64_t event_id = -1;
    (void)AttrUtils::GetInt(attr, ATTR_NAME_ATTACHED_RESOURCE_ID, event_id);
    event_ids.emplace_back(event_id);
  }
  return event_ids;
}

}  // namespace

TEST_F(UtestAssignAttachedEventPass, Case0) {
  const auto graph = CreateGraph0();
  const auto &assign_attached_event_pass = MakeShared<AssignAttachedEventPass>();
  EXPECT_NE(assign_attached_event_pass, nullptr);
  uint32_t event_num = 0U;
  EXPECT_EQ((assign_attached_event_pass->Run(graph, event_num)), SUCCESS);

  // check 分配了attach event的节点才会分从event
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);
  // relu1,relu2 分配了不一样的attached event id
  std::vector<int64_t> event_id1;
  std::vector<int64_t> event_id2;
  EXPECT_TRUE(AttrUtils::GetListInt(relu1->GetOpDescBarePtr(), RECV_ATTR_EVENT_ID, event_id1));
  EXPECT_TRUE(AttrUtils::GetListInt(relu2->GetOpDescBarePtr(), RECV_ATTR_EVENT_ID, event_id2));
  EXPECT_NE(event_id1[0], event_id2[0]);
  // 总的event个数为2
  EXPECT_EQ(event_num, 2);
}

TEST_F(UtestAssignAttachedEventPass, Case1) {
  const auto graph = CreateGraph1();
  const auto &assign_attached_event_pass = MakeShared<AssignAttachedEventPass>();
  EXPECT_NE(assign_attached_event_pass, nullptr);
  uint32_t event_num = 0U;
  EXPECT_EQ((assign_attached_event_pass->Run(graph, event_num)), SUCCESS);

  // check 分配了attach event的节点才会分从event
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);
  // relu1,relu2 分配了一样的attached event id
  std::vector<int64_t> event_id1;
  std::vector<int64_t> event_id2;
  EXPECT_TRUE(AttrUtils::GetListInt(relu1->GetOpDescBarePtr(), RECV_ATTR_EVENT_ID, event_id1));
  EXPECT_TRUE(AttrUtils::GetListInt(relu2->GetOpDescBarePtr(), RECV_ATTR_EVENT_ID, event_id2));
  EXPECT_EQ(event_id1, event_id2);
  EXPECT_EQ(event_id1[0], 0U);
  // 总的event个数为1
  EXPECT_EQ(event_num, 1);
}

TEST_F(UtestAssignAttachedEventPass, Case1_With_Subgraph) {
  const auto graph = CreateGraph1WithSubgraphs();
  auto parent_node = graph->FindNode("partitioncall_0");
  EXPECT_NE(parent_node, nullptr);
  auto sub_graph = ge::NodeUtils::GetSubgraph(*parent_node, 0U);
  EXPECT_NE(sub_graph, nullptr);

  Graph2SubGraphInfoList graph_2_sub_graph_info_list{{graph, {}}, {sub_graph, {}}};
  const auto &allocator = MakeShared<StreamAllocator>(graph, graph_2_sub_graph_info_list);
  EXPECT_EQ((allocator->AssignAttachedEventResource()), SUCCESS);

  // check 分配了attach event的节点才会分从event
  auto relu1 = sub_graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = sub_graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);
  // relu1,relu2 分配了一样的attached event id
  std::vector<int64_t> event_id1;
  std::vector<int64_t> event_id2;
  EXPECT_TRUE(AttrUtils::GetListInt(relu1->GetOpDescBarePtr(), RECV_ATTR_EVENT_ID, event_id1));
  EXPECT_TRUE(AttrUtils::GetListInt(relu2->GetOpDescBarePtr(), RECV_ATTR_EVENT_ID, event_id2));
  EXPECT_EQ(event_id1, event_id2);
  EXPECT_EQ(event_id1[0], 0U);
}

TEST_F(UtestAssignAttachedEventPass, Case2) {
  const auto graph = CreateGraph2();
  const auto &assign_attached_event_pass = MakeShared<AssignAttachedEventPass>();
  EXPECT_NE(assign_attached_event_pass, nullptr);
  uint32_t event_num = 2U;
  EXPECT_EQ((assign_attached_event_pass->Run(graph, event_num)), SUCCESS);

  // check 分配了attach event的节点才会分从event
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);
  // relu1,relu2 分配了不一样的attached event id
  std::vector<int64_t> event_id1;
  std::vector<int64_t> event_id2;
  EXPECT_TRUE(AttrUtils::GetListInt(relu1->GetOpDescBarePtr(), RECV_ATTR_EVENT_ID, event_id1));
  EXPECT_TRUE(AttrUtils::GetListInt(relu2->GetOpDescBarePtr(), RECV_ATTR_EVENT_ID, event_id2));
  EXPECT_NE(event_id1[0], event_id2[0]);
  // 总的event个数为2 + 2
  EXPECT_EQ(event_num, 2 + 2);
}

TEST_F(UtestAssignAttachedEventPass, Case3) {
  const auto graph = CreateGraph3();
  const auto &assign_attached_event_pass = MakeShared<AssignAttachedEventPass>();
  EXPECT_NE(assign_attached_event_pass, nullptr);
  uint32_t event_num = 0U;
  EXPECT_EQ((assign_attached_event_pass->Run(graph, event_num)), SUCCESS);

  // check 分配了attach event的节点才会分从event
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);
  // relu1,relu2 没有分配attached event id
  std::vector<int64_t> event_id1;
  std::vector<int64_t> event_id2;
  EXPECT_FALSE(AttrUtils::GetListInt(relu1->GetOpDescBarePtr(), RECV_ATTR_EVENT_ID, event_id1));
  EXPECT_FALSE(AttrUtils::GetListInt(relu2->GetOpDescBarePtr(), RECV_ATTR_EVENT_ID, event_id2));
  // 总的event个数为0
  EXPECT_EQ(event_num, 0);
}

TEST_F(UtestAssignAttachedEventPass, Case4) {
  const auto graph = CreateGraph4();
  const auto &assign_attached_event_pass = MakeShared<AssignAttachedEventPass>();
  EXPECT_NE(assign_attached_event_pass, nullptr);
  uint32_t event_num = 0U;
  EXPECT_EQ((assign_attached_event_pass->Run(graph, event_num)), SUCCESS);

  // check 分配了attach event的节点才会分从event
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);
  // relu1,relu2 分配了不一样的attached event id
  std::vector<int64_t> event_id1;
  std::vector<int64_t> event_id2;
  EXPECT_TRUE(AttrUtils::GetListInt(relu1->GetOpDescBarePtr(), RECV_ATTR_EVENT_ID, event_id1));
  EXPECT_TRUE(AttrUtils::GetListInt(relu2->GetOpDescBarePtr(), RECV_ATTR_EVENT_ID, event_id2));
  EXPECT_EQ(event_id1.size(), 2U);
  EXPECT_EQ(event_id2.size(), 2U);
  EXPECT_EQ(event_id1[0], event_id2[0]);
  EXPECT_NE(event_id1[1], event_id2[1]);
  // 总的event个数为3
  EXPECT_EQ(event_num, 3);
}

/*
 * 没传reusekey，默认不复用
 * */
TEST_F(UtestAssignAttachedEventPass, Case0_not_reuse_V2) {
  const auto graph = CreateGraph0_V2();
  const auto &assign_attached_event_pass = MakeShared<AssignAttachedEventPass>();
  EXPECT_NE(assign_attached_event_pass, nullptr);
  uint32_t event_num = 0U;
  EXPECT_EQ((assign_attached_event_pass->Run(graph, event_num)), SUCCESS);

  // check 分配了attach event的节点才会分从event
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);

  std::vector<int64_t> event_id1 = GetAttachedEventIds_V2(relu1);
  std::vector<int64_t> event_id2 = GetAttachedEventIds_V2(relu2);
  EXPECT_EQ(event_id1.size(), 1U);
  EXPECT_EQ(event_id2.size(), 1U);

  // relu1,relu2 分配了不一样的attached event id
  EXPECT_NE(event_id1[0], event_id2[0]);
  EXPECT_NE(event_id1[0], -1);

  // 总的event个数为2
  EXPECT_EQ(event_num, 2);

  std::vector<NamedAttrs> attached_sync_res_info_list_from_attr;
  AttrUtils::GetListNamedAttrs(relu1->GetOpDescBarePtr(), ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST,
                               attached_sync_res_info_list_from_attr);
  EXPECT_EQ(attached_sync_res_info_list_from_attr.size(), 1);
  int64_t event_id_relu_1;
  AttrUtils::GetInt(attached_sync_res_info_list_from_attr[0], ATTR_NAME_ATTACHED_RESOURCE_ID, event_id_relu_1);

  attached_sync_res_info_list_from_attr.clear();
  AttrUtils::GetListNamedAttrs(relu2->GetOpDescBarePtr(), ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST,
                               attached_sync_res_info_list_from_attr);
  EXPECT_EQ(attached_sync_res_info_list_from_attr.size(), 1);
  int64_t event_id_relu_2;
  AttrUtils::GetInt(attached_sync_res_info_list_from_attr[0], ATTR_NAME_ATTACHED_RESOURCE_ID, event_id_relu_2);

  // relu1,relu2 分配了不一样的attached event id
  EXPECT_NE(event_id_relu_1, event_id_relu_2);

  // 总的event个数为2
  EXPECT_EQ(event_num, 2);
}

/*
 * 传reusekey，event复用
 * */

TEST_F(UtestAssignAttachedEventPass, Case1_V2) {
  const auto graph = CreateGraph1_V2();
  const auto &assign_attached_event_pass = MakeShared<AssignAttachedEventPass>();
  EXPECT_NE(assign_attached_event_pass, nullptr);
  uint32_t event_num = 0U;
  EXPECT_EQ((assign_attached_event_pass->Run(graph, event_num)), SUCCESS);

  // check 分配了attach event的节点才会分从event
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);

  std::vector<int64_t> event_id1 = GetAttachedEventIds_V2(relu1);
  std::vector<int64_t> event_id2 = GetAttachedEventIds_V2(relu2);
  EXPECT_EQ(event_id1.size(), 1U);
  EXPECT_EQ(event_id2.size(), 1U);

  // relu1,relu2 分配了一样的attached event id
  EXPECT_EQ(event_id1[0], event_id2[0]);
  EXPECT_EQ(event_id1[0], 0U);

  // 总的event个数为1
  EXPECT_EQ(event_num, 1);
}

/*
 * 传reusekey，event复用
 * */
TEST_F(UtestAssignAttachedEventPass, Case1_With_Subgraph_V2) {
  const auto graph = CreateGraph1WithSubgraphs_V2();
  auto parent_node = graph->FindNode("partitioncall_0");
  EXPECT_NE(parent_node, nullptr);
  auto sub_graph = ge::NodeUtils::GetSubgraph(*parent_node, 0U);
  EXPECT_NE(sub_graph, nullptr);

  Graph2SubGraphInfoList graph_2_sub_graph_info_list{{graph, {}}, {sub_graph, {}}};
  const auto &allocator = MakeShared<StreamAllocator>(graph, graph_2_sub_graph_info_list);
  EXPECT_EQ((allocator->AssignAttachedEventResource()), SUCCESS);

  // check 分配了attach event的节点才会分从event
  auto relu1 = sub_graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = sub_graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);

  std::vector<int64_t> event_id1 = GetAttachedEventIds_V2(relu1);
  std::vector<int64_t> event_id2 = GetAttachedEventIds_V2(relu2);
  EXPECT_EQ(event_id1.size(), 1U);
  EXPECT_EQ(event_id2.size(), 1U);

  EXPECT_EQ(event_id1[0], event_id2[0]);
  EXPECT_EQ(event_id1[0], 0U);
}

TEST_F(UtestAssignAttachedEventPass, Case2_V2) {
  const auto graph = CreateGraph2_V2();
  const auto &assign_attached_event_pass = MakeShared<AssignAttachedEventPass>();
  EXPECT_NE(assign_attached_event_pass, nullptr);
  uint32_t event_num = 2U;
  EXPECT_EQ((assign_attached_event_pass->Run(graph, event_num)), SUCCESS);

  // check 分配了attach event的节点才会分从event
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);

  // relu1,relu2 分配了不一样的attached event id
  std::vector<int64_t> event_id1 = GetAttachedEventIds_V2(relu1);
  std::vector<int64_t> event_id2 = GetAttachedEventIds_V2(relu2);
  EXPECT_EQ(event_id1.size(), 1);
  EXPECT_EQ(event_id2.size(), 1);

  EXPECT_NE(event_id1[0], event_id2[0]);
  // 总的event个数为2 + 2 (初始化默认event_num是2)
  EXPECT_EQ(event_num, 2 + 2);
}

TEST_F(UtestAssignAttachedEventPass, Case3_V2) {
  const auto graph = CreateGraph3_V2();
  const auto &assign_attached_event_pass = MakeShared<AssignAttachedEventPass>();
  EXPECT_NE(assign_attached_event_pass, nullptr);
  uint32_t event_num = 0U;
  EXPECT_EQ((assign_attached_event_pass->Run(graph, event_num)), SUCCESS);

  // check 分配了attach event的节点才会分从event
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);

  // relu1,relu2 没有分配attached event id
  std::vector<int64_t> event_id1 = GetAttachedEventIds_V2(relu1);
  std::vector<int64_t> event_id2 = GetAttachedEventIds_V2(relu2);
  EXPECT_EQ(event_id1.size(), 1U);
  EXPECT_EQ(event_id1[0], -1);
  EXPECT_EQ(event_id2.size(), 0U);

  // 总的event个数为0
  EXPECT_EQ(event_num, 0);
}

TEST_F(UtestAssignAttachedEventPass, Case4_V2) {
  const auto graph = CreateGraph4_V2();
  const auto &assign_attached_event_pass = MakeShared<AssignAttachedEventPass>();
  EXPECT_NE(assign_attached_event_pass, nullptr);
  uint32_t event_num = 0U;
  EXPECT_EQ((assign_attached_event_pass->Run(graph, event_num)), SUCCESS);

  // check 分配了attach event的节点才会分从event
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);

  // relu1,relu2 分配了不一样的attached event id
  std::vector<int64_t> event_id1 = GetAttachedEventIds_V2(relu1);
  std::vector<int64_t> event_id2 = GetAttachedEventIds_V2(relu2);
  EXPECT_EQ(event_id1.size(), 2U);
  EXPECT_EQ(event_id2.size(), 2U);
  EXPECT_EQ(event_id1[0], event_id2[0]);
  EXPECT_NE(event_id1[1], event_id2[1]);
  // 总的event个数为3
  EXPECT_EQ(event_num, 3);
}

/*
 * 设置的required_flag是false，不会分event
 * */
TEST_F(UtestAssignAttachedEventPass, Not_Required_V2) {
  const auto graph = CreateGraph1_V2();

  // check 分配了attach event的节点才会分从event
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);

  std::vector<ge::NamedAttrs> named_attr_list;
  EXPECT_TRUE(AttrUtils::GetListNamedAttrs(relu1->GetOpDesc(), ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST, named_attr_list));
  EXPECT_EQ(named_attr_list.size(), 1);
  EXPECT_TRUE(AttrUtils::SetBool(named_attr_list[0], ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG, false));

  EXPECT_TRUE(AttrUtils::SetListNamedAttrs(relu1->GetOpDesc(), ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST, named_attr_list));
  EXPECT_TRUE(AttrUtils::SetListNamedAttrs(relu2->GetOpDesc(), ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST, named_attr_list));

  const auto &assign_attached_event_pass = MakeShared<AssignAttachedEventPass>();
  EXPECT_NE(assign_attached_event_pass, nullptr);
  uint32_t event_num = 0U;
  EXPECT_EQ((assign_attached_event_pass->Run(graph, event_num)), SUCCESS);

  // relu1,relu2 不会分配event_id
  std::vector<int64_t> event_id1 = GetAttachedEventIds_V2(relu1);
  std::vector<int64_t> event_id2 = GetAttachedEventIds_V2(relu2);
  EXPECT_EQ(event_id1.size(), 1U);
  EXPECT_EQ(event_id2.size(), 1U);
  EXPECT_EQ(event_id1[0], -1);
  EXPECT_EQ(event_id2[0], -1);
  // 总的event个数为0
  EXPECT_EQ(event_num, 0);
}

/*
 * 兼容场景，老的v1流程跟新的v2流程均存在
 * relu1、relu2共用一个event，relu3、relu4共用一个event
 * */
TEST_F(UtestAssignAttachedEventPass, Mix_V1_V2) {
  const auto graph = CreateGraph5_V2();
  const auto &assign_attached_event_pass = MakeShared<AssignAttachedEventPass>();
  EXPECT_NE(assign_attached_event_pass, nullptr);
  uint32_t event_num = 0U;
  EXPECT_EQ((assign_attached_event_pass->Run(graph, event_num)), SUCCESS);

  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);

  auto relu3 = graph->FindNode("relu3");
  EXPECT_NE(relu3, nullptr);
  auto relu4 = graph->FindNode("relu4");
  EXPECT_NE(relu4, nullptr);

  auto event_ids1 = GetAttachedEventIds_V2(relu1);
  auto event_ids2 = GetAttachedEventIds_V2(relu2);
  EXPECT_EQ(event_ids1.size(), 1);
  EXPECT_EQ(event_ids2.size(), 1);
  EXPECT_EQ(event_ids1[0], event_ids2[0]);
  EXPECT_EQ(event_ids1[0], 1);

  std::vector<int64_t> event_id3;
  std::vector<int64_t> event_id4;
  EXPECT_TRUE(AttrUtils::GetListInt(relu3->GetOpDescBarePtr(), RECV_ATTR_EVENT_ID, event_id3));
  EXPECT_TRUE(AttrUtils::GetListInt(relu4->GetOpDescBarePtr(), RECV_ATTR_EVENT_ID, event_id4));
  EXPECT_EQ(event_id3.size(), 1U);
  EXPECT_EQ(event_id4.size(), 1U);
  EXPECT_EQ(event_id3[0], event_id4[0]);
  EXPECT_EQ(event_id3[0], 0);

  // relu1、relu2共用一个event，relu3、relu4共用一个event
  EXPECT_EQ(event_num, 2);
}

/*
 * Invalid case
 * 某个op_desc在分attached_resource的pass里面，打了attached_stream属性
 * */

TEST_F(UtestAssignAttachedEventPass, Invalid_Case_V2) {
  const auto graph = CreateGraph1_V2();
  auto relu3 = graph->FindNode("relu3");
  EXPECT_NE(relu3, nullptr);
  std::vector<ge::NamedAttrs> attached_stream_attrs;
  ge::GeAttrValue::NAMED_ATTRS attached_stream;
  (void)ge::AttrUtils::SetStr(attached_stream, ATTR_NAME_ATTACHED_RESOURCE_NAME, "tiling");
  (void)ge::AttrUtils::SetStr(attached_stream, ge::ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, "tiling");
  (void)ge::AttrUtils::SetBool(attached_stream, ge::ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG, true);
  attached_stream_attrs.emplace_back(attached_stream);
  (void)ge::AttrUtils::SetListNamedAttrs(relu3->GetOpDesc(), ge::ATTR_NAME_ATTACHED_STREAM_INFO_LIST,
                                         attached_stream_attrs);

  const auto &assign_attached_event_pass = MakeShared<AssignAttachedEventPass>();
  EXPECT_NE(assign_attached_event_pass, nullptr);
  uint32_t event_num = 0U;
  EXPECT_EQ((assign_attached_event_pass->Run(graph, event_num)), SUCCESS);

  // invalid
  std::vector<int64_t> event_id3 = GetAttachedEventIds_V2(relu3);
  EXPECT_EQ(event_id3.size(), 0U);

  // 总的event个数为1
  EXPECT_EQ(event_num, 1);
}
}  // namespace ge

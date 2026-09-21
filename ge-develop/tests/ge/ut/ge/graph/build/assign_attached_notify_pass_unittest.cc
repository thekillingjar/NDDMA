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
#include "graph/build/stream/assign_attached_notify_pass.h"
#include "graph/build/stream/stream_allocator.h"
#include "macro_utils/dt_public_unscope.h"
#include "graph/debug/ge_attr_define.h"

using namespace std;

namespace ge {
class UtestAssignAttachedNotifyPass : public testing::Test {
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
 * 图中有一个通信域， 通信域中的节点设置了相同的attach notify名称
 * @return
 */
ComputeGraphPtr CreateGraph0() {
  const auto &graph = MakeGraph();
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);
  NamedAttrs named_attrs;
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, ATTR_NAME_ATTACHED_NOTIFY_POLICY, GROUP_POLICY));
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, ATTR_NAME_ATTACHED_NOTIFY_KEY, "as0"));
  EXPECT_TRUE(AttrUtils::SetNamedAttrs(relu1->GetOpDesc(), ATTR_NAME_ATTACHED_NOTIFY_INFO, named_attrs));
  EXPECT_TRUE(AttrUtils::SetNamedAttrs(relu2->GetOpDesc(), ATTR_NAME_ATTACHED_NOTIFY_INFO, named_attrs));
  EXPECT_TRUE(AttrUtils::SetStr(relu1->GetOpDesc(), GROUP_POLICY, "group0"));
  EXPECT_TRUE(AttrUtils::SetStr(relu2->GetOpDesc(), GROUP_POLICY, "group0"));

  return graph;
}

ComputeGraphPtr CreateGraph1WithSubgraphs() {
  auto sub_graph = MakeGraph();
  auto relu1 = sub_graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = sub_graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);
  NamedAttrs named_attrs;
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, ATTR_NAME_ATTACHED_NOTIFY_POLICY, GROUP_POLICY));
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, ATTR_NAME_ATTACHED_NOTIFY_KEY, "as0"));
  EXPECT_TRUE(AttrUtils::SetInt(named_attrs, ATTR_NAME_ATTACHED_NOTIFY_TYPE, 0));
  EXPECT_TRUE(AttrUtils::SetStr(relu1->GetOpDesc(), GROUP_POLICY, "group0"));
  EXPECT_TRUE(AttrUtils::SetStr(relu2->GetOpDesc(), GROUP_POLICY, "group0"));
  EXPECT_TRUE(AttrUtils::SetNamedAttrs(relu1->GetOpDesc(), ATTR_NAME_ATTACHED_NOTIFY_INFO, named_attrs));
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, ATTR_NAME_ATTACHED_NOTIFY_KEY, "as1"));
  EXPECT_TRUE(AttrUtils::SetInt(named_attrs, ATTR_NAME_ATTACHED_NOTIFY_TYPE, 1));
  EXPECT_TRUE(AttrUtils::SetNamedAttrs(relu2->GetOpDesc(), ATTR_NAME_ATTACHED_NOTIFY_INFO, named_attrs));

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
 * 图中有一个通信域， 通信域中的节点设置了不同的attach notify名称
 * @return
 */
ComputeGraphPtr CreateGraph1() {
  const auto &graph = MakeGraph();
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);
  NamedAttrs named_attrs;
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, ATTR_NAME_ATTACHED_NOTIFY_POLICY, GROUP_POLICY));
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, ATTR_NAME_ATTACHED_NOTIFY_KEY, "as0"));
  EXPECT_TRUE(AttrUtils::SetNamedAttrs(relu1->GetOpDesc(), ATTR_NAME_ATTACHED_NOTIFY_INFO, named_attrs));
  NamedAttrs named_attrs2;
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs2, ATTR_NAME_ATTACHED_NOTIFY_POLICY, GROUP_POLICY));
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs2, ATTR_NAME_ATTACHED_NOTIFY_KEY, "as1"));
  EXPECT_TRUE(AttrUtils::SetNamedAttrs(relu2->GetOpDesc(), ATTR_NAME_ATTACHED_NOTIFY_INFO, named_attrs2));

  EXPECT_TRUE(AttrUtils::SetStr(relu1->GetOpDesc(), GROUP_POLICY, "group0"));
  EXPECT_TRUE(AttrUtils::SetStr(relu2->GetOpDesc(), GROUP_POLICY, "group0"));

  return graph;
}

/**
 * 图中有两个通信域， 每个通信域内的节点设置了不同的attach notify名称， 两个通信域的attach notify名称不同
 * @return
 */
ComputeGraphPtr CreateGraph2() {
  const auto &graph = MakeGraph();
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);

  NamedAttrs named_attrs;
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, ATTR_NAME_ATTACHED_NOTIFY_POLICY, GROUP_POLICY));
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, ATTR_NAME_ATTACHED_NOTIFY_KEY, "as0"));
  EXPECT_TRUE(AttrUtils::SetInt(named_attrs, ATTR_NAME_ATTACHED_NOTIFY_TYPE, 0));
  EXPECT_TRUE(AttrUtils::SetStr(relu1->GetOpDesc(), GROUP_POLICY, "group0"));
  EXPECT_TRUE(AttrUtils::SetStr(relu2->GetOpDesc(), GROUP_POLICY, "group0"));
  EXPECT_TRUE(AttrUtils::SetNamedAttrs(relu1->GetOpDesc(), ATTR_NAME_ATTACHED_NOTIFY_INFO, named_attrs));
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, ATTR_NAME_ATTACHED_NOTIFY_KEY, "as1"));
  EXPECT_TRUE(AttrUtils::SetInt(named_attrs, ATTR_NAME_ATTACHED_NOTIFY_TYPE, 1));
  EXPECT_TRUE(AttrUtils::SetNamedAttrs(relu2->GetOpDesc(), ATTR_NAME_ATTACHED_NOTIFY_INFO, named_attrs));

  auto relu3 = graph->FindNode("relu3");
  EXPECT_NE(relu3, nullptr);
  auto relu4 = graph->FindNode("relu4");
  EXPECT_NE(relu4, nullptr);
  NamedAttrs named_attrs2;
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs2, ATTR_NAME_ATTACHED_NOTIFY_POLICY, GROUP_POLICY));
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs2, ATTR_NAME_ATTACHED_NOTIFY_KEY, "as2"));
  EXPECT_TRUE(AttrUtils::SetInt(named_attrs2, ATTR_NAME_ATTACHED_NOTIFY_TYPE, 2));

  EXPECT_TRUE(AttrUtils::SetStr(relu3->GetOpDesc(), GROUP_POLICY, "group1"));
  EXPECT_TRUE(AttrUtils::SetStr(relu4->GetOpDesc(), GROUP_POLICY, "group1"));
  EXPECT_TRUE(AttrUtils::SetNamedAttrs(relu3->GetOpDesc(), ATTR_NAME_ATTACHED_NOTIFY_INFO, named_attrs2));
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs2, ATTR_NAME_ATTACHED_NOTIFY_KEY, "as3"));
  EXPECT_TRUE(AttrUtils::SetInt(named_attrs2, ATTR_NAME_ATTACHED_NOTIFY_TYPE, 3));
  EXPECT_TRUE(AttrUtils::SetNamedAttrs(relu4->GetOpDesc(), ATTR_NAME_ATTACHED_NOTIFY_INFO, named_attrs2));
  return graph;
}

/**
 * 图中有两个通信域， 所有的通信域中的节点设置了一样的attach notify名称
 * @return
 */
ComputeGraphPtr CreateGraph3() {
  const auto &graph = MakeGraph();
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);
  NamedAttrs named_attrs;
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, ATTR_NAME_ATTACHED_NOTIFY_POLICY, GROUP_POLICY));
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, ATTR_NAME_ATTACHED_NOTIFY_KEY, "as0"));

  EXPECT_TRUE(AttrUtils::SetStr(relu1->GetOpDesc(), GROUP_POLICY, "group0"));
  EXPECT_TRUE(AttrUtils::SetStr(relu2->GetOpDesc(), GROUP_POLICY, "group0"));
  EXPECT_TRUE(AttrUtils::SetNamedAttrs(relu1->GetOpDesc(), ATTR_NAME_ATTACHED_NOTIFY_INFO, named_attrs));
  EXPECT_TRUE(AttrUtils::SetNamedAttrs(relu2->GetOpDesc(), ATTR_NAME_ATTACHED_NOTIFY_INFO, named_attrs));
  auto relu3 = graph->FindNode("relu3");
  EXPECT_NE(relu3, nullptr);
  auto relu4 = graph->FindNode("relu4");
  EXPECT_NE(relu4, nullptr);

  EXPECT_TRUE(AttrUtils::SetStr(relu3->GetOpDesc(), GROUP_POLICY, "group1"));
  EXPECT_TRUE(AttrUtils::SetStr(relu4->GetOpDesc(), GROUP_POLICY, "group1"));
  EXPECT_TRUE(AttrUtils::SetNamedAttrs(relu3->GetOpDesc(), ATTR_NAME_ATTACHED_NOTIFY_INFO, named_attrs));
  EXPECT_TRUE(AttrUtils::SetNamedAttrs(relu4->GetOpDesc(), ATTR_NAME_ATTACHED_NOTIFY_INFO, named_attrs));
  return graph;
}

/**
 * 异常用例：
 * 图中有两个通信域， 所有的通信域中的节点设置了一样的attach notify名称, 但是attach notify type不相同
 * @return
 */
ComputeGraphPtr CreateGraph4() {
  const auto &graph = MakeGraph();
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);
  NamedAttrs named_attrs;
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, ATTR_NAME_ATTACHED_NOTIFY_POLICY, GROUP_POLICY));
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, ATTR_NAME_ATTACHED_NOTIFY_KEY, "as0"));

  EXPECT_TRUE(AttrUtils::SetStr(relu1->GetOpDesc(), GROUP_POLICY, "group0"));
  EXPECT_TRUE(AttrUtils::SetStr(relu2->GetOpDesc(), GROUP_POLICY, "group0"));
  EXPECT_TRUE(AttrUtils::SetNamedAttrs(relu1->GetOpDesc(), ATTR_NAME_ATTACHED_NOTIFY_INFO, named_attrs));
  EXPECT_TRUE(AttrUtils::SetNamedAttrs(relu2->GetOpDesc(), ATTR_NAME_ATTACHED_NOTIFY_INFO, named_attrs));
  auto relu3 = graph->FindNode("relu3");
  EXPECT_NE(relu3, nullptr);
  auto relu4 = graph->FindNode("relu4");
  EXPECT_NE(relu4, nullptr);

  EXPECT_TRUE(AttrUtils::SetStr(relu3->GetOpDesc(), GROUP_POLICY, "group1"));
  EXPECT_TRUE(AttrUtils::SetStr(relu4->GetOpDesc(), GROUP_POLICY, "group1"));
  EXPECT_TRUE(AttrUtils::SetNamedAttrs(relu3->GetOpDesc(), ATTR_NAME_ATTACHED_NOTIFY_INFO, named_attrs));
  // 更改type
  EXPECT_TRUE(AttrUtils::SetInt(named_attrs, ATTR_NAME_ATTACHED_NOTIFY_TYPE, 1));
  EXPECT_TRUE(AttrUtils::SetNamedAttrs(relu4->GetOpDesc(), ATTR_NAME_ATTACHED_NOTIFY_INFO, named_attrs));
  return graph;
}

/**
 * 适配V2场景，属性名字发生变化，采用V2版本表示新的属性方式;
 * 图中有一个通信域， 通信域中的节点设置了相同的attach notify名称
 * @return
 */
ComputeGraphPtr CreateGraph0_V2() {
  const auto &graph = MakeGraph();
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);

  ge::GeAttrValue::NAMED_ATTRS attached_notify;
  (void)ge::AttrUtils::SetStr(attached_notify, ATTR_NAME_ATTACHED_RESOURCE_NAME, "group0");
  (void)ge::AttrUtils::SetStr(attached_notify, ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, "an0");
  (void)ge::AttrUtils::SetInt(attached_notify, ge::ATTR_NAME_ATTACHED_RESOURCE_TYPE,
                              static_cast<int64_t>(SyncResType::kSyncResNotify));
  (void)ge::AttrUtils::SetBool(attached_notify, ge::ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG, true);
  std::vector<NamedAttrs> attached_sync_res_info_list_from_attr;
  attached_sync_res_info_list_from_attr.emplace_back(attached_notify);

  (void)ge::AttrUtils::SetListNamedAttrs(relu1->GetOpDesc(), ge::ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST,
                                         attached_sync_res_info_list_from_attr);
  (void)ge::AttrUtils::SetListNamedAttrs(relu2->GetOpDesc(), ge::ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST,
                                         attached_sync_res_info_list_from_attr);

  return graph;
}

ComputeGraphPtr CreateGraph1WithSubgraphs_V2() {
  auto sub_graph = MakeGraph();
  auto relu1 = sub_graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = sub_graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);

  ge::GeAttrValue::NAMED_ATTRS attached_notify;
  (void)ge::AttrUtils::SetStr(attached_notify, ATTR_NAME_ATTACHED_RESOURCE_NAME, "group0");
  (void)ge::AttrUtils::SetStr(attached_notify, ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, "an0");
  (void)ge::AttrUtils::SetInt(attached_notify, ge::ATTR_NAME_ATTACHED_RESOURCE_TYPE,
                              static_cast<int64_t>(SyncResType::kSyncResNotify));
  (void)ge::AttrUtils::SetBool(attached_notify, ge::ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG, true);
  std::vector<NamedAttrs> attached_sync_res_info_list_from_attr;
  attached_sync_res_info_list_from_attr.emplace_back(attached_notify);

  (void)ge::AttrUtils::SetListNamedAttrs(relu1->GetOpDesc(), ge::ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST,
                                         attached_sync_res_info_list_from_attr);

  (void)ge::AttrUtils::SetStr(attached_sync_res_info_list_from_attr[0], ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, "an1");
  (void)ge::AttrUtils::SetListNamedAttrs(relu2->GetOpDesc(), ge::ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST,
                                         attached_sync_res_info_list_from_attr);
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
 * 适配V2场景，属性名字发生变化，采用V2版本表示新的属性方式;
 * 图中有一个通信域， 通信域中的节点设置了不同的attach notify名称
 * @return
 */
ComputeGraphPtr CreateGraph1_V2() {
  const auto &graph = MakeGraph();
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);

  ge::GeAttrValue::NAMED_ATTRS attached_notify;
  (void)ge::AttrUtils::SetStr(attached_notify, ATTR_NAME_ATTACHED_RESOURCE_NAME, "group0");
  (void)ge::AttrUtils::SetStr(attached_notify, ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, "an0");
  (void)ge::AttrUtils::SetInt(attached_notify, ge::ATTR_NAME_ATTACHED_RESOURCE_TYPE,
                              static_cast<int64_t>(SyncResType::kSyncResNotify));
  (void)ge::AttrUtils::SetBool(attached_notify, ge::ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG, true);
  std::vector<NamedAttrs> attached_sync_res_info_list_from_attr;
  attached_sync_res_info_list_from_attr.emplace_back(attached_notify);

  (void)ge::AttrUtils::SetListNamedAttrs(relu1->GetOpDesc(), ge::ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST,
                                         attached_sync_res_info_list_from_attr);

  (void)ge::AttrUtils::SetStr(attached_sync_res_info_list_from_attr[0], ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, "an1");
  (void)ge::AttrUtils::SetListNamedAttrs(relu2->GetOpDesc(), ge::ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST,
                                         attached_sync_res_info_list_from_attr);

  return graph;
}

/**
 * 适配V2场景，属性名字发生变化，采用V2版本表示新的属性方式;
 * 图中有两个通信域， 每个通信域内的节点设置了不同的attach notify名称， 两个通信域的attach notify名称不同
 * @return
 */
ComputeGraphPtr CreateGraph2_V2() {
  const auto &graph = MakeGraph();
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);

  ge::GeAttrValue::NAMED_ATTRS attached_notify;
  (void)ge::AttrUtils::SetStr(attached_notify, ATTR_NAME_ATTACHED_RESOURCE_NAME, "group0");
  (void)ge::AttrUtils::SetStr(attached_notify, ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, "an0");
  (void)ge::AttrUtils::SetInt(attached_notify, ge::ATTR_NAME_ATTACHED_RESOURCE_TYPE,
                              static_cast<int64_t>(SyncResType::kSyncResNotify));
  (void)ge::AttrUtils::SetBool(attached_notify, ge::ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG, true);
  std::vector<NamedAttrs> attached_sync_res_info_list_from_attr;
  attached_sync_res_info_list_from_attr.emplace_back(attached_notify);

  (void)ge::AttrUtils::SetListNamedAttrs(relu1->GetOpDesc(), ge::ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST,
                                         attached_sync_res_info_list_from_attr);
  (void)ge::AttrUtils::SetStr(attached_sync_res_info_list_from_attr[0], ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, "an1");
  (void)ge::AttrUtils::SetListNamedAttrs(relu2->GetOpDesc(), ge::ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST,
                                         attached_sync_res_info_list_from_attr);

  auto relu3 = graph->FindNode("relu3");
  EXPECT_NE(relu3, nullptr);
  auto relu4 = graph->FindNode("relu4");
  EXPECT_NE(relu4, nullptr);

  ge::GeAttrValue::NAMED_ATTRS attached_notify2;
  EXPECT_TRUE(ge::AttrUtils::SetStr(attached_notify2, ATTR_NAME_ATTACHED_RESOURCE_NAME, "group1"));
  EXPECT_TRUE(ge::AttrUtils::SetStr(attached_notify2, ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, "an2"));
  EXPECT_TRUE(ge::AttrUtils::SetInt(attached_notify2, ge::ATTR_NAME_ATTACHED_RESOURCE_TYPE,
                              static_cast<int64_t>(SyncResType::kSyncResNotify)));
  EXPECT_TRUE(ge::AttrUtils::SetBool(attached_notify2, ge::ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG, true));
  std::vector<NamedAttrs> attached_sync_res_info_list_from_attr2;
  attached_sync_res_info_list_from_attr2.emplace_back(attached_notify2);

  EXPECT_TRUE(ge::AttrUtils::SetListNamedAttrs(relu3->GetOpDesc(), ge::ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST,
                                               attached_sync_res_info_list_from_attr2));

  EXPECT_TRUE(ge::AttrUtils::SetStr(attached_sync_res_info_list_from_attr2[0], ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, "an3"));
  EXPECT_TRUE(ge::AttrUtils::SetListNamedAttrs(relu4->GetOpDesc(), ge::ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST,
                                         attached_sync_res_info_list_from_attr2));
  return graph;
}

/**
 * 适配V2场景，属性名字发生变化，采用V2版本表示新的属性方式;
 * 图中有两个通信域， 所有的通信域中的节点设置了一样的attach notify名称
 * @return
 */
ComputeGraphPtr CreateGraph3_V2() {
  const auto &graph = MakeGraph();
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);
  ge::GeAttrValue::NAMED_ATTRS attached_notify;
  (void)ge::AttrUtils::SetStr(attached_notify, ATTR_NAME_ATTACHED_RESOURCE_NAME, "group0");
  (void)ge::AttrUtils::SetStr(attached_notify, ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, "an0");
  (void)ge::AttrUtils::SetInt(attached_notify, ge::ATTR_NAME_ATTACHED_RESOURCE_TYPE,
                              static_cast<int64_t>(SyncResType::kSyncResNotify));
  (void)ge::AttrUtils::SetBool(attached_notify, ge::ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG, true);
  std::vector<NamedAttrs> attached_sync_res_info_list_from_attr;
  attached_sync_res_info_list_from_attr.emplace_back(attached_notify);

  (void)ge::AttrUtils::SetListNamedAttrs(relu1->GetOpDesc(), ge::ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST,
                                         attached_sync_res_info_list_from_attr);
  (void)ge::AttrUtils::SetListNamedAttrs(relu2->GetOpDesc(), ge::ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST,
                                         attached_sync_res_info_list_from_attr);

  auto relu3 = graph->FindNode("relu3");
  EXPECT_NE(relu3, nullptr);
  auto relu4 = graph->FindNode("relu4");
  EXPECT_NE(relu4, nullptr);
  (void)ge::AttrUtils::SetStr(attached_sync_res_info_list_from_attr[0], ATTR_NAME_ATTACHED_RESOURCE_NAME, "group1");
  (void)ge::AttrUtils::SetListNamedAttrs(relu3->GetOpDesc(), ge::ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST,
                                         attached_sync_res_info_list_from_attr);
  (void)ge::AttrUtils::SetListNamedAttrs(relu4->GetOpDesc(), ge::ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST,
                                         attached_sync_res_info_list_from_attr);
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
                              static_cast<int64_t>(SyncResType::kSyncResNotify));
  (void)ge::AttrUtils::SetBool(attached_event, ge::ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG, true);
  (void)ge::AttrUtils::SetListNamedAttrs(relu1->GetOpDesc(), ge::ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST,
                                         {attached_event});
  (void)ge::AttrUtils::SetListNamedAttrs(relu2->GetOpDesc(), ge::ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST,
                                         {attached_event});

  NamedAttrs named_attrs;
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, ATTR_NAME_ATTACHED_NOTIFY_POLICY, GROUP_POLICY));
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, ATTR_NAME_ATTACHED_NOTIFY_KEY, "as0"));
  EXPECT_TRUE(AttrUtils::SetStr(relu3->GetOpDesc(), GROUP_POLICY, "group1"));
  EXPECT_TRUE(AttrUtils::SetStr(relu4->GetOpDesc(), GROUP_POLICY, "group1"));
  EXPECT_TRUE(AttrUtils::SetNamedAttrs(relu3->GetOpDesc(), ATTR_NAME_ATTACHED_NOTIFY_INFO, named_attrs));
  EXPECT_TRUE(AttrUtils::SetNamedAttrs(relu4->GetOpDesc(), ATTR_NAME_ATTACHED_NOTIFY_INFO, named_attrs));
  return graph;
}

/**
 * 图中一个节点设置了attached res type，但不是event；一个节点未设置attach res info
 * @return
 */
ComputeGraphPtr CreateGraph4_V2() {
  const auto &graph = MakeGraph();
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);

  NamedAttrs named_attrs;
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, ATTR_NAME_ATTACHED_RESOURCE_NAME, "notify"));
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, "res_key0"));
  EXPECT_TRUE(AttrUtils::SetInt(named_attrs, ATTR_NAME_ATTACHED_RESOURCE_TYPE,
                                static_cast<int64_t>(SyncResType::kSyncResEvent)));
  EXPECT_TRUE(AttrUtils::SetBool(named_attrs, ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG, true));
  EXPECT_TRUE(AttrUtils::SetListNamedAttrs(relu1->GetOpDesc(), ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST, {named_attrs}));
  return graph;
}

// 提供一个从属性中获取notify_id列表的接口
std::vector<int64_t> GetAttachedNotifyIds_V2(const NodePtr node) {
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
TEST_F(UtestAssignAttachedNotifyPass, Case0) {
  const auto graph = CreateGraph0();
  const auto &assign_attached_notify_pass = MakeShared<AssignAttachedNotifyPass>();
  EXPECT_NE(assign_attached_notify_pass, nullptr);
  std::vector<uint32_t> notify_types;
  uint32_t notify_num = 0U;
  EXPECT_EQ((assign_attached_notify_pass->Run(graph, notify_num, notify_types)), SUCCESS);

  // check 通信域内的节点才会分从notify
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);
  auto relu3 = graph->FindNode("relu3");
  EXPECT_NE(relu3, nullptr);
  auto relu4 = graph->FindNode("relu4");
  EXPECT_NE(relu4, nullptr);
  // 通信域内的relu1,relu2 分配了一样的attached notify id
  std::vector<int64_t> notify_id1;
  std::vector<int64_t> notify_id2;
  EXPECT_TRUE(AttrUtils::GetListInt(relu1->GetOpDescBarePtr(), RECV_ATTR_NOTIFY_ID, notify_id1));
  EXPECT_TRUE(AttrUtils::GetListInt(relu2->GetOpDescBarePtr(), RECV_ATTR_NOTIFY_ID, notify_id2));
  EXPECT_EQ(notify_id1, notify_id2);
  EXPECT_EQ(notify_id1[0], 0U);
  // 通信域外的relu3, relu4 的attached notify id还是无效值
  EXPECT_FALSE(AttrUtils::GetListInt(relu3->GetOpDescBarePtr(), RECV_ATTR_NOTIFY_ID, notify_id1));
  EXPECT_FALSE(AttrUtils::GetListInt(relu4->GetOpDescBarePtr(), RECV_ATTR_NOTIFY_ID, notify_id2));
  // 总的notify个数为1
  EXPECT_EQ(notify_num, 1);
}

TEST_F(UtestAssignAttachedNotifyPass, Case1) {
  const auto graph = CreateGraph1();
  const auto &assign_attached_notify_pass = MakeShared<AssignAttachedNotifyPass>();
  EXPECT_NE(assign_attached_notify_pass, nullptr);
  std::vector<uint32_t> notify_types;
  uint32_t notify_num = 2U;  // 假设已经分了两个普通的notify id
  notify_types.resize(notify_num, 0U);
  EXPECT_EQ((assign_attached_notify_pass->Run(graph, notify_num, notify_types)), SUCCESS);

  // check 同一个通信域内的从notify按照attach notify key来复用，attach notify key不一样的话，notify也不一样
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);
  auto relu3 = graph->FindNode("relu3");
  EXPECT_NE(relu3, nullptr);
  auto relu4 = graph->FindNode("relu4");
  EXPECT_NE(relu4, nullptr);
  // 通信域内的relu1,relu2 分配了不一样的attached notify id
  std::vector<int64_t> notify_id1;
  std::vector<int64_t> notify_id2;
  EXPECT_TRUE(AttrUtils::GetListInt(relu1->GetOpDescBarePtr(), RECV_ATTR_NOTIFY_ID, notify_id1));
  EXPECT_TRUE(AttrUtils::GetListInt(relu2->GetOpDescBarePtr(), RECV_ATTR_NOTIFY_ID, notify_id2));
  EXPECT_NE(notify_id1, notify_id2);
  EXPECT_TRUE(notify_id1[0] == 2U || notify_id1[0] == 3U);
  EXPECT_TRUE(notify_id2[0] == 2U || notify_id2[0] == 3U);
  // 通信域外的relu3, relu4 的attached notify id还是无效值
  EXPECT_FALSE(AttrUtils::GetListInt(relu3->GetOpDescBarePtr(), RECV_ATTR_NOTIFY_ID, notify_id1));
  EXPECT_FALSE(AttrUtils::GetListInt(relu4->GetOpDescBarePtr(), RECV_ATTR_NOTIFY_ID, notify_id2));
  // 总的notify个数为4
  EXPECT_EQ(notify_num, 2 + 2);
}

TEST_F(UtestAssignAttachedNotifyPass, Case2) {
  const auto graph = CreateGraph2();
  const auto &assign_attached_notify_pass = MakeShared<AssignAttachedNotifyPass>();
  EXPECT_NE(assign_attached_notify_pass, nullptr);
  uint32_t notify_num = 2U;  // 假设已经分了两个普通的notify id
  std::vector<uint32_t> notify_types;
  notify_types.resize(notify_num, 0U);
  EXPECT_EQ((assign_attached_notify_pass->Run(graph, notify_num, notify_types)), SUCCESS);
  // check 同一个通信域内的从notify按照attach notify key来复用，attach notify key一样的话，notify也一样
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);
  auto relu3 = graph->FindNode("relu3");
  EXPECT_NE(relu3, nullptr);
  auto relu4 = graph->FindNode("relu4");
  EXPECT_NE(relu4, nullptr);
  // 通信域g0内的relu1,relu2 分配了不一样的attached notify id
  std::vector<int64_t> notify_id1;
  std::vector<int64_t> notify_id2;
  EXPECT_TRUE(AttrUtils::GetListInt(relu1->GetOpDescBarePtr(), RECV_ATTR_NOTIFY_ID, notify_id1));
  EXPECT_TRUE(AttrUtils::GetListInt(relu2->GetOpDescBarePtr(), RECV_ATTR_NOTIFY_ID, notify_id2));
  EXPECT_NE(notify_id1[0], notify_id2[0]);

  // 通信域g1内的relu3,relu4 分配了不一样的attached notify id
  std::vector<int64_t> notify_id3;
  std::vector<int64_t> notify_id4;
  EXPECT_TRUE(AttrUtils::GetListInt(relu3->GetOpDescBarePtr(), RECV_ATTR_NOTIFY_ID, notify_id3));
  EXPECT_TRUE(AttrUtils::GetListInt(relu4->GetOpDescBarePtr(), RECV_ATTR_NOTIFY_ID, notify_id4));
  EXPECT_NE(notify_id3[0], notify_id4[0]);
  // 两个通信域的notify不同
  EXPECT_NE(notify_id1[0], notify_id3[0]);
  // 总的notify个数为6
  EXPECT_EQ(notify_num, 2 + 4);
}

TEST_F(UtestAssignAttachedNotifyPass, Case3) {
  const auto graph = CreateGraph3();
  const auto &assign_attached_notify_pass = MakeShared<AssignAttachedNotifyPass>();
  EXPECT_NE(assign_attached_notify_pass, nullptr);
  std::vector<uint32_t> notify_types;
  uint32_t notify_num = 0U;
  EXPECT_EQ((assign_attached_notify_pass->Run(graph, notify_num, notify_types)), SUCCESS);

  // check 不同的通信域的从notify不复用
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);
  auto relu3 = graph->FindNode("relu3");
  EXPECT_NE(relu3, nullptr);
  auto relu4 = graph->FindNode("relu4");
  EXPECT_NE(relu4, nullptr);
  // 通信域g0内的relu1,relu2 分配了一样的attached notify id
  std::vector<int64_t> notify_id1;
  std::vector<int64_t> notify_id2;
  EXPECT_TRUE(AttrUtils::GetListInt(relu1->GetOpDescBarePtr(), RECV_ATTR_NOTIFY_ID, notify_id1));
  EXPECT_TRUE(AttrUtils::GetListInt(relu2->GetOpDescBarePtr(), RECV_ATTR_NOTIFY_ID, notify_id2));
  EXPECT_EQ(notify_id1[0], notify_id2[0]);
  // 通信域g1内的relu3,relu4 分配了一样的attached notify id
  std::vector<int64_t> notify_id3;
  std::vector<int64_t> notify_id4;
  EXPECT_TRUE(AttrUtils::GetListInt(relu3->GetOpDescBarePtr(), RECV_ATTR_NOTIFY_ID, notify_id3));
  EXPECT_TRUE(AttrUtils::GetListInt(relu4->GetOpDescBarePtr(), RECV_ATTR_NOTIFY_ID, notify_id4));
  EXPECT_EQ(notify_id3[0], notify_id4[0]);

  // 两个通信域的notify不同， 虽然通信域的attached notify key一样
  EXPECT_NE(notify_id1[0], notify_id3[0]);
  // 总的notify个数为2
  EXPECT_EQ(notify_num, 2);
}

TEST_F(UtestAssignAttachedNotifyPass, InvalidCase) {
  const auto graph = CreateGraph4();
  const auto &assign_attached_notify_pass = MakeShared<AssignAttachedNotifyPass>();
  EXPECT_NE(assign_attached_notify_pass, nullptr);
  std::vector<uint32_t> notify_types;
  uint32_t notify_num = 0U;
  EXPECT_NE((assign_attached_notify_pass->Run(graph, notify_num, notify_types)), SUCCESS);
}

TEST_F(UtestAssignAttachedNotifyPass, Assign_With_Subgraph) {
  const auto graph = CreateGraph1WithSubgraphs();
  auto parent_node = graph->FindNode("partitioncall_0");
  EXPECT_NE(parent_node, nullptr);
  auto sub_graph = ge::NodeUtils::GetSubgraph(*parent_node, 0U);
  EXPECT_NE(sub_graph, nullptr);

  Graph2SubGraphInfoList graph_2_sub_graph_info_list{{graph, {}}, {sub_graph, {}}};
  const auto &allocator = MakeShared<StreamAllocator>(graph, graph_2_sub_graph_info_list);
  EXPECT_EQ(  allocator->AssignAttachedNotifyResource(), SUCCESS);
  auto relu1 = sub_graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = sub_graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);
  std::vector<int64_t> notify_id1;
  std::vector<int64_t> notify_id2;
  EXPECT_TRUE(AttrUtils::GetListInt(relu1->GetOpDescBarePtr(), RECV_ATTR_NOTIFY_ID, notify_id1));
  EXPECT_TRUE(AttrUtils::GetListInt(relu2->GetOpDescBarePtr(), RECV_ATTR_NOTIFY_ID, notify_id2));
  EXPECT_EQ(notify_id1.size(), 1UL);
  EXPECT_EQ(notify_id2.size(), 1UL);
  EXPECT_EQ(notify_id1[0], 0);
  EXPECT_EQ(notify_id2[0], 1);
}

TEST_F(UtestAssignAttachedNotifyPass, Case0_V2) {
  const auto graph = CreateGraph0_V2();
  const auto &assign_attached_notify_pass = MakeShared<AssignAttachedNotifyPass>();
  EXPECT_NE(assign_attached_notify_pass, nullptr);
  std::vector<uint32_t> notify_types;
  uint32_t notify_num = 0U;
  EXPECT_EQ((assign_attached_notify_pass->Run(graph, notify_num, notify_types)), SUCCESS);

  // check 通信域内的节点才会分从notify
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);
  auto relu3 = graph->FindNode("relu3");
  EXPECT_NE(relu3, nullptr);
  auto relu4 = graph->FindNode("relu4");
  EXPECT_NE(relu4, nullptr);

  auto notify_id1 = GetAttachedNotifyIds_V2(relu1);
  auto notify_id2 = GetAttachedNotifyIds_V2(relu2);
  auto notify_id3 = GetAttachedNotifyIds_V2(relu3);
  auto notify_id4 = GetAttachedNotifyIds_V2(relu4);

  // 通信域内的relu1,relu2 分配了一样的attached notify id
  EXPECT_EQ(notify_id1.size(), 1);
  EXPECT_EQ(notify_id2.size(), 1);
  EXPECT_EQ(notify_id1[0], notify_id2[0]);
  EXPECT_EQ(notify_id1[0], 0);

  // 通信域外的relu3, relu4 的attached notify id还是无效值
  EXPECT_EQ(notify_id3.size(), 0);
  EXPECT_EQ(notify_id4.size(), 0);
  EXPECT_EQ(notify_num, 1);
}

TEST_F(UtestAssignAttachedNotifyPass, Case1_V2) {
  const auto graph = CreateGraph1_V2();
  const auto &assign_attached_notify_pass = MakeShared<AssignAttachedNotifyPass>();
  EXPECT_NE(assign_attached_notify_pass, nullptr);
  std::vector<uint32_t> notify_types;
  uint32_t notify_num = 2U;  // 假设已经分了两个普通的notify id
  notify_types.resize(notify_num, 0U);
  EXPECT_EQ((assign_attached_notify_pass->Run(graph, notify_num, notify_types)), SUCCESS);

  // check 同一个通信域内的从notify按照attach notify key来复用，attach notify key不一样的话，notify也不一样
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);
  auto relu3 = graph->FindNode("relu3");
  EXPECT_NE(relu3, nullptr);
  auto relu4 = graph->FindNode("relu4");
  EXPECT_NE(relu4, nullptr);
  // 通信域内的relu1,relu2 分配了不一样的attached notify id
  std::vector<int64_t> notify_id1 = GetAttachedNotifyIds_V2(relu1);
  std::vector<int64_t> notify_id2 = GetAttachedNotifyIds_V2(relu2);
  EXPECT_EQ(notify_id1.size(), 1);
  EXPECT_EQ(notify_id2.size(), 1);
  EXPECT_NE(notify_id1, notify_id2);
  EXPECT_TRUE(notify_id1[0] == 2U || notify_id1[0] == 3U);
  EXPECT_TRUE(notify_id2[0] == 2U || notify_id2[0] == 3U);


  // 通信域外的relu3, relu4 的attached notify id还是无效值
  std::vector<int64_t> notify_id3 = GetAttachedNotifyIds_V2(relu3);
  std::vector<int64_t> notify_id4 = GetAttachedNotifyIds_V2(relu4);
  EXPECT_EQ(notify_id3.size(), 0);
  EXPECT_EQ(notify_id4.size(), 0);

  // 总的notify个数为4
  EXPECT_EQ(notify_num, 2 + 2);
}

TEST_F(UtestAssignAttachedNotifyPass, Case2_V2) {
  const auto graph = CreateGraph2_V2();
  const auto &assign_attached_notify_pass = MakeShared<AssignAttachedNotifyPass>();
  EXPECT_NE(assign_attached_notify_pass, nullptr);
  uint32_t notify_num = 2U;  // 假设已经分了两个普通的notify id
  std::vector<uint32_t> notify_types;
  notify_types.resize(notify_num, 0U);
  EXPECT_EQ((assign_attached_notify_pass->Run(graph, notify_num, notify_types)), SUCCESS);
  // check 同一个通信域内的从notify按照attach notify key来复用，attach notify key一样的话，notify也一样
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);
  auto relu3 = graph->FindNode("relu3");
  EXPECT_NE(relu3, nullptr);
  auto relu4 = graph->FindNode("relu4");
  EXPECT_NE(relu4, nullptr);
  // 通信域g0内的relu1,relu2 分配了不一样的attached notify id
  std::vector<int64_t> notify_id1 = GetAttachedNotifyIds_V2(relu1);
  std::vector<int64_t> notify_id2 = GetAttachedNotifyIds_V2(relu2);
  EXPECT_EQ(notify_id1.size(), 1);
  EXPECT_EQ(notify_id2.size(), 1);
  EXPECT_NE(notify_id1[0], notify_id2[0]);

  // 通信域g1内的relu3,relu4 分配了不一样的attached notify id
  std::vector<int64_t> notify_id3 = GetAttachedNotifyIds_V2(relu3);
  std::vector<int64_t> notify_id4 = GetAttachedNotifyIds_V2(relu4);
  EXPECT_EQ(notify_id3.size(), 1);
  EXPECT_EQ(notify_id4.size(), 1);
  EXPECT_NE(notify_id3[0], notify_id4[0]);

  // 两个通信域的notify不同
  EXPECT_NE(notify_id1[0], notify_id3[0]);
  // 总的notify个数为6
  EXPECT_EQ(notify_num, 2 + 4);
}

TEST_F(UtestAssignAttachedNotifyPass, Case3_V2) {
  const auto graph = CreateGraph3_V2();
  const auto &assign_attached_notify_pass = MakeShared<AssignAttachedNotifyPass>();
  EXPECT_NE(assign_attached_notify_pass, nullptr);
  std::vector<uint32_t> notify_types;
  uint32_t notify_num = 0U;
  EXPECT_EQ((assign_attached_notify_pass->Run(graph, notify_num, notify_types)), SUCCESS);

  // check 不同的通信域的从notify不复用
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);
  auto relu3 = graph->FindNode("relu3");
  EXPECT_NE(relu3, nullptr);
  auto relu4 = graph->FindNode("relu4");
  EXPECT_NE(relu4, nullptr);
  // 通信域g0内的relu1,relu2 分配了一样的attached notify id
  std::vector<int64_t> notify_id1 = GetAttachedNotifyIds_V2(relu1);
  std::vector<int64_t> notify_id2 = GetAttachedNotifyIds_V2(relu2);
  EXPECT_EQ(notify_id1.size(), 1);
  EXPECT_EQ(notify_id2.size(), 1);
  EXPECT_EQ(notify_id1[0], notify_id2[0]);

  // 通信域g1内的relu3,relu4 分配了一样的attached notify id
  std::vector<int64_t> notify_id3 = GetAttachedNotifyIds_V2(relu3);
  std::vector<int64_t> notify_id4 = GetAttachedNotifyIds_V2(relu4);
  EXPECT_EQ(notify_id3.size(), 1);
  EXPECT_EQ(notify_id4.size(), 1);
  EXPECT_EQ(notify_id3[0], notify_id4[0]);
  EXPECT_NE(notify_id1[0], notify_id3[0]);
  // 总的notify个数为2
  EXPECT_EQ(notify_num, 2);
}

TEST_F(UtestAssignAttachedNotifyPass, Assign_With_Subgraph_V2) {
  const auto graph = CreateGraph1WithSubgraphs_V2();
  auto parent_node = graph->FindNode("partitioncall_0");
  EXPECT_NE(parent_node, nullptr);
  auto sub_graph = ge::NodeUtils::GetSubgraph(*parent_node, 0U);
  EXPECT_NE(sub_graph, nullptr);

  Graph2SubGraphInfoList graph_2_sub_graph_info_list{{graph, {}}, {sub_graph, {}}};
  const auto &allocator = MakeShared<StreamAllocator>(graph, graph_2_sub_graph_info_list);
  EXPECT_EQ(  allocator->AssignAttachedNotifyResource(), SUCCESS);
  auto relu1 = sub_graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = sub_graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);
  std::vector<int64_t> notify_id1 = GetAttachedNotifyIds_V2(relu1);
  std::vector<int64_t> notify_id2 = GetAttachedNotifyIds_V2(relu2);

  EXPECT_EQ(notify_id1.size(), 1UL);
  EXPECT_EQ(notify_id2.size(), 1UL);
  EXPECT_EQ(notify_id1[0], 0);
  EXPECT_EQ(notify_id2[0], 1);
}

TEST_F(UtestAssignAttachedNotifyPass, Not_Require_V2) {
  const auto graph = CreateGraph0_V2();
  // check 通信域内的节点才会分从notify
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);

  std::vector<NamedAttrs> attached_sync_res_info_list_from_attr;
  (void)ge::AttrUtils::GetListNamedAttrs(relu1->GetOpDesc(), ge::ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST,
                                         attached_sync_res_info_list_from_attr);
  EXPECT_EQ(attached_sync_res_info_list_from_attr.size(), 1);
  EXPECT_TRUE(ge::AttrUtils::SetBool(attached_sync_res_info_list_from_attr[0], ge::ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG, false));
  (void)ge::AttrUtils::SetListNamedAttrs(relu1->GetOpDesc(), ge::ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST,
                                         attached_sync_res_info_list_from_attr);
  (void)ge::AttrUtils::SetListNamedAttrs(relu2->GetOpDesc(), ge::ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST,
                                         attached_sync_res_info_list_from_attr);

  const auto &assign_attached_notify_pass = MakeShared<AssignAttachedNotifyPass>();
  EXPECT_NE(assign_attached_notify_pass, nullptr);
  std::vector<uint32_t> notify_types;
  uint32_t notify_num = 0U;
  EXPECT_EQ((assign_attached_notify_pass->Run(graph, notify_num, notify_types)), SUCCESS);

  auto notify_id1 = GetAttachedNotifyIds_V2(relu1);
  auto notify_id2 = GetAttachedNotifyIds_V2(relu2);

  // required为false，不分配资源
  EXPECT_EQ(notify_id1.size(), 1);
  EXPECT_EQ(notify_id2.size(), 1);
  EXPECT_EQ(notify_id1[0], notify_id2[0]);
  EXPECT_EQ(notify_id1[0], -1);

  EXPECT_EQ(notify_num, 0);
}

/*
 * 兼容场景，老的v1流程跟新的v2流程均存在
 * relu1、relu2共用一个event，relu3、relu4共用一个event
 * */
TEST_F(UtestAssignAttachedNotifyPass, Mix_V1_V2) {
  const auto graph = CreateGraph5_V2();
  const auto &assign_attached_notify_pass = MakeShared<AssignAttachedNotifyPass>();
  EXPECT_NE(assign_attached_notify_pass, nullptr);
  std::vector<uint32_t> notify_types;
  uint32_t notify_num = 0U;
  EXPECT_EQ((assign_attached_notify_pass->Run(graph, notify_num, notify_types)), SUCCESS);

  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);

  auto relu3 = graph->FindNode("relu3");
  EXPECT_NE(relu3, nullptr);
  auto relu4 = graph->FindNode("relu4");
  EXPECT_NE(relu4, nullptr);

  auto notify_ids1 = GetAttachedNotifyIds_V2(relu1);
  auto notify_ids2 = GetAttachedNotifyIds_V2(relu2);
  EXPECT_EQ(notify_ids1.size(), 1);
  EXPECT_EQ(notify_ids2.size(), 1);
  EXPECT_EQ(notify_ids1[0], notify_ids2[0]);
  EXPECT_EQ(notify_ids1[0], 1);

  std::vector<int64_t> notify_id3;
  std::vector<int64_t> notify_id4;
  EXPECT_TRUE(AttrUtils::GetListInt(relu3->GetOpDescBarePtr(), RECV_ATTR_NOTIFY_ID, notify_id3));
  EXPECT_TRUE(AttrUtils::GetListInt(relu4->GetOpDescBarePtr(), RECV_ATTR_NOTIFY_ID, notify_id4));
  EXPECT_EQ(notify_id3.size(), 1U);
  EXPECT_EQ(notify_id4.size(), 1U);
  EXPECT_EQ(notify_id3[0], notify_id4[0]);
  EXPECT_EQ(notify_id3[0], 0);

  // relu1、relu2共用一个event，relu3、relu4共用一个event
  EXPECT_EQ(notify_num, 2);
}

TEST_F(UtestAssignAttachedNotifyPass, Not_Notify_V2) {
  const auto graph = CreateGraph4_V2();
  const auto &assign_attached_notify_pass = MakeShared<AssignAttachedNotifyPass>();
  EXPECT_NE(assign_attached_notify_pass, nullptr);
  std::vector<uint32_t> notify_types;
  uint32_t notify_num = 0U;
  EXPECT_EQ((assign_attached_notify_pass->Run(graph, notify_num, notify_types)), SUCCESS);

  // check 分配了attach event的节点才会分从event
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);

  // relu1,relu2 没有分配attached event id
  std::vector<int64_t> event_id1 = GetAttachedNotifyIds_V2(relu1);
  std::vector<int64_t> event_id2 = GetAttachedNotifyIds_V2(relu2);
  EXPECT_EQ(event_id1.size(), 1U);
  EXPECT_EQ(event_id1[0], -1);
  EXPECT_EQ(event_id2.size(), 0U);

  // 总的event个数为0
  EXPECT_EQ(notify_num, 0);
}

}  // namespace ge

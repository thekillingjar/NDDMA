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
#include "graph/build/stream/assign_attached_stream_pass.h"
#include "graph/ge_local_context.h"
#include "macro_utils/dt_public_unscope.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"

using namespace std;
namespace {
struct StreamInfo {
  ge::AscendString name;
  ge::AscendString reuse_key;
  std::vector<int64_t> depend_value_input_indices;
  bool required{true};
  bool is_valid{false};
  int64_t stream_id{-1};
};

enum class SyncResType {
  SYNC_RES_EVENT,
  SYNC_RES_NOTIFY,
  END
};

struct SyncResInfo {
  SyncResType type;
  ge::AscendString name;
  ge::AscendString reuse_key;
  bool required{true};
  bool is_valid{false};
  int32_t sync_res_id{-1};
};
}
namespace ge {
class UtestAssignAttachedStreamPass : public testing::Test {
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
 * 图中有一个通信域， 通信域中的节点设置了相同的attach stream名称
 * 其余两个节点使用新的打属性方式
 * @return
 */
ComputeGraphPtr CreateGraph0() {
  const auto &graph = MakeGraph();
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);
  NamedAttrs named_attrs;
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, ATTR_NAME_ATTACHED_STREAM_POLICY, GROUP_POLICY));
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, ATTR_NAME_ATTACHED_STREAM_KEY, "as0"));
  EXPECT_TRUE(AttrUtils::SetNamedAttrs(relu1->GetOpDesc(), ATTR_NAME_ATTACHED_STREAM_INFO, named_attrs));
  EXPECT_TRUE(AttrUtils::SetNamedAttrs(relu2->GetOpDesc(), ATTR_NAME_ATTACHED_STREAM_INFO, named_attrs));
  EXPECT_TRUE(AttrUtils::SetStr(relu1->GetOpDesc(), GROUP_POLICY, "group0"));
  EXPECT_TRUE(AttrUtils::SetStr(relu2->GetOpDesc(), GROUP_POLICY, "group0"));

  return graph;
}

/**
 * 图中有一个通信域， 通信域中的节点设置了不同的attach stream名称
 * @return
 */
ComputeGraphPtr CreateGraph1() {
  const auto &graph = MakeGraph();
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);
  NamedAttrs named_attrs;
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, ATTR_NAME_ATTACHED_STREAM_POLICY, GROUP_POLICY));
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, ATTR_NAME_ATTACHED_STREAM_KEY, "as0"));
  EXPECT_TRUE(AttrUtils::SetNamedAttrs(relu1->GetOpDesc(), ATTR_NAME_ATTACHED_STREAM_INFO, named_attrs));
  NamedAttrs named_attrs2;
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs2, ATTR_NAME_ATTACHED_STREAM_POLICY, GROUP_POLICY));
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs2, ATTR_NAME_ATTACHED_STREAM_KEY, "as1"));
  EXPECT_TRUE(AttrUtils::SetNamedAttrs(relu2->GetOpDesc(), ATTR_NAME_ATTACHED_STREAM_INFO, named_attrs2));

  EXPECT_TRUE(AttrUtils::SetStr(relu1->GetOpDesc(), GROUP_POLICY, "group0"));
  EXPECT_TRUE(AttrUtils::SetStr(relu2->GetOpDesc(), GROUP_POLICY, "group0"));

  return graph;
}

/**
 * 图中有两个通信域， 每个通信域内的节点设置了相同的attach stream名称， 两个通信域的attach stream名称不同
 * @return
 */
ComputeGraphPtr CreateGraph2() {
  const auto &graph = MakeGraph();
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);

  NamedAttrs named_attrs;
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, ATTR_NAME_ATTACHED_STREAM_POLICY, GROUP_POLICY));
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, ATTR_NAME_ATTACHED_STREAM_KEY, "as0"));
  EXPECT_TRUE(AttrUtils::SetStr(relu1->GetOpDesc(), GROUP_POLICY, "group0"));
  EXPECT_TRUE(AttrUtils::SetStr(relu2->GetOpDesc(), GROUP_POLICY, "group0"));
  EXPECT_TRUE(AttrUtils::SetNamedAttrs(relu1->GetOpDesc(), ATTR_NAME_ATTACHED_STREAM_INFO, named_attrs));
  EXPECT_TRUE(AttrUtils::SetNamedAttrs(relu2->GetOpDesc(), ATTR_NAME_ATTACHED_STREAM_INFO, named_attrs));

  auto relu3 = graph->FindNode("relu3");
  EXPECT_NE(relu3, nullptr);
  auto relu4 = graph->FindNode("relu4");
  EXPECT_NE(relu4, nullptr);
  NamedAttrs named_attrs2;
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs2, ATTR_NAME_ATTACHED_STREAM_POLICY, GROUP_POLICY));
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs2, ATTR_NAME_ATTACHED_STREAM_KEY, "as1"));
  EXPECT_TRUE(AttrUtils::SetStr(relu3->GetOpDesc(), GROUP_POLICY, "group1"));
  EXPECT_TRUE(AttrUtils::SetStr(relu4->GetOpDesc(), GROUP_POLICY, "group1"));
  EXPECT_TRUE(AttrUtils::SetNamedAttrs(relu3->GetOpDesc(), ATTR_NAME_ATTACHED_STREAM_INFO, named_attrs2));
  EXPECT_TRUE(AttrUtils::SetNamedAttrs(relu4->GetOpDesc(), ATTR_NAME_ATTACHED_STREAM_INFO, named_attrs2));
  return graph;
}

/**
 * 图中有两个通信域， 所有的通信域中的节点设置了一样的attach stream名称
 * @return
 */
ComputeGraphPtr CreateGraph3(const std::string &graph_name = "g1") {
  const auto &graph = MakeGraph();
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);
  NamedAttrs named_attrs;
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, ATTR_NAME_ATTACHED_STREAM_POLICY, GROUP_POLICY));
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, ATTR_NAME_ATTACHED_STREAM_KEY, "as0"));

  EXPECT_TRUE(AttrUtils::SetStr(relu1->GetOpDesc(), GROUP_POLICY, "group0"));
  EXPECT_TRUE(AttrUtils::SetStr(relu2->GetOpDesc(), GROUP_POLICY, "group0"));
  EXPECT_TRUE(AttrUtils::SetNamedAttrs(relu1->GetOpDesc(), ATTR_NAME_ATTACHED_STREAM_INFO, named_attrs));
  EXPECT_TRUE(AttrUtils::SetNamedAttrs(relu2->GetOpDesc(), ATTR_NAME_ATTACHED_STREAM_INFO, named_attrs));
  auto relu3 = graph->FindNode("relu3");
  EXPECT_NE(relu3, nullptr);
  auto relu4 = graph->FindNode("relu4");
  EXPECT_NE(relu4, nullptr);

  EXPECT_TRUE(AttrUtils::SetStr(relu3->GetOpDesc(), GROUP_POLICY, "group1"));
  EXPECT_TRUE(AttrUtils::SetStr(relu4->GetOpDesc(), GROUP_POLICY, "group1"));
  EXPECT_TRUE(AttrUtils::SetNamedAttrs(relu3->GetOpDesc(), ATTR_NAME_ATTACHED_STREAM_INFO, named_attrs));
  EXPECT_TRUE(AttrUtils::SetNamedAttrs(relu4->GetOpDesc(), ATTR_NAME_ATTACHED_STREAM_INFO, named_attrs));
  graph->SetName(graph_name);
  return graph;
}

/**
 * 
 * @return
 */
ComputeGraphPtr CreateGraph4() {
  const auto &graph = MakeGraph();
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);
  NamedAttrs named_attrs;
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, "_attached_stream_key", "tilingsink"));
  relu1->GetOpDesc()->SetStreamId(0U);
  relu2->GetOpDesc()->SetStreamId(1U);
  EXPECT_TRUE(AttrUtils::SetListNamedAttrs(relu1->GetOpDesc(), ATTR_NAME_ATTACHED_STREAM_INFO, {named_attrs}));
  EXPECT_TRUE(AttrUtils::SetListNamedAttrs(relu2->GetOpDesc(), ATTR_NAME_ATTACHED_STREAM_INFO, {named_attrs}));
  return graph;
}

/**
 * v1控制算子子图，不应该分配从流
 * @return
 */
ComputeGraphPtr CreateV1CtrlGraph() {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("data1", DATA)->NODE("relu1", RELU)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("data2", DATA)->NODE("relu2", RELU)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("data3", DATA)->NODE("relu3", RELU)->NODE("merge", STREAMMERGE)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("data4", DATA)->NODE("relu4", RELU)->NODE("merge", STREAMMERGE));
                };
  auto graph1 = ToGeGraph(g1);
  auto graph = ge::GraphUtilsEx::GetComputeGraph(graph1);
  graph->TopologicalSortingGraph();

  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);

  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);
  auto merge_op = graph->FindNode("merge");
  EXPECT_NE(merge_op, nullptr);

  NamedAttrs named_attrs;
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, ATTR_NAME_ATTACHED_STREAM_POLICY, GROUP_POLICY));
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, ATTR_NAME_ATTACHED_STREAM_KEY, "as0"));
  EXPECT_TRUE(AttrUtils::SetNamedAttrs(relu1->GetOpDesc(), ATTR_NAME_ATTACHED_STREAM_INFO, named_attrs));
  EXPECT_TRUE(AttrUtils::SetNamedAttrs(relu2->GetOpDesc(), ATTR_NAME_ATTACHED_STREAM_INFO, named_attrs));
  EXPECT_TRUE(AttrUtils::SetStr(relu1->GetOpDesc(), GROUP_POLICY, "group0"));
  EXPECT_TRUE(AttrUtils::SetStr(relu2->GetOpDesc(), GROUP_POLICY, "group0"));
  EXPECT_TRUE(AttrUtils::SetBool(relu1->GetOpDesc(), "_disable_attached_resource", true));
  EXPECT_TRUE(AttrUtils::SetBool(relu2->GetOpDesc(), "_disable_attached_resource", true));

  return graph;
}

ComputeGraphPtr CreateGraphWithSubgraphs(const std::vector<ComputeGraphPtr> &sub_graphs) {
  const size_t subgraph_num = sub_graphs.size();
  const std::string prefix = "partitioncall_";
  ComputeGraphPtr root_graph;
  DEF_GRAPH(g1) {
    for (size_t i = 0; i < subgraph_num; ++i) {
      CHAIN(NODE("data1", DATA)->NODE(prefix + std::to_string(i), PARTITIONEDCALL)->NODE("netoutput", NETOUTPUT));
    }
  };
  auto graph1 = ToGeGraph(g1);
  root_graph = ge::GraphUtilsEx::GetComputeGraph(graph1);

  size_t i = 0U;
  for (const auto &sub_graph : sub_graphs) {
    auto node_with_subgraph = root_graph->FindNode(prefix + std::to_string(i));
    EXPECT_NE(node_with_subgraph, nullptr);
    node_with_subgraph->GetOpDesc()->AddSubgraphName(sub_graph->GetName());
    node_with_subgraph->GetOpDesc()->SetSubgraphInstanceName(0, sub_graph->GetName());
    sub_graph->SetParentNode(node_with_subgraph);
    sub_graph->SetParentGraph(root_graph);
    root_graph->AddSubGraph(sub_graph);
    ++i;
  }
  EXPECT_EQ(root_graph->TopologicalSorting(), GRAPH_SUCCESS);
  return root_graph;
}

/**
 * 适配V2场景，属性名字发生变化，采用V2版本表示新的属性方式;
 * 图中有一个通信域， 通信域中的节点设置了相同的attach stream名称
 * 其余两个节点使用新的打属性方式
 * @return
 */
ComputeGraphPtr CreateGraph0_V2() {
  const auto &graph = MakeGraph();
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);

  std::vector<ge::NamedAttrs> attached_stream_info;
  ge::GeAttrValue::NAMED_ATTRS attached_stream;
  (void)ge::AttrUtils::SetStr(attached_stream, ATTR_NAME_ATTACHED_RESOURCE_NAME, "tiling");
  (void)ge::AttrUtils::SetStr(attached_stream, ge::ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, "tiling");
  std::vector<int64_t> depend_value_input_indices{0};
  (void)ge::AttrUtils::SetListInt(attached_stream, ge::ATTR_NAME_ATTACHED_RESOURCE_DEPEND_VALUE_LIST_INT,
                                  depend_value_input_indices);
  (void)ge::AttrUtils::SetBool(attached_stream, ge::ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG, true);
  attached_stream_info.emplace_back(attached_stream);
  (void)ge::AttrUtils::SetListNamedAttrs(relu1->GetOpDesc(), ge::ATTR_NAME_ATTACHED_STREAM_INFO_LIST,
                                         attached_stream_info);

  attached_stream_info.clear();
  (void)ge::AttrUtils::SetStr(attached_stream, ATTR_NAME_ATTACHED_RESOURCE_NAME, "tiling");
  (void)ge::AttrUtils::SetStr(attached_stream, ge::ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, "tiling");
  (void)ge::AttrUtils::SetListInt(attached_stream, ge::ATTR_NAME_ATTACHED_RESOURCE_DEPEND_VALUE_LIST_INT,
                                  depend_value_input_indices);
  (void)ge::AttrUtils::SetBool(attached_stream, ge::ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG, true);
  attached_stream_info.emplace_back(attached_stream);
  (void)ge::AttrUtils::SetListNamedAttrs(relu2->GetOpDesc(), ge::ATTR_NAME_ATTACHED_STREAM_INFO_LIST,
                                         attached_stream_info);

  return graph;
}

/**
 * 适配V2场景，属性名字发生变化，采用V2版本表示新的属性方式;
 * 图中有一个通信域， 通信域中的节点设置了不同的attach stream名称
 * @return
 */
ComputeGraphPtr CreateGraph1_V2() {
  const auto &graph = MakeGraph();
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);

  std::vector<ge::NamedAttrs> attached_stream_info;
  ge::GeAttrValue::NAMED_ATTRS attached_stream;
  (void)ge::AttrUtils::SetStr(attached_stream, ATTR_NAME_ATTACHED_RESOURCE_NAME, "mc2");
  (void)ge::AttrUtils::SetStr(attached_stream, ge::ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, "as0");
  (void)ge::AttrUtils::SetBool(attached_stream, ge::ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG, true);
  attached_stream_info.emplace_back(attached_stream);
  (void)ge::AttrUtils::SetListNamedAttrs(relu1->GetOpDesc(), ge::ATTR_NAME_ATTACHED_STREAM_INFO_LIST,
                                         attached_stream_info);

  attached_stream_info.clear();
  (void)ge::AttrUtils::SetStr(attached_stream, ATTR_NAME_ATTACHED_RESOURCE_NAME, "mc2");
  (void)ge::AttrUtils::SetStr(attached_stream, ge::ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, "as1");
  (void)ge::AttrUtils::SetBool(attached_stream, ge::ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG, true);
  attached_stream_info.emplace_back(attached_stream);
  (void)ge::AttrUtils::SetListNamedAttrs(relu2->GetOpDesc(), ge::ATTR_NAME_ATTACHED_STREAM_INFO_LIST,
                                         attached_stream_info);

  return graph;
}

/**
 * 适配V2场景，属性名字发生变化，采用V2版本表示新的属性方式;
 * 图中有两个通信域， 每个通信域内的节点设置了相同的attach stream名称， 两个通信域的attach stream名称不同
 * @return
// */
ComputeGraphPtr CreateGraph2_V2() {
  const auto &graph = MakeGraph();
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);

  std::vector<ge::NamedAttrs> attached_stream_info;
  ge::GeAttrValue::NAMED_ATTRS attached_stream;
  (void)ge::AttrUtils::SetStr(attached_stream, ATTR_NAME_ATTACHED_RESOURCE_NAME, "group0");
  (void)ge::AttrUtils::SetStr(attached_stream, ge::ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, "as0");
  (void)ge::AttrUtils::SetBool(attached_stream, ge::ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG, true);
  attached_stream_info.emplace_back(attached_stream);
  (void)ge::AttrUtils::SetListNamedAttrs(relu1->GetOpDesc(), ge::ATTR_NAME_ATTACHED_STREAM_INFO_LIST,
                                         attached_stream_info);

  attached_stream_info.clear();
  (void)ge::AttrUtils::SetStr(attached_stream, ATTR_NAME_ATTACHED_RESOURCE_NAME, "group0");
  (void)ge::AttrUtils::SetStr(attached_stream, ge::ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, "as0");
  (void)ge::AttrUtils::SetBool(attached_stream, ge::ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG, true);
  attached_stream_info.emplace_back(attached_stream);
  (void)ge::AttrUtils::SetListNamedAttrs(relu2->GetOpDesc(), ge::ATTR_NAME_ATTACHED_STREAM_INFO_LIST,
                                         attached_stream_info);

  auto relu3 = graph->FindNode("relu3");
  EXPECT_NE(relu3, nullptr);
  auto relu4 = graph->FindNode("relu4");
  EXPECT_NE(relu4, nullptr);
  NamedAttrs named_attrs2;

  attached_stream_info.clear();
  (void)ge::AttrUtils::SetStr(attached_stream, ATTR_NAME_ATTACHED_RESOURCE_NAME, "group1");
  (void)ge::AttrUtils::SetStr(attached_stream, ge::ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, "as1");
  (void)ge::AttrUtils::SetBool(attached_stream, ge::ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG, true);
  attached_stream_info.emplace_back(attached_stream);
  (void)ge::AttrUtils::SetListNamedAttrs(relu3->GetOpDesc(), ge::ATTR_NAME_ATTACHED_STREAM_INFO_LIST,
                                         attached_stream_info);

  attached_stream_info.clear();
  (void)ge::AttrUtils::SetStr(attached_stream, ATTR_NAME_ATTACHED_RESOURCE_NAME, "group1");
  (void)ge::AttrUtils::SetStr(attached_stream, ge::ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, "as1");
  (void)ge::AttrUtils::SetBool(attached_stream, ge::ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG, true);
  attached_stream_info.emplace_back(attached_stream);
  (void)ge::AttrUtils::SetListNamedAttrs(relu4->GetOpDesc(), ge::ATTR_NAME_ATTACHED_STREAM_INFO_LIST,
                                         attached_stream_info);
  return graph;
}

/**
 * 适配V2场景，属性名字发生变化，采用V2版本表示新的属性方式;
 * 图中有两个通信域， 所有的通信域中的节点设置了一样的attach stream名称
 * @return
 */
ComputeGraphPtr CreateGraph3_V2(const std::string &graph_name = "g1") {
  const auto &graph = MakeGraph();
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);

  std::vector<ge::NamedAttrs> attached_stream_info;
  ge::GeAttrValue::NAMED_ATTRS attached_stream;
  (void)ge::AttrUtils::SetStr(attached_stream, ATTR_NAME_ATTACHED_RESOURCE_NAME, "group0");
  (void)ge::AttrUtils::SetStr(attached_stream, ge::ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, "as0");
  (void)ge::AttrUtils::SetBool(attached_stream, ge::ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG, true);
  attached_stream_info.emplace_back(attached_stream);
  (void)ge::AttrUtils::SetListNamedAttrs(relu1->GetOpDesc(), ge::ATTR_NAME_ATTACHED_STREAM_INFO_LIST,
                                         attached_stream_info);
  (void)ge::AttrUtils::SetListNamedAttrs(relu2->GetOpDesc(), ge::ATTR_NAME_ATTACHED_STREAM_INFO_LIST,
                                         attached_stream_info);

  auto relu3 = graph->FindNode("relu3");
  EXPECT_NE(relu3, nullptr);
  auto relu4 = graph->FindNode("relu4");
  EXPECT_NE(relu4, nullptr);

  attached_stream_info.clear();
  (void)ge::AttrUtils::SetStr(attached_stream, ATTR_NAME_ATTACHED_RESOURCE_NAME, "group1");
  (void)ge::AttrUtils::SetStr(attached_stream, ge::ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, "as0");
  (void)ge::AttrUtils::SetBool(attached_stream, ge::ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG, true);
  attached_stream_info.emplace_back(attached_stream);

  (void)ge::AttrUtils::SetListNamedAttrs(relu3->GetOpDesc(), ge::ATTR_NAME_ATTACHED_STREAM_INFO_LIST,
                                         attached_stream_info);
  (void)ge::AttrUtils::SetListNamedAttrs(relu4->GetOpDesc(), ge::ATTR_NAME_ATTACHED_STREAM_INFO_LIST,
                                         attached_stream_info);
  graph->SetName(graph_name);
  return graph;
}

/**
 * 适配V2场景，属性名字发生变化，采用V2版本表示新的属性方式;
 * 两个节点归属的主流不同
 * @return
 */
ComputeGraphPtr CreateGraph4_V2() {
  const auto &graph = MakeGraph();
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);
  relu1->GetOpDesc()->SetStreamId(0U);
  relu2->GetOpDesc()->SetStreamId(1U);

  ge::GeAttrValue::NAMED_ATTRS attached_stream;
  (void)ge::AttrUtils::SetStr(attached_stream, ATTR_NAME_ATTACHED_RESOURCE_NAME, "group0");
  (void)ge::AttrUtils::SetStr(attached_stream, ge::ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, "as0");
  (void)ge::AttrUtils::SetBool(attached_stream, ge::ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG, true);
  (void)ge::AttrUtils::SetListNamedAttrs(relu1->GetOpDesc(), ge::ATTR_NAME_ATTACHED_STREAM_INFO_LIST,
                                         {attached_stream});
  (void)ge::AttrUtils::SetListNamedAttrs(relu2->GetOpDesc(), ge::ATTR_NAME_ATTACHED_STREAM_INFO_LIST,
                                         {attached_stream});
  return graph;
}

/**
 * v1控制算子子图，不应该分配从流
 * @return
 */
ComputeGraphPtr CreateV1CtrlGraph_V2() {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", DATA)->NODE("relu1", RELU)->NODE("netoutput", NETOUTPUT));
    CHAIN(NODE("data2", DATA)->NODE("relu2", RELU)->NODE("netoutput", NETOUTPUT));
    CHAIN(NODE("data3", DATA)->NODE("relu3", RELU)->NODE("merge", STREAMMERGE)->NODE("netoutput", NETOUTPUT));
    CHAIN(NODE("data4", DATA)->NODE("relu4", RELU)->NODE("merge", STREAMMERGE));
  };
  auto graph1 = ToGeGraph(g1);
  auto graph = ge::GraphUtilsEx::GetComputeGraph(graph1);
  graph->TopologicalSortingGraph();

  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);

  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);
  auto merge_op = graph->FindNode("merge");
  EXPECT_NE(merge_op, nullptr);

  NamedAttrs named_attrs;
  ge::GeAttrValue::NAMED_ATTRS attached_stream;
  (void)ge::AttrUtils::SetStr(attached_stream, ATTR_NAME_ATTACHED_RESOURCE_NAME, "group0");
  (void)ge::AttrUtils::SetStr(attached_stream, ge::ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, "as0");
  (void)ge::AttrUtils::SetBool(attached_stream, ge::ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG, true);
  (void)ge::AttrUtils::SetListNamedAttrs(relu1->GetOpDesc(), ge::ATTR_NAME_ATTACHED_STREAM_INFO_LIST,
                                         {attached_stream});
  (void)ge::AttrUtils::SetListNamedAttrs(relu2->GetOpDesc(), ge::ATTR_NAME_ATTACHED_STREAM_INFO_LIST,
                                         {attached_stream});

  EXPECT_TRUE(AttrUtils::SetBool(relu1->GetOpDesc(), "_disable_attached_resource", true));
  EXPECT_TRUE(AttrUtils::SetBool(relu2->GetOpDesc(), "_disable_attached_resource", true));

  return graph;
}

ComputeGraphPtr CreateGraphWithSubgraphs_V2(const std::vector<ComputeGraphPtr> &sub_graphs) {
  const size_t subgraph_num = sub_graphs.size();
  const std::string prefix = "partitioncall_";
  ComputeGraphPtr root_graph;
  DEF_GRAPH(g1) {
    for (size_t i = 0; i < subgraph_num; ++i) {
      CHAIN(NODE("data1", DATA)->NODE(prefix + std::to_string(i), PARTITIONEDCALL)->NODE("netoutput", NETOUTPUT));
    }
  };
  auto graph1 = ToGeGraph(g1);
  root_graph = ge::GraphUtilsEx::GetComputeGraph(graph1);

  size_t i = 0U;
  for (const auto &sub_graph : sub_graphs) {
    auto node_with_subgraph = root_graph->FindNode(prefix + std::to_string(i));
    EXPECT_NE(node_with_subgraph, nullptr);
    node_with_subgraph->GetOpDesc()->AddSubgraphName(sub_graph->GetName());
    node_with_subgraph->GetOpDesc()->SetSubgraphInstanceName(0, sub_graph->GetName());
    sub_graph->SetParentNode(node_with_subgraph);
    sub_graph->SetParentGraph(root_graph);
    root_graph->AddSubGraph(sub_graph);
    ++i;
  }
  EXPECT_EQ(root_graph->TopologicalSorting(), GRAPH_SUCCESS);
  return root_graph;
}

/**
 * 适配V2场景，属性名字发生变化，采用V2版本表示新的属性方式;
 * 资源申请的属性设置为false
 * @return
 */
ComputeGraphPtr CreateGraph5_V2() {
  const auto &graph = MakeGraph();
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);

  ge::GeAttrValue::NAMED_ATTRS attached_stream;
  (void)ge::AttrUtils::SetStr(attached_stream, ATTR_NAME_ATTACHED_RESOURCE_NAME, "group0");
  (void)ge::AttrUtils::SetStr(attached_stream, ge::ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, "as0");
  (void)ge::AttrUtils::SetBool(attached_stream, ge::ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG, false);
  (void)ge::AttrUtils::SetListNamedAttrs(relu1->GetOpDesc(), ge::ATTR_NAME_ATTACHED_STREAM_INFO_LIST,
                                         {attached_stream});
  (void)ge::AttrUtils::SetListNamedAttrs(relu2->GetOpDesc(), ge::ATTR_NAME_ATTACHED_STREAM_INFO_LIST,
                                         {attached_stream});
  return graph;
}

/**
 * 适配V2场景，属性名字发生变化，采用V2版本表示新的属性方式;
 * V1 V2场景均存在，需要兼容处理
 * @return
 */
ComputeGraphPtr CreateGraph6_V2() {
  const auto &graph = MakeGraph();
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);

  auto relu3 = graph->FindNode("relu3");
  EXPECT_NE(relu3, nullptr);
  auto relu4 = graph->FindNode("relu4");
  EXPECT_NE(relu4, nullptr);

  ge::GeAttrValue::NAMED_ATTRS attached_stream;
  (void)ge::AttrUtils::SetStr(attached_stream, ATTR_NAME_ATTACHED_RESOURCE_NAME, "group0");
  (void)ge::AttrUtils::SetStr(attached_stream, ge::ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, "as0");
  (void)ge::AttrUtils::SetBool(attached_stream, ge::ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG, true);
  (void)ge::AttrUtils::SetListNamedAttrs(relu1->GetOpDesc(), ge::ATTR_NAME_ATTACHED_STREAM_INFO_LIST,
                                         {attached_stream});
  (void)ge::AttrUtils::SetListNamedAttrs(relu2->GetOpDesc(), ge::ATTR_NAME_ATTACHED_STREAM_INFO_LIST,
                                         {attached_stream});

  NamedAttrs named_attrs;
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, ATTR_NAME_ATTACHED_STREAM_POLICY, GROUP_POLICY));
  EXPECT_TRUE(AttrUtils::SetStr(named_attrs, ATTR_NAME_ATTACHED_STREAM_KEY, "as0"));
  EXPECT_TRUE(AttrUtils::SetStr(relu3->GetOpDesc(), GROUP_POLICY, "group0"));
  EXPECT_TRUE(AttrUtils::SetStr(relu4->GetOpDesc(), GROUP_POLICY, "group0"));
  EXPECT_TRUE(AttrUtils::SetNamedAttrs(relu3->GetOpDesc(), ATTR_NAME_ATTACHED_STREAM_INFO, named_attrs));
  EXPECT_TRUE(AttrUtils::SetNamedAttrs(relu4->GetOpDesc(), ATTR_NAME_ATTACHED_STREAM_INFO, named_attrs));
  return graph;
}

/**
 * 适配V2场景，属性名字发生变化，采用V2版本表示新的属性方式;
 * 继承现有实现，reuse_key必须是不同的，附属流才会分配新的stream id
 * @return
 */
ComputeGraphPtr CreateGraph7_V2() {
  const auto &graph = MakeGraph();
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);

  ge::GeAttrValue::NAMED_ATTRS attached_stream;
  (void)ge::AttrUtils::SetStr(attached_stream, ATTR_NAME_ATTACHED_RESOURCE_NAME, "group0");
  (void)ge::AttrUtils::SetStr(attached_stream, ge::ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, "as0");
  (void)ge::AttrUtils::SetBool(attached_stream, ge::ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG, true);

  std::vector<ge::NamedAttrs> attached_stream_list;
  attached_stream_list.emplace_back(attached_stream);

  ge::GeAttrValue::NAMED_ATTRS attached_stream1;
  (void)ge::AttrUtils::SetStr(attached_stream1, ATTR_NAME_ATTACHED_RESOURCE_NAME, "group0");
  (void)ge::AttrUtils::SetStr(attached_stream1, ge::ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, "as1");
  attached_stream_list.emplace_back(attached_stream1);

  (void)ge::AttrUtils::SetListNamedAttrs(relu1->GetOpDesc(), ge::ATTR_NAME_ATTACHED_STREAM_INFO_LIST,
                                         attached_stream_list);
  return graph;
}

}  // namespace
TEST_F(UtestAssignAttachedStreamPass, Case0) {
  const auto graph = CreateGraph0();
  const auto &assign_attached_stream_pass = MakeShared<AssignAttachedStreamPass>();
  EXPECT_NE(assign_attached_stream_pass, nullptr);
  LogicalStreamPass::Context context;
  EXPECT_EQ((assign_attached_stream_pass->Run(graph, {}, context)), SUCCESS);

  // check 通信域内的节点才会分从流
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);
  auto relu3 = graph->FindNode("relu3");
  EXPECT_NE(relu3, nullptr);
  auto relu4 = graph->FindNode("relu4");
  EXPECT_NE(relu4, nullptr);
  // 通信域内的relu1,relu2 分配了一样的attached stream id
  EXPECT_EQ(relu1->GetOpDescBarePtr()->GetAttachedStreamId(), 0);
  EXPECT_EQ(relu2->GetOpDescBarePtr()->GetAttachedStreamId(), 0);
  // 通信域外的relu3, relu4 的attached stream id还是无效值
  EXPECT_EQ(relu3->GetOpDescBarePtr()->GetAttachedStreamId(), -1);
  EXPECT_EQ(relu4->GetOpDescBarePtr()->GetAttachedStreamId(), -1);
  // 总的流个数为1
  EXPECT_EQ(context.next_stream, 1);
}

TEST_F(UtestAssignAttachedStreamPass, Case1) {
  const auto graph = CreateGraph1();
  const auto &assign_attached_stream_pass = MakeShared<AssignAttachedStreamPass>();
  EXPECT_NE(assign_attached_stream_pass, nullptr);
  LogicalStreamPass::Context context;
  EXPECT_EQ((assign_attached_stream_pass->Run(graph, {}, context)), SUCCESS);

  // check 同一个通信域内的从流按照attach stream key来复用，attach stream key不一样的话，流也不一样
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);
  auto relu3 = graph->FindNode("relu3");
  EXPECT_NE(relu3, nullptr);
  auto relu4 = graph->FindNode("relu4");
  EXPECT_NE(relu4, nullptr);
  // 通信域内的relu1,relu2 分配了不一样的attached stream id
  EXPECT_NE(relu1->GetOpDescBarePtr()->GetAttachedStreamId(), -1);
  EXPECT_NE(relu2->GetOpDescBarePtr()->GetAttachedStreamId(), -1);
  EXPECT_NE(relu2->GetOpDescBarePtr()->GetAttachedStreamId(), relu1->GetOpDescBarePtr()->GetAttachedStreamId());
  // 通信域外的relu3, relu4 的attached stream id还是无效值
  EXPECT_EQ(relu3->GetOpDescBarePtr()->GetAttachedStreamId(), -1);
  EXPECT_EQ(relu4->GetOpDescBarePtr()->GetAttachedStreamId(), -1);
  // 总的流个数为2
  EXPECT_EQ(context.next_stream, 2);
}

TEST_F(UtestAssignAttachedStreamPass, Case2) {
  const auto graph = CreateGraph2();
  const auto &assign_attached_stream_pass = MakeShared<AssignAttachedStreamPass>();
  EXPECT_NE(assign_attached_stream_pass, nullptr);
  LogicalStreamPass::Context context;
  EXPECT_EQ((assign_attached_stream_pass->Run(graph, {}, context)), SUCCESS);
  // check 同一个通信域内的从流按照attach stream key来复用，attach stream key一样的话，流也一样
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);
  auto relu3 = graph->FindNode("relu3");
  EXPECT_NE(relu3, nullptr);
  auto relu4 = graph->FindNode("relu4");
  EXPECT_NE(relu4, nullptr);
  // 通信域g0内的relu1,relu2 分配了一样的attached stream id
  EXPECT_NE(relu1->GetOpDescBarePtr()->GetAttachedStreamId(), -1);
  EXPECT_NE(relu2->GetOpDescBarePtr()->GetAttachedStreamId(), -1);
  EXPECT_EQ(relu2->GetOpDescBarePtr()->GetAttachedStreamId(), relu1->GetOpDescBarePtr()->GetAttachedStreamId());
  // 通信域g1内的relu3,relu4 分配了一样的attached stream id
  EXPECT_NE(relu3->GetOpDescBarePtr()->GetAttachedStreamId(), -1);
  EXPECT_NE(relu4->GetOpDescBarePtr()->GetAttachedStreamId(), -1);
  EXPECT_EQ(relu3->GetOpDescBarePtr()->GetAttachedStreamId(), relu4->GetOpDescBarePtr()->GetAttachedStreamId());

  // 两个通信域的流不同
  EXPECT_NE(relu1->GetOpDescBarePtr()->GetAttachedStreamId(), relu3->GetOpDescBarePtr()->GetAttachedStreamId());
  // 总的流个数为2
  EXPECT_EQ(context.next_stream, 2);
}

TEST_F(UtestAssignAttachedStreamPass, Case3) {
  const auto graph = CreateGraph3();
  const auto &assign_attached_stream_pass = MakeShared<AssignAttachedStreamPass>();
  EXPECT_NE(assign_attached_stream_pass, nullptr);
  LogicalStreamPass::Context context;
  EXPECT_EQ((assign_attached_stream_pass->Run(graph, {}, context)), SUCCESS);

  // check 不同的通信域的从流不复用
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);
  auto relu3 = graph->FindNode("relu3");
  EXPECT_NE(relu3, nullptr);
  auto relu4 = graph->FindNode("relu4");
  EXPECT_NE(relu4, nullptr);
  // 通信域g0内的relu1,relu2 分配了一样的attached stream id
  EXPECT_NE(relu1->GetOpDescBarePtr()->GetAttachedStreamId(), -1);
  EXPECT_NE(relu2->GetOpDescBarePtr()->GetAttachedStreamId(), -1);
  EXPECT_EQ(relu2->GetOpDescBarePtr()->GetAttachedStreamId(), relu1->GetOpDescBarePtr()->GetAttachedStreamId());
  // 通信域g1内的relu3,relu4 分配了一样的attached stream id
  EXPECT_NE(relu3->GetOpDescBarePtr()->GetAttachedStreamId(), -1);
  EXPECT_NE(relu4->GetOpDescBarePtr()->GetAttachedStreamId(), -1);
  EXPECT_EQ(relu3->GetOpDescBarePtr()->GetAttachedStreamId(), relu4->GetOpDescBarePtr()->GetAttachedStreamId());

  // 两个通信域的流不同， 虽然通信域的attached stream key一样
  EXPECT_NE(relu1->GetOpDescBarePtr()->GetAttachedStreamId(), relu3->GetOpDescBarePtr()->GetAttachedStreamId());
  // 总的流个数为2
  EXPECT_EQ(context.next_stream, 2);
}

TEST_F(UtestAssignAttachedStreamPass, Case3_With_SubGraph) {
  const auto graph = CreateGraph3();
  const auto graph2 = CreateGraph3("g2");
  const auto root_graph = CreateGraphWithSubgraphs({graph, graph2});
  const std::map<std::string, int32_t> max_parallel_num;
  const auto &passes = MakeShared<LogicalStreamAllocator>(max_parallel_num);
  EXPECT_NE(passes, nullptr);
  int64_t total_stream_num = 0;
  int64_t main_stream_num = 0;
  Graph2SubGraphInfoList graph_2_sub_graph_info_list{{root_graph, {}}, {graph, {}}, {graph2, {}}};
  EXPECT_EQ((passes->Assign(root_graph, graph_2_sub_graph_info_list, total_stream_num, main_stream_num)), SUCCESS);

  auto check_func = [](const ComputeGraphPtr &graph) {
    auto relu1 = graph->FindNode("relu1");
    EXPECT_NE(relu1, nullptr);
    auto relu2 = graph->FindNode("relu2");
    EXPECT_NE(relu2, nullptr);
    auto relu3 = graph->FindNode("relu3");
    EXPECT_NE(relu3, nullptr);
    auto relu4 = graph->FindNode("relu4");
    EXPECT_NE(relu4, nullptr);
    // 通信域g0内的relu1,relu2 分配了一样的attached stream id
    EXPECT_NE(relu1->GetOpDescBarePtr()->GetAttachedStreamId(), -1);
    EXPECT_NE(relu2->GetOpDescBarePtr()->GetAttachedStreamId(), -1);
    EXPECT_EQ(relu2->GetOpDescBarePtr()->GetAttachedStreamId(), relu1->GetOpDescBarePtr()->GetAttachedStreamId());
    // 通信域g1内的relu3,relu4 分配了一样的attached stream id
    EXPECT_NE(relu3->GetOpDescBarePtr()->GetAttachedStreamId(), -1);
    EXPECT_NE(relu4->GetOpDescBarePtr()->GetAttachedStreamId(), -1);
    EXPECT_EQ(relu3->GetOpDescBarePtr()->GetAttachedStreamId(), relu4->GetOpDescBarePtr()->GetAttachedStreamId());

    // 两个通信域的流不同， 虽然通信域的attached stream key一样
    EXPECT_NE(relu1->GetOpDescBarePtr()->GetAttachedStreamId(), relu3->GetOpDescBarePtr()->GetAttachedStreamId());
  };
  check_func(graph);
  check_func(graph2);
  // 总的流个数为2+2
  EXPECT_EQ(total_stream_num, 4);
}

TEST_F(UtestAssignAttachedStreamPass, Case4) {
  const auto graph = CreateGraph4();
  const auto &assign_attached_stream_pass = MakeShared<AssignAttachedStreamPass>();
  EXPECT_NE(assign_attached_stream_pass, nullptr);
  LogicalStreamPass::Context context;
  EXPECT_EQ((assign_attached_stream_pass->Run(graph, {}, context)), SUCCESS);

  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);
  auto relu3 = graph->FindNode("relu3");
  EXPECT_NE(relu3, nullptr);
  auto relu4 = graph->FindNode("relu4");
  EXPECT_NE(relu4, nullptr);
  // 通信域内的relu1,relu2 分配了不一样的attached stream id，key一样但主流不一样
  EXPECT_NE(relu1->GetOpDescBarePtr()->GetAttachedStreamId(), -1);
  EXPECT_NE(relu2->GetOpDescBarePtr()->GetAttachedStreamId(), -1);
  EXPECT_NE(relu2->GetOpDescBarePtr()->GetAttachedStreamId(), relu1->GetOpDescBarePtr()->GetAttachedStreamId());
  // 通信域外的relu3, relu4 的attached stream id还是无效值
  EXPECT_EQ(relu3->GetOpDescBarePtr()->GetAttachedStreamId(), -1);
  EXPECT_EQ(relu4->GetOpDescBarePtr()->GetAttachedStreamId(), -1);
  // 总的流个数为2
  EXPECT_EQ(context.next_stream, 2);
}

TEST_F(UtestAssignAttachedStreamPass, SkipAssignForV1CtrlOp) {
  const auto graph = CreateV1CtrlGraph();
  const auto &assign_attached_stream_pass = MakeShared<AssignAttachedStreamPass>();
  EXPECT_NE(assign_attached_stream_pass, nullptr);
  LogicalStreamPass::Context context;
  EXPECT_EQ((assign_attached_stream_pass->Run(graph, {}, context)), SUCCESS);

  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);
  auto relu3 = graph->FindNode("relu3");
  EXPECT_NE(relu3, nullptr);
  auto relu4 = graph->FindNode("relu4");
  EXPECT_NE(relu4, nullptr);

  EXPECT_EQ(relu1->GetOpDescBarePtr()->GetAttachedStreamId(), -1);
  EXPECT_EQ(relu2->GetOpDescBarePtr()->GetAttachedStreamId(), -1);
  EXPECT_EQ(relu3->GetOpDescBarePtr()->GetAttachedStreamId(), -1);
  EXPECT_EQ(relu4->GetOpDescBarePtr()->GetAttachedStreamId(), -1);
  EXPECT_EQ(context.next_stream, 0);
}

TEST_F(UtestAssignAttachedStreamPass, Case0_V2) {
  const auto graph = CreateGraph0_V2();
  const auto &assign_attached_stream_pass = MakeShared<AssignAttachedStreamPass>();
  EXPECT_NE(assign_attached_stream_pass, nullptr);
  LogicalStreamPass::Context context;
  EXPECT_EQ((assign_attached_stream_pass->Run(graph, {}, context)), SUCCESS);

  // check 通信域内的节点才会分从流
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);
  auto relu3 = graph->FindNode("relu3");
  EXPECT_NE(relu3, nullptr);
  auto relu4 = graph->FindNode("relu4");
  EXPECT_NE(relu4, nullptr);
  // 通信域内的relu1,relu2 分配了一样的attached stream id
  EXPECT_EQ(relu1->GetOpDescBarePtr()->GetAttachedStreamIds().size(), 1);
  EXPECT_EQ(relu1->GetOpDescBarePtr()->GetAttachedStreamIds().at(0), 0);

  EXPECT_EQ(relu2->GetOpDescBarePtr()->GetAttachedStreamIds().size(), 1);
  EXPECT_EQ(relu2->GetOpDescBarePtr()->GetAttachedStreamIds().at(0), 0);

  // 通信域外的relu3, relu4 的attached stream id还是无效值
  EXPECT_EQ(relu3->GetOpDescBarePtr()->GetAttachedStreamIds().size(), 0);

  EXPECT_EQ(relu4->GetOpDescBarePtr()->GetAttachedStreamIds().size(), 0);

  // 总的流个数为1
  EXPECT_EQ(context.next_stream, 1);
}


TEST_F(UtestAssignAttachedStreamPass, Case1_V2) {
  const auto graph = CreateGraph1_V2();
  const auto &assign_attached_stream_pass = MakeShared<AssignAttachedStreamPass>();
  EXPECT_NE(assign_attached_stream_pass, nullptr);
  LogicalStreamPass::Context context;
  EXPECT_EQ((assign_attached_stream_pass->Run(graph, {}, context)), SUCCESS);

  // check 同一个通信域内的从流按照attach stream key来复用，attach stream key不一样的话，流也不一样
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);
  auto relu3 = graph->FindNode("relu3");
  EXPECT_NE(relu3, nullptr);
  auto relu4 = graph->FindNode("relu4");
  EXPECT_NE(relu4, nullptr);
  // 通信域内的relu1,relu2 分配了不一样的attached stream id
  EXPECT_EQ(relu1->GetOpDescBarePtr()->GetAttachedStreamIds().size(), 1);
  EXPECT_EQ(relu2->GetOpDescBarePtr()->GetAttachedStreamIds().size(), 1);
  EXPECT_NE(relu2->GetOpDescBarePtr()->GetAttachedStreamIds()[0], relu1->GetOpDescBarePtr()->GetAttachedStreamIds()[0]);
  // 通信域外的relu3, relu4 的attached stream id还是无效值
  EXPECT_EQ(relu3->GetOpDescBarePtr()->GetAttachedStreamIds().size(), 0);
  EXPECT_EQ(relu4->GetOpDescBarePtr()->GetAttachedStreamIds().size(), 0);
  // 总的流个数为2
  EXPECT_EQ(context.next_stream, 2);
}

TEST_F(UtestAssignAttachedStreamPass, Case2_V2) {
  const auto graph = CreateGraph2_V2();
  const auto &assign_attached_stream_pass = MakeShared<AssignAttachedStreamPass>();
  EXPECT_NE(assign_attached_stream_pass, nullptr);
  LogicalStreamPass::Context context;
  EXPECT_EQ((assign_attached_stream_pass->Run(graph, {}, context)), SUCCESS);
  // check 同一个通信域内的从流按照attach stream key来复用，attach stream key一样的话，流也一样
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);
  auto relu3 = graph->FindNode("relu3");
  EXPECT_NE(relu3, nullptr);
  auto relu4 = graph->FindNode("relu4");
  EXPECT_NE(relu4, nullptr);
  // 通信域g0内的relu1,relu2 分配了一样的attached stream id
  EXPECT_EQ(relu1->GetOpDescBarePtr()->GetAttachedStreamIds().size(), 1);
  EXPECT_EQ(relu2->GetOpDescBarePtr()->GetAttachedStreamIds().size(), 1);
  EXPECT_EQ(relu2->GetOpDescBarePtr()->GetAttachedStreamIds()[0], relu1->GetOpDescBarePtr()->GetAttachedStreamIds()[0]);
  // 通信域g1内的relu3,relu4 分配了一样的attached stream id
  EXPECT_EQ(relu3->GetOpDescBarePtr()->GetAttachedStreamIds().size(), 1);
  EXPECT_EQ(relu4->GetOpDescBarePtr()->GetAttachedStreamIds().size(), 1);
  EXPECT_EQ(relu3->GetOpDescBarePtr()->GetAttachedStreamIds()[0], relu4->GetOpDescBarePtr()->GetAttachedStreamIds()[0]);

  // 两个通信域的流不同
  EXPECT_NE(relu1->GetOpDescBarePtr()->GetAttachedStreamIds()[0], relu3->GetOpDescBarePtr()->GetAttachedStreamIds()[0]);
  // 总的流个数为2
  EXPECT_EQ(context.next_stream, 2);
}

TEST_F(UtestAssignAttachedStreamPass, Case3_V2) {
  const auto graph = CreateGraph3_V2();
  const auto &assign_attached_stream_pass = MakeShared<AssignAttachedStreamPass>();
  EXPECT_NE(assign_attached_stream_pass, nullptr);
  LogicalStreamPass::Context context;
  EXPECT_EQ((assign_attached_stream_pass->Run(graph, {}, context)), SUCCESS);

  // check 不同的通信域的从流不复用
  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);
  auto relu3 = graph->FindNode("relu3");
  EXPECT_NE(relu3, nullptr);
  auto relu4 = graph->FindNode("relu4");
  EXPECT_NE(relu4, nullptr);
  // 通信域g0内的relu1,relu2 分配了一样的attached stream id
  EXPECT_EQ(relu1->GetOpDescBarePtr()->GetAttachedStreamIds().size(), 1);
  EXPECT_EQ(relu2->GetOpDescBarePtr()->GetAttachedStreamIds().size(), 1);
  EXPECT_EQ(relu2->GetOpDescBarePtr()->GetAttachedStreamIds()[0], relu1->GetOpDescBarePtr()->GetAttachedStreamIds()[0]);

  // 通信域g1内的relu3,relu4 分配了一样的attached stream id
  EXPECT_EQ(relu3->GetOpDescBarePtr()->GetAttachedStreamIds().size(), 1);
  EXPECT_EQ(relu4->GetOpDescBarePtr()->GetAttachedStreamIds().size(), 1);
  EXPECT_EQ(relu3->GetOpDescBarePtr()->GetAttachedStreamIds()[0], relu4->GetOpDescBarePtr()->GetAttachedStreamIds()[0]);

  // 两个通信域的流不同， 虽然通信域的attached stream key一样
  EXPECT_NE(relu1->GetOpDescBarePtr()->GetAttachedStreamIds()[0], relu3->GetOpDescBarePtr()->GetAttachedStreamIds()[0]);
  // 总的流个数为2
  EXPECT_EQ(context.next_stream, 2);
}

TEST_F(UtestAssignAttachedStreamPass, Case3_With_SubGraph_V2) {
  const auto graph = CreateGraph3_V2();
  const auto graph2 = CreateGraph3_V2("g2");
  const auto root_graph = CreateGraphWithSubgraphs_V2({graph, graph2});
  const std::map<std::string, int32_t> max_parallel_num;
  const auto &passes = MakeShared<LogicalStreamAllocator>(max_parallel_num);
  EXPECT_NE(passes, nullptr);
  int64_t total_stream_num = 0;
  int64_t main_stream_num = 0;
  Graph2SubGraphInfoList graph_2_sub_graph_info_list{{root_graph, {}}, {graph, {}}, {graph2, {}}};
  EXPECT_EQ((passes->Assign(root_graph, graph_2_sub_graph_info_list, total_stream_num, main_stream_num)), SUCCESS);

  auto check_func = [](const ComputeGraphPtr &graph) {
    auto relu1 = graph->FindNode("relu1");
    EXPECT_NE(relu1, nullptr);
    auto relu2 = graph->FindNode("relu2");
    EXPECT_NE(relu2, nullptr);
    auto relu3 = graph->FindNode("relu3");
    EXPECT_NE(relu3, nullptr);
    auto relu4 = graph->FindNode("relu4");
    EXPECT_NE(relu4, nullptr);
    // 通信域g0内的relu1,relu2 分配了一样的attached stream id
    EXPECT_EQ(relu1->GetOpDescBarePtr()->GetAttachedStreamIds().size(), 1);
    EXPECT_EQ(relu2->GetOpDescBarePtr()->GetAttachedStreamIds().size(), 1);
    EXPECT_EQ(relu2->GetOpDescBarePtr()->GetAttachedStreamIds()[0], relu1->GetOpDescBarePtr()->GetAttachedStreamIds()[0]);
    // 通信域g1内的relu3,relu4 分配了一样的attached stream id
    EXPECT_EQ(relu3->GetOpDescBarePtr()->GetAttachedStreamIds().size(), 1);
    EXPECT_EQ(relu4->GetOpDescBarePtr()->GetAttachedStreamIds().size(), 1);
    EXPECT_EQ(relu3->GetOpDescBarePtr()->GetAttachedStreamIds()[0], relu4->GetOpDescBarePtr()->GetAttachedStreamIds()[0]);

    // 两个通信域的流不同， 虽然通信域的attached stream key一样
    EXPECT_NE(relu1->GetOpDescBarePtr()->GetAttachedStreamIds()[0], relu3->GetOpDescBarePtr()->GetAttachedStreamIds()[0]);
  };
  check_func(graph);
  check_func(graph2);
  // 总的流个数为2+2
  EXPECT_EQ(total_stream_num, 4);
}

TEST_F(UtestAssignAttachedStreamPass, Case4_V2) {
  const auto graph = CreateGraph4_V2();
  const auto &assign_attached_stream_pass = MakeShared<AssignAttachedStreamPass>();
  EXPECT_NE(assign_attached_stream_pass, nullptr);
  LogicalStreamPass::Context context;
  EXPECT_EQ((assign_attached_stream_pass->Run(graph, {}, context)), SUCCESS);

  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);
  auto relu3 = graph->FindNode("relu3");
  EXPECT_NE(relu3, nullptr);
  auto relu4 = graph->FindNode("relu4");
  EXPECT_NE(relu4, nullptr);
  // 通信域内的relu1,relu2 分配了不一样的attached stream id，key一样但主流不一样
  EXPECT_EQ(relu1->GetOpDescBarePtr()->GetAttachedStreamIds().size(), 1);
  EXPECT_EQ(relu2->GetOpDescBarePtr()->GetAttachedStreamIds().size(), 1);
  EXPECT_NE(relu2->GetOpDescBarePtr()->GetAttachedStreamIds()[0], relu1->GetOpDescBarePtr()->GetAttachedStreamIds()[0]);
  // 通信域外的relu3, relu4 的attached stream id还是无效值
  EXPECT_EQ(relu3->GetOpDescBarePtr()->GetAttachedStreamIds().size(), 0);
  EXPECT_EQ(relu4->GetOpDescBarePtr()->GetAttachedStreamIds().size(), 0);
  // 总的流个数为2
  EXPECT_EQ(context.next_stream, 2);
}

TEST_F(UtestAssignAttachedStreamPass, SkipAssignForV1CtrlOp_V2) {
  const auto graph = CreateV1CtrlGraph_V2();
  const auto &assign_attached_stream_pass = MakeShared<AssignAttachedStreamPass>();
  EXPECT_NE(assign_attached_stream_pass, nullptr);
  LogicalStreamPass::Context context;
  EXPECT_EQ((assign_attached_stream_pass->Run(graph, {}, context)), SUCCESS);

  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);
  auto relu3 = graph->FindNode("relu3");
  EXPECT_NE(relu3, nullptr);
  auto relu4 = graph->FindNode("relu4");
  EXPECT_NE(relu4, nullptr);

  EXPECT_EQ(relu1->GetOpDescBarePtr()->GetAttachedStreamIds().size(), 1);
  EXPECT_EQ(relu1->GetOpDescBarePtr()->GetAttachedStreamIds()[0], -1);

  EXPECT_EQ(relu2->GetOpDescBarePtr()->GetAttachedStreamIds().size(), 1);
  EXPECT_EQ(relu2->GetOpDescBarePtr()->GetAttachedStreamIds()[0], -1);

  EXPECT_EQ(relu3->GetOpDescBarePtr()->GetAttachedStreamIds().size(), 0);
  EXPECT_EQ(relu4->GetOpDescBarePtr()->GetAttachedStreamIds().size(), 0);
  EXPECT_EQ(context.next_stream, 0);
}

TEST_F(UtestAssignAttachedStreamPass, Not_Required_Source_V2) {
  const auto graph = CreateGraph5_V2();
  const auto &assign_attached_stream_pass = MakeShared<AssignAttachedStreamPass>();
  EXPECT_NE(assign_attached_stream_pass, nullptr);
  LogicalStreamPass::Context context;
  EXPECT_EQ((assign_attached_stream_pass->Run(graph, {}, context)), SUCCESS);

  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);

  auto relu3 = graph->FindNode("relu3");
  EXPECT_NE(relu3, nullptr);
  auto relu4 = graph->FindNode("relu4");
  EXPECT_NE(relu4, nullptr);

  EXPECT_EQ(relu1->GetOpDescBarePtr()->GetAttachedStreamIds().size(), 1);
  EXPECT_EQ(relu1->GetOpDescBarePtr()->GetAttachedStreamIds()[0], -1);

  EXPECT_EQ(relu2->GetOpDescBarePtr()->GetAttachedStreamIds().size(), 1);
  EXPECT_EQ(relu2->GetOpDescBarePtr()->GetAttachedStreamIds()[0], -1);

  EXPECT_EQ(relu3->GetOpDescBarePtr()->GetAttachedStreamIds().size(), 0);
  EXPECT_EQ(relu4->GetOpDescBarePtr()->GetAttachedStreamIds().size(), 0);
  EXPECT_EQ(context.next_stream, 0);
}

/*
 * 兼容场景，老的v1流程跟新的v2流程均存在
 * */
TEST_F(UtestAssignAttachedStreamPass, Mix_V1_V2) {
  const auto graph = CreateGraph6_V2();
  const auto &assign_attached_stream_pass = MakeShared<AssignAttachedStreamPass>();
  EXPECT_NE(assign_attached_stream_pass, nullptr);
  LogicalStreamPass::Context context;
  EXPECT_EQ((assign_attached_stream_pass->Run(graph, {}, context)), SUCCESS);

  auto relu1 = graph->FindNode("relu1");
  EXPECT_NE(relu1, nullptr);
  auto relu2 = graph->FindNode("relu2");
  EXPECT_NE(relu2, nullptr);

  auto relu3 = graph->FindNode("relu3");
  EXPECT_NE(relu3, nullptr);
  auto relu4 = graph->FindNode("relu4");
  EXPECT_NE(relu4, nullptr);

  EXPECT_EQ(relu1->GetOpDescBarePtr()->GetAttachedStreamIds().size(), 1);
  EXPECT_EQ(relu1->GetOpDescBarePtr()->GetAttachedStreamIds()[0], 1);

  EXPECT_EQ(relu2->GetOpDescBarePtr()->GetAttachedStreamIds().size(), 1);
  EXPECT_EQ(relu2->GetOpDescBarePtr()->GetAttachedStreamIds()[0], 1);

  EXPECT_EQ(relu3->GetOpDescBarePtr()->GetAttachedStreamIds().size(), 1);
  EXPECT_EQ(relu3->GetOpDescBarePtr()->GetAttachedStreamIds()[0], 0);

  EXPECT_EQ(relu4->GetOpDescBarePtr()->GetAttachedStreamIds().size(), 1);
  EXPECT_EQ(relu4->GetOpDescBarePtr()->GetAttachedStreamIds()[0], 0);

  EXPECT_EQ(context.next_stream, 2);
}

TEST_F(UtestAssignAttachedStreamPass, SupportMutliAttachedStream) {
  dlog_setlevel(0,0,0);
  const auto graph = CreateGraph7_V2();
  const auto &assign_attached_stream_pass = MakeShared<AssignAttachedStreamPass>();
  EXPECT_NE(assign_attached_stream_pass, nullptr);
  LogicalStreamPass::Context context;
  EXPECT_EQ((assign_attached_stream_pass->Run(graph, {}, context)), SUCCESS);

  auto relu1 = graph->FindNode("relu1");
  EXPECT_EQ(relu1->GetOpDesc()->GetAttachedStreamIds().size(), 2);
}

/*
 * 某个op_desc在分attached_stream的pass里面，打了attached_resource属性
 * */
TEST_F(UtestAssignAttachedStreamPass, InvalidCaseV2) {
  const auto graph = CreateGraph0_V2();
  auto relu3 = graph->FindNode("relu3");
  EXPECT_NE(relu3, nullptr);
  std::vector<ge::NamedAttrs> attached_resource_info_list;
  ge::GeAttrValue::NAMED_ATTRS attached_resource;
  (void)ge::AttrUtils::SetStr(attached_resource, ATTR_NAME_ATTACHED_RESOURCE_NAME, "tiling");
  (void)ge::AttrUtils::SetStr(attached_resource, ge::ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, "tiling");
  (void)ge::AttrUtils::SetInt(attached_resource, ge::ATTR_NAME_ATTACHED_RESOURCE_TYPE, 0);
  (void)ge::AttrUtils::SetBool(attached_resource, ge::ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG, true);
  attached_resource_info_list.emplace_back(attached_resource);
  (void)ge::AttrUtils::SetListNamedAttrs(relu3->GetOpDesc(), ge::ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST,
                                         attached_resource_info_list);

  const auto &assign_attached_stream_pass = MakeShared<AssignAttachedStreamPass>();
  EXPECT_NE(assign_attached_stream_pass, nullptr);
  LogicalStreamPass::Context context;
  EXPECT_EQ((assign_attached_stream_pass->Run(graph, {}, context)), SUCCESS);
  // 无效的
  EXPECT_EQ(relu3->GetOpDescBarePtr()->GetAttachedStreamIds().size(), 0);

  // 总的流个数为1
  EXPECT_EQ(context.next_stream, 1);
}

}  // namespace ge

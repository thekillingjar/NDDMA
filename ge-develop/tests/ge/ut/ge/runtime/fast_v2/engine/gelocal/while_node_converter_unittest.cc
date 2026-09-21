/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "engine/gelocal/while_converter.h"
#include "engine/gelocal/subgraph_io_converter.h"

#include <gtest/gtest.h>

#include "ge_graph_dsl/graph_dsl.h"
#include "engine/gelocal/inputs_converter.h"
#include "graph/utils/graph_utils.h"
#include "graph_builder/exe_graph_comparer.h"
#include "common/share_graph.h"
#include "faker/global_data_faker.h"
#include "common/bg_test.h"
#include "lowering/graph_converter.h"
#include "op_impl/less_important_op_impl.h"

#include "register/op_impl_registry.h"
#include "framework/common/types.h"
#include "graph/utils/graph_dump_utils.h"

using namespace ge;
namespace gert {
IMPL_OP(Foo);
IMPL_OP(Bar);

class WhileNodeConverterUT : public bg::BgTest {};
TEST_F(WhileNodeConverterUT, ConvertWhileNode) {
  for (auto &main_graph : {ShareGraph::WhileGraph(), ShareGraph::WhileGraph(true)}) {
    ASSERT_NE(main_graph, nullptr);
    auto while_node = main_graph->FindFirstNodeMatchType("While");
    ASSERT_NE(while_node, nullptr);

    MockLessImportantNodeConverter(main_graph, {ge::WHILE, ge::DATA, ge::NETOUTPUT});
    auto root_model = GeModelBuilder(main_graph).BuildGeRootModel();
    auto global_data = GlobalDataFaker(root_model).Build();

    main_graph->TopologicalSorting();
    ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
    model_desc_holder.SetSpaceRegistry(SpaceRegistryFaker().Build());
    auto lower_3stage_graph = GraphConverter()
                                  .SetModelDescHolder(&model_desc_holder)
                                  .ConvertComputeGraphToExecuteGraph(main_graph, global_data);
    ASSERT_NE(lower_3stage_graph, nullptr);
    auto lower_graph = ge::FastNodeUtils::GetSubgraphFromNode(
        ge::ExecuteGraphUtils::FindFirstNodeMatchType(lower_3stage_graph.get(), "Main"), 0);
    ASSERT_NE(lower_graph, nullptr);
    auto lower_while_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(lower_graph, "While");
    ASSERT_NE(lower_while_node, nullptr);

    auto subgraph_names = lower_while_node->GetOpDescBarePtr()->GetSubgraphNameIndexes();
    std::map<uint32_t, std::string> sorted_subgraph_names;
    for (auto &item : subgraph_names) {
      sorted_subgraph_names[item.second] = item.first;
    }
    ASSERT_EQ(sorted_subgraph_names.size(), 2);
    auto iter = sorted_subgraph_names.begin();
    EXPECT_EQ((iter++)->second, "control_frame");
    EXPECT_EQ(iter->second, "body");

    auto control_frame_graph = ge::FastNodeUtils::GetSubgraphFromNode(lower_while_node, 0);
    ASSERT_NE(control_frame_graph, nullptr);
    auto subgraph_call = ge::ExecuteGraphUtils::FindFirstNodeMatchType(control_frame_graph, "SubgraphCall");
    ASSERT_NE(subgraph_call, nullptr);

    auto body_graph =  ge::FastNodeUtils::GetSubgraphFromNode(lower_while_node, 1);
    ASSERT_NE(body_graph, nullptr);
    auto identity_shape_and_addr = ge::ExecuteGraphUtils::FindFirstNodeMatchType(body_graph, "IdentityShapeAndAddr");
    ASSERT_NE(identity_shape_and_addr, nullptr);

    ASSERT_EQ(identity_shape_and_addr->GetDataInNum(), 2U * while_node->GetOutDataNodesSize());
    ASSERT_EQ(identity_shape_and_addr->GetDataOutNum(), 2U * while_node->GetOutDataNodesSize());

    subgraph_names = subgraph_call->GetOpDescBarePtr()->GetSubgraphNameIndexes();
    sorted_subgraph_names.clear();
    for (auto &item : subgraph_names) {
      sorted_subgraph_names[item.second] = item.first;
    }
    ASSERT_EQ(sorted_subgraph_names.size(), 2);
    iter = sorted_subgraph_names.begin();
    EXPECT_EQ(sorted_subgraph_names.size(), 2);
    EXPECT_EQ((iter++)->second, "control_frame");
    EXPECT_EQ(iter->second, "f");

    auto cond_graph = ge::FastNodeUtils::GetSubgraphFromNode(subgraph_call, 1);
    auto while_cond = ge::ExecuteGraphUtils::FindFirstNodeMatchType(cond_graph, "GenCondForWhile");
    ASSERT_NE(while_cond, nullptr);

    ge::DumpGraph(lower_3stage_graph.get(), "WhileExecGraph");
  }
}

TEST_F(WhileNodeConverterUT, ConvertWhileNodeXBody) {
  for (auto &main_graph : {ShareGraph::WhileGraphXBody()}) {
    ASSERT_NE(main_graph, nullptr);
    auto while_node = main_graph->FindFirstNodeMatchType("While");
    ASSERT_NE(while_node, nullptr);

    MockLessImportantNodeConverter(main_graph, {ge::WHILE, ge::DATA, ge::NETOUTPUT}, {}, true);
    auto root_model = GeModelBuilder(main_graph).BuildGeRootModel();
    auto global_data = GlobalDataFaker(root_model).Build();

    main_graph->TopologicalSorting();
    ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
    model_desc_holder.SetSpaceRegistry(SpaceRegistryFaker().Build());
    auto lower_3stage_graph = GraphConverter()
                                  .SetModelDescHolder(&model_desc_holder)
                                  .ConvertComputeGraphToExecuteGraph(main_graph, global_data);
    ASSERT_NE(lower_3stage_graph, nullptr);
    auto lower_graph = ge::FastNodeUtils::GetSubgraphFromNode(
        ge::ExecuteGraphUtils::FindFirstNodeMatchType(lower_3stage_graph.get(), "Main"), 0);
    ASSERT_NE(lower_graph, nullptr);
    auto lower_while_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(lower_graph, "While");
    ASSERT_NE(lower_while_node, nullptr);

    auto subgraph_names = lower_while_node->GetOpDescBarePtr()->GetSubgraphNameIndexes();
    std::map<uint32_t, std::string> sorted_subgraph_names;
    for (auto &item : subgraph_names) {
      sorted_subgraph_names[item.second] = item.first;
    }
    ASSERT_EQ(sorted_subgraph_names.size(), 2);
    EXPECT_EQ(sorted_subgraph_names.rbegin()->second, "body");

    auto body_graph = ge::FastNodeUtils::GetSubgraphFromNode(lower_while_node, 1);
    ASSERT_NE(body_graph, nullptr);
    auto identity_shape_and_addr = ge::ExecuteGraphUtils::FindFirstNodeMatchType(body_graph, "IdentityShapeAndAddr");
    ASSERT_NE(identity_shape_and_addr, nullptr);

    auto data_nodes = identity_shape_and_addr->GetInDataNodes();
    std::vector<ge::FastNode*> data_nodes_vec(data_nodes.begin(), data_nodes.end());
    ASSERT_EQ(data_nodes_vec.size(), 2 * 2); // 2 * num_inputs

    for (size_t i = 0; i < 2; ++i) {
      ASSERT_EQ(data_nodes_vec[i]->GetType(), "IdentityShape");
      ASSERT_EQ(data_nodes_vec[i + 2]->GetType(), "IdentityAddr");
    }

    auto identity_addr = data_nodes_vec[2];
    auto free_nodes = identity_addr->GetAllOutNodes();
    std::vector<ge::FastNode*> free_nodes_vec(free_nodes.begin(), free_nodes.end());
    ASSERT_EQ(free_nodes_vec.size(), 2 + 2); // 2 addr and 2 free
    size_t num_addr = 0U;
    size_t num_free = 0U;
    for (auto &node : free_nodes_vec) {
      if (node->GetType() == "IdentityShapeAndAddr") {
        num_addr++;
      } else if (node->GetType() == "FreeMemory") {
        num_free++;
      } else {
        ASSERT_EQ(node->GetType(), node->GetType() + ":Unexpect");
      }
    }
    ASSERT_EQ(num_addr, 2U);
    ASSERT_EQ(num_free, 2U);
  }
}

/*
 * Netoutput of while body has input node(Add) from other stream
 * There will be an AccessMemCrossStream between Add and netoutput
 */

TEST_F(WhileNodeConverterUT, ConvertWhileNode_multi_stream_inside_body) {
  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistry(true);
  for (auto &main_graph : {ShareGraph::WhileGraph(), ShareGraph::WhileGraph(true)}) {
    ASSERT_NE(main_graph, nullptr);
    auto while_node = main_graph->FindFirstNodeMatchType("While");
    ASSERT_NE(while_node, nullptr);

    // assign add node inside body to stream1
    const auto &body_graph_name = while_node->GetOpDescBarePtr()->GetSubgraphInstanceName(1);
    auto compute_body_graph = main_graph->GetSubgraph(body_graph_name);
    auto add_node = compute_body_graph->FindFirstNodeMatchType(ADD);
    add_node->GetOpDescBarePtr()->SetStreamId(1);

    MockLessImportantNodeConverter(main_graph, {ge::WHILE, ge::DATA, ge::NETOUTPUT});
    auto root_model = GeModelBuilder(main_graph).BuildGeRootModel();
    auto global_data = GlobalDataFaker(root_model).Build();

    main_graph->TopologicalSorting();
    ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
    model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
    auto lower_3stage_graph = GraphConverter()
        .SetModelDescHolder(&model_desc_holder)
        .ConvertComputeGraphToExecuteGraph(main_graph, global_data);
    ASSERT_NE(lower_3stage_graph, nullptr);
    auto lower_graph = ge::FastNodeUtils::GetSubgraphFromNode(
        ge::ExecuteGraphUtils::FindFirstNodeMatchType(lower_3stage_graph.get(), "Main"), 0);
    ASSERT_NE(lower_graph, nullptr);
    auto lower_while_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(lower_graph, "While");
    ASSERT_NE(lower_while_node, nullptr);

    auto subgraph_names = lower_while_node->GetOpDescBarePtr()->GetSubgraphNameIndexes();
    std::map<uint32_t, std::string> sorted_subgraph_names;
    for (auto &item : subgraph_names) {
      sorted_subgraph_names[item.second] = item.first;
    }
    ASSERT_EQ(sorted_subgraph_names.size(), 2);
    auto iter = sorted_subgraph_names.begin();
    EXPECT_EQ((iter++)->second, "control_frame");
    EXPECT_EQ(iter->second, "body");

    auto control_frame_graph = ge::FastNodeUtils::GetSubgraphFromNode(lower_while_node, 0);
    ASSERT_NE(control_frame_graph, nullptr);
    auto subgraph_call = ge::ExecuteGraphUtils::FindFirstNodeMatchType(control_frame_graph, "SubgraphCall");
    ASSERT_NE(subgraph_call, nullptr);

    auto body_graph = ge::FastNodeUtils::GetSubgraphFromNode(lower_while_node, 1);
    ASSERT_NE(body_graph, nullptr);
    // check there is identity_shape_addr
    auto identity_shape_and_addr = ge::ExecuteGraphUtils::FindFirstNodeMatchType(body_graph, "IdentityShapeAndAddr");
    ASSERT_NE(identity_shape_and_addr, nullptr);
    ASSERT_EQ(identity_shape_and_addr->GetDataInNum(), 2U * while_node->GetOutDataNodesSize());
    ASSERT_EQ(identity_shape_and_addr->GetDataOutNum(), 2U * while_node->GetOutDataNodesSize());

    // check there is asscessMemCrossStream between add & identity_shape_addr
    auto access_mem_cross = identity_shape_and_addr->GetInDataEdgeByIndex(4)->src;
    EXPECT_EQ(access_mem_cross->GetType(), "AccessMemCrossStream");
    auto in_node = access_mem_cross->GetInDataEdgeByIndex(0)->src;
    EXPECT_EQ(in_node->GetType(), "Add");

    ge::DumpGraph(lower_3stage_graph.get(), "WhileExecGraph");
  }
}
}  // namespace gert
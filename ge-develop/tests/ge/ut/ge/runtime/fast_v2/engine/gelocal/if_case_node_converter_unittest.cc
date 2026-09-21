/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "engine/gelocal/if_case_converter.h"
#include "engine/gelocal/subgraph_io_converter.h"

#include <gtest/gtest.h>

#include "ge_graph_dsl/graph_dsl.h"
#include "engine/gelocal/inputs_converter.h"
#include "graph/utils/graph_utils.h"
#include "graph_builder/exe_graph_comparer.h"
#include "graph_builder/value_holder_generator.h"
#include "common/share_graph.h"
#include "faker/global_data_faker.h"
#include "common/bg_test.h"
#include "common/topo_checker.h"
#include "lowering/graph_converter.h"
#include "register/op_impl_registry.h"
#include "framework/common/types.h"
#include "faker/magic_ops.h"
#include "common/const_data_helper.h"
#include "graph/utils/graph_dump_utils.h"

using namespace ge;
namespace gert {
REGISTER_NODE_CONVERTER("_subgraph_input", LoweringSubgraphInput);
REGISTER_NODE_CONVERTER("_subgraph_output", LoweringSubgraphOutput);
IMPL_OP(Shape);
IMPL_OP(NetOutput);
const static std::string kStubLowerFunc = "_stub_lower_func";
LowerResult LoweringStubNode(const ge::NodePtr &node, const LowerInput &lower_input) {
  LowerResult ret;
  auto merged_holders = lower_input.input_shapes;
  merged_holders.insert(merged_holders.end(), lower_input.input_addrs.begin(), lower_input.input_addrs.end());
  if (node->GetAllOutDataAnchorsSize() == 0U) {
    const static std::string kLowerPrefix = "Lower_";
    ret.order_holders.push_back(bg::ValueHolder::CreateVoid<bg::ValueHolder>((kLowerPrefix + node->GetType()).c_str(), merged_holders));
  } else {
    auto holders = bg::DevMemValueHolder::CreateDataOutput(node->GetType().c_str(), merged_holders,
                                                           node->GetAllOutDataAnchorsSize() << 1U,
                                                           node->GetOpDesc()->GetStreamId());
    std::vector<bg::ValueHolderPtr> launch_inputs;
    launch_inputs.emplace_back(lower_input.global_data->GetStreamById(0));
    launch_inputs.insert(launch_inputs.cend(), holders.cbegin() + node->GetAllOutDataAnchorsSize(), holders.cend());
    auto launch_holder = bg::ValueHolder::CreateVoid<bg::ValueHolder>("Launch", launch_inputs);
    ret.order_holders.emplace_back(launch_holder);
    ret.out_shapes.insert(ret.out_shapes.end(), holders.begin(), holders.begin() + node->GetAllOutDataAnchorsSize());
    ret.out_addrs.insert(ret.out_addrs.end(), holders.begin() + node->GetAllOutDataAnchorsSize(), holders.end());
  }
  return ret;
}
REGISTER_NODE_CONVERTER(kStubLowerFunc.c_str(), LoweringStubNode);

class IfOrCaseNodeConverterUT : public bg::BgTestAutoCreate3StageFrame {
 public:
  void ConnectFromOuter(ge::FastNode *node, int32_t dst_index, const ge::FastNode *outer, int32_t src_index) {
    std::stringstream path_ss;
    path_ss << node->GetName() << ":" << dst_index;
    while (true) {
      auto in_edge = node->GetInDataEdgeByIndex(dst_index);
      ASSERT_NE(in_edge, nullptr);
      auto src_node = in_edge->src;
      ASSERT_NE(src_node, nullptr);
      if (src_node == outer) {
        return;
      }
      ASSERT_TRUE((src_node->GetType() == "InnerData" || src_node->GetType() == "Data"));

      int32_t parent_index;
      ASSERT_TRUE(ge::AttrUtils::GetInt(src_node->GetOpDescBarePtr(), "index", parent_index))
                    << "Failed to find outer node " << outer->GetName() << " index " << src_index << " from inner node "
                    << node->GetName() << " index " << dst_index << ", path " << path_ss.str();
      auto parent_graph = src_node->GetExtendInfo()->GetOwnerGraphBarePtr();
      ASSERT_NE(parent_graph, nullptr);
      auto parent_node = parent_graph->GetParentNodeBarePtr();
      ASSERT_NE(parent_node, nullptr);
      node = parent_node;
      dst_index = parent_index;

      path_ss << " -> " << node->GetName() << ":" << dst_index;
    }
  }
  void TestOneEmptyGraph(ge::ComputeGraphPtr &main_graph) {
    for (auto &clean_node : ge::GraphUtils::FindNodesByTypeFromAllNodes(main_graph, "Clean")) {
      ge::AttrUtils::SetStr(clean_node->GetOpDesc(), "_ge_attr_lowering_func", kStubLowerFunc);
    }

    auto if_node = main_graph->FindFirstNodeMatchType("If");
    ASSERT_NE(if_node, nullptr);
    auto root_model = GeModelBuilder(main_graph).BuildGeRootModel();
    auto global_data = GlobalDataFaker(root_model).Build();
    auto stream_on_main = global_data.GetStreamById(0);
    bg::LowerConstDataNode(global_data);
    LowerInput data_input = {{}, {}, &global_data};
    LowerInput if_input = {{}, {}, &global_data};
    for (auto &node : main_graph->GetDirectNode()) {
      if (node->GetType() == ge::DATA) {
        auto lr = LoweringDataNode(node, data_input);
        EXPECT_TRUE(lr.result.IsSuccess());
        if_input.input_shapes.push_back(lr.out_shapes[0]);
        if_input.input_addrs.push_back(lr.out_addrs[0]);
      }
    }


    auto if_lower_result = LoweringIf(if_node, if_input);
    EXPECT_TRUE(if_lower_result.result.IsSuccess());
    ASSERT_EQ(if_lower_result.out_shapes.size(), 0);
    ASSERT_EQ(if_lower_result.out_addrs.size(), 0);
    ASSERT_EQ(if_lower_result.order_holders.size(), 1);

    auto execute_graph =
        bg::ValueHolder::PopGraphFrame(ConvertDevMemValueHoldersToValueHolders(if_lower_result.out_addrs),
                                       if_lower_result.order_holders)
            ->GetExecuteGraph();

    EXPECT_NE(execute_graph, nullptr);

    auto exe_if_node = ExecuteGraphUtils::FindFirstNodeMatchType(execute_graph.get(), "If");
    ASSERT_NE(exe_if_node, nullptr);
    std::map<std::string, uint32_t> subgraph_names_to_index{{"CondGraph", 0}, {"then_graph", 1}, {"else_graph", 2}};
    ASSERT_EQ(exe_if_node->GetOpDescPtr()->GetSubgraphNameIndexes(), subgraph_names_to_index);

    std::vector<FastSrcNode> exe_if_inputs;
    exe_if_inputs.emplace_back("GenIndexForIf", 0);
    for (const auto &input_shape : if_input.input_shapes) {
      exe_if_inputs.emplace_back(input_shape);
    }
    for (const auto &input_addr : if_input.input_addrs) {
      exe_if_inputs.emplace_back(input_addr);
    }
    ASSERT_EQ(FastNodeTopoChecker(exe_if_node).StrictConnectFrom(exe_if_inputs), "success");
    ASSERT_TRUE(FastNodeTopoChecker(exe_if_node)
                    .InChecker()
                    .DataFromByType("GenIndexForIf")
                    .DataFrom(if_input.input_shapes[0])
                    .IsOk());

    ASSERT_NE(FastNodeUtils::GetSubgraphFromNode(exe_if_node, 0), nullptr);
    ASSERT_NE(FastNodeUtils::GetSubgraphFromNode(exe_if_node, 1), nullptr);
    ASSERT_NE(FastNodeUtils::GetSubgraphFromNode(exe_if_node, 2), nullptr);

    DumpGraph(execute_graph.get(), "IfCaseExecGraph");
  }
};

TEST_F(IfOrCaseNodeConverterUT, ConvertIf_InputIndexCorrect) {
  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistry();
  auto main_graph = ShareGraph::IfGraph();
  ASSERT_NE(main_graph, nullptr);
  for (auto &clean_node : ge::GraphUtils::FindNodesByTypeFromAllNodes(main_graph, "Shape")) {
    ge::AttrUtils::SetStr(clean_node->GetOpDesc(), "_ge_attr_lowering_func", kStubLowerFunc);
  }

  auto if_node = main_graph->FindFirstNodeMatchType("If");
  ASSERT_NE(if_node, nullptr);
  auto root_model = GeModelBuilder(main_graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();

  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};
  LowerInput if_input = {{}, {}, &global_data};
  for (auto &node : main_graph->GetDirectNode()) {
    if (node->GetType() == ge::DATA) {
      auto lr = LoweringDataNode(node, data_input);
      EXPECT_TRUE(lr.result.IsSuccess());
      if_input.input_shapes.push_back(lr.out_shapes[0]);
      if_input.input_addrs.push_back(lr.out_addrs[0]);
    }
  }

  auto if_lower_result = LoweringIf(if_node, if_input);
  EXPECT_TRUE(if_lower_result.result.IsSuccess());
  ASSERT_EQ(if_lower_result.out_shapes.size(), 1);
  ASSERT_EQ(if_lower_result.out_addrs.size(), 1);
  ASSERT_EQ(if_lower_result.order_holders.size(), 1);

  auto execute_graph =
      bg::ValueHolder::PopGraphFrame(ConvertDevMemValueHoldersToValueHolders(if_lower_result.out_addrs),
                                     if_lower_result.order_holders)
          ->GetExecuteGraph();
  EXPECT_NE(execute_graph, nullptr);

  auto exe_if_node = ExecuteGraphUtils::FindFirstNodeMatchType(execute_graph.get(), "If");
  ASSERT_NE(exe_if_node, nullptr);

  auto shape_in = exe_if_node->GetInDataEdgeByIndex(2);
  ASSERT_NE(shape_in, nullptr);
  auto addr_in = exe_if_node->GetInDataEdgeByIndex(4);
  ASSERT_NE(addr_in, nullptr);

  ASSERT_NE(ge::FastNodeUtils::GetSubgraphFromNode(exe_if_node, 1), nullptr);
  auto shape_node =
      ge::ExecuteGraphUtils::FindFirstNodeMatchType(ge::FastNodeUtils::GetSubgraphFromNode(exe_if_node, 1), "Shape");
  ASSERT_EQ(FastNodeTopoChecker(shape_node).StrictConnectFrom({{"InnerData"}, {"InnerData"}}), "success");
  ASSERT_NE(shape_node, nullptr);
  ConnectFromOuter(shape_node, 0, shape_in->src, shape_in->src_output);
  ConnectFromOuter(shape_node, 1, addr_in->src, addr_in->src_output);
}

TEST_F(IfOrCaseNodeConverterUT, ConvertIf_Success_TwoGraphs) {
  auto main_graph = ShareGraph::IfGraph();
  ASSERT_NE(main_graph, nullptr);
  for (auto &clean_node : ge::GraphUtils::FindNodesByTypeFromAllNodes(main_graph, "Shape")) {
    ge::AttrUtils::SetStr(clean_node->GetOpDesc(), "_ge_attr_lowering_func", kStubLowerFunc);
  }

  auto if_node = main_graph->FindFirstNodeMatchType("If");
  ASSERT_NE(if_node, nullptr);
  auto root_model = GeModelBuilder(main_graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();

  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};
  LowerInput if_input = {{}, {}, &global_data};
  for (auto &node : main_graph->GetDirectNode()) {
    if (node->GetType() == ge::DATA) {
      auto lr = LoweringDataNode(node, data_input);
      EXPECT_TRUE(lr.result.IsSuccess());
      if_input.input_shapes.push_back(lr.out_shapes[0]);
      if_input.input_addrs.push_back(lr.out_addrs[0]);
    }
  }

  auto if_lower_result = LoweringIf(if_node, if_input);
  EXPECT_TRUE(if_lower_result.result.IsSuccess());
  ASSERT_EQ(if_lower_result.out_shapes.size(), 1);
  ASSERT_EQ(if_lower_result.out_addrs.size(), 1);
  ASSERT_EQ(if_lower_result.order_holders.size(), 1);

  auto execute_graph =
      bg::ValueHolder::PopGraphFrame(ConvertDevMemValueHoldersToValueHolders(if_lower_result.out_addrs), if_lower_result.order_holders)->GetExecuteGraph();
  EXPECT_NE(execute_graph, nullptr);

  auto exe_if_node = ExecuteGraphUtils::FindFirstNodeMatchType(execute_graph.get(), "If");
  ASSERT_NE(exe_if_node, nullptr);
  std::map<std::string, uint32_t> subgraph_names_to_index{{"CondGraph", 0}, {"then_graph", 1}, {"else_graph", 2}};
  ASSERT_EQ(exe_if_node->GetOpDescPtr()->GetSubgraphNameIndexes(), subgraph_names_to_index);

  std::vector<FastSrcNode> exe_if_inputs;
  exe_if_inputs.emplace_back("GenIndexForIf", 0);
  for (const auto &input_shape : if_input.input_shapes) {
    exe_if_inputs.emplace_back(input_shape);
  }
  for (const auto &input_addr : if_input.input_addrs) {
    exe_if_inputs.emplace_back(input_addr);
  }
  exe_if_inputs.emplace_back(global_data.GetStreamById(0));
  ASSERT_EQ(FastNodeTopoChecker(exe_if_node).StrictConnectFrom(exe_if_inputs), "success");
  ASSERT_TRUE(FastNodeTopoChecker(exe_if_node)
                  .InChecker()
                  .DataFromByType("GenIndexForIf")
                  .DataFrom(if_input.input_shapes[0])
                  .IsOk());

  ASSERT_NE(FastNodeUtils::GetSubgraphFromNode(exe_if_node, 0), nullptr);
  ASSERT_NE(FastNodeUtils::GetSubgraphFromNode(exe_if_node, 1), nullptr);
  ASSERT_NE(FastNodeUtils::GetSubgraphFromNode(exe_if_node, 2), nullptr);

  DumpGraph(execute_graph.get(), "IfCaseExecGraph");
}

TEST_F(IfOrCaseNodeConverterUT, ConvertIf_Success_OneGraphOnlyDataAndNetOutput) {
  auto main_graph = ShareGraph::IfOneBranchGraph();
  ASSERT_NE(main_graph, nullptr);
  TestOneEmptyGraph(main_graph);
}
TEST_F(IfOrCaseNodeConverterUT, ConvertIf_Success_OneEmptyGraph) {
  auto main_graph = ShareGraph::IfOneBranchGraph2();
  ASSERT_NE(main_graph, nullptr);
  TestOneEmptyGraph(main_graph);
}

//                                    ┌──────────────────────────────────┐
//                                    │                                  │
//                                    │ ┌──────┐   ┌─────────────────┐   │
//                         -----------│ │ Data ├───►  RequireDevice  │   │
//                         -          │ └──────┘   └─────────────────┘   │
//                         -          │                                  │
//                         -          └──────────────────────────────────┘
// ┌───────────┐      ┌──────────┐
// │ GenDevice ├─────►│   If     │
// └───────────┘      └──────────┘     ┌─────────────────────────────────┐
//                         ▲ -         │                                 │
// ┌───────────┐           │ -         │ ┌──────┐   ┌──────────────┐     │
// │   Data    ├───────────┘ -         │ │ Data ├───► RequireHost  │     │
// └───────────┘             ----------│ └──────┘   └──────────────┘     │
//                                     │                                 │
//                                     └─────────────────────────────────┘
/**
 * 用例描述：测试If节点两个子图要求的输入Placement不同时，能正确Lowering，并生成对应的MakeSure执行节点
 *
 * 预置条件：
 * 1.通过MagicOpFakerBuilder，为主图If输入、Then子图、Else子图节点打桩，分别设置对应的输出placement和输入placement要求
 * 2.主图If输入的placement为Device，then子图节点需求输入位于Device，else子图节点需求输入位于Host
 *
 * 测试步骤
 * 1.构造子图需求不同Placement的if节点，参见IfWithDifferentPlacementSubgraph示意图
 * 2.执行LowerIf调用
 *
 * 预期结果
 * 1.执行成功
 * 2.校验then子图中不含有MakeSureTensorAtHost，校验else子图中包含MakeSureTensorAtHost
 */
TEST_F(IfOrCaseNodeConverterUT, ConvertIf_Success_SubgraphInputPlacementCorrect) {
  auto graph = ShareGraph::IfWithDifferentPlacementSubgraph();
  auto data = graph->FindNode("input");
  auto gen_device = graph->FindNode("gen_device");
  auto if_op = graph->FindNode("if");
  gert::GertRuntimeStub stub;
  MagicOpFakerBuilder("IfOrCaseNodeConverterUT_ReqDevice").RequireInputPlacement(kOnDeviceHbm).Build(stub);
  MagicOpFakerBuilder("IfOrCaseNodeConverterUT_ReqHost").RequireInputPlacement(kOnHost).Build(stub);
  MagicOpFakerBuilder("IfOrCaseNodeConverterUT_GenDevice").OutputPlacement(kOnDeviceHbm).Build(stub);

auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput if_input = {{}, {}, &global_data};
  for (auto &node : {data, gen_device}) {
    LowerInput lower_input = {{}, {}, &global_data};
    auto lr = *LoweringNode(node, lower_input, {});
    ASSERT_TRUE(lr.result.IsSuccess());
    if_input.input_shapes.push_back(lr.out_shapes[0]);
    if_input.input_addrs.push_back(lr.out_addrs[0]);
  }
  auto lr = LoweringIf(if_op, if_input);
  std::cerr << lr.result.GetErrorMessage() << std::endl;
  ASSERT_TRUE(lr.result.IsSuccess());

  auto execute_graph =
      bg::ValueHolder::PopGraphFrame(ConvertDevMemValueHoldersToValueHolders(lr.out_addrs), lr.order_holders)
          ->GetExecuteGraph();
  ASSERT_NE(execute_graph, nullptr);

  auto exe_if_node = ExecuteGraphUtils::FindFirstNodeMatchType(execute_graph.get(), "If");
  ASSERT_NE(exe_if_node, nullptr);
  auto then_exec = FastNodeUtils::GetSubgraphFromNode(exe_if_node, 1);
  auto else_exec = FastNodeUtils::GetSubgraphFromNode(exe_if_node, 2);
  ASSERT_NE(then_exec, nullptr);
  ASSERT_NE(else_exec, nullptr);

  ASSERT_EQ(ExecuteGraphUtils::FindFirstNodeMatchType(then_exec, "MakeSureTensorAtHost"), nullptr);
  ASSERT_NE(ExecuteGraphUtils::FindFirstNodeMatchType(else_exec, "MakeSureTensorAtHost"), nullptr);
}
}  // namespace gert

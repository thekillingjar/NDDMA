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
#include "engine/gelocal/inputs_converter.h"
#include "common/share_graph.h"
#include "faker/global_data_faker.h"
#include "common/bg_test.h"
#include "common/topo_checker.h"
#include "common/ge_inner_attrs.h"
#include "lowering/static_model_output_allocator.h"
#include "common/const_data_helper.h"
#include "graph/utils/op_desc_utils_ex.h"

using namespace ge;
using namespace gert::bg;

namespace gert {
class OutputsReuseHelperUT : public bg::BgTestAutoCreate3StageFrame {};

TEST_F(OutputsReuseHelperUT, GenerateOutputsReuseInfosSuccess) {
  auto graph = ShareGraph::BuildWithKnownSubgraph();
  auto parent_node = graph->FindNode(ge::PARTITIONEDCALL);

auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Conv2d", false).Build();
  LowerInput data_input = {{}, {}, &global_data};
  auto data_a_ret = LoweringDataNode(graph->FindNode("data_a"), data_input);
  ASSERT_TRUE(data_a_ret.result.IsSuccess());
  ASSERT_EQ(data_a_ret.out_shapes.size(), 1);
  LowerInput conv2d_input = {{data_a_ret.out_shapes[0]},
                             {data_a_ret.out_addrs[0]},
                             &global_data};

  std::vector<OutputReuseInfo> output_reuse_infos;
  auto ret = StaticModelOutputAllocator::GenerateOutputsReuseInfos(
      graph->GetSubgraph(parent_node->GetOpDesc()->GetSubgraphInstanceName(0)), output_reuse_infos);
  ASSERT_EQ(ret, ge::SUCCESS);
  // check output num
  EXPECT_EQ(output_reuse_infos.size(), 3);

  StaticModelOutputAllocator output_allocator(nullptr, data_a_ret.out_addrs);
  auto lower_result = output_allocator.AllocAllOutputs(output_reuse_infos, global_data);
  EXPECT_EQ(lower_result.result.IsSuccess(), true);
  EXPECT_EQ(lower_result.out_addrs.size(), 3);
  EXPECT_EQ(lower_result.out_shapes.size(), 3);
  EXPECT_EQ(lower_result.order_holders.size(), 1);
}

TEST_F(OutputsReuseHelperUT, GenerateOutputsReuseInfos_Success_PartitionedCallConnectedToNetOutput) {
  auto graph = ShareGraph::BuildWithNestingKnownSubgraph();
  auto parent_node = graph->FindNode(ge::PARTITIONEDCALL);

auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Conv2d", false).Build();
  LowerInput data_input = {{}, {}, &global_data};
  auto data_a_ret = LoweringDataNode(graph->FindNode("data_a"), data_input);
  ASSERT_TRUE(data_a_ret.result.IsSuccess());
  ASSERT_EQ(data_a_ret.out_shapes.size(), 1);
  LowerInput conv2d_input = {{data_a_ret.out_shapes[0]},
                             {data_a_ret.out_addrs[0]},
                             &global_data};

  std::vector<OutputReuseInfo> output_reuse_infos;
  auto ret = StaticModelOutputAllocator::GenerateOutputsReuseInfos(
      graph->GetSubgraph(parent_node->GetOpDesc()->GetSubgraphInstanceName(0)), output_reuse_infos);
  ASSERT_EQ(ret, ge::SUCCESS);
  // check output num
  EXPECT_EQ(output_reuse_infos.size(), 3);
  EXPECT_TRUE(output_reuse_infos[0].is_reuse);
  EXPECT_TRUE(output_reuse_infos[1].is_reuse);
  EXPECT_TRUE(output_reuse_infos[2].is_reuse);
  EXPECT_EQ(output_reuse_infos[1].reuse_type, OutputReuseType::kReuseOutput);
  EXPECT_EQ(output_reuse_infos[2].reuse_type, OutputReuseType::kReuseOutput);
}

TEST_F(OutputsReuseHelperUT, RefOutpusOffsetCorrect) {
  auto graph = ShareGraph::BuildWithKnownSubgraph();
  auto parent_node = graph->FindNode(ge::PARTITIONEDCALL);

  auto sub_graph = ge::NodeUtils::GetSubgraph(*parent_node, 0U);
  ASSERT_NE(sub_graph, nullptr);
  auto net_output = sub_graph->FindFirstNodeMatchType("NetOutput");
  ASSERT_NE(net_output, nullptr);
  ASSERT_NE(net_output->GetOpDesc(), nullptr);

  const auto &tensor_desc = net_output->GetOpDesc()->MutableInputDesc(2U);
  (void)ge::TensorUtils::SetDataOffset(*tensor_desc, 0x120);

  const auto &tensor_desc1 = net_output->GetOpDesc()->MutableInputDesc(1U);
  (void)ge::TensorUtils::SetDataOffset(*tensor_desc1, 0x90);

auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Conv2d", false).Build();
  LowerInput data_input = {{}, {}, &global_data};
  auto data_a_ret = LoweringDataNode(graph->FindNode("data_a"), data_input);
  ASSERT_TRUE(data_a_ret.result.IsSuccess());
  ASSERT_EQ(data_a_ret.out_shapes.size(), 1);
  LowerInput conv2d_input = {{data_a_ret.out_shapes[0]},
                             {data_a_ret.out_addrs[0]},
                             &global_data};

  std::vector<OutputReuseInfo> output_reuse_infos;
  auto ret = StaticModelOutputAllocator::GenerateOutputsReuseInfos(
      graph->GetSubgraph(parent_node->GetOpDesc()->GetSubgraphInstanceName(0)), output_reuse_infos);
  ASSERT_EQ(ret, ge::SUCCESS);
  ASSERT_EQ(output_reuse_infos.size(), 3);

  EXPECT_EQ(output_reuse_infos[0].is_reuse, false);

  EXPECT_EQ(output_reuse_infos[1].is_reuse, true);
  EXPECT_EQ(output_reuse_infos[1].reuse_type, OutputReuseType::kReuseInput);
  EXPECT_EQ(output_reuse_infos[1].reuse_index, 0);

  EXPECT_EQ(output_reuse_infos[2].is_reuse, true);
  EXPECT_EQ(output_reuse_infos[2].reuse_type, OutputReuseType::kRefOutput);
  EXPECT_EQ(output_reuse_infos[1].mem_base_type_offset.base_type, kernel::MemoryBaseType::kMemoryBaseTypeWeight);
}

TEST_F(OutputsReuseHelperUT, RefOutpusOffsetCorrect_WithTwoConst) {
  auto graph = ShareGraph::BuildWithKnownSubgraphWithTwoConst();
  auto parent_node = graph->FindNode(ge::PARTITIONEDCALL);

  auto sub_graph = ge::NodeUtils::GetSubgraph(*parent_node, 0U);
  ASSERT_NE(sub_graph, nullptr);
  auto net_output = sub_graph->FindFirstNodeMatchType("NetOutput");
  ASSERT_NE(net_output, nullptr);
  ASSERT_NE(net_output->GetOpDesc(), nullptr);

  const auto &tensor_desc = net_output->GetOpDesc()->MutableInputDesc(2U);
  (void)ge::TensorUtils::SetDataOffset(*tensor_desc, 0x120);

  const auto &tensor_desc1 = net_output->GetOpDesc()->MutableInputDesc(1U);
  (void)ge::TensorUtils::SetDataOffset(*tensor_desc1, 0x90);

  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Conv2d", false).Build();
  LowerInput data_input = {{}, {}, &global_data};
  auto data_a_ret = LoweringDataNode(graph->FindNode("data_a"), data_input);
  ASSERT_TRUE(data_a_ret.result.IsSuccess());
  ASSERT_EQ(data_a_ret.out_shapes.size(), 1);
  LowerInput conv2d_input = {{data_a_ret.out_shapes[0]},
                             {data_a_ret.out_addrs[0]},
                             &global_data};

  std::vector<OutputReuseInfo> output_reuse_infos;
  auto ret = StaticModelOutputAllocator::GenerateOutputsReuseInfos(
      graph->GetSubgraph(parent_node->GetOpDesc()->GetSubgraphInstanceName(0)), output_reuse_infos);
  ASSERT_EQ(ret, ge::SUCCESS);
  ASSERT_EQ(output_reuse_infos.size(), 3);

  EXPECT_EQ(output_reuse_infos[0].is_reuse, true);
  EXPECT_EQ(output_reuse_infos[0].reuse_type, OutputReuseType::kRefOutput);

  EXPECT_EQ(output_reuse_infos[1].is_reuse, true);
  EXPECT_EQ(output_reuse_infos[1].reuse_type, OutputReuseType::kReuseInput);
  EXPECT_EQ(output_reuse_infos[1].reuse_index, 0);

  EXPECT_EQ(output_reuse_infos[2].is_reuse, true);
  EXPECT_EQ(output_reuse_infos[2].reuse_type, OutputReuseType::kRefOutput);
  EXPECT_EQ(output_reuse_infos[1].mem_base_type_offset.base_type, kernel::MemoryBaseType::kMemoryBaseTypeWeight);
}

TEST_F(OutputsReuseHelperUT, RefVariableCorrect) {
  auto graph = ShareGraph::BuildWithKnownSubgraph();
  auto parent_node = graph->FindNode(ge::PARTITIONEDCALL);

  auto sub_graph = ge::NodeUtils::GetSubgraph(*parent_node, 0U);
  ASSERT_NE(sub_graph, nullptr);

  // const --> Variable
  auto const1 = sub_graph->FindNode("const1");
  auto op_desc = const1->GetOpDesc();
  ge::OpDescUtilsEx::SetType(op_desc, "Variable");

  auto net_output = sub_graph->FindFirstNodeMatchType("NetOutput");
  ASSERT_NE(net_output, nullptr);
  ASSERT_NE(net_output->GetOpDesc(), nullptr);

  const auto &tensor_desc = net_output->GetOpDesc()->MutableInputDesc(2U);
  (void)ge::TensorUtils::SetDataOffset(*tensor_desc, 0x120);

  const auto &tensor_desc1 = net_output->GetOpDesc()->MutableInputDesc(1U);
  (void)ge::TensorUtils::SetDataOffset(*tensor_desc1, 0x90);

auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Conv2d", false).Build();
  bg::LowerConstDataNode(global_data);

  LowerInput data_input = {{}, {}, &global_data};
  auto data_a_ret = LoweringDataNode(graph->FindNode("data_a"), data_input);
  ASSERT_TRUE(data_a_ret.result.IsSuccess());
  ASSERT_EQ(data_a_ret.out_shapes.size(), 1);
  LowerInput conv2d_input = {{data_a_ret.out_shapes[0]},
                             {data_a_ret.out_addrs[0]},
                             &global_data};

  std::vector<OutputReuseInfo> output_reuse_infos;
  auto ret = StaticModelOutputAllocator::GenerateOutputsReuseInfos(
      graph->GetSubgraph(parent_node->GetOpDesc()->GetSubgraphInstanceName(0)), output_reuse_infos);
  ASSERT_EQ(ret, ge::SUCCESS);
  ASSERT_EQ(output_reuse_infos.size(), 3);

  EXPECT_EQ(output_reuse_infos[0].is_reuse, false);

  EXPECT_EQ(output_reuse_infos[1].is_reuse, true);
  EXPECT_EQ(output_reuse_infos[1].reuse_type, OutputReuseType::kReuseInput);
  EXPECT_EQ(output_reuse_infos[1].reuse_index, 0);

  EXPECT_EQ(output_reuse_infos[2].is_reuse, true);
  EXPECT_EQ(output_reuse_infos[2].reuse_type, OutputReuseType::kRefVariable);
  EXPECT_EQ(output_reuse_infos[2].var_name, "const1");

  StaticModelOutputAllocator output_allocator(nullptr, data_a_ret.out_addrs);
  auto lower_result = output_allocator.AllocAllOutputs(output_reuse_infos, global_data);
  EXPECT_EQ(lower_result.result.IsSuccess(), true);
  EXPECT_EQ(lower_result.out_addrs.size(), 3);
  EXPECT_EQ(lower_result.out_shapes.size(), 3);
  EXPECT_EQ(lower_result.order_holders.size(), 1);
  ASSERT_NE(lower_result.out_addrs[2]->GetFastNode(), nullptr);
  EXPECT_EQ(lower_result.out_addrs[2]->GetFastNode()->GetType(), "Init");
}

TEST_F(OutputsReuseHelperUT, RefConstantCorrect) {
  auto graph = ShareGraph::BuildWithKnownSubgraph();
  auto parent_node = graph->FindNode(ge::PARTITIONEDCALL);

  auto sub_graph = ge::NodeUtils::GetSubgraph(*parent_node, 0U);
  ASSERT_NE(sub_graph, nullptr);

  // const --> Variable
  auto const1 = sub_graph->FindNode("const1");
  auto op_desc = const1->GetOpDesc();
  ge::OpDescUtilsEx::SetType(op_desc, "Constant");

  auto net_output = sub_graph->FindFirstNodeMatchType("NetOutput");
  ASSERT_NE(net_output, nullptr);
  ASSERT_NE(net_output->GetOpDesc(), nullptr);

  const auto &tensor_desc = net_output->GetOpDesc()->MutableInputDesc(2U);
  (void)ge::TensorUtils::SetDataOffset(*tensor_desc, 0x120);

  const auto &tensor_desc1 = net_output->GetOpDesc()->MutableInputDesc(1U);
  (void)ge::TensorUtils::SetDataOffset(*tensor_desc1, 0x90);

auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Conv2d", false).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};
  auto data_a_ret = LoweringDataNode(graph->FindNode("data_a"), data_input);
  ASSERT_TRUE(data_a_ret.result.IsSuccess());
  ASSERT_EQ(data_a_ret.out_shapes.size(), 1);
  LowerInput conv2d_input = {{data_a_ret.out_shapes[0]},
                             {data_a_ret.out_addrs[0]},
                             &global_data};

  std::vector<OutputReuseInfo> output_reuse_infos;
  auto ret = StaticModelOutputAllocator::GenerateOutputsReuseInfos(
      graph->GetSubgraph(parent_node->GetOpDesc()->GetSubgraphInstanceName(0)), output_reuse_infos);
  ASSERT_EQ(ret, ge::SUCCESS);
  ASSERT_EQ(output_reuse_infos.size(), 3);

  EXPECT_EQ(output_reuse_infos[0].is_reuse, false);

  EXPECT_EQ(output_reuse_infos[1].is_reuse, true);
  EXPECT_EQ(output_reuse_infos[1].reuse_type, OutputReuseType::kReuseInput);
  EXPECT_EQ(output_reuse_infos[1].reuse_index, 0);

  EXPECT_EQ(output_reuse_infos[2].is_reuse, true);
  EXPECT_EQ(output_reuse_infos[2].reuse_type, OutputReuseType::kRefVariable);
  EXPECT_EQ(output_reuse_infos[2].var_name, "const1");

  StaticModelOutputAllocator output_allocator(nullptr, data_a_ret.out_addrs);
  auto lower_result = output_allocator.AllocAllOutputs(output_reuse_infos, global_data);
  EXPECT_EQ(lower_result.result.IsSuccess(), true);
  EXPECT_EQ(lower_result.out_addrs.size(), 3);
  EXPECT_EQ(lower_result.out_shapes.size(), 3);
  EXPECT_EQ(lower_result.order_holders.size(), 1);
  EXPECT_EQ(lower_result.out_addrs[2]->GetFastNode()->GetType(), "Init");
}

TEST_F(OutputsReuseHelperUT, StringTypeNoReuseOutputSizeCorrect) {
  auto graph = ShareGraph::BuildStringNodeGraph();
auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  std::vector<OutputReuseInfo> output_reuse_infos;
  auto ret = StaticModelOutputAllocator::GenerateOutputsReuseInfos(graph, output_reuse_infos);
  ASSERT_EQ(ret, ge::SUCCESS);
  ASSERT_EQ(output_reuse_infos.size(), 1);
  EXPECT_EQ(output_reuse_infos[0].is_reuse, false);
  StaticModelOutputAllocator output_allocator(nullptr, {});
  auto lower_ret = output_allocator.AllocAllOutputs(output_reuse_infos, global_data);
  ASSERT_TRUE(lower_ret.result.IsSuccess());

  ASSERT_EQ(lower_ret.out_addrs.size(), 1);
  auto malloc_node = lower_ret.out_addrs[0]->GetFastNode();
  ASSERT_EQ(malloc_node->GetType(), "AllocMemHbm");
  ASSERT_EQ(malloc_node->GetInDataNodes().size(), 2);
  ASSERT_EQ(malloc_node->GetInDataNodes().at(1)->GetType(), "Const");
  auto const_node = malloc_node->GetInDataNodes().at(1);
  Buffer buffer;
  ASSERT_TRUE(ge::AttrUtils::GetZeroCopyBytes(const_node->GetOpDescPtr(), "value", buffer));
  ASSERT_EQ(buffer.size(), sizeof(uint64_t));
  // sizeof(uint64_t) * 2 * (2 * 3)
  ASSERT_EQ(*reinterpret_cast<uint64_t *>(buffer.GetData()), 128);
}

// todo need fix refdata addr in static sub graph, and release this ut
/*
TEST_F(OutputsReuseHelperUT, ParseReuseInputsReuse_SrcNodeNotData) {
  int64_t stream_num, event_num;
  auto graph = ShareGraph::MultiStreamGraphRefMemCrossStream(stream_num, event_num);
  auto netoutput = graph->FindFirstNodeMatchType(NETOUTPUT);
  netoutput->GetOpDesc()->SetInputOffset({0, 1, 2});
  auto assign = graph->FindNode("assign");
  std::vector<OutputReuseInfo> output_reuse_infos;
  EXPECT_EQ(StaticModelOutputAllocator::GenerateOutputsReuseInfos(graph, output_reuse_infos), ge::SUCCESS);
  ASSERT_EQ(output_reuse_infos.size(), 1);
  EXPECT_EQ(output_reuse_infos[0].is_reuse, true);
}
*/

TEST_F(OutputsReuseHelperUT, ParseReuseInputsReuse) {
  auto graph = ShareGraph::BuildWithKnownSubgraphWithRefNode();
  auto sub_graph = graph->GetSubgraph("g2");
  EXPECT_NE(sub_graph, nullptr);
  auto data = sub_graph->FindFirstNodeMatchType(DATA);
  EXPECT_NE(data, nullptr);
  TensorUtils::SetDataOffset(*(data->GetOpDesc()->MutableOutputDesc(0)), 1);
  std::vector<OutputReuseInfo> output_reuse_infos;
  EXPECT_EQ(StaticModelOutputAllocator::GenerateOutputsReuseInfos(sub_graph, output_reuse_infos), ge::SUCCESS);
  ASSERT_EQ(output_reuse_infos.size(), 1);
  EXPECT_EQ(output_reuse_infos[0].is_reuse, true);
}
} // namespace gert

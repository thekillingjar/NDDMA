/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_builder/bg_memory.h"
#include <gtest/gtest.h>
#include "register/node_converter_registry.h"
#include <common/const_data_helper.h>
#include "common/bg_test.h"
#include "common/topo_checker.h"
#include "faker/node_faker.h"
#include "faker/global_data_faker.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "common/share_graph.h"
#include "graph_builder/bg_rt_session.h"

namespace gert {
using namespace bg;
class BgMemoryUT : public BgTestAutoCreate3StageFrame {};
TEST_F(BgMemoryUT, AllocHbmWorkspace) {
  auto graph = std::make_shared<ge::ComputeGraph>("tmp");
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  LoweringGlobalData global_data = GlobalDataFaker(root_model).Build();
  auto size = ValueHolder::CreateFeed(0);
  auto mem = AllocWorkspaceMem(kOnDeviceHbm, size, global_data);
  ASSERT_NE(mem, nullptr);
  EXPECT_EQ(mem->GetPlacement(), kOnDeviceHbm);
  EXPECT_EQ(mem->GetFastNode()->GetType(), "AllocBatchHbm");
  auto free_mem = FreeWorkspaceMem(kOnDeviceHbm, mem);
  ASSERT_NE(free_mem, nullptr);
  EXPECT_EQ(free_mem->GetFastNode()->GetType(), "FreeBatchHbm");
  int64_t index;
  EXPECT_TRUE(ge::AttrUtils::GetInt(free_mem->GetFastNode()->GetOpDescBarePtr(), "ReleaseResourceIndex", index));
  EXPECT_EQ(index, 0);
  FastNodeTopoChecker checker(mem);
  EXPECT_EQ(checker.StrictConnectFrom(std::vector<FastSrcNode>({{"SelectL2Allocator"}, size}), true), "success");
  EXPECT_EQ(checker.OutChecker().DataToByType("FreeBatchHbm").Result(), "success");
}
TEST_F(BgMemoryUT, AllocHostOutput) {
  LoweringGlobalData global_data;
  auto size1 = ValueHolder::CreateFeed(0);
  auto size2 = ValueHolder::CreateFeed(1);
  auto node = ComputeNodeFaker().IoNum(2, 2).Build();

  auto out_mems = AllocOutputMemory(kOnHost, node, {size1, size2}, global_data);
  ASSERT_EQ(out_mems.size(), 2);

  auto allocator = global_data.GetOrCreateAllocator({kOnHost, AllocatorUsage::kAllocNodeOutput});
  ASSERT_NE(allocator, nullptr);

  auto mem1 = out_mems[0];
  EXPECT_EQ(mem1->GetPlacement(), kOnHost);
  FastNodeTopoChecker checker(mem1);
  EXPECT_EQ(checker.StrictConnectFrom(std::vector<FastSrcNode>({allocator, size1}), true), "success");
  EXPECT_EQ(checker.OutChecker().DataToByType("FreeMemory").Result(), "success");

  auto mem2 = out_mems[1];
  EXPECT_EQ(mem1->GetPlacement(), kOnHost);
  checker = FastNodeTopoChecker(mem2);
  EXPECT_EQ(checker.StrictConnectFrom(std::vector<FastSrcNode>({allocator, size2}), true), "success");
  EXPECT_EQ(checker.OutChecker().DataToByType("FreeMemory").Result(), "success");
}

TEST_F(BgMemoryUT, AllocFftsMem) {
  auto graph = std::make_shared<ge::ComputeGraph>("tmp");
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  LoweringGlobalData global_data = GlobalDataFaker(root_model).Build();
  uint32_t window_size = 2UL;
  auto window_size_holder = ValueHolder::CreateConst(&window_size, sizeof(window_size));
  auto ffts_allocator = CreateFftsMemAllocator(window_size_holder, global_data);
  ASSERT_NE(ffts_allocator, nullptr);

  auto batch_sizes = ValueHolder::CreateFeed(1);
  auto batch_mems =  AllocateBatchFftsMems(ffts_allocator, batch_sizes, 0);
  ASSERT_NE(batch_mems, nullptr);
  FastNodeTopoChecker batch_checker(batch_mems);
  EXPECT_EQ(batch_checker.OutChecker().DataToByType("FreeBatchFftsMems").Result(), "success");

  size_t block_size = 2UL;
  auto block_size_holder = ValueHolder::CreateConst(&block_size, sizeof(block_size));
  auto mems =  AllocateFftsMems(ffts_allocator, 0, {block_size_holder});

  EXPECT_EQ(mems.size(), 1UL);
  ASSERT_NE(mems[0], nullptr);
  EXPECT_EQ(mems[0]->GetPlacement(), kOnDeviceHbm);

  auto void_guard = RecycleFftsMems(ffts_allocator);
  ASSERT_NE(void_guard, nullptr);
  FastNodeTopoChecker checker(mems[0]);
  EXPECT_EQ(checker.OutChecker().DataToByType("FreeFftsMem").Result(), "success");
}

/*  
   连接到op*的输出边ref自refdata
                                                                +------------------------------+
    data(refdata)      data0           data1       原始计算图    |  refdata       op       op   |
       |                  |             |            <======    |                 \      /     |
  split_tensor      GetVariableAddr   AllocMemHbm               |                 fake_node    | 
                                                                |                   /    \     |
                                                                |                 op*      op  |
                                                                +------------------------------+
*/
TEST_F(BgMemoryUT, AllocRefOutput_RefwithRefData) {
  // fake that refdata has been lowered
  // 模拟refdata lowering后的结果，在global data中会有一个uniquevalueholder，key为refdata name。
  std::string ref_data_name = "ref_data";
  auto graph = std::make_shared<ge::ComputeGraph>("tmp");
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  LoweringGlobalData global_data = GlobalDataFaker(root_model).Build();
  auto ref_data_exe = ValueHolder::CreateFeed(2);
  auto ref_data_addr = DevMemValueHolder::CreateSingleDataOutput("SplitTensor", {ref_data_exe}, 0);
  auto builder = [&ref_data_addr]() -> std::vector<bg::ValueHolderPtr> { return {ref_data_addr, ref_data_addr}; };
  global_data.GetOrCreateUniqueValueHolder(ref_data_name, builder);

  // 构造原始的计算图
  auto ref_data_node = ComputeNodeFaker().NameAndType(ref_data_name, "RefData").IoNum(1, 1).Build();
  auto compute_node_faker = ComputeNodeFaker();
  auto node = compute_node_faker.IoNum(2, 2).Build();
  (void)ge::AttrUtils::SetStr(node->GetOpDesc()->MutableOutputDesc(0), ge::REF_VAR_SRC_VAR_NAME, ref_data_name);
  auto root_graph = node->GetOwnerComputeGraph();
  root_graph->AddNodeFront(ref_data_node->GetOpDesc());

  // 调用分配fake node的输出内存
  auto size1 = ValueHolder::CreateFeed(0);
  auto size2 = ValueHolder::CreateFeed(1);
  auto out_mems = AllocOutputMemory(kOnDeviceHbm, node, {size1, size2}, global_data);
  ASSERT_EQ(out_mems.size(), 2);

  auto allocator = global_data.GetOrCreateAllocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
  ASSERT_NE(allocator, nullptr);

  // 此时mem0 应该就是split tensor的输出
  auto mem0 = out_mems[0];
  EXPECT_NE(mem0, nullptr);
  EXPECT_EQ(mem0->GetPlacement(), kOnDeviceHbm);
  FastNodeTopoChecker checker(mem0);
  EXPECT_EQ(checker.StrictConnectFrom(std::vector<FastSrcNode>({ref_data_exe}), true), "success");

  auto mem1 = out_mems[1];
  EXPECT_NE(mem1, nullptr);
  EXPECT_EQ(mem1->GetPlacement(), kOnDeviceHbm);
  checker = FastNodeTopoChecker(mem1);
  EXPECT_EQ(checker.StrictConnectFrom(std::vector<FastSrcNode>({allocator, size2}), true), "success");
  EXPECT_EQ(checker.OutChecker().DataToByType("FreeMemory").Result(), "success");
}

TEST_F(BgMemoryUT, AllocOutputMemoryForVariable_WrongPlacement) {
  auto graph = SingleNodeGraphBuilder("g", "Foo").NumInputs(3).NumOutputs(2).Build();
  auto foo = graph->FindFirstNodeMatchType("Foo");
  auto desc1 = foo->GetOpDesc()->MutableOutputDesc(0);
  ge::AttrUtils::SetStr(desc1, ge::ASSIGN_VAR_NAME, "var_1");
  auto desc2 = foo->GetOpDesc()->MutableOutputDesc(1);

  ge::AttrUtils::SetBool(foo->GetOpDesc(), ge::ATTR_NAME_REFERENCE, true);
  foo->GetOpDesc()->MutableInputDesc(1)->SetDataType(ge::DT_UNDEFINED); // Unfed optional input
  foo->GetOpDesc()->MutableInputDesc(1)->SetFormat(ge::FORMAT_RESERVED);
  ASSERT_NE(foo->GetOpDesc()->GetAllInputsSize(), foo->GetOpDesc()->GetInputsSize());

  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  bg::LowerConstDataNode(global_data);

  LowerInput lower_input;
  lower_input.global_data = &global_data;

  size_t output_size = 64;
  auto output_size_holder = bg::ValueHolder::CreateConst(&output_size, sizeof(output_size));
  std::vector<bg::ValueHolderPtr> output_sizes(2, output_size_holder);

  auto input_addr_0 = bg::DevMemValueHolder::CreateConst(&output_size, sizeof(output_size), 0);
  auto input_addr_1 = bg::DevMemValueHolder::CreateConst(&output_size, sizeof(output_size), 0);
  std::vector<bg::DevMemValueHolderPtr> input_addrs{input_addr_0, input_addr_1};  // Unfed optional input "y"

  auto allocated_memories =
      bg::AllocOutputMemory(TensorPlacement::kFollowing, foo, output_sizes, input_addrs, global_data);

  ASSERT_EQ(allocated_memories.size(), 0);
}

TEST_F(BgMemoryUT, AllocSessionFixedMemory_Success) {
  auto graph = SingleNodeGraphBuilder("g", "Foo").NumInputs(3).NumOutputs(2).Build();
  auto foo = graph->FindFirstNodeMatchType("Foo");
  auto desc1 = foo->GetOpDesc()->MutableOutputDesc(0);
  ge::AttrUtils::SetStr(desc1, ge::ASSIGN_VAR_NAME, "var_1");
  auto desc2 = foo->GetOpDesc()->MutableOutputDesc(1);

  ge::AttrUtils::SetBool(foo->GetOpDesc(), ge::ATTR_NAME_REFERENCE, true);
  foo->GetOpDesc()->MutableInputDesc(1)->SetDataType(ge::DT_UNDEFINED); // Unfed optional input
  foo->GetOpDesc()->MutableInputDesc(1)->SetFormat(ge::FORMAT_RESERVED);
  ASSERT_NE(foo->GetOpDesc()->GetAllInputsSize(), foo->GetOpDesc()->GetInputsSize());

  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).Build();
  bg::LowerConstDataNode(global_data);

  LowerInput lower_input;
  lower_input.global_data = &global_data;

  auto session_id_holder = bg::FrameSelector::OnInitRoot(
      [&global_data]() -> std::vector<bg::ValueHolderPtr> {
        return {bg::GetSessionId(global_data)};
      });

  auto output_size_holder = bg::FrameSelector::OnInitRoot(
      []() -> std::vector<bg::ValueHolderPtr> {
        size_t output_size = 64;
        return {bg::ValueHolder::CreateConst(&output_size, sizeof(output_size))};
      });
  auto allocated_memory =
      bg::AllocFixedFeatureMemory(session_id_holder[0], TensorPlacement::kOnDeviceHbm,  output_size_holder[0], global_data);

  ASSERT_NE(allocated_memory, nullptr);
}

}  // namespace gert
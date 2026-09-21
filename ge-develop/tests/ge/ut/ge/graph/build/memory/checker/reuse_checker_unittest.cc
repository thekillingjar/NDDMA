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
#include <gmock/gmock.h>

#include "graph/build/memory/checker/reuse_checker.h"
#include "common/share_graph.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/utils/graph_utils_ex.h"
#include "stub/gert_runtime_stub.h"
#include "graph/ge_local_context.h"
#include "graph/ge_context.h"
#include "common/ge_common/ge_types.h"
#include "framework/common/types.h"
#include "framework/memory/memory_assigner.h"
#include "../../test_memory_shared_graph.h"
#include "common/mem_conflict_share_graph.h"

namespace ge {
class UtestReuseChecker : public testing::Test {
 protected:
  void SetUp() {
    global_options_ = ge::GetThreadLocalContext().GetAllGlobalOptions();
    graph_options_ = ge::GetThreadLocalContext().GetAllGraphOptions();
    session_options_ = ge::GetThreadLocalContext().GetAllSessionOptions();

    std::map<std::string, std::string> tmp_global_option;
    tmp_global_option.insert(std::make_pair(ge::OPTION_TOPOSORTING_MODE, "0"));
    ge::GetThreadLocalContext().SetGlobalOption(tmp_global_option);
  }
  void TearDown() {
    ge::GetThreadLocalContext().SetGlobalOption(global_options_);
    ge::GetThreadLocalContext().SetGraphOption(graph_options_);
    ge::GetThreadLocalContext().SetSessionOption(session_options_);
  }
  std::map<std::string, std::string> global_options_;
  std::map<std::string, std::string> graph_options_;
  std::map<std::string, std::string> session_options_;
};

/*
 *    var a
 *     \  /
 *     assign ref_var_src_var_name
 *       |
 *    assign_add ref_var_src_var_name
 *       |
 *     netoutput
 *   assign and assign_add use same var addreess
 */
TEST_F(UtestReuseChecker, RefVarSrcVarName_CheckSkip) {
  VarManager::Instance(0)->Destory();
  VarManager::Instance(0)->Init(0, 0, 0, 0);
  auto graph = block_mem_ut::BuildGraphWithRefVarSrcVarName();
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
}

/*
 *    var a
 *     \  /
 *     assign ref_var_src_var_name
 *       |
 *    assign_add
 *       |
 *     netoutput
 *   assign and assign_add use same var addreess
 */
TEST_F(UtestReuseChecker, RefVarSrcVarNameAbsent_CheckSkip) {
  VarManager::Instance(0)->Destory();
  VarManager::Instance(0)->Init(0, 0, 0, 0);
  auto graph = block_mem_ut::BuildGraphWithRefVarSrcVarNameAbsent();
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
}

//    const
//      |                +---------------+
//  partitioncall0-------| data          |
//      |                |  |            |
//      c (stream 0)     |  b (stream 1) |
//      |                |  |            |
//      d (stream 0)     | netoutput1    |
//      |                +---------------+
//   netoutput
//
// 子图data引用父节点的const输入，offset与const输出offset一样，和feature map的offset重叠
TEST_F(UtestReuseChecker, RefFromConst_CheckSkip) {
  VarManager::Instance(0)->Destory();
  VarManager::Instance(0)->Init(0, 0, 0, 0);
  auto graph = block_mem_ut::BuildDataRefConst();
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
}

//    a   b
//     \ /
//     pc
//      |
//  netoutput
//
TEST_F(UtestReuseChecker, ContinuousInputAndOutputOffsetForBufferFusion_ModifyMemorySize) {
  auto graph = block_mem_ut::BuildContinuousInputAndOutputOffsetForBufferFusion();
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
}

//          a   b   c   d
//           \ /     \ /       abcde output shape: [1,1,48,448]
//           pc1  e  pc2
//[1,2,48,448]\   |  / [1,2,48,448]
//               pc3
//                |  [1,5,48,448]
//            netoutput
//
// 级联phony concat的特点是，pc1 pc2设置的size和pc3的size一样，都是最终一整块内存的size，
// pc2输入节点上设置的ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION，也是该输入节点在最终一整块内存上的offset。
//
// 对notask的节点不作检查，因为这些节点不真正写输出内存，所以只需要检查真正写输出内存的节点就可以
TEST_F(UtestReuseChecker, PhonyConcatCascated_SkipNoTask) {
  auto graph = block_mem_ut::BuildPhonyConcatCascated();
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
}

//    a   b
//     \ /
//     hcom input need p2p, output need hbm. means that a and b memory type is p2p
//      |
//  netoutput
//
TEST_F(UtestReuseChecker, MemoryTypeOfP2p) {
  auto graph = block_mem_ut::BuildHcomWithP2pMemoryType();
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
}

//    const
//      |                +---------------+
//  partitioncall0-------| data          |
//      |                |  |            |
//      c (stream 0)     |  b (stream 1) |
//      |                |  |            |
//      d (stream 0)     | netoutput1    |
//      |                +---------------+
//   netoutput
// 带有这个属性ATTR_NAME_MEMORY_SIZE_CALC_TYPE，且属性数值为1的，不分配内存，所以其offset会为0.，检查内存需要跳过
//
TEST_F(UtestReuseChecker, SkipWithAttr_ATTR_NAME_MEMORY_SIZE_CALC_TYPE) {
  auto graph = block_mem_ut::BuildDataRefConst();
  const auto c = graph->FindNode("c");
  ASSERT_NE(c, nullptr);
  ge::AttrUtils::SetInt(c->GetOpDesc()->MutableOutputDesc(0), ATTR_NAME_MEMORY_SIZE_CALC_TYPE, 1);
  const auto netoutput = graph->FindNode("NETOUTPUT");
  ASSERT_NE(netoutput, nullptr);
  netoutput->GetOpDescBarePtr()->SetType("Add");
  graph->TopologicalSorting();
  HybridMemAssigner hybrid_mem_assigner(graph);
  ASSERT_EQ(hybrid_mem_assigner.Assign(), SUCCESS);
  ASSERT_NE(hybrid_mem_assigner.GetReuseChecker(), nullptr);

  const auto d = graph->FindNode("d");
  ASSERT_NE(d, nullptr);
  d->GetOpDescBarePtr()->SetOutputOffset({0});
  c->GetOpDescBarePtr()->SetOutputOffset({0});
  EXPECT_EQ(hybrid_mem_assigner.GetReuseChecker()->Check(), SUCCESS);
}

//    const
//      |                +---------------+
//  partitioncall0-------| data          |
//      |                |  |            |
//      c (stream 0)     |  b (stream 1) |
//      |                |  |            |
//      d (stream 0)     | netoutput1    |
//      |                +---------------+
//   netoutput
//
TEST_F(UtestReuseChecker, SkipMemoryTypeOfL1) {
  auto graph = block_mem_ut::BuildDataRefConst();
  const auto c = graph->FindNode("c");
  ASSERT_NE(c, nullptr);
  int64_t memory_type = RT_MEMORY_L1;
  ge::AttrUtils::SetInt(c->GetOpDescBarePtr()->MutableOutputDesc(0), ATTR_NAME_TENSOR_MEM_TYPE, memory_type);
  const auto netoutput = graph->FindNode("NETOUTPUT");
  ASSERT_NE(netoutput, nullptr);
  netoutput->GetOpDescBarePtr()->SetType("Add");
  graph->TopologicalSorting();
  HybridMemAssigner hybrid_mem_assigner(graph);
  ASSERT_EQ(hybrid_mem_assigner.Assign(), SUCCESS);
  ASSERT_NE(hybrid_mem_assigner.GetReuseChecker(), nullptr);

  const auto d = graph->FindNode("d");
  ASSERT_NE(d, nullptr);
  d->GetOpDescBarePtr()->SetOutputOffset({0});
  c->GetOpDescBarePtr()->SetOutputOffset({0});
  EXPECT_EQ(hybrid_mem_assigner.GetReuseChecker()->Check(), SUCCESS);
}

//    const
//      |                +---------------+
//  partitioncall0-------| data          |
//      |                |  |            |
//      c (stream 0)     |  b (stream 1) |
//      |                |  |            |
//      d (stream 0)     | netoutput1    |
//      |                +---------------+
//   netoutput
//
TEST_F(UtestReuseChecker, ReuseCheckFailed) {
  auto graph = block_mem_ut::BuildDataRefConst();
  const auto netoutput = graph->FindNode("NETOUTPUT");
  ASSERT_NE(netoutput, nullptr);
  netoutput->GetOpDescBarePtr()->SetType("Add");
  graph->TopologicalSorting();
  HybridMemAssigner hybrid_mem_assigner(graph);
  ASSERT_EQ(hybrid_mem_assigner.Assign(), SUCCESS);
  ASSERT_NE(hybrid_mem_assigner.GetReuseChecker(), nullptr);
  EXPECT_EQ(hybrid_mem_assigner.GetReuseChecker()->Check(), SUCCESS);

  const auto c = graph->FindNode("c");
  ASSERT_NE(c, nullptr);
  c->GetOpDescBarePtr()->SetOutputOffset({0});

  EXPECT_NE(hybrid_mem_assigner.GetReuseChecker()->Check(), SUCCESS);
}

// a-b-c-d-e
TEST_F(UtestReuseChecker, CCannotReuseD_CheckError) {
  auto graph = block_mem_ut::BuildFiveNodesGraph();
  HybridMemAssigner hybrid_mem_assigner(graph);
  ASSERT_EQ(hybrid_mem_assigner.Assign(), SUCCESS);
  ASSERT_NE(hybrid_mem_assigner.GetReuseChecker(), nullptr);

  const auto c = graph->FindNode("c");
  const auto d = graph->FindNode("d");
  ASSERT_NE(c, nullptr);
  ASSERT_NE(d, nullptr);
  d->GetOpDescBarePtr()->SetOutputOffset(c->GetOpDescBarePtr()->GetOutputOffset());
  EXPECT_NE(hybrid_mem_assigner.GetReuseChecker()->Check(), SUCCESS);
}

// stream0 a-b-c
// stream2 d-e
TEST_F(UtestReuseChecker, CCannotReuseD_DiffStream_CheckError) {
  auto graph = block_mem_ut::BuildFiveNodesDiffStreamGraph();
  HybridMemAssigner hybrid_mem_assigner(graph);
  ASSERT_EQ(hybrid_mem_assigner.Assign(), SUCCESS);
  ASSERT_NE(hybrid_mem_assigner.GetReuseChecker(), nullptr);

  const auto b = graph->FindNode("b");
  const auto d = graph->FindNode("d");
  ASSERT_NE(b, nullptr);
  ASSERT_NE(d, nullptr);
  d->GetOpDescBarePtr()->SetOutputOffset(b->GetOpDescBarePtr()->GetOutputOffset());
  EXPECT_NE(hybrid_mem_assigner.GetReuseChecker()->Check(), SUCCESS);
}

//     const
//      |
//      a
//      |                +-----------+
//  partitioncall0-------| data      |
//      |                |  |        |
//      d                |  b        |
//      |                |  |        |
//  netoutput            |  c        |
//                       |  |        |
//                       |netoutput1 |
//                       +-----------+
//
TEST_F(UtestReuseChecker, ReuseWithOutputNodeWrapper) {
  auto graph = block_mem_ut::BuildReuseWithOutNodeWrapper();
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
}

//    a   b
//     \ /
//     hcom
//      |
//  netoutput
//
TEST_F(UtestReuseChecker, SkipNodeWithBufferPoolId) {
  auto graph = block_mem_ut::BuildHcomWithBufferPoolId();
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
}

//    a   b
//     \ /
//      c
//      |
//  netoutput
//
TEST_F(UtestReuseChecker, OutputNoAssignSkip) {
  auto graph = block_mem_ut::BuildGraphWithOutputNotAssign();
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
}

//          a
//          |
//          b   c  shape[60] no alignaed size: 240, after 32 aligned: 256, after +32padding: 288
//           \ /
//           pc1 shape[120], no alignaed size: 480, after 32 aligned: 480, after +32padding: 512
//            |
//         netoutput
//
//  240(b) + 288(c) = 528 > 512
//
TEST_F(UtestReuseChecker, ModifyPhonyConcatLastInputMemSize) {
  auto graph = block_mem_ut::BuildPhonyConcatSizeLessThanSumOfAllInputs();
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
}
} // namespace ge

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

#include "graph/build/memory/checker/special_node_checker.h"
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

namespace ge {
class UtestSpecialNodeChecker : public testing::Test {};

//          a   b   c   d
//           \ /     \ /       abcde output shape: [1,1,48,448]
//           pc1  e  pc2
//[1,2,48,448]\   |  / [1,2,48,448]
//               pc3
//                |  [1,5,48,448]
//                f
//                |
//             netoutput
//
TEST_F(UtestSpecialNodeChecker, NoPaddingContinuousInput_Check) {
  auto graph = block_mem_ut::BuildPhonyConcatCascated();
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);

  SpecialNodeChecker checker;
  EXPECT_EQ(checker.Check(graph), SUCCESS);

  auto e = graph->FindNode("e");
  auto offset = e->GetOpDescBarePtr()->GetOutputOffset();
  offset[0] += 1;
  e->GetOpDescBarePtr()->SetOutputOffset(offset);
  EXPECT_NE(checker.Check(graph), SUCCESS);
}

//          a
//          |
//          b   c
//           \ /
//           pc1
//            |
//            d
//            |
//         netoutput
//
TEST_F(UtestSpecialNodeChecker, NoPaddingContinuousInput_Check_WithOutputOffsetList) {
  auto graph = block_mem_ut::BuildPhonyConcatWithOutputOffsetList();
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);

  SpecialNodeChecker checker;
  EXPECT_EQ(checker.Check(graph), SUCCESS);

  auto b = graph->FindNode("b");
  std::vector<int64_t> offset_list = {490, 480};
  (void)AttrUtils::SetListInt(b->GetOpDescBarePtr(), ATTR_NAME_OUTPUT_OFFSET_LIST_FOR_CONTINUOUS, offset_list);
  EXPECT_NE(checker.Check(graph), SUCCESS);
}

/*
          a
          |
          ps
         / \
        b   c
        \   /
       netoutput
*/
TEST_F(UtestSpecialNodeChecker, NoPaddingContinuousOutput_Check_WithOutputOffsetList) {
  auto graph = block_mem_ut::BuildPhonySplitWithOutputOffsetList();
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);

  SpecialNodeChecker checker;
  EXPECT_EQ(checker.Check(graph), SUCCESS);

  auto b = graph->FindNode("b");
  std::vector<int64_t> offset_list = {240, 480};
  (void)AttrUtils::SetListInt(b->GetOpDescBarePtr(), ATTR_NAME_INPUT_OFFSET_LIST_FOR_CONTINUOUS, offset_list);
  EXPECT_NE(checker.Check(graph), SUCCESS);
}

/*
          a
          |
          ps
         / \
        b
        \
       netoutput
*/
TEST_F(UtestSpecialNodeChecker, NoPaddingContinuousOutput_Check_WithOutputOffsetListAndSuspendOutput) {
  auto graph = block_mem_ut::BuildPhonySplitSuspendOutputWithOutputOffsetList();
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
TEST_F(UtestSpecialNodeChecker, ContinuousInput_Check) {
  auto graph = block_mem_ut::BuildHcomWithBufferPoolId();
  auto hcom = graph->FindNode("hcom");
  AttrUtils::ClearAllAttrs(hcom->GetOpDescBarePtr());
  ge::AttrUtils::SetBool(hcom->GetOpDescBarePtr(), ATTR_NAME_CONTINUOUS_INPUT, true);
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);

  SpecialNodeChecker checker;
  EXPECT_EQ(checker.Check(graph), SUCCESS);

  auto b = graph->FindNode("b");
  auto offset = b->GetOpDescBarePtr()->GetOutputOffset();
  offset[0] += 1;
  b->GetOpDescBarePtr()->SetOutputOffset(offset);
  EXPECT_NE(checker.Check(graph), SUCCESS);
}

/*
 *   a  b
 *    \ /
 *    hcom
 *     |
 *    netoutput
 */
TEST_F(UtestSpecialNodeChecker, ContinuousOutput_Check) {
  auto graph = block_mem_ut::BuildHcomWithContinuousOutputGraph();
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);

  SpecialNodeChecker checker;
  EXPECT_EQ(checker.Check(graph), SUCCESS);

  auto hcom = graph->FindNode("hcom");
  auto offset = hcom->GetOpDescBarePtr()->GetOutputOffset();
  offset[0] += 1;
  hcom->GetOpDescBarePtr()->SetOutputOffset(offset);
  EXPECT_NE(checker.Check(graph), SUCCESS);
}

/*
 *      c
 *      |
 *     split
 *      /\
 *     a  b
 */
TEST_F(UtestSpecialNodeChecker, NoPaddingContinuousOutput_Check) {
  auto graph = block_mem_ut::BuildPhonySplitGraph();
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);

  SpecialNodeChecker checker;
  EXPECT_EQ(checker.Check(graph), SUCCESS);

  auto split = graph->FindNode("split");
  auto offset = split->GetOpDescBarePtr()->GetOutputOffset();
  offset[0] += 1;
  split->GetOpDescBarePtr()->SetOutputOffset(offset);
  EXPECT_NE(checker.Check(graph), SUCCESS);
}

//          a   b   c   d
//           \ /     \ /       abcde output shape: [1,1,48,448]
//           pc1  e  pc2
//[1,2,48,448]\   |  / [1,2,48,448]
//               pc3
//                |  [1,5,48,448]
//                f
//                |
//             netoutput
//
TEST_F(UtestSpecialNodeChecker, NoPaddingContinuousInputMemSize_Success) {
  auto graph = block_mem_ut::BuildPhonyConcatCascated();
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);

  SpecialNodeChecker checker;
  EXPECT_EQ(checker.Check(graph), SUCCESS);
}

//          a   b   c   d
//           \ /     \ /       abcde output shape: [1,1,48,448]
//           pc1  e  pc2
//[1,2,48,448]\   |  / [1,2,48,448]
//               pc3
//                |  [1,5,48,448]
//                f
//                |
//             netoutput
//
TEST_F(UtestSpecialNodeChecker, NoPaddingContinuousInputMemSize_Failed_BecauseLastRealInputShapeTooBig) {
  auto graph = block_mem_ut::BuildPhonyConcatCascated();
  auto d = graph->FindNode("d");
  std::vector<int64_t> offsets_of_fusion;
  AttrUtils::GetListInt(d->GetOpDescBarePtr(), ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION, offsets_of_fusion);
  ASSERT_FALSE(offsets_of_fusion.empty());

  d->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1,3,48,448})));

  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_NE(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
}

/*
 *             f
 *             |
 *            ps1 [1,5,48,448]
 *          /  |  \
 *        ps2  e  ps3 [1,2,48,448]
 *        / \     /  \
 *       a   b   c    d  abcde output shape: [1,1,48,448]
 */
TEST_F(UtestSpecialNodeChecker, NoPaddingContinuousOutputMemSize_Failed_BecausePhonySplitInputNodeSizeTooSmall) {
  auto graph = block_mem_ut::BuildPhonySplitCascated();
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);

  auto f = graph->FindNode("f");
  ASSERT_NE(f, nullptr);

  const auto tensor_desc = f->GetOpDescBarePtr()->GetOutputDesc(0);
  int64_t out_size;
  TensorUtils::GetSize(tensor_desc, out_size);
  TensorUtils::SetSize(*f->GetOpDescBarePtr()->MutableOutputDesc(0), out_size - 60);

  EXPECT_NE(SpecialNodeChecker::Check(graph), SUCCESS);
}


} // namespace ge
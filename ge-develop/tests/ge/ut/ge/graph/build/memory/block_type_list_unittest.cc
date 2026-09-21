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
#include "graph/build/memory/block_mem_assigner.h"
#include "graph/build/memory/block_type_list.h"
#include "ge/ut/ge/graph/build/test_memory_shared_graph.h"
namespace ge {
class UtestBlockTypeListTest : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestBlockTypeListTest, NodeMemAttrConflictRegister_HasConflict) {
  EXPECT_TRUE(NodeMemAttrConflictRegister::HasConflict(NodeMemAttr::kData, NodeMemAttr::kConcentrateAtomic));
  EXPECT_TRUE(NodeMemAttrConflictRegister::HasConflict(NodeMemAttr::kConcentrateAtomic, NodeMemAttr::kData));
}

TEST_F(UtestBlockTypeListTest, NodeMemAttrConflictRegister_NotConflict) {
  EXPECT_FALSE(NodeMemAttrConflictRegister::HasConflict(NodeMemAttr::kData, NodeMemAttr::kData));
}

TEST_F(UtestBlockTypeListTest, NodeMemAttrConflictRegister_NotConflict2) {
  EXPECT_FALSE(NodeMemAttrConflictRegister::HasConflict(NodeMemAttr::kConcentrateAtomic, NodeMemAttr::kConcentrateAtomic));
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
//                       | hcom(输入清零)|
//                       |  |        |
//                       |netoutput1 |
//                       +-----------+
TEST_F(UtestBlockTypeListTest, NodeMemAttrUtils_GetNodeMemAttrs) {
  auto graph = block_mem_ut::BuildAtomicCleanNotReuseSubGraphData();
  Node *c = nullptr;
  Node *data = nullptr;
  for (const auto node : graph->GetAllNodesPtr()) {
    if (node->GetName() == "c") {
      c = node;
    }
    if (node->GetName() == "partitioncall_0_data") {
      data = node;
    }
  }
  ASSERT_NE(c, nullptr);
  ASSERT_NE(data, nullptr);
  {
    NodeTypeIndex node_type_index{data, kOutput, 0};
    auto attrs = NodeMemAttrUtils::GetNodeMemAttrs(node_type_index);
    ASSERT_EQ(attrs.size(), 1U);
    EXPECT_EQ(attrs.front(), NodeMemAttr::kData);
  }

  {
    NodeTypeIndex node_type_index{c, kOutput, 0};
    auto attrs = NodeMemAttrUtils::GetNodeMemAttrs(node_type_index);
    ASSERT_EQ(attrs.size(), 1U);
    EXPECT_EQ(attrs.front(), NodeMemAttr::kConcentrateAtomic);
  }
}

//     const
//      |
//      a
//      |                +-----------+
//  partitioncall0-------| data      |
//      |                |  |        |
//      d                |  b        |
//      |                |  |        |
//  netoutput            |  c(输出引用输入)|
//                       |  |        |
//                       | hcom(输入清零)|
//                       |  |        |
//                       |netoutput1 |
//                       +-----------+
TEST_F(UtestBlockTypeListTest, NodeMemAttrUtils_IsConcentrateAtomic_ThroughRefNode) {
  auto graph = block_mem_ut::BuildAtomicCleanNotReuseSubGraphData();
  Node *c = nullptr;
  Node *b = nullptr;
  for (const auto node : graph->GetAllNodesPtr()) {
    if (node->GetName() == "c") {
      c = node;
    }
    if (node->GetName() == "b") {
      b = node;
    }
  }
  ASSERT_NE(c, nullptr);
  ASSERT_NE(b, nullptr);
  c->GetOpDescBarePtr()->MutableAllInputName() = {{"x", 0}};
  c->GetOpDescBarePtr()->MutableAllOutputName() = {{"x", 0}};
  (void)ge::AttrUtils::SetBool(c->GetOpDescBarePtr(), ATTR_NAME_REFERENCE, true);
  EXPECT_TRUE(NodeMemAttrUtils::IsConcentrateAtomic(b, 0));
}

/*
 *         data       mem_set
 *          |        /  /(ctrl)
 *     a scatter_nd    /
 *      \  /          /
 *      Reshape+-----+
 *        |
 *     netoutput
 *
 *  Reshape是输出引用输入，但是比较特殊，在建符号表的时候特别判断了节点类型
 */
TEST_F(UtestBlockTypeListTest, NodeMemAttrUtils_IsConcentrateAtomic_Output) {
  auto graph = block_mem_ut::AtomicNodeConnectReShapeConnectNetoutput();
  auto scatter_nd = graph->FindNode("scatter_nd");
  ASSERT_NE(scatter_nd, nullptr);
  EXPECT_TRUE(NodeMemAttrUtils::IsConcentrateAtomic(scatter_nd.get(), 0));
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
//                       | hcom(输入清零)|
//                       |  |        |
//                       |netoutput1 |
//                       +-----------+
TEST_F(UtestBlockTypeListTest, BlockTypeList_CheckConflict) {
  auto graph = block_mem_ut::BuildAtomicCleanNotReuseSubGraphData();
  auto a = graph->FindNode("a").get();
  Node *c = nullptr;
  Node *data = nullptr;
  for (const auto node : graph->GetAllNodesPtr()) {
    if (node->GetName() == "c") {
      c = node;
    }
    if (node->GetName() == "partitioncall_0_data") {
      data = node;
    }
  }
  ASSERT_NE(a, nullptr);
  ASSERT_NE(c, nullptr);
  ASSERT_NE(data, nullptr);

  BlockTypeList data_type;
  BlockTypeList atomic_clean_type;

  NodeTypeIndex data_node_type_index{data, kOutput, 0};
  data_type.WithAdded(data_node_type_index);
  EXPECT_STREQ(data_type.ToString().c_str(), "data ");

  NodeTypeIndex atomic_node_type_index{c, kOutput, 0};
  atomic_clean_type.WithAdded(atomic_node_type_index);
  EXPECT_STREQ(atomic_clean_type.ToString().c_str(), "concentrate_atomic ");
  EXPECT_STREQ(NodeMemAttrUtils::GetAttrStr(atomic_node_type_index).c_str(), "concentrate_atomic ");

  EXPECT_TRUE(data_type.IsConflictWithBlock(atomic_clean_type));
  EXPECT_TRUE(data_type.IsConflictWithOneNode(atomic_node_type_index));
  EXPECT_TRUE(atomic_clean_type.IsConflictWithOneNode(data_node_type_index));

  ReuseStrategy strategy;
  MemoryBlock block{strategy, 1};
  NodeTypeIndex normal{a, kOutput, 0};
  block.AddNodeTypeIndex(normal, 1, 1, 1);
  data_type.WithDeleted(block, data_node_type_index);
  EXPECT_STREQ(data_type.ToString().c_str(), "");
  EXPECT_FALSE(data_type.IsConflictWithBlock(atomic_clean_type));
}
} // namespace ge

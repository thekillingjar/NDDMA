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

#include "graph/build/memory/checker/atomic_clean_checker.h"
#include "common/share_graph.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/utils/graph_utils_ex.h"
#include "stub/gert_runtime_stub.h"
#include "graph/ge_local_context.h"
#include "graph/ge_context.h"
#include "graph/passes/memory_conflict/atomic_addr_clean_pass.h"
#include "common/ge_common/ge_types.h"
#include "framework/common/types.h"
#include "framework/memory/memory_assigner.h"
#include "../../test_memory_shared_graph.h"

namespace ge {
class UtestAtomicCleanChecker : public testing::Test {};

/*
 *         data       mem_set
 *          |        /  /(ctrl)
 *     a  scatter_nd   /
 *      \  /          /
 *      Reshape+-----+
 *        |
 *     netoutput
 */
TEST_F(UtestAtomicCleanChecker, AtomicCleanOutputCheckSuccess) {
  auto root_graph = block_mem_ut::AtomicNodeConnectReShapeConnectNetoutput();

  MemoryAssigner mem_assigner(root_graph);
  std::map<uint64_t, size_t> mem_offsets;
  size_t zero_copy_mem_size;
  EXPECT_EQ(mem_assigner.AssignMemory(mem_offsets, zero_copy_mem_size), SUCCESS);
}

/*
 *         data       mem_set
 *          |        /  /(ctrl)
 *     a  scatter_nd   /
 *      \  /          /
 *      Reshape+-----+
 *        |
 *     netoutput
 */
TEST_F(UtestAtomicCleanChecker, AtomicCleanOutputCheckFailed) {
  auto root_graph = block_mem_ut::AtomicNodeConnectReShapeConnectNetoutput();

  MemoryAssigner mem_assigner(root_graph);
  std::map<uint64_t, size_t> mem_offsets;
  size_t zero_copy_mem_size;
  EXPECT_EQ(mem_assigner.AssignMemory(mem_offsets, zero_copy_mem_size), SUCCESS);

  // 构造memset中的地址错误
  auto mem_set_node = root_graph->FindNode("mem_set");
  ASSERT_NE(mem_set_node, nullptr);
  const auto memset_origin_workspace = mem_set_node->GetOpDescBarePtr()->GetWorkspace();
  mem_set_node->GetOpDescBarePtr()->SetWorkspace({10000000});

  AtomicCleanChecker checker1(mem_assigner.GetGraphMemoryAssigner().get());
  auto ret = checker1.Check(root_graph);
  EXPECT_NE(ret, SUCCESS);
  mem_set_node->GetOpDescBarePtr()->SetWorkspace(memset_origin_workspace);

  // 构造memset中的size错误
  const auto memset_origin_size = mem_set_node->GetOpDescBarePtr()->GetWorkspaceBytes();
  mem_set_node->GetOpDescBarePtr()->SetWorkspaceBytes({2});
  AtomicCleanChecker checker2(mem_assigner.GetGraphMemoryAssigner().get());
  ret = checker2.Check(root_graph);
  EXPECT_NE(ret, SUCCESS);
  mem_set_node->GetOpDescBarePtr()->SetWorkspaceBytes(memset_origin_size);

  // 构造memset中的type错误
  std::vector<int64_t> workspace_type_list{64};
  ge::AttrUtils::SetListInt(mem_set_node->GetOpDescBarePtr(), ATTR_NAME_WORKSPACE_TYPE_LIST, workspace_type_list);
  AtomicCleanChecker checker22(mem_assigner.GetGraphMemoryAssigner().get());
  ret = checker22.Check(root_graph);
  EXPECT_NE(ret, SUCCESS);
  ge::AttrUtils::SetListInt(mem_set_node->GetOpDescBarePtr(), ATTR_NAME_WORKSPACE_TYPE_LIST, {});

  // 构造scatter_nd的地址错误
  auto scatter_nd = root_graph->FindNode("scatter_nd");
  ASSERT_NE(scatter_nd, nullptr);

  const auto origin_offset = scatter_nd->GetOpDescBarePtr()->GetOutputOffset();
  scatter_nd->GetOpDescBarePtr()->SetOutputOffset({100555});
  AtomicCleanChecker checker3(mem_assigner.GetGraphMemoryAssigner().get());
  ret = checker3.Check(root_graph);
  EXPECT_NE(ret, SUCCESS);
  scatter_nd->GetOpDescBarePtr()->SetOutputOffset(origin_offset);

  // 构造scatter_nd输出size过大
  auto tensor_desc = scatter_nd->GetOpDescBarePtr()->MutableOutputDesc(0);
  int64_t origin_mem_size = 0;
  TensorUtils::GetSize(*tensor_desc, origin_mem_size);
  TensorUtils::SetSize(*tensor_desc, 2 * origin_mem_size);
  AtomicCleanChecker checker4(mem_assigner.GetGraphMemoryAssigner().get());
  ret = checker4.Check(root_graph);
  EXPECT_NE(ret, SUCCESS);
  TensorUtils::SetSize(*tensor_desc, origin_mem_size);

  // 找不到memset节点
  mem_set_node->GetOpDescBarePtr()->SetType("Unknow");
  AtomicCleanChecker checker5(mem_assigner.GetGraphMemoryAssigner().get());
  ret = checker5.Check(root_graph);
  EXPECT_NE(ret, SUCCESS);
}
/**
 *      data  data
 *        \   /
 *         add
 *          |
 *         hcom
 *        /
 *     netout
 */
TEST_F(UtestAtomicCleanChecker, AtomicCleanInputCheckFailed) {
  auto root_graph = block_mem_ut::BuildAtomicCleanInputGraph();

  AtomicAddrCleanPass pass;
  EXPECT_EQ(pass.Run(root_graph), SUCCESS);

  MemoryAssigner mem_assigner(root_graph);
  std::map<uint64_t, size_t> mem_offsets;
  size_t zero_copy_mem_size;
  EXPECT_EQ(mem_assigner.AssignMemory(mem_offsets, zero_copy_mem_size), SUCCESS);

  // 构造memset中的地址错误
  auto mem_set_node = root_graph->FindFirstNodeMatchType("AtomicAddrClean");
  ASSERT_NE(mem_set_node, nullptr);
  const auto memset_origin_workspace = mem_set_node->GetOpDescBarePtr()->GetWorkspace();
  mem_set_node->GetOpDescBarePtr()->SetWorkspace({10000000});

  AtomicCleanChecker checker1(mem_assigner.GetGraphMemoryAssigner().get());
  auto ret = checker1.Check(root_graph);
  EXPECT_NE(ret, SUCCESS);
  mem_set_node->GetOpDescBarePtr()->SetWorkspace(memset_origin_workspace);

  // 构造memset中的size错误
  const auto memset_origin_size = mem_set_node->GetOpDescBarePtr()->GetWorkspaceBytes();
  mem_set_node->GetOpDescBarePtr()->SetWorkspaceBytes({2});
  AtomicCleanChecker checker2(mem_assigner.GetGraphMemoryAssigner().get());
  ret = checker2.Check(root_graph);
  EXPECT_NE(ret, SUCCESS);
  mem_set_node->GetOpDescBarePtr()->SetWorkspaceBytes(memset_origin_size);

  // 构造memset中的type错误
  std::vector<int64_t> workspace_type_list{64};
  ge::AttrUtils::SetListInt(mem_set_node->GetOpDescBarePtr(), ATTR_NAME_WORKSPACE_TYPE_LIST, workspace_type_list);
  AtomicCleanChecker checker22(mem_assigner.GetGraphMemoryAssigner().get());
  ret = checker22.Check(root_graph);
  EXPECT_NE(ret, SUCCESS);
  ge::AttrUtils::SetListInt(mem_set_node->GetOpDescBarePtr(), ATTR_NAME_WORKSPACE_TYPE_LIST, {});

  // 构造b的地址错误
  auto hcom_1 = root_graph->FindNode("hcom_1");
  ASSERT_NE(hcom_1, nullptr);

  const auto origin_offset = hcom_1->GetOpDescBarePtr()->GetInputOffset();
  hcom_1->GetOpDescBarePtr()->SetInputOffset({100555});
  AtomicCleanChecker checker3(mem_assigner.GetGraphMemoryAssigner().get());
  ret = checker3.Check(root_graph);
  EXPECT_NE(ret, SUCCESS);
  hcom_1->GetOpDescBarePtr()->SetInputOffset(origin_offset);

  // 构造scatter_nd输出size过大
  auto tensor_desc = hcom_1->GetOpDescBarePtr()->MutableInputDesc(0);
  int64_t origin_mem_size = 0;
  TensorUtils::GetSize(*tensor_desc, origin_mem_size);
  TensorUtils::SetSize(*tensor_desc, 1000 * origin_mem_size);
  AtomicCleanChecker checker4(mem_assigner.GetGraphMemoryAssigner().get());
  ret = checker4.Check(root_graph);
  EXPECT_NE(ret, SUCCESS);
  TensorUtils::SetSize(*tensor_desc, origin_mem_size);

  // 找不到memset节点
  mem_set_node->GetOpDescBarePtr()->SetType("Unknow");
  AtomicCleanChecker checker5(mem_assigner.GetGraphMemoryAssigner().get());
  ret = checker5.Check(root_graph);
  EXPECT_NE(ret, SUCCESS);
}

/*
 *         data       mem_set
 *          |        /  /(ctrl)
 *     a  scatter_nd   /
 *      \  /          /
 *        b+-----+
 *        |
 *     netoutput
 */
TEST_F(UtestAtomicCleanChecker, AtomicCleanWorkspaceCheckFailed) {
  auto root_graph = block_mem_ut::BuildAtomicCleanWorkspaceGraph();

  MemoryAssigner mem_assigner(root_graph);
  std::map<uint64_t, size_t> mem_offsets;
  size_t zero_copy_mem_size;
  EXPECT_EQ(mem_assigner.AssignMemory(mem_offsets, zero_copy_mem_size), SUCCESS);

  // 构造memset中的地址错误
  auto mem_set_node = root_graph->FindNode("mem_set");
  ASSERT_NE(mem_set_node, nullptr);
  const auto memset_origin_workspace = mem_set_node->GetOpDescBarePtr()->GetWorkspace();
  mem_set_node->GetOpDescBarePtr()->SetWorkspace({10000000});

  AtomicCleanChecker checker1(mem_assigner.GetGraphMemoryAssigner().get());
  auto ret = checker1.Check(root_graph);
  EXPECT_NE(ret, SUCCESS);
  mem_set_node->GetOpDescBarePtr()->SetWorkspace(memset_origin_workspace);

  // 构造memset中的size错误
  const auto memset_origin_size = mem_set_node->GetOpDescBarePtr()->GetWorkspaceBytes();
  mem_set_node->GetOpDescBarePtr()->SetWorkspaceBytes({2});
  AtomicCleanChecker checker2(mem_assigner.GetGraphMemoryAssigner().get());
  ret = checker2.Check(root_graph);
  EXPECT_NE(ret, SUCCESS);
  mem_set_node->GetOpDescBarePtr()->SetWorkspaceBytes(memset_origin_size);

  // 构造memset中的type错误
  std::vector<int64_t> workspace_type_list{64};
  ge::AttrUtils::SetListInt(mem_set_node->GetOpDescBarePtr(), ATTR_NAME_WORKSPACE_TYPE_LIST, workspace_type_list);
  AtomicCleanChecker checker22(mem_assigner.GetGraphMemoryAssigner().get());
  ret = checker22.Check(root_graph);
  EXPECT_NE(ret, SUCCESS);
  ge::AttrUtils::SetListInt(mem_set_node->GetOpDescBarePtr(), ATTR_NAME_WORKSPACE_TYPE_LIST, {});

  // 构造scatter_nd的地址错误
  auto scatter_nd = root_graph->FindNode("scatter_nd");
  ASSERT_NE(scatter_nd, nullptr);

  const auto origin_offset = scatter_nd->GetOpDescBarePtr()->GetWorkspace();
  scatter_nd->GetOpDescBarePtr()->SetWorkspace({0, 100555});
  AtomicCleanChecker checker3(mem_assigner.GetGraphMemoryAssigner().get());
  ret = checker3.Check(root_graph);
  EXPECT_NE(ret, SUCCESS);
  scatter_nd->GetOpDescBarePtr()->SetWorkspace(origin_offset);

  // 构造scatter_nd输出size过大
  auto origin_w_size = scatter_nd->GetOpDescBarePtr()->GetWorkspaceBytes();
  scatter_nd->GetOpDescBarePtr()->SetWorkspaceBytes({10240, 20480});
  AtomicCleanChecker checker4(mem_assigner.GetGraphMemoryAssigner().get());
  ret = checker4.Check(root_graph);
  EXPECT_NE(ret, SUCCESS);
  scatter_nd->GetOpDescBarePtr()->SetWorkspaceBytes(origin_w_size);

  // workspace中offset较多，超过日志长度，分多行打印
  std::vector<int64_t> long_workspace;
  std::vector<int64_t> long_workspace_bytes;
  for (int64_t i = 0; i < 1000; ++i) {
    long_workspace.emplace_back(i);
    long_workspace_bytes.emplace_back(1);
  }
  mem_set_node->GetOpDescBarePtr()->SetWorkspace(long_workspace);
  mem_set_node->GetOpDescBarePtr()->SetWorkspaceBytes(long_workspace_bytes);
  AtomicCleanChecker checker6(mem_assigner.GetGraphMemoryAssigner().get());
  ret = checker6.Check(root_graph);
  EXPECT_NE(ret, SUCCESS);
  mem_set_node->GetOpDescBarePtr()->SetWorkspace(memset_origin_workspace);
  mem_set_node->GetOpDescBarePtr()->SetWorkspaceBytes(memset_origin_size);

  // 找不到memset节点
  mem_set_node->GetOpDescBarePtr()->SetType("Unknow");
  AtomicCleanChecker checker5(mem_assigner.GetGraphMemoryAssigner().get());
  ret = checker5.Check(root_graph);
  EXPECT_NE(ret, SUCCESS);
}

/*
 *         data       mem_set
 *          |        /  /(ctrl)
 *     a  scatter_nd   /
 *      \  /          /
 *      Reshape+-----+
 *        |
 *     netoutput
 */
TEST_F(UtestAtomicCleanChecker, AtomicCleanOutput_MemsetOffsetOverlap) {
  auto root_graph = block_mem_ut::AtomicNodeConnectReShapeConnectNetoutput();

  MemoryAssigner mem_assigner(root_graph);
  std::map<uint64_t, size_t> mem_offsets;
  size_t zero_copy_mem_size;
  EXPECT_EQ(mem_assigner.AssignMemory(mem_offsets, zero_copy_mem_size), SUCCESS);

  // 构造memset的workspace中的offset，size有包含关系的场景
  auto mem_set_node = root_graph->FindNode("mem_set");
  ASSERT_NE(mem_set_node, nullptr);
  const auto memset_origin_workspace = mem_set_node->GetOpDescBarePtr()->GetWorkspace();
  std::vector<int64_t> fake_workspace;
  fake_workspace.emplace_back(memset_origin_workspace.back() - 2);
  fake_workspace.emplace_back(memset_origin_workspace.back() - 1);
  fake_workspace.emplace_back(memset_origin_workspace.back() + 1);
  mem_set_node->GetOpDescBarePtr()->SetWorkspace(fake_workspace);

  const auto memset_origin_workspace_sizes = mem_set_node->GetOpDescBarePtr()->GetWorkspaceBytes();
  std::vector<int64_t> fake_sizes;
  fake_sizes.emplace_back(memset_origin_workspace_sizes.back() + 10);
  fake_sizes.emplace_back(memset_origin_workspace_sizes.back() - 10);
  fake_sizes.emplace_back(memset_origin_workspace_sizes.back() - 10);
  mem_set_node->GetOpDescBarePtr()->SetWorkspaceBytes(fake_sizes);

  std::vector<int32_t> data_type_list{ge::DataType::DT_INT16, ge::DataType::DT_INT16, ge::DataType::DT_INT16};
  (void) AttrUtils::SetListInt(mem_set_node->GetOpDesc(), ATTR_NAME_ATOMIC_MEMSET_DTYPES, data_type_list);
  std::vector<int32_t> int_list = {0x1, 2, 3};
  (void) AttrUtils::SetListInt(mem_set_node->GetOpDesc(), ge::ATTR_NAME_ATOMIC_MEMSET_VALUES_INT, int_list);

  AtomicCleanChecker checker1(mem_assigner.GetGraphMemoryAssigner().get());
  auto ret = checker1.Check(root_graph);
  EXPECT_EQ(ret, SUCCESS);

  // 构造memset中的workspace offset有重叠的场景
  auto fake_workspace2 = memset_origin_workspace;
  fake_workspace2.emplace_back(memset_origin_workspace.back());
  mem_set_node->GetOpDescBarePtr()->SetWorkspace(fake_workspace2);

  auto fake_sizes2 = memset_origin_workspace_sizes;
  fake_sizes2.emplace_back(memset_origin_workspace_sizes.back() / 2);
  mem_set_node->GetOpDescBarePtr()->SetWorkspaceBytes(fake_sizes2);

  std::vector<int32_t> data_type_list2{ge::DataType::DT_INT16, ge::DataType::DT_INT16};
  (void) AttrUtils::SetListInt(mem_set_node->GetOpDesc(), ATTR_NAME_ATOMIC_MEMSET_DTYPES, data_type_list2);
  std::vector<int32_t> int_list3 = {0x1, 2};
  (void) AttrUtils::SetListInt(mem_set_node->GetOpDesc(), ge::ATTR_NAME_ATOMIC_MEMSET_VALUES_INT, int_list3);

  AtomicCleanChecker checker2(mem_assigner.GetGraphMemoryAssigner().get());
  auto ret2 = checker2.Check(root_graph);
  EXPECT_EQ(ret2, SUCCESS);


}
} // namespace ge
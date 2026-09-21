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
#include <memory>

#include "graph/build/memory/continuous_mem.h"
#include "../test_memory_shared_graph.h"

#include "macro_utils/dt_public_scope.h"
#include "graph/build/memory/max_block_mem_assigner.h"
#include "graph/manager/mem_manager.h"
#include "graph/ge_tensor.h"
#include "graph/normal_graph/op_desc_impl.h"
#include "graph/normal_graph/ge_tensor_impl.h"
#include "graph/attr_value.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"

namespace ge {
namespace {

struct NodeNameIndex {
  std::string name;
  uint32_t idx;
};
std::vector<NodeIndexIO> GetNodeIndexIoByName(const ComputeGraphPtr &graph,
                                                 const std::vector<NodeNameIndex> &nodes) {
  std::vector<NodeIndexIO> exp_nodes_out;
  for (const auto &name_index : nodes) {
    auto node = graph->FindNode(name_index.name);
    if (node == nullptr) {
      std::cerr << "find node by name [" << name_index.name << "] failed" << std::endl;
      return {};
    }
    if (node->GetOpDescBarePtr()->GetOutputsSize() < name_index.idx) {
      std::cerr << "node[" << name_index.name << "] only has " << node->GetOpDescBarePtr()->GetOutputsSize()
                << " outputs, but given idx: " << name_index.idx << std::endl;
      return {};
    }
    exp_nodes_out.emplace_back(NodeIndexIO{node, name_index.idx, kOut});
  }
  return exp_nodes_out;
}

Status CheckContinuousMem(const ComputeGraphPtr &graph, const std::vector<NodeNameIndex> &nodes) {
  ContinuousMemMng mng;
  if (mng.Init(graph) != SUCCESS) {
    std::cerr << "init failed." << std::endl;
    return FAILED;
  }
  if (nodes.empty()) {
    return SUCCESS;
  }

  const auto exp_nodes_out = GetNodeIndexIoByName(graph ,nodes);
  if (exp_nodes_out.empty()) {
    std::cerr << "GetNodeIndexIoByName return empty vector" << std::endl;
    return SUCCESS;
  }
  const auto continuous_mem = mng.GetContinuousMem(exp_nodes_out.front().node_ptr_, exp_nodes_out.front().index_);
  const auto act_nodes_out = continuous_mem.GetContinuousNodeOut();
  if (act_nodes_out.size() != exp_nodes_out.size()) {
    std::cerr << "continuous outs number not equal. expect: " << exp_nodes_out.size() <<
        ", but actual: " << act_nodes_out.size() << std::endl;
    return FAILED;
  }
  if (continuous_mem.GetAlignedSizes().size() != exp_nodes_out.size()) {
    std::cerr << "continuous mem GetAlignedSizes().size error. expect size: " << exp_nodes_out.size()
              << exp_nodes_out.size() << ", actual size: " << continuous_mem.GetAlignedSizes().size()
              << std::endl;
    return FAILED;
  }
  size_t exp_total_size = 0U;
  for (size_t i = 0U; i < act_nodes_out.size(); ++i) {
    const auto &exp = exp_nodes_out.at(i);
    const auto &act = act_nodes_out.at(i);
    if (exp.node_ptr_ != act.node_ptr_ || exp.index_ != act.index_) {
      std::cerr << "continous outs[" << i << "] check failed. expect: " << exp.node_ptr_->GetName()
                << "[out:" << exp.index_ << "]"
                << ", actual: " << act.node_ptr_->GetName() << "[out:" << act.index_ << "]" << std::endl;
      return FAILED;
    }
    auto exp_tensor = exp.node_ptr_->GetOpDescBarePtr()->GetOutputDesc(exp.index_);
    int64_t exp_size = 0;
    TensorUtils::GetSize(exp_tensor, exp_size);
    auto align_exp_size = static_cast<size_t>(exp_size);
    MemReuseUtils::AlignMemOffset(align_exp_size);

    if (align_exp_size != continuous_mem.GetAlignedSizes().at(i)) {
      std::cerr << "size check failed. index: " << i << ", align_exp_size: " << align_exp_size << ", actual size: "
                << continuous_mem.GetAlignedSizes().at(i) << ", node: " << exp.node_ptr_->GetName() << ", out_index: "
                << exp.index_ << std::endl;
      return FAILED;
    }
    exp_total_size += align_exp_size;
    const auto is_first_node = continuous_mem.IsFirstNodeOut(exp.node_ptr_, exp.index_);
    if ((i == 0U) != is_first_node) {
      std::cerr << "IsFirstNodeOut check failed. i: " << i << ", is_first_node: " << is_first_node << ", node: "
                << exp.node_ptr_->GetName() << ", out_index: " << exp.index_ << std::endl;
      return FAILED;
    }
    if (!mng.IsFound(exp.node_ptr_, exp.index_)) {
      std::cerr << "IsFound check failed. i: " << i << ", node: "
                << exp.node_ptr_->GetName() << ", out_index: " << exp.index_ << std::endl;
      return FAILED;
    }
    const auto is_need_assign = mng.IsNeedAssignMemory(exp.node_ptr_, exp.index_);
    if ((i == 0U) != is_need_assign) {
      std::cerr << "IsNeedAssignMemory check failed. i: " << i << ", is_need_assign: " << is_need_assign << ", node: "
                << exp.node_ptr_->GetName() << ", out_index: " << exp.index_ << std::endl;
      return FAILED;
    }
  }
  if (exp_total_size != continuous_mem.GetTotalSize()) {
    std::cerr << "expect total size: " << exp_total_size << ", actual total size: " << continuous_mem.GetTotalSize()
              << std::endl;
    return FAILED;
  }
  return SUCCESS;
}
}
class UtestContinuousMem : public testing::Test {};

/*
 *     hcom1   b
 *    /  | |  /
 *   a   hcom2
 */
TEST_F(UtestContinuousMem, IsTargetScenario_ReturnTrue) {
  auto graph = block_mem_ut::BuildContinuousOutInWithOutAsHeader();
  auto hcom2 = graph->FindNode("hcom2");
  ASSERT_NE(hcom2, nullptr);
  ContinuousMemMng mng;
  ContinuousMemScenario scenario;
  EXPECT_TRUE(mng.IsTargetScenario(hcom2.get(), scenario));
  EXPECT_EQ(scenario, ContinuousMemScenario::kContinuousMemScenarioOutIn);
}

/*
 *     hcom1   b
 *    /  | |  /
 *   a   hcom2
 */
TEST_F(UtestContinuousMem, ContinuousOutIn_SUCCESS_OutAsHeader) {
  auto graph = block_mem_ut::BuildContinuousOutInWithOutAsHeader();
  ContinuousMemMng mng;
  ASSERT_EQ(mng.Init(graph), SUCCESS);

  auto hcom1 = graph->FindNode("hcom1");
  ASSERT_NE(hcom1, nullptr);
  const auto &continuous_mem = mng.GetContinuousMem(hcom1.get(), 0);
  EXPECT_TRUE(continuous_mem.IsUseOneBlock());
  EXPECT_FALSE(continuous_mem.IsReuse());

  EXPECT_EQ(CheckContinuousMem(graph, {{"hcom1", 0}, {"hcom1", 1}, {"hcom1", 2}, {"b", 0}}), SUCCESS);
}

/*
 *     hcom1
 *    / | |  \
 *   a  hcom2  b
 */
TEST_F(UtestContinuousMem, ContinuousOutIn_SUCCESS_OutAsHeaderAndTail) {
  auto graph = block_mem_ut::BuildContinuousOutInWithOutAsHeaderAndTail();
  ContinuousMemMng mng;
  ASSERT_EQ(mng.Init(graph), SUCCESS);

  auto hcom1 = graph->FindNode("hcom1");
  ASSERT_NE(hcom1, nullptr);
  const auto &continuous_mem = mng.GetContinuousMem(hcom1.get(), 0);
  EXPECT_TRUE(continuous_mem.IsUseOneBlock());
  EXPECT_FALSE(continuous_mem.IsReuse());

  EXPECT_EQ(CheckContinuousMem(graph, {{"hcom1", 0}, {"hcom1", 1}, {"hcom1", 2}, {"hcom1", 3}}), SUCCESS);
}

/*
 *  a  hcom1
 *   \  | | \
 *     hcom2 b
 */
TEST_F(UtestContinuousMem, ContinuousOutIn_SUCCESS_InAsHeader) {
  auto graph = block_mem_ut::BuildContinuousOutInWithInAsHeader();
  ContinuousMemMng mng;
  ASSERT_EQ(mng.Init(graph), SUCCESS);

  auto hcom1 = graph->FindNode("hcom1");
  ASSERT_NE(hcom1, nullptr);
  const auto &continuous_mem = mng.GetContinuousMem(hcom1.get(), 0);
  EXPECT_TRUE(continuous_mem.IsUseOneBlock());
  EXPECT_FALSE(continuous_mem.IsReuse());

  EXPECT_EQ(CheckContinuousMem(graph, {{"a", 0}, {"hcom1", 0}, {"hcom1", 1}, {"hcom1", 2}}), SUCCESS);
}

/*
 *  a  hcom1 b
 *   \  | | /
 *     hcom2
 */
TEST_F(UtestContinuousMem, ContinuousOutIn_SUCCESS_InAsHeaderAndTail) {
  auto graph = block_mem_ut::BuildContinuousOutInWithInAsHeaderAndTail();
  ContinuousMemMng mng;
  ASSERT_EQ(mng.Init(graph), SUCCESS);

  auto hcom1 = graph->FindNode("hcom1");
  ASSERT_NE(hcom1, nullptr);
  const auto &continuous_mem = mng.GetContinuousMem(hcom1.get(), 0);
  EXPECT_TRUE(continuous_mem.IsUseOneBlock());
  EXPECT_FALSE(continuous_mem.IsReuse());

  EXPECT_EQ(CheckContinuousMem(graph, {{"a", 0}, {"hcom1", 0}, {"hcom1", 1}, {"b", 0}}), SUCCESS);
}

/*
 *      hcom1
 *   / ||   ||  \
 *  a hcom2 hcom3 b
 */
TEST_F(UtestContinuousMem, ContinuousOutIn_SUCCESS_TwoContinuousInNode) {
  auto graph = block_mem_ut::BuildContinuousOutInWithTwoContinuousInNode();
  ContinuousMemMng mng;
  ASSERT_EQ(mng.Init(graph), SUCCESS);

  auto hcom1 = graph->FindNode("hcom1");
  ASSERT_NE(hcom1, nullptr);
  const auto &continuous_mem = mng.GetContinuousMem(hcom1.get(), 0);
  EXPECT_TRUE(continuous_mem.IsUseOneBlock());
  EXPECT_FALSE(continuous_mem.IsReuse());

  EXPECT_EQ(CheckContinuousMem(graph, {{"hcom1", 0}, {"hcom1", 1}, {"hcom1", 2}, {"hcom1", 3}, {"hcom1", 4}, {"hcom1", 5}}), SUCCESS);
}
/*
 *      hcom1        c d
 *   / ||   ||  \  \ | |
 *  a hcom2 hcom3 b hcom4
 */
TEST_F(UtestContinuousMem, ContinuousOutIn_SUCCESS_TwoContinuousInNodeAndRefNode) {
  auto graph = block_mem_ut::BuildContinuousOutInWithTwoContinuousInNodeAndRefNode();
  ContinuousMemMng mng;
  ASSERT_EQ(mng.Init(graph), SUCCESS);

  auto hcom1 = graph->FindNode("hcom1");
  ASSERT_NE(hcom1, nullptr);
  const auto &continuous_mem = mng.GetContinuousMem(hcom1.get(), 0);
  EXPECT_TRUE(continuous_mem.IsUseOneBlock());
  EXPECT_FALSE(continuous_mem.IsReuse());

  EXPECT_EQ(CheckContinuousMem(graph, {{"hcom1", 0}, {"hcom1", 1}, {"hcom1", 2}, {"hcom1", 3}, {"hcom1", 4},
                                       {"hcom1", 5}, {"hcom1", 6}, {"c", 0}, {"d", 0}}), SUCCESS);
}
/*
 *    a hcom1 b hcom2 c
 *    \  ||   |  ||   |
 *          hcom3
 */
TEST_F(UtestContinuousMem, ContinuousOutIn_SUCCESS_TwoContinuousOutNodeAndRefNode) {
  auto graph = block_mem_ut::BuildContinuousOutInWithTwoContinuousOutNodeAndRefNode();
  ContinuousMemMng mng;
  ASSERT_EQ(mng.Init(graph), SUCCESS);

  auto hcom1 = graph->FindNode("hcom1");
  ASSERT_NE(hcom1, nullptr);
  const auto &continuous_mem = mng.GetContinuousMem(hcom1.get(), 0);
  EXPECT_TRUE(continuous_mem.IsUseOneBlock());
  EXPECT_FALSE(continuous_mem.IsReuse());

  EXPECT_EQ(CheckContinuousMem(graph, {{"a", 0}, {"hcom1", 0}, {"hcom1", 1}, {"b", 0}, {"hcom2", 0},
                                       {"hcom2", 1}, {"c", 0}}), SUCCESS);
}
/*
 *    hcom1
 *   / |   \
 *  a hcom2  c
 */
TEST_F(UtestContinuousMem, ContinuousOutIn_SUCCESS_OneInput) {
  auto graph = block_mem_ut::BuildContinuousOutInWithOneInput();
  EXPECT_EQ(CheckContinuousMem(graph, {}), SUCCESS);
}
/*
 *    hcom1
 *   / |   \
 *  a hcom2  c
 */
TEST_F(UtestContinuousMem, ContinuousOutIn_SUCCESS_OneOutput) {
  auto graph = block_mem_ut::BuildContinuousOutInWithOneOutput();
  EXPECT_EQ(CheckContinuousMem(graph, {}), SUCCESS);
}
/*
 *  a  hcom1 hcom2 hcom3
 *  | |    | |   | |   |
 *  hcom4 hcom5  hcom6 b
 */
TEST_F(UtestContinuousMem, ContinuousOutIn_SUCCESS_ThreeNode) {
  auto graph = block_mem_ut::BuildContinuousOutInWithThreeNode();
  EXPECT_EQ(CheckContinuousMem(graph, {{"a", 0}, {"hcom1", 0}, {"hcom1", 1}
                                       , {"hcom2", 0}, {"hcom2", 1}
                                       , {"hcom3", 0}, {"hcom3", 1}}), SUCCESS);
}

/*
 *  a  hcom1
 *  |  | | \
 *  hcom2 hcom3
 */
TEST_F(UtestContinuousMem, ContinuousOutIn_SUCCESS_VisitLastNodeFirst) {
  auto graph = block_mem_ut::BuildContinuousOutInVisitLastNodeFirst();
  EXPECT_EQ(CheckContinuousMem(graph, {{"a", 0}, {"hcom1", 0}
                                       , {"hcom1", 1}, {"hcom1", 2}}), SUCCESS);
}

TEST_F(UtestContinuousMem, GetContinuousMemFailed) {
  ContinuousMemMng mng;
  auto ret = mng.GetContinuousMem(nullptr, 0);
  EXPECT_EQ(ret.GetAlignedSizes().size(), 0);

  auto graph = block_mem_ut::BuildContinuousOutInWithOneOutput();
  auto node = graph->FindNode("a");
  ret = mng.GetContinuousMem(node.get(), 0);
  EXPECT_EQ(ret.GetAlignedSizes().size(), 0);
  EXPECT_TRUE(mng.IsNeedAssignMemory(node.get(), 0));
}
} // namespace ge


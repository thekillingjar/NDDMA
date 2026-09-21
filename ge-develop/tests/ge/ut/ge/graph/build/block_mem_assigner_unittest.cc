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
#include <vector>

#include "macro_utils/dt_public_scope.h"
#include "graph/build/memory/block_mem_assigner.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/tensor_utils.h"
#include "graph/manager/graph_var_manager.h"
#include "common/sgt_slice_type.h"

#include "macro_utils/dt_public_unscope.h"
#include "graph/build/memory/binary_block_mem_assigner.h"
#include "graph/ge_context.h"
#include "test_memory_shared_graph.h"
#include "framework/memory/memory_assigner.h"
#include "common/mem_conflict_share_graph.h"
using namespace std;
using namespace testing;

namespace ge {
using namespace block_mem_ut;
namespace {
ge::OpDescPtr CreateOpWithWsSize(const string &name, int64_t wsByte, const string &type = "some",
                                 int64_t size = 1024, std::vector<int64_t> shape = {1, 1, 16, 8},
                                 Format format = FORMAT_NCHW, DataType data_type = DT_FLOAT) {
  ge::OpDescPtr op_def = std::make_shared<ge::OpDesc>(name, type);
  auto desc_temp_ptr = std::make_shared<ge::GeTensorDesc>();
  desc_temp_ptr->SetShape(GeShape(shape));
  desc_temp_ptr->SetFormat(format);
  desc_temp_ptr->SetDataType(data_type);
  desc_temp_ptr->SetOriginFormat(format);
  desc_temp_ptr->SetOriginShape(GeShape(shape));
  desc_temp_ptr->SetOriginDataType(data_type);

  auto desc_temp = *desc_temp_ptr;

  TensorUtils::SetSize(desc_temp, size);
  op_def->AddInputDesc(desc_temp);
  op_def->AddOutputDesc(desc_temp);
  if (wsByte != 0) {
    std::vector<int64_t> workspace_bytes;
    workspace_bytes.push_back(wsByte);
    op_def->SetWorkspaceBytes(workspace_bytes);
  }
  return op_def;
}

bool CheckIntersection(const std::vector<int64_t> &left_range, const std::vector<int64_t> &right_range) {
  if (left_range.empty() || right_range.empty()) {
    return false;
  }
  if (left_range[0] > right_range[1] || left_range[1] < right_range[0]) {
    return false;
  }
  return true;
}
}
class UtestBlockMemAssigner : public testing::Test {
 protected:
  void SetUp() {
  }
  void TearDown() {
  }

  class FakBlockMemAssigner : public BlockMemAssigner {
   public:
    FakBlockMemAssigner(MemAssistInfo &mem_assist_info) : BlockMemAssigner(mem_assist_info){};

   public:
    virtual Status GetMemoryRanges(std::vector<int64_t> &ranges) override {
      ranges.push_back(1);
      return SUCCESS;
    };
  };
  ReuseStrategy reuse_strategy_{};
};

TEST_F(UtestBlockMemAssigner, Normal) {
    EXPECT_NO_THROW(auto p1 = std::make_shared<MemoryBlock>(reuse_strategy_, 1024));
    auto p1 = std::make_shared<MemoryBlock>(reuse_strategy_, 1024);
    EXPECT_EQ(p1->Size(), 1024);
}

TEST_F(UtestBlockMemAssigner, AssignOutputMemoryWithReuse) {
    auto builder = std::make_shared<block_mem_ut::GraphBuilder>("graph");
    auto node = builder->AddNode("node", DATA, 1, 1);
    ComputeGraphPtr compute_graph = builder->GetGraph();
    MemAssistInfo mem_assist_info;
    mem_assist_info.compute_graph = compute_graph;
    auto p1 = std::make_shared<FakBlockMemAssigner>(mem_assist_info);
    std::vector<int64_t> ranges;
    p1->op_reuse_env_valid_ = true;
    EXPECT_EQ(p1->AssignOutputMemoryWithReuse(node, ranges), SUCCESS);
    std::vector<int64_t> memorys_type;
    ge::AttrUtils::SetListInt(node->GetOpDesc(), "_output_memory_type", memorys_type);
    EXPECT_EQ(p1->AssignOutputMemoryWithReuse(node, ranges), INTERNAL_ERROR);
}

TEST_F(UtestBlockMemAssigner, AssignOutputMemoryWithReuseL1)
{
  auto root_builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &add = root_builder.AddNode("add", ADD, 0, 1);
  std::vector<int64_t> memorys_type = {RT_MEMORY_L1};
  ge::AttrUtils::SetListInt(add->GetOpDesc(), "_output_memory_type", memorys_type);

  const auto &netout = root_builder.AddNode("NETOUTPUT", NETOUTPUT, 1, 1);
  root_builder.AddDataEdge(add, 0, netout, 0);
  ComputeGraphPtr compute_graph = root_builder.GetGraph();

  MemAssistInfo mem_assist_info;
  mem_assist_info.compute_graph = compute_graph;
  auto p1 = std::make_shared<FakBlockMemAssigner>(mem_assist_info);
  std::vector<int64_t> ranges;
  p1->op_reuse_env_valid_ = false;
  EXPECT_EQ(p1->AssignOutputMemoryWithReuse(add, ranges), SUCCESS);
}

TEST_F(UtestBlockMemAssigner, AssignMemoryWithReuse) {
    const char *const OP_NO_REUSE_MEM = "OP_NO_REUSE_MEM";
    setenv(OP_NO_REUSE_MEM, "FusedMulAddN,BatchNorm", 1);
    auto builder = std::make_shared<block_mem_ut::GraphBuilder>("graph");
    ASSERT_NE(builder, nullptr);
    auto node = builder->AddNode("node", DATA, 1, 1);
    ASSERT_NE(node, nullptr);
    ComputeGraphPtr root_graph = builder->GetGraph();
    ASSERT_NE(root_graph, nullptr);
    auto p1_sub_builder = block_mem_ut::GraphBuilder("partitioncall_0_sub");
    const auto &partitioncall_0_const1 = p1_sub_builder.AddNode("partitioncall_0_const1", CONSTANT, 0, 1);
    const auto &partitioncall_0_netoutput = p1_sub_builder.AddNode("partitioncall_0_netoutput", NETOUTPUT, 1, 1);
    const auto &sub_graph = p1_sub_builder.GetGraph();
    sub_graph->SetParentNode(node);
    sub_graph->SetParentGraph(root_graph);
    ASSERT_EQ(root_graph->AddSubgraph(sub_graph->GetName(), sub_graph), SUCCESS);
    MemAssistInfo mem_assist_info;
    mem_assist_info.compute_graph = sub_graph;
    auto p1 = std::make_shared<FakBlockMemAssigner>(mem_assist_info);
    std::vector<int64_t> ranges;
    std::vector<int64_t> tvm_workspace_memory_type;
    tvm_workspace_memory_type.push_back(1);
    AttrUtils::SetListInt(partitioncall_0_const1->GetOpDesc(), "tvm_workspace_type", tvm_workspace_memory_type);
    EXPECT_NO_THROW(p1->AssignMemoryWithReuse(ranges));

    auto p2 = std::make_shared<FakBlockMemAssigner>(mem_assist_info);
    EXPECT_NO_THROW(p2->AssignMemoryWithReuse(ranges));
    unsetenv(OP_NO_REUSE_MEM);
}

TEST_F(UtestBlockMemAssigner, Assign) {
    auto builder = std::make_shared<block_mem_ut::GraphBuilder>("graph");
    auto node = builder->AddNode("node", DATA, 1, 1);
    ComputeGraphPtr compute_graph = builder->GetGraph();
    MemAssistInfo mem_assist_info;
    mem_assist_info.compute_graph = compute_graph;
    auto p1 = std::make_shared<FakBlockMemAssigner>(mem_assist_info);
    EXPECT_EQ(p1->Assign(), SUCCESS);
}

TEST_F(UtestBlockMemAssigner, GetWorkSpaceMemoryType) {
    auto builder = std::make_shared<block_mem_ut::GraphBuilder>("graph");
    auto node = builder->AddNode("node", DATA, 1, 1);
    ComputeGraphPtr compute_graph = builder->GetGraph();
    MemAssistInfo mem_assist_info;
    mem_assist_info.compute_graph = compute_graph;
    auto p1 = std::make_shared<FakBlockMemAssigner>(mem_assist_info);
    uint64_t memory_type;
    std::vector<bool> workspace_reuse_flag;

    size_t no_reuse_scope_size = 3;
    size_t index = 0U;
    bool is_p2p_memory = true;
    bool session_scope_memory = true;
    memory_type = p1->GetWorkSpaceMemoryType(no_reuse_scope_size, index, is_p2p_memory, session_scope_memory,
                                             workspace_reuse_flag);
    EXPECT_EQ(memory_type, RT_MEMORY_P2P_DDR);

    is_p2p_memory = false;
    memory_type = p1->GetWorkSpaceMemoryType(no_reuse_scope_size, index, is_p2p_memory, session_scope_memory,
                                             workspace_reuse_flag);
    EXPECT_EQ(memory_type, (kSessionScopeMemory | RT_MEMORY_HBM));

    session_scope_memory = false;
    memory_type = p1->GetWorkSpaceMemoryType(no_reuse_scope_size, index, is_p2p_memory, session_scope_memory,
                                             workspace_reuse_flag);
    EXPECT_EQ(memory_type, RT_MEMORY_HBM);
}

TEST_F(UtestBlockMemAssigner, IsZeroCopyBlock) {
  auto builder = std::make_shared<block_mem_ut::GraphBuilder>("graph");
  auto node = builder->AddNode("node", DATA, 1, 1);
  ComputeGraphPtr compute_graph = builder->GetGraph();
  MemAssistInfo mem_assist_info;
  mem_assist_info.compute_graph = compute_graph;
  auto p1 = std::make_shared<FakBlockMemAssigner>(mem_assist_info);
  ge::AttrUtils::SetBool(compute_graph, "_dynamic_shape_partitioned", true);
  EXPECT_EQ(p1->IsZeroCopyBlock(node, 0, false), true);
}

TEST_F(UtestBlockMemAssigner, IsZeroCopyBlock_Disable_zerocopy) {
  auto builder = std::make_shared<block_mem_ut::GraphBuilder>("graph");
  auto node = builder->AddNode("node", DATA, 1, 1);
  ComputeGraphPtr compute_graph = builder->GetGraph();
  MemAssistInfo mem_assist_info;
  mem_assist_info.compute_graph = compute_graph;
  auto p1 = std::make_shared<FakBlockMemAssigner>(mem_assist_info);
  ge::AttrUtils::SetBool(node->GetOpDesc(), "_is_multi_batch_shape_data", true);
  EXPECT_EQ(p1->IsZeroCopyBlock(node, 0, false), false);
}

TEST_F(UtestBlockMemAssigner, IsZeroCopyBlock_data_fb_refreshable) {
  ge::GetThreadLocalContext().SetGraphOption({{"ge.featureBaseRefreshable", "1"}});
  auto builder = std::make_shared<block_mem_ut::GraphBuilder>("graph");
  auto node = builder->AddNode("node", DATA, 1, 1);
  ComputeGraphPtr compute_graph = builder->GetGraph();
  MemAssistInfo mem_assist_info;
  mem_assist_info.compute_graph = compute_graph;
  auto p1 = std::make_shared<FakBlockMemAssigner>(mem_assist_info);
  EXPECT_EQ(p1->IsZeroCopyBlock(node, 0, false), true);

  ge::GetThreadLocalContext().SetGraphOption({{}});
}

TEST_F(UtestBlockMemAssigner, IsZeroCopyBlock_alloc_ByGE) {
  const uint64_t input_fusion_size = 25600U;
  std::map<std::string, std::string> options_map;
  options_map["ge.exec.graphIOMemAllocMode"] = "ByGE";
  options_map["ge.exec.input_fusion_size"] = std::to_string(input_fusion_size);
  ge::GetThreadLocalContext().SetGraphOption(options_map);

  auto builder = std::make_shared<block_mem_ut::GraphBuilder>("graph");
  auto node = builder->AddNode("node", DATA, 1, 1);
  ComputeGraphPtr compute_graph = builder->GetGraph();
  MemAssistInfo mem_assist_info;
  mem_assist_info.compute_graph = compute_graph;
  auto p1 = std::make_shared<FakBlockMemAssigner>(mem_assist_info);
  EXPECT_EQ(p1->IsZeroCopyBlock(node, 0, false, input_fusion_size + 1U), false);  // size > input_fusion_size
  EXPECT_EQ(p1->IsZeroCopyBlock(node, 0, false, input_fusion_size), true);        // size <= input_fusion_size
  ge::GetThreadLocalContext().SetGraphOption({{}});
}

TEST_F(UtestBlockMemAssigner, IsZeroCopyBlock_data_out_anchor_null) {
  auto builder = std::make_shared<block_mem_ut::GraphBuilder>("graph");
  auto node = builder->AddNode("node", DATA, 1, 1);
  ComputeGraphPtr compute_graph = builder->GetGraph();
  MemAssistInfo mem_assist_info;
  mem_assist_info.compute_graph = compute_graph;
  auto p1 = std::make_shared<FakBlockMemAssigner>(mem_assist_info);
  EXPECT_EQ(p1->IsZeroCopyBlock(node, 1, false), false);
}

TEST_F(UtestBlockMemAssigner, IsZeroCopyBlock_unsupport_dsa)
{
  auto root_builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &add = root_builder.AddNode("add", ADD, 0, 1);
  const auto &netout = root_builder.AddNode("NETOUTPUT", NETOUTPUT, 1, 1);
  root_builder.AddDataEdge(add, 0, netout, 0);
  const auto &root_graph = root_builder.GetGraph();

  auto op_desc = root_graph->FindNode("add")->GetOpDesc();
  op_desc->SetOpKernelLibName(ge::kEngineNameDsa.c_str());

  MemAssistInfo mem_assist_info;
  mem_assist_info.compute_graph = root_graph;
  auto p1 = std::make_shared<FakBlockMemAssigner>(mem_assist_info);
  EXPECT_EQ(p1->IsZeroCopyBlock(add, 0, false), false);
  EXPECT_EQ(p1->IsZeroCopyBlock(netout, 0, false), true);
}

TEST_F(UtestBlockMemAssigner, IsZeroCopyBlock_unsupport_hccl)
{
  auto root_builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &add = root_builder.AddNode("add", ADD, 0, 1);
  const auto &netout = root_builder.AddNode("NETOUTPUT", NETOUTPUT, 1, 1);
  root_builder.AddDataEdge(add, 0, netout, 0);
  const auto &root_graph = root_builder.GetGraph();

  auto op_desc = root_graph->FindNode("add")->GetOpDesc();
  op_desc->SetOpKernelLibName(ge::kEngineNameHccl.c_str());

  MemAssistInfo mem_assist_info;
  mem_assist_info.compute_graph = root_graph;
  auto p1 = std::make_shared<FakBlockMemAssigner>(mem_assist_info);
  EXPECT_EQ(p1->IsZeroCopyBlock(add, 0, false), false);
}

TEST_F(UtestBlockMemAssigner, IsZeroCopyBlock_unsupport_hccl_with_dynamic)
{
  ge::GetThreadLocalContext().SetGraphOption({{"ge.exec.static_model_addr_fixed", "1"}});
  auto root_builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &add = root_builder.AddNode("add", ADD, 0, 1);
  const auto &netout = root_builder.AddNode("NETOUTPUT", NETOUTPUT, 1, 1);
  root_builder.AddDataEdge(add, 0, netout, 0);
  const auto &root_graph = root_builder.GetGraph();

  auto op_desc = root_graph->FindNode("add")->GetOpDesc();
  op_desc->SetOpKernelLibName(ge::kEngineNameHccl.c_str());

  MemAssistInfo mem_assist_info;
  mem_assist_info.compute_graph = root_graph;
  (void)AttrUtils::SetBool(root_graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, true);
  auto p1 = std::make_shared<FakBlockMemAssigner>(mem_assist_info);
  EXPECT_EQ(p1->IsZeroCopyBlock(add, 0, false), false);
  ge::GetThreadLocalContext().SetGraphOption({{}});
}

TEST_F(UtestBlockMemAssigner, IsZeroCopyBlock_unsupport_addhccl)
{
  auto root_builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &add = root_builder.AddNode("add", ADD, 0, 1);
  const auto &addhccl = root_builder.AddNode("hccl", ADD, 0, 1);
  const auto &netout = root_builder.AddNode("NETOUTPUT", NETOUTPUT, 1, 1);
  root_builder.AddDataEdge(add, 0, netout, 0);
  root_builder.AddDataEdge(add, 0, addhccl, 0);
  const auto &root_graph = root_builder.GetGraph();

  auto op_desc = root_graph->FindNode("add")->GetOpDesc();
  op_desc->SetOpKernelLibName(ge::kEngineNameHccl.c_str());

  MemAssistInfo mem_assist_info;
  mem_assist_info.compute_graph = root_graph;
  auto p1 = std::make_shared<FakBlockMemAssigner>(mem_assist_info);
  EXPECT_EQ(p1->IsZeroCopyBlock(add, 0, false), false);
}

TEST_F(UtestBlockMemAssigner, IsZeroCopyBlockWithSubgraph)
{
  // root graph builder
  auto root_builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &data = root_builder.AddNode("data", DATA, 0, 1);
  const auto &a = root_builder.AddNode("A", ADD, 1, 1);
  const auto &partitioncall_0 = root_builder.AddNode("partitioncall_0", PARTITIONEDCALL, 1, 1);
  const auto &b = root_builder.AddNode("B", ADD, 1, 1);
  const auto &netout = root_builder.AddNode("NETOUTPUT", NETOUTPUT, 1, 1);

  root_builder.AddDataEdge(data, 0, a, 0);
  root_builder.AddDataEdge(a, 0, partitioncall_0, 0);
  root_builder.AddDataEdge(partitioncall_0, 0, b, 0);
  root_builder.AddDataEdge(b, 0, netout, 0);
  const auto &root_graph = root_builder.GetGraph();

  // partitioncall_0 sub graph build
  auto sub_builder = block_mem_ut::GraphBuilder("partitioncall_0_sub");
  const auto &partitioncall_0_data = sub_builder.AddNode("partitioncall_0_data", DATA, 1, 1);
  const auto &partitioncall_0_a = sub_builder.AddNode("partitioncall_0_A", ADD, 1, 1);
  const auto &partitioncall_0_netoutput = sub_builder.AddNode("partitioncall_0_netoutput", NETOUTPUT, 1, 1);

  AttrUtils::SetInt(partitioncall_0_data->GetOpDesc(), "_parent_node_index", 0);
  AttrUtils::SetInt(partitioncall_0_netoutput->GetOpDesc()->MutableInputDesc(0), "_parent_node_index", 0);

  sub_builder.AddDataEdge(partitioncall_0_data, 0, partitioncall_0_a, 0);
  sub_builder.AddDataEdge(partitioncall_0_a, 0, partitioncall_0_netoutput, 0);
  const auto &sub_graph = sub_builder.GetGraph();
  sub_graph->SetParentNode(partitioncall_0);
  sub_graph->SetParentGraph(root_graph);
  partitioncall_0->GetOpDesc()->AddSubgraphName("partitioncall_0");
  partitioncall_0->GetOpDesc()->SetSubgraphInstanceName(0, "partitioncall_0_sub");

  root_graph->AddSubgraph(sub_graph->GetName(), sub_graph);
  root_graph->TopologicalSorting();

  MemAssistInfo mem_assist_info;
  mem_assist_info.compute_graph = root_graph;
  auto p1 = std::make_shared<FakBlockMemAssigner>(mem_assist_info);
  EXPECT_EQ(p1->IsZeroCopyBlock(partitioncall_0_a, 0, false), false);

  ge::AttrUtils::SetBool(root_graph, "_dynamic_shape_partitioned", true);
  EXPECT_EQ(p1->IsZeroCopyBlock(partitioncall_0_a, 0, false), false);
}

TEST_F(UtestBlockMemAssigner, GetLifeEnd) {
  size_t block_size = 512;
  int64_t stream_id = 0;
  bool is_reuse_mem = false;
  OpMemoryType memory_type = kOutput;
  MemoryBlock block(reuse_strategy_, block_size, stream_id, is_reuse_mem, memory_type);
  auto builder = std::make_shared<block_mem_ut::GraphBuilder>("graph");
  auto n = builder->AddNode("node", DATA, 1, 1);
  NodeTypeIndex node_type_index{n.get(), memory_type, 0, false, 0, -1};
  block.AddNodeTypeIndex(node_type_index, 512, 512, 0);
  block.SetLifeTimeEnd(0, 1);
  EXPECT_EQ(block.GetLifeEnd(1), 0);
}

TEST_F(UtestBlockMemAssigner, IsPostReuse_True) {
  auto builder = std::make_shared<block_mem_ut::GraphBuilder>("graph");
  auto n = builder->AddNode("node", DATA, 1, 1);
  NodeIndexIO node_index_io(n, 0, kOut);
  MemAssistInfo mem_assist_info;
  mem_assist_info.anchor_to_symbol[node_index_io.ToString()] = node_index_io.ToString();

  std::list<NodeIndexIO> symbol_list;
  symbol_list.push_back(node_index_io);
  mem_assist_info.symbol_to_anchors.insert(pair<std::string, std::list<NodeIndexIO>>("node_out_0", symbol_list));

  BinaryBlockMemAssigner mem_assigner(mem_assist_info);
  bool is_reuse_zero_copy = false;
  EXPECT_EQ(mem_assigner.GetAllRefCount(node_index_io, is_reuse_zero_copy), 0);
  bool diff_stream_prior = false;
  EXPECT_TRUE(mem_assigner.IsPostReuse(node_index_io.ToString(), diff_stream_prior));
}

TEST_F(UtestBlockMemAssigner, IsPostReuse_False) {
  MemAssistInfo mem_assist_info;
  BinaryBlockMemAssigner mem_assigner(mem_assist_info);
  EXPECT_FALSE(mem_assigner.IsPostReuse(nullptr));
}

TEST_F(UtestBlockMemAssigner, IsNodeOutputUseSameMemWithNetOutput_False_SymbolNotFound) {
  auto builder = std::make_shared<block_mem_ut::GraphBuilder>("graph");
  auto a = builder->AddNode("node", DATA, 1, 1);
  MemAssistInfo mem_assist_info;
  BinaryBlockMemAssigner mem_assigner(mem_assist_info);
  EXPECT_FALSE(mem_assigner.IsNodeOutputUseSameMemWithNetOutput(a, 0));
}

TEST_F(UtestBlockMemAssigner, IsNodeOutputUseSameMemWithNetOutput_False_AnchorNotFound) {
  auto builder = std::make_shared<block_mem_ut::GraphBuilder>("graph");
  auto a = builder->AddNode("node", DATA, 1, 1);
  NodeIndexIO node_index_io(a, 0, kOut);
  MemAssistInfo mem_assist_info;
  mem_assist_info.anchor_to_symbol[node_index_io.ToString()] = node_index_io.ToString();
  BinaryBlockMemAssigner mem_assigner(mem_assist_info);
  EXPECT_FALSE(mem_assigner.IsNodeOutputUseSameMemWithNetOutput(a, 0));
}

TEST_F(UtestBlockMemAssigner, IsNodeOutputUseSameMemWithNetOutput_True) {
  auto builder = std::make_shared<block_mem_ut::GraphBuilder>("graph");
  auto a = builder->AddNode("node", DATA, 1, 1);
  auto b = builder->AddNode("node_b", NETOUTPUT, 1, 1);
  NodeIndexIO node_index_io_a(a, 0, kOut);
  NodeIndexIO node_index_io_b(b, 0, kOut);
  MemAssistInfo mem_assist_info;
  mem_assist_info.anchor_to_symbol[node_index_io_a.ToString()] = node_index_io_a.ToString();
  mem_assist_info.symbol_to_anchors[node_index_io_a.ToString()].emplace_back(node_index_io_b);
  BinaryBlockMemAssigner mem_assigner(mem_assist_info);
  EXPECT_TRUE(mem_assigner.IsNodeOutputUseSameMemWithNetOutput(a, 0));
}

//     data
//      |
//      a (stream 0)
//      |                +---------------+
//  partitioncall0-------| data          |
//      |                |  |            |
//      c (stream 0)     |  b (stream 1) |
//      |                |  |            |
//      d (stream 0)     | netoutput1    |
//      |                +---------------+
//   netoutput
//
// 子图data连接的节点stream和子图输入不一致，校验data的复用
TEST_F(UtestBlockMemAssigner, SubgraphDataStreamIsDifferentWithInput_CheckReuse) {
  ge::ComputeGraphPtr graph = BuildSubGraphWithDiffStream();
  MemAssistInfo mem_assist_info;
  mem_assist_info.compute_graph = graph;
  auto ret = GraphUtils::GetRefMapping(graph, mem_assist_info.symbol_to_anchors,
                                       mem_assist_info.anchor_to_symbol);
  EXPECT_EQ(ret, SUCCESS);
  std::vector<int64_t> ranges;
  BinaryBlockMemAssigner assigner(mem_assist_info);
  assigner.SetReuseStrategy(ReuseStrategy{false, true, false, true});
  assigner.GetMemoryRanges(ranges);
  assigner.AssignMemoryWithReuse(ranges);
  const auto blocks = assigner.GetMemoryBlocks();
  bool has_checked = false;
  assigner.SetOpMemOffset(false);
  for (const auto block : blocks) {
    for (const auto &node : block->node_type_index_list_) {
      if (node.mem_type_ != kOutput) {
        continue;
      }
      if (node.node_->GetOpDesc()->GetName() == "a") {
        EXPECT_EQ(block->NodeTypeIndexList().back().out_stream_count_, 1U);
        EXPECT_EQ(block->GetLifeEnd(0), kMaxLifeTime);
        EXPECT_EQ(block->GetLifeEnd(1), 4);
        int64_t end_stream = 0;
        EXPECT_EQ(block->GetLifeEnd(1, end_stream), 4);
        EXPECT_EQ(end_stream, 1);
        has_checked = true;
      }
    }
  }
  EXPECT_TRUE(has_checked);
}

//     data
//      |
//      a (stream 2)
//      +---------------+
//      |               |
//      b (stream 1)  RefNode (stream 0)
//      |               |
//      |               c (stream 0)
//      |               |
//      |               d (stream 0)
//      |
//   netoutput
//
// a单输出多引用，同时给b和 RefNode， 流不同。
// 校验a所在block的out_stream_count_为2（block的out_stream_count实际就是block上最后一个节点的）
//
// 要解决的问题: a的输出实际是给到了两个stream，但是在二级复用时，调用MemoryBlock::GetLifeEnd获取block的life end时，
// node_type_index_list_.back().out_stream_count_ 实际为1，导致误判。
//
// 同时也修改了GetNodeMaxLifeBySymbol中streams添加的逻辑，对于partitioned_call，data这种节点，并不执行，没有真正的stream，
// 所以不往streams中添加。否则SubgraphDataStreamIsDifferentWithInput_CheckReuse这个用例中的场景，由于partitioncall0是流0，b是流1，
// 所以会认为a的输出给到了2个流，但是实际上partitioned_call不执行，且子图中data也不执行，所以a的内存只给到了b，也就是流1.
TEST_F(UtestBlockMemAssigner, SingleOutputConnectMultiStreamAndRefNode_CheckBlockOutStreamCount) {
  ge::ComputeGraphPtr graph = BuildSingleOutputConnectMultiStreamAndRefNode();
  MemAssistInfo mem_assist_info;
  mem_assist_info.compute_graph = graph;
  auto ret = GraphUtils::GetRefMapping(graph, mem_assist_info.symbol_to_anchors,
                                       mem_assist_info.anchor_to_symbol);
  EXPECT_EQ(ret, SUCCESS);
  std::vector<int64_t> ranges;
  BinaryBlockMemAssigner assigner(mem_assist_info);
  assigner.SetReuseStrategy(ReuseStrategy{false, true, false, true});
  assigner.GetMemoryRanges(ranges);
  assigner.AssignMemoryWithReuse(ranges);
  const auto blocks = assigner.GetMemoryBlocks();
  bool has_checked = false;
  assigner.SetOpMemOffset(false);
  for (const auto block : blocks) {
    for (const auto &node : block->node_type_index_list_) {
      if (node.mem_type_ != kOutput) {
        continue;
      }
      if (node.node_->GetOpDesc()->GetName() == "a") {
        EXPECT_EQ(block->NodeTypeIndexList().back().out_stream_count_, 2U);
        EXPECT_EQ(block->GetLifeEnd(0), kMaxLifeTime);
        EXPECT_EQ(block->GetLifeEnd(1), kMaxLifeTime);
        has_checked = true;
      }
    }
  }
  EXPECT_TRUE(has_checked);
}

//     data
//      |
//      a (stream 2)
//      +-------------------+------------------------+
//      |                   |                        |
//      b (stream 1)-ctr-> RefNode (stream 0)    RefNode2 (stream 2)
//      |                    |                       |
//      |                    c (stream 0)            e (stream 2)
//      |                    |                       |
//      |                    d (stream 0)            | d不要连接控制边到refnode2
//      |                    |                       |
//      +---------------------+----------------------+
//      f (stream 2)
//      |
//   netoutput
//
// 与上面用例的区别在于多了一个RefNode2，并且RefNode2的stream与a是相同的。
//
// 要解决的问题: 在问题代码中，ReleaseMemory时，a所在block最后一个节点是RefNode2, e去释放a的block时，不会走到need_process_diff_stream的分支中，
// 导致a的block的GetLifeEnd函数拿到的life_end是错误的。应该是f，实际是e。
//
TEST_F(UtestBlockMemAssigner, SingleOutputConnectMultiStreamAndRefNode_LastRefNodeSameStream_CheckBlockOutStreamCount) {
  ge::ComputeGraphPtr graph = BuildSingleOutputConnectMultiStreamAndRefNode3();
  MemAssistInfo mem_assist_info;
  mem_assist_info.compute_graph = graph;
  auto ret = GraphUtils::GetRefMapping(graph, mem_assist_info.symbol_to_anchors,
                                       mem_assist_info.anchor_to_symbol);
  EXPECT_EQ(ret, SUCCESS);
  std::vector<int64_t> ranges;
  BinaryBlockMemAssigner assigner(mem_assist_info);
  assigner.SetReuseStrategy(ReuseStrategy{false, true, false, true});
  assigner.GetMemoryRanges(ranges);
  assigner.AssignMemoryWithReuse(ranges);
  const auto blocks = assigner.GetMemoryBlocks();
  bool has_checked = false;
  assigner.SetOpMemOffset(false);
  auto f = graph->FindNode("f");
  for (const auto block : blocks) {
    for (const auto &node : block->node_type_index_list_) {
      if (node.mem_type_ != kOutput) {
        continue;
      }
      if (node.node_->GetOpDesc()->GetName() == "a") {
        EXPECT_EQ(block->GetLifeEnd(2), f->GetOpDesc()->GetId());
        has_checked = true;
      }
    }
  }
  EXPECT_TRUE(has_checked);
}

//     data
//      |
//      a (stream 2)
//      +-------------------+------------------------+
//      |                   |                        |
//      b (stream 1)-ctr-> RefNode (stream 0)  +--> RefNode2 (stream 2)
//      |                    |                 |     |
//      |                    c (stream 0)     ctrl    e (stream 2)(a的终点错误的找到e，正确的应该找到f，f作为流1，流0回到流2的终点)
//      |                    |                |       |
//      |                    d (stream 0) ----+      ctrl
//      |                    |                        |
//      +---------------------+------------------------+
//      f (stream 2)
//      |
//   netoutput
//
// 要看护的场景，在e(7)释放a所在block时，GetNodeMaxLife中返回的值应该时符号和diff stream最大的值，但是由于逻辑错误，导致返回值比symbol的还小。
//
TEST_F(UtestBlockMemAssigner, SingleOutputConnectMultiStreamAndRefNode_LifeEndIsBiggerThanSymbolMax) {
  ge::ComputeGraphPtr graph = BuildSingleOutputConnectMultiStreamAndRefNode2();
  MemAssistInfo mem_assist_info;
  mem_assist_info.compute_graph = graph;
  auto ret = GraphUtils::GetRefMapping(graph, mem_assist_info.symbol_to_anchors,
                                       mem_assist_info.anchor_to_symbol);
  EXPECT_EQ(ret, SUCCESS);
  std::vector<int64_t> ranges;
  BinaryBlockMemAssigner assigner(mem_assist_info);
  assigner.SetReuseStrategy(ReuseStrategy{false, true, false, true});
  assigner.GetMemoryRanges(ranges);
  assigner.AssignMemoryWithReuse(ranges);
  const auto blocks = assigner.GetMemoryBlocks();
  bool has_checked = false;
  assigner.SetOpMemOffset(false);
  auto e = graph->FindNode("e");
  for (const auto block : blocks) {
    for (const auto &node : block->node_type_index_list_) {
      if (node.mem_type_ != kOutput) {
        continue;
      }
      if (node.node_->GetOpDesc()->GetName() == "a") {
        EXPECT_EQ(block->GetLifeEnd(2), e->GetOpDesc()->GetId());
        has_checked = true;
      }
    }
  }
  EXPECT_TRUE(has_checked);
}

TEST_F(UtestBlockMemAssigner, SeperateAtomicCleanAndContinousInput) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph_continuous_input_reuse");
  /**
   *          a:1  b:1
   *           |___|
   *             |
   *            c:3
   *             |
   *            d:2
   */
  // node c padding input continuous
  const string &type = "some";
  ge::OpDescPtr op_def_a = CreateOpWithWsSize("A", 1024, type, 2048);
  op_def_a->SetStreamId(1);
  op_def_a->AddOutputDesc(op_def_a->GetInputDesc(0));

  ge::OpDescPtr op_def_b = CreateOpWithWsSize("B", 1024, type, 2048);
  op_def_b->SetStreamId(1);
  ge::OpDescPtr op_def_c = CreateOpWithWsSize("C", 1024, type, 4096);
  ge::AttrUtils::SetBool(op_def_c, ATTR_NAME_CONTINUOUS_INPUT, true);
  ge::AttrUtils::SetBool(op_def_c, "need_gentask_atomic", true);
  std::vector<int32_t> input_indexes = {-1};
  (void)ge::AttrUtils::SetListInt(op_def_c, ATOMIC_ATTR_INPUT_INDEX, input_indexes);
  op_def_c->SetStreamId(3);
  ge::OpDescPtr op_def_d = CreateOpWithWsSize("D", 1024, type, 2048);
  op_def_d->SetStreamId(2);
  auto desc_temp_ptr = std::make_shared<ge::GeTensorDesc>();
  auto desc_temp = *desc_temp_ptr;
  TensorUtils::SetSize(desc_temp, 2048);
  op_def_c->AddInputDesc(desc_temp);
  // add node
  ge::NodePtr node_a = graph->AddNode(op_def_a);
  ge::NodePtr node_b = graph->AddNode(op_def_b);
  ge::NodePtr node_c = graph->AddNode(op_def_c);
  ge::NodePtr node_d = graph->AddNode(op_def_d);

  // add edge
  ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(1), node_c->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_b->GetOutDataAnchor(0), node_c->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(node_c->GetOutDataAnchor(0), node_d->GetInDataAnchor(0));
  graph->TopologicalSorting();

  MemAssistInfo mem_assist_info;
  mem_assist_info.compute_graph = graph;
  auto ret = GraphUtils::GetRefMapping(graph, mem_assist_info.symbol_to_anchors,
                                       mem_assist_info.anchor_to_symbol);
  EXPECT_EQ(ret, SUCCESS);
  std::vector<int64_t> ranges;
  BinaryBlockMemAssigner assigner(mem_assist_info);
  assigner.SetReuseStrategy(ReuseStrategy{false, true, false, true});
  assigner.GetMemoryRanges(ranges);
  assigner.AssignMemoryWithReuse(ranges);
  const auto blocks = assigner.GetMemoryBlocks();
  assigner.SetOpMemOffset(false);
  bool has_checked = false;
  for (const auto block : blocks) {
    for (const auto &node : block->node_type_index_list_) {
      if (node.mem_type_ != kOutput) {
        continue;
      }
      if ((node.node_->GetOpDesc()->GetName() == "A") && (node.index_ == 1)) {
        //此用例校验连续内存的复用关系
        //Node[A] stream[1] output[1]'s life time is max of [2][2][4294967295], node_io[C], stream_id[3]
        EXPECT_EQ(block->NodeTypeIndexList().back().out_stream_count_, 1U);
        EXPECT_EQ(block->GetLifeEnd(1), kMaxLifeTime);
        EXPECT_EQ(block->GetLifeEnd(3), 2);
        EXPECT_EQ(block->GetLifeEnd(2), kMaxLifeTime);
        int64_t end_stream = 0;
        EXPECT_EQ(block->GetLifeEnd(2, end_stream), 2);
        EXPECT_EQ(end_stream, 3);
        ASSERT_FALSE(block->real_size_list_.empty());
        // 校验a的输出size是C所有输入的size之和
        EXPECT_EQ(block->real_size_list_[0], 4096);
        has_checked = true;
      }

      // 此用例校验不同流间的复用关系，涉及函数GetNodeMaxLifeBySymbol
      // Node[C] stream[3] output[0]'s life time is max of [3][3][4294967295], node_io[D], stream_id[2].
      if ((node.node_->GetOpDesc()->GetName() == "C") && (node.index_ == 0)) {
        EXPECT_EQ(block->NodeTypeIndexList().back().out_stream_count_, 1U);
        EXPECT_EQ(block->GetLifeEnd(3), kMaxLifeTime);
        EXPECT_EQ(block->GetLifeEnd(2), 3);
        int64_t end_stream = 0;
        EXPECT_EQ(block->GetLifeEnd(2, end_stream), 3);
        EXPECT_EQ(end_stream, 2);
        has_checked = true;
      }
    }
  }
  EXPECT_TRUE(has_checked);
}

  /**
   *            A:1
   *          ___|___
   *         |       |
   *        B:2     C:2
   *         |       |
   *         |      D:2
   *         |_______|
   *             |
   *            E:1
   */
  // check stream 1 lifend
TEST_F(UtestBlockMemAssigner, ContinousOutputResueLifeCheck) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph_continuous_output_reuse");
  const string &type = "some";
  ge::OpDescPtr op_def_a = CreateOpWithWsSize("A", 1024, type, 2048);
  op_def_a->SetStreamId(1);
  op_def_a->AddOutputDesc(op_def_a->GetInputDesc(0));
  ge::AttrUtils::SetBool(op_def_a, ATTR_NAME_CONTINUOUS_OUTPUT, true);

  ge::OpDescPtr op_def_b = CreateOpWithWsSize("B", 1024, type, 2048);
  op_def_b->SetStreamId(2);
  ge::OpDescPtr op_def_c = CreateOpWithWsSize("C", 1024, type, 2048);
  op_def_c->SetStreamId(2);
  ge::OpDescPtr op_def_d = CreateOpWithWsSize("D", 1024, type, 2048);
  op_def_d->SetStreamId(2);
  ge::OpDescPtr op_def_e = CreateOpWithWsSize("E", 1024, type, 4096);
  op_def_e->SetStreamId(1);

  auto desc_temp_ptr = std::make_shared<ge::GeTensorDesc>();
  auto desc_temp = *desc_temp_ptr;
  TensorUtils::SetSize(desc_temp, 2048);
  op_def_e->AddInputDesc(desc_temp);
  // add node
  ge::NodePtr node_a = graph->AddNode(op_def_a);
  ge::NodePtr node_b = graph->AddNode(op_def_b);
  ge::NodePtr node_c = graph->AddNode(op_def_c);
  ge::NodePtr node_d = graph->AddNode(op_def_d);
  ge::NodePtr node_e = graph->AddNode(op_def_e);
  // add edge
  ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_b->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_b->GetOutDataAnchor(0), node_e->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(1), node_c->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_c->GetOutDataAnchor(0), node_d->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_d->GetOutDataAnchor(0), node_e->GetInDataAnchor(1));
  graph->TopologicalSorting();

  (void)AttrUtils::SetListInt(node_a->GetOpDesc(), TVM_ATTR_NAME_WORKSPACE_TYPE, {kRtMemoryUB});

  MemAssistInfo mem_assist_info;
  mem_assist_info.compute_graph = graph;
  auto ret = GraphUtils::GetRefMapping(graph, mem_assist_info.symbol_to_anchors,
                                       mem_assist_info.anchor_to_symbol);
  EXPECT_EQ(ret, SUCCESS);
  std::vector<int64_t> ranges;
  BinaryBlockMemAssigner assigner(mem_assist_info);
  assigner.SetReuseStrategy(ReuseStrategy{false, true, false, true});
  assigner.GetMemoryRanges(ranges);
  assigner.AssignMemoryWithReuse(ranges);
  const auto blocks = assigner.GetMemoryBlocks();
  assigner.SetOpMemOffset(false);
  bool has_checked = false;
  for (const auto block : blocks) {
    for (const auto &node : block->node_type_index_list_) {
      if (node.mem_type_ != kOutput) {
        continue;
      }
      if ((node.node_->GetOpDesc()->GetName() == "A") && (node.index_ == 0)) {
        //此用例校验连续内存的复用关系
        //Node[A] stream[1] output[0]'s life time is max of [1][1][4], node_io[B], stream_id[2].
        //[A] optype[some] output[0] life time begin[0] life time end[4]
        EXPECT_EQ(block->NodeTypeIndexList().back().out_stream_count_, 1U);
        EXPECT_EQ(block->GetLifeEnd(1), 4);
        has_checked = true;
      }

      if (node.node_->GetOpDesc()->GetName() == "B") {
        //此用例校验普通内存的复用关系
        //Node[B] stream[2] output[0]'s life time is max of [4][4][4294967295], node_io[E], stream_id[1].
        //name[B] optype[some] output[0] life time begin[1] life time end[4--4294967295]
        EXPECT_EQ(block->NodeTypeIndexList().back().out_stream_count_, 1U);
        EXPECT_EQ(block->GetLifeEnd(1), 4);
        EXPECT_EQ(block->GetLifeEnd(2), kMaxLifeTime);
        has_checked = true;
      }
    }
  }
  EXPECT_TRUE(has_checked);
}

  /**
   *            A:1
   *          ___|___
   *         |       |
   *        B:2     C:2
   *         |       |
   *         |      D:2
   *         |_______|
   *             |
   *            E:1
   */
  // check stream 1 lifend
TEST_F(UtestBlockMemAssigner, FixedAddrPriorCheck) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph_fixed_addr_reuse");
  const string &type = "some";
  ge::OpDescPtr op_def_a = CreateOpWithWsSize("A", 1024, type, 2048);
  op_def_a->SetStreamId(1);
  ge::AttrUtils::SetBool(op_def_a, ATTR_NAME_IS_FIXED_ADDR_PRIOR, true);

  ge::OpDescPtr op_def_b = CreateOpWithWsSize("B", 1024, type, 2048);
  op_def_b->SetStreamId(2);
  ge::OpDescPtr op_def_c = CreateOpWithWsSize("C", 1024, type, 2048);
  op_def_c->SetStreamId(2);
  ge::AttrUtils::SetBool(op_def_c, ATTR_NAME_IS_FIXED_ADDR_PRIOR, true);
  ge::OpDescPtr op_def_d = CreateOpWithWsSize("D", 1024, type, 2048);
  op_def_d->SetStreamId(2);
  ge::OpDescPtr op_def_e = CreateOpWithWsSize("E", 1024, type, 4096);
  op_def_e->SetStreamId(1);
  ge::AttrUtils::SetBool(op_def_e, ATTR_NAME_IS_FIXED_ADDR_PRIOR, true);

  auto desc_temp_ptr = std::make_shared<ge::GeTensorDesc>();
  auto desc_temp = *desc_temp_ptr;
  TensorUtils::SetSize(desc_temp, 2048);
  op_def_e->AddInputDesc(desc_temp);
  // add node
  ge::NodePtr node_a = graph->AddNode(op_def_a);
  ge::NodePtr node_b = graph->AddNode(op_def_b);
  ge::NodePtr node_c = graph->AddNode(op_def_c);
  ge::NodePtr node_d = graph->AddNode(op_def_d);
  ge::NodePtr node_e = graph->AddNode(op_def_e);
  // add edge
  ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_b->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_b->GetOutDataAnchor(0), node_e->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(1), node_c->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_c->GetOutDataAnchor(0), node_d->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_d->GetOutDataAnchor(0), node_e->GetInDataAnchor(1));
  graph->TopologicalSorting();

  MemAssistInfo mem_assist_info;
  mem_assist_info.compute_graph = graph;
  auto ret = GraphUtils::GetRefMapping(graph, mem_assist_info.symbol_to_anchors,
                                       mem_assist_info.anchor_to_symbol);
  EXPECT_EQ(ret, SUCCESS);
  std::vector<int64_t> ranges;
  BinaryBlockMemAssigner assigner(mem_assist_info);
  assigner.SetReuseStrategy(ReuseStrategy{false, true, false, true});
  assigner.GetMemoryRanges(ranges);
  assigner.AssignMemoryWithReuse(ranges);
  const auto blocks = assigner.GetMemoryBlocks();
  assigner.SetOpMemOffset(false);
}

TEST_F(UtestBlockMemAssigner, DT_VARIANT_NotPostReuse) {
  auto graph = BuildGraphWithDtVariant();

  MemAssistInfo mem_assist_info;
  mem_assist_info.compute_graph = graph;
  auto ret = GraphUtils::GetRefMapping(graph, mem_assist_info.symbol_to_anchors,
                                       mem_assist_info.anchor_to_symbol);
  EXPECT_EQ(ret, SUCCESS);

  std::vector<int64_t> ranges;
  BinaryBlockMemAssigner assigner(mem_assist_info);
  assigner.GetMemoryRanges(ranges);
  assigner.AssignMemoryWithReuse(ranges);
  const auto blocks = assigner.GetMemoryBlocks();
  bool has_checked = false;
  for (const auto block : blocks) {
    for (const auto &node : block->node_type_index_list_) {
      if (node.mem_type_ != kOutput) {
        continue;
      }
      std::cout << block->String() << std::endl;
      if (node.node_->GetName() == "tensor_list_length") {
        if (block->node_type_index_list_.size() == 1U) {
          EXPECT_TRUE(block->child_blocks_.empty());
          ASSERT_FALSE(block->child_block_);
          has_checked = true;
        }
      }
    }
  }
  EXPECT_TRUE(has_checked);
}

TEST_F(UtestBlockMemAssigner, Resize) {
  size_t block_size = 1024;
  int64_t stream_id = 0;
  bool is_reuse_mem = false;
  OpMemoryType memory_type = kOutput;
  MemoryBlock parent(reuse_strategy_, block_size, stream_id, is_reuse_mem, memory_type);
  MemoryBlock child(reuse_strategy_, block_size, stream_id, is_reuse_mem, memory_type);
  child.first_continuous_block_ = true;
  child.last_continuous_block_ = true;
  parent.child_blocks_.emplace_back(&child);
  auto builder = std::make_shared<block_mem_ut::GraphBuilder>("graph");
  auto n = builder->AddNode("node", DATA, 1, 1);
  NodeTypeIndex node_type_index{n.get(), memory_type, 0, false, 0, -1};
  parent.AddNodeTypeIndex(node_type_index, 1024, 1024, 0);
  child.AddNodeTypeIndex(node_type_index, 1024, 1024, 0);
  parent.Resize();
  size_t parent_size = parent.Size();
  size_t child_size = child.Size();
  EXPECT_EQ(parent_size - child_size, 0);
}

TEST_F(UtestBlockMemAssigner, MatchNoReuseType) {
  auto builder = std::make_shared<block_mem_ut::GraphBuilder>("graph");
  auto a = builder->AddNode("node", DROPOUTGENMASK, 1, 1);
  auto b = builder->AddNode("node_b", DROPOUTGENMASKV3, 1, 1);
  auto c = builder->AddNode("node_c", "DropOutGenMaskV5", 1, 1);
  ge::GraphUtils::AddEdge(a->GetOutDataAnchor(0), b->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(b->GetOutDataAnchor(0), c->GetInDataAnchor(0));
  NodeIndexIO node_index_io_a(a, 0, kOut);
  NodeIndexIO node_index_io_b(b, 0, kOut);
  NodeIndexIO node_index_io_c(c, 0, kOut);
  MemAssistInfo mem_assist_info;
  mem_assist_info.compute_graph = builder->GetGraph();
  EXPECT_EQ(GraphUtils::GetRefMapping(mem_assist_info.compute_graph, mem_assist_info.symbol_to_anchors,
                                      mem_assist_info.anchor_to_symbol), GRAPH_SUCCESS);
  BlockMemAssigner::PreparationForAssign(mem_assist_info);
  BinaryBlockMemAssigner mem_assigner(mem_assist_info);

  std::vector<int64_t> range_ceils;
  mem_assigner.GetMemoryRanges(range_ceils);
  EXPECT_EQ(mem_assigner.symbol_mem_reuse_info_[node_index_io_a.ToString()].pre_reuse_flag_, false);
  EXPECT_EQ(mem_assigner.symbol_mem_reuse_info_[node_index_io_a.ToString()].post_reuse_flag_, false);
  EXPECT_EQ(mem_assigner.symbol_mem_reuse_info_[node_index_io_b.ToString()].pre_reuse_flag_, false);
  EXPECT_EQ(mem_assigner.symbol_mem_reuse_info_[node_index_io_b.ToString()].post_reuse_flag_, false);
  EXPECT_EQ(mem_assigner.symbol_mem_reuse_info_[node_index_io_c.ToString()].pre_reuse_flag_, true);
  EXPECT_EQ(mem_assigner.symbol_mem_reuse_info_[node_index_io_c.ToString()].post_reuse_flag_, true);
  mem_assigner.symbol_mem_reuse_info_[node_index_io_a.ToString()].pre_reuse_flag_ = true;
  mem_assigner.AssignMemoryWithReuse(range_ceils);
}

TEST_F(UtestBlockMemAssigner, ContinousInputToDiffStreamsNotPostReuse) {
ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph_continuous_input_reuse");
  /**
   *          a:0  b:1
   *           |___|  \
   *             |     d
   *            c:2
   */
  // node c padding input continuous
  const string &type = "some";
  ge::OpDescPtr op_def_a = CreateOpWithWsSize("A", 1024, type, 2048);
  op_def_a->SetStreamId(1);
  op_def_a->AddOutputDesc(op_def_a->GetInputDesc(0));

  ge::OpDescPtr op_def_b = CreateOpWithWsSize("B", 1024, type, 2048);
  op_def_b->SetStreamId(1);
  ge::OpDescPtr op_def_c = CreateOpWithWsSize("C", 1024, type, 4096);
  ge::OpDescPtr op_def_d = CreateOpWithWsSize("D", 1024, type, 4096);
  op_def_d->SetStreamId(2);
  ge::AttrUtils::SetBool(op_def_c, ATTR_NAME_CONTINUOUS_INPUT, true);
  ge::AttrUtils::SetBool(op_def_c, "need_gentask_atomic", true);
  std::vector<int32_t> input_indexes = {-1};
  (void)ge::AttrUtils::SetListInt(op_def_c, ATOMIC_ATTR_INPUT_INDEX, input_indexes);
  op_def_c->SetStreamId(3);
  auto desc_temp_ptr = std::make_shared<ge::GeTensorDesc>();
  auto desc_temp = *desc_temp_ptr;
  TensorUtils::SetSize(desc_temp, 2048);
  op_def_c->AddInputDesc(desc_temp);
  // add node
  ge::NodePtr node_a = graph->AddNode(op_def_a);
  ge::NodePtr node_b = graph->AddNode(op_def_b);
  ge::NodePtr node_c = graph->AddNode(op_def_c);
  ge::NodePtr node_d = graph->AddNode(op_def_d);

  // add edge
  ge::GraphUtils::AddEdge(node_a->GetOutDataAnchor(1), node_c->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_b->GetOutDataAnchor(0), node_c->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(node_b->GetOutDataAnchor(0), node_d->GetInDataAnchor(0));
  graph->TopologicalSorting();

  MemAssistInfo mem_assist_info;
  mem_assist_info.compute_graph = graph;
  auto ret = GraphUtils::GetRefMapping(graph, mem_assist_info.symbol_to_anchors,
                                       mem_assist_info.anchor_to_symbol);
  EXPECT_EQ(ret, SUCCESS);
  std::vector<int64_t> ranges;
  BinaryBlockMemAssigner assigner(mem_assist_info);
  assigner.SetReuseStrategy(ReuseStrategy{false, true, false, true});
  assigner.GetMemoryRanges(ranges);
  assigner.AssignMemoryWithReuse(ranges);
  assigner.SetOpMemOffset(false);
  const auto blocks = assigner.GetMemoryBlocks();
  bool has_checked = false;
  for (const auto block : blocks) {
    for (const auto &node : block->node_type_index_list_) {
      if (node.mem_type_ != kOutput) {
        continue;
      }
      if ((node.node_->GetOpDesc()->GetName() == "A") && (node.index_ == 1)) {
        EXPECT_EQ(block->NodeTypeIndexList().back().out_stream_count_, 2U);
        EXPECT_EQ(block->GetLifeEnd(1), kMaxLifeTime);
        EXPECT_EQ(block->GetLifeEnd(3), kMaxLifeTime);
        has_checked = true;
      }
    }
  }
  EXPECT_TRUE(has_checked);
}

TEST_F(UtestBlockMemAssigner, ContinuousOutputCanNotZeroCpy) {
  auto builder = std::make_shared<block_mem_ut::GraphBuilder>("graph");
  auto node = builder->AddNode("node", DATA, 1, 1);
  auto node2 = builder->AddNode("streamswitch", STREAMSWITCH, 1, 1);
  GraphUtils::AddEdge(node->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
  ComputeGraphPtr compute_graph = builder->GetGraph();
  MemAssistInfo mem_assist_info;
  mem_assist_info.compute_graph = compute_graph;
  BinaryBlockMemAssigner assigner(mem_assist_info);
  const string &type = "some";
  ge::OpDescPtr node_op_desc = CreateOpWithWsSize("A", 1024, type, 2048);
  node_op_desc->AddOutputDesc(node_op_desc->GetOutputDesc(0));
  ge::AttrUtils::SetBool(node_op_desc, ATTR_NAME_CONTINUOUS_OUTPUT, true);
  ge::NodePtr node_a = compute_graph->AddNode(node_op_desc);
  GraphUtils::AddEdge(node2->GetOutDataAnchor(0), node_a->GetInDataAnchor(0));

  ge::OpDescPtr b_node_op_desc = CreateOpWithWsSize("B", 1024, type, 2048);
  ge::NodePtr node_b = compute_graph->AddNode(b_node_op_desc);
  ge::OpDescPtr c_node_op_desc = CreateOpWithWsSize("C", 1024, type, 2048);
  ge::NodePtr node_c = compute_graph->AddNode(c_node_op_desc);
  GraphUtils::AddEdge(node_a->GetOutDataAnchor(0), node_b->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_a->GetOutDataAnchor(1), node_c->GetInDataAnchor(0));

  size_t block_size = 1024;
  int64_t stream_id = 0;
  bool is_reuse_mem = false;
  OpMemoryType memory_type = kOutput;
  MemoryBlock parent(reuse_strategy_, block_size, stream_id, is_reuse_mem, memory_type);
  MemoryBlock child(reuse_strategy_, block_size, stream_id, is_reuse_mem, memory_type);
  child.is_reuse_zero_copy_ = true;
  child.is_zero_copy_ = true;
  assigner.MarkReuseZeroCopyBlockFlag(node_a, &child, 0);
  EXPECT_EQ(child.is_reuse_zero_copy_, false);
  EXPECT_EQ(child.is_zero_copy_, true);

  assigner.MarkReuseZeroCopyBlockFlag(node, &child, 0);
  EXPECT_EQ(child.is_reuse_zero_copy_, false);
  EXPECT_EQ(child.is_zero_copy_, false);
}

TEST_F(UtestBlockMemAssigner, DataOutDiffStream) {
  auto builder = std::make_shared<block_mem_ut::GraphBuilder>("graph");
  auto a = builder->AddNode("Data", DATA, 0, 1);
  a->GetOpDesc()->SetStreamId(-1);
  auto b = builder->AddNode("node_b", DROPOUTGENMASKV3, 1, 1);
  b->GetOpDesc()->SetStreamId(0);
  auto c = builder->AddNode("node_c", DROPOUTGENMASKV3, 1, 1);
  c->GetOpDesc()->SetStreamId(2);
  ge::GraphUtils::AddEdge(a->GetOutDataAnchor(0), b->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(a->GetOutDataAnchor(0), c->GetInDataAnchor(0));
  MemAssistInfo mem_assist_info;
  mem_assist_info.compute_graph = builder->GetGraph();
  EXPECT_EQ(GraphUtils::GetRefMapping(mem_assist_info.compute_graph, mem_assist_info.symbol_to_anchors,
                                      mem_assist_info.anchor_to_symbol), GRAPH_SUCCESS);
  BlockMemAssigner::PreparationForAssign(mem_assist_info);
  BinaryBlockMemAssigner mem_assigner(mem_assist_info);
  std::vector<int64_t> range_ceils;
  mem_assigner.GetMemoryRanges(range_ceils);
  mem_assigner.AssignMemoryWithReuse(range_ceils);
  int64_t sub_stream_id = ge::kInvalidStreamId;
  ge::AttrUtils::GetInt(a->GetOpDesc(), ge::ATTR_NAME_SUB_STREAM_ID, sub_stream_id);
  EXPECT_TRUE(sub_stream_id == ge::kInvalidStreamId);
}

TEST_F(UtestBlockMemAssigner, DataOutSameStream) {
  auto builder = std::make_shared<block_mem_ut::GraphBuilder>("graph");
  auto a = builder->AddNode("Data", DATA, 0, 1);
  a->GetOpDesc()->SetStreamId(-1);
  auto b = builder->AddNode("node_b", DROPOUTGENMASKV3, 1, 1);
  b->GetOpDesc()->SetStreamId(0);
  auto c = builder->AddNode("node_c", DROPOUTGENMASKV3, 1, 1);
  c->GetOpDesc()->SetStreamId(0);
  ge::GraphUtils::AddEdge(a->GetOutDataAnchor(0), b->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(a->GetOutDataAnchor(0), c->GetInDataAnchor(0));
  MemAssistInfo mem_assist_info;
  mem_assist_info.compute_graph = builder->GetGraph();
  EXPECT_EQ(GraphUtils::GetRefMapping(mem_assist_info.compute_graph, mem_assist_info.symbol_to_anchors,
                                      mem_assist_info.anchor_to_symbol), GRAPH_SUCCESS);
  BlockMemAssigner::PreparationForAssign(mem_assist_info);
  BinaryBlockMemAssigner mem_assigner(mem_assist_info);
  std::vector<int64_t> range_ceils;
  mem_assigner.GetMemoryRanges(range_ceils);
  mem_assigner.AssignMemoryWithReuse(range_ceils);
  int64_t sub_stream_id = ge::kInvalidStreamId;
  ge::AttrUtils::GetInt(a->GetOpDesc(), ge::ATTR_NAME_SUB_STREAM_ID, sub_stream_id);
  EXPECT_EQ(sub_stream_id, 0);
}

TEST_F(UtestBlockMemAssigner, BlockContinueFlagCheck) {
  auto builder = std::make_shared<block_mem_ut::GraphBuilder>("graph");
  auto a = builder->AddNode("Data", DATA, 0, 1);
  auto b = builder->AddNode("node_b", DROPOUTGENMASKV3, 1, 1);
  auto c = builder->AddNode("node_c", DROPOUTGENMASKV3, 1, 1);
  ge::GraphUtils::AddEdge(a->GetOutDataAnchor(0), b->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(a->GetOutDataAnchor(0), c->GetInDataAnchor(0));

  size_t block_size = 1024;
  int64_t stream_id = 0;
  bool is_reuse_mem = false;
  OpMemoryType memory_type = kOutput;
  MemoryBlock parent(reuse_strategy_, block_size, stream_id, is_reuse_mem, memory_type);
  NodeTypeIndex node_type_index{b.get(), memory_type, 0, false, 0, stream_id};
  node_type_index.SetFirstContinuousNode(true);
  node_type_index.SetLastContinuousNode(true);
  node_type_index.SetContinuousNode(true);
  parent.AddNodeTypeIndex(node_type_index, 1024, 1024, stream_id);
  EXPECT_TRUE(parent.GetFirstContinuousFlag());
  EXPECT_TRUE(parent.GetLastContinuousFlag());
  EXPECT_TRUE(parent.GetContinuousFlag());
  parent.UpdateContinuousFlag();
  auto iter = parent.NodeTypeIndexList().cbegin();
  parent.DelNode(iter);
  parent.UpdateContinuousFlag();
  EXPECT_FALSE(parent.GetFirstContinuousFlag());
  EXPECT_FALSE(parent.GetLastContinuousFlag());
  EXPECT_FALSE(parent.GetContinuousFlag());
}

TEST_F(UtestBlockMemAssigner, MemoryBlockAddField) {
  // 如果MemoryBlock里添加了成员变量，需要在MemoryBlock::Clone()里添加处理
  EXPECT_EQ(sizeof(ge::MemoryBlock), 384);
}

TEST_F(UtestBlockMemAssigner, AssignOutputMemoryWithMemoryScopeReuse) {
  auto builder = std::make_shared<block_mem_ut::GraphBuilder>("graph");
  auto node = builder->AddNode("node", DATA, 1, 1);
  ComputeGraphPtr compute_graph = builder->GetGraph();
  MemAssistInfo mem_assist_info;
  mem_assist_info.compute_graph = compute_graph;
  auto p1 = std::make_shared<FakBlockMemAssigner>(mem_assist_info);
  std::vector<int64_t> ranges;
  p1->op_reuse_env_valid_ = true;
  ge::AttrUtils::SetInt(node->GetOpDesc()->MutableOutputDesc(0), ATTR_NAME_TENSOR_MEMORY_SCOPE, 2);
  ge::AttrUtils::SetBool(node->GetOpDesc(), ATTR_NAME_CONTINUOUS_OUTPUT, true);
  EXPECT_EQ(p1->AssignOutputMemoryWithReuse(node, ranges), SUCCESS);
}

TEST_F(UtestBlockMemAssigner, clone_block) {
  ReuseStrategy strategy;
  strategy.memory_priority_mode_ = true;
  MemoryBlock block(strategy, 512, 0);
  auto clone_block = block.Clone();
  GELOGI("clone ReuseStrategy %p", &clone_block->GetReuseStrategy());
  EXPECT_EQ(clone_block->GetReuseStrategy().memory_priority_mode_, true);
  delete clone_block;
}

//     data
//      |
//      a (stream 0)
//      |                +---------------+
//  partitioncall0-------| data          |
//      |                |  |            |
//      c (stream 0)     |  b (stream 1) |
//      |                |  |            |
//      d (stream 0)     | netoutput1    |
//      |                +---------------+
//   netoutput
//
// 子图data连接的节点stream和子图输入不一致，校验data的复用
TEST_F(UtestBlockMemAssigner, GetRealStreamIdForParentNode_Success) {
  ge::ComputeGraphPtr graph = BuildSubGraphWithDiffStream();
  MemAssistInfo mem_assist_info;
  mem_assist_info.compute_graph = graph;
  auto ret = GraphUtils::GetRefMapping(graph, mem_assist_info.symbol_to_anchors,
                                       mem_assist_info.anchor_to_symbol);
  EXPECT_EQ(ret, SUCCESS);
  BlockMemAssigner::PreparationForAssign(mem_assist_info);
  std::vector<int64_t> ranges;
  BinaryBlockMemAssigner assigner(mem_assist_info);
  assigner.SetReuseStrategy(ReuseStrategy{false, true, false, true});
  assigner.GetMemoryRanges(ranges);
  assigner.AssignMemoryWithReuse(ranges);
  const auto blocks = assigner.GetMemoryBlocks();
  bool has_checked = false;
  assigner.SetOpMemOffset(false);
  for (const auto block : blocks) {
    for (const auto &node : block->node_type_index_list_) {
      if (node.mem_type_ != kOutput) {
        continue;
      }
      if (node.node_->GetOpDesc()->GetName() == "partitioncall0") {
        EXPECT_EQ(block->NodeTypeIndexList().back().stream_id_, 1U);
        has_checked = true;
      }
    }
  }
  EXPECT_TRUE(has_checked);
}

//     data
//      |
//      a
//      |                +----------+
//  partitioncall0-------|   data   |
//      |                |    |     |
//      e                |    b     |
//      |                |    |     |
//      f                |    c     |
//      |                |    |     |
//   netoutput           |    d     |
//                       |    |     |
//                       |netoutput1|
//                       +----------+
// 动态shape静态子图输出内存复用
TEST_F(UtestBlockMemAssigner, KnownSubGraphOutputReuse) {
  ge::ComputeGraphPtr root_graph = BuildKnownSubGraph();
  auto graph = root_graph->GetAllSubgraphs().front();
  MemAssistInfo mem_assist_info;
  mem_assist_info.compute_graph = graph;
  auto ret = GraphUtils::GetRefMapping(graph, mem_assist_info.symbol_to_anchors,
                                       mem_assist_info.anchor_to_symbol);
  EXPECT_EQ(ret, SUCCESS);
  BlockMemAssigner::PreparationForAssign(mem_assist_info);
  std::vector<int64_t> ranges;
  BinaryBlockMemAssigner assigner(mem_assist_info);
  assigner.SetReuseStrategy(ReuseStrategy{false, true, false, true});
  assigner.GetMemoryRanges(ranges);
  assigner.AssignMemoryWithReuse(ranges);
  const auto blocks = assigner.GetMemoryBlocks();
  bool has_checked = false;
  assigner.SetOpMemOffset(false);
  for (const auto block : blocks) {
    for (const auto &node : block->node_type_index_list_) {
      if (node.mem_type_ != kOutput) {
        continue;
      }
      if (node.node_->GetOpDesc()->GetName() == "d") {
        EXPECT_EQ(block->child_blocks_.size(), 1U);
        has_checked = true;
      }
    }
  }
  EXPECT_TRUE(has_checked);
}

/*
 *  a       b          c
 *  |       |          |
 *  |   hcombroadcast  |
 *   \      |         /
 *     hcombroadcast2
 *          |
 *          d
 */
TEST_F(UtestBlockMemAssigner, RefNodeConnectContinuousNode) {
  auto graph = BuildRefNodeConnectContinuousInputNode();
  MemAssistInfo mem_assist_info;
  mem_assist_info.compute_graph = graph;
  auto ret = GraphUtils::GetRefMapping(graph, mem_assist_info.symbol_to_anchors,
                                       mem_assist_info.anchor_to_symbol);
  EXPECT_EQ(ret, SUCCESS);
  BlockMemAssigner::PreparationForAssign(mem_assist_info);
  std::vector<int64_t> ranges;
  BinaryBlockMemAssigner assigner(mem_assist_info);
  assigner.SetReuseStrategy(ReuseStrategy{false, false, false, true});
  assigner.GetMemoryRanges(ranges);
  assigner.AssignMemoryWithReuse(ranges);
  const auto blocks = assigner.GetMemoryBlocks();
  bool has_checked = false;
  assigner.SetOpMemOffset(false);
  for (const auto block : blocks) {
    for (const auto &node : block->node_type_index_list_) {
      if (node.mem_type_ != kOutput) {
        continue;
      }
      if (node.node_->GetOpDesc()->GetName() == "b") {
        EXPECT_TRUE(block->continuous_block_);
        has_checked = true;
      }
    }
  }
  EXPECT_TRUE(has_checked);
}

/*
 *     a            b
 *     |            |
 *  PhonyConcat     |
 *           \      |
 *         hcombroadcast
 *              ||
 *               c
 */
TEST_F(UtestBlockMemAssigner, SingleNoPaddingContinuousConnectContinuousNode) {
  auto graph = BuildSingleNoPaddingContinuousConnectContinuousInputNode();
  MemAssistInfo mem_assist_info;
  mem_assist_info.compute_graph = graph;
  auto ret = GraphUtils::GetRefMapping(graph, mem_assist_info.symbol_to_anchors,
                                       mem_assist_info.anchor_to_symbol);
  EXPECT_EQ(ret, SUCCESS);
  BlockMemAssigner::PreparationForAssign(mem_assist_info);
  std::vector<int64_t> ranges;
  BinaryBlockMemAssigner assigner(mem_assist_info);
  assigner.SetReuseStrategy(ReuseStrategy{false, false, false, true});
  assigner.GetMemoryRanges(ranges);
  assigner.AssignMemoryWithReuse(ranges);
  const auto blocks = assigner.GetMemoryBlocks();
  bool has_checked = false;
  assigner.SetOpMemOffset(false);
  for (const auto block : blocks) {
    for (const auto &node : block->node_type_index_list_) {
      if (node.mem_type_ != kOutput) {
        continue;
      }
      if (node.node_->GetOpDesc()->GetName() == "a" || node.node_->GetOpDesc()->GetName() == "b") {
        EXPECT_TRUE(block->continuous_block_);
        has_checked = true;
      }
    }
  }
  EXPECT_TRUE(has_checked);
}

/*
 *         data
 *          |
 *          a(1)---ctrl
 *          |       |
 *          b(2)    |
 *          |      c(3) (stream 1)
 *          d(4)  /
 *           \   /
 *         PhonyConcat(5)
 *            |
 *            g(6)
 *  topo id: b is smaller than c
 *  size: a/PhonyConcat 8M, b/d 4M, others 2k
 *  预期a和d不复用，d所在block的same_stream_标记为false，以前就修改了，可以看  UtestMemoryAssignerTest, DiffMergeInputNodesNotReuse
 */
TEST_F(UtestBlockMemAssigner, NoPaddingContinuousInput_MultiInputDiffStream) {
  auto graph = BuildNoPaddingContinuousMultiInputDiffStream();
  MemAssistInfo mem_assist_info;
  mem_assist_info.compute_graph = graph;
  auto ret = GraphUtils::GetRefMapping(graph, mem_assist_info.symbol_to_anchors,
                                       mem_assist_info.anchor_to_symbol);
  EXPECT_EQ(ret, SUCCESS);
  BlockMemAssigner::PreparationForAssign(mem_assist_info);
  std::vector<int64_t> ranges;
  BinaryBlockMemAssigner assigner(mem_assist_info);
  assigner.SetReuseStrategy(ReuseStrategy{false, false, false, true});
  assigner.GetMemoryRanges(ranges);
  assigner.AssignMemoryWithReuse(ranges);
  const auto blocks = assigner.GetMemoryBlocks();
  assigner.SetOpMemOffset(false);

  auto a = graph->FindNode("a");
  auto d = graph->FindNode("d");
  ASSERT_NE(a, nullptr);
  ASSERT_NE(d, nullptr);
  auto a_out_offsets = a->GetOpDesc()->GetOutputOffset();
  auto d_out_offsets = d->GetOpDesc()->GetOutputOffset();
  ASSERT_EQ(a_out_offsets.size(), 1);
  ASSERT_EQ(d_out_offsets.size(), 1);

  auto reuse = CheckIntersection({a_out_offsets.at(0), a_out_offsets.at(0) + 8 * 1024 * 1024 - 1}, {d_out_offsets.at(0), d_out_offsets.at(0) + 4 * 1024 * 1024});
  std::cout << "z_out_offset: " << a_out_offsets.at(0) << " , e_out_offset: " << d_out_offsets.at(0) << std::endl;
  EXPECT_FALSE(reuse);
}

/*
 *(stream 1)a(0)   b(1)
 *          |     / \
 *          |  ctrl  |
 *          | /      /
 *          c(2)    /
 *          |      /
 *          d(3)  /
 *           \  /
 *         PhonyConcat(4)
 *            |
 *            g(5)
 *  topo id: b is smaller than c
 *  size: a/PhonyConcat 8M, b/d 4M, others 2k
 *  看护多流+NoPadding连续输入叠加场景内存复用。该场景未发现问题，原本怀疑d的生命周期起点是3，比c（2）要大，会导致d和a的复用，
 *  但是经过实测发现a和d没有复用，且d的生命周期起点已经标记为1（b）了。
 */
TEST_F(UtestBlockMemAssigner, NoPaddingContinuousInputAndMultiStream_FirstInputNotSmallestTopoId_SameSizeReuse) {
  auto graph = BuildNoPaddingContinuousAndMultiStreamGraph();
  MemAssistInfo mem_assist_info;
  mem_assist_info.compute_graph = graph;
  auto ret = GraphUtils::GetRefMapping(graph, mem_assist_info.symbol_to_anchors,
                                       mem_assist_info.anchor_to_symbol);
  EXPECT_EQ(ret, SUCCESS);
  BlockMemAssigner::PreparationForAssign(mem_assist_info);
  std::vector<int64_t> ranges;
  BinaryBlockMemAssigner assigner(mem_assist_info);
  assigner.SetReuseStrategy(ReuseStrategy{false, false, false, true});
  assigner.GetMemoryRanges(ranges);
  assigner.AssignMemoryWithReuse(ranges);
  const auto blocks = assigner.GetMemoryBlocks();
  assigner.SetOpMemOffset(false);

  auto a = graph->FindNode("a");
  auto d = graph->FindNode("d");
  ASSERT_NE(a, nullptr);
  ASSERT_NE(d, nullptr);
  auto a_out_offsets = a->GetOpDesc()->GetOutputOffset();
  auto d_out_offsets = d->GetOpDesc()->GetOutputOffset();
  ASSERT_EQ(a_out_offsets.size(), 1);
  ASSERT_EQ(d_out_offsets.size(), 1);

  auto reuse = CheckIntersection({a_out_offsets.at(0), a_out_offsets.at(0) + 8 * 1024 * 1024}, {d_out_offsets.at(0), d_out_offsets.at(0) + 4 * 1024 * 1024});
  std::cout << "z_out_offset: " << a_out_offsets.at(0) << " , e_out_offset: " << d_out_offsets.at(0) << std::endl;
  EXPECT_FALSE(reuse);
}

/*
 *  data1  data2
 *     \    /
 *       if
 *       |
 *       op
 *       |
 *    netoutput0
 *
 * then_subgraph                             else_subgraph
 * +-------------------------------------+   +------------------------------------+
 * |    data3                            |   |                                    |
 * |     |                               |   |                                    |
 * |    cast                             |   |    data5                           |
 * |     |                               |   |     |                              |
 * | partitioned_call1   +------------+  |   | partitioned_call2  +------------+  |
 * |     |               |   data4    |  |   |     |              |    data6   |  |
 * |  netoutput1         |     |      |  |   |  netoutput3        |      |     |  |
 * |                     | netoutput2 |  |   |                    | netoutput4 |  |
 * |                     +------------+  |   |                    +------------+  |
 * +-------------------------------------+   +------------------------------------+
 *
 * partitioned_call1(stream1), partitioned_call2(stream2), data4(stream4), data6(stream6)
 * if output 0 should use data4 and data6 stream id, but they are different, so set if output 0 no reuse
 */
TEST_F(UtestBlockMemAssigner, GetRealStreamIdForParentNode_NestingAndDiffStream) {
  ge::ComputeGraphPtr graph = BuildNestingWrapperWithSubgraphNodeDiffStream();
  MemAssistInfo mem_assist_info;
  mem_assist_info.compute_graph = graph;
  auto ret = GraphUtils::GetRefMapping(graph, mem_assist_info.symbol_to_anchors,
                                       mem_assist_info.anchor_to_symbol);
  EXPECT_EQ(ret, SUCCESS);
  BlockMemAssigner::PreparationForAssign(mem_assist_info);

  std::vector<int64_t> ranges;
  BinaryBlockMemAssigner assigner(mem_assist_info);
  assigner.SetReuseStrategy(ReuseStrategy{false, true, false, true});
  assigner.GetMemoryRanges(ranges);
  assigner.AssignMemoryWithReuse(ranges);
  const auto blocks = assigner.GetMemoryBlocks();
  bool has_checked = false;
  assigner.SetOpMemOffset(false);
  for (const auto block : blocks) {
    for (const auto &node : block->node_type_index_list_) {
      if (node.mem_type_ != kOutput) {
        continue;
      }
      if (node.node_->GetOpDesc()->GetName() == "if") {
        EXPECT_FALSE(block->reuse_mem_);
        has_checked = true;
      }
    }
  }
  EXPECT_TRUE(has_checked);
}
/*
 *  stream0 stream1 stream2
 *     a(0)----+
 *     |       |
 *     b(1)-+  |
 *     |    +->c(2)----+
 *     |       |       |
 *     +------>d(3)-+  |
 *             |    +->e(4)
 *             |       |
 *             +------>f(5)
 */
TEST_F(UtestBlockMemAssigner, GetDiffStreamEdgeLife_Success_CheckAllEdge) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("a", RELU)->NODE("b", RELU));
    CHAIN(NODE("a", RELU)->NODE("c", RELU));
    CHAIN(NODE("b", RELU)->NODE("c", RELU));
    CHAIN(NODE("b", RELU)->NODE("d", RELU));
    CHAIN(NODE("c", RELU)->NODE("d", RELU));
    CHAIN(NODE("c", RELU)->NODE("e", RELU));
    CHAIN(NODE("d", RELU)->NODE("e", RELU));
    CHAIN(NODE("d", RELU)->NODE("f", RELU));
    CHAIN(NODE("e", RELU)->NODE("f", RELU));
  };
  auto graph = ToComputeGraph(g1);
  auto a = graph->FindNode("a");
  auto b = graph->FindNode("b");
  auto c = graph->FindNode("c");
  auto d = graph->FindNode("d");
  auto e = graph->FindNode("e");
  auto f = graph->FindNode("f");
  a->GetOpDesc()->SetStreamId(0);
  b->GetOpDesc()->SetStreamId(0);
  c->GetOpDesc()->SetStreamId(1);
  d->GetOpDesc()->SetStreamId(1);
  e->GetOpDesc()->SetStreamId(2);
  f->GetOpDesc()->SetStreamId(2);

  MemAssistInfo mem_assist_info;
  mem_assist_info.compute_graph = graph;
  auto ret = GraphUtils::GetRefMapping(graph, mem_assist_info.symbol_to_anchors,
                                       mem_assist_info.anchor_to_symbol);
  EXPECT_EQ(ret, SUCCESS);
  BlockMemAssigner::PreparationForAssign(mem_assist_info);

  std::vector<int64_t> ranges;
  BinaryBlockMemAssigner assigner(mem_assist_info);
  assigner.SetReuseStrategy(ReuseStrategy{false, true, false, true});
  
  ASSERT_EQ(assigner.GetMemoryRanges(ranges), SUCCESS);
  ASSERT_EQ(assigner.in_stream_edges_[1][0].size(), 1U);
  ASSERT_EQ(assigner.in_stream_edges_[2][0].size(), 1U);
  ASSERT_EQ(assigner.in_stream_edges_[2][1].size(), 1U);
  ASSERT_EQ(assigner.out_stream_edges_[0][1].size(), 1U);
  ASSERT_EQ(assigner.out_stream_edges_[0][2].size(), 1U);
  ASSERT_EQ(assigner.out_stream_edges_[1][2].size(), 1U);

  EXPECT_EQ(assigner.in_stream_edges_[1][0].begin()->node_id, 2);
  EXPECT_EQ(assigner.in_stream_edges_[1][0].begin()->peer_node_id, 1);
  EXPECT_EQ(assigner.in_stream_edges_[2][0].begin()->node_id, 4);
  EXPECT_EQ(assigner.in_stream_edges_[2][0].begin()->peer_node_id, 1);
  EXPECT_EQ(assigner.in_stream_edges_[2][1].begin()->node_id, 4);
  EXPECT_EQ(assigner.in_stream_edges_[2][1].begin()->peer_node_id, 3);

  EXPECT_EQ(assigner.out_stream_edges_[0][1].begin()->node_id, 1);
  EXPECT_EQ(assigner.out_stream_edges_[0][1].begin()->peer_node_id, 2);
  EXPECT_EQ(assigner.out_stream_edges_[0][2].begin()->node_id, 1);
  EXPECT_EQ(assigner.out_stream_edges_[0][2].begin()->peer_node_id, 4);
  EXPECT_EQ(assigner.out_stream_edges_[1][2].begin()->node_id, 3);
  EXPECT_EQ(assigner.out_stream_edges_[1][2].begin()->peer_node_id, 4);
}

/*
 *  stream0  stream1  stream2
 *  +--a(0)
 *  |
 *  |  b(1)-----+
 *  |           |
 *  |           c(2)-----+
 *  |                    |
 *  |                    d(3)
 *  |
 *  +------------------->e(4)
 */
TEST_F(UtestBlockMemAssigner, GetDiffStreamEdgeLife_Success_EraseIntersectedEdge) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("a", RELU)->NODE("e", RELU));
    CHAIN(NODE("b", RELU)->NODE("c", RELU));
    CHAIN(NODE("c", RELU)->NODE("d", RELU));
  };
  auto graph = ToComputeGraph(g1);
  auto a = graph->FindNode("a");
  auto b = graph->FindNode("b");
  auto c = graph->FindNode("c");
  auto d = graph->FindNode("d");
  auto e = graph->FindNode("e");
  a->GetOpDesc()->SetStreamId(0);
  b->GetOpDesc()->SetStreamId(0);
  c->GetOpDesc()->SetStreamId(1);
  d->GetOpDesc()->SetStreamId(2);
  e->GetOpDesc()->SetStreamId(2);

  MemConflictShareGraph::TopologicalSortingMock(graph, {"a", "b", "c", "d", "e"});
  MemAssistInfo mem_assist_info;
  mem_assist_info.compute_graph = graph;
  auto ret = GraphUtils::GetRefMapping(graph, mem_assist_info.symbol_to_anchors,
                                       mem_assist_info.anchor_to_symbol);
  EXPECT_EQ(ret, SUCCESS);
  BlockMemAssigner::PreparationForAssign(mem_assist_info);

  std::vector<int64_t> ranges;
  BinaryBlockMemAssigner assigner(mem_assist_info);
  assigner.SetReuseStrategy(ReuseStrategy{false, true, false, true});

  ASSERT_EQ(assigner.GetMemoryRanges(ranges), SUCCESS);
  auto edge_set = assigner.in_stream_edges_[2][0];
  ASSERT_EQ(edge_set.size(), 1U);
  ASSERT_EQ(edge_set.begin()->node_id, 3);
  ASSERT_EQ(edge_set.begin()->peer_node_id, 1);
}

/*
 *  stream0      stream2
 *     a(2052)--+---+
 *              |   |
 *              |   b(2060)
 *              |
 *     c(2078)  +---+
 *     |            |
 *     +----------> d(2091)
 *
 * 问题场景：
 * OutEdge                      InEdge
 * only insert {2052->2060}     only insert {2060<-2052}
 * not erase, not insert        only insert {2091<-2052}
 * only insert {2078->2091}     erase and insert, erase {2091<-2052}, and insert {2091<-2078}, 误删 OutEdge 2052->2060
 */
TEST_F(UtestBlockMemAssigner, GetDiffStreamEdgeLife_Success_CheckOutEdge) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("a", RELU)->NODE("b", RELU));
    CHAIN(NODE("a", RELU)->NODE("d", RELU));
    CHAIN(NODE("c", RELU)->NODE("d", RELU));
  };
  auto graph = ToComputeGraph(g1);
  auto a = graph->FindNode("a");
  auto b = graph->FindNode("b");
  auto c = graph->FindNode("c");
  auto d = graph->FindNode("d");
  a->GetOpDesc()->SetId(2052);
  b->GetOpDesc()->SetId(2060);
  c->GetOpDesc()->SetId(2078);
  d->GetOpDesc()->SetId(2091);
  a->GetOpDesc()->SetStreamId(0);
  b->GetOpDesc()->SetStreamId(1);
  c->GetOpDesc()->SetStreamId(0);
  d->GetOpDesc()->SetStreamId(1);

  MemAssistInfo mem_assist_info;
  mem_assist_info.compute_graph = graph;
  auto ret = GraphUtils::GetRefMapping(graph, mem_assist_info.symbol_to_anchors,
                                       mem_assist_info.anchor_to_symbol);
  EXPECT_EQ(ret, SUCCESS);
  BlockMemAssigner::PreparationForAssign(mem_assist_info);

  std::vector<int64_t> ranges;
  BinaryBlockMemAssigner assigner(mem_assist_info);
  assigner.SetReuseStrategy(ReuseStrategy{false, true, false, true});

  ASSERT_EQ(assigner.GetMemoryRanges(ranges), SUCCESS);
  ASSERT_EQ(assigner.in_stream_edges_[1][0].size(), 2U);
  ASSERT_EQ(assigner.out_stream_edges_[0][1].size(), 2U);
}

/*
               a
              /   \
      ref_node1<--ref_node2
             \     /  \
               pc       b
                |
                c
                |
             netoutput
 用例关注点：
 1 a的同一个输出给到了两个ref_node，并且这两个ref_node的输出又给到了同一个pc
 2 ref_node2上带有input offset，表示从a的某个偏移上读取数据
*/
TEST_F(UtestBlockMemAssigner, GetNoNeedAssignMemoryFlag_AsFirstAndSecondInput) {
  auto graph = block_mem_ut::BuildPhonyConcatWithSameInputThrougRefNode();
  MemAssistInfo mem_assist_info;
  mem_assist_info.compute_graph = graph;
  auto ret = GraphUtils::GetRefMapping(graph, mem_assist_info.symbol_to_anchors,
                                       mem_assist_info.anchor_to_symbol);
  EXPECT_EQ(ret, SUCCESS);
  BlockMemAssigner::PreparationForAssign(mem_assist_info);

  std::vector<int64_t> ranges;
  BinaryBlockMemAssigner assigner(mem_assist_info);
  assigner.SetReuseStrategy(ReuseStrategy{false, true, false, true});
  assigner.GetMemoryRanges(ranges);

  bool no_need_assign_memory_flag = false;

  auto a = graph->FindNode("a");
  ASSERT_NE(a, nullptr);
  ASSERT_EQ(assigner.GetNoNeedAssignMemoryFlag(a, 0, no_need_assign_memory_flag), SUCCESS);
  EXPECT_FALSE(no_need_assign_memory_flag);
}

/*
 *(stream 1)a(0)   b(1)
 *          |     / \
 *          |  ctrl  |
 *          | /      /
 *          c(2)    /
 *          |      /
 *          d(3)  /
 *           \  /
 *         PhonyConcat(4)
 *            |
 *            g(5)
 *  topo id: b is smaller than c
 *  size: a/PhonyConcat 8M, b/d 4M, others 2k
 */
TEST_F(UtestBlockMemAssigner, GetNoNeedAssignMemoryFlag_PhonyConcat) {
  auto graph = BuildNoPaddingContinuousAndMultiStreamGraph();
  MemAssistInfo mem_assist_info;
  mem_assist_info.compute_graph = graph;
  auto ret = GraphUtils::GetRefMapping(graph, mem_assist_info.symbol_to_anchors,
                                       mem_assist_info.anchor_to_symbol);
  EXPECT_EQ(ret, SUCCESS);
  BlockMemAssigner::PreparationForAssign(mem_assist_info);

  std::vector<int64_t> ranges;
  BinaryBlockMemAssigner assigner(mem_assist_info);
  assigner.SetReuseStrategy(ReuseStrategy{false, true, false, true});
  assigner.GetMemoryRanges(ranges);

  bool no_need_assign_memory_flag = false;

  auto d = graph->FindNode("d");
  ASSERT_NE(d, nullptr);
  ASSERT_EQ(assigner.GetNoNeedAssignMemoryFlag(d, 0, no_need_assign_memory_flag), SUCCESS);
  EXPECT_FALSE(no_need_assign_memory_flag);

  auto b = graph->FindNode("b");
  ASSERT_NE(b, nullptr);
  ASSERT_EQ(assigner.GetNoNeedAssignMemoryFlag(b, 0, no_need_assign_memory_flag), SUCCESS);
  EXPECT_TRUE(no_need_assign_memory_flag);
}

/*
 *     a            b
 *     |            |
 *  PhonyConcat     |
 *           \      |
 *         hcombroadcast
 *              ||
 *               c
 */
TEST_F(UtestBlockMemAssigner, GetNoNeedAssignMemoryFlag_Hcom_WithoutLxFusion) {
  auto graph = BuildSingleNoPaddingContinuousConnectContinuousInputNode();
  MemAssistInfo mem_assist_info;
  mem_assist_info.compute_graph = graph;
  auto ret = GraphUtils::GetRefMapping(graph, mem_assist_info.symbol_to_anchors,
                                       mem_assist_info.anchor_to_symbol);
  EXPECT_EQ(ret, SUCCESS);
  BlockMemAssigner::PreparationForAssign(mem_assist_info);

  std::vector<int64_t> ranges;
  BinaryBlockMemAssigner assigner(mem_assist_info);
  assigner.SetReuseStrategy(ReuseStrategy{false, true, false, true});
  assigner.GetMemoryRanges(ranges);

  bool no_need_assign_memory_flag = false;

  auto a = graph->FindNode("a");
  ASSERT_NE(a, nullptr);
  ASSERT_EQ(assigner.GetNoNeedAssignMemoryFlag(a, 0, no_need_assign_memory_flag), SUCCESS);
  EXPECT_FALSE(no_need_assign_memory_flag);

  auto b = graph->FindNode("b");
  ASSERT_NE(b, nullptr);
  ASSERT_EQ(assigner.GetNoNeedAssignMemoryFlag(b, 0, no_need_assign_memory_flag), SUCCESS);
  EXPECT_FALSE(no_need_assign_memory_flag);
}

/*
 *     a            b
 *     |            |
 *  PhonyConcat     |
 *           \      |
 *         hcombroadcast
 *              ||
 *               c
 */
TEST_F(UtestBlockMemAssigner, GetNoNeedAssignMemoryFlag_Hcom_WithLxFusion) {
  auto graph = BuildSingleNoPaddingContinuousConnectContinuousInputNode();
  auto b = graph->FindNode("b");
  ASSERT_NE(b, nullptr);
  std::vector<int64_t> offsets_for_fusion = {};
  AttrUtils::SetListInt(b->GetOpDesc(), ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION, offsets_for_fusion);

  MemAssistInfo mem_assist_info;
  mem_assist_info.compute_graph = graph;
  auto ret = GraphUtils::GetRefMapping(graph, mem_assist_info.symbol_to_anchors,
                                       mem_assist_info.anchor_to_symbol);
  EXPECT_EQ(ret, SUCCESS);
  BlockMemAssigner::PreparationForAssign(mem_assist_info);

  std::vector<int64_t> ranges;
  BinaryBlockMemAssigner assigner(mem_assist_info);
  assigner.SetReuseStrategy(ReuseStrategy{false, true, false, true});
  assigner.GetMemoryRanges(ranges);

  bool no_need_assign_memory_flag = false;

  auto a = graph->FindNode("a");
  ASSERT_NE(a, nullptr);
  ASSERT_EQ(assigner.GetNoNeedAssignMemoryFlag(a, 0, no_need_assign_memory_flag), SUCCESS);
  EXPECT_FALSE(no_need_assign_memory_flag);

  ASSERT_EQ(assigner.GetNoNeedAssignMemoryFlag(b, 0, no_need_assign_memory_flag), SUCCESS);
  EXPECT_TRUE(no_need_assign_memory_flag);
}

/*
 *    a    b   c
 *    |___|___|
 *        |
 *    d  pc1  f
 *    |___|___|
 *        |
 *       pc2
 */
TEST_F(UtestBlockMemAssigner, GetNoNeedAssignMemoryFlag_Cascaded_Success) {
  auto graph = MemConflictShareGraph::BuildNoPaddingContinuousInCascadedGraph();
  MemAssistInfo mem_assist_info;
  mem_assist_info.compute_graph = graph;
  auto ret = GraphUtils::GetRefMapping(graph, mem_assist_info.symbol_to_anchors,
                                       mem_assist_info.anchor_to_symbol);
  EXPECT_EQ(ret, SUCCESS);
  BlockMemAssigner::PreparationForAssign(mem_assist_info);

  std::vector<int64_t> ranges;
  BinaryBlockMemAssigner assigner(mem_assist_info);
  assigner.SetReuseStrategy(ReuseStrategy{false, true, false, true});
  assigner.GetMemoryRanges(ranges);

  bool no_need_assign_memory_flag = false;

  auto a = graph->FindNode("a");
  ASSERT_NE(a, nullptr);
  ASSERT_EQ(assigner.GetNoNeedAssignMemoryFlag(a, 0, no_need_assign_memory_flag), SUCCESS);
  EXPECT_TRUE(no_need_assign_memory_flag);

  auto b = graph->FindNode("b");
  ASSERT_NE(b, nullptr);
  ASSERT_EQ(assigner.GetNoNeedAssignMemoryFlag(b, 0, no_need_assign_memory_flag), SUCCESS);
  EXPECT_TRUE(no_need_assign_memory_flag);

  auto c = graph->FindNode("c");
  ASSERT_NE(c, nullptr);
  ASSERT_EQ(assigner.GetNoNeedAssignMemoryFlag(c, 0, no_need_assign_memory_flag), SUCCESS);
  EXPECT_TRUE(no_need_assign_memory_flag);

  auto d = graph->FindNode("d");
  ASSERT_NE(d, nullptr);
  ASSERT_EQ(assigner.GetNoNeedAssignMemoryFlag(d, 0, no_need_assign_memory_flag), SUCCESS);
  EXPECT_FALSE(no_need_assign_memory_flag);

  auto f = graph->FindNode("f");
  ASSERT_NE(f, nullptr);
  ASSERT_EQ(assigner.GetNoNeedAssignMemoryFlag(f, 0, no_need_assign_memory_flag), SUCCESS);
  EXPECT_TRUE(no_need_assign_memory_flag);

  auto pc1 = graph->FindNode("pc1");
  ASSERT_NE(pc1, nullptr);
  ASSERT_EQ(assigner.GetNoNeedAssignMemoryFlag(pc1, 0, no_need_assign_memory_flag), SUCCESS);
  EXPECT_TRUE(no_need_assign_memory_flag);
}
} // namespace ge

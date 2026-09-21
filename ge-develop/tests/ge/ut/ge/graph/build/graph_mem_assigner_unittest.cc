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

#include "macro_utils/dt_public_scope.h"
#include "graph/build/memory/binary_block_mem_assigner.h"
#include "framework/memory/memory_assigner.h"
#include "graph/build/memory/hybrid_mem_assigner.h"
#include "graph/build/memory/max_block_mem_assigner.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/manager/mem_manager.h"
#include "graph/ge_tensor.h"
#include "graph/node.h"
#include "graph/anchor.h"
#include "graph/op_desc.h"
#include "graph/normal_graph/op_desc_impl.h"
#include "graph/normal_graph/ge_tensor_impl.h"
#include "graph/attr_value.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "graph/utils/tensor_utils.h"
#include "omg/omg_inner_types.h"
#include "../passes/graph_builder_utils.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/build/model_builder.h"
#include "macro_utils/dt_public_unscope.h"
#include "test_memory_shared_graph.h"
#include "common/mem_conflict_share_graph.h"
#include "stub/gert_runtime_stub.h"

using namespace std;
using namespace testing;
using domi::GetContext;

namespace ge {
Status CalculateTensorRealSizeAndOutSize(const ConstGeTensorDescPtr &output_desc, int64_t dim_index,
                                         int64_t &output_mem_size, int64_t &batch_dim_num, int64_t &out_size);
namespace {
constexpr const char_t *TBE_OP_ATOMIC_DTYPES = "tbe_op_atomic_dtypes";
constexpr const char_t *TBE_OP_ATOMIC_INT64_VALUES = "tbe_op_atomic_int64_values";
constexpr const char_t *TBE_OP_ATOMIC_FLOAT_VALUES = "tbe_op_atomic_float_values";
constexpr const char_t *ATOMIC_ATTR_HAS_ASSIGNED = "_atomic_attr_has_assigned";
struct ContinuousType {
  static const uint32_t kTypeInput = 1U;
  static const uint32_t kTypeInputNoPadding = 2U;
  static const uint32_t kTypeOutput = 4U;
  static const uint32_t kTypeOutputNoPadding = 8U;
};
void SetDefaultTensorDesc(GeTensorDescPtr &tensor_desc) {
  std::vector<int64_t> shape = {1, 1, 224, 224};
  tensor_desc->SetShape(GeShape(shape));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);
  tensor_desc->SetOriginFormat(FORMAT_NCHW);
  tensor_desc->SetOriginShape(GeShape(shape));
  tensor_desc->SetOriginDataType(DT_FLOAT);
}

void SetAtomicDataTypeList(const NodePtr &atomic_node, const std::vector<int32_t> &data_types) {
  (void) AttrUtils::SetListInt(atomic_node->GetOpDesc(), TBE_OP_ATOMIC_DTYPES, data_types);
}

void SetAtomicIntValList(const NodePtr &atomic_node, const std::vector<int32_t> &int_list) {
  (void) AttrUtils::SetListInt(atomic_node->GetOpDesc(), TBE_OP_ATOMIC_INT64_VALUES, int_list);
}

void SetAtomicFloatValList(const NodePtr &atomic_node, const std::vector<float32_t> &float_list) {
  (void) AttrUtils::SetListFloat(atomic_node->GetOpDesc(), TBE_OP_ATOMIC_FLOAT_VALUES, float_list);
}

std::vector<int32_t> GetAtomicDataTypeList(const NodePtr &atomic_node) {
  std::vector<int32_t> data_type_list;
  (void) AttrUtils::GetListInt(atomic_node->GetOpDesc(), TBE_OP_ATOMIC_DTYPES, data_type_list);
  return data_type_list;
}

std::vector<int32_t> GetMemsetDataTypeList(const NodePtr &atomic_node) {
  std::vector<int32_t> data_type_list;
  (void) AttrUtils::GetListInt(atomic_node->GetOpDesc(), ATTR_NAME_ATOMIC_MEMSET_DTYPES, data_type_list);
  return data_type_list;
}

std::vector<int32_t> GetAtomicIntValList(const NodePtr &atomic_node) {
  std::vector<int32_t> int_list;
  (void) AttrUtils::GetListInt(atomic_node->GetOpDesc(), TBE_OP_ATOMIC_INT64_VALUES, int_list);
  return int_list;
}

std::vector<int32_t> GetMemsetIntValList(const NodePtr &atomic_node) {
  std::vector<int32_t> int_list;
  (void) AttrUtils::GetListInt(atomic_node->GetOpDesc(), ATTR_NAME_ATOMIC_MEMSET_VALUES_INT, int_list);
  return int_list;
}

std::vector<float32_t> GetAtomicFloatValList(const NodePtr &atomic_node) {
  std::vector<float32_t> float_list;
  (void) AttrUtils::GetListFloat(atomic_node->GetOpDesc(), TBE_OP_ATOMIC_FLOAT_VALUES, float_list);
  return float_list;
}

std::vector<float32_t> GetMemsetFloatValList(const NodePtr &atomic_node) {
  std::vector<float32_t> float_list;
  (void) AttrUtils::GetListFloat(atomic_node->GetOpDesc(), ATTR_NAME_ATOMIC_MEMSET_VALUES_FLOAT, float_list);
  return float_list;
}

std::vector<int64_t> GetAtomicAddrList(const NodePtr &atomic_node) {
  std::vector<int64_t> mem_start_vector;
  (void) ge::AttrUtils::GetListInt(atomic_node->GetOpDesc(), ATTR_NAME_AUTOMIC_ADD_START, mem_start_vector);
  return mem_start_vector;
}
std::vector<int64_t> GetAtomicMemSizeList(const NodePtr &atomic_node) {
  std::vector<int64_t> mem_size_vector;
  (void) ge::AttrUtils::GetListInt(atomic_node->GetOpDesc(), ATTR_NAME_ATOMIC_MEMSET_SIZES, mem_size_vector);
  return mem_size_vector;
}

void SetTensorSize(const GeShape &shape,
                   const Format format,
                   const DataType data_type,
                   GeTensorDesc &tensor_desc) {
  int64_t tensor_size = 0;
  TensorUtils::CalcTensorMemSize(shape, format, data_type, tensor_size);
  TensorUtils::SetSize(tensor_desc, tensor_size);
}

void UpdateGraphTensorSize(ComputeGraphPtr &graph) {
  for (auto &node : graph->GetAllNodes()) {
    for (auto &input_name : node->GetOpDesc()->GetAllInputNames()) {
      auto input = node->GetOpDesc()->MutableInputDesc(input_name);
      SetTensorSize(input->GetShape(), input->GetFormat(), input->GetDataType(), *input);
    }
    auto out_size = node->GetOpDesc()->GetAllOutputsDescSize();
    for (int32_t id = 0; id < static_cast<int32_t>(out_size); id++) {
      auto output = node->GetOpDesc()->MutableOutputDesc(id);
      SetTensorSize(output->GetShape(), output->GetFormat(), output->GetDataType(), *output);
    }
  }
}
}

class UtestGraphMemAssigner : public testing::Test {
 public:
    ge::ComputeGraphPtr BuildGraphWithVar(int64_t session_id) {
      // init
      MemManager::Instance().Initialize(std::vector<rtMemType_t>({RT_MEMORY_HBM}));
      VarManager::Instance(session_id)->Init(0, 0, 0, 0);
      ge::ut::GraphBuilder builder("graph");
      auto var_input = builder.AddNode("var", "Variable", 1, 1);
      auto const_input = builder.AddNode("const", "Const", 1, 1);
      auto assign = builder.AddNode("assgin", "Assign", 2, 1);
      // add link
      builder.AddDataEdge(var_input, 0, assign, 0);
      builder.AddDataEdge(const_input, 0, assign, 1);
      // set offset
      var_input->GetOpDesc()->SetOutputOffset({10000});
      const_input->GetOpDesc()->SetOutputOffset({1000});
      assign->GetOpDesc()->SetInputOffset({10100, 1000});
      assign->GetOpDesc()->SetOutputOffset({10100});
      // set inner offset
      int64_t inner_offset = 100;
      ge::AttrUtils::SetInt(assign->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_INNER_OFFSET, inner_offset);
      ge::AttrUtils::SetInt(assign->GetOpDesc()->MutableOutputDesc(0), ATTR_NAME_INNER_OFFSET, inner_offset);
      // add var addr
      VarManager::Instance(session_id)->var_resource_->var_offset_map_.emplace(10000, RT_MEMORY_HBM);

      return builder.GetGraph();
    }

protected:
    void SetUp() {}
    void TearDown() {}
};


NodePtr UtAddNode(ComputeGraphPtr &graph, std::string name, std::string type, int in_cnt, int out_cnt) {
  auto op_desc = std::make_shared<OpDesc>(name, type);
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  SetDefaultTensorDesc(tensor_desc);

  for (int i = 0; i < in_cnt; ++i) {
    op_desc->AddInputDesc(tensor_desc->Clone());
  }
  for (int i = 0; i < out_cnt; ++i) {
    op_desc->AddOutputDesc(tensor_desc->Clone());
  }
  return graph->AddNode(op_desc);
}

TEST_F(UtestGraphMemAssigner, graph_memory_assign_fail_case) {
  ge::ComputeGraphPtr compute_graph = std::make_shared<ge::ComputeGraph>("");
  GraphMemoryAssigner graph_mem_assigner(compute_graph);
  MemoryOffset mem_offset(2, 65UL * 1024Ul * 1024UL * 1024UL);
  graph_mem_assigner.memory_offset_.insert({2, mem_offset});
  VarManager::Instance(0)->use_max_mem_size_ = 0;

  map<uint64_t, size_t> mem_type_to_offset = {};
  Status ret = graph_mem_assigner.ReAssignMemory(mem_type_to_offset);
  EXPECT_EQ(ret, ACL_ERROR_GE_MEMORY_ALLOCATION);
}

TEST_F(UtestGraphMemAssigner, graph_memory_get_type_case) {
  ge::ComputeGraphPtr compute_graph = std::make_shared<ge::ComputeGraph>("");
  GraphMemoryAssigner graph_mem_assigner(compute_graph);
  ge::ut::GraphBuilder builder("graph");
  NodePtr node_one = builder.AddNode("node_one", ATTR_NAME_CONTINUOUS_INPUT, 1, 1);
  std::vector<int64_t> mem_type_list;
  mem_type_list.emplace_back(55);
  mem_type_list.emplace_back(66);
  ge::AttrUtils::SetListInt(node_one->GetOpDesc(), ATTR_NAME_INPUT_MEM_TYPE_LIST, mem_type_list);
  compute_graph->AddNode(node_one);
  int64_t memory_type = 0;
  Status ret = graph_mem_assigner.GetNodeMemoryType(node_one, memory_type, "input");
  EXPECT_EQ(ret, FAILED);

  NodePtr node_two = builder.AddNode("node_two", ATTR_NAME_CONTINUOUS_OUTPUT, 1, 1);
  ge::AttrUtils::SetListInt(node_two->GetOpDesc(), ATTR_NAME_OUTPUT_MEM_TYPE_LIST, mem_type_list);
  compute_graph->AddNode(node_two);
  ret = graph_mem_assigner.GetNodeMemoryType(node_one, memory_type, "output");
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestGraphMemAssigner, Assign) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("");
  ge::VariableMemoryAssigner vma(graph);
  vma.compute_graph_ = nullptr;
  EXPECT_EQ(vma.Assign(), FAILED);
}

TEST_F(UtestGraphMemAssigner, AssignMemory) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("");
  ge::GraphMemoryAssigner gma(graph);
  EXPECT_EQ(gma.AssignMemory(), SUCCESS);
}

TEST_F(UtestGraphMemAssigner, AssignContinuousInputMemory) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node1 = UtAddNode(graph, "data1", "DATA", 1, 1);
  GraphMemoryAssigner graph_mem_assigner(graph);
  EXPECT_EQ(graph_mem_assigner.AssignContinuousInputMemory(node1, 2, true), FAILED);
}

TEST_F(UtestGraphMemAssigner, AssignContinuousInputMemory_multi_output_continuous_success) {
  vector<std::string> engine_list = {"AIcoreEngine"};
  std::vector<int64_t> memtype_list = {RT_MEMORY_HBM, RT_MEMORY_HBM};
  auto hcom1 = OP_CFG(HCOMALLREDUCE)
                  .InCnt(2)
                  .OutCnt(2)
                  .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224})
                  .Attr(ATTR_NAME_INPUT_MEM_TYPE_LIST, memtype_list)
                  .Attr(ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memtype_list)
                  .Attr(ATTR_NAME_CONTINUOUS_OUTPUT, true);
  auto hcom2 = OP_CFG(HCOMALLREDUCE)
                .InCnt(2)
                .OutCnt(2)
                .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224})
                .Attr(ATTR_NAME_INPUT_MEM_TYPE_LIST, memtype_list)
                .Attr(ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memtype_list)
                .Attr(ATTR_NAME_CONTINUOUS_INPUT, true);

  auto data1 = OP_CFG(DATA);
  auto data2 = OP_CFG(DATA);
  auto data3 = OP_CFG("Print");
  auto data4 = OP_CFG("Print");

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("hcom_1", hcom1));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("hcom_1", hcom1));
    CHAIN(NODE("hcom_1", hcom1)->EDGE(0, 0)->NODE("hcom_2", hcom2));
    CHAIN(NODE("hcom_1", hcom1)->EDGE(1, 1)->NODE("hcom_2", hcom2));
    CHAIN(NODE("hcom_2", hcom2)->EDGE(0, 0)->NODE("data_3", data3));
    CHAIN(NODE("hcom_2", hcom2)->EDGE(1, 0)->NODE("data_4", data4));
  };

  auto compute_graph = ToComputeGraph(g1);
  UpdateGraphTensorSize(compute_graph);

  GraphMemoryAssigner graph_mem_assigner(compute_graph);
  MemoryOffset memory_offset(RT_MEMORY_HBM, 0UL, 0x10);
  graph_mem_assigner.memory_offset_.emplace(RT_MEMORY_HBM, memory_offset);
  auto vec1 = std::vector<int64_t>();
  vec1.push_back(1);
  vec1.push_back(10);
  compute_graph->FindNode("hcom_1")->GetOpDesc()->SetOutputOffset(vec1);
  auto node_hcom2 = compute_graph->FindNode("hcom_2");
  EXPECT_EQ(graph_mem_assigner.AssignContinuousInputMemory(node_hcom2, 0, true),
            SUCCESS);
}

TEST_F(UtestGraphMemAssigner, AssignContinuousInputMemory_multi_output_continuous) {
  vector<std::string> engine_list = {"AIcoreEngine"};
  std::vector<int64_t> memtype_list = {RT_MEMORY_HBM, RT_MEMORY_HBM};
  auto hcom1 = OP_CFG(HCOMALLREDUCE)
                  .InCnt(2)
                  .OutCnt(2)
                  .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224})
                  .Attr(ATTR_NAME_INPUT_MEM_TYPE_LIST, memtype_list)
                  .Attr(ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memtype_list)
                  .Attr(ATTR_NAME_CONTINUOUS_OUTPUT, true);
  auto hcom2 = OP_CFG(HCOMALLREDUCE)
                .InCnt(2)
                .OutCnt(2)
                .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224})
                .Attr(ATTR_NAME_INPUT_MEM_TYPE_LIST, memtype_list)
                .Attr(ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memtype_list)
                .Attr(ATTR_NAME_CONTINUOUS_INPUT, true);

  auto data1 = OP_CFG(DATA);
  auto data2 = OP_CFG(DATA);
  auto data3 = OP_CFG("Print");
  auto data4 = OP_CFG("Print");

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("hcom_1", hcom1));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("hcom_1", hcom1));
    CHAIN(NODE("hcom_1", hcom1)->EDGE(1, 0)->NODE("hcom_2", hcom2));
    CHAIN(NODE("hcom_1", hcom1)->EDGE(0, 1)->NODE("hcom_2", hcom2));
    CHAIN(NODE("hcom_2", hcom2)->EDGE(0, 0)->NODE("data_3", data3));
    CHAIN(NODE("hcom_2", hcom2)->EDGE(1, 0)->NODE("data_4", data4));
  };

  auto compute_graph = ToComputeGraph(g1);
  UpdateGraphTensorSize(compute_graph);

  GraphMemoryAssigner graph_mem_assigner(compute_graph);
  MemoryOffset memory_offset(RT_MEMORY_HBM, 0UL, 0x10);
  graph_mem_assigner.memory_offset_.emplace(RT_MEMORY_HBM, memory_offset);
  auto vec1 = std::vector<int64_t>();
  vec1.push_back(1);
  vec1.push_back(10);
  compute_graph->FindNode("hcom_1")->GetOpDesc()->SetOutputOffset(vec1);
  auto node_hcom2 = compute_graph->FindNode("hcom_2");
  EXPECT_EQ(graph_mem_assigner.AssignContinuousInputMemory(node_hcom2, 0, true), SUCCESS);
}

TEST_F(UtestGraphMemAssigner, CleanMemInfo_Success_MergeCleanMemInfos) {
  std::set<CleanMemInfo> info_set;
  CleanMemInfo mem1;
  mem1.offset = 112708094;
  mem1.size = 3584;
  info_set.insert(mem1);

  CleanMemInfo mem2;
  mem2.offset = 112708094;
  mem2.size = 12800;
  info_set.insert(mem2);

  ASSERT_TRUE(info_set.size() == 2U);
  GraphMemoryAssigner assigner(nullptr);
  const auto mem_vec = assigner.MergeCleanMemInfos(info_set, {});
  ASSERT_TRUE(mem_vec.size() == 1U);
  ASSERT_TRUE(mem_vec.front().size == 12800);
}

TEST_F(UtestGraphMemAssigner, CleanMemInfo_Success_MergeCleanMemInfos_SameOffset) {
  std::set<CleanMemInfo> info_set;
  {
    CleanMemInfo mem;
    mem.offset = 0;
    mem.size = 512;
    info_set.insert(mem);
  }
  {
    CleanMemInfo mem;
    mem.offset = 0;
    mem.size = 512;
    info_set.insert(mem);
  }
  {
    CleanMemInfo mem;
    mem.offset = 512;
    mem.size = 512;
    info_set.insert(mem);
  }
  {
    CleanMemInfo mem;
    mem.offset = 0;
    mem.size = 512;
    info_set.insert(mem);
  }
  {
    CleanMemInfo mem;
    mem.offset = 0;
    mem.size = 512;
    info_set.insert(mem);
  }
  {
    CleanMemInfo mem;
    mem.offset = 0;
    mem.size = 1536;
    info_set.insert(mem);
  }
  ASSERT_TRUE(info_set.size() == 3U);
  GraphMemoryAssigner assigner(nullptr);
  const auto mem_vec = assigner.MergeCleanMemInfos(info_set, {});
  ASSERT_TRUE(mem_vec.size() == 1U);
  ASSERT_TRUE(mem_vec.front().size == 1536);
}

/*
 * data_1 data_2
 *   |  \  /
 *   |  hcom_1
 *   \   |
 *   hcom_2
 *   |  |
 *   a  b
 *
 *  预期会报错。因为hcom1要求data_1后紧挨着data_2，但是hcom2要求data_1后面紧挨着hcom1输出，分配内存会出错。
 *  如果用户构造这种图，会有pass在data_1后面插入identity，不允许一个out_anchor同时连接两个需要连续输入的节点上。
 */
TEST_F(UtestGraphMemAssigner, AssignContinuousInputMemory_Failed_SameOutAnchorToMultiContinuousIn) {
  vector<std::string> engine_list = {"AIcoreEngine"};
  std::vector<int64_t> memtype_list = {RT_MEMORY_HBM, RT_MEMORY_HBM};
  auto hcom1 = OP_CFG(HCOMALLREDUCE)
                  .InCnt(2)
                  .OutCnt(2)
                  .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224})
                  .Attr(ATTR_NAME_INPUT_MEM_TYPE_LIST, memtype_list)
                  .Attr(ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memtype_list)
                  .Attr(ATTR_NAME_CONTINUOUS_INPUT, true);
  auto hcom2 = OP_CFG(HCOMALLREDUCE)
                .InCnt(2)
                .OutCnt(2)
                .TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1, 224, 224})
                .Attr(ATTR_NAME_INPUT_MEM_TYPE_LIST, memtype_list)
                .Attr(ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memtype_list)
                .Attr(ATTR_NAME_CONTINUOUS_INPUT, true);

  auto data1 = OP_CFG(DATA);
  auto data2 = OP_CFG(DATA);
  auto a = OP_CFG("Print");
  auto b = OP_CFG("Print");

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("hcom_1", hcom1));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("hcom_1", hcom1));
    CHAIN(NODE("data_1", hcom1)->EDGE(0, 0)->NODE("hcom_2", hcom2));
    CHAIN(NODE("hcom_1", hcom1)->EDGE(0, 1)->NODE("hcom_2", hcom2));
    CHAIN(NODE("hcom_2", hcom2)->EDGE(0, 0)->NODE("a", a));
    CHAIN(NODE("hcom_2", hcom2)->EDGE(1, 0)->NODE("b", b));
  };

  auto compute_graph = ToComputeGraph(g1);
  compute_graph->TopologicalSorting();
  UpdateGraphTensorSize(compute_graph);

  std::map<uint64_t, size_t> mem_type_to_mem_offset;
  size_t zero_copy_mem_size;
  VarManager::Instance(compute_graph->GetSessionID())->Init(0, compute_graph->GetSessionID(), 0, 0);
  MemoryAssigner mem_assigner(compute_graph);
  EXPECT_NE(mem_assigner.AssignMemory(mem_type_to_mem_offset, zero_copy_mem_size), SUCCESS);
}

/*
 *   data_1 data_2
 *       \  /
 *      hcom_1 (ATTR_NAME_CONTINUOUS_OUTPUT)
 *       | |
 *      hcom_2 (ATTR_NAME_CONTINUOUS_INPUT)
 *       | |
 *       a b
 *       | |
 *       c d
 */
TEST_F(UtestGraphMemAssigner, AssignContinuousInputMemory_SUCCESS_TwoOutConnectTwoIn) {
  auto compute_graph = block_mem_ut::BuildContinuousOutInWithTwoOutIn();

  std::map<uint64_t, size_t> mem_type_to_mem_offset;
  size_t zero_copy_mem_size;
  VarManager::Instance(compute_graph->GetSessionID())->Init(0, compute_graph->GetSessionID(), 0, 0);
  MemoryAssigner mem_assigner(compute_graph);
  EXPECT_EQ(mem_assigner.AssignMemory(mem_type_to_mem_offset, zero_copy_mem_size), SUCCESS);
}

/*
 *   data_1 data_2
 *       \  /
 *      hcom_1 (ATTR_NAME_CONTINUOUS_OUTPUT)
 *       | |
 *      hcom_2 (ATTR_NAME_CONTINUOUS_INPUT)
 *       | |
 *       a b
 *       | |
 *       c d
 */
TEST_F(UtestGraphMemAssigner, AssignContinuousInputMemory_SUCCESS_CanReuseInSeparatAtomicClean) {
  OptionSetter option_setter({{ATOMIC_CLEAN_POLICY, "1"}});

  auto compute_graph = block_mem_ut::BuildContinuousOutInWithTwoOutIn();
  std::map<uint64_t, size_t> mem_type_to_mem_offset;
  size_t zero_copy_mem_size;
  VarManager::Instance(compute_graph->GetSessionID())->Init(0, compute_graph->GetSessionID(), 0, 0);
  MemoryAssigner mem_assigner(compute_graph);
  EXPECT_EQ(mem_assigner.AssignMemory(mem_type_to_mem_offset, zero_copy_mem_size), SUCCESS);
}

TEST_F(UtestGraphMemAssigner, AssignContinuousInputMemoryWithOffset) {
  ge::ComputeGraphPtr graph2 = std::make_shared<ge::ComputeGraph>("graph");
  auto node3 = UtAddNode(graph2, "data3", DATA, 1, 1);
  auto node4 = UtAddNode(graph2, "data4", DATA, 1, 1);
  node4->GetOutDataAnchor(0)->SetIdx(0);
  EXPECT_EQ(node3->GetInDataAnchor(0)->LinkFrom(node4->GetOutDataAnchor(0)), GRAPH_SUCCESS);
  std::vector<int64_t> offsets_of_continuous = {0};
  AttrUtils::SetListInt(node4->GetOpDesc(), ATTR_NAME_OUTPUT_OFFSET_LIST_FOR_CONTINUOUS, offsets_of_continuous);
  GraphMemoryAssigner graph_mem_assigner2(graph2);
  graph_mem_assigner2.memory_offset_.insert(std::make_pair<int64_t, MemoryOffset>(2, MemoryOffset((rtMemType_t)2, (size_t)1024)));

  node3->GetOpDesc()->SetOutputOffset({100});
  node4->GetOpDesc()->SetOutputOffset({0});
  EXPECT_EQ(graph_mem_assigner2.AssignContinuousInputMemory(node3, 2, true), SUCCESS);
  EXPECT_EQ(node4->GetOpDesc()->GetOutputOffset()[0], 100);
}

TEST_F(UtestGraphMemAssigner, AssignContinuousInputMemoryFailedIdx) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node1 = UtAddNode(graph, "data1", DATA, 1, 1);
  auto node2 = UtAddNode(graph, "data2", DATA, 1, 1);
  node2->GetOutDataAnchor(0)->SetIdx(10);
  EXPECT_EQ(node1->GetInDataAnchor(0)->LinkFrom(node2->GetOutDataAnchor(0)), GRAPH_SUCCESS);
  std::vector<int64_t> offsets_of_fusion = {1,2,3};
  AttrUtils::SetListInt(node2->GetOpDesc(), ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION, offsets_of_fusion);
  GraphMemoryAssigner graph_mem_assigner(graph);
  graph_mem_assigner.memory_offset_.insert(std::make_pair<int64_t, MemoryOffset>(2, MemoryOffset((rtMemType_t)2, (size_t)1024)));

  EXPECT_EQ(graph_mem_assigner.AssignContinuousInputMemory(node1, 2, true), FAILED);

  ge::ComputeGraphPtr graph2 = std::make_shared<ge::ComputeGraph>("graph");
  auto node3 = UtAddNode(graph2, "data3", DATA, 1, 1);
  auto node4 = UtAddNode(graph2, "data4", DATA, 1, 1);
  node4->GetOutDataAnchor(0)->SetIdx(10);
  EXPECT_EQ(node3->GetInDataAnchor(0)->LinkFrom(node4->GetOutDataAnchor(0)), GRAPH_SUCCESS);
  std::vector<int64_t> offsets_of_continuous = {1,2,3};
  AttrUtils::SetListInt(node4->GetOpDesc(), ATTR_NAME_OUTPUT_OFFSET_LIST_FOR_CONTINUOUS, offsets_of_continuous);
  GraphMemoryAssigner graph_mem_assigner2(graph2);
  graph_mem_assigner2.memory_offset_.insert(std::make_pair<int64_t, MemoryOffset>(2, MemoryOffset((rtMemType_t)2, (size_t)1024)));
  EXPECT_NE(graph_mem_assigner2.AssignContinuousInputMemory(node3, 2, true), SUCCESS);
}

TEST_F(UtestGraphMemAssigner, AssignContinuousInputMemoryFailed) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node1 = UtAddNode(graph, "data1", DATA, 1, 1);
  auto node2 = UtAddNode(graph, "data2", DATA, 1, 1);
  EXPECT_EQ(node1->GetInDataAnchor(0)->LinkFrom(node2->GetOutDataAnchor(0)), GRAPH_SUCCESS);
  std::vector<int64_t> offsets_of_fusion = {1,2,3};
  AttrUtils::SetListInt(node2->GetOpDesc(), ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION, offsets_of_fusion);
  GraphMemoryAssigner graph_mem_assigner(graph);
  graph_mem_assigner.memory_offset_.insert(std::make_pair<int64_t, MemoryOffset>(2, MemoryOffset((rtMemType_t)2, (size_t)1024)));
  EXPECT_NE(graph_mem_assigner.AssignContinuousInputMemory(node1, 2, true), SUCCESS);
}

TEST_F(UtestGraphMemAssigner, AssignContinuousInputMemoryWithAtomicProcessDirectly) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node1 = UtAddNode(graph, "data1", DATA, 1, 1);
  auto node2 = UtAddNode(graph, "var2", VARIABLE, 1, 1);
  EXPECT_EQ(node1->GetInDataAnchor(0)->LinkFrom(node2->GetOutDataAnchor(0)), GRAPH_SUCCESS);
  GraphMemoryAssigner graph_mem_assigner(graph);
  auto nmap = std::map<NodePtr, uint32_t>();
  nmap[node1] = 4;
  EXPECT_EQ(graph_mem_assigner.IsAssignContinuousInputMemoryDirectly(node1, nmap), true);
  auto node3 = UtAddNode(graph, "data3", DATA, 1, 1);
  auto node4 = UtAddNode(graph, "data4", DATA, 1, 1);
  EXPECT_EQ(node3->GetInDataAnchor(0)->LinkFrom(node4->GetOutDataAnchor(0)), GRAPH_SUCCESS);
  nmap[node3] = 1U;
  nmap[node4] = 2U;
  EXPECT_EQ(graph_mem_assigner.IsAssignContinuousInputMemoryDirectly(node3, nmap), false);
  EXPECT_EQ(ge::AttrUtils::SetBool(node3->GetOpDesc(), ATTR_NAME_CONTINUOUS_INPUT, true), true);
  EXPECT_EQ(graph_mem_assigner.IsAssignContinuousInputMemoryDirectly(node4, nmap), true);
}

TEST_F(UtestGraphMemAssigner, CheckOffset) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node1 = UtAddNode(graph, "data1", DATA, 1, 1);
  auto node4 = UtAddNode(graph, "data4", DATA, 1, 1);
  EXPECT_EQ(node1->GetInDataAnchor(0)->LinkFrom(node4->GetOutDataAnchor(0)), GRAPH_SUCCESS);
  GraphMemoryAssigner graph_mem_assigner(graph);
  graph_mem_assigner.mem_assigner_.reset(new(std::nothrow) HybridMemAssigner(graph));
  EXPECT_EQ(graph_mem_assigner.CheckOffset(), SUCCESS);
  ge::ComputeGraphPtr graph2 = std::make_shared<ge::ComputeGraph>("graph2");
  auto node2 = UtAddNode(graph2, "data2", DATA, 1, 1);
  GraphMemoryAssigner graph_mem_assigner2(graph2);
  graph_mem_assigner2.mem_assigner_.reset(new(std::nothrow) HybridMemAssigner(graph));
  auto vec1 = std::vector<int64_t>();
  auto vec2 = std::vector<int64_t>();
  vec1.push_back(1);
  vec2.push_back(-1);
  node2->GetOpDesc()->SetInputOffset(vec1);
  node2->GetOpDesc()->SetOutputOffset(vec2);
  EXPECT_NE(graph_mem_assigner2.CheckOffset(), SUCCESS);
  vec1.push_back(-1);
  node2->GetOpDesc()->SetInputOffset(vec1);
  EXPECT_NE(graph_mem_assigner2.CheckOffset(), SUCCESS);
  vec1.clear();
  vec2.clear();
  vec1.push_back(1);
  vec2.push_back(1);
  ge::ComputeGraphPtr graph3 = std::make_shared<ge::ComputeGraph>("graph3");
  auto node3 = UtAddNode(graph3, "iden3", IDENTITY, 1, 1);
  GraphMemoryAssigner graph_mem_assigner3(graph3);
  graph_mem_assigner3.mem_assigner_.reset(new(std::nothrow) HybridMemAssigner(graph));
  vec1.push_back(1);
  vec2.push_back(1);
  node3->GetOpDesc()->SetInputOffset(vec1);
  node3->GetOpDesc()->SetOutputOffset(vec2);
  EXPECT_EQ(graph_mem_assigner3.CheckOffset(), SUCCESS);
}

TEST_F(UtestGraphMemAssigner, CheckOffset_UnfedOptionalInput_CheckSuccess) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");

  auto op_desc = std::make_shared<OpDesc>("node1", "UnfedOptionalInputOp");
  ge::AttrUtils::SetBool(op_desc, ATTR_NAME_REFERENCE, true);
  auto invalid_td = std::make_shared<GeTensorDesc>();
  invalid_td->SetDataType(DT_UNDEFINED);
  invalid_td->SetFormat(FORMAT_RESERVED);
  op_desc->AddInputDesc(invalid_td->Clone());

  auto in_tensor_desc = std::make_shared<GeTensorDesc>();
  auto out_tensor_desc = std::make_shared<GeTensorDesc>();
  SetDefaultTensorDesc(in_tensor_desc);
  SetDefaultTensorDesc(out_tensor_desc);
  op_desc->AddInputDesc("reuse_desc0", in_tensor_desc->Clone());
  op_desc->AddOutputDesc("reuse_desc0", out_tensor_desc->Clone());

  auto node = graph->AddNode(op_desc);
  GraphMemoryAssigner graph_mem_assigner3(graph);
  graph_mem_assigner3.mem_assigner_.reset(new(std::nothrow) HybridMemAssigner(graph));
  auto vec1 = std::vector<int64_t>();
  auto vec2 = std::vector<int64_t>();
  vec1.push_back(1);
  vec2.push_back(1);
  node->GetOpDesc()->SetInputOffset(vec1);
  node->GetOpDesc()->SetOutputOffset(vec2);
  EXPECT_EQ(graph_mem_assigner3.CheckOffset(), SUCCESS);
}

TEST_F(UtestGraphMemAssigner, CheckOffset_Workspace) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = UtAddNode(graph, "data", DATA, 1, 1);
  GraphMemoryAssigner graph_mem_assigner(graph);
  graph_mem_assigner.mem_assigner_.reset(new(std::nothrow) HybridMemAssigner(graph));
  auto vec1 = std::vector<int64_t>();
  auto vec2 = std::vector<int64_t>();
  auto vec3 = std::vector<int64_t>();
  vec1.push_back(1);
  vec2.push_back(1);
  vec3.push_back(-1);
  node->GetOpDesc()->SetInputOffset(vec1);
  node->GetOpDesc()->SetOutputOffset(vec2);
  node->GetOpDesc()->SetWorkspace(vec3);
  EXPECT_NE(graph_mem_assigner.CheckOffset(), SUCCESS);
}

TEST_F(UtestGraphMemAssigner, AssignContinuousOutputMemory) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = UtAddNode(graph, "data", DATA, 1, 1);
  auto node2 = UtAddNode(graph, "data2", DATA, 0, 0);
  auto node3 = UtAddNode(graph, "data3", DATA, 1, 1);
  EXPECT_EQ(node->GetInDataAnchor(0)->LinkFrom(node3->GetOutDataAnchor(0)), GRAPH_SUCCESS);
  GraphMemoryAssigner graph_mem_assigner(graph);
  EXPECT_EQ(graph_mem_assigner.AssignContinuousOutputMemory(node, 1, 8), FAILED);
  auto output = vector<int64_t>();
  output.push_back(1);
  node->GetOpDesc()->SetOutputOffset(output);
  EXPECT_EQ(graph_mem_assigner.AssignContinuousOutputMemory(node, 1, 127), FAILED);
  auto output3 = vector<int64_t>();
  output3.push_back(1);
  output3.push_back(2);
  output3.push_back(3);
  node3->GetOpDesc()->SetOutputOffset(output3);
  EXPECT_EQ(node3->GetOpDesc()->GetOutputOffset().size(), 3);
  EXPECT_EQ(node3->GetOutDataAnchor(0)->GetIdx(), 0);
  EXPECT_EQ(graph_mem_assigner.AssignContinuousOutputMemory(node, 1, 127), FAILED);
  AttrUtils::SetBool(node->GetOpDesc(), "reference", true);
  EXPECT_EQ(graph_mem_assigner.AssignContinuousOutputMemory(node, 1, 4), SUCCESS);
  auto output2 = vector<int64_t>();
  output2.push_back(1);
  node2->GetOpDesc()->SetOutputOffset(output2);
  EXPECT_EQ(graph_mem_assigner.AssignContinuousOutputMemory(node2, 1, 127), FAILED);
}

TEST_F(UtestGraphMemAssigner, AssignContinuousOutputMemoryCal) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = UtAddNode(graph, "data", DATA, 1, 1);
  auto node3 = UtAddNode(graph, "data3", DATA, 1, 1);
  EXPECT_EQ(node->GetInDataAnchor(0)->LinkFrom(node3->GetOutDataAnchor(0)), GRAPH_SUCCESS);
  GraphMemoryAssigner graph_mem_assigner(graph);
  auto output = vector<int64_t>();
  output.push_back(1);
  output.push_back(2);
  output.push_back(3);
  node->GetOpDesc()->SetOutputOffset(output);
  auto output3 = vector<int64_t>();
  output3.push_back(1);
  output3.push_back(2);
  output3.push_back(3);
  node3->GetOpDesc()->SetOutputOffset(output3);
  EXPECT_EQ(node->GetOpDesc()->GetOutputOffset().size(), 3);
  EXPECT_EQ(node->GetOutDataAnchor(0)->GetIdx(), 0);
  ge::AttrUtils::SetInt(node->GetOpDesc(), "_reuse_input_on_dim_index", 2);
  ConstGeTensorDescPtr output_desc = node->GetOpDesc()->GetOutputDescPtr(node->GetOutDataAnchor(0)->GetIdx());
  EXPECT_EQ(output_desc->GetShape().GetDims().size(), 4);
  EXPECT_EQ(graph_mem_assigner.AssignContinuousOutputMemory(node, 1, 15), SUCCESS);
}

TEST_F(UtestGraphMemAssigner, CheckInputIsSupportAtomic) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = UtAddNode(graph, "data", DATA, 1, 1);
  GraphMemoryAssigner graph_mem_assigner(graph);
  EXPECT_EQ(graph_mem_assigner.CheckInputIsSupportAtomic(node.get()), true);
  auto node2 = UtAddNode(graph, "data2", DATA, 1, 1);
  EXPECT_EQ(node->GetInDataAnchor(0)->LinkFrom(node2->GetOutDataAnchor(0)), GRAPH_SUCCESS);
  auto op_desc = node2->GetOpDesc();
  ge::OpDescUtilsEx::SetType(op_desc, VARIABLE);
  EXPECT_EQ(graph_mem_assigner.CheckInputIsSupportAtomic(node.get()), false);
}

/*
 *     data
 *      |
 *   atomic_node
 *     / \
 *   Node_Output
 */
TEST_F(UtestGraphMemAssigner, SetAtomicCleanOffset_CheckAttrs_Success) {
  DEF_GRAPH(graph) {
    auto atomic_memset = OP_CFG(MEMSET)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});
    auto fake_type2_op1 = OP_CFG("AtomicNode")
        .InCnt(1)
        .OutCnt(2)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});
    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(2)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {-1});
    CHAIN(NODE("data", atomic_memset)->NODE("atomic_node", fake_type2_op1)->NODE("Node_Output", net_output));
    CHAIN(NODE("atomic_node", fake_type2_op1)->NODE("Node_Output", net_output));
  };
  auto root_graph = ToComputeGraph(graph);
  GraphMemoryAssigner graph_mem_assigner(root_graph);
  UpdateGraphTensorSize(root_graph);
  auto atomic_node = root_graph->FindFirstNodeMatchType("AtomicNode");
  auto atomic_clean_node = root_graph->FindFirstNodeMatchType(MEMSET);
  ASSERT_NE(atomic_node, nullptr);
  ASSERT_NE(atomic_clean_node, nullptr);
  InControlAnchorPtr in_ctrl_anchor = atomic_node->GetInControlAnchor();
  OutControlAnchorPtr out_ctrl_anchor = atomic_clean_node->GetOutControlAnchor();
  EXPECT_EQ(GraphUtils::AddEdge(out_ctrl_anchor, in_ctrl_anchor), SUCCESS);
  std::vector<int32_t> data_list = {ge::DataType::DT_INT16, ge::DataType::DT_INT16};
  std::vector<int32_t> int_list = {0x55, 0xaa};
  std::vector<float32_t> float_list = {};
  SetAtomicDataTypeList(atomic_node, data_list);
  SetAtomicIntValList(atomic_node, int_list);
  SetAtomicFloatValList(atomic_node, float_list);
  AttrUtils::SetBool(atomic_node->GetOpDesc(), ATOMIC_ATTR_IS_ATOMIC_NODE, true);
  EXPECT_EQ(AttrUtils::SetListInt(atomic_node->GetOpDesc(), ATOMIC_ATTR_OUTPUT_INDEX, {0, 1}), true);
  auto mem_offset_end = std::vector<int64_t>({0x20, 0x30});
  auto real_atomic_sizes = std::vector<int64_t>({0x10, 0x10});
  atomic_node->GetOpDesc()->SetOutputOffset({0, 512});
  MemConflictShareGraph::SetSizeForAllNodes(root_graph);

  EXPECT_EQ(graph_mem_assigner.SetAtomicCleanOffset(), SUCCESS);
  EXPECT_EQ(GetAtomicDataTypeList(atomic_node).size(), data_list.size());
  EXPECT_EQ(GetAtomicIntValList(atomic_node).size(), int_list.size());
  EXPECT_EQ(GetAtomicFloatValList(atomic_node).size(), float_list.size());
  auto addr_list = GetAtomicAddrList(atomic_clean_node);
  ASSERT_EQ(addr_list.size(), mem_offset_end.size());
  size_t id = 0UL;
  std::vector<int64_t> expect_mem_offset({0, 512});
  for (auto addr : addr_list) {
    EXPECT_EQ(addr, expect_mem_offset[id]);
    id++;
  }
  id = 0UL;
  auto mem_size_list = GetAtomicMemSizeList(atomic_clean_node);
  EXPECT_EQ(mem_size_list.size(), real_atomic_sizes.size());
  for (auto mem_size : mem_size_list) {
    EXPECT_EQ(mem_size, mem_size_list[id]);
    id++;
  }
}

TEST_F(UtestGraphMemAssigner, SetAtomicCleanOffset_AppendAtomicCleanAttr) {
  DEF_GRAPH(graph) {
    auto atomic_memset = OP_CFG(MEMSET)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op1 = OP_CFG("AtomicNode")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {-1});

    CHAIN(NODE("atomic_memset", atomic_memset)->Ctrl()->NODE("atomic_node", fake_type2_op1)->NODE("Node_Output", net_output));
  };
  auto root_graph = ToComputeGraph(graph);
  UpdateGraphTensorSize(root_graph);
  GraphMemoryAssigner graph_mem_assigner(root_graph);
  auto atomic_node = root_graph->FindFirstNodeMatchType("AtomicNode");
  std::vector<int32_t> data_list = {ge::DataType::DT_INT16};
  std::vector<int32_t> int_list = {0x1};
  std::vector<float32_t> float_list = {};
  SetAtomicDataTypeList(atomic_node, data_list);
  SetAtomicIntValList(atomic_node, int_list);
  SetAtomicFloatValList(atomic_node, float_list);
  atomic_node->GetOpDesc()->SetOutputOffset({10240});

  auto atomic_memset = root_graph->FindFirstNodeMatchType(MEMSET);
  AttrUtils::SetBool(atomic_node->GetOpDesc(), ATOMIC_ATTR_IS_ATOMIC_NODE, true);
  EXPECT_EQ(AttrUtils::SetListInt(atomic_node->GetOpDesc(), ATOMIC_ATTR_OUTPUT_INDEX, {0}), true);
  EXPECT_EQ(graph_mem_assigner.SetAtomicCleanOffset(), SUCCESS);
  EXPECT_EQ(GetMemsetDataTypeList(atomic_memset).size(), data_list.size());
  EXPECT_EQ(GetMemsetIntValList(atomic_memset).size(), int_list.size());
  EXPECT_EQ(GetMemsetFloatValList(atomic_memset).size(), float_list.size());
}

TEST_F(UtestGraphMemAssigner, SetAtomicCleanOffset_AppendMultiAtomicCleanAttr) {
  DEF_GRAPH(graph) {
                     auto atomic_memset = OP_CFG(MEMSET)
                         .InCnt(1)
                         .OutCnt(1)
                         .TensorDesc(FORMAT_ND, DT_INT32, {16});

                     auto fake_type2_op1 = OP_CFG("AtomicNode")
                         .InCnt(1)
                         .OutCnt(1)
                         .TensorDesc(FORMAT_ND, DT_INT32, {16});

                     auto fake_type2_op2 = OP_CFG("AtomicNode")
                         .InCnt(1)
                         .OutCnt(1)
                         .TensorDesc(FORMAT_ND, DT_INT32, {16});

                     auto net_output = OP_CFG(NETOUTPUT)
                         .InCnt(1)
                         .OutCnt(1)
                         .TensorDesc(FORMAT_ND, DT_INT32, {-1});

                     CHAIN(NODE("data", atomic_memset)->Ctrl()->NODE("atomic_node1", fake_type2_op1)->NODE("Node_Output", net_output));
                     CHAIN(NODE("data", atomic_memset)->Ctrl()->NODE("atomic_node2", fake_type2_op2)->NODE("Node_Output", net_output));
                   };
  auto root_graph = ToComputeGraph(graph);
  UpdateGraphTensorSize(root_graph);
  GraphMemoryAssigner graph_mem_assigner(root_graph);
  auto atomic_node = root_graph->FindNode("atomic_node1");
  std::vector<int32_t> data_list = {ge::DataType::DT_INT16};
  std::vector<int32_t> int_list = {0x1};
  std::vector<float32_t> float_list = {};
  SetAtomicDataTypeList(atomic_node, data_list);
  SetAtomicIntValList(atomic_node, int_list);
  SetAtomicFloatValList(atomic_node, float_list);
  atomic_node->GetOpDesc()->SetOutputOffset({1024});

  auto atomic_node2 = root_graph->FindNode("atomic_node2");
  SetAtomicDataTypeList(atomic_node, data_list);
  SetAtomicIntValList(atomic_node, int_list);
  SetAtomicFloatValList(atomic_node, float_list);
  atomic_node2->GetOpDesc()->SetOutputOffset({10240});
  AttrUtils::SetListInt(atomic_node2->GetOpDesc(), ATTR_NAME_OUTPUT_MEM_TYPE_LIST, {RT_MEMORY_P2P_DDR});

  auto atomic_memset = root_graph->FindFirstNodeMatchType(MEMSET);
  AttrUtils::SetBool(atomic_node->GetOpDesc(), ATOMIC_ATTR_IS_ATOMIC_NODE, true);
  AttrUtils::SetBool(atomic_node2->GetOpDesc(), ATOMIC_ATTR_IS_ATOMIC_NODE, true);
  EXPECT_EQ(AttrUtils::SetListInt(atomic_node->GetOpDesc(), ATOMIC_ATTR_OUTPUT_INDEX, {0}), true);
  EXPECT_EQ(AttrUtils::SetListInt(atomic_node2->GetOpDesc(), ATOMIC_ATTR_OUTPUT_INDEX, {0}), true);

  EXPECT_EQ(graph_mem_assigner.SetAtomicCleanOffset(),SUCCESS);
  std::vector<int64_t> mem_type_list;
  ASSERT_TRUE(ge::AttrUtils::GetListInt(atomic_memset->GetOpDesc(), ATTR_NAME_WORKSPACE_TYPE_LIST, mem_type_list));
  ASSERT_EQ(mem_type_list.size(), 2U);
  EXPECT_EQ(mem_type_list[0], RT_MEMORY_HBM);
  EXPECT_EQ(mem_type_list[1], RT_MEMORY_P2P_DDR);
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
TEST_F(UtestGraphMemAssigner, AtomicNodeConnectReShapeConnectNetoutput_CheckMemSetAddr) {
  auto root_graph = block_mem_ut::AtomicNodeConnectReShapeConnectNetoutput();
  MemoryAssigner mem_assigner(root_graph);
  std::map<uint64_t, size_t> mem_offsets;
  size_t zero_copy_mem_size;
  EXPECT_EQ(mem_assigner.AssignMemory(mem_offsets, zero_copy_mem_size), SUCCESS);
  auto atomic_memset = root_graph->FindFirstNodeMatchType(MEMSET);
  auto clean_addrs = atomic_memset->GetOpDescBarePtr()->GetWorkspace();

  auto atomic_node = root_graph->FindNode("scatter_nd");
  auto offsets = atomic_node->GetOpDescBarePtr()->GetOutputOffset();
  ASSERT_FALSE(offsets.empty());
  ASSERT_FALSE(clean_addrs.empty());

  // check atomic clean addrs
  EXPECT_EQ(clean_addrs[0], offsets[0]);
  GELOGI("clean_addrs[0]: %lld, offsets[0]: %lld, ", clean_addrs[0], offsets[0]);

  // 这种场景目前还不支持零拷贝
}

//                        g
//                       /
//          a   b   c   d (output ref input)
//           \ /     \ /       abcde output shape: [1,1,48,448]
//           pc1  e  pc2
//[1,2,48,448]\   |  / [1,2,48,448]
//               pc3
//                |  [1,5,48,448]
//                f
//                |
//             netoutput
//
TEST_F(UtestGraphMemAssigner, PhonyConcatCascatedConnectRefNode_AssignSuccess) {
  auto root_graph = block_mem_ut::BuildPhonyConcatCascatedConnectRefNode();

  MemoryAssigner mem_assigner(root_graph);
  std::map<uint64_t, size_t> mem_offsets;
  size_t zero_copy_mem_size;
  EXPECT_EQ(mem_assigner.AssignMemory(mem_offsets, zero_copy_mem_size), SUCCESS);
}

//
//  a--c
// 添加控制边保证拓扑序为，a b c d
//            d       b
//            |       |
//      ref_node1   ref_node2
//             \     /
//               pc
//                |
//                e
//                |
//             netoutput
//
// a的生命周期在c结束，那么d可以复用吗？答案是不能。
// 因为d和b的输出是连续内存，由于b不能复用a的内存，所以d也不能复用a的内存。
//
TEST_F(UtestGraphMemAssigner, PhonyConcatWithRefNodeInputs_AssignSuccess) {
  auto root_graph = block_mem_ut::BuildPhonyConcatWithRefNodeInputs();
  MemoryAssigner mem_assigner(root_graph);
  std::map<uint64_t, size_t> mem_offsets;
  size_t zero_copy_mem_size;
  EXPECT_EQ(mem_assigner.AssignMemory(mem_offsets, zero_copy_mem_size), SUCCESS);

  auto b = root_graph->FindNode("b");
  ASSERT_NE(b, nullptr);
  auto ref_node2 = root_graph->FindNode("ref_node2");
  ASSERT_NE(ref_node2, nullptr);
  EXPECT_EQ(b->GetOpDesc()->GetOutputOffset().at(0), ref_node2->GetOpDesc()->GetOutputOffset().at(0));
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
TEST_F(UtestGraphMemAssigner, PhonyConcatWithSameInputThrougRefNode_AssignSuccess) {
  auto root_graph = block_mem_ut::BuildPhonyConcatWithSameInputThrougRefNode();
  MemoryAssigner mem_assigner(root_graph);
  std::map<uint64_t, size_t> mem_offsets;
  size_t zero_copy_mem_size;
  EXPECT_EQ(mem_assigner.AssignMemory(mem_offsets, zero_copy_mem_size), SUCCESS);
}

/*
               a
              /   \
      ref_node1--> ref_node2
             \     /  \
               pc       b
                |
                c
                |
             netoutput
 与上面的用例区别在与ref_node1和ref_node2的拓扑序。反向刷新时，ref_node1和ref_node2谁前进栈都得保证给a刷新的offset时正确的
*/
TEST_F(UtestGraphMemAssigner, PhonyConcatWithSameInputThrougRefNode_AssignSuccess2) {
  auto root_graph = block_mem_ut::BuildPhonyConcatWithSameInputThrougRefNode2();
  MemoryAssigner mem_assigner(root_graph);
  std::map<uint64_t, size_t> mem_offsets;
  size_t zero_copy_mem_size;
  EXPECT_EQ(mem_assigner.AssignMemory(mem_offsets, zero_copy_mem_size), SUCCESS);
}

TEST_F(UtestGraphMemAssigner, AssignContinuousInputMemoryWithAtomicProcess) {
  DEF_GRAPH(graph) {
     auto atomic_memset = OP_CFG(MEMSET)
         .InCnt(1)
         .OutCnt(1)
         .TensorDesc(FORMAT_ND, DT_INT32, {16});
     auto data = OP_CFG(DATA)
         .InCnt(1)
         .OutCnt(1)
         .TensorDesc(FORMAT_ND, DT_INT32, {16});

     auto fake_type2_op1 = OP_CFG("HcomNode")
         .InCnt(2)
         .OutCnt(1)
         .TensorDesc(FORMAT_ND, DT_INT32, {16});

     auto net_output = OP_CFG(NETOUTPUT)
         .InCnt(1)
         .OutCnt(1)
         .TensorDesc(FORMAT_ND, DT_INT32, {-1});

     CHAIN(NODE("data", data)->NODE("atomic_node", fake_type2_op1)->NODE("Node_Output", net_output));
     CHAIN(NODE("data", data)->NODE("atomic_node", fake_type2_op1));
     CTRL_CHAIN(NODE("memset", atomic_memset)->NODE("atomic_node", fake_type2_op1));
  };
  auto root_graph = ToComputeGraph(graph);
  GraphMemoryAssigner graph_mem_assigner(root_graph);
  auto atomic_mem_start = std::vector<int64_t>(0x1, 0x2);
  auto atomic_mem_size = std::vector<int64_t>(0x1, 0x1);
  auto atomic_node = root_graph->FindFirstNodeMatchType("HcomNode");
  auto data = root_graph->FindFirstNodeMatchType(DATA);
  MemoryOffset memory_offset(RT_MEMORY_HBM, 1UL, 0UL);
  graph_mem_assigner.memory_offset_.emplace(RT_MEMORY_HBM, memory_offset);
  graph_mem_assigner.memory_offset_.emplace(RT_MEMORY_HBM, memory_offset);
  std::vector<int32_t> mem_type_list = {RT_MEMORY_HBM, RT_MEMORY_HBM};
  ge::AttrUtils::SetListInt(atomic_node->GetOpDesc(), ATTR_NAME_INPUT_MEM_TYPE_LIST, mem_type_list);
  data->GetOpDesc()->SetOutputOffset({0, 1});

  auto atomic_memset = root_graph->FindFirstNodeMatchType(MEMSET);
  AttrUtils::SetBool(atomic_node->GetOpDesc(), ATOMIC_ATTR_IS_ATOMIC_NODE, true);
  EXPECT_EQ(AttrUtils::SetListInt(atomic_node->GetOpDesc(), ATOMIC_ATTR_INPUT_INDEX, {-1}), true);
  EXPECT_EQ(graph_mem_assigner.AssignContinuousInputMemory(atomic_node, ContinuousType::kTypeInput, false),
      SUCCESS);
}

TEST_F(UtestGraphMemAssigner, CheckContinuousMemType) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = UtAddNode(graph, "data", DATA, 1, 1);
  GraphMemoryAssigner graph_mem_assigner(graph);
  auto mem_type_list = std::vector<int64_t>();
  EXPECT_EQ(graph_mem_assigner.CheckContinuousMemType(mem_type_list), true);
}

TEST_F(UtestGraphMemAssigner, ReAssignAtomicMemory) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node1 = UtAddNode(graph, "atom1", ATOMICADDRCLEAN, 1, 1);
  auto node2 = UtAddNode(graph, "atom2", ATOMICADDRCLEAN, 1, 1);
  EXPECT_EQ(node1->GetOutControlAnchor()->LinkTo(node2->GetInControlAnchor()), GRAPH_SUCCESS);
  AttrUtils::SetBool(node2->GetOpDesc(), "is_atomic_node", true);
  std::vector<uint32_t> value;
  value.push_back(1);
  value.push_back(2);
  value.push_back(3);
  value.push_back(4);
  value.push_back(5);
  AttrUtils::SetListInt(node2->GetOpDesc(), "atomic_output_index", value);
  EXPECT_EQ(node2->GetOpDesc()->GetOutputsSize(), 1);
  GraphMemoryAssigner graph_mem_assigner(graph);
  EXPECT_FALSE(graph_mem_assigner.CheckAtomicNodeIsSupportRef(node2));
  graph_mem_assigner.ReAssignAtomicMemory();
}

TEST_F(UtestGraphMemAssigner, ReAssignAtomicMemoryMergeNodesAttrs) {
  DEF_GRAPH(ge_graph) {
   auto atomic_memset = OP_CFG(MEMSET)
       .InCnt(1)
       .OutCnt(1)
       .TensorDesc(FORMAT_ND, DT_INT32, {16});

   auto node1 = OP_CFG("AtomicNode")
       .InCnt(1)
       .OutCnt(1)
       .TensorDesc(FORMAT_ND, DT_INT32, {16});
   auto node2 = OP_CFG("AtomicNode")
       .InCnt(1)
       .OutCnt(1)
       .TensorDesc(FORMAT_ND, DT_INT32, {16});
   auto node3 = OP_CFG("AtomicNode")
       .InCnt(1)
       .OutCnt(1)
       .TensorDesc(FORMAT_ND, DT_INT32, {16});

   auto net_output = OP_CFG(NETOUTPUT)
       .InCnt(1)
       .OutCnt(1)
       .TensorDesc(FORMAT_ND, DT_INT32, {-1});

   CHAIN(NODE("atomic_node0", node1)->NODE("Node_Output", net_output));
   CHAIN(NODE("atomic_node1", node2)->NODE("Node_Output", net_output));
   CHAIN(NODE("atomic_node2", node3)->NODE("Node_Output", net_output));
   CHAIN(NODE("atomic_memset", atomic_memset)->NODE("Node_Output", net_output));
 };
  auto graph = ToComputeGraph(ge_graph);
  UpdateGraphTensorSize(graph);
  GraphMemoryAssigner graph_mem_assigner(graph);
  auto atomic_node0 = graph->FindNode("atomic_node0");
  auto atomic_node1 = graph->FindNode("atomic_node1");
  auto atomic_node2 = graph->FindNode("atomic_node2");
  auto atomic_memset = graph->FindNode("atomic_memset");
  atomic_node0->GetOpDesc()->SetOutputOffset({0});
  atomic_node1->GetOpDesc()->SetOutputOffset({16});
  atomic_node2->GetOpDesc()->SetOutputOffset({32});

  EXPECT_EQ(atomic_memset->GetOutControlAnchor()->LinkTo(atomic_node0->GetInControlAnchor()), GRAPH_SUCCESS);
  EXPECT_EQ(atomic_memset->GetOutControlAnchor()->LinkTo(atomic_node1->GetInControlAnchor()), GRAPH_SUCCESS);
  EXPECT_EQ(atomic_memset->GetOutControlAnchor()->LinkTo(atomic_node2->GetInControlAnchor()), GRAPH_SUCCESS);
  EXPECT_EQ(AttrUtils::SetBool(atomic_node0->GetOpDesc(), "is_atomic_node", true), true);
  EXPECT_EQ(AttrUtils::SetBool(atomic_node1->GetOpDesc(), "is_atomic_node", true), true);
  EXPECT_EQ(AttrUtils::SetBool(atomic_node2->GetOpDesc(), "is_atomic_node", true), true);
  std::vector<uint32_t> value;
  value.push_back(0);
  EXPECT_EQ(AttrUtils::SetListInt(atomic_node0->GetOpDesc(), "atomic_output_index", value), true);
  EXPECT_EQ(AttrUtils::SetListInt(atomic_node1->GetOpDesc(), "atomic_output_index", value), true);
  EXPECT_EQ(AttrUtils::SetListInt(atomic_node2->GetOpDesc(), "atomic_output_index", value), true);
  // init graph_mem_assigner
  graph_mem_assigner.mem_assigner_.reset(new(std::nothrow) HybridMemAssigner(graph));
  MemoryOffset memory_offset(RT_MEMORY_HBM, 0UL, 0x10);
  graph_mem_assigner.memory_offset_.emplace(RT_MEMORY_HBM, memory_offset);
  EXPECT_EQ(graph_mem_assigner.ReAssignAtomicMemory(), SUCCESS);
  EXPECT_EQ(graph_mem_assigner.SetAtomicCleanOffset(), SUCCESS);
  EXPECT_EQ(GetAtomicMemSizeList(atomic_memset).size(), 1);
  EXPECT_EQ(GetAtomicAddrList(atomic_memset).size(), 1);
  std::vector<int64_t> expect_memory_size{MEM_ALIGN_SIZE * 3};
  std::vector<int64_t> expect_memory_start{0};
  EXPECT_EQ(GetAtomicMemSizeList(atomic_memset), expect_memory_size);
  EXPECT_EQ(GetAtomicAddrList(atomic_memset), expect_memory_start);
  EXPECT_EQ(GetMemsetDataTypeList(atomic_memset).size(), 1);
  EXPECT_EQ(GetMemsetFloatValList(atomic_memset).size(), 1);
  EXPECT_EQ(GetMemsetIntValList(atomic_memset).size(), 0);
}

TEST_F(UtestGraphMemAssigner, ReAssignAtomicMemoryWithOutMergeNodesAttrs) {
  DEF_GRAPH(ge_graph) {
    auto atomic_memset = OP_CFG(MEMSET)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto node1 = OP_CFG("AtomicNode")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});
    auto node2 = OP_CFG("AtomicNode")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});
    auto node3 = OP_CFG("AtomicNode")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {-1});

    CHAIN(NODE("atomic_node0", node1)->NODE("Node_Output", net_output));
    CHAIN(NODE("atomic_node1", node2)->NODE("Node_Output", net_output));
    CHAIN(NODE("atomic_node2", node3)->NODE("Node_Output", net_output));
    CHAIN(NODE("atomic_memset", atomic_memset)->NODE("Node_Output", net_output));
  };
  auto graph = ToComputeGraph(ge_graph);
  UpdateGraphTensorSize(graph);
  GraphMemoryAssigner graph_mem_assigner(graph);
  auto atomic_node0 = graph->FindNode("atomic_node0");
  auto atomic_node1 = graph->FindNode("atomic_node1");
  auto atomic_node2 = graph->FindNode("atomic_node2");
  auto atomic_memset = graph->FindNode("atomic_memset");
  atomic_node0->GetOpDesc()->SetOutputOffset({0});
  atomic_node1->GetOpDesc()->SetOutputOffset({16});
  atomic_node2->GetOpDesc()->SetOutputOffset({32});

  EXPECT_EQ(atomic_memset->GetOutControlAnchor()->LinkTo(atomic_node0->GetInControlAnchor()), GRAPH_SUCCESS);
  EXPECT_EQ(atomic_memset->GetOutControlAnchor()->LinkTo(atomic_node1->GetInControlAnchor()), GRAPH_SUCCESS);
  EXPECT_EQ(atomic_memset->GetOutControlAnchor()->LinkTo(atomic_node2->GetInControlAnchor()), GRAPH_SUCCESS);
  EXPECT_EQ(AttrUtils::SetBool(atomic_node0->GetOpDesc(), "is_atomic_node", true), true);
  EXPECT_EQ(AttrUtils::SetBool(atomic_node1->GetOpDesc(), "is_atomic_node", true), true);
  EXPECT_EQ(AttrUtils::SetBool(atomic_node2->GetOpDesc(), "is_atomic_node", true), true);
  std::vector<uint32_t> value;
  value.push_back(0);
  EXPECT_EQ(AttrUtils::SetListInt(atomic_node0->GetOpDesc(), "atomic_output_index", value), true);
  EXPECT_EQ(AttrUtils::SetListInt(atomic_node1->GetOpDesc(), "atomic_output_index", value), true);
  EXPECT_EQ(AttrUtils::SetListInt(atomic_node2->GetOpDesc(), "atomic_output_index", value), true);
  // init graph_mem_assigner
  MemoryOffset memory_offset(RT_MEMORY_HBM, 0UL, 0x10);
  graph_mem_assigner.memory_offset_.emplace(RT_MEMORY_HBM, memory_offset);
  // mock fe set data_type/val_int/val_float attr
  EXPECT_EQ(AttrUtils::SetListInt(atomic_node0->GetOpDesc(), TBE_OP_ATOMIC_DTYPES, {DT_INT32}), true);
  EXPECT_EQ(AttrUtils::SetListInt(atomic_node1->GetOpDesc(), TBE_OP_ATOMIC_DTYPES, {DT_FLOAT16}), true);
  EXPECT_EQ(AttrUtils::SetListInt(atomic_node2->GetOpDesc(), TBE_OP_ATOMIC_DTYPES, {DT_INT64}), true);

  EXPECT_EQ(AttrUtils::SetListInt(atomic_node0->GetOpDesc(), TBE_OP_ATOMIC_INT64_VALUES, {0}), true);
  EXPECT_EQ(AttrUtils::SetListFloat(atomic_node1->GetOpDesc(), TBE_OP_ATOMIC_FLOAT_VALUES, {0.1}), true);
  EXPECT_EQ(AttrUtils::SetListInt(atomic_node2->GetOpDesc(), TBE_OP_ATOMIC_INT64_VALUES, {0}), true);
  graph_mem_assigner.mem_assigner_.reset(new(std::nothrow) HybridMemAssigner(graph));

  EXPECT_EQ(graph_mem_assigner.ReAssignAtomicMemory(), SUCCESS);
  EXPECT_EQ(graph_mem_assigner.SetAtomicCleanOffset(), SUCCESS);
  ASSERT_EQ(GetAtomicMemSizeList(atomic_memset).size(), 3);
  ASSERT_EQ(GetAtomicAddrList(atomic_memset).size(), 3);
  std::vector<int64_t> expect_memory_size{MEM_ALIGN_SIZE, MEM_ALIGN_SIZE, MEM_ALIGN_SIZE};
  std::vector<int64_t> expect_memory_start{0, MEM_ALIGN_SIZE, MEM_ALIGN_SIZE * 2};
  EXPECT_EQ(GetAtomicMemSizeList(atomic_memset), expect_memory_size);
  EXPECT_EQ(GetAtomicAddrList(atomic_memset), expect_memory_start);
  EXPECT_EQ(GetMemsetDataTypeList(atomic_memset).size(), 3);
  EXPECT_EQ(GetMemsetFloatValList(atomic_memset).size(), 1);
  EXPECT_EQ(GetMemsetIntValList(atomic_memset).size(), 2);
}

TEST_F(UtestGraphMemAssigner, ReAssignAtomicMemoryWithOutMergeNodesMultiAttrs) {
  DEF_GRAPH(ge_graph) {
    auto atomic_memset = OP_CFG(MEMSET)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto node1 = OP_CFG("AtomicNode")
        .InCnt(1)
        .OutCnt(2)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});
    auto node2 = OP_CFG("AtomicNode")
        .InCnt(1)
        .OutCnt(2)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});
    auto node3 = OP_CFG("AtomicNode")
        .InCnt(1)
        .OutCnt(2)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {-1});

    CHAIN(NODE("atomic_node0", node1)->NODE("Node_Output", net_output));
    CHAIN(NODE("atomic_node0", node1)->NODE("Node_Output", net_output));
    CHAIN(NODE("atomic_node1", node2)->NODE("Node_Output", net_output));
    CHAIN(NODE("atomic_node1", node2)->NODE("Node_Output", net_output));
    CHAIN(NODE("atomic_node2", node3)->NODE("Node_Output", net_output));
    CHAIN(NODE("atomic_node2", node3)->NODE("Node_Output", net_output));
    CHAIN(NODE("atomic_memset", atomic_memset)->NODE("Node_Output", net_output));
  };
  auto graph = ToComputeGraph(ge_graph);
  UpdateGraphTensorSize(graph);
  GraphMemoryAssigner graph_mem_assigner(graph);
  auto atomic_node0 = graph->FindNode("atomic_node0");
  auto atomic_node1 = graph->FindNode("atomic_node1");
  auto atomic_node2 = graph->FindNode("atomic_node2");
  auto atomic_memset = graph->FindNode("atomic_memset");
  atomic_node0->GetOpDesc()->SetOutputOffset({0, 16});
  atomic_node1->GetOpDesc()->SetOutputOffset({32,48});
  atomic_node2->GetOpDesc()->SetOutputOffset({64, 80});

  EXPECT_EQ(atomic_memset->GetOutControlAnchor()->LinkTo(atomic_node0->GetInControlAnchor()), GRAPH_SUCCESS);
  EXPECT_EQ(atomic_memset->GetOutControlAnchor()->LinkTo(atomic_node1->GetInControlAnchor()), GRAPH_SUCCESS);
  EXPECT_EQ(atomic_memset->GetOutControlAnchor()->LinkTo(atomic_node2->GetInControlAnchor()), GRAPH_SUCCESS);
  EXPECT_EQ(AttrUtils::SetBool(atomic_node0->GetOpDesc(), ATOMIC_ATTR_IS_ATOMIC_NODE, true), true);
  EXPECT_EQ(AttrUtils::SetBool(atomic_node1->GetOpDesc(), ATOMIC_ATTR_IS_ATOMIC_NODE, true), true);
  EXPECT_EQ(AttrUtils::SetBool(atomic_node2->GetOpDesc(), ATOMIC_ATTR_IS_ATOMIC_NODE, true), true);
  std::vector<uint32_t> value{0,1};
  EXPECT_EQ(AttrUtils::SetListInt(atomic_node0->GetOpDesc(), ATOMIC_ATTR_OUTPUT_INDEX, value), true);
  EXPECT_EQ(AttrUtils::SetListInt(atomic_node1->GetOpDesc(), ATOMIC_ATTR_OUTPUT_INDEX, value), true);
  EXPECT_EQ(AttrUtils::SetListInt(atomic_node2->GetOpDesc(), ATOMIC_ATTR_OUTPUT_INDEX, value), true);
  // init graph_mem_assigner
  MemoryOffset memory_offset(RT_MEMORY_HBM, 0UL, 0x10);
  graph_mem_assigner.memory_offset_.emplace(RT_MEMORY_HBM, memory_offset);
  // mock fe set data_type/val_int/val_float attr
  EXPECT_EQ(AttrUtils::SetListInt(atomic_node0->GetOpDesc(), TBE_OP_ATOMIC_DTYPES, {DT_INT32, DT_INT32}), true);
  EXPECT_EQ(AttrUtils::SetListInt(atomic_node1->GetOpDesc(), TBE_OP_ATOMIC_DTYPES, {DT_FLOAT16, DT_FLOAT16}), true);
  EXPECT_EQ(AttrUtils::SetListInt(atomic_node2->GetOpDesc(), TBE_OP_ATOMIC_DTYPES, {DT_INT64, DT_INT64}), true);

  EXPECT_EQ(AttrUtils::SetListInt(atomic_node0->GetOpDesc(), TBE_OP_ATOMIC_INT64_VALUES, {0, 0}), true);
  EXPECT_EQ(AttrUtils::SetListFloat(atomic_node1->GetOpDesc(), TBE_OP_ATOMIC_FLOAT_VALUES, {0.1, 0.1}), true);
  EXPECT_EQ(AttrUtils::SetListInt(atomic_node2->GetOpDesc(), TBE_OP_ATOMIC_INT64_VALUES, {0, 0}), true);
  graph_mem_assigner.mem_assigner_.reset(new(std::nothrow) HybridMemAssigner(graph));

  EXPECT_EQ(graph_mem_assigner.ReAssignAtomicMemory(), SUCCESS);
  EXPECT_EQ(graph_mem_assigner.SetAtomicCleanOffset(), SUCCESS);
  ASSERT_EQ(GetAtomicMemSizeList(atomic_memset).size(), 3);
  ASSERT_EQ(GetAtomicAddrList(atomic_memset).size(), 3);
  std::vector<int64_t> expect_memory_size{MEM_ALIGN_SIZE * 2, MEM_ALIGN_SIZE * 2, MEM_ALIGN_SIZE * 2};
  std::vector<int64_t> expect_memory_start{0, MEM_ALIGN_SIZE * 2, MEM_ALIGN_SIZE * 4};
  EXPECT_EQ(GetAtomicMemSizeList(atomic_memset), expect_memory_size);
  EXPECT_EQ(GetAtomicAddrList(atomic_memset), expect_memory_start);
  EXPECT_EQ(GetMemsetDataTypeList(atomic_memset).size(), 3);
  EXPECT_EQ(GetMemsetFloatValList(atomic_memset).size(), 1);
  EXPECT_EQ(GetMemsetIntValList(atomic_memset).size(), 2);
}

TEST_F(UtestGraphMemAssigner, CheckAtomicNodeIsSupportRef) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node1 = UtAddNode(graph, "atom1", ATOMICADDRCLEAN, 1, 1);
  auto node2 = UtAddNode(graph, "atom2", ATOMICADDRCLEAN, 1, 1);
  EXPECT_EQ(node1->GetOutControlAnchor()->LinkTo(node2->GetInControlAnchor()), GRAPH_SUCCESS);
  AttrUtils::SetBool(node2->GetOpDesc(), "is_atomic_node", true);
  std::vector<uint32_t> value;
  value.push_back(0);
  AttrUtils::SetListInt(node2->GetOpDesc(), "atomic_output_index", value);
  EXPECT_EQ(node2->GetOpDesc()->GetOutputsSize(), 1);
  AttrUtils::SetBool(node2->GetOpDesc(), "is_atomic_node", true);
  AttrUtils::SetStr(node2->GetOpDesc(), "_batch_label", "batch_label");
  std::vector<int32_t> is_connecting_output;
  AttrUtils::SetListInt(node2->GetOpDesc(), "_is_connected_to_netoutput", is_connecting_output);
  GraphMemoryAssigner graph_mem_assigner(graph);
  graph_mem_assigner.memory_offset_.insert(std::make_pair<int64_t, MemoryOffset>(2, MemoryOffset(1, 0)));
  EXPECT_EQ(graph_mem_assigner.CheckAtomicNodeIsSupportRef(node2), true);
}

TEST_F(UtestGraphMemAssigner, ReAssignAtomicMemoryConnectToNetoutput) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node1 = UtAddNode(graph, "atom1", ATOMICADDRCLEAN, 1, 1);
  auto node2 = UtAddNode(graph, "atom2", ATOMICADDRCLEAN, 1, 1);
  EXPECT_EQ(node1->GetOutControlAnchor()->LinkTo(node2->GetInControlAnchor()), GRAPH_SUCCESS);
  AttrUtils::SetBool(node2->GetOpDesc(), "is_atomic_node", true);
  std::vector<uint32_t> value{0};
  AttrUtils::SetListInt(node2->GetOpDesc(), "atomic_output_index", value);
  node2->GetOpDesc()->SetOutputOffset({0});
  AttrUtils::SetBool(node2->GetOpDesc(), "is_atomic_node", true);
  AttrUtils::SetStr(node2->GetOpDesc(), "_batch_label", "batch_label");
  std::vector<int32_t> is_connecting_output;
  AttrUtils::SetListInt(node2->GetOpDesc(), "_is_connected_to_netoutput", is_connecting_output);
  GraphMemoryAssigner graph_mem_assigner(graph);
  graph_mem_assigner.mem_assigner_.reset(new(std::nothrow) HybridMemAssigner(graph));
  graph_mem_assigner.memory_offset_.insert(std::make_pair<int64_t, MemoryOffset>(2, MemoryOffset(1, 0)));
  graph_mem_assigner.memory_offset_.insert(std::make_pair<int64_t, MemoryOffset>(17, MemoryOffset(17, 2000)));
  EXPECT_EQ(graph_mem_assigner.CheckAtomicNodeIsSupportRef(node2), true);
  EXPECT_EQ(graph_mem_assigner.ReAssignAtomicMemory(), SUCCESS);
  is_connecting_output.push_back(1);
  AttrUtils::SetListInt(node2->GetOpDesc(), "_is_connected_to_netoutput", is_connecting_output);
  EXPECT_EQ(graph_mem_assigner.ReAssignAtomicMemory(), SUCCESS);
}

TEST_F(UtestGraphMemAssigner, ReAssignMemoryFailed) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = UtAddNode(graph, "data", DATA, 1, 1);
  GraphMemoryAssigner graph_mem_assigner(graph);
  graph_mem_assigner.memory_offset_.clear();
  std::map<uint64_t, size_t> mem_type_to_offset;
  EXPECT_EQ(graph_mem_assigner.ReAssignMemory(mem_type_to_offset), FAILED);
}

TEST_F(UtestGraphMemAssigner, AssignReferenceMemory) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = UtAddNode(graph, "data", DATA, 1, 1);
  AttrUtils::SetBool(node->GetOpDesc(), ATTR_NAME_REFERENCE, true);
  GraphMemoryAssigner graph_mem_assigner(graph);
  EXPECT_EQ(graph_mem_assigner.AssignReferenceMemory(), SUCCESS);
}

TEST_F(UtestGraphMemAssigner, AssignReferenceMemoryOutput) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = UtAddNode(graph, "data", DATA, 1, 1);
  std::vector<int64_t> output_list;
  output_list.push_back(1);
  output_list.push_back(2);
  node->GetOpDesc()->SetOutputOffset(output_list);
  GraphMemoryAssigner graph_mem_assigner(graph);
  EXPECT_EQ(graph_mem_assigner.AssignReferenceMemory(), SUCCESS);
}

TEST_F(UtestGraphMemAssigner, AssignReferenceMemoryOutputGood) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = UtAddNode(graph, "data", DATA, 1, 1);
  AttrUtils::SetBool(node->GetOpDesc(), ATTR_NAME_REFERENCE, true);
  std::vector<int64_t> output_list;
  output_list.push_back(1);
  output_list.push_back(2);
  node->GetOpDesc()->SetOutputOffset(output_list);
  GraphMemoryAssigner graph_mem_assigner(graph);
  EXPECT_EQ(graph_mem_assigner.AssignReferenceMemory(), SUCCESS);
}

TEST_F(UtestGraphMemAssigner, GetMemoryAssignmentStatus) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = UtAddNode(graph, "data", DATA, 1, 1);
  auto node2 = UtAddNode(graph, "data2", DATA, 1, 1);
  EXPECT_EQ(node->GetOutDataAnchor(0)->LinkTo(node2->GetInDataAnchor(0)), GRAPH_SUCCESS);
  int64_t output_index = 10;
  bool is_mem_assigned = true;
  GraphMemoryAssigner graph_mem_assigner(graph);
  EXPECT_EQ(graph_mem_assigner.GetMemoryAssignmentStatus(node, output_index, is_mem_assigned), PARAM_INVALID);
  output_index = 0;
  EXPECT_EQ(graph_mem_assigner.GetMemoryAssignmentStatus(node, output_index, is_mem_assigned), SUCCESS);
}

TEST_F(UtestGraphMemAssigner, CheckContinuousMemTypeAll) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  std::vector<int64_t> mem_type_list;
  mem_type_list.push_back(1);
  GraphMemoryAssigner graph_mem_assigner(graph);
  EXPECT_EQ(graph_mem_assigner.CheckContinuousMemType(mem_type_list), false);
  graph_mem_assigner.memory_offset_.insert(std::make_pair<int64_t, MemoryOffset>(1, MemoryOffset(1, 0)));
  EXPECT_EQ(graph_mem_assigner.CheckContinuousMemType(mem_type_list), true);
  mem_type_list.push_back(2);
  EXPECT_EQ(graph_mem_assigner.CheckContinuousMemType(mem_type_list), false);
}

TEST_F(UtestGraphMemAssigner, AssignAtomicOutputMemory) {
  std::vector<int64_t> mem_offset_end;
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = UtAddNode(graph, "data", DATA, 1, 1);
  std::vector<int64_t> output_list;
  output_list.push_back(1);
  node->GetOpDesc()->SetOutputOffset(output_list);
  std::vector<int64_t> atomic_output_index;
  atomic_output_index.push_back(1);
  ge::AttrUtils::SetListInt(node->GetOpDesc(), "atomic_output_index", atomic_output_index);
  GraphMemoryAssigner graph_mem_assigner(graph);
  std::vector<int64_t> real_atomic_sizes;

  graph_mem_assigner.mem_assigner_.reset(new(std::nothrow) HybridMemAssigner(graph));
  std::map<int64_t, std::vector<int64_t>> mem_type_to_offset_end;
  std::map<int64_t, std::vector<int64_t>> mem_type_to_real_atomic_sizes;
  EXPECT_EQ(graph_mem_assigner.AssignAtomicOutputMemory(node, mem_type_to_offset_end, mem_type_to_real_atomic_sizes),
            PARAM_INVALID);
  atomic_output_index.push_back(2);
  ge::AttrUtils::SetListInt(node->GetOpDesc(), "atomic_output_index", atomic_output_index);
  mem_type_to_offset_end.clear();
  mem_type_to_real_atomic_sizes.clear();
  EXPECT_EQ(graph_mem_assigner.AssignAtomicOutputMemory(node, mem_type_to_offset_end, mem_type_to_real_atomic_sizes),
            FAILED);
}

TEST_F(UtestGraphMemAssigner, AssignAtomicOutputMemoryParamInvalid) {
  std::map<int64_t, std::vector<int64_t>> mem_type_to_offset_end;
  std::map<int64_t, std::vector<int64_t>> mem_type_to_real_atomic_sizes;
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = UtAddNode(graph, "data", DATA, 1, 1);
  std::vector<int64_t> output_list;
  output_list.push_back(1);
  node->GetOpDesc()->SetOutputOffset(output_list);
  std::vector<int64_t> atomic_output_index;
  atomic_output_index.push_back(10);
  ge::AttrUtils::SetListInt(node->GetOpDesc(), "atomic_output_index", atomic_output_index);
  GraphMemoryAssigner graph_mem_assigner(graph);
  graph_mem_assigner.memory_offset_.insert(std::make_pair<int64_t, MemoryOffset>(2, MemoryOffset(1, 0)));
  EXPECT_EQ(graph_mem_assigner.AssignAtomicOutputMemory(node, mem_type_to_offset_end, mem_type_to_real_atomic_sizes),
            PARAM_INVALID);
}

TEST_F(UtestGraphMemAssigner, AssignOrdinaryAtomicWorkspaceMemory) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = UtAddNode(graph, "data", DATA, 1, 1);
  std::map<std::string, std::map<int64_t, int64_t>> workspace_info;
  std::map<int64_t, std::vector<int64_t>> mem_type_to_offset_end;
  std::map<int64_t, std::vector<int64_t>> mem_type_to_real_atomic_sizes;
  GraphMemoryAssigner graph_mem_assigner(graph);
  graph_mem_assigner.memory_offset_.insert(std::make_pair<int64_t, MemoryOffset>(2, MemoryOffset(1, 0)));
  workspace_info["noname"] = std::map<int64_t, int64_t>();
  EXPECT_EQ(graph_mem_assigner.AssignOrdinaryAtomicWorkspaceMemory(
                node->GetOpDesc(), workspace_info, mem_type_to_offset_end, mem_type_to_real_atomic_sizes),
            PARAM_INVALID);
  graph_mem_assigner.memory_offset_.clear();
  mem_type_to_offset_end.clear();
  mem_type_to_real_atomic_sizes.clear();
  EXPECT_EQ(graph_mem_assigner.AssignOrdinaryAtomicWorkspaceMemory(
                node->GetOpDesc(), workspace_info, mem_type_to_offset_end, mem_type_to_real_atomic_sizes),
            FAILED);
}

TEST_F(UtestGraphMemAssigner, AssignFusionAtomicWorkspaceMemory1) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = UtAddNode(graph, "data", DATA, 1, 1);
  std::map<std::string, std::map<int64_t, int64_t>> workspace_info;
  std::map<int64_t, int64_t> tmp{{1, 512}};
  workspace_info.insert(std::make_pair("tmp", tmp));
  std::map<int64_t, std::vector<int64_t>> mem_type_to_offset_end;
  std::map<int64_t, std::vector<int64_t>> mem_type_to_real_atomic_sizes;
  GraphMemoryAssigner graph_mem_assigner(graph);
  graph_mem_assigner.memory_offset_.clear();
  graph_mem_assigner.memory_offset_.insert(std::make_pair<int64_t, MemoryOffset>(RT_MEMORY_HBM, MemoryOffset(1, 1024)));
  EXPECT_EQ(graph_mem_assigner.AssignFusionAtomicWorkspaceMemory(node->GetOpDesc(), workspace_info,
                                                                 mem_type_to_offset_end, mem_type_to_real_atomic_sizes),
            SUCCESS);
  for (auto size : mem_type_to_real_atomic_sizes[RT_MEMORY_HBM]) {
    EXPECT_EQ(size % 512, 0);
  }
}

TEST_F(UtestGraphMemAssigner, AssignFusionAtomicWorkspaceMemory) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = UtAddNode(graph, "data", DATA, 1, 1);
  std::map<std::string, std::map<int64_t, int64_t>> workspace_info;
  std::map<int64_t, std::vector<int64_t>> mem_type_to_offset_end;
  std::map<int64_t, std::vector<int64_t>> mem_type_to_real_atomic_sizes;
  GraphMemoryAssigner graph_mem_assigner(graph);
  EXPECT_EQ(graph_mem_assigner.AssignFusionAtomicWorkspaceMemory(node->GetOpDesc(), workspace_info,
                                                                 mem_type_to_offset_end, mem_type_to_real_atomic_sizes),
            FAILED);
}

TEST_F(UtestGraphMemAssigner, SetInputOffset) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = UtAddNode(graph, "data", DATA, 1, 1);
  GraphMemoryAssigner graph_mem_assigner(graph);
  graph_mem_assigner.memory_offset_.insert(std::make_pair<int64_t, MemoryOffset>(RT_MEMORY_HBM, MemoryOffset(1, 1000)));
  EXPECT_EQ(graph_mem_assigner.SetInputOffset(), SUCCESS);
}

TEST_F(UtestGraphMemAssigner, UpdateOpInputOffset) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = UtAddNode(graph, "hcom", HCOMBROADCAST , 3, 1);
  auto node1 = UtAddNode(graph, "hcom1", HCOMBROADCAST , 1, 1);
  auto node2 = UtAddNode(graph, "var2", VARIABLE , 1, 1);
  EXPECT_EQ(node->GetInDataAnchor(0)->LinkFrom(node1->GetOutDataAnchor(0)), GRAPH_SUCCESS);
  EXPECT_EQ(node->GetInDataAnchor(1)->LinkFrom(node2->GetOutDataAnchor(0)), GRAPH_SUCCESS);
  GraphMemoryAssigner graph_mem_assigner(graph);
  EXPECT_EQ(graph_mem_assigner.UpdateOpInputOffset(node), SUCCESS);
}

TEST_F(UtestGraphMemAssigner, CalculateScaleTensorRealSize) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  std::vector<int64_t> shape = {};
  tensor_desc->SetShape(GeShape(shape));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);
  tensor_desc->SetOriginFormat(FORMAT_NCHW);
  tensor_desc->SetOriginShape(GeShape(shape));
  tensor_desc->SetOriginDataType(DT_FLOAT);
  int64_t output_mem_size = 0;
  int64_t batch_dim_num = 0;
  int64_t out_size = 0;
  EXPECT_EQ(CalculateTensorRealSizeAndOutSize(tensor_desc, 0, output_mem_size, batch_dim_num, out_size), SUCCESS);
}

TEST_F(UtestGraphMemAssigner, UpdateOpInputOffset_with_cut_const) {
  ge::ComputeGraphPtr root_graph = std::make_shared<ge::ComputeGraph>("root_graph");
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("sub_graph");
  root_graph->AddSubgraph(graph);

  auto const_0 = UtAddNode(root_graph, "const_0", CONSTANT , 0, 1);
  const_0->GetOpDescBarePtr()->SetOutputOffset({1024});
  auto partitioned_call = UtAddNode(root_graph, "partitioned_call", PARTITIONEDCALL , 1, 1);
  partitioned_call->GetOpDesc()->SetSubgraphInstanceName(0, graph->GetName());
  EXPECT_EQ(partitioned_call->GetInDataAnchor(0)->LinkFrom(const_0->GetOutDataAnchor(0)), GRAPH_SUCCESS);
  graph->SetParentGraph(root_graph);
  graph->SetParentNode(partitioned_call);

  auto data_0 = UtAddNode(graph, "data_0", DATA , 0, 1);
  data_0->GetOpDescBarePtr()->SetOutputOffset({1024});
  auto data_1 = UtAddNode(graph, "data_1", DATA , 0, 1);
  data_1->GetOpDescBarePtr()->SetOutputOffset({1024});
  auto add0 = UtAddNode(graph, "add_lxslice0", ADD, 1, 1);
  auto add1 = UtAddNode(graph, "add_lxslice1", ADD, 1, 1);

  TensorUtils::SetDataOffset(*const_0->GetOpDescBarePtr()->MutableOutputDesc(0), 1024);
  TensorUtils::SetDataOffset(*add0->GetOpDescBarePtr()->MutableInputDesc(0), 1024);
  TensorUtils::SetDataOffset(*add1->GetOpDescBarePtr()->MutableInputDesc(0), 1024);

  (void)AttrUtils::SetInt(data_0->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  (void)AttrUtils::SetInt(data_1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  (void)AttrUtils::SetInt(*data_0->GetOpDescBarePtr()->MutableOutputDesc(0), ATTR_NAME_INNER_OFFSET, 0);
  (void)AttrUtils::SetInt(*data_1->GetOpDescBarePtr()->MutableOutputDesc(0), ATTR_NAME_INNER_OFFSET, 1024);

  EXPECT_EQ(add0->GetInDataAnchor(0)->LinkFrom(data_0->GetOutDataAnchor(0)), GRAPH_SUCCESS);
  EXPECT_EQ(add1->GetInDataAnchor(0)->LinkFrom(data_1->GetOutDataAnchor(0)), GRAPH_SUCCESS);
  GraphMemoryAssigner graph_mem_assigner(root_graph);
  EXPECT_EQ(graph_mem_assigner.UpdateOpInputOffset(add0), SUCCESS);
  EXPECT_EQ(graph_mem_assigner.UpdateOpInputOffset(add1), SUCCESS);

  int64_t data_offset{0L};
  TensorUtils::GetDataOffset(const_0->GetOpDescBarePtr()->GetOutputDesc(0), data_offset);
  EXPECT_EQ(data_offset, 1024);
  int64_t data_offset_add0{0L};
  TensorUtils::GetDataOffset(add0->GetOpDescBarePtr()->GetInputDesc(0), data_offset_add0);
  EXPECT_EQ(data_offset_add0, 1024);
  int64_t data_offset_add1{0L};
  TensorUtils::GetDataOffset(add1->GetOpDescBarePtr()->GetInputDesc(0), data_offset_add1);
  EXPECT_EQ(data_offset_add1, 2048);
}

TEST_F(UtestGraphMemAssigner, UpdateDataOutputOffset_with_nested_static_subgraph) {
  ge::ComputeGraphPtr root_graph = std::make_shared<ge::ComputeGraph>("root_graph");
  auto dynamic_const = UtAddNode(root_graph, "dynamic_const", CONSTANT , 0, 1);
  dynamic_const->GetOpDescBarePtr()->SetOutputOffset({0});
  root_graph->SetGraphUnknownFlag(true);
  auto partitioned_call_1 = UtAddNode(root_graph, "partitioned_call1", PARTITIONEDCALL , 1, 1);
  EXPECT_EQ(partitioned_call_1->GetInDataAnchor(0)->LinkFrom(dynamic_const->GetOutDataAnchor(0)), GRAPH_SUCCESS);

  ge::ComputeGraphPtr static_sub_graph1 = std::make_shared<ge::ComputeGraph>("sub_graph_1");
  partitioned_call_1->GetOpDesc()->AddSubgraphName(static_sub_graph1->GetName());
  partitioned_call_1->GetOpDesc()->SetSubgraphInstanceName(0, static_sub_graph1->GetName());
  static_sub_graph1->SetParentGraph(root_graph);
  static_sub_graph1->SetParentNode(partitioned_call_1);
  root_graph->AddSubgraph(static_sub_graph1);

  auto data_0 = UtAddNode(static_sub_graph1, "data_0", DATA , 0, 1);
  data_0->GetOpDescBarePtr()->SetOutputOffset({1024});
  TensorUtils::SetDataOffset(*dynamic_const->GetOpDescBarePtr()->MutableOutputDesc(0), 0);

  (void)AttrUtils::SetInt(data_0->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  (void)AttrUtils::SetInt(*data_0->GetOpDescBarePtr()->MutableOutputDesc(0), ATTR_NAME_INNER_OFFSET, 0);

  auto partitioned_call_2 = UtAddNode(static_sub_graph1, "partitioned_call2", PARTITIONEDCALL , 1, 1);
  EXPECT_EQ(partitioned_call_2->GetInDataAnchor(0)->LinkFrom(data_0->GetOutDataAnchor(0)), GRAPH_SUCCESS);
  ge::ComputeGraphPtr static_sub_graph2 = std::make_shared<ge::ComputeGraph>("sub_graph_2");
  partitioned_call_2->GetOpDesc()->AddSubgraphName(static_sub_graph2->GetName());
  partitioned_call_2->GetOpDesc()->SetSubgraphInstanceName(0, static_sub_graph2->GetName());
  static_sub_graph2->SetParentGraph(static_sub_graph1);
  static_sub_graph2->SetParentNode(partitioned_call_2);
  root_graph->AddSubgraph(static_sub_graph2);

  auto data_1 = UtAddNode(static_sub_graph2, "data_1", DATA , 0, 1);
  data_1->GetOpDescBarePtr()->SetOutputOffset({1024});
  (void)AttrUtils::SetInt(data_1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  (void)AttrUtils::SetInt(*data_1->GetOpDescBarePtr()->MutableOutputDesc(0), ATTR_NAME_INNER_OFFSET, 0);

  GraphMemoryAssigner graph_mem_assigner(static_sub_graph2);
  EXPECT_EQ(graph_mem_assigner.UpdateOpInputOffset(data_1), SUCCESS);


  int64_t const_offset{0L};
  TensorUtils::GetDataOffset(dynamic_const->GetOpDescBarePtr()->GetOutputDesc(0), const_offset);
  EXPECT_EQ(const_offset, 0);

  std::vector<int64_t> output_offset = data_1->GetOpDescBarePtr()->GetOutputOffset();
  EXPECT_TRUE(!output_offset.empty());
  EXPECT_EQ(output_offset[0], 1024);
}

TEST_F(UtestGraphMemAssigner, UpdateDataOutputOffset_with_parent_input_var) {
  ge::ComputeGraphPtr root_graph = std::make_shared<ge::ComputeGraph>("root_graph");
  auto var_0 = UtAddNode(root_graph, "var_0", VARIABLE , 0, 1);
  var_0->GetOpDescBarePtr()->SetOutputOffset({1024});
  auto partitioned_call = UtAddNode(root_graph, "partitioned_call", PARTITIONEDCALL , 1, 1);
  EXPECT_EQ(partitioned_call->GetInDataAnchor(0)->LinkFrom(var_0->GetOutDataAnchor(0)), GRAPH_SUCCESS);

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("sub_graph");
  partitioned_call->GetOpDesc()->AddSubgraphName(graph->GetName());
  partitioned_call->GetOpDesc()->SetSubgraphInstanceName(0, graph->GetName());
  graph->SetParentGraph(root_graph);
  graph->SetParentNode(partitioned_call);
  root_graph->AddSubgraph(graph);

  auto data_0 = UtAddNode(graph, "data_0", DATA , 0, 1);
  data_0->GetOpDescBarePtr()->SetOutputOffset({0}); // default as 0
  auto data_1 = UtAddNode(graph, "data_1", DATA , 0, 1);
  data_1->GetOpDescBarePtr()->SetOutputOffset({1024});
  auto add0 = UtAddNode(graph, "add_lxslice0", ADD, 1, 1);
  auto add1 = UtAddNode(graph, "add_lxslice1", ADD, 1, 1);

  TensorUtils::SetDataOffset(*var_0->GetOpDescBarePtr()->MutableOutputDesc(0), 1024);
  TensorUtils::SetDataOffset(*add0->GetOpDescBarePtr()->MutableInputDesc(0), 1024);
  TensorUtils::SetDataOffset(*add1->GetOpDescBarePtr()->MutableInputDesc(0), 1024);

  (void)AttrUtils::SetInt(data_0->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
  (void)AttrUtils::SetInt(data_1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);
  (void)AttrUtils::SetInt(*data_0->GetOpDescBarePtr()->MutableOutputDesc(0), ATTR_NAME_INNER_OFFSET, 0);
  (void)AttrUtils::SetInt(*data_1->GetOpDescBarePtr()->MutableOutputDesc(0), ATTR_NAME_INNER_OFFSET, 1024);

  EXPECT_EQ(add0->GetInDataAnchor(0)->LinkFrom(data_0->GetOutDataAnchor(0)), GRAPH_SUCCESS);
  EXPECT_EQ(add1->GetInDataAnchor(0)->LinkFrom(data_1->GetOutDataAnchor(0)), GRAPH_SUCCESS);
  GraphMemoryAssigner graph_mem_assigner(graph);
  EXPECT_EQ(graph_mem_assigner.UpdateOpInputOffset(data_0), SUCCESS);

  int64_t data_offset{0L};
  TensorUtils::GetDataOffset(var_0->GetOpDescBarePtr()->GetOutputDesc(0), data_offset);
  EXPECT_EQ(data_offset, 1024);

  std::vector<int64_t> output_offset = data_0->GetOpDescBarePtr()->GetOutputOffset();
  EXPECT_TRUE(!output_offset.empty());
  EXPECT_EQ(output_offset[0], 1024); // expect same as var
}

TEST_F(UtestGraphMemAssigner, static_graph_no_split) { // split默认打开了
  std::string memory_optimization_policy = "MemoryPriority";
  const auto old_options = ge::GetThreadLocalContext().GetAllGlobalOptions();
  auto new_options = old_options;
  new_options.insert(std::make_pair(MEMORY_OPTIMIZATION_POLICY, memory_optimization_policy));
  ge::GetThreadLocalContext().SetGlobalOption(new_options);
  auto hcom1 = OP_CFG(HCOMALLREDUCE).InCnt(2).OutCnt(2).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1024, 1024, 224});
  auto hcom2 = OP_CFG(HCOMALLREDUCE).InCnt(2).OutCnt(2).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1024, 1024, 224})
      .Attr(ATTR_NAME_CONTINUOUS_INPUT, true);

  auto data1 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 50, 1024, 1024});
  auto data2 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 64, 1024, 1024});
  auto data3 = OP_CFG("Print");
  auto data4 = OP_CFG("Print");

  DEF_GRAPH(g1) {
      CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("hcom_1", hcom1));
      CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("hcom_1", hcom1));
      CHAIN(NODE("hcom_1", hcom1)->EDGE(0, 0)->NODE("hcom_2", hcom2));
      CHAIN(NODE("data_1", data1)->EDGE(0, 1)->NODE("hcom_2", hcom2));
      CHAIN(NODE("hcom_2", hcom2)->EDGE(0, 0)->NODE("data_3", data3));
      CHAIN(NODE("hcom_2", hcom2)->EDGE(1, 0)->NODE("data_4", data4));
  };

  auto graph = ToComputeGraph(g1);
  UpdateGraphTensorSize(graph);
  graph->TopologicalSorting();

  std::map<std::string, std::string> option;
  Graph2SubGraphInfoList subgraphs;
  std::map<std::string, int> stream_max_parallel_num;
  ge::ModelBuilder builder(0, graph, subgraphs, stream_max_parallel_num, false);
  MemoryAssigner mem_assigner(graph);
  EXPECT_EQ(mem_assigner.AssignMemory(builder.mem_type_to_mem_offset_, builder.zero_copy_mem_size_), SUCCESS);
  builder.sub_mem_offsets_ = mem_assigner.GetSubMemOffsets();
  ge::Model model;
  EXPECT_EQ(builder.BuildModelDefForMem(model), SUCCESS);
  std::vector<std::vector<int64_t>> sub_mem_offsets;
  ge::AttrUtils::GetListListInt(&model, ATTR_MODEL_SUB_MEMORY_INFO, sub_mem_offsets);
  EXPECT_EQ(sub_mem_offsets.size(), 4U);
  ge::GetThreadLocalContext().SetGlobalOption(old_options);
}

TEST_F(UtestGraphMemAssigner, small_static_sub_graph_no_split) { // 静态子图，小于阀值，不分段
  std::string memory_optimization_policy = "MemoryPriority";
  const auto old_options = ge::GetThreadLocalContext().GetAllGlobalOptions();
  auto new_options = old_options;
  new_options.insert(std::make_pair(MEMORY_OPTIMIZATION_POLICY, memory_optimization_policy));
  ge::GetThreadLocalContext().SetGlobalOption(new_options);
  auto hcom1 = OP_CFG(HCOMALLREDUCE).InCnt(2).OutCnt(2).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1024, 1024, 1});
  auto hcom2 = OP_CFG(HCOMALLREDUCE).InCnt(2).OutCnt(2).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1024, 1024, 1})
      .Attr(ATTR_NAME_CONTINUOUS_INPUT, true);

  auto data1 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 50, 1024, 1});
  auto data2 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 64, 1024, 1});
  auto data3 = OP_CFG("Print");
  auto data4 = OP_CFG("Print");

  DEF_GRAPH(g1) {
      CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("hcom_1", hcom1));
      CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("hcom_1", hcom1));
      CHAIN(NODE("hcom_1", hcom1)->EDGE(0, 0)->NODE("hcom_2", hcom2));
      CHAIN(NODE("data_1", data1)->EDGE(0, 1)->NODE("hcom_2", hcom2));
      CHAIN(NODE("hcom_2", hcom2)->EDGE(0, 0)->NODE("data_3", data3));
      CHAIN(NODE("hcom_2", hcom2)->EDGE(1, 0)->NODE("data_4", data4));
  };

  auto graph = ToComputeGraph(g1);
  UpdateGraphTensorSize(graph);
  graph->TopologicalSorting();
  bool dynamic_shape_partition = true;
  (void) ge::AttrUtils::SetBool(graph, ge::ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, dynamic_shape_partition);

  std::map<std::string, std::string> option;
  Graph2SubGraphInfoList subgraphs;
  std::map<std::string, int> stream_max_parallel_num;
  ge::ModelBuilder builder(0, graph, subgraphs, stream_max_parallel_num, false);
  MemoryAssigner mem_assigner(graph);
  EXPECT_EQ(mem_assigner.AssignMemory(builder.mem_type_to_mem_offset_, builder.zero_copy_mem_size_), SUCCESS);
  builder.sub_mem_offsets_ = mem_assigner.GetSubMemOffsets();
  ge::Model model;
  EXPECT_EQ(builder.BuildModelDefForMem(model), SUCCESS);
  std::vector<std::vector<int64_t>> sub_mem_offsets;
  ge::AttrUtils::GetListListInt(&model, ATTR_MODEL_SUB_MEMORY_INFO, sub_mem_offsets);
  EXPECT_EQ(sub_mem_offsets.size(), 2U);
  ge::GetThreadLocalContext().SetGlobalOption(old_options);
}

TEST_F(UtestGraphMemAssigner, large_static_sub_graph_split_no_atomic) {
  std::string memory_optimization_policy = "MemoryPriority";
  const auto old_options = ge::GetThreadLocalContext().GetAllGlobalOptions();
  auto new_options = old_options;
  new_options.insert(std::make_pair(MEMORY_OPTIMIZATION_POLICY, memory_optimization_policy));
  ge::GetThreadLocalContext().SetGlobalOption(new_options);
  auto hcom1 = OP_CFG(HCOMALLREDUCE).InCnt(2).OutCnt(2).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1024, 1024, 224});
  auto hcom2 = OP_CFG(HCOMALLREDUCE).InCnt(2).OutCnt(2).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1024, 1024, 224})
      .Attr(ATTR_NAME_CONTINUOUS_INPUT, true);

  auto data1 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 50, 1024, 512});
  auto data2 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 64, 1024, 512});
  auto data3 = OP_CFG("Print");
  auto data4 = OP_CFG("Print");

  DEF_GRAPH(g1) {
      CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("hcom_1", hcom1));
      CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("hcom_1", hcom1));
      CHAIN(NODE("hcom_1", hcom1)->EDGE(0, 0)->NODE("hcom_2", hcom2));
      CHAIN(NODE("data_1", data1)->EDGE(0, 1)->NODE("hcom_2", hcom2));
      CHAIN(NODE("hcom_2", hcom2)->EDGE(0, 0)->NODE("data_3", data3));
      CHAIN(NODE("hcom_2", hcom2)->EDGE(1, 0)->NODE("data_4", data4));
  };

  auto graph = ToComputeGraph(g1);
  UpdateGraphTensorSize(graph);
  graph->TopologicalSorting();
  bool dynamic_shape_partition = true;
  (void) ge::AttrUtils::SetBool(graph, ge::ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, dynamic_shape_partition);

  std::map<std::string, std::string> option;
  Graph2SubGraphInfoList subgraphs;
  std::map<std::string, int> stream_max_parallel_num;
  ge::ModelBuilder builder(0, graph, subgraphs, stream_max_parallel_num, false);
  MemoryAssigner mem_assigner(graph);
  EXPECT_EQ(mem_assigner.AssignMemory(builder.mem_type_to_mem_offset_, builder.zero_copy_mem_size_), SUCCESS);
  builder.sub_mem_offsets_ = mem_assigner.GetSubMemOffsets();
  ge::Model model;
  EXPECT_EQ(builder.BuildModelDefForMem(model), SUCCESS);
  std::vector<std::vector<int64_t>> sub_mem_offsets;
  ge::AttrUtils::GetListListInt(&model, ATTR_MODEL_SUB_MEMORY_INFO, sub_mem_offsets);
  EXPECT_EQ(sub_mem_offsets.size(), 4U);
  int64_t total_size = 0;
  for (size_t i = 0U; i < sub_mem_offsets.size(); ++i) {
    EXPECT_GE(sub_mem_offsets[i].size(), 3U);
    total_size += sub_mem_offsets[i][2];
  }
  EXPECT_EQ(total_size, builder.mem_type_to_mem_offset_[RT_MEMORY_HBM]);
  ge::GetThreadLocalContext().SetGlobalOption(old_options);
}

TEST_F(UtestGraphMemAssigner, large_static_sub_graph_split) {
  std::string memory_optimization_policy = "MemoryPriority";
  const auto old_options = ge::GetThreadLocalContext().GetAllGlobalOptions();
  auto new_options = old_options;
  new_options.insert(std::make_pair(MEMORY_OPTIMIZATION_POLICY, memory_optimization_policy));
  ge::GetThreadLocalContext().SetGlobalOption(new_options);
  auto hcom1 = OP_CFG(HCOMALLREDUCE).InCnt(3).OutCnt(2).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1024, 1024, 32});
  auto hcom2 = OP_CFG(HCOMALLREDUCE).InCnt(3).OutCnt(2).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1024, 1024, 32})
      .Attr(ATTR_NAME_CONTINUOUS_INPUT, true);
  auto hcom3 = OP_CFG(HCOMALLREDUCE).InCnt(1).OutCnt(1).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1024, 1024, 32})
    .Attr(ATTR_NAME_CONTINUOUS_INPUT, true).Attr(ATTR_NAME_CONTINUOUS_OUTPUT, true);

  std::vector<uint32_t> value{0};
  auto cast0 = OP_CFG(CAST).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 32, 512, 512})
    .Attr(ATOMIC_ATTR_IS_ATOMIC_NODE, true);
  auto cast1 = OP_CFG(CAST).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 32, 512, 512})
    .Attr(ATOMIC_ATTR_IS_ATOMIC_NODE, true);
  auto data1 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 50, 512, 512});
  auto data2 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 32, 512, 512});
  auto data5 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 32, 512, 1024});
  auto data3 = OP_CFG("Print");
  auto data4 = OP_CFG("Print");
  auto atomic_memset = OP_CFG(MEMSET).InCnt(1).OutCnt(1).TensorDesc(FORMAT_ND, DT_INT32, {16});
  auto net_output = OP_CFG(NETOUTPUT).InCnt(1).OutCnt(0).TensorDesc(FORMAT_ND, DT_INT32, {-1});
  DEF_GRAPH(g1) {
      CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("hcom_1", hcom1));
      CHAIN(NODE("cast_0", cast0)->EDGE(0, 1)->NODE("hcom_1", hcom1));
      CHAIN(NODE("cast_1", cast1)->EDGE(0, 2)->NODE("hcom_1", hcom1));
      CHAIN(NODE("hcom_1", hcom1)->EDGE(1, 0)->NODE("hcom_3", hcom3));
      CHAIN(NODE("hcom_1", hcom1)->EDGE(0, 0)->NODE("hcom_2", hcom2));
      CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("hcom_2", hcom2));
      CHAIN(NODE("data_5", data5)->EDGE(0, 2)->NODE("hcom_2", hcom2));
      CHAIN(NODE("hcom_2", hcom2)->EDGE(0, 0)->NODE("data_3", data3));
      CHAIN(NODE("hcom_2", hcom2)->EDGE(1, 0)->NODE("data_4", data4));
      CHAIN(NODE("atomic_memset", atomic_memset)->NODE("Node_Output", net_output));
  };

  auto graph = ToComputeGraph(g1);
  auto atomic_memset_node = graph->FindNode("atomic_memset");
  auto cast0_node = graph->FindNode("cast_0");
  auto cast1_node = graph->FindNode("cast_1");
  AttrUtils::SetBool(cast0_node->GetOpDesc(), ATOMIC_ATTR_IS_ATOMIC_NODE, true);
  AttrUtils::SetBool(cast1_node->GetOpDesc(), ATOMIC_ATTR_IS_ATOMIC_NODE, true);
  EXPECT_EQ(AttrUtils::SetListInt(cast0_node->GetOpDesc(), ATOMIC_ATTR_OUTPUT_INDEX, value), true);
  EXPECT_EQ(AttrUtils::SetListInt(cast1_node->GetOpDesc(), ATOMIC_ATTR_OUTPUT_INDEX, value), true);
  EXPECT_EQ(atomic_memset_node->GetOutControlAnchor()->LinkTo(cast0_node->GetInControlAnchor()), GRAPH_SUCCESS);
  EXPECT_EQ(atomic_memset_node->GetOutControlAnchor()->LinkTo(cast1_node->GetInControlAnchor()), GRAPH_SUCCESS);

  UpdateGraphTensorSize(graph);
  graph->TopologicalSorting();
  bool dynamic_shape_partition = true;
  (void) ge::AttrUtils::SetBool(graph, ge::ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, dynamic_shape_partition);

  std::map<std::string, std::string> option;
  Graph2SubGraphInfoList subgraphs;
  std::map<std::string, int> stream_max_parallel_num;
  ge::ModelBuilder builder(0, graph, subgraphs, stream_max_parallel_num, false);
  MemoryAssigner mem_assigner(graph);

  ASSERT_EQ(mem_assigner.AssignMemory(builder.mem_type_to_mem_offset_, builder.zero_copy_mem_size_), SUCCESS);
  builder.sub_mem_offsets_ = mem_assigner.GetSubMemOffsets();
  ge::Model model;
  ASSERT_EQ(builder.BuildModelDefForMem(model), SUCCESS);
  std::vector<std::vector<int64_t>> sub_mem_offsets;
  ge::AttrUtils::GetListListInt(&model, ATTR_MODEL_SUB_MEMORY_INFO, sub_mem_offsets);
  EXPECT_FALSE(sub_mem_offsets.empty());
  int64_t total_size = 0;
  for (size_t i = 0U; i < sub_mem_offsets.size(); ++i) {
    EXPECT_GE(sub_mem_offsets[i].size(), 3U);
    total_size += sub_mem_offsets[i][2];
  }
  EXPECT_EQ(total_size, builder.mem_type_to_mem_offset_[RT_MEMORY_HBM]);
  ge::GetThreadLocalContext().SetGlobalOption(old_options);
}

TEST_F(UtestGraphMemAssigner, static_graph_split) {
  std::string memory_optimization_policy = "MemoryPriority";
  const auto old_options = ge::GetThreadLocalContext().GetAllGlobalOptions();
  auto new_options = old_options;
  new_options.insert(std::make_pair(STATIC_MEMORY_POLICY, "2"));
  new_options.insert(std::make_pair(MEMORY_OPTIMIZATION_POLICY, memory_optimization_policy));
  ge::GetThreadLocalContext().SetGlobalOption(new_options);
  auto hcom1 = OP_CFG(HCOMALLREDUCE).InCnt(2).OutCnt(2).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1024, 1024, 224});
  auto hcom2 = OP_CFG(HCOMALLREDUCE).InCnt(2).OutCnt(2).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 1024, 1024, 224})
      .Attr(ATTR_NAME_CONTINUOUS_INPUT, true);

  auto data1 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 50, 1024, 1024});
  auto data2 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_FLOAT, {1, 64, 1024, 1024});
  auto data3 = OP_CFG("Print");
  auto data4 = OP_CFG("Print");

  DEF_GRAPH(g1)
  {
    CHAIN(NODE("data_1", data1)->EDGE(0, 0)->NODE("hcom_1", hcom1));
    CHAIN(NODE("data_2", data2)->EDGE(0, 1)->NODE("hcom_1", hcom1));
    CHAIN(NODE("hcom_1", hcom1)->EDGE(0, 0)->NODE("hcom_2", hcom2));
    CHAIN(NODE("data_1", data1)->EDGE(0, 1)->NODE("hcom_2", hcom2));
    CHAIN(NODE("hcom_2", hcom2)->EDGE(0, 0)->NODE("data_3", data3));
    CHAIN(NODE("hcom_2", hcom2)->EDGE(1, 0)->NODE("data_4", data4));
  };

  auto graph = ToComputeGraph(g1);
  UpdateGraphTensorSize(graph);
  graph->TopologicalSorting();

  std::map<std::string, std::string> option;
  Graph2SubGraphInfoList subgraphs;
  std::map<std::string, int> stream_max_parallel_num;
  ge::ModelBuilder builder(0, graph, subgraphs, stream_max_parallel_num, false);
  MemoryAssigner mem_assigner(graph);
  EXPECT_EQ(mem_assigner.AssignMemory(builder.mem_type_to_mem_offset_, builder.zero_copy_mem_size_), SUCCESS);
  builder.sub_mem_offsets_ = mem_assigner.GetSubMemOffsets();
  ge::Model model;
  gert::GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().NoConsoleOut().SetLevelInfo();
  EXPECT_EQ(builder.BuildModelDefForMem(model), SUCCESS);  EXPECT_NE(runtime_stub.GetSlogStub().FindLog(DLOG_INFO, "weight_offset_:"), -1);  // logs for test case
  std::vector<std::vector<int64_t>> sub_mem_offsets;
  ge::AttrUtils::GetListListInt(&model, ATTR_MODEL_SUB_MEMORY_INFO, sub_mem_offsets);
  EXPECT_EQ(sub_mem_offsets.size(), 4U);
  ge::GetThreadLocalContext().SetGlobalOption(old_options);
}

TEST_F(UtestGraphMemAssigner, ConstConnectNoPaddingContinuousOutput_CheckOutputOffset) {
  vector<int64_t> perm1{0, 3, 1, 2};
  GeTensorDesc tensor_desc1(GeShape(vector<int64_t>{4}));
  GeTensorPtr const_tensor1 =
      std::make_shared<GeTensor>(tensor_desc1, reinterpret_cast<uint8_t *>(perm1.data()) , sizeof(int64_t)*perm1.size());
  auto const1 = OP_CFG(CONSTANTOP).Weight(const_tensor1);
  auto const2 = OP_CFG(CONSTANTOP).Weight(const_tensor1);
  DEF_GRAPH(g1) {
                  CHAIN(NODE("const2", const2)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("const1", const1)->NODE("split", PHONYSPLIT)->NODE("a", RELU)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("split", PHONYSPLIT)->NODE("b", RELU)->NODE("netoutput", NETOUTPUT));
                };

  auto graph = ToComputeGraph(g1);
  auto split_node = graph->FindNode("split");
  ASSERT_NE(split_node, nullptr);
  (void)ge::AttrUtils::SetBool(split_node->GetOpDesc(), ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT, true);
  (void)ge::AttrUtils::SetBool(split_node->GetOpDesc(), ATTR_NAME_OUTPUT_REUSE_INPUT, true);
  (void)ge::AttrUtils::SetInt(split_node->GetOpDesc(), ge::ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);
  split_node->GetOpDescBarePtr()->UpdateOutputDesc(0, tensor_desc1);
  split_node->GetOpDescBarePtr()->UpdateOutputDesc(1, tensor_desc1);
  UpdateGraphTensorSize(graph);
  graph->TopologicalSorting();

  std::map<uint64_t, size_t> mem_type_to_mem_offset;
  size_t zero_copy_mem_size;
  VarManager::Instance(graph->GetSessionID())->Init(0, graph->GetSessionID(), 0, 0);
  MemoryAssigner mem_assigner(graph);
  EXPECT_EQ(mem_assigner.AssignMemory(mem_type_to_mem_offset, zero_copy_mem_size), SUCCESS);

  auto op_desc = split_node->GetOpDesc();
  auto split_output_offset = op_desc->GetOutputOffset();
  ASSERT_EQ(split_output_offset.size(), 2U);

  auto split_input_offset = op_desc->GetInputOffset();
  ASSERT_EQ(split_input_offset.size(), 1U);

  EXPECT_EQ(split_input_offset[0], split_output_offset[0]);

  int64_t output_0_size = 0;
  TensorUtils::GetTensorSizeInBytes(op_desc->GetOutputDesc(0), output_0_size);
  EXPECT_EQ(split_output_offset[1], split_output_offset[0] + output_0_size);
}

TEST_F(UtestGraphMemAssigner, ConstConnectPhonySplit_CheckOutputOffset) {
  vector<int64_t> perm1{0, 3, 1, 2};
  GeTensorDesc tensor_desc1(GeShape(vector<int64_t>{4}));
  GeTensorPtr const_tensor1 =
      std::make_shared<GeTensor>(tensor_desc1, reinterpret_cast<uint8_t *>(perm1.data()) , sizeof(int64_t)*perm1.size());
  auto const1 = OP_CFG(CONSTANTOP).Weight(const_tensor1);
  auto const2 = OP_CFG(CONSTANTOP).Weight(const_tensor1);
  DEF_GRAPH(g1) {
                  CHAIN(NODE("const2", const2)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("const1", const1)->NODE("split", PHONYSPLIT)->NODE("a", RELU)->NODE("netoutput", NETOUTPUT));
                  CHAIN(NODE("split", PHONYSPLIT)->NODE("b", RELU)->NODE("netoutput", NETOUTPUT));
                };

  auto graph = ToComputeGraph(g1);
  auto split_node = graph->FindNode("split");
  ASSERT_NE(split_node, nullptr);
  (void)ge::AttrUtils::SetBool(split_node->GetOpDesc(), ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT, true);
  (void)ge::AttrUtils::SetBool(split_node->GetOpDesc(), ATTR_NAME_OUTPUT_REUSE_INPUT, true);
  (void)ge::AttrUtils::SetInt(split_node->GetOpDesc(), ge::ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);

  split_node->GetOpDescBarePtr()->MutableOutputDesc(0)->SetShape(GeShape(vector<int64_t>{2}));
  split_node->GetOpDescBarePtr()->MutableOutputDesc(1)->SetShape(GeShape(vector<int64_t>{2}));
  split_node->GetOpDescBarePtr()->MutableInputDesc(0)->SetShape(GeShape(vector<int64_t>{4}));

  UpdateGraphTensorSize(graph);
  graph->TopologicalSorting();

  int64_t output_0_size = 0;
  TensorUtils::GetTensorSizeInBytes(split_node->GetOpDesc()->GetOutputDesc(0), output_0_size);

  auto a_node = graph->FindNode("a");
  ASSERT_NE(a_node, nullptr);
  auto b_node = graph->FindNode("b");
  ASSERT_NE(b_node, nullptr);
  std::vector<int64_t> a_in_offset_list = {0};
  ge::AttrUtils::SetListInt(a_node->GetOpDesc(), ATTR_NAME_INPUT_OFFSET_LIST_FOR_CONTINUOUS, a_in_offset_list);
  std::vector<int64_t> b_in_offset_list = {output_0_size};
  ge::AttrUtils::SetListInt(b_node->GetOpDesc(), ATTR_NAME_INPUT_OFFSET_LIST_FOR_CONTINUOUS, b_in_offset_list);

  std::map<uint64_t, size_t> mem_type_to_mem_offset;
  size_t zero_copy_mem_size;
  VarManager::Instance(graph->GetSessionID())->Init(0, graph->GetSessionID(), 0, 0);
  MemoryAssigner mem_assigner(graph);
  EXPECT_EQ(mem_assigner.AssignMemory(mem_type_to_mem_offset, zero_copy_mem_size), SUCCESS);

  auto op_desc = split_node->GetOpDesc();
  auto split_output_offset = op_desc->GetOutputOffset();
  ASSERT_EQ(split_output_offset.size(), 2U);

  auto split_input_offset = op_desc->GetInputOffset();
  ASSERT_EQ(split_input_offset.size(), 1U);

  EXPECT_EQ(split_input_offset[0], split_output_offset[0]);
  EXPECT_EQ(split_output_offset[1], split_output_offset[0] + output_0_size);
}
TEST_F(UtestGraphMemAssigner, PhonyConcatConnectToHcomBroadCast_CheckOffsetSuccess) {
  auto hcombroadcast = OP_CFG(HCOMBROADCAST).Attr(ATTR_NAME_REFERENCE, true).InNames({"x", "y"}).OutNames({"x", "y"});
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", CAST)->NODE("hcombroadcast", hcombroadcast)->NODE("d", ADD));
                  CHAIN(NODE("b", CAST)->NODE("c", CAST)->NODE("pc", PHONYCONCAT)->NODE("hcombroadcast", hcombroadcast)->NODE("d", ADD)->NODE("netoutput", NETOUTPUT));
                };

  auto graph = ToComputeGraph(g1);
  auto pc_node = graph->FindNode("pc");
  ASSERT_NE(pc_node, nullptr);
  (void)ge::AttrUtils::SetBool(pc_node->GetOpDesc(), ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
  (void)ge::AttrUtils::SetBool(pc_node->GetOpDesc(), ATTR_NAME_OUTPUT_REUSE_INPUT, true);
  (void)ge::AttrUtils::SetInt(pc_node->GetOpDesc(), ge::ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 0);

  UpdateGraphTensorSize(graph);
  graph->TopologicalSorting();

  std::map<uint64_t, size_t> mem_type_to_mem_offset;
  size_t zero_copy_mem_size;
  VarManager::Instance(graph->GetSessionID())->Init(0, graph->GetSessionID(), 0, 0);
  MemoryAssigner mem_assigner(graph);
  EXPECT_EQ(mem_assigner.AssignMemory(mem_type_to_mem_offset, zero_copy_mem_size), SUCCESS);

  auto a_node = graph->FindNode("a");
  EXPECT_FALSE(MemReuseUtils::IsAllOutRefAllInput(a_node));
}
} // namespace ge

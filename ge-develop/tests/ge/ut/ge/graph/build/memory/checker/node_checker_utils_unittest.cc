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

#include "graph/build/memory/checker/node_checker_utils.h"
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
namespace {
void SetSizeAndOffset(ComputeGraphPtr graph) {
  for (auto &node : graph->GetAllNodes()) {
    auto input_size = node->GetOpDesc()->GetAllInputsSize();
    std::vector<int64_t> input_offsets(1024, input_size);
    node->GetOpDescBarePtr()->SetInputOffset(input_offsets);
    for (auto &input_name : node->GetOpDesc()->GetAllInputNames()) {
      auto input = node->GetOpDesc()->MutableInputDesc(input_name);
      int64_t tensor_size = 0;
      TensorUtils::CalcTensorMemSize(input->GetShape(), input->GetFormat(), input->GetDataType(), tensor_size);
      tensor_size = (tensor_size + 32 - 1) / 32 * 32 + 32;
      TensorUtils::SetSize(*input, tensor_size);
    }
    auto out_size = node->GetOpDesc()->GetAllOutputsDescSize();
    std::vector<int64_t> out_offsets(1024, out_size);
    node->GetOpDescBarePtr()->SetOutputOffset(out_offsets);
    for (int32_t id = 0; id < static_cast<int32_t>(out_size); id++) {
      auto output = node->GetOpDesc()->MutableOutputDesc(id);
      int64_t tensor_size = 0;
      TensorUtils::CalcTensorMemSize(output->GetShape(), output->GetFormat(), output->GetDataType(), tensor_size);
      tensor_size = (tensor_size + 32 - 1) / 32 * 32 + 32;
      TensorUtils::SetSize(*output, tensor_size);
    }
  }
}

Status GetNoAlignedSize(const GeTensorDesc &tensor_desc, int64_t &no_aligned_size) {
  const Format out_format = tensor_desc.GetFormat();
  const DataType data_type = tensor_desc.GetDataType();
  const auto &shape = tensor_desc.GetShape();
  GE_ASSERT_SUCCESS(ge::TensorUtils::CalcTensorMemSize(shape, out_format, data_type, no_aligned_size));
  return SUCCESS;
}
}
class UtestNodeCheckerUtils : public testing::Test {};

TEST_F(UtestNodeCheckerUtils, AlignMemSize) {
  int64_t size = 2;
  ASSERT_EQ(NodeCheckerUtils::AlignMemSize(size), SUCCESS);
  ASSERT_EQ(size, 512);
}

TEST_F(UtestNodeCheckerUtils, NodeName_ParamNull) {
  ASSERT_STRCASEEQ(NodeCheckerUtils::NodeName(nullptr).c_str(), "node is null");
}

TEST_F(UtestNodeCheckerUtils, NodeName_NameTooLong) {
  auto root_builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &node_var = root_builder.AddNode("var", VARIABLE, 0, 1, 1, {1, 1, 44, 448});
  const auto &root_graph = root_builder.GetGraph();
  root_graph->TopologicalSorting();

  std::string long_name;
  for (size_t i = 0; i < 300U; ++i) {
    long_name.push_back('a');
  }
  node_var->GetOpDescBarePtr()->SetName(long_name);
  std::string expect_name;
  for (size_t i = 0; i < 200U; ++i) {
    expect_name.push_back('a');
  }
  expect_name += "...topo_id_0";
  EXPECT_STREQ(NodeCheckerUtils::NodeName(node_var.get()).c_str(), expect_name.c_str());
}

TEST_F(UtestNodeCheckerUtils, NodeName_ShortName) {
  auto root_builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &node_var = root_builder.AddNode("var", VARIABLE, 0, 1, 1, {1, 1, 44, 448});
  const auto &root_graph = root_builder.GetGraph();
  root_graph->TopologicalSorting();

  std::string expect_name = "var, topo_id: 0";
  EXPECT_STREQ(NodeCheckerUtils::NodeName(node_var.get()).c_str(), expect_name.c_str());
}

TEST_F(UtestNodeCheckerUtils, GetSpecialNodeTypes) {
  auto root_builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &a = root_builder.AddNode("a", VARIABLE, 0, 1, 1, {1, 1, 44, 448});
  const auto &b = root_builder.AddNode("b", VARIABLE, 0, 1, 1, {1, 1, 44, 448});
  const auto &c = root_builder.AddNode("c", VARIABLE, 1, 1, 1, {1, 1, 44, 448});
  const auto &d = root_builder.AddNode("d", VARIABLE, 1, 1, 1, {1, 1, 44, 448});
  const auto &pc = root_builder.AddNode("pc", PHONYCONCAT, 2, 2, 1, {1, 1, 44, 448});

  root_builder.AddDataEdge(a, 0, pc, 0);
  root_builder.AddDataEdge(b, 0, pc, 1);
  root_builder.AddDataEdge(pc, 0, c, 0);
  root_builder.AddDataEdge(pc, 1, d, 0);
  const auto &root_graph = root_builder.GetGraph();
  root_graph->TopologicalSorting();

  (void)AttrUtils::SetBool(pc->GetOpDesc(), ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
  (void)AttrUtils::SetBool(pc->GetOpDesc(), ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT, true);
  (void)AttrUtils::SetBool(pc->GetOpDesc(), ATTR_NAME_CONTINUOUS_INPUT, true);
  (void)AttrUtils::SetBool(pc->GetOpDesc(), ATTR_NAME_CONTINUOUS_OUTPUT, true);

  const auto types = NodeCheckerUtils::GetSpecialNodeTypes(pc.get());
  ASSERT_EQ(types.size(), 4);

  for (int64_t i = 0; i < 4; i++) {
    ASSERT_EQ(types.at(i), i);
  }
}

TEST_F(UtestNodeCheckerUtils, GetOutputOffset) {
  auto root_builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &a = root_builder.AddNode("a", VARIABLE, 0, 3, 1, {1, 1, 44, 448});
  const auto &root_graph = root_builder.GetGraph();
  root_graph->TopologicalSorting();

  a->GetOpDescBarePtr()->SetOutputOffset({1024, 2048, 3096});
  int64_t offset;
  NodeCheckerUtils::GetOutputOffset(a->GetOpDescBarePtr(), 0, offset);
  ASSERT_EQ(offset, 1024);
  NodeCheckerUtils::GetOutputOffset(a->GetOpDescBarePtr(), 1, offset);
  ASSERT_EQ(offset, 2048);
  NodeCheckerUtils::GetOutputOffset(a->GetOpDescBarePtr(), 2, offset);
  ASSERT_EQ(offset, 3096);

  ASSERT_NE(NodeCheckerUtils::GetOutputOffset(a->GetOpDescBarePtr(), 3, offset), SUCCESS);
}

TEST_F(UtestNodeCheckerUtils, GetStrideForContinuousInput_Success) {
  auto graph = block_mem_ut::BuildContinuousInputAndOutputOffsetForBufferFusion();
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);

  auto a = graph->FindNode("a");
  ASSERT_NE(a, nullptr);
  int64_t stride;
  ASSERT_EQ(NodeCheckerUtils::GetStrideForContinuousInput(a.get(), 0, stride), SUCCESS);


  std::vector<int64_t> offsets_for_fusion = {};
  const bool has_lx_fusion_attr = AttrUtils::GetListInt(a->GetOpDescBarePtr(),
                                                        ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION, offsets_for_fusion);
  ASSERT_TRUE(has_lx_fusion_attr);
  ASSERT_FALSE(offsets_for_fusion.empty());
  EXPECT_EQ(offsets_for_fusion.at(0), stride);
}

TEST_F(UtestNodeCheckerUtils, GetStrideForNoPaddingContinuousInput_UseOutputOffsetForBufferFusion) {
  auto graph = block_mem_ut::BuildPhonyConcatCascated();
  auto pc1 = graph->FindNode("pc1");
  auto a = graph->FindNode("a");
  ASSERT_NE(pc1, nullptr);
  ASSERT_NE(a, nullptr);

  int64_t stride;
  ASSERT_EQ(NodeCheckerUtils::GetStrideForNoPaddingContinuousInput(pc1.get(), a.get(), 0, stride), SUCCESS);
  std::vector<int64_t> offsets_for_fusion = {};
  const bool has_lx_fusion_attr = AttrUtils::GetListInt(a->GetOpDescBarePtr(),
                                                        ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION, offsets_for_fusion);
  ASSERT_TRUE(has_lx_fusion_attr);
  ASSERT_FALSE(offsets_for_fusion.empty());
  ASSERT_EQ(offsets_for_fusion.at(0), stride);
}

TEST_F(UtestNodeCheckerUtils, GetStrideForNoPaddingContinuousInput_UseDimIndex) {
  auto graph = block_mem_ut::BuildPhonyConcatCascated();
  auto pc1 = graph->FindNode("pc1");
  auto a = graph->FindNode("a");
  ASSERT_NE(pc1, nullptr);
  ASSERT_NE(a, nullptr);

  AttrUtils::ClearAllAttrs(a->GetOpDescBarePtr());
  std::vector<int64_t> offsets_for_fusion = {};
  const bool has_lx_fusion_attr = AttrUtils::GetListInt(a->GetOpDescBarePtr(),
                                                        ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION, offsets_for_fusion);
  ASSERT_FALSE(has_lx_fusion_attr);

  int64_t stride;
  (void)ge::AttrUtils::SetInt(pc1->GetOpDesc(), ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 3);
  ASSERT_EQ(NodeCheckerUtils::GetStrideForNoPaddingContinuousInput(pc1.get(), a.get(), 0, stride), SUCCESS);
  auto a_dims = a->GetOpDescBarePtr()->GetOutputDesc(0).GetShape().GetDims();

  ASSERT_EQ(a_dims[3] * 4, stride);
}

TEST_F(UtestNodeCheckerUtils, GetStrideForNoPaddingContinuousOutput_UseDimIndex) {
  auto graph = block_mem_ut::BuildPhonyConcatCascated();
  auto pc1 = graph->FindNode("pc1");
  auto a = graph->FindNode("a");
  ASSERT_NE(pc1, nullptr);
  int64_t stride;
  (void)ge::AttrUtils::SetInt(pc1->GetOpDesc(), ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX, 3);
  ASSERT_EQ(NodeCheckerUtils::GetStrideForNoPaddingContinuousOutput(pc1.get(), 0, stride), SUCCESS);
  auto a_dims = a->GetOpDescBarePtr()->GetOutputDesc(0).GetShape().GetDims();

  ASSERT_EQ(a_dims[3] * 4, stride);
}

TEST_F(UtestNodeCheckerUtils, ErrorLogAllInputs) {
  auto graph = block_mem_ut::BuildPhonyConcatCascated();
  MemoryAssigner memory_assigner(graph);
  map<uint64_t, size_t> mem_offset;
  size_t zero_memory_size = 0;
  EXPECT_EQ(memory_assigner.AssignMemory(mem_offset, zero_memory_size), GRAPH_SUCCESS);
  auto pc1 = graph->FindNode("pc1");
  auto a = graph->FindNode("a");
  ASSERT_NE(pc1, nullptr);

  NodeCheckerParam param{graph, pc1.get(), kSpecialNodeTypeContinuousInput};
  ASSERT_EQ(NodeCheckerUtils::ErrorLogAllInputs(param), SUCCESS);
}

TEST_F(UtestNodeCheckerUtils, ErrorLogAllOutputs) {
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

  for (auto &node : graph->GetAllNodes()) {
    for (auto &input_name : node->GetOpDesc()->GetAllInputNames()) {
      auto input = node->GetOpDesc()->MutableInputDesc(input_name);
      int64_t tensor_size = 0;
      TensorUtils::CalcTensorMemSize(input->GetShape(), input->GetFormat(), input->GetDataType(), tensor_size);
      TensorUtils::SetSize(*input, tensor_size);
    }
    auto out_size = node->GetOpDesc()->GetAllOutputsDescSize();
    for (int32_t id = 0; id < static_cast<int32_t>(out_size); id++) {
      auto output = node->GetOpDesc()->MutableOutputDesc(id);
      int64_t tensor_size = 0;
      TensorUtils::CalcTensorMemSize(output->GetShape(), output->GetFormat(), output->GetDataType(), tensor_size);
      TensorUtils::SetSize(*output, tensor_size);
    }
  }
  graph->TopologicalSorting();

  std::map<uint64_t, size_t> mem_type_to_mem_offset;
  size_t zero_copy_mem_size;
  VarManager::Instance(graph->GetSessionID())->Init(0, graph->GetSessionID(), 0, 0);
  MemoryAssigner mem_assigner(graph);
  EXPECT_EQ(mem_assigner.AssignMemory(mem_type_to_mem_offset, zero_copy_mem_size), SUCCESS);

  NodeCheckerParam param{graph, split_node.get(), kSpecialNodeTypeContinuousOutput};
  ASSERT_EQ(NodeCheckerUtils::ErrorLogAllOutputs(param), SUCCESS);
}

TEST_F(UtestNodeCheckerUtils, CheckNoPaddingContinuousInputNodeAttrs_Failed_BecauseNotAllInputNodesHasAttr) {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", CAST)->NODE("pc", PHONYCONCAT));
                  CHAIN(NODE("b", CAST)->NODE("pc", PHONYCONCAT));
                };
  auto graph = ToComputeGraph(g1);
  graph->TopologicalSorting();
  auto pc = graph->FindNode("pc");
  AttrUtils::SetBool(pc->GetOpDescBarePtr(), ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
  SetSizeAndOffset(graph);

  auto in0 = graph->FindNode("a");
  ASSERT_NE(in0, nullptr);
  std::vector<int64_t> offset_list = {1024};
  AttrUtils::SetListInt(in0->GetOpDescBarePtr(), ATTR_NAME_OUTPUT_OFFSET_LIST_FOR_CONTINUOUS, offset_list);
  ASSERT_NE(NodeCheckerUtils::CheckNoPaddingContinuousInputNodeAttrs(pc.get()), SUCCESS);

  auto in1 = graph->FindNode("b");
  ASSERT_NE(in1, nullptr);
  AttrUtils::SetListInt(in1->GetOpDescBarePtr(), ATTR_NAME_OUTPUT_OFFSET_LIST_FOR_CONTINUOUS, offset_list);
  AttrUtils::ClearAllAttrs(in0->GetOpDescBarePtr());

  ASSERT_NE(NodeCheckerUtils::CheckNoPaddingContinuousInputNodeAttrs(pc.get()), SUCCESS);
}

TEST_F(UtestNodeCheckerUtils, CheckNoPaddingContinuousInputNodeAttrs_Failed_Because2KindsAttrsMixed) {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", CAST)->NODE("pc", PHONYCONCAT));
                  CHAIN(NODE("b", CAST)->NODE("pc", PHONYCONCAT));
                };
  auto graph = ToComputeGraph(g1);
  graph->TopologicalSorting();
  auto pc = graph->FindNode("pc");
  AttrUtils::SetBool(pc->GetOpDescBarePtr(), ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
  SetSizeAndOffset(graph);

  auto in0 = graph->FindNode("a");
  ASSERT_NE(in0, nullptr);
  std::vector<int64_t> offset_list = {1024};
  AttrUtils::SetListInt(in0->GetOpDescBarePtr(), ATTR_NAME_OUTPUT_OFFSET_LIST_FOR_CONTINUOUS, offset_list);

  auto in1 = graph->FindNode("b");
  ASSERT_NE(in1, nullptr);
  AttrUtils::SetListInt(in1->GetOpDescBarePtr(), ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION, offset_list);

  ASSERT_NE(NodeCheckerUtils::CheckNoPaddingContinuousInputNodeAttrs(pc.get()), SUCCESS);
}

TEST_F(UtestNodeCheckerUtils, CheckNoPaddingContinuousInputNodeAttrs_Success) {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", CAST)->NODE("pc", PHONYCONCAT));
                  CHAIN(NODE("b", CAST)->NODE("pc", PHONYCONCAT));
                };
  auto graph = ToComputeGraph(g1);
  graph->TopologicalSorting();
  auto pc = graph->FindNode("pc");
  AttrUtils::SetBool(pc->GetOpDescBarePtr(), ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);

  // all input nodes do not has ATTR_NAME_OUTPUT_OFFSET_LIST_FOR_CONTINUOUS
  ASSERT_EQ(NodeCheckerUtils::CheckNoPaddingContinuousInputNodeAttrs(pc.get()), SUCCESS);

  // all input nodes has ATTR_NAME_OUTPUT_OFFSET_LIST_FOR_CONTINUOUS
  auto in0 = graph->FindNode("a");
  ASSERT_NE(in0, nullptr);
  std::vector<int64_t> offset_list = {1024};
  AttrUtils::SetListInt(in0->GetOpDescBarePtr(), ATTR_NAME_OUTPUT_OFFSET_LIST_FOR_CONTINUOUS, offset_list);

  auto in1 = graph->FindNode("b");
  ASSERT_NE(in1, nullptr);
  AttrUtils::SetListInt(in1->GetOpDescBarePtr(), ATTR_NAME_OUTPUT_OFFSET_LIST_FOR_CONTINUOUS, offset_list);

  ASSERT_EQ(NodeCheckerUtils::CheckNoPaddingContinuousInputNodeAttrs(pc.get()), SUCCESS);

  ASSERT_EQ(NodeCheckerUtils::CheckNoPaddingContinuousInputNodeAttrs(in0.get()), SUCCESS);
  ASSERT_EQ(NodeCheckerUtils::CheckNoPaddingContinuousInputNodeAttrs(in1.get()), SUCCESS);
}

TEST_F(UtestNodeCheckerUtils, CheckNoPaddingContinuousInputNodeAttrs_Cascated_Success) {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("a", CAST)->NODE("pc1", PHONYCONCAT));
                  CHAIN(NODE("b", CAST)->NODE("pc1", PHONYCONCAT));
                  CHAIN(NODE("c", CAST)->NODE("pc2", PHONYCONCAT));
                  CHAIN(NODE("pc1", CAST)->NODE("pc2", PHONYCONCAT));
                };
  auto graph = ToComputeGraph(g1);
  graph->TopologicalSorting();
  auto pc1 = graph->FindNode("pc1");
  AttrUtils::SetBool(pc1->GetOpDescBarePtr(), ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
  auto pc2= graph->FindNode("pc2");
  AttrUtils::SetBool(pc2->GetOpDescBarePtr(), ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);

  // all input nodes has ATTR_NAME_OUTPUT_OFFSET_LIST_FOR_CONTINUOUS
  auto in0 = graph->FindNode("a");
  ASSERT_NE(in0, nullptr);
  std::vector<int64_t> offset_list = {1024};
  AttrUtils::SetListInt(in0->GetOpDescBarePtr(), ATTR_NAME_OUTPUT_OFFSET_LIST_FOR_CONTINUOUS, offset_list);

  auto in1 = graph->FindNode("b");
  ASSERT_NE(in1, nullptr);
  AttrUtils::SetListInt(in1->GetOpDescBarePtr(), ATTR_NAME_OUTPUT_OFFSET_LIST_FOR_CONTINUOUS, offset_list);
  auto in2 = graph->FindNode("c");
  ASSERT_NE(in2, nullptr);
  AttrUtils::SetListInt(in2->GetOpDescBarePtr(), ATTR_NAME_OUTPUT_OFFSET_LIST_FOR_CONTINUOUS, offset_list);

  ASSERT_EQ(NodeCheckerUtils::CheckNoPaddingContinuousInputNodeAttrs(pc1.get()), SUCCESS);
  ASSERT_EQ(NodeCheckerUtils::CheckNoPaddingContinuousInputNodeAttrs(pc2.get()), SUCCESS);
}

TEST_F(UtestNodeCheckerUtils, CheckNoPaddingContinuousOutputNodeAttrs_Failed_BecauseNotAllOutputsHasAttr) {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("ps", PHONYSPLIT)->NODE("a", CAST));
                  CHAIN(NODE("ps", PHONYSPLIT)->NODE("b", CAST));
                  CHAIN(NODE("ps", PHONYSPLIT)->EDGE(0, 0)->NODE("c", CAST));
                };
  auto graph = ToComputeGraph(g1);
  graph->TopologicalSorting();
  auto ps = graph->FindNode("ps");
  AttrUtils::SetBool(ps->GetOpDescBarePtr(), ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT, true);
  SetSizeAndOffset(graph);

  auto out0 = graph->FindNode("a");
  ASSERT_NE(out0, nullptr);
  std::vector<int64_t> offset_list = {1024};
  AttrUtils::SetListInt(out0->GetOpDescBarePtr(), ATTR_NAME_INPUT_OFFSET_LIST_FOR_CONTINUOUS, offset_list);
  ASSERT_NE(NodeCheckerUtils::CheckNoPaddingContinuousOutputNodeAttrs(ps.get()), SUCCESS);

  auto out1 = graph->FindNode("b");
  ASSERT_NE(out1, nullptr);
  AttrUtils::SetListInt(out1->GetOpDescBarePtr(), ATTR_NAME_INPUT_OFFSET_LIST_FOR_CONTINUOUS, offset_list);
  AttrUtils::ClearAllAttrs(out0->GetOpDescBarePtr());

  ASSERT_NE(NodeCheckerUtils::CheckNoPaddingContinuousOutputNodeAttrs(ps.get()), SUCCESS);
}

/*
 *    ps
 *   /  \
 *  a
 */
TEST_F(UtestNodeCheckerUtils, CheckNoPaddingContinuousOutputNodeAttrs_Success_WithSuspendOutAnchor) {
  auto root_builder = block_mem_ut::GraphBuilder("root_graph");
  const auto &ps = root_builder.AddNode("ps", PHONYSPLIT, 0, 2);
  const auto &a = root_builder.AddNode("a", CAST, 1, 1);
  root_builder.AddDataEdge(ps, 0, a, 0);

  const auto &graph = root_builder.GetGraph();
  graph->TopologicalSorting();
  AttrUtils::SetBool(ps->GetOpDescBarePtr(), ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT, true);

  auto out0 = graph->FindNode("a");
  ASSERT_NE(out0, nullptr);
  std::vector<int64_t> offset_list = {1024};
  AttrUtils::SetListInt(out0->GetOpDescBarePtr(), ATTR_NAME_INPUT_OFFSET_LIST_FOR_CONTINUOUS, offset_list);
  ASSERT_EQ(NodeCheckerUtils::CheckNoPaddingContinuousOutputNodeAttrs(ps.get()), SUCCESS);

  ASSERT_EQ(NodeCheckerUtils::ErrorLogAllOutputs({graph, ps.get(), kSpecialNodeTypeNoPaddingContinuousOutput}),
            SUCCESS);
}

TEST_F(UtestNodeCheckerUtils, CheckNoPaddingContinuousOutputNodeAttrs_Success) {
  DEF_GRAPH(g1) {
                  CHAIN(NODE("ps1", PHONYSPLIT)->NODE("a", CAST));
                  CHAIN(NODE("ps1", PHONYSPLIT)->NODE("b", CAST));
                  CHAIN(NODE("ps1", PHONYSPLIT)->NODE("ps2", PHONYSPLIT));
                  CHAIN(NODE("ps2", PHONYSPLIT)->NODE("c", CAST));
                  CHAIN(NODE("ps2", PHONYSPLIT)->NODE("d", CAST));
                };
  auto graph = ToComputeGraph(g1);
  graph->TopologicalSorting();
  auto ps1 = graph->FindNode("ps1");
  AttrUtils::SetBool(ps1->GetOpDescBarePtr(), ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT, true);
  auto ps2 = graph->FindNode("ps2");
  AttrUtils::SetBool(ps2->GetOpDescBarePtr(), ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT, true);
  SetSizeAndOffset(graph);

  auto out0 = graph->FindNode("a");
  ASSERT_NE(out0, nullptr);
  std::vector<int64_t> offset_list = {1024};
  AttrUtils::SetListInt(out0->GetOpDescBarePtr(), ATTR_NAME_INPUT_OFFSET_LIST_FOR_CONTINUOUS, offset_list);

  auto out1 = graph->FindNode("b");
  ASSERT_NE(out1, nullptr);
  AttrUtils::SetListInt(out1->GetOpDescBarePtr(), ATTR_NAME_INPUT_OFFSET_LIST_FOR_CONTINUOUS, offset_list);

  auto out2 = graph->FindNode("c");
  ASSERT_NE(out2, nullptr);
  AttrUtils::SetListInt(out2->GetOpDescBarePtr(), ATTR_NAME_INPUT_OFFSET_LIST_FOR_CONTINUOUS, offset_list);

  auto out3 = graph->FindNode("d");
  ASSERT_NE(out3, nullptr);
  AttrUtils::SetListInt(out3->GetOpDescBarePtr(), ATTR_NAME_INPUT_OFFSET_LIST_FOR_CONTINUOUS, offset_list);

  ASSERT_EQ(NodeCheckerUtils::CheckNoPaddingContinuousOutputNodeAttrs(ps1.get()), SUCCESS);
  ASSERT_EQ(NodeCheckerUtils::CheckNoPaddingContinuousOutputNodeAttrs(ps2.get()), SUCCESS);
}

/*
 *    a
 *    |
 *   refnode   b
 *        \  /
 *         pc
 */
TEST_F(UtestNodeCheckerUtils, GetInputOutputSizeSuccess) {
  const auto refnode = OP_CFG(ASSIGNADD).Attr(ATTR_NAME_REFERENCE, true).InNames({"ref"}).OutNames({"ref"});
  DEF_GRAPH(g1) {
    CHAIN(NODE("a", RELU)->NODE("refnode", refnode)->NODE("pc", PHONYCONCAT)->NODE("netoutput", NETOUTPUT));
    CHAIN(NODE("b", RELU)->EDGE(0, 1)->NODE("pc", PHONYCONCAT));
  };
  auto graph = ToComputeGraph(g1);
  auto pc = graph->FindNode("pc");
  (void)ge::AttrUtils::SetBool(pc->GetOpDesc(), ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);

  graph->TopologicalSorting();
  SetSizeAndOffset(graph);

  auto ref_node = graph->FindNode("refnode");

  // 由于refnode的输出连接的是NOPADDING连续输入，所以这里获取的size是no aligned size
  int64_t mem_size = 0;
  auto ret = NodeCheckerUtils::GetInputSize(ref_node.get(), 0, mem_size);
  ASSERT_EQ(ret, SUCCESS);

  int64_t noalign_size = 0;
  GetNoAlignedSize(ref_node->GetOpDescBarePtr()->GetInputDesc(0), noalign_size);
  EXPECT_EQ(mem_size, noalign_size);

  // 由于a的输出经RefNode连接到了NOPADDING连续输入，所以这里获取的size是no aligned size
  auto a = graph->FindNode("a");
  ret = NodeCheckerUtils::GetOutputSize(a.get(), 0, mem_size);
  ASSERT_EQ(ret, SUCCESS);

  noalign_size = 0;
  GetNoAlignedSize(a->GetOpDescBarePtr()->GetOutputDesc(0), noalign_size);
  EXPECT_EQ(mem_size, noalign_size);

  auto b = graph->FindNode("b");
  ret = NodeCheckerUtils::GetOutputSize(b.get(), 0, mem_size);
  ASSERT_EQ(ret, SUCCESS);
  noalign_size = 0;
  GetNoAlignedSize(b->GetOpDescBarePtr()->GetOutputDesc(0), noalign_size);
  EXPECT_EQ(mem_size, noalign_size);
}
} // namespace ge
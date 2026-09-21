/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <numeric>
#include <gtest/gtest.h>

#include "ge_tensor.h"
#include "operator_reg.h"
#include "easy_graph/builder/graph_dsl.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "ge_graph_dsl/op_desc/op_desc_cfg_box.h"
#include "pattern_fusion/concat_slice_simplification_pass.h"

#include "all_ops_cpp.h"
#include "graph/compute_graph.h"
#include "framework/omg/parser/parser_types.h"
#include "esb_graph.h"
#include "esb_funcs_cpp.h"
#include "graph_utils.h"
#include "graph_utils_ex.h"
#include "node_utils.h"
#include "lowering/lowerings.h"

namespace ge {
class ConcatSliceSimplificationPassTest : public testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

namespace {
const std::string CONSTANT = "Const";
const std::string NETOUTPUT = "NetOutput";
constexpr auto kOpTypeSlice = "Slice";
constexpr auto kOpTypeSliceD = "SliceD";
constexpr auto kOpTypeStridedSlice = "StridedSlice";
constexpr auto kOpTypeStridedSliceD = "StridedSliceD";
constexpr auto kOpTypeConcat = "Concat";
constexpr auto kOpTypeConcatD = "ConcatD";
constexpr auto kOpTypeConcatV2 = "ConcatV2";
constexpr auto kOpTypeConcatV2D = "ConcatV2D";
constexpr auto kAttrNameN = "N";

template <typename T>
es::Tensor CreateConst(es::Graph &graph, ge::DataType dtype, const std::vector<int64_t> &dims,
                       const std::vector<T> &value) {
  auto result = es::FileConstant(graph, dims, dtype);
  GeTensorDesc desc(GeShape(dims), ge::FORMAT_ND, dtype);
  GeTensorPtr tensor =
      std::make_shared<GeTensor>(desc, reinterpret_cast<const uint8_t *>(value.data()), sizeof(T) * value.size());
  AttrUtils::SetTensor(result.GetEsbTensor()->GetProducer()->GetOpDesc(), "value", tensor);
  result.GetEsbTensor()->GetProducer()->GetOpDesc()->SetType(ge::CONSTANT);
  result.GetEsbTensor()->GetProducer()->GetOpDesc()->UpdateOutputDesc(0, tensor->GetTensorDesc());
  return result;
}

REG_OP(SliceD)
    .INPUT(x, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(offsets, ListInt)
    .REQUIRED_ATTR(size, ListInt)
    .OP_END_FACTORY_REG(SliceD)
}  // namespace


TEST_F(ConcatSliceSimplificationPassTest, OptimizeSuccessAndPrune) {
  auto dtype = DT_INT32;
  auto shape = GeShape(std::vector<int64_t>({32, 1024, 1024}));
  GeTensorDesc desc(shape, FORMAT_ND, dtype);
  auto num_elements = shape.GetShapeSize();
  std::vector<int32_t> buffer(shape.GetShapeSize());
  std::iota(buffer.begin(), buffer.end(), 0);
  ::es::Graph es_graph("graph");
  {
    auto data_0 = es_graph.CreateInput(0, "data_0", nullptr);
    data_0.SetSymbolShape({"32", "1"});
    data_0.SetShape({32, 1});
    auto data_1 = es_graph.CreateInput(1, "data_1", nullptr);
    data_1.SetSymbolShape({"32", "4"});
    data_1.SetShape({32, 4});
    auto concat_0 = es::ConcatD({data_0, data_1}, 1, 2);
    concat_0.SetSymbolShape({"32", "5"});
    concat_0.SetShape({32, 5});
    auto slice_0 = es::SliceD(concat_0, {1, 2}, {4, 2});
    slice_0.SetSymbolShape({"4", "2"});
    auto slice_1 = es::SliceD(concat_0, {1, 2}, {4, -1});
    slice_1.SetSymbolShape({"4", "3"});
    es_graph.SetOutput(slice_0, 0);
    es_graph.SetOutput(slice_1, 1);
  }
  auto test_graph = es_graph.Build();
  auto graph = GraphUtilsEx::GetComputeGraph(*test_graph);
  for (const auto &node : graph->GetAllNodes()) {
    for (size_t i = 0; i < node->GetAllOutDataAnchorsSize(); ++i) {
      const auto &src_shape = node->GetOpDesc()->GetOutputDesc(i).GetShape();
      for (const auto &[in_anchor, peer_node] :
           NodeUtils::GetOutDataNodesWithAnchorByIndex(*node, static_cast<int32_t>(i))) {
        peer_node->GetOpDesc()->MutableInputDesc(in_anchor->GetIdx())->SetShape(src_shape);
      }
    }
  }
  GraphPasses graph_passes;
  bool prune_invoked = false;
  bool const_folding_invoked = false;
  graph_passes.prune_graph_func = [&prune_invoked](const ComputeGraphPtr &graph) -> Status {
    (void) graph;
    prune_invoked = true;
    return 0;
  };
  graph_passes.constant_folding_func = [&const_folding_invoked](NodePtr &node) -> graphStatus {
    (void) node;
    const_folding_invoked = true;
    return 0;
  };
  bool changed = false;
  EXPECT_EQ(ConcatSliceSimplificationPass().Run(graph, graph_passes, changed), SUCCESS);
  auto data_1_node = graph->FindNode("data_1");
  ASSERT_TRUE(data_1_node != nullptr);
  EXPECT_EQ(data_1_node->GetOutDataNodes().size(), 3);
  EXPECT_TRUE(prune_invoked);
  EXPECT_FALSE(const_folding_invoked);
}

TEST_F(ConcatSliceSimplificationPassTest, OptimizeSuccessAndFold) {
  auto dtype = DT_INT32;
  auto shape = GeShape(std::vector<int64_t>({32, 1024, 1024}));
  GeTensorDesc desc(shape, FORMAT_ND, dtype);
  auto num_elements = shape.GetShapeSize();
  std::vector<int32_t> buffer(shape.GetShapeSize());
  std::iota(buffer.begin(), buffer.end(), 0);
  ::es::Graph es_graph("graph");
  {
    auto data_0 = es_graph.CreateInput(0, "data_0", nullptr);
    data_0.SetSymbolShape({"32", "1"});
    data_0.SetShape({32, 1});
    auto const_0 = CreateConst(es_graph, dtype, shape.GetDims(), buffer);
    const_0.SetSymbolShape({"32", "4"});
    const_0.SetShape({32, 4});
    auto concat_0 = es::ConcatD({data_0, const_0}, 1, 2);
    concat_0.SetSymbolShape({"32", "5"});
    concat_0.SetShape({32, 5});
    auto slice_0 = es::SliceD(concat_0, {1, 2}, {4, 2});
    slice_0.SetSymbolShape({"4", "2"});
    es_graph.SetOutput(slice_0, 0);
  }
  auto test_graph = es_graph.Build();
  auto graph = GraphUtilsEx::GetComputeGraph(*test_graph);
  for (const auto &node : graph->GetAllNodes()) {
    for (size_t i = 0; i < node->GetAllOutDataAnchorsSize(); ++i) {
      const auto &src_shape = node->GetOpDesc()->GetOutputDesc(i).GetShape();
      for (const auto &[in_anchor, peer_node] :
           NodeUtils::GetOutDataNodesWithAnchorByIndex(*node, static_cast<int32_t>(i))) {
        peer_node->GetOpDesc()->MutableInputDesc(in_anchor->GetIdx())->SetShape(src_shape);
      }
    }
  }
  GraphPasses graph_passes;
  bool prune_invoked = false;
  bool const_folding_invoked = false;
  graph_passes.prune_graph_func = [&prune_invoked](const ComputeGraphPtr &graph) -> Status {
    (void) graph;
    prune_invoked = true;
    return 0;
  };
  graph_passes.constant_folding_func = [&const_folding_invoked](NodePtr &node) -> graphStatus {
    (void) node;
    const_folding_invoked = true;
    return 0;
  };
  bool changed = false;
  EXPECT_EQ(ConcatSliceSimplificationPass().Run(graph, graph_passes, changed), SUCCESS);
  auto src_node = graph->FindFirstNodeMatchType("Const");
  ASSERT_TRUE(src_node != nullptr);
  EXPECT_EQ(src_node->GetOutDataNodes().size(), 2);
  EXPECT_TRUE(prune_invoked);
  EXPECT_TRUE(const_folding_invoked);
}
}  // namespace ge

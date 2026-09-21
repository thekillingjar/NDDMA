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
#include "pattern_fusion/decompose_large_const_pass.h"

#include "all_ops_cpp.h"
#include "graph/compute_graph.h"
#include "framework/omg/parser/parser_types.h"
#include "esb_graph.h"
#include "esb_funcs_cpp.h"
#include "graph_utils_ex.h"
#include "lowering/lowerings.h"

namespace ge {
class DecomposeLargeConstPassTest : public testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

namespace {
const std::string CONSTANT = "Const";
const std::string NETOUTPUT = "NetOutput";
const std::string RELU = "ReLU";
const std::string TILE = "Tile";
const std::string FILL = "Fill";

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
}  // namespace

REG_OP(Tile)
    .INPUT(x, TensorType::BasicType())
    .INPUT(multiples, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(Tile)

REG_OP(Fill)
    .INPUT(dims, TensorType::IndexNumberType())
    .INPUT(value, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8, DT_INT16,
                              DT_INT8, DT_COMPLEX64, DT_INT64, DT_BOOL, DT_QINT8,
                              DT_QUINT8, DT_QINT32, DT_QINT16, DT_QUINT16, DT_UINT16,
                              DT_COMPLEX128, DT_FLOAT16, DT_UINT32, DT_UINT64}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8, DT_INT16,
                              DT_INT8, DT_COMPLEX64, DT_INT64, DT_BOOL, DT_QINT8,
                              DT_QUINT8, DT_QINT32, DT_QINT16, DT_QUINT16, DT_UINT16,
                              DT_COMPLEX128, DT_FLOAT16, DT_UINT32, DT_UINT64}))
    .OP_END_FACTORY_REG(Fill)

TEST_F(DecomposeLargeConstPassTest, test_decompose_const_not_needed) {
  auto dtype = DT_INT32;
  auto shape = GeShape(std::vector<int64_t>({2, 128, 128}));
  GeTensorDesc desc(shape, FORMAT_ND, dtype);
  auto num_elements = shape.GetShapeSize();
  std::vector<int32_t> buffer(shape.GetShapeSize());
  auto const_tensor = std::make_shared<GeTensor>(desc, reinterpret_cast<uint8_t *>(buffer.data()),
                                                 GetSizeByDataType(dtype) * num_elements);
  auto const_0 = OP_CFG(CONSTANT)
                     .TensorDesc(desc.GetFormat(), dtype, shape.GetDims())
                     .OutCnt(1)
                     .Weight(const_tensor)
                     .Build("const_0");
  DEF_GRAPH(test_graph) {
    CHAIN(NODE(const_0)->NODE("output_0", NETOUTPUT));
  };
  auto graph = ToComputeGraph(test_graph);
  EXPECT_EQ(DecomposeLargeConstPass::Run(graph), GRAPH_SUCCESS);
  EXPECT_TRUE(graph->FindNode("const_0") != nullptr);
}

TEST_F(DecomposeLargeConstPassTest, test_decompose_const_not_all_pointwise) {
  auto dtype = DT_INT32;
  auto shape = GeShape(std::vector<int64_t>({32, 1024, 1024}));
  GeTensorDesc desc(shape, FORMAT_ND, dtype);
  auto num_elements = shape.GetShapeSize();
  std::vector<int32_t> buffer(shape.GetShapeSize());
  std::iota(buffer.begin(), buffer.end(), 0);
  ::es::Graph es_graph("graph");
  {
    auto const_0 = CreateConst(es_graph, dtype, shape.GetDims(), buffer);
    const_0.SetSymbolShape({"32", "1024", "1024"});
    auto relu_0 = es::Relu(const_0);
    relu_0.SetSymbolShape({"32", "1024", "1024"});
    auto concat_0 = es::ConcatD({const_0, relu_0}, 0);
    concat_0.SetSymbolShape({"64", "1024", "1024"});
    es_graph.SetOutput(concat_0, 0);
  }
  auto test_graph = es_graph.Build();
  auto graph = GraphUtilsEx::GetComputeGraph(*test_graph);
  auto orig_const_node = graph->FindFirstNodeMatchType(CONSTANT);
  EXPECT_EQ(DecomposeLargeConstPass::Run(graph), GRAPH_SUCCESS);
  EXPECT_TRUE(graph->FindNode(orig_const_node->GetName()) != nullptr);
}

TEST_F(DecomposeLargeConstPassTest, test_decompose_const_not_broadcasted) {
  auto dtype = DT_INT32;
  auto shape = GeShape(std::vector<int64_t>({32, 1024, 1024}));
  GeTensorDesc desc(shape, FORMAT_ND, dtype);
  auto num_elements = shape.GetShapeSize();
  std::vector<int32_t> buffer(shape.GetShapeSize());
  std::iota(buffer.begin(), buffer.end(), 0);
  ::es::Graph es_graph("graph");
  {
    auto const_0 = CreateConst(es_graph, dtype, shape.GetDims(), buffer);
    const_0.SetSymbolShape({"32", "1024", "1024"});
    auto relu_0 = es::Relu(const_0);
    relu_0.SetSymbolShape({"32", "1024", "1024"});
    es_graph.SetOutput(relu_0, 0);
  }
  auto test_graph = es_graph.Build();
  auto graph = GraphUtilsEx::GetComputeGraph(*test_graph);
  auto orig_const_node = graph->FindFirstNodeMatchType(CONSTANT);
  EXPECT_EQ(DecomposeLargeConstPass::Run(graph), GRAPH_SUCCESS);
  EXPECT_TRUE(graph->FindNode(orig_const_node->GetName()) != nullptr);
}

TEST_F(DecomposeLargeConstPassTest, test_decompose_const_brc_first_dim) {
  auto dtype = DT_INT32;
  auto shape = GeShape(std::vector<int64_t>({32, 1024, 1024}));
  std::vector<int32_t> buffer(shape.GetShapeSize());
  for (int i = 0; i < 32; ++i) {
    std::iota(buffer.begin() + i * 1024 * 1024, buffer.begin() + (i + 1) * 1024 * 1024, 0);
  }
  ::es::Graph es_graph("graph");
  {
    auto const_0 = CreateConst(es_graph, dtype, shape.GetDims(), buffer);
    const_0.SetSymbolShape({"32", "1024", "1024"});
    auto relu_0 = es::Relu(const_0);
    relu_0.SetSymbolShape({"32", "1024", "1024"});
    es_graph.SetOutput(relu_0, 0);
  }
  auto test_graph = es_graph.Build();
  auto graph = GraphUtilsEx::GetComputeGraph(*test_graph);
  EXPECT_EQ(DecomposeLargeConstPass::Run(graph), GRAPH_SUCCESS);
  auto new_const_node = graph->FindFirstNodeMatchType(CONSTANT);
  EXPECT_TRUE(new_const_node != nullptr);
  ASSERT_TRUE(graph->FindFirstNodeMatchType(TILE) != nullptr);
  auto node = graph->FindFirstNodeMatchType(TILE);
  LoweringManager::Lowering(node);
  const auto node_out_anchor = node->GetOutDataAnchor(0);
  const auto node_kernel_box = loop::GetKernelBox(node_out_anchor);
  EXPECT_EQ(FuseTypeToString(node_kernel_box.Type()), "pointwise");
}

TEST_F(DecomposeLargeConstPassTest, test_decompose_const_brc_all) {
  auto dtype = DT_INT32;
  auto shape = GeShape(std::vector<int64_t>({32, 1024, 1024}));
  GeTensorDesc desc(shape, FORMAT_ND, dtype);
  std::vector<int32_t> buffer(shape.GetShapeSize());
  ::es::Graph es_graph("graph");
  {
    auto const_0 = CreateConst(es_graph, dtype, shape.GetDims(), buffer);
    const_0.SetSymbolShape({"32", "1024", "1024"});
    auto relu_0 = es::Relu(const_0);
    relu_0.SetSymbolShape({"32", "1024", "1024"});
    es_graph.SetOutput(relu_0, 0);
  }
  auto test_graph = es_graph.Build();
  auto graph = GraphUtilsEx::GetComputeGraph(*test_graph);
  EXPECT_EQ(DecomposeLargeConstPass::Run(graph), GRAPH_SUCCESS);
  EXPECT_TRUE(graph->FindNode("const_0") == nullptr);
  EXPECT_TRUE(graph->FindFirstNodeMatchType(FILL) != nullptr);
  auto new_const_node = graph->FindFirstNodeMatchType(CONSTANT);
  EXPECT_TRUE(new_const_node != nullptr);
  EXPECT_TRUE(new_const_node->GetOpDesc()->GetOutputDesc(0).GetShape().IsScalar());

  auto node = graph->FindFirstNodeMatchType(FILL);
  ASSERT_TRUE(node != nullptr);
  LoweringManager::Lowering(node);
  const auto node_out_anchor = node->GetOutDataAnchor(0);
  const auto node_kernel_box = loop::GetKernelBox(node_out_anchor);
  EXPECT_EQ(FuseTypeToString(node_kernel_box.Type()), "pointwise");
}
}  // namespace ge
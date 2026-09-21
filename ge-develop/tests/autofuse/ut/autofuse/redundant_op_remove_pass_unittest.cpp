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
#include "pattern_fusion/cascade_reshape_remove_pass.h"
#include "pattern_fusion/redundant_slice_remove_pass.h"

#include "all_ops_cpp.h"
#include "graph/compute_graph.h"
#include "framework/omg/parser/parser_types.h"
#include "esb_graph.h"
#include "esb_funcs_cpp.h"
#include "graph_utils.h"
#include "graph_utils_ex.h"
#include "lowering/lowerings.h"
#include "pattern_fusion/pattern_fusion.h"
#include "utils/autofuse_utils.h"

namespace ge {
class RedundantOpRemovePassTest : public testing::Test {
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

}  // namespace

TEST_F(RedundantOpRemovePassTest, RemoveReshapeWithCtrlEdge) {
  auto data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4}).InCnt(0).OutCnt(1).Build("Data");
  auto reshape1 = OP_CFG("Reshape").TensorDesc(FORMAT_NCHW, DT_FLOAT, {24}).InCnt(1).OutCnt(1).Build("reshape1");
  auto reshape2 = OP_CFG("Reshape").TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, 12}).InCnt(1).OutCnt(1).Build("reshape2");
  auto reshape3 = OP_CFG("Reshape").TensorDesc(FORMAT_NCHW, DT_FLOAT, {3, 8}).InCnt(1).OutCnt(1).Build("reshape3");
  auto reshape4 = OP_CFG("Reshape").TensorDesc(FORMAT_NCHW, DT_FLOAT, {24}).InCnt(1).OutCnt(1).Build("reshape4");
  auto reshape5 = OP_CFG("Reshape").TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, 12}).InCnt(1).OutCnt(1).Build("reshape5");
  auto reshape6 = OP_CFG("Reshape").TensorDesc(FORMAT_NCHW, DT_FLOAT, {3, 8}).InCnt(1).OutCnt(1).Build("reshape6");
  auto relu = OP_CFG("Relu").TensorDesc(FORMAT_NCHW, DT_FLOAT, {3, 8}).InCnt(1).OutCnt(1).Build("relu");

  DEF_GRAPH(test_graph) {
    CHAIN(NODE(data)
              ->NODE(reshape1)
              ->NODE(reshape2)
              ->NODE(reshape3)
              ->NODE(reshape4)
              ->NODE(reshape5)
              ->NODE(reshape6)
              ->NODE(relu)
              ->NODE("output_0", NETOUTPUT));
    CHAIN(NODE(reshape3)->CTRL_EDGE()->NODE(relu));
  };
  auto graph = ToComputeGraph(test_graph);
  bool changed = false;
  EXPECT_EQ(CascadeReshapeRemovePass().Run(graph, changed), GRAPH_SUCCESS);
  EXPECT_TRUE(graph->FindNode("reshape1") == nullptr);
  EXPECT_TRUE(graph->FindNode("reshape2") == nullptr);
  EXPECT_TRUE(graph->FindNode("reshape3") == nullptr);
  EXPECT_TRUE(graph->FindNode("reshape4") == nullptr);
  EXPECT_TRUE(graph->FindNode("reshape5") == nullptr);
  EXPECT_TRUE(graph->FindNode("reshape6") != nullptr);
}

TEST_F(RedundantOpRemovePassTest, RemoveReshapeWithEdge) {
  auto data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4}).InCnt(0).OutCnt(1).Build("Data");
  auto reshape1 = OP_CFG("Reshape").TensorDesc(FORMAT_NCHW, DT_FLOAT, {24}).InCnt(1).OutCnt(1).Build("reshape1");
  auto reshape2 = OP_CFG("Reshape").TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, 12}).InCnt(1).OutCnt(1).Build("reshape2");
  auto reshape3 = OP_CFG("Reshape").TensorDesc(FORMAT_NCHW, DT_FLOAT, {3, 8}).InCnt(1).OutCnt(1).Build("reshape3");
  auto reshape4 = OP_CFG("Reshape").TensorDesc(FORMAT_NCHW, DT_FLOAT, {24}).InCnt(1).OutCnt(1).Build("reshape4");
  auto reshape5 = OP_CFG("Reshape").TensorDesc(FORMAT_NCHW, DT_FLOAT, {2, 12}).InCnt(1).OutCnt(1).Build("reshape5");
  auto reshape6 = OP_CFG("Reshape").TensorDesc(FORMAT_NCHW, DT_FLOAT, {3, 8}).InCnt(1).OutCnt(1).Build("reshape6");
  auto add = OP_CFG("Add").TensorDesc(FORMAT_NCHW, DT_FLOAT, {3, 8}).InCnt(2).OutCnt(1).Build("add");

  DEF_GRAPH(test_graph) {
    CHAIN(NODE(data)
              ->NODE(reshape1)
              ->NODE(reshape2)
              ->NODE(reshape3)
              ->NODE(reshape4)
              ->NODE(reshape5)
              ->NODE(reshape6)
              ->NODE(add)
              ->NODE("output_0", NETOUTPUT));
    CHAIN(NODE(reshape3)->DATA_EDGE(0, 1)->NODE(add));
  };
  auto graph = ToComputeGraph(test_graph);
  bool changed = false;
  EXPECT_EQ(CascadeReshapeRemovePass().Run(graph, changed), GRAPH_SUCCESS);
  EXPECT_TRUE(graph->FindNode("reshape1") == nullptr);
  EXPECT_TRUE(graph->FindNode("reshape2") == nullptr);
  EXPECT_TRUE(graph->FindNode("reshape3") != nullptr);
  EXPECT_TRUE(graph->FindNode("reshape4") == nullptr);
  EXPECT_TRUE(graph->FindNode("reshape5") == nullptr);
  EXPECT_TRUE(graph->FindNode("reshape6") != nullptr);
}

TEST_F(RedundantOpRemovePassTest, RemoveReshapeWithSlice) {
  auto dtype = DT_INT32;
  auto shape = GeShape(std::vector<int64_t>({1}));
  GeTensorDesc desc(shape, FORMAT_ND, dtype);
  auto num_elements = shape.GetShapeSize();
  std::vector<int32_t> begin_buffer{0};
  auto begin_tensor = std::make_shared<GeTensor>(desc, reinterpret_cast<uint8_t *>(begin_buffer.data()),
                                                 GetSizeByDataType(dtype) * num_elements);
  auto const_0 = OP_CFG(CONSTANT)
      .TensorDesc(desc.GetFormat(), dtype, shape.GetDims())
      .OutCnt(1)
      .Weight(begin_tensor)
      .Build("const_0");

  std::vector<int32_t> end_buffer{24};
  auto end_tensor = std::make_shared<GeTensor>(desc, reinterpret_cast<uint8_t *>(end_buffer.data()),
                                               GetSizeByDataType(dtype) * num_elements);
  auto const_1 = OP_CFG(CONSTANT)
      .TensorDesc(desc.GetFormat(), dtype, shape.GetDims())
      .OutCnt(1)
      .Weight(end_tensor)
      .Build("const_1");
  auto slice1 = OP_CFG("Slice")
      .TensorDesc(FORMAT_NCHW, DT_FLOAT, {24})
      .InCnt(3)
      .OutCnt(1)
      .InNames({"x", "offsets", "size"})
      .Build("slice1");

  auto data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4}).InCnt(0).OutCnt(1).Build("Data");
  auto reshape1 = OP_CFG("Reshape").TensorDesc(FORMAT_NCHW, DT_FLOAT, {24}).InCnt(1).OutCnt(1).Build("reshape1");
  auto reshape2 = OP_CFG("Reshape").TensorDesc(FORMAT_NCHW, DT_FLOAT, {3, 8}).InCnt(1).OutCnt(1).Build("reshape2");

  auto abs = OP_CFG("Abs").TensorDesc(FORMAT_NCHW, DT_FLOAT, {3, 8}).InCnt(1).OutCnt(1).Build("abs");

  DEF_GRAPH(test_graph) {
    CHAIN(NODE(data)->NODE(reshape1)->NODE(slice1)->NODE(reshape2)->NODE(abs)->NODE("output_0", NETOUTPUT));
    CHAIN(NODE(const_0)->DATA_EDGE(0, 1)->NODE(slice1));
    CHAIN(NODE(const_1)->DATA_EDGE(0, 2)->NODE(slice1));
  };
  auto graph = ToComputeGraph(test_graph);
  EXPECT_EQ(PatternFusion::RunEarlyPasses(graph), GRAPH_SUCCESS);
  EXPECT_TRUE(graph->FindNode("reshape1") == nullptr);
  EXPECT_TRUE(graph->FindNode("reshape2") != nullptr);
  EXPECT_TRUE(graph->FindNode("slice1") == nullptr);
}


TEST_F(RedundantOpRemovePassTest, RemoveSlice) {
  auto dtype = DT_INT32;
  auto shape = GeShape(std::vector<int64_t>({2}));
  GeTensorDesc desc(shape, FORMAT_ND, dtype);
  auto num_elements = shape.GetShapeSize();
  std::vector<int32_t> begin_buffer{0, 0};
  auto begin_tensor = std::make_shared<GeTensor>(desc, reinterpret_cast<uint8_t *>(begin_buffer.data()),
                                                 GetSizeByDataType(dtype) * num_elements);
  auto const_0 = OP_CFG(CONSTANT)
                     .TensorDesc(desc.GetFormat(), dtype, shape.GetDims())
                     .OutCnt(1)
                     .Weight(begin_tensor)
                     .Build("const_0");

  std::vector<int32_t> end_buffer{128, 256};
  auto end_tensor = std::make_shared<GeTensor>(desc, reinterpret_cast<uint8_t *>(end_buffer.data()),
                                               GetSizeByDataType(dtype) * num_elements);
  auto const_1 = OP_CFG(CONSTANT)
                     .TensorDesc(desc.GetFormat(), dtype, shape.GetDims())
                     .OutCnt(1)
                     .Weight(end_tensor)
                     .Build("const_1");

  auto data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {128, 256}).InCnt(0).OutCnt(1).Build("Data");
  auto slice1 = OP_CFG("Slice")
                    .TensorDesc(FORMAT_NCHW, DT_FLOAT, {128, 256})
                    .InCnt(3)
                    .OutCnt(1)
                    .InNames({"x", "offsets", "size"})
                    .Build("slice1");
  auto relu = OP_CFG("Relu").TensorDesc(FORMAT_NCHW, DT_FLOAT, {128, 256}).InCnt(1).OutCnt(1).Build("relu");

  DEF_GRAPH(test_graph) {
    CHAIN(NODE(data)->NODE(slice1)->NODE(relu)->NODE("output_0", NETOUTPUT));
    CHAIN(NODE(const_0)->DATA_EDGE(0, 1)->NODE(slice1));
    CHAIN(NODE(const_1)->DATA_EDGE(0, 2)->NODE(slice1));
  };
  auto graph = ToComputeGraph(test_graph);

  bool changed = false;
  EXPECT_EQ(RedundantSliceRemovePass().Run(graph, changed), GRAPH_SUCCESS);
  EXPECT_TRUE(graph->FindNode("slice1") == nullptr);
  auto relu_node = graph->FindNode("relu");
  ASSERT_NE(relu_node, nullptr);
  EXPECT_EQ(relu_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "Data");
}

TEST_F(RedundantOpRemovePassTest, SkipRemoveSlice) {
  auto dtype = DT_INT32;
  auto shape = GeShape(std::vector<int64_t>({2}));
  GeTensorDesc desc(shape, FORMAT_ND, dtype);
  auto num_elements = shape.GetShapeSize();
  std::vector<int32_t> begin_buffer{0, 0};
  auto begin_tensor = std::make_shared<GeTensor>(desc, reinterpret_cast<uint8_t *>(begin_buffer.data()),
                                                 GetSizeByDataType(dtype) * num_elements);
  auto const_0 = OP_CFG(CONSTANT)
                     .TensorDesc(desc.GetFormat(), dtype, shape.GetDims())
                     .OutCnt(1)
                     .Weight(begin_tensor)
                     .Build("const_0");

  std::vector<int32_t> end_buffer{128, 128};
  auto end_tensor = std::make_shared<GeTensor>(desc, reinterpret_cast<uint8_t *>(end_buffer.data()),
                                               GetSizeByDataType(dtype) * num_elements);
  auto const_1 = OP_CFG(CONSTANT)
                     .TensorDesc(desc.GetFormat(), dtype, shape.GetDims())
                     .OutCnt(1)
                     .Weight(end_tensor)
                     .Build("const_1");

  auto data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {128, 256}).InCnt(0).OutCnt(1).Build("Data");
  auto slice1 = OP_CFG("Slice")
                    .TensorDesc(FORMAT_NCHW, DT_FLOAT, {128, 256})
                    .InCnt(3)
                    .OutCnt(1)
                    .InNames({"x", "offsets", "size"})
                    .Build("slice1");
  auto relu = OP_CFG("Relu").TensorDesc(FORMAT_NCHW, DT_FLOAT, {128, 256}).InCnt(1).OutCnt(1).Build("relu");
  auto slice2 = OP_CFG("Slice")
                    .TensorDesc(FORMAT_NCHW, DT_FLOAT, {128, 256})
                    .InCnt(3)
                    .OutCnt(1)
                    .InNames({"x", "offsets", "size"})
                    .Build("slice2");

  DEF_GRAPH(test_graph) {
    CHAIN(NODE(data)->NODE(slice1)->NODE(relu)->NODE("output_0", NETOUTPUT));
    CHAIN(NODE(const_0)->DATA_EDGE(0, 1)->NODE(slice1));
    CHAIN(NODE(const_1)->DATA_EDGE(0, 2)->NODE(slice1));

    CHAIN(NODE(data)->NODE(slice2)->NODE("output_0", NETOUTPUT));
    CHAIN(NODE(data)->DATA_EDGE(0, 1)->NODE(slice2));
    CHAIN(NODE(data)->DATA_EDGE(0, 2)->NODE(slice2));
  };
  auto graph = ToComputeGraph(test_graph);

  bool changed = false;
  EXPECT_EQ(RedundantSliceRemovePass().Run(graph, changed), GRAPH_SUCCESS);
  EXPECT_TRUE(graph->FindNode("slice1") != nullptr);
  EXPECT_TRUE(graph->FindNode("slice2") != nullptr);
  auto relu_node = graph->FindNode("relu");
  ASSERT_NE(relu_node, nullptr);
  EXPECT_EQ(relu_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "slice1");
}

// 测试混合情况：部分维度 size=-1
TEST_F(RedundantOpRemovePassTest, RemoveSliceWithMixedMinusOneSize) {
  auto dtype = DT_INT32;
  auto shape = GeShape(std::vector<int64_t>({3}));
  GeTensorDesc desc(shape, FORMAT_ND, dtype);
  auto num_elements = shape.GetShapeSize();

  // offsets = [0, 0, 0]
  std::vector<int32_t> begin_buffer{0, 0, 0};
  auto begin_tensor = std::make_shared<GeTensor>(desc, reinterpret_cast<uint8_t *>(begin_buffer.data()),
                                                 GetSizeByDataType(dtype) * num_elements);
  auto const_0 = OP_CFG(CONSTANT)
                     .TensorDesc(desc.GetFormat(), dtype, shape.GetDims())
                     .OutCnt(1)
                     .Weight(begin_tensor)
                     .Build("const_0");

  // size = [64, -1, 128]，第二个维度用-1
  std::vector<int32_t> end_buffer{64, -1, 128};
  auto end_tensor = std::make_shared<GeTensor>(desc, reinterpret_cast<uint8_t *>(end_buffer.data()),
                                               GetSizeByDataType(dtype) * num_elements);
  auto const_1 = OP_CFG(CONSTANT)
                     .TensorDesc(desc.GetFormat(), dtype, shape.GetDims())
                     .OutCnt(1)
                     .Weight(end_tensor)
                     .Build("const_1");

  auto data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {64, 128, 128}).InCnt(0).OutCnt(1).Build("Data");
  auto slice1 = OP_CFG("Slice")
                    .TensorDesc(FORMAT_NCHW, DT_FLOAT, {64, 128, 128})
                    .InCnt(3)
                    .OutCnt(1)
                    .InNames({"x", "offsets", "size"})
                    .Build("slice1");
  auto relu = OP_CFG("Relu").TensorDesc(FORMAT_NCHW, DT_FLOAT, {64, 128, 128}).InCnt(1).OutCnt(1).Build("relu");

  DEF_GRAPH(test_graph) {
    CHAIN(NODE(data)->NODE(slice1)->NODE(relu)->NODE("output_0", NETOUTPUT));
    CHAIN(NODE(const_0)->DATA_EDGE(0, 1)->NODE(slice1));
    CHAIN(NODE(const_1)->DATA_EDGE(0, 2)->NODE(slice1));
  };
  auto graph = ToComputeGraph(test_graph);

  bool changed = false;
  EXPECT_EQ(RedundantSliceRemovePass().Run(graph, changed), GRAPH_SUCCESS);
  // [0,0,0] + [64,-1,128] 对于输入 [64,128,128] 是冗余的
  EXPECT_TRUE(graph->FindNode("slice1") == nullptr);
  auto relu_node = graph->FindNode("relu");
  ASSERT_NE(relu_node, nullptr);
  EXPECT_EQ(relu_node->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode()->GetName(), "Data");
}
}  // namespace ge
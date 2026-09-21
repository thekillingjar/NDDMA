/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "lowering/exe_graph_serializer.h"
#include <gtest/gtest.h>
#include "exe_graph/lowering/value_holder.h"

#include "common/bg_test.h"
#include "common/share_graph.h"
#include "runtime/model_desc.h"
#include "graph/debug/ge_attr_define.h"
#include "faker/model_desc_holder_faker.h"

namespace gert {
class ExeGraphSerializerUT : public bg::BgTest {};
TEST_F(ExeGraphSerializerUT, SerializeModelDesc_Success) {
  auto compute_graph = ShareGraph::BuildSingleNodeGraph("Add", {{-1, 2, 3, 4}, {1, -1, 3, 4}, {-1}, {-1, -1, 3, 4}},
                                                        {{1, 2, 3, 4}, {1, 1, 3, 4}, {1}, {1, 1, 3, 4}},
                                                        {{100, 2, 3, 4}, {1, 100, 3, 4}, {100}, {100, 100, 3, 4}});
  auto netout_node = compute_graph->FindNode("NetOutput");
  const auto &op_desc = netout_node->GetOpDesc();
  auto input_desc = op_desc->MutableInputDesc(static_cast<uint32_t>(0U));

  auto frame = bg::ValueHolder::GetCurrentFrame();
  bg::BufferPool bp;
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build(2, 1);
  model_desc_holder.SetOutputNodeName({"add:0"});
  ASSERT_EQ(ExeGraphSerializer(*frame).SetComputeGraph(compute_graph).SetModelDescHolder(&model_desc_holder).SerializeModelDesc(bp), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(bp.GetSize() >= 1);
  auto model_desc = reinterpret_cast<const ModelDesc *>(bp.GetBufById(bp.GetSize() - 1));
  ASSERT_NE(model_desc, nullptr);

  ASSERT_EQ(model_desc->GetInputNum(), 2);
  ASSERT_EQ(model_desc->GetOutputNum(), 1);
  ASSERT_EQ(model_desc->GetReusableStreamNum(), 2);
  ASSERT_EQ(model_desc->GetReusableEventNum(), 1);

  auto input0 = model_desc->GetInputDesc(0);
  ASSERT_NE(input0, nullptr);
  auto index = reinterpret_cast<size_t>(input0->GetName());
  ASSERT_STREQ(bp.GetBufById(index), "data1");
  EXPECT_EQ(input0->GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(input0->GetOriginFormat(), ge::FORMAT_ND);
  EXPECT_EQ(input0->GetStorageFormat(), ge::FORMAT_ND);
  Shape expect_shape{{-1, 2, 3, 4}};
  EXPECT_EQ(input0->GetOriginShape(), expect_shape);
  EXPECT_EQ(input0->GetStorageShape(), expect_shape);
  ShapeRange expect_shape_range{{1, 2, 3, 4}, {100, 2, 3, 4}};
  EXPECT_EQ(input0->GetOriginShapeRange(), expect_shape_range);
  EXPECT_EQ(input0->GetStorageShapeRange(), expect_shape_range);

  auto input1 = model_desc->GetInputDesc(1);
  ASSERT_NE(input1, nullptr);
  index = reinterpret_cast<size_t>(input1->GetName());
  ASSERT_STREQ(bp.GetBufById(index), "data2");
  EXPECT_EQ(input1->GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(input1->GetOriginFormat(), ge::FORMAT_ND);
  EXPECT_EQ(input1->GetStorageFormat(), ge::FORMAT_ND);
  expect_shape = Shape{{1, -1, 3, 4}};
  EXPECT_EQ(input1->GetOriginShape(), expect_shape);
  EXPECT_EQ(input1->GetStorageShape(), expect_shape);
  expect_shape_range = ShapeRange{{1, 1, 3, 4}, {1, 100, 3, 4}};
  EXPECT_EQ(input1->GetOriginShapeRange(), expect_shape_range);
  EXPECT_EQ(input1->GetStorageShapeRange(), expect_shape_range);

  auto output0 = model_desc->GetOutputDesc(0);
  ASSERT_NE(output0, nullptr);
  index = reinterpret_cast<size_t>(output0->GetName());
  ASSERT_STREQ(bp.GetBufById(index), "add:0");
  EXPECT_EQ(output0->GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(output0->GetOriginFormat(), ge::FORMAT_ND);
  EXPECT_EQ(output0->GetStorageFormat(), ge::FORMAT_ND);
  expect_shape = Shape{{-1, -1, 3, 4}};
  EXPECT_EQ(output0->GetOriginShape(), expect_shape);
  EXPECT_EQ(output0->GetStorageShape(), expect_shape);
  expect_shape_range = ShapeRange{{1, 1, 3, 4}, {100, 100, 3, 4}};
  EXPECT_EQ(output0->GetOriginShapeRange(), expect_shape_range);
  EXPECT_EQ(output0->GetStorageShapeRange(), expect_shape_range);
}

TEST_F(ExeGraphSerializerUT, SerializeModelDesc_Success_Without_ATTR_NAME_ORIGIN_OUTPUT_TENSOR_NAME) {
  auto compute_graph = ShareGraph::BuildSingleNodeGraph("Add", {{-1, 2, 3, 4}, {1, -1, 3, 4}, {-1}, {-1, -1, 3, 4}},
                                                        {{1, 2, 3, 4}, {1, 1, 3, 4}, {1}, {1, 1, 3, 4}},
                                                        {{100, 2, 3, 4}, {1, 100, 3, 4}, {100}, {100, 100, 3, 4}});
  auto frame = bg::ValueHolder::GetCurrentFrame();
  bg::BufferPool bp;
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build(2, 1);
  model_desc_holder.SetOutputNodeName({"add:0"});
  ASSERT_EQ(ExeGraphSerializer(*frame).SetComputeGraph(compute_graph).SetModelDescHolder(&model_desc_holder).SerializeModelDesc(bp), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(bp.GetSize() >= 1);
  auto model_desc = reinterpret_cast<const ModelDesc *>(bp.GetBufById(bp.GetSize() - 1));
  ASSERT_NE(model_desc, nullptr);

  ASSERT_EQ(model_desc->GetInputNum(), 2);
  ASSERT_EQ(model_desc->GetOutputNum(), 1);

  auto output0 = model_desc->GetOutputDesc(0);
  ASSERT_NE(output0, nullptr);
  auto index = reinterpret_cast<size_t>(output0->GetName());
  ASSERT_STREQ(bp.GetBufById(index), "add:0");
  EXPECT_EQ(output0->GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(output0->GetOriginFormat(), ge::FORMAT_ND);
  EXPECT_EQ(output0->GetStorageFormat(), ge::FORMAT_ND);
  auto expect_shape = Shape{{-1, -1, 3, 4}};
  EXPECT_EQ(output0->GetOriginShape(), expect_shape);
  EXPECT_EQ(output0->GetStorageShape(), expect_shape);
  auto expect_shape_range = ShapeRange{{1, 1, 3, 4}, {100, 100, 3, 4}};
  EXPECT_EQ(output0->GetOriginShapeRange(), expect_shape_range);
  EXPECT_EQ(output0->GetStorageShapeRange(), expect_shape_range);
}

TEST_F(ExeGraphSerializerUT, SerializeModelDesc_Failed) {
  auto compute_graph = ShareGraph::BuildSingleNodeGraph("Add", {{-1, 2, 3, 4}, {1, -1, 3, 4}, {-1}, {-1, -1, 3, 4}},
                                                        {{1, 2, 3, 4}, {1, 1, 3, 4}, {1}, {1, 1, 3, 4}},
                                                        {{100, 2, 3, 4}, {1, 100, 3, 4}, {100}, {100, 100, 3, 4}});
  auto netout_node = compute_graph->FindNode("NetOutput");
  const auto &op_desc = netout_node->GetOpDesc();
  auto input_desc = op_desc->MutableInputDesc(static_cast<uint32_t>(0U));
  const std::string out_tensor_name = "out_tensor_name";
  ge::AttrUtils::SetStr(input_desc, ge::ATTR_NAME_ORIGIN_OUTPUT_TENSOR_NAME, out_tensor_name);
  auto add_node = compute_graph->FindNode("add1");
  // Netoutput 对端出现了空的Anchor，与原来逻辑保持一致，跳过此处理
  ge::GraphUtils::RemoveEdge(netout_node->GetInAnchor(0U), add_node->GetOutAnchor(0U));

  auto frame = bg::ValueHolder::GetCurrentFrame();
  bg::BufferPool bp;
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build(2, 1);
  ASSERT_EQ(ExeGraphSerializer(*frame).SetComputeGraph(compute_graph).SetModelDescHolder(&model_desc_holder).SerializeModelDesc(bp), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(bp.GetSize() >= 1);
  auto model_desc = reinterpret_cast<const ModelDesc *>(bp.GetBufById(bp.GetSize() - 1));
  ASSERT_NE(model_desc, nullptr);

  ASSERT_EQ(model_desc->GetInputNum(), 2);
  ASSERT_EQ(model_desc->GetOutputNum(), 1);
}

TEST_F(ExeGraphSerializerUT, SerializeModelDesc_CheckInputOrder) {
  auto compute_graph = ShareGraph::BuildSingleNodeGraph("Add", {{-1, 2, 3, 4}, {1, -1, 3, 4}, {-1}, {-1, -1, 3, 4}},
                                                        {{1, 2, 3, 4}, {1, 1, 3, 4}, {1}, {1, 1, 3, 4}},
                                                        {{100, 2, 3, 4}, {1, 100, 3, 4}, {100}, {100, 100, 3, 4}});
  ASSERT_NE(compute_graph, nullptr);

  auto data1_node = compute_graph->FindNode("data1");
  ASSERT_NE(data1_node, nullptr);
  ASSERT_TRUE(ge::AttrUtils::SetInt(data1_node->GetOpDesc(), ge::ATTR_NAME_INDEX, 200));

  auto data2_node = compute_graph->FindNode("data2");  
  ASSERT_NE(data2_node, nullptr);
  ASSERT_TRUE(ge::AttrUtils::SetInt(data2_node->GetOpDesc(), ge::ATTR_NAME_INDEX, 100));

  auto frame = bg::ValueHolder::GetCurrentFrame();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  bg::BufferPool bp;
  ASSERT_EQ(ExeGraphSerializer(*frame).SetComputeGraph(compute_graph).SetModelDescHolder(&model_desc_holder).SerializeModelDesc(bp), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(bp.GetSize() >= 1);
  auto model_desc = reinterpret_cast<const ModelDesc *>(bp.GetBufById(bp.GetSize() - 1));
  ASSERT_NE(model_desc, nullptr);

  ASSERT_EQ(model_desc->GetInputNum(), 2);
  ASSERT_EQ(model_desc->GetOutputNum(), 1);

  auto input0 = model_desc->GetInputDesc(0);
  ASSERT_NE(input0, nullptr);
  auto index = reinterpret_cast<size_t>(input0->GetName());
  ASSERT_STREQ(bp.GetBufById(index), "data2");

  auto input1 = model_desc->GetInputDesc(1);
  ASSERT_NE(input1, nullptr);
  index = reinterpret_cast<size_t>(input1->GetName());
  ASSERT_STREQ(bp.GetBufById(index), "data1");

  ASSERT_TRUE(ge::AttrUtils::SetInt(data1_node->GetOpDesc(), ge::ATTR_NAME_INDEX, 100));
  ASSERT_TRUE(ge::AttrUtils::SetInt(data2_node->GetOpDesc(), ge::ATTR_NAME_INDEX, 200));
  ASSERT_EQ(ExeGraphSerializer(*frame).SetComputeGraph(compute_graph).SetModelDescHolder(&model_desc_holder).SerializeModelDesc(bp), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(bp.GetSize() >= 1);
  model_desc = reinterpret_cast<const ModelDesc *>(bp.GetBufById(bp.GetSize() - 1));
  ASSERT_NE(model_desc, nullptr);

  ASSERT_EQ(model_desc->GetInputNum(), 2);
  ASSERT_EQ(model_desc->GetOutputNum(), 1);

  input0 = model_desc->GetInputDesc(0);
  ASSERT_NE(input0, nullptr);
  index = reinterpret_cast<size_t>(input0->GetName());
  ASSERT_STREQ(bp.GetBufById(index), "data1");

  input1 = model_desc->GetInputDesc(1);
  ASSERT_NE(input1, nullptr);
  index = reinterpret_cast<size_t>(input1->GetName());
  ASSERT_STREQ(bp.GetBufById(index), "data2");
}
}  // namespace gert
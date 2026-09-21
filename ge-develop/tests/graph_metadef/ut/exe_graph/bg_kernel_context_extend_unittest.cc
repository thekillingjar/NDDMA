/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/lowering/bg_kernel_context_extend.h"
#include <gtest/gtest.h>
#include <memory>
#include <algorithm>
#include <cstdint>
#include "graph/compute_graph.h"
#include "exe_graph/runtime/context_extend.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "exe_graph/runtime/tensor.h"
#include "graph/debug/ge_attr_define.h"
#include "expand_dimension.h"
#include "graph/utils/graph_utils.h"
#include "runtime/runtime_attrs_def.h"
namespace {
bool isMemoryCleared(const uint8_t *ptr, size_t size) {
  if (ptr == nullptr) {
    return false;
  }
  return std::all_of(ptr, ptr + size, [](uint8_t byte) { return byte == 0; });
}
}
namespace gert {
class BgKernelContextExtendUT : public testing::Test {};
TEST_F(BgKernelContextExtendUT, BuildRequiredInput) {
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  ge::GeTensorDesc tensor_desc;
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetFormat(ge::FORMAT_NC1HWC0);
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetShape(ge::GeShape({8, 1, 224, 224, 16}));
  tensor_desc.SetOriginShape(ge::GeShape({8, 3, 224, 224}));
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AppendIrInput("x", ge::kIrInputRequired);

  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);
  auto data_op_desc = std::make_shared<ge::OpDesc>("data", "Data");
  data_op_desc->AddOutputDesc("x", tensor_desc);
  auto data_node = graph->AddNode(data_op_desc);
  ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), node->GetInDataAnchor(0));

  bg::BufferPool buffer_pool;
  auto ret = bg::CreateComputeNodeInfo(node, buffer_pool);
  ASSERT_NE(ret, nullptr);
  auto compute_node_info = reinterpret_cast<ComputeNodeInfo *>(ret.get());
  ASSERT_EQ(compute_node_info->GetInputsNum(), 1);
  ASSERT_EQ(compute_node_info->GetOutputsNum(), 0);
  ASSERT_EQ(compute_node_info->GetIrInputsNum(), 1);
  auto td = compute_node_info->GetInputTdInfo(0);
  ASSERT_NE(td, nullptr);
  EXPECT_EQ(td->GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(td->GetOriginFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(td->GetStorageFormat(), ge::FORMAT_NC1HWC0);
  auto expand_dims_type = td->GetExpandDimsType();
  Shape origin_shape({8, 3, 224, 224});
  Shape storage_shape;
  expand_dims_type.Expand(origin_shape, storage_shape);
  EXPECT_EQ(storage_shape, origin_shape);

  auto ins_info = compute_node_info->GetInputInstanceInfo(0);
  ASSERT_NE(ins_info, nullptr);
  EXPECT_EQ(ins_info->GetInstanceNum(), 1);
  EXPECT_EQ(ins_info->GetInstanceStart(), 0);

  EXPECT_EQ(compute_node_info->GetAttrs()->GetAttrNum(), 0);
  EXPECT_EQ(compute_node_info->GetAttrs()->GetAttrPointer<char>(0), nullptr);
}
TEST_F(BgKernelContextExtendUT, BuildEmptyRequiredInput) {
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  ge::GeTensorDesc tensor_desc;
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetFormat(ge::FORMAT_NC1HWC0);
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetShape(ge::GeShape({8, 1, 224, 224, 16}));
  tensor_desc.SetOriginShape(ge::GeShape({8, 3, 224, 224}));
  op_desc->AddInputDesc("x1", tensor_desc);
  op_desc->AppendIrInput("x1", ge::kIrInputRequired);
  op_desc->AppendIrInput("x2", ge::kIrInputRequired);

  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);

  bg::BufferPool buffer_pool;
  auto ret = bg::CreateComputeNodeInfo(node, buffer_pool);
  EXPECT_NE(ret, nullptr);
}
TEST_F(BgKernelContextExtendUT, UknownInputFailed) {
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  ge::GeTensorDesc tensor_desc;
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetFormat(ge::FORMAT_NC1HWC0);
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetShape(ge::GeShape({8, 1, 224, 224, 16}));
  tensor_desc.SetOriginShape(ge::GeShape({8, 3, 224, 224}));
  op_desc->AddInputDesc("x2", tensor_desc);
  op_desc->AppendIrInput("x1", ge::kIrInputOptional);

  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);

  bg::BufferPool buffer_pool;
  auto ret = bg::CreateComputeNodeInfo(node, buffer_pool);
  EXPECT_NE(ret, nullptr);

  auto compute_node_info = reinterpret_cast<ComputeNodeInfo *>(ret.get());
  auto ins_info = compute_node_info->GetInputInstanceInfo(0);
  ASSERT_NE(ins_info, nullptr);
  EXPECT_EQ(ins_info->GetInstanceNum(), 0);
}
TEST_F(BgKernelContextExtendUT, BuildWithOptionalInputs) {
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  ge::GeTensorDesc tensor_desc;
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetFormat(ge::FORMAT_NC1HWC0);
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetShape(ge::GeShape({8, 1, 224, 224, 16}));
  tensor_desc.SetOriginShape(ge::GeShape({8, 3, 224, 224}));
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AppendIrInput("x", ge::kIrInputOptional);

  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);
  auto data_op_desc = std::make_shared<ge::OpDesc>("data", "Data");
  data_op_desc->AddOutputDesc("x", tensor_desc);
  auto data_node = graph->AddNode(data_op_desc);
  ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), node->GetInDataAnchor(0));

  bg::BufferPool buffer_pool;
  auto ret = bg::CreateComputeNodeInfo(node, buffer_pool);
  ASSERT_NE(ret, nullptr);
  auto compute_node_info = reinterpret_cast<ComputeNodeInfo *>(ret.get());
  ASSERT_EQ(compute_node_info->GetInputsNum(), 1);
  ASSERT_EQ(compute_node_info->GetOutputsNum(), 0);
  ASSERT_EQ(compute_node_info->GetIrInputsNum(), 1);
  auto td = compute_node_info->GetInputTdInfo(0);
  ASSERT_NE(td, nullptr);
  EXPECT_EQ(td->GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(td->GetOriginFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(td->GetStorageFormat(), ge::FORMAT_NC1HWC0);

  auto ins_info = compute_node_info->GetInputInstanceInfo(0);
  ASSERT_NE(ins_info, nullptr);
  EXPECT_EQ(ins_info->GetInstanceNum(), 1);
  EXPECT_EQ(ins_info->GetInstanceStart(), 0);

  EXPECT_EQ(compute_node_info->GetAttrs()->GetAttrNum(), 0);
  EXPECT_EQ(compute_node_info->GetAttrs()->GetAttrPointer<char>(0), nullptr);
}
TEST_F(BgKernelContextExtendUT, BuildWithOptionalInputsNotExists) {
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  ge::GeTensorDesc tensor_desc;
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetFormat(ge::FORMAT_NC1HWC0);
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetShape(ge::GeShape({8, 1, 224, 224, 16}));
  tensor_desc.SetOriginShape(ge::GeShape({8, 3, 224, 224}));
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AppendIrInput("y", ge::kIrInputOptional);
  op_desc->AppendIrInput("x", ge::kIrInputOptional);

  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);
  auto data_op_desc = std::make_shared<ge::OpDesc>("data", "Data");
  data_op_desc->AddOutputDesc("x", tensor_desc);
  auto data_node = graph->AddNode(data_op_desc);
  ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), node->GetInDataAnchor(0));

  bg::BufferPool buffer_pool;
  auto ret = bg::CreateComputeNodeInfo(node, buffer_pool);
  ASSERT_NE(ret, nullptr);
  auto compute_node_info = reinterpret_cast<ComputeNodeInfo *>(ret.get());
  ASSERT_EQ(compute_node_info->GetInputsNum(), 1);
  ASSERT_EQ(compute_node_info->GetOutputsNum(), 0);
  ASSERT_EQ(compute_node_info->GetIrInputsNum(), 2);
  auto td = compute_node_info->GetInputTdInfo(0);
  ASSERT_NE(td, nullptr);
  EXPECT_EQ(td->GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(td->GetOriginFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(td->GetStorageFormat(), ge::FORMAT_NC1HWC0);

  auto ins_info = compute_node_info->GetInputInstanceInfo(0);
  ASSERT_NE(ins_info, nullptr);
  EXPECT_EQ(ins_info->GetInstanceNum(), 0);
  ins_info = compute_node_info->GetInputInstanceInfo(1);
  ASSERT_NE(ins_info, nullptr);
  EXPECT_EQ(ins_info->GetInstanceNum(), 1);
  EXPECT_EQ(ins_info->GetInstanceStart(), 0);

  EXPECT_EQ(compute_node_info->GetAttrs()->GetAttrNum(), 0);
  EXPECT_EQ(compute_node_info->GetAttrs()->GetAttrPointer<char>(0), nullptr);
}
TEST_F(BgKernelContextExtendUT, BuildWithMultipleOptionalInputsIns) {
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  ge::GeTensorDesc tensor_desc;
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetFormat(ge::FORMAT_NC1HWC0);
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetShape(ge::GeShape({8, 1, 224, 224, 16}));
  tensor_desc.SetOriginShape(ge::GeShape({8, 3, 224, 224}));
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddInputDesc("y", tensor_desc);
  op_desc->AppendIrInput("x", ge::kIrInputOptional);
  op_desc->AppendIrInput("y", ge::kIrInputOptional);
  op_desc->AppendIrInput("z", ge::kIrInputOptional);

  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);

  auto data_op_desc = std::make_shared<ge::OpDesc>("data", "Data");
  data_op_desc->AddOutputDesc("x", tensor_desc);
  data_op_desc->AddOutputDesc("y", tensor_desc);
  auto data_node = graph->AddNode(data_op_desc);
  ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(1), node->GetInDataAnchor(1));

  bg::BufferPool buffer_pool;
  auto ret = bg::CreateComputeNodeInfo(node, buffer_pool);
  ASSERT_NE(ret, nullptr);
  auto compute_node_info = reinterpret_cast<ComputeNodeInfo *>(ret.get());
  ASSERT_EQ(compute_node_info->GetInputsNum(), 2);
  ASSERT_EQ(compute_node_info->GetOutputsNum(), 0);
  ASSERT_EQ(compute_node_info->GetIrInputsNum(), 3);
  auto td = compute_node_info->GetInputTdInfo(0);
  ASSERT_NE(td, nullptr);
  EXPECT_EQ(td->GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(td->GetOriginFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(td->GetStorageFormat(), ge::FORMAT_NC1HWC0);

  auto ins_info = compute_node_info->GetInputInstanceInfo(0);
  ASSERT_NE(ins_info, nullptr);
  EXPECT_EQ(ins_info->GetInstanceNum(), 1);
  EXPECT_EQ(ins_info->GetInstanceStart(), 0);
  ins_info = compute_node_info->GetInputInstanceInfo(1);
  ASSERT_NE(ins_info, nullptr);
  EXPECT_EQ(ins_info->GetInstanceNum(), 1);
  EXPECT_EQ(ins_info->GetInstanceStart(), 1);
  ins_info = compute_node_info->GetInputInstanceInfo(2);
  ASSERT_NE(ins_info, nullptr);
  EXPECT_EQ(ins_info->GetInstanceNum(), 0);

  EXPECT_EQ(compute_node_info->GetAttrs()->GetAttrNum(), 0);
  EXPECT_EQ(compute_node_info->GetAttrs()->GetAttrPointer<char>(0), nullptr);
}
TEST_F(BgKernelContextExtendUT, BuildWithDynamicInputs) {
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  ge::GeTensorDesc tensor_desc;
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetFormat(ge::FORMAT_NC1HWC0);
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetShape(ge::GeShape({8, 1, 224, 224, 16}));
  tensor_desc.SetOriginShape(ge::GeShape({8, 3, 224, 224}));
  op_desc->AddInputDesc("x0", tensor_desc);
  op_desc->AppendIrInput("x", ge::kIrInputDynamic);

  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);
  auto data_op_desc = std::make_shared<ge::OpDesc>("data", "Data");
  data_op_desc->AddOutputDesc("x", tensor_desc);
  auto data_node = graph->AddNode(data_op_desc);
  ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), node->GetInDataAnchor(0));

  bg::BufferPool buffer_pool;
  auto ret = bg::CreateComputeNodeInfo(node, buffer_pool);
  ASSERT_NE(ret, nullptr);
  auto compute_node_info = reinterpret_cast<ComputeNodeInfo *>(ret.get());
  ASSERT_EQ(compute_node_info->GetInputsNum(), 1);
  ASSERT_EQ(compute_node_info->GetOutputsNum(), 0);
  ASSERT_EQ(compute_node_info->GetIrInputsNum(), 1);
  auto td = compute_node_info->GetInputTdInfo(0);
  ASSERT_NE(td, nullptr);
  EXPECT_EQ(td->GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(td->GetOriginFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(td->GetStorageFormat(), ge::FORMAT_NC1HWC0);

  auto ins_info = compute_node_info->GetInputInstanceInfo(0);
  ASSERT_NE(ins_info, nullptr);
  EXPECT_EQ(ins_info->GetInstanceNum(), 1);
  EXPECT_EQ(ins_info->GetInstanceStart(), 0);

  EXPECT_EQ(compute_node_info->GetAttrs()->GetAttrNum(), 0);
  EXPECT_EQ(compute_node_info->GetAttrs()->GetAttrPointer<char>(0), nullptr);
}
TEST_F(BgKernelContextExtendUT, BuildWithMultiInstanceDynamicInputs) {
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  ge::GeTensorDesc tensor_desc;
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetFormat(ge::FORMAT_NC1HWC0);
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetShape(ge::GeShape({8, 1, 224, 224, 16}));
  tensor_desc.SetOriginShape(ge::GeShape({8, 3, 224, 224}));
  op_desc->AddInputDesc("x0", tensor_desc);
  op_desc->AddInputDesc("x1", tensor_desc);
  op_desc->AddInputDesc("y0", tensor_desc);
  op_desc->AddInputDesc("y1", tensor_desc);
  op_desc->AddInputDesc("y2", tensor_desc);
  op_desc->AppendIrInput("x", ge::kIrInputDynamic);
  op_desc->AppendIrInput("y", ge::kIrInputDynamic);

  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);
  auto data_op_desc = std::make_shared<ge::OpDesc>("data", "Data");
  data_op_desc->AddOutputDesc("x0", tensor_desc);
  data_op_desc->AddOutputDesc("x1", tensor_desc);
  data_op_desc->AddOutputDesc("y0", tensor_desc);
  data_op_desc->AddOutputDesc("y1", tensor_desc);
  data_op_desc->AddOutputDesc("y2", tensor_desc);
  auto data_node = graph->AddNode(data_op_desc);
  ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(1), node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(2), node->GetInDataAnchor(2));
  ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(3), node->GetInDataAnchor(3));
  ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(4), node->GetInDataAnchor(4));

  bg::BufferPool buffer_pool;
  auto ret = bg::CreateComputeNodeInfo(node, buffer_pool);
  ASSERT_NE(ret, nullptr);
  auto compute_node_info = reinterpret_cast<ComputeNodeInfo *>(ret.get());
  ASSERT_EQ(compute_node_info->GetInputsNum(), 5);
  ASSERT_EQ(compute_node_info->GetOutputsNum(), 0);
  ASSERT_EQ(compute_node_info->GetIrInputsNum(), 2);
  auto td = compute_node_info->GetInputTdInfo(0);
  ASSERT_NE(td, nullptr);
  EXPECT_EQ(td->GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(td->GetOriginFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(td->GetStorageFormat(), ge::FORMAT_NC1HWC0);

  auto ins_info = compute_node_info->GetInputInstanceInfo(0);
  ASSERT_NE(ins_info, nullptr);
  EXPECT_EQ(ins_info->GetInstanceNum(), 2);
  EXPECT_EQ(ins_info->GetInstanceStart(), 0);
  ins_info = compute_node_info->GetInputInstanceInfo(1);
  ASSERT_NE(ins_info, nullptr);
  EXPECT_EQ(ins_info->GetInstanceNum(), 3);
  EXPECT_EQ(ins_info->GetInstanceStart(), 2);

  EXPECT_EQ(compute_node_info->GetAttrs()->GetAttrNum(), 0);
  EXPECT_EQ(compute_node_info->GetAttrs()->GetAttrPointer<char>(0), nullptr);
}
TEST_F(BgKernelContextExtendUT, BuildWithEmptyDynamicInputs) {
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  ge::GeTensorDesc tensor_desc;
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetFormat(ge::FORMAT_NC1HWC0);
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetShape(ge::GeShape({8, 1, 224, 224, 16}));
  tensor_desc.SetOriginShape(ge::GeShape({8, 3, 224, 224}));
  op_desc->AddInputDesc("x0", tensor_desc);
  op_desc->AddInputDesc("x1", tensor_desc);
  op_desc->AppendIrInput("y", ge::kIrInputDynamic);
  op_desc->AppendIrInput("x", ge::kIrInputDynamic);

  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);
  auto data_op_desc = std::make_shared<ge::OpDesc>("data", "Data");
  data_op_desc->AddOutputDesc("x0", tensor_desc);
  data_op_desc->AddOutputDesc("x1", tensor_desc);
  auto data_node = graph->AddNode(data_op_desc);
  ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(1), node->GetInDataAnchor(1));

  bg::BufferPool buffer_pool;
  auto ret = bg::CreateComputeNodeInfo(node, buffer_pool);
  ASSERT_NE(ret, nullptr);
  auto compute_node_info = reinterpret_cast<ComputeNodeInfo *>(ret.get());
  ASSERT_EQ(compute_node_info->GetInputsNum(), 2);
  ASSERT_EQ(compute_node_info->GetOutputsNum(), 0);
  ASSERT_EQ(compute_node_info->GetIrInputsNum(), 2);
  auto td = compute_node_info->GetInputTdInfo(0);
  ASSERT_NE(td, nullptr);
  EXPECT_EQ(td->GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(td->GetOriginFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(td->GetStorageFormat(), ge::FORMAT_NC1HWC0);

  auto ins_info = compute_node_info->GetInputInstanceInfo(0);
  ASSERT_NE(ins_info, nullptr);
  EXPECT_EQ(ins_info->GetInstanceNum(), 0);
  ins_info = compute_node_info->GetInputInstanceInfo(1);
  ASSERT_NE(ins_info, nullptr);
  EXPECT_EQ(ins_info->GetInstanceNum(), 2);
  EXPECT_EQ(ins_info->GetInstanceStart(), 0);

  EXPECT_EQ(compute_node_info->GetAttrs()->GetAttrNum(), 0);
  EXPECT_EQ(compute_node_info->GetAttrs()->GetAttrPointer<char>(0), nullptr);
}
TEST_F(BgKernelContextExtendUT, BuildWithOneAttr) {
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  ge::GeTensorDesc tensor_desc;
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetFormat(ge::FORMAT_NC1HWC0);
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetShape(ge::GeShape({8, 1, 224, 224, 16}));
  tensor_desc.SetOriginShape(ge::GeShape({8, 3, 224, 224}));
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AppendIrInput("x", ge::kIrInputRequired);
  op_desc->AddOutputDesc("y", tensor_desc);

  ge::AttrUtils::SetStr(op_desc, "a1", "Hello");
  op_desc->AppendIrAttrName("a1");

  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);
  auto data_op_desc = std::make_shared<ge::OpDesc>("data", "Data");
  data_op_desc->AddOutputDesc("x", tensor_desc);
  auto data_node = graph->AddNode(data_op_desc);
  ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), node->GetInDataAnchor(0));

  bg::BufferPool buffer_pool;
  auto ret = bg::CreateComputeNodeInfo(node, buffer_pool);
  ASSERT_NE(ret, nullptr);
  auto compute_node_info = reinterpret_cast<ComputeNodeInfo *>(ret.get());
  ASSERT_EQ(compute_node_info->GetInputsNum(), 1);
  ASSERT_EQ(compute_node_info->GetOutputsNum(), 1);
  ASSERT_EQ(compute_node_info->GetIrInputsNum(), 1);
  auto td = compute_node_info->GetInputTdInfo(0);
  ASSERT_NE(td, nullptr);
  EXPECT_EQ(td->GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(td->GetOriginFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(td->GetStorageFormat(), ge::FORMAT_NC1HWC0);

  auto ins_info = compute_node_info->GetInputInstanceInfo(0);
  ASSERT_NE(ins_info, nullptr);
  EXPECT_EQ(ins_info->GetInstanceNum(), 1);
  EXPECT_EQ(ins_info->GetInstanceStart(), 0);

  EXPECT_EQ(compute_node_info->GetAttrs()->GetAttrNum(), 1);
  ASSERT_NE(compute_node_info->GetAttrs()->GetAttrPointer<char>(0), nullptr);
  EXPECT_STREQ(compute_node_info->GetAttrs()->GetAttrPointer<char>(0), "Hello");
}
TEST_F(BgKernelContextExtendUT, BuildWithAttrs) {
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  ge::GeTensorDesc tensor_desc;
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetFormat(ge::FORMAT_NC1HWC0);
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetShape(ge::GeShape({8, 1, 224, 224, 16}));
  tensor_desc.SetOriginShape(ge::GeShape({8, 3, 224, 224}));
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AppendIrInput("x", ge::kIrInputRequired);
  op_desc->AddOutputDesc("y", tensor_desc);

  ge::GeTensorDesc ge_td;
  ge_td.SetOriginFormat(ge::FORMAT_NHWC);
  ge_td.SetFormat(ge::FORMAT_NC1HWC0);
  ge_td.SetDataType(ge::DT_FLOAT16);
  ge_td.SetOriginShape(ge::GeShape({8, 224, 224, 3}));
  ge_td.SetShape(ge::GeShape({8, 1, 224, 224, 16}));
  ge::GeTensor ge_tensor(ge_td);
  std::vector<uint16_t> fake_data(8 * 224 * 224 * 16);
  for (size_t i = 0; i < fake_data.size(); ++i) {
    fake_data[i] = static_cast<uint16_t>(rand() % std::numeric_limits<uint16_t>::max());
  }
  ge_tensor.SetData(reinterpret_cast<uint8_t *>(fake_data.data()), fake_data.size() * 2);

  ge::AttrUtils::SetStr(op_desc, "a1", "Hello");
  ge::AttrUtils::SetInt(op_desc, "a2", 10240);
  ge::AttrUtils::SetBool(op_desc, "a3", true);
  ge::AttrUtils::SetFloat(op_desc, "a4", 1024.0021);
  ge::AttrUtils::SetListInt(op_desc, "a5", std::vector<int32_t>({10, 200, 3000, 4096}));
  ge::AttrUtils::SetTensor(op_desc, "a6", ge_tensor);
  ge::AttrUtils::SetStr(op_desc, "b1", "World");
  ge::AttrUtils::SetInt(op_desc, "b2", 1024000);
  ge::AttrUtils::SetBool(op_desc, "b3", false);
  ge::AttrUtils::SetFloat(op_desc, "b4", 1024.1);
  ge::AttrUtils::SetListInt(op_desc, "b5", std::vector<int64_t>({10, 400, 3000, 8192}));
  ge::AttrUtils::SetTensor(op_desc, "b6", ge_tensor);
  ge::AttrUtils::SetListStr(op_desc, "c1", std::vector<std::string>({"hello", "world", "world1", "hello1"}));
  ge::AttrUtils::SetListDataType(op_desc, "c2", std::vector<ge::DataType>({ge::DT_FLOAT, ge::DT_STRING, ge::DT_UINT16, ge::DT_BOOL}));

  op_desc->AppendIrAttrName("b1");
  op_desc->AppendIrAttrName("b2");
  op_desc->AppendIrAttrName("b3");
  op_desc->AppendIrAttrName("b4");
  op_desc->AppendIrAttrName("b5");
  op_desc->AppendIrAttrName("b6");
  op_desc->AppendIrAttrName("a1");
  op_desc->AppendIrAttrName("a2");
  op_desc->AppendIrAttrName("a3");
  op_desc->AppendIrAttrName("a4");
  op_desc->AppendIrAttrName("a5");
  op_desc->AppendIrAttrName("a6");
  op_desc->AppendIrAttrName("c1");
  op_desc->AppendIrAttrName("c2");

  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);
  auto data_op_desc = std::make_shared<ge::OpDesc>("data", "Data");
  data_op_desc->AddOutputDesc("x", tensor_desc);
  auto data_node = graph->AddNode(data_op_desc);
  ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), node->GetInDataAnchor(0));

  bg::BufferPool buffer_pool;
  auto ret = bg::CreateComputeNodeInfo(node, buffer_pool);
  ASSERT_NE(ret, nullptr);
  auto compute_node_info = reinterpret_cast<ComputeNodeInfo *>(ret.get());
  ASSERT_EQ(compute_node_info->GetInputsNum(), 1);
  ASSERT_EQ(compute_node_info->GetOutputsNum(), 1);
  ASSERT_EQ(compute_node_info->GetIrInputsNum(), 1);
  auto td = compute_node_info->GetInputTdInfo(0);
  ASSERT_NE(td, nullptr);
  EXPECT_EQ(td->GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(td->GetOriginFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(td->GetStorageFormat(), ge::FORMAT_NC1HWC0);

  auto ins_info = compute_node_info->GetInputInstanceInfo(0);
  ASSERT_NE(ins_info, nullptr);
  EXPECT_EQ(ins_info->GetInstanceNum(), 1);
  EXPECT_EQ(ins_info->GetInstanceStart(), 0);

  auto attrs = compute_node_info->GetAttrs();
  EXPECT_EQ(attrs->GetAttrNum(), 14);

  EXPECT_STREQ(attrs->GetAttrPointer<char>(0), "World");
  EXPECT_STREQ(attrs->GetStr(0), "World");
  EXPECT_EQ(*attrs->GetAttrPointer<int64_t>(1), 1024000);
  EXPECT_EQ(*attrs->GetInt(1), 1024000);
  EXPECT_EQ(*attrs->GetAttrPointer<bool>(2), false);
  EXPECT_EQ(*attrs->GetBool(2), false);
  EXPECT_FLOAT_EQ(*attrs->GetAttrPointer<float>(3), 1024.1);
  EXPECT_FLOAT_EQ(*attrs->GetFloat(3), 1024.1);
  auto list_int = attrs->GetAttrPointer<gert::ContinuousVector>(4);
  ASSERT_NE(list_int, nullptr);
  ASSERT_EQ(list_int->GetSize(), 4);
  EXPECT_EQ(memcmp(list_int->GetData(), std::vector<int64_t>({10, 400, 3000, 8192}).data(), 4 * sizeof(int64_t)), 0);
  auto typed_list_int = attrs->GetListInt(4);
  ASSERT_NE(typed_list_int, nullptr);
  ASSERT_EQ(typed_list_int->GetSize(), 4);
  EXPECT_EQ(typed_list_int->GetData()[0], 10);
  EXPECT_EQ(typed_list_int->GetData()[1], 400);
  EXPECT_EQ(typed_list_int->GetData()[2], 3000);
  EXPECT_EQ(typed_list_int->GetData()[3], 8192);

  auto gert_tensor = attrs->GetAttrPointer<gert::Tensor>(5);
  EXPECT_EQ(attrs->GetTensor(5), gert_tensor);
  ASSERT_NE(gert_tensor, nullptr);
  EXPECT_EQ(gert_tensor->GetOriginShape(), gert::Shape({8, 224, 224, 3}));
  EXPECT_EQ(gert_tensor->GetStorageShape(), gert::Shape({8, 1, 224, 224, 16}));
  EXPECT_EQ(gert_tensor->GetOriginFormat(), ge::FORMAT_NHWC);
  EXPECT_EQ(gert_tensor->GetStorageFormat(), ge::FORMAT_NC1HWC0);
  EXPECT_EQ(gert_tensor->GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(memcmp(gert_tensor->GetData<uint16_t>(), fake_data.data(), fake_data.size() * sizeof(uint16_t)), 0);

  EXPECT_STREQ(attrs->GetAttrPointer<char>(6), "Hello");
  EXPECT_EQ(*attrs->GetAttrPointer<int64_t>(7), 10240);
  EXPECT_EQ(*attrs->GetAttrPointer<bool>(8), true);
  EXPECT_FLOAT_EQ(*attrs->GetAttrPointer<float>(9), 1024.0021);
  list_int = attrs->GetAttrPointer<gert::ContinuousVector>(10);
  ASSERT_NE(list_int, nullptr);
  ASSERT_EQ(list_int->GetSize(), 4);
  EXPECT_EQ(memcmp(list_int->GetData(), std::vector<int64_t>({10, 200, 3000, 4096}).data(), 4 * sizeof(int64_t)), 0);
  gert_tensor = attrs->GetAttrPointer<gert::Tensor>(11);
  ASSERT_NE(gert_tensor, nullptr);
  EXPECT_EQ(gert_tensor->GetOriginShape(), gert::Shape({8, 224, 224, 3}));
  EXPECT_EQ(gert_tensor->GetStorageShape(), gert::Shape({8, 1, 224, 224, 16}));
  EXPECT_EQ(gert_tensor->GetOriginFormat(), ge::FORMAT_NHWC);
  EXPECT_EQ(gert_tensor->GetStorageFormat(), ge::FORMAT_NC1HWC0);
  EXPECT_EQ(gert_tensor->GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(memcmp(gert_tensor->GetData<uint16_t>(), fake_data.data(), fake_data.size() * sizeof(uint16_t)), 0);

  auto list_str = attrs->GetAttrPointer<gert::ContinuousVector>(12);
  ASSERT_NE(list_str, nullptr);
  ASSERT_EQ(list_str->GetSize(), 4);
  EXPECT_STREQ(reinterpret_cast<const char *>(list_str->GetData()), "hello");
  EXPECT_STREQ(reinterpret_cast<const char *>(list_str->GetData()) + 6, "world");
  EXPECT_STREQ(reinterpret_cast<const char *>(list_str->GetData()) + 12, "world1");
  EXPECT_STREQ(reinterpret_cast<const char *>(list_str->GetData()) + 19, "hello1");

  auto list_datatype = attrs->GetAttrPointer<gert::ContinuousVector>(13);
  ASSERT_NE(list_datatype, nullptr);
  ASSERT_EQ(list_datatype->GetSize(), 4);
  EXPECT_EQ(memcmp(list_datatype->GetData(), std::vector<ge::DataType>({ge::DT_FLOAT, ge::DT_STRING, ge::DT_UINT16, ge::DT_BOOL}).data(), 4 * sizeof(ge::DataType)), 0);
  auto typed_list_datatype = attrs->GetAttrPointer<TypedContinuousVector<ge::DataType>>(13);
  ASSERT_NE(typed_list_datatype, nullptr);
  ASSERT_EQ(typed_list_datatype->GetSize(), 4);
  EXPECT_EQ((ge::DataType)(typed_list_datatype->GetData()[0]), ge::DT_FLOAT);
  EXPECT_EQ((ge::DataType)typed_list_datatype->GetData()[1], ge::DT_STRING);
  EXPECT_EQ((ge::DataType)typed_list_datatype->GetData()[2], ge::DT_UINT16);
  EXPECT_EQ((ge::DataType)typed_list_datatype->GetData()[3], ge::DT_BOOL);

}
TEST_F(BgKernelContextExtendUT, IgnoreNoneIrAttr) {
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  ge::GeTensorDesc tensor_desc;
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetFormat(ge::FORMAT_NC1HWC0);
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetShape(ge::GeShape({8, 1, 224, 224, 16}));
  tensor_desc.SetOriginShape(ge::GeShape({8, 3, 224, 224}));
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AppendIrInput("x", ge::kIrInputRequired);
  op_desc->AddOutputDesc("y", tensor_desc);

  ge::AttrUtils::SetStr(op_desc, "a1", "Hello");
  ge::AttrUtils::SetInt(op_desc, "a2", 10240);
  ge::AttrUtils::SetBool(op_desc, "a3", true);
  ge::AttrUtils::SetFloat(op_desc, "a4", 1024.0021);
  ge::AttrUtils::SetStr(op_desc, "b1", "World");
  ge::AttrUtils::SetInt(op_desc, "b2", 1024000);
  ge::AttrUtils::SetBool(op_desc, "b3", false);
  ge::AttrUtils::SetFloat(op_desc, "b4", 1024.1);

  op_desc->AppendIrAttrName("b1");
  op_desc->AppendIrAttrName("b3");
  op_desc->AppendIrAttrName("a2");
  op_desc->AppendIrAttrName("a4");

  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);
  auto data_op_desc = std::make_shared<ge::OpDesc>("data", "Data");
  data_op_desc->AddOutputDesc("x", tensor_desc);
  auto data_node = graph->AddNode(data_op_desc);
  ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), node->GetInDataAnchor(0));

  bg::BufferPool buffer_pool;
  auto ret = bg::CreateComputeNodeInfo(node, buffer_pool);
  ASSERT_NE(ret, nullptr);
  auto compute_node_info = reinterpret_cast<ComputeNodeInfo *>(ret.get());
  ASSERT_EQ(compute_node_info->GetInputsNum(), 1);
  ASSERT_EQ(compute_node_info->GetOutputsNum(), 1);
  ASSERT_EQ(compute_node_info->GetIrInputsNum(), 1);
  auto td = compute_node_info->GetInputTdInfo(0);
  ASSERT_NE(td, nullptr);
  EXPECT_EQ(td->GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(td->GetOriginFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(td->GetStorageFormat(), ge::FORMAT_NC1HWC0);

  auto ins_info = compute_node_info->GetInputInstanceInfo(0);
  ASSERT_NE(ins_info, nullptr);
  EXPECT_EQ(ins_info->GetInstanceNum(), 1);
  EXPECT_EQ(ins_info->GetInstanceStart(), 0);

  EXPECT_EQ(compute_node_info->GetAttrs()->GetAttrNum(), 4);
  EXPECT_STREQ(compute_node_info->GetAttrs()->GetAttrPointer<char>(0), "World");
  EXPECT_EQ(*compute_node_info->GetAttrs()->GetAttrPointer<bool>(1), false);
  EXPECT_EQ(*compute_node_info->GetAttrs()->GetAttrPointer<int64_t>(2), 10240);
  EXPECT_FLOAT_EQ(*compute_node_info->GetAttrs()->GetAttrPointer<float>(3), 1024.0021);
}

// 测试构造kernel context的时候从tensor desc上获取ATTR_NAME_RESHAPE_TYPE_MASK并设置到compile time tensor desc 上
// 同时测试调用Expand是否能够得到正确的扩维shape
TEST_F(BgKernelContextExtendUT, BuildRequiredInputWithExpandDimsType) {
  vector<int64_t> origin_shape = {5, 6, 7};
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  ge::GeTensorDesc tensor_desc;
  tensor_desc.SetFormat(ge::FORMAT_NC1HWC0);
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetShape(ge::GeShape({5, 6, 7, 1}));
  tensor_desc.SetOriginShape(ge::GeShape(origin_shape));
  // get reshape type 此处模拟FE调用transformer中方法获取int类型的reshape type
  int64_t int_reshape_type = transformer::ExpandDimension::GenerateReshapeType(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0,
                                                                               origin_shape.size(), "NCH");
  (void) ge::AttrUtils::SetInt(tensor_desc, ge::ATTR_NAME_RESHAPE_TYPE_MASK, int_reshape_type);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AppendIrInput("x", ge::kIrInputRequired);

  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);
  auto data_op_desc = std::make_shared<ge::OpDesc>("data", "Data");
  data_op_desc->AddOutputDesc("x", tensor_desc);
  auto data_node = graph->AddNode(data_op_desc);
  ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), node->GetInDataAnchor(0));

  bg::BufferPool buffer_pool;
  auto ret = bg::CreateComputeNodeInfo(node, buffer_pool);
  ASSERT_NE(ret, nullptr);
  auto compute_node_info = reinterpret_cast<ComputeNodeInfo *>(ret.get());
  ASSERT_EQ(compute_node_info->GetInputsNum(), 1);
  ASSERT_EQ(compute_node_info->GetOutputsNum(), 0);
  ASSERT_EQ(compute_node_info->GetIrInputsNum(), 1);
  auto td = compute_node_info->GetInputTdInfo(0);
  ASSERT_NE(td, nullptr);
  EXPECT_EQ(td->GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(td->GetOriginFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(td->GetStorageFormat(), ge::FORMAT_NC1HWC0);

  auto expand_dims_type = td->GetExpandDimsType();
  auto shape = Shape{5, 6, 7};
  Shape out_shape;
  expand_dims_type.Expand(shape, out_shape);
  ASSERT_EQ(4, out_shape.GetDimNum());
  ASSERT_EQ(out_shape, Shape({5, 6, 7, 1}));
}

TEST_F(BgKernelContextExtendUT, BuildWithMultiInstanceDynamicInputsWithNoMatchingNameBetweenIrAndNode) {
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  ge::GeTensorDesc tensor_desc;
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetFormat(ge::FORMAT_NC1HWC0);
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetShape(ge::GeShape({8, 1, 224, 224, 16}));
  tensor_desc.SetOriginShape(ge::GeShape({8, 3, 224, 224}));
  op_desc->AddInputDesc("y1", tensor_desc);
  op_desc->AddInputDesc("y2", tensor_desc);
  op_desc->AddInputDesc("x1", tensor_desc);
  op_desc->AddInputDesc("x2", tensor_desc);

  op_desc->AppendIrInput("x", ge::kIrInputDynamic);
  op_desc->AppendIrInput("y", ge::kIrInputDynamic);

  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);
  auto data_op_desc = std::make_shared<ge::OpDesc>("data", "Data");
  data_op_desc->AddOutputDesc("y0", tensor_desc);
  data_op_desc->AddOutputDesc("y1", tensor_desc);
  data_op_desc->AddOutputDesc("x0", tensor_desc);
  data_op_desc->AddOutputDesc("x1", tensor_desc);
  auto data_node = graph->AddNode(data_op_desc);
  ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(1), node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(2), node->GetInDataAnchor(2));
  ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(3), node->GetInDataAnchor(3));

  bg::BufferPool buffer_pool;
  auto ret = bg::CreateComputeNodeInfo(node, buffer_pool);
  ASSERT_NE(ret, nullptr);
  auto compute_node_info = reinterpret_cast<ComputeNodeInfo *>(ret.get());
  ASSERT_EQ(compute_node_info->GetInputsNum(), 4);
  ASSERT_EQ(compute_node_info->GetOutputsNum(), 0);
  ASSERT_EQ(compute_node_info->GetIrInputsNum(), 2);
  auto td = compute_node_info->GetInputTdInfo(0);
  ASSERT_NE(td, nullptr);
  EXPECT_EQ(td->GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(td->GetOriginFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(td->GetStorageFormat(), ge::FORMAT_NC1HWC0);

  auto ins_info = compute_node_info->GetInputInstanceInfo(0);
  ASSERT_NE(ins_info, nullptr);
  EXPECT_EQ(ins_info->GetInstanceNum(), 0);
  EXPECT_EQ(ins_info->GetInstanceStart(), 0);
  ins_info = compute_node_info->GetInputInstanceInfo(1);
  ASSERT_NE(ins_info, nullptr);
  EXPECT_EQ(ins_info->GetInstanceNum(), 0);
  EXPECT_EQ(ins_info->GetInstanceStart(), 0);

  EXPECT_EQ(compute_node_info->GetAttrs()->GetAttrNum(), 0);
  EXPECT_EQ(compute_node_info->GetAttrs()->GetAttrPointer<char>(0), nullptr);
}

TEST_F(BgKernelContextExtendUT, BuildWithRequiredInputWithNoMatchingNameBetweenIrAndNode) {
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  ge::GeTensorDesc tensor_desc;
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetFormat(ge::FORMAT_NC1HWC0);
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetShape(ge::GeShape({8, 1, 224, 224, 16}));
  tensor_desc.SetOriginShape(ge::GeShape({8, 3, 224, 224}));

  op_desc->AddInputDesc("x1", tensor_desc);
  op_desc->AppendIrInput("x", ge::kIrInputRequired);

  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);
  auto data_op_desc = std::make_shared<ge::OpDesc>("data", "Data");
  data_op_desc->AddOutputDesc("x1", tensor_desc);
  auto data_node = graph->AddNode(data_op_desc);
  ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), node->GetInDataAnchor(0));

  bg::BufferPool buffer_pool;
  auto ret = bg::CreateComputeNodeInfo(node, buffer_pool);
  ASSERT_NE(ret, nullptr);
  auto compute_node_info = reinterpret_cast<ComputeNodeInfo *>(ret.get());
  ASSERT_EQ(compute_node_info->GetInputsNum(), 1);
  ASSERT_EQ(compute_node_info->GetOutputsNum(), 0);
  ASSERT_EQ(compute_node_info->GetIrInputsNum(), 1);
  auto td = compute_node_info->GetInputTdInfo(0);
  ASSERT_NE(td, nullptr);
  EXPECT_EQ(td->GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(td->GetOriginFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(td->GetStorageFormat(), ge::FORMAT_NC1HWC0);

  auto ins_info = compute_node_info->GetInputInstanceInfo(0);
  ASSERT_NE(ins_info, nullptr);
  EXPECT_EQ(ins_info->GetInstanceNum(), 1);
  EXPECT_EQ(ins_info->GetInstanceStart(), 0);
}

/*
 *    net_output
 *       |
 *     conv
 *    /  \
 *  x     w
 */
ge::ComputeGraphPtr GetNoPeerOutputAnchorGraph() {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  // add x node
  ge::OpDescPtr x_op_desc = std::make_shared<ge::OpDesc>("x", "Data");
  ge::GeTensorDesc x_td(ge::GeShape({1, 256, 256, 3}), ge::FORMAT_NHWC, ge::DT_FLOAT);
  x_td.SetFormat(ge::FORMAT_NHWC);
  x_td.SetOriginFormat(ge::FORMAT_NHWC);
  x_op_desc->AddOutputDesc(x_td);
  auto x_node = graph->AddNode(x_op_desc);
  x_node->Init();
  // add w node
  ge::OpDescPtr w_op_desc = std::make_shared<ge::OpDesc>("w", "Data");
  ge::GeTensorDesc w_td(ge::GeShape({1, 1, 1, 1}), ge::FORMAT_HWCN, ge::DT_FLOAT);
  w_op_desc->AddOutputDesc(w_td);
  auto w_node = graph->AddNode(w_op_desc);
  w_node->Init();
  // add conv node
  ge::OpDescPtr conv_op_desc = std::make_shared<ge::OpDesc>("conv", "Conv2D");
  conv_op_desc->AddInputDesc(x_td);
  conv_op_desc->AddInputDesc(w_td);
  ge::GeTensorDesc bias_td(ge::GeShape({1}));
  conv_op_desc->AddInputDesc("bias", bias_td);
  conv_op_desc->AppendIrInput("x", ge::kIrInputRequired);
  conv_op_desc->AppendIrInput("w", ge::kIrInputRequired);
  conv_op_desc->AppendIrInput("bias", ge::kIrInputOptional);
  auto conv_node = graph->AddNode(conv_op_desc);
  conv_node->Init();
  // add edge
  ge::GraphUtils::AddEdge(x_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(w_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(1));
  return graph;
}

TEST_F(BgKernelContextExtendUT, BuildOptionalInputWithNoPeerOutputAnchor) {
  auto graph = GetNoPeerOutputAnchorGraph();
  auto conv_node = graph->FindNode("conv");
  bg::BufferPool buffer_pool;
  size_t total_size = 0;
  auto ret = bg::CreateComputeNodeInfo(conv_node, buffer_pool, total_size);
  ASSERT_NE(ret, nullptr);
  size_t buf_size = sizeof(ComputeNodeInfo) + sizeof(AnchorInstanceInfo) * 3 +
      sizeof(CompileTimeTensorDesc) * 2 + sizeof(RuntimeAttrsDef);
  ASSERT_EQ(total_size, buf_size);

  auto compute_node_info = reinterpret_cast<ComputeNodeInfo *>(ret.get());
  ASSERT_EQ(compute_node_info->GetInputsNum(), 2);
  ASSERT_EQ(compute_node_info->GetOutputsNum(), 0);
  ASSERT_EQ(compute_node_info->GetIrInputsNum(), 3);
  auto td = compute_node_info->GetInputTdInfo(0);
  ASSERT_NE(td, nullptr);
  EXPECT_EQ(td->GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(td->GetOriginFormat(), ge::FORMAT_NHWC);

  auto ins_info = compute_node_info->GetInputInstanceInfo(0);
  ASSERT_NE(ins_info, nullptr);
  EXPECT_EQ(ins_info->GetInstanceNum(), 1);
  EXPECT_EQ(ins_info->GetInstanceStart(), 0);
  ins_info = compute_node_info->GetInputInstanceInfo(1);
  ASSERT_NE(ins_info, nullptr);
  EXPECT_EQ(ins_info->GetInstanceNum(), 1);
  EXPECT_EQ(ins_info->GetInstanceStart(), 1);
  ins_info = compute_node_info->GetInputInstanceInfo(2);
}

TEST_F(BgKernelContextExtendUT, CheckCompileTimeTensorDescReservedMemClean) {
  auto graph = GetNoPeerOutputAnchorGraph();
  auto conv_node = graph->FindNode("conv");
  bg::BufferPool buffer_pool;
  size_t total_size = 0;
  auto ret = bg::CreateComputeNodeInfo(conv_node, buffer_pool, total_size);
  ASSERT_NE(ret, nullptr);
  size_t buf_size = sizeof(ComputeNodeInfo) + sizeof(AnchorInstanceInfo) * 3 +
      sizeof(CompileTimeTensorDesc) * 2 + sizeof(RuntimeAttrsDef);
  ASSERT_EQ(total_size, buf_size);

  auto compute_node_info = reinterpret_cast<ComputeNodeInfo *>(ret.get());
  ASSERT_EQ(compute_node_info->GetInputsNum(), 2);
  ASSERT_EQ(compute_node_info->GetOutputsNum(), 0);
  ASSERT_EQ(compute_node_info->GetIrInputsNum(), 3);
  auto td = compute_node_info->GetInputTdInfo(0);
  ASSERT_NE(td, nullptr);
  EXPECT_EQ(td->GetDataType(), ge::DT_FLOAT);
  EXPECT_EQ(td->GetOriginFormat(), ge::FORMAT_NHWC);
  EXPECT_FALSE(isMemoryCleared(reinterpret_cast<const uint8_t *>(td), sizeof(CompileTimeTensorDesc)));
  auto reserved_size = sizeof(CompileTimeTensorDesc) - sizeof(ge::DataType) - sizeof(StorageFormat);
  EXPECT_TRUE(isMemoryCleared(reinterpret_cast<const uint8_t *>(td) + sizeof(ge::DataType) + sizeof(StorageFormat),
                              reserved_size));
}

TEST_F(BgKernelContextExtendUT, GetPrivateAttrInComputeNodeInfoOK) {
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("test0", "Test");
  const char *attr_name_1 = "private_attr1";
  const char *attr_name_2 = "private_attr2";
  constexpr int64_t attr_value_1 = 10;
  const std::string attr_value_2 = "20";
  ge::AnyValue av1 = ge::AnyValue::CreateFrom<int64_t>(attr_value_1);
  ge::AnyValue av2 = ge::AnyValue::CreateFrom<std::string>(attr_value_2);
  op_desc->AppendIrAttrName("ir_attr_1");
  (void)op_desc->SetAttr("ir_attr_1", av2);
  (void)op_desc->SetAttr(attr_name_1, av1);
  (void)op_desc->SetAttr(attr_name_2, av2);
  std::vector<std::pair<ge::AscendString, ge::AnyValue>> private_attrs;
  private_attrs.emplace_back(std::make_pair(attr_name_1, av1));
  private_attrs.emplace_back(std::make_pair(attr_name_2, av2));
  bg::BufferPool buffer_pool;
  size_t attr_size;
  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);
  auto compute_node_info_holder = bg::CreateComputeNodeInfo(node, buffer_pool, private_attrs, attr_size);
  ASSERT_NE(compute_node_info_holder, nullptr);
  auto compute_node_info = reinterpret_cast<ComputeNodeInfo *>(compute_node_info_holder.get());
  EXPECT_EQ(compute_node_info->GetAttrs()->GetAttrNum(), 3);
  EXPECT_STREQ(compute_node_info->GetAttrs()->GetStr(0), "20");
  EXPECT_EQ(*(compute_node_info->GetAttrs()->GetInt(1)), 10);
  EXPECT_STREQ(compute_node_info->GetAttrs()->GetStr(2), "20");
}

TEST_F(BgKernelContextExtendUT, GetPrivateAttrInComputeNodeInfoByDefault) {
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("test0", "Test");
  const char *attr_name_1 = "private_attr1";
  const char *attr_name_2 = "private_attr2";
  constexpr int64_t attr_value_1 = 10;
  const std::string attr_value_2 = "20";
  ge::AnyValue av1 = ge::AnyValue::CreateFrom<int64_t>(attr_value_1);
  ge::AnyValue av2 = ge::AnyValue::CreateFrom<std::string>(attr_value_2);
  op_desc->AppendIrAttrName("ir_attr_1");
  (void)op_desc->SetAttr("ir_attr_1", av2);
  std::vector<std::pair<ge::AscendString, ge::AnyValue>> private_attrs;
  private_attrs.emplace_back(std::make_pair(attr_name_1, av1));
  private_attrs.emplace_back(std::make_pair(attr_name_2, av2));
  bg::BufferPool buffer_pool;
  size_t attr_size;
  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);
  auto compute_node_info_holder = bg::CreateComputeNodeInfo(node, buffer_pool, private_attrs, attr_size);
  ASSERT_NE(compute_node_info_holder, nullptr);
  auto compute_node_info = reinterpret_cast<ComputeNodeInfo *>(compute_node_info_holder.get());
  EXPECT_EQ(compute_node_info->GetAttrs()->GetAttrNum(), 3);
  EXPECT_STREQ(compute_node_info->GetAttrs()->GetStr(0), "20");
  EXPECT_EQ(*(compute_node_info->GetAttrs()->GetInt(1)), 10);
  EXPECT_STREQ(compute_node_info->GetAttrs()->GetStr(2), "20");
}

TEST_F(BgKernelContextExtendUT, CreateComputeNodeInfoFailedWhenNotRegisteringPrivateAttr) {
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("test0", "Test");
  const char *attr_name_1 = "private_attr1";
  const char *attr_name_2 = "private_attr2";
  constexpr int64_t attr_value_1 = 10;
  const std::string attr_value_2 = "20";
  ge::AnyValue av1 = ge::AnyValue::CreateFrom<int64_t>(attr_value_1);
  ge::AnyValue av2 = ge::AnyValue::CreateFrom<std::string>(attr_value_2);
  op_desc->AppendIrAttrName("ir_attr_1");
  (void)op_desc->SetAttr("ir_attr_1", av2);
  std::vector<std::pair<ge::AscendString, ge::AnyValue>> private_attrs;
  private_attrs.emplace_back(std::make_pair(attr_name_1, av1));
  private_attrs.emplace_back(std::make_pair(attr_name_2, ge::AnyValue()));
  bg::BufferPool buffer_pool;
  size_t attr_size;
  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);
  auto compute_node_info_holder = bg::CreateComputeNodeInfo(node, buffer_pool, private_attrs, attr_size);
  EXPECT_EQ(compute_node_info_holder, nullptr);
}

TEST_F(BgKernelContextExtendUT, CreateComputeNodeInfoWithoutIrAttr) {
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("test0", "Test");
  const char *attr_name_1 = "private_attr1";
  const char *attr_name_2 = "private_attr2";
  constexpr int64_t attr_value_1 = 10;
  const std::string attr_value_2 = "20";
  ge::AnyValue av1 = ge::AnyValue::CreateFrom<int64_t>(attr_value_1);
  ge::AnyValue av2 = ge::AnyValue::CreateFrom<std::string>(attr_value_2);
  op_desc->AppendIrAttrName("ir_attr_1");
  (void)op_desc->SetAttr("ir_attr_1", av2);
  std::vector<std::pair<ge::AscendString, ge::AnyValue>> private_attrs;
  private_attrs.emplace_back(std::make_pair(attr_name_1, av1));
  private_attrs.emplace_back(std::make_pair(attr_name_2, av2));
  bg::BufferPool buffer_pool;
  size_t attr_size;
  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);
  auto compute_node_info_holder = bg::CreateComputeNodeInfoWithoutIrAttr(node, buffer_pool, private_attrs, attr_size);
  EXPECT_NE(compute_node_info_holder, nullptr);
  auto compute_node_info = reinterpret_cast<ComputeNodeInfo *>(compute_node_info_holder.get());
  EXPECT_EQ(compute_node_info->GetAttrs()->GetAttrNum(), 2);
  EXPECT_EQ(*(compute_node_info->GetAttrs()->GetInt(0)), 10);
  EXPECT_STREQ(compute_node_info->GetAttrs()->GetStr(1), "20");
}

TEST_F(BgKernelContextExtendUT, BuildRequiredOutput) {
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  ge::GeTensorDesc tensor_desc;
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetFormat(ge::FORMAT_NC1HWC0);
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetShape(ge::GeShape({8, 1, 224, 224, 16}));
  tensor_desc.SetOriginShape(ge::GeShape({8, 3, 224, 224}));
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AppendIrInput("x", ge::kIrInputRequired);
  op_desc->AddOutputDesc("y", tensor_desc);
  op_desc->AppendIrOutput("y", ge::kIrOutputRequired);

  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);
  auto data_op_desc = std::make_shared<ge::OpDesc>("data", "Data");
  data_op_desc->AddOutputDesc("x", tensor_desc);
  auto data_node = graph->AddNode(data_op_desc);
  ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), node->GetInDataAnchor(0));

  bg::BufferPool buffer_pool;
  auto ret = bg::CreateComputeNodeInfo(node, buffer_pool);
  ASSERT_NE(ret, nullptr);
  auto compute_node_info = reinterpret_cast<ComputeNodeInfo *>(ret.get());
  ASSERT_EQ(compute_node_info->GetInputsNum(), 1);
  ASSERT_EQ(compute_node_info->GetOutputsNum(), 1);
  ASSERT_EQ(compute_node_info->GetIrInputsNum(), 1);
  ASSERT_EQ(compute_node_info->GetIrOutputsNum(), 1);
  auto td = compute_node_info->GetInputTdInfo(0);
  ASSERT_NE(td, nullptr);
  EXPECT_EQ(td->GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(td->GetOriginFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(td->GetStorageFormat(), ge::FORMAT_NC1HWC0);
  auto expand_dims_type = td->GetExpandDimsType();
  Shape origin_shape({8, 3, 224, 224});
  Shape storage_shape;
  expand_dims_type.Expand(origin_shape, storage_shape);
  EXPECT_EQ(storage_shape, origin_shape);

  auto ins_info = compute_node_info->GetInputInstanceInfo(0);
  ASSERT_NE(ins_info, nullptr);
  EXPECT_EQ(ins_info->GetInstanceNum(), 1);
  EXPECT_EQ(ins_info->GetInstanceStart(), 0);

  auto out_ins_info = compute_node_info->GetOutputInstanceInfo(0);
  ASSERT_NE(out_ins_info, nullptr);
  EXPECT_EQ(out_ins_info->GetInstanceNum(), 1);
  EXPECT_EQ(out_ins_info->GetInstanceStart(), 0);

  EXPECT_EQ(compute_node_info->GetAttrs()->GetAttrNum(), 0);
  EXPECT_EQ(compute_node_info->GetAttrs()->GetAttrPointer<char>(0), nullptr);
}
TEST_F(BgKernelContextExtendUT, BuildWithDynamicOutputs) {
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  ge::GeTensorDesc tensor_desc;
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetFormat(ge::FORMAT_NC1HWC0);
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetShape(ge::GeShape({8, 1, 224, 224, 16}));
  tensor_desc.SetOriginShape(ge::GeShape({8, 3, 224, 224}));
  op_desc->AddInputDesc("x0", tensor_desc);
  op_desc->AppendIrInput("x", ge::kIrInputDynamic);
  op_desc->AddOutputDesc("y0", tensor_desc);
  op_desc->AppendIrOutput("y", ge::kIrOutputDynamic);

  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);
  auto data_op_desc = std::make_shared<ge::OpDesc>("data", "Data");
  data_op_desc->AddOutputDesc("x", tensor_desc);
  auto data_node = graph->AddNode(data_op_desc);
  ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), node->GetInDataAnchor(0));

  bg::BufferPool buffer_pool;
  auto ret = bg::CreateComputeNodeInfo(node, buffer_pool);
  ASSERT_NE(ret, nullptr);
  auto compute_node_info = reinterpret_cast<ComputeNodeInfo *>(ret.get());
  ASSERT_EQ(compute_node_info->GetInputsNum(), 1);
  ASSERT_EQ(compute_node_info->GetOutputsNum(), 1);
  ASSERT_EQ(compute_node_info->GetIrInputsNum(), 1);
  ASSERT_EQ(compute_node_info->GetIrOutputsNum(), 1);
  auto td = compute_node_info->GetInputTdInfo(0);
  ASSERT_NE(td, nullptr);
  EXPECT_EQ(td->GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(td->GetOriginFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(td->GetStorageFormat(), ge::FORMAT_NC1HWC0);
  auto td_out = compute_node_info->GetOutputTdInfo(0);
  ASSERT_NE(td_out, nullptr);
  EXPECT_EQ(td_out->GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(td_out->GetOriginFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(td_out->GetStorageFormat(), ge::FORMAT_NC1HWC0);

  auto ins_info = compute_node_info->GetInputInstanceInfo(0);
  ASSERT_NE(ins_info, nullptr);
  EXPECT_EQ(ins_info->GetInstanceNum(), 1);
  EXPECT_EQ(ins_info->GetInstanceStart(), 0);

  auto out_ins_info = compute_node_info->GetOutputInstanceInfo(0);
  ASSERT_NE(out_ins_info, nullptr);
  EXPECT_EQ(out_ins_info->GetInstanceNum(), 1);
  EXPECT_EQ(out_ins_info->GetInstanceStart(), 0);

  EXPECT_EQ(compute_node_info->GetAttrs()->GetAttrNum(), 0);
  EXPECT_EQ(compute_node_info->GetAttrs()->GetAttrPointer<char>(0), nullptr);
}
TEST_F(BgKernelContextExtendUT, BuildWithMultiInstanceDynamicOutputs) {
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  ge::GeTensorDesc tensor_desc;
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetFormat(ge::FORMAT_NC1HWC0);
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetShape(ge::GeShape({8, 1, 224, 224, 16}));
  tensor_desc.SetOriginShape(ge::GeShape({8, 3, 224, 224}));
  op_desc->AddInputDesc("x0", tensor_desc);
  op_desc->AddInputDesc("x1", tensor_desc);
  op_desc->AddInputDesc("y0", tensor_desc);
  op_desc->AddInputDesc("y1", tensor_desc);
  op_desc->AddInputDesc("y2", tensor_desc);
  op_desc->AppendIrInput("x", ge::kIrInputDynamic);
  op_desc->AppendIrInput("y", ge::kIrInputDynamic);

  op_desc->AddOutputDesc("xx0", tensor_desc);
  op_desc->AddOutputDesc("xx1", tensor_desc);
  op_desc->AddOutputDesc("yy0", tensor_desc);
  op_desc->AddOutputDesc("yy1", tensor_desc);
  op_desc->AddOutputDesc("yy2", tensor_desc);
  op_desc->AppendIrOutput("xx", ge::kIrOutputDynamic);
  op_desc->AppendIrOutput("yy", ge::kIrOutputDynamic);

  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);
  auto data_op_desc = std::make_shared<ge::OpDesc>("data", "Data");
  data_op_desc->AddOutputDesc("x0", tensor_desc);
  data_op_desc->AddOutputDesc("x1", tensor_desc);
  data_op_desc->AddOutputDesc("y0", tensor_desc);
  data_op_desc->AddOutputDesc("y1", tensor_desc);
  data_op_desc->AddOutputDesc("y2", tensor_desc);
  auto data_node = graph->AddNode(data_op_desc);
  ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(1), node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(2), node->GetInDataAnchor(2));
  ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(3), node->GetInDataAnchor(3));
  ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(4), node->GetInDataAnchor(4));

  bg::BufferPool buffer_pool;
  auto ret = bg::CreateComputeNodeInfo(node, buffer_pool);
  ASSERT_NE(ret, nullptr);
  auto compute_node_info = reinterpret_cast<ComputeNodeInfo *>(ret.get());
  ASSERT_EQ(compute_node_info->GetInputsNum(), 5);
  ASSERT_EQ(compute_node_info->GetOutputsNum(), 5);
  ASSERT_EQ(compute_node_info->GetIrInputsNum(), 2);
  ASSERT_EQ(compute_node_info->GetIrOutputsNum(), 2);
  auto td = compute_node_info->GetInputTdInfo(0);
  ASSERT_NE(td, nullptr);
  EXPECT_EQ(td->GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(td->GetOriginFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(td->GetStorageFormat(), ge::FORMAT_NC1HWC0);

  auto ins_info = compute_node_info->GetInputInstanceInfo(0);
  ASSERT_NE(ins_info, nullptr);
  EXPECT_EQ(ins_info->GetInstanceNum(), 2);
  EXPECT_EQ(ins_info->GetInstanceStart(), 0);
  ins_info = compute_node_info->GetInputInstanceInfo(1);
  ASSERT_NE(ins_info, nullptr);
  EXPECT_EQ(ins_info->GetInstanceNum(), 3);
  EXPECT_EQ(ins_info->GetInstanceStart(), 2);

  auto out_ins_info = compute_node_info->GetOutputInstanceInfo(0);
  ASSERT_NE(out_ins_info, nullptr);
  EXPECT_EQ(out_ins_info->GetInstanceNum(), 2);
  EXPECT_EQ(out_ins_info->GetInstanceStart(), 0);
  out_ins_info = compute_node_info->GetOutputInstanceInfo(1);
  ASSERT_NE(out_ins_info, nullptr);
  EXPECT_EQ(out_ins_info->GetInstanceNum(), 3);
  EXPECT_EQ(out_ins_info->GetInstanceStart(), 2);

  EXPECT_EQ(compute_node_info->GetAttrs()->GetAttrNum(), 0);
  EXPECT_EQ(compute_node_info->GetAttrs()->GetAttrPointer<char>(0), nullptr);
}
TEST_F(BgKernelContextExtendUT, BuildWithEmptyDynamicOutputs) {
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  ge::GeTensorDesc tensor_desc;
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetFormat(ge::FORMAT_NC1HWC0);
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetOriginDataType(ge::DT_FLOAT);
  tensor_desc.SetShape(ge::GeShape({8, 1, 224, 224, 16}));
  tensor_desc.SetOriginShape(ge::GeShape({8, 3, 224, 224}));
  op_desc->AddInputDesc("x0", tensor_desc);
  op_desc->AddInputDesc("x1", tensor_desc);
  op_desc->AppendIrInput("y", ge::kIrInputDynamic);
  op_desc->AppendIrInput("x", ge::kIrInputDynamic);
  op_desc->AddOutputDesc("xx0", tensor_desc);
  op_desc->AddOutputDesc("xx1", tensor_desc);
  op_desc->AppendIrOutput("yy", ge::kIrOutputDynamic);
  op_desc->AppendIrOutput("xx", ge::kIrOutputDynamic);

  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);
  auto data_op_desc = std::make_shared<ge::OpDesc>("data", "Data");
  data_op_desc->AddOutputDesc("x0", tensor_desc);
  data_op_desc->AddOutputDesc("x1", tensor_desc);
  auto data_node = graph->AddNode(data_op_desc);
  ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(1), node->GetInDataAnchor(1));

  bg::BufferPool buffer_pool;
  auto ret = bg::CreateComputeNodeInfo(node, buffer_pool);
  ASSERT_NE(ret, nullptr);
  auto compute_node_info = reinterpret_cast<ComputeNodeInfo *>(ret.get());
  ASSERT_EQ(compute_node_info->GetInputsNum(), 2);
  ASSERT_EQ(compute_node_info->GetOutputsNum(), 2);
  ASSERT_EQ(compute_node_info->GetIrInputsNum(), 2);
  ASSERT_EQ(compute_node_info->GetIrOutputsNum(), 2);
  auto td = compute_node_info->GetInputTdInfo(0);
  ASSERT_NE(td, nullptr);
  EXPECT_EQ(td->GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(td->GetOriginFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(td->GetStorageFormat(), ge::FORMAT_NC1HWC0);

  auto td_out = compute_node_info->GetOutputTdInfo(0);
  ASSERT_NE(td_out, nullptr);
  EXPECT_EQ(td_out->GetDataType(), ge::DT_FLOAT16);
  EXPECT_EQ(td_out->GetOriginFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(td_out->GetStorageFormat(), ge::FORMAT_NC1HWC0);

  auto ins_info = compute_node_info->GetInputInstanceInfo(0);
  ASSERT_NE(ins_info, nullptr);
  EXPECT_EQ(ins_info->GetInstanceNum(), 0);
  ins_info = compute_node_info->GetInputInstanceInfo(1);
  ASSERT_NE(ins_info, nullptr);
  EXPECT_EQ(ins_info->GetInstanceNum(), 2);
  EXPECT_EQ(ins_info->GetInstanceStart(), 0);

  auto out_ins_info = compute_node_info->GetOutputInstanceInfo(0);
  ASSERT_NE(out_ins_info, nullptr);
  EXPECT_EQ(out_ins_info->GetInstanceNum(), 0);
  out_ins_info = compute_node_info->GetOutputInstanceInfo(1);
  ASSERT_NE(ins_info, nullptr);
  EXPECT_EQ(ins_info->GetInstanceNum(), 2);
  EXPECT_EQ(ins_info->GetInstanceStart(), 0);

  EXPECT_EQ(compute_node_info->GetAttrs()->GetAttrNum(), 0);
  EXPECT_EQ(compute_node_info->GetAttrs()->GetAttrPointer<char>(0), nullptr);
}
// todo lowering时，不需要构造attr
// todo infershape、tiling utils重新看一下输入是否正确
// todo kernel中获取attr的方式变化
}  // namespace gert

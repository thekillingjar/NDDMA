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

#include "macro_utils/dt_public_scope.h"
#include "host_kernels/array_ops/unsqueezev3_kernel.h"

#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/ge_inner_error_codes.h"
#include "common/op/ge_op_utils.h"
#include "common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/types.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "host_kernels/kernel_factory.h"
#include "macro_utils/dt_public_unscope.h"

using namespace testing;
using namespace ge;

class UtestFoldingKernelUnsqueezeV3Kernel : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

namespace {
void TestValidUnsqueezeV3(vector<int64_t> &data_vec, vector<int8_t> &dim_value_vec) {

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");

  ge::OpDescPtr data_op_desc = std::make_shared<ge::OpDesc>("data", CONSTANTOP);
  int64_t dims_size = 1;
  for_each(data_vec.begin(), data_vec.end(), [&](int64_t &data) { dims_size *= data; });
  vector<int8_t> data_value_vec(dims_size, 4);
  GeTensorDesc data_tensor_desc(GeShape(data_vec), FORMAT_NCHW, ge::DT_INT8);
  auto data_tensor = std::make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value_vec.data(),
                                                       data_value_vec.size() * sizeof(int8_t));
  OpDescUtils::SetWeights(data_op_desc, data_tensor);
  data_op_desc->AddOutputDesc(data_tensor_desc);
  NodePtr data_node = graph->AddNode(data_op_desc);
  data_node->Init();

  // add axes node
  ge::OpDescPtr dim_op_desc = std::make_shared<ge::OpDesc>("axes", CONSTANTOP);
  GeTensorDesc dim_tensor_desc(ge::GeShape(), FORMAT_NCHW, ge::DT_INT8);
  auto dim_tensor = std::make_shared<GeTensor>(dim_tensor_desc, (uint8_t *)dim_value_vec.data(),
                                                      dim_value_vec.size() * sizeof(int8_t));
  OpDescUtils::SetWeights(dim_op_desc, dim_tensor);
  dim_op_desc->AddOutputDesc(dim_tensor_desc);
  NodePtr dim_node = graph->AddNode(dim_op_desc);
  dim_node->Init();

  // add SqueezeV3 node
  OpDescPtr unsqueezev3_op_desc = std::make_shared<OpDesc>("UnsqueezeV3", UNSQUEEZEV3);
  unsqueezev3_op_desc->AddInputDesc(data_tensor_desc);
  unsqueezev3_op_desc->AddInputDesc(dim_tensor_desc);
  GeTensorDesc tensor_desc_out(GeShape(), FORMAT_NCHW, DT_INT32);
  unsqueezev3_op_desc->AddOutputDesc(tensor_desc_out);
  auto node_ptr = graph->AddNode(unsqueezev3_op_desc);
  node_ptr->Init();

  // add edge
  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), node_ptr->GetInDataAnchor(0));
  GraphUtils::AddEdge(dim_node->GetOutDataAnchor(0), node_ptr->GetInDataAnchor(1));

  Status status;
  auto kernel = KernelFactory::Instance().Create(UNSQUEEZEV3);
  status = kernel->Compute(node_ptr);
  EXPECT_EQ(ge::SUCCESS, status);

  std::vector<GeTensorPtr> v_output;
  std::vector<ConstGeTensorPtr> input = {data_tensor, dim_tensor};
  status = kernel->Compute(node_ptr->GetOpDesc(), input, v_output);
  EXPECT_EQ(ge::SUCCESS, status);
  EXPECT_EQ(1, v_output.size());
  int8_t *data_buf = (int8_t *)v_output[0]->GetData().data();
  for (size_t i = 0; i < 6; i++) {
    EXPECT_EQ(*(data_buf + i), 4);
  }
}

void TestInvalidUnsqueezeV3(vector<int64_t> &data_vec, vector<int8_t> &dim_value_vec) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");

  ge::OpDescPtr data_op_desc = std::make_shared<ge::OpDesc>("data", CONSTANTOP);
  int64_t dims_size = 1;
  for_each(data_vec.begin(), data_vec.end(), [&](int64_t &data) { dims_size *= data; });
  vector<int8_t> data_value_vec(dims_size, 2);
  GeTensorDesc data_tensor_desc(GeShape(data_vec), FORMAT_NCHW, ge::DT_INT8);
  auto data_tensor = std::make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value_vec.data(),
                                                       data_value_vec.size() * sizeof(int8_t));
  OpDescUtils::SetWeights(data_op_desc, data_tensor);
  data_op_desc->AddOutputDesc(data_tensor_desc);
  NodePtr data_node = graph->AddNode(data_op_desc);
  data_node->Init();

  // add axes node
  ge::OpDescPtr dim_op_desc = std::make_shared<ge::OpDesc>("axes", CONSTANTOP);
  GeTensorDesc dim_tensor_desc(ge::GeShape(), FORMAT_NCHW, ge::DT_INT8);
  auto dim_tensor = std::make_shared<GeTensor>(dim_tensor_desc, (uint8_t *)dim_value_vec.data(),
                                                      dim_value_vec.size() * sizeof(int8_t));
  OpDescUtils::SetWeights(dim_op_desc, dim_tensor);
  dim_op_desc->AddOutputDesc(dim_tensor_desc);
  NodePtr dim_node = graph->AddNode(dim_op_desc);
  dim_node->Init();

  // add SqueezeV3 node
  OpDescPtr squeezev3_op_desc = std::make_shared<OpDesc>("UnsqueezeV3", UNSQUEEZEV3);
  squeezev3_op_desc->AddInputDesc(data_tensor_desc);
  squeezev3_op_desc->AddInputDesc(dim_tensor_desc);

  auto node_ptr = graph->AddNode(squeezev3_op_desc);
  node_ptr->Init();

  Status status;
  auto kernel = KernelFactory::Instance().Create(UNSQUEEZEV3);
  status = kernel->Compute(node_ptr);
  EXPECT_EQ(ge::NOT_CHANGED, status);


  std::vector<GeTensorPtr> v_output;
  std::vector<ConstGeTensorPtr> input = {data_tensor, dim_tensor};
  status = kernel->Compute(node_ptr->GetOpDesc(), input, v_output);
  EXPECT_EQ(ge::NOT_CHANGED, status);

  // add edge
  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), node_ptr->GetInDataAnchor(0));
  GraphUtils::AddEdge(dim_node->GetOutDataAnchor(0), node_ptr->GetInDataAnchor(1));

  GeTensorDesc tensor_desc_out(GeShape(), FORMAT_NCHW, DT_INT32);
  squeezev3_op_desc->AddOutputDesc(tensor_desc_out);
  std::vector<ConstGeTensorPtr> input_2 = {nullptr, nullptr};
  status = kernel->Compute(node_ptr->GetOpDesc(), input_2, v_output);
  EXPECT_EQ(ge::PARAM_INVALID, status);
}
}

TEST_F(UtestFoldingKernelUnsqueezeV3Kernel, ValidFoldingTest) {
  vector<int64_t> data_vec = {1, 2, 3};
  vector<int8_t> dim_value_vec = {0};
  TestValidUnsqueezeV3(data_vec, dim_value_vec);
}

TEST_F(UtestFoldingKernelUnsqueezeV3Kernel, InvalidFoldingTest) {
  vector<int64_t> data_vec = {1, 2, 3};
  vector<int8_t> dim_value_vec = {0};
  TestInvalidUnsqueezeV3(data_vec, dim_value_vec);
}

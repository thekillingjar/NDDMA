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
#include "host_kernels/array_ops/unsqueeze_kernel.h"

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

class NodeBuilder {
 public:
  NodeBuilder(const std::string &name, const std::string &type) {
    op_desc_ = std::make_shared<OpDesc>(name, type);
  }
  NodeBuilder &AddInputDesc(std::initializer_list<int64_t> shape, ge::Format format = FORMAT_NCHW,
                            ge::DataType data_type = DT_FLOAT) {
    op_desc_->AddInputDesc(CreateTensorDesc(shape, format, data_type)->Clone());
    return *this;
  }

  NodeBuilder &AddOutputDesc(std::initializer_list<int64_t> shape, ge::Format format = FORMAT_NCHW,
                             ge::DataType data_type = DT_FLOAT) {
    op_desc_->AddOutputDesc(CreateTensorDesc(shape, format, data_type)->Clone());
    return *this;
  }

  ge::NodePtr Build(const ge::ComputeGraphPtr &graph) { return graph->AddNode(op_desc_); }

 private:
  ge::GeTensorDescPtr CreateTensorDesc(std::initializer_list<int64_t> shape, ge::Format format = FORMAT_NCHW,
                                       ge::DataType data_type = DT_FLOAT) {
    GeShape ge_shape{std::vector<int64_t>(shape)};
    ge::GeTensorDescPtr tensor_desc = std::make_shared<ge::GeTensorDesc>();
    tensor_desc->SetShape(ge_shape);
    tensor_desc->SetFormat(format);
    tensor_desc->SetDataType(data_type);
    return tensor_desc;
  }

  ge::OpDescPtr op_desc_;
};

class UtestGraphPassesFoldingKernelUnsqueezeKernel : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestGraphPassesFoldingKernelUnsqueezeKernel, KernelCompute1) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  // data
  ge::NodePtr node_data = NodeBuilder("data", DATA).AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT).Build(graph);
  // const
  ge::NodePtr node_const =
      NodeBuilder("const", CONSTANT).AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT).Build(graph);
  // relu
  ge::NodePtr node_relu = NodeBuilder("node_relu1", RELU)
                              .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                              .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                              .Build(graph);
  // sinh
  ge::NodePtr node_sinh = NodeBuilder("node_sinh", SINH)
                              .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                              .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                              .Build(graph);

  NodePtr node_nul = nullptr;
  UnsqueezeKernel kernel;

  Status status = kernel.Compute(node_nul);
  EXPECT_NE(SUCCESS, status);

  status = kernel.Compute(node_data);
  EXPECT_NE(SUCCESS, status);

  status = kernel.Compute(node_const);
  EXPECT_NE(SUCCESS, status);

  status = kernel.Compute(node_relu);
  EXPECT_EQ(SUCCESS, status);

  status = kernel.Compute(node_sinh);
  EXPECT_EQ(SUCCESS, status);
}

TEST_F(UtestGraphPassesFoldingKernelUnsqueezeKernel, KernelCompute2) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("UNSQUEEZEV2", UNSQUEEZEV2);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_INT32);

  std::string name1 = "name1";
  GeTensorDesc output_desc1;
  op_desc_ptr->AddOutputDesc(name1, output_desc1);

  std::string name2 = "name2";
  GeTensorDesc output_desc2;
  op_desc_ptr->AddOutputDesc(name2, output_desc2);

  int32_t start = 1, limit = 20, delta = 2;

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {start};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_1;
  vector<int32_t> data_vec_1 = {limit};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_2;
  vector<int32_t> data_vec_2 = {delta};
  GeTensorDesc tensor_desc_2(GeShape(dims_vec_2), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_2, (uint8_t *)data_vec_2.data(), data_vec_2.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2};
  vector<GeTensorPtr> outputs;

  UnsqueezeKernel kernel;
  Status status = kernel.Compute(op_desc_ptr, input, outputs);
  EXPECT_EQ(NOT_CHANGED, status);
}

TEST_F(UtestGraphPassesFoldingKernelUnsqueezeKernel, KernelCompute3) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("UNSQUEEZE", UNSQUEEZE);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_INT32);

  std::string name1 = "name1";
  GeTensorDesc output_desc1;
  op_desc_ptr->AddOutputDesc(name1, output_desc1);

  GeTensorDesc input_desc1;
  op_desc_ptr->AddInputDesc(0, input_desc1);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0,};
  vector<GeTensorPtr> outputs;

  UnsqueezeKernel kernel;
  Status status = kernel.Compute(op_desc_ptr, input, outputs);
  EXPECT_EQ(SUCCESS, status);
}

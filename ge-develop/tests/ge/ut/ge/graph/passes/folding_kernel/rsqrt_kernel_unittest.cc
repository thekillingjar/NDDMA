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
#include "host_kernels/elewise_calculation_ops/rsqrt_kernel.h"

#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
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

class UtestFoldingKernelRsqrtKernel : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

// optimize op of sqrt success
TEST_F(UtestFoldingKernelRsqrtKernel, RsqrtOptimizerSuccess) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("RSQRT", RSQRT);

  vector<int64_t> dims_vec_0 = {3, 2};
  vector<float> data_vec_0 = {4.0, 16.0, 25.0, 100.0, 400.0};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(RSQRT);
  Status status = kernel->Compute(op_desc_ptr, input, outputs);

  float *outdata = (float *)outputs[0]->GetData().data();

  EXPECT_EQ(SUCCESS, status);
  EXPECT_FLOAT_EQ(outdata[0], 0.5);
  EXPECT_FLOAT_EQ(outdata[1], 0.25);
  EXPECT_FLOAT_EQ(outdata[2], 0.2);
  EXPECT_FLOAT_EQ(outdata[3], 0.1);
  EXPECT_FLOAT_EQ(outdata[4], 0.05);
}

// optimize op of sqrt fail(include 0)
TEST_F(UtestFoldingKernelRsqrtKernel, RsqrtOptimizerHasZero) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("RSQRT", RSQRT);

  vector<int64_t> dims_vec_0 = {2};
  vector<float> data_vec_0 = {4.0, 0.0};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(RSQRT);
  ge::Status status = kernel->Compute(op_desc_ptr, input, outputs);

  EXPECT_EQ(NOT_CHANGED, status);
}

TEST_F(UtestFoldingKernelRsqrtKernel, InputCheckFailed) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("RSQRT", RSQRT);

  vector<int64_t> dims_vec_0 = {3, 2};
  vector<float> data_vec_0 = {4.0, 16.0, 25.0, 100.0, 400.0};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(RSQRT);
  Status status = kernel->Compute(op_desc_ptr, input, outputs);
  EXPECT_EQ(NOT_CHANGED, status);
}

TEST_F(UtestFoldingKernelRsqrtKernel, DoubleSuccess) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("RSQRT", RSQRT);

  vector<int64_t> dims_vec_0 = {3, 2};
  vector<double> data_vec_0 = {4.0, 16.0, 25.0, 100.0, 400.0};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_DOUBLE);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(double));

  vector<ConstGeTensorPtr> input = {tensor_0};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(RSQRT);
  Status status = kernel->Compute(op_desc_ptr, input, outputs);
  EXPECT_EQ(SUCCESS, status);
}

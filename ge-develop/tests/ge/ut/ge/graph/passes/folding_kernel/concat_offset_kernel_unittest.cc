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
#include "host_kernels/split_combination_ops/concat_offset_kernel.h"

#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/ge_inner_error_codes.h"
#include "common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/passes/standard_optimize/constant_folding/dimension_compute_pass.h"
#include "host_kernels/kernel_utils.h"
#include "graph/types.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "host_kernels/kernel_factory.h"
#include "macro_utils/dt_public_unscope.h"

using namespace testing;
using namespace ge;

class UtestGraphPassesFoldingKernelConcatOffsetKernel : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestGraphPassesFoldingKernelConcatOffsetKernel, CheckAttrFail) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("ConcatOffset", "ConcatOffset");

  vector<ConstGeTensorPtr> input = {};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(CONCATOFFSET);
  ge::Status status = kernel->Compute(op_desc_ptr, input, outputs);
  EXPECT_EQ(NOT_CHANGED, status);
  status = kernel->Compute(nullptr, input, outputs);
  EXPECT_EQ(PARAM_INVALID, status);
}

TEST_F(UtestGraphPassesFoldingKernelConcatOffsetKernel, CheckInputSize) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("ConcatOffset", "ConcatOffset");
  AttrUtils::SetInt(op_desc_ptr, "N", 2);
  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_INT32);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);

  vector<int64_t> dims_vec_0 = {1, 1, 1, 1};
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(CONCATOFFSET);
  ge::Status status = kernel->Compute(op_desc_ptr, input, outputs);
  EXPECT_EQ(NOT_CHANGED, status);
}

TEST_F(UtestGraphPassesFoldingKernelConcatOffsetKernel, ComputeSuccess) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("ConcatOffset", "ConcatOffset");
  (void)AttrUtils::SetInt(op_desc_ptr, "N", 3);
  GeTensorDesc dims_tensor_desc(GeShape({0, 0, 0, 0}), FORMAT_NCHW, DT_INT32);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(1, dims_tensor_desc);
  op_desc_ptr->AddInputDesc(2, dims_tensor_desc);

  vector<int64_t> dims_vec_0 = {0};
  vector<int32_t> data_vec_0 = {0, 0, 0, 0};
  vector<int32_t> data_vec_1 = {1, 1, 1, 1};
  vector<int32_t> data_vec_2 = {1, 0, 0, 0};
  GeTensorDesc tensor_desc_0(GeShape({0}), FORMAT_ND, DT_INT32);
  GeTensorDesc tensor_desc_1(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_dim =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)dims_vec_0.data(), dims_vec_0.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_2.data(), data_vec_2.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_dim, tensor_0, tensor_1, tensor_2};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(CONCATOFFSET);
  ge::Status status = kernel->Compute(op_desc_ptr, input, outputs);

  EXPECT_EQ(ge::SUCCESS, status);
}

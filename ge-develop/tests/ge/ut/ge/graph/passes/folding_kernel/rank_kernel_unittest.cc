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
#include "host_kernels/array_ops/rank_kernel.h"

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
using ge::SUCCESS;

class UtestGraphPassesFoldingKernelRankKernel : public testing::Test {
 protected:
  void SetUp() {
    graph = std::make_shared<ge::ComputeGraph>("default");
    op_desc_ptr = std::make_shared<OpDesc>("Rank", RANK);
    node = std::make_shared<Node>(op_desc_ptr, graph);
    kernel = KernelFactory::Instance().Create(RANK);
  }

  void TearDown() {}

  ge::ComputeGraphPtr graph;
  OpDescPtr op_desc_ptr;
  NodePtr node;
  shared_ptr<Kernel> kernel;
};

TEST_F(UtestGraphPassesFoldingKernelRankKernel, RankIsOne) {
  vector<int64_t> dims_vec_0 = {4};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc out_tensor_desc_0(GeShape(), FORMAT_NCHW, DT_INT32);
  op_desc_ptr->AddInputDesc(tensor_desc_0);
  op_desc_ptr->AddOutputDesc(out_tensor_desc_0);
  std::vector<GeTensorPtr> v_output;

  Status status = kernel->Compute(nullptr, v_output);
  EXPECT_EQ(FAILED, status);
  status = kernel->Compute(node, v_output);
  EXPECT_EQ(SUCCESS, status);
  EXPECT_EQ(v_output[0]->GetTensorDesc().GetDataType(), DT_INT32);
  EXPECT_EQ(v_output[0]->GetTensorDesc().GetShape().GetDimNum(), 0);
  EXPECT_EQ(*(int32_t *)(v_output[0]->GetData().data()), 1);
}

TEST_F(UtestGraphPassesFoldingKernelRankKernel, RankIsThree) {
  vector<int64_t> dims_vec_0 = {4, 2, 2};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc out_tensor_desc_0(GeShape(), FORMAT_NCHW, DT_INT32);
  op_desc_ptr->AddInputDesc(tensor_desc_0);
  op_desc_ptr->AddOutputDesc(out_tensor_desc_0);
  std::vector<GeTensorPtr> v_output;
  Status status = kernel->Compute(node, v_output);

  EXPECT_EQ(SUCCESS, status);
  EXPECT_EQ(v_output[0]->GetTensorDesc().GetDataType(), DT_INT32);
  EXPECT_EQ(v_output[0]->GetTensorDesc().GetShape().GetDimNum(), 0);
  EXPECT_EQ(*(int32_t *)(v_output[0]->GetData().data()), 3);
}

TEST_F(UtestGraphPassesFoldingKernelRankKernel, InvalidCaseInputSizeIsZero) {
  std::vector<GeTensorPtr> v_output;
  Status status = kernel->Compute(node, v_output);

  EXPECT_NE(SUCCESS, status);
}

TEST_F(UtestGraphPassesFoldingKernelRankKernel, InvalidCaseInputSizeIsMoreThanOne) {
  vector<int64_t> dims_vec_0 = {4, 2, 2};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddInputDesc(tensor_desc_0);
  op_desc_ptr->AddInputDesc(tensor_desc_0);

  std::vector<GeTensorPtr> v_output;
  Status status = kernel->Compute(node, v_output);

  EXPECT_NE(SUCCESS, status);
}

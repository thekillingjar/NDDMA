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
#include "host_kernels/kernel_utils.h"

#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/ge_inner_error_codes.h"
#include "common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/passes/standard_optimize/constant_folding/dimension_compute_pass.h"
#include "graph/types.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "host_kernels/kernel_factory.h"
#include "macro_utils/dt_public_unscope.h"

using namespace testing;
using namespace ge;

class UtestGraphPassesFoldingKernelkernelUtils : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestGraphPassesFoldingKernelkernelUtils, KernelUtilsFailed) {
  int64_t d0 = static_cast<int64_t>(INT_MAX);
  int64_t d1 = d0 + 1;
  vector<int64_t> data = {d0, d1};

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {0};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(PACK);
  Status status = KernelUtils::ConstructTensorDescWithData(tensor_desc_0, data, outputs, true);
  EXPECT_EQ(PARAM_INVALID, status);

  NodePtr node_ptr = nullptr;
  status = KernelUtils::CheckDimensionNodeInfo(node_ptr);
  EXPECT_EQ(FAILED, status);
  bool ret = KernelUtils::CheckFormatSupported(node_ptr);
  EXPECT_EQ(false, ret);

  ConstGeTensorPtr const_weight_ptr = nullptr;
  OpDescPtr op_desc_ptr = nullptr;
  ret = KernelUtils::CheckSizeForTransOp(const_weight_ptr, op_desc_ptr);
  EXPECT_EQ(false, ret);
}

TEST_F(UtestGraphPassesFoldingKernelkernelUtils, IsUnknownShape) {
  std::vector<int64_t> shape = {-1, 256, 256};
  GeShape dynamic_shape(shape);
  EXPECT_EQ(KernelUtils::IsUnknownShape(dynamic_shape), true);
}

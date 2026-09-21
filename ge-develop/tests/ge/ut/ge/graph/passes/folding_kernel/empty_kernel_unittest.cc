/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <assert.h>
#include <gtest/gtest.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "macro_utils/dt_public_scope.h"
#include "host_kernels/array_ops/empty_kernel.h"

#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/types.h"
#include "folding_kernel_unittest_utils.h"
#include "framework/common/ge_inner_error_codes.h"
#include "ge/ge_api.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/operator.h"
#include "graph/passes/standard_optimize/constant_folding/constant_folding_pass.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "host_kernels/kernel_factory.h"
#include "macro_utils/dt_public_unscope.h"

using namespace testing;
using namespace ge;
using namespace ge::test;

class UtestEmptyKernel : public testing::Test {
 protected:
  void SetUp() { init(); }

  void TearDown() { destory(); }

 private:
  void init() {
    pass_ = new ConstantFoldingPass();
    graph_ = std::make_shared<ge::ComputeGraph>("default");
    op_desc_ptr_ = std::make_shared<OpDesc>("Empty", EMPTY);
    node_ = std::make_shared<Node>(op_desc_ptr_, graph_);
  }
  void destory() {
    delete pass_;
    pass_ = NULL;
  }

 protected:
  template <typename InT, typename OutT>
  bool TestOtherDataType(DataType in_type, DataType out_type) {
    string op_type = EMPTY;
    vector<vector<int64_t>> input_shape_dims({
        {5},
    });
    vector<vector<InT>> input_data({
        {2, 2, 1, 3, 1},
    });

    vector<vector<int64_t>> output_shape_dims({
        {2, 2, 1, 3, 1},
    });
    vector<vector<OutT>> output_data({
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    });

    return ge::test::ConstFoldingKernelCheckShapeAndOutput(op_type, input_shape_dims, input_data, in_type,
                                                           output_shape_dims, output_data, out_type);
  }
  ConstantFoldingPass *pass_;

  ge::ComputeGraphPtr graph_;
  OpDescPtr op_desc_ptr_;
  NodePtr node_;
};

TEST_F(UtestEmptyKernel, ShapeDimCheckFail) {
  vector<int64_t> dims_vec_0 = {8, 2};
  vector<int64_t> data_vec_0 = {2, 1, 4, 1, 2};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int64_t));
  op_desc_ptr_->AddInputDesc(tensor_desc_0);

  GeTensorDesc tensor_desc_out(GeShape(), FORMAT_NCHW, DT_INT64);
  op_desc_ptr_->AddOutputDesc(tensor_desc_out);

  vector<ConstGeTensorPtr> input = {tensor_0};

  std::vector<GeTensorPtr> outputs;
  auto kernel_ptr = KernelFactory::Instance().Create(EMPTY);
  if (kernel_ptr != nullptr) {
    Status status = kernel_ptr->Compute(op_desc_ptr_, input, outputs);
    EXPECT_EQ(NOT_CHANGED, status);
  }
}

TEST_F(UtestEmptyKernel, ShapeDataTypeCheclFail) {
  vector<int64_t> dims_vec_0 = {5};
  vector<int64_t> data_vec_0 = {2, 1, 4, 1, 2};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int64_t));
  op_desc_ptr_->AddInputDesc(tensor_desc_0);

  GeTensorDesc tensor_desc_out(GeShape(), FORMAT_NCHW, DT_INT64);
  op_desc_ptr_->AddOutputDesc(tensor_desc_out);

  vector<ConstGeTensorPtr> input = {tensor_0};

  std::vector<GeTensorPtr> outputs;
  auto kernel_ptr = KernelFactory::Instance().Create(EMPTY);
  if (kernel_ptr != nullptr) {
    Status status = kernel_ptr->Compute(op_desc_ptr_, input, outputs);
    EXPECT_EQ(NOT_CHANGED, status);
  }
}

TEST_F(UtestEmptyKernel, SizeCheckFail) {
  vector<int64_t> dims_vec_0 = {-1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr_->AddInputDesc(tensor_desc_0);

  vector<int64_t> dims_vec_1 = {-1};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr_->AddInputDesc(tensor_desc_1);

  GeTensorDesc tensor_desc_out_1(GeShape(), FORMAT_NCHW, DT_INT64);
  op_desc_ptr_->AddOutputDesc(tensor_desc_out_1);

  vector<int64_t> data_vec_0 = {2, 1, 4, 1, 2};
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int64_t));

  vector<int64_t> data_vec_1 = {2, 2, 1, 3, 1};
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int64_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};

  std::vector<GeTensorPtr> outputs;

  auto kernel_ptr = KernelFactory::Instance().Create(EMPTY);
  if (kernel_ptr != nullptr) {
    Status status = kernel_ptr->Compute(op_desc_ptr_, input, outputs);
    EXPECT_EQ(NOT_CHANGED, status);
  }

  Status status = kernel_ptr->Compute(nullptr, input, outputs);
  EXPECT_NE(SUCCESS, status);
}

TEST_F(UtestEmptyKernel, CheckOutputNormalIn64Out64) {
  string op_type = EMPTY;
  vector<vector<int64_t>> input_shape_dims({
      {5},
  });
  vector<vector<int64_t>> input_data({
      {2, 2, 1, 3, 1},
  });

  vector<vector<int64_t>> output_shape_dims({
      {2, 2, 1, 3, 1},
  });
  vector<vector<int64_t>> output_data({
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
  });

  bool result = ge::test::ConstFoldingKernelCheckShapeAndOutput(op_type, input_shape_dims, input_data, DT_INT64,
                                                                output_shape_dims, output_data, DT_INT64);
  EXPECT_EQ(result, true);
}
TEST_F(UtestEmptyKernel, CheckOutputNormalIn32Out64) {
  string op_type = EMPTY;
  vector<vector<int64_t>> input_shape_dims({
      {5},
  });
  vector<vector<int32_t>> input_data({
      {2, 2, 1, 3, 1},
  });

  vector<vector<int64_t>> output_shape_dims({
      {2, 2, 1, 3, 1},
  });
  vector<vector<int64_t>> output_data({
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
  });

  bool result = ge::test::ConstFoldingKernelCheckShapeAndOutput(op_type, input_shape_dims, input_data, DT_INT32,
                                                                output_shape_dims, output_data, DT_INT64);
  EXPECT_EQ(result, true);
}

TEST_F(UtestEmptyKernel, CheckOutputNormalIn64Out32) {
  string op_type = EMPTY;
  vector<vector<int64_t>> input_shape_dims({
      {5},
  });
  vector<vector<int64_t>> input_data({
      {2, 2, 1, 3, 1},
  });

  vector<vector<int64_t>> output_shape_dims({
      {2, 2, 1, 3, 1},
  });
  vector<vector<int32_t>> output_data({
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
  });

  bool result = ge::test::ConstFoldingKernelCheckShapeAndOutput(op_type, input_shape_dims, input_data, DT_INT64,
                                                                output_shape_dims, output_data, DT_INT32);
  EXPECT_EQ(result, true);
}

TEST_F(UtestEmptyKernel, CheckOutputNormalOtherType) {
  bool result = false;

#define TESTBYTYPE(data_type, T)                               \
  result = TestOtherDataType<int32_t, T>(DT_INT32, data_type); \
  EXPECT_EQ(result, true);

  TESTBYTYPE(DT_FLOAT, float)
  TESTBYTYPE(DT_INT8, int8_t)
  TESTBYTYPE(DT_INT16, int16_t)
  TESTBYTYPE(DT_UINT16, uint16_t)
  TESTBYTYPE(DT_UINT8, uint8_t)
  TESTBYTYPE(DT_INT32, int32_t)
  TESTBYTYPE(DT_INT64, int64_t)
  TESTBYTYPE(DT_UINT32, uint32_t)
  TESTBYTYPE(DT_UINT64, uint64_t)
  TESTBYTYPE(DT_BOOL, bool)
  TESTBYTYPE(DT_DOUBLE, double)
#undef TESTBYTYPE
}

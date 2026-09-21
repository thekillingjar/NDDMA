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
#include "host_kernels/array_ops/gathershapes_kernel.h"

#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/op/ge_op_utils.h"
#include "common/types.h" 
#include "graph/types.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "host_kernels/kernel_factory.h"
#include "macro_utils/dt_public_unscope.h"

using namespace testing;
using namespace ge;

class UtestGraphPassesFoldingKernelGatherShapesKernel : public testing::Test {
 protected:
  void SetUp() {
    std::cout << "GatherShapes kernel test begin." << std::endl;
  }
  void TearDown() {
    std::cout << "GatherShapes kernel test end." << std::endl;
  }
};

TEST_F(UtestGraphPassesFoldingKernelGatherShapesKernel, ValidInputTest) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("GatherShapes", "GatherShapes");
  GeTensorDesc tensor_desc_x1(GeShape({1, 2, 3}), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_x2(GeShape({3, 4, 5}), FORMAT_NHWC, DT_INT32);
  op_desc_ptr->AddInputDesc(0, tensor_desc_x1);
  op_desc_ptr->AddInputDesc(1, tensor_desc_x2);
  op_desc_ptr->SetAttr("axes",
                       AnyValue::CreateFrom(std::vector<std ::vector<int64_t>>{{0, 1}, {0, 2}, {1, 0}, {1, 2}}));
  GeTensorDesc tensor_desc_out(GeShape(), FORMAT_NCHW, DT_INT32);
  op_desc_ptr->AddOutputDesc(tensor_desc_out);
  shared_ptr<Kernel> gathershapes_kernel = KernelFactory::Instance().Create(GATHERSHAPES);
  ge::ComputeGraphPtr graph_ = std::make_shared<ge::ComputeGraph>("default");
  NodePtr node_ = graph_->AddNode(op_desc_ptr);
  std::vector<GeTensorPtr> v_output;
  ge::Status status = gathershapes_kernel->Compute(node_, v_output);
  ASSERT_EQ(ge::SUCCESS, status);
  ASSERT_EQ(v_output.size(), 1);
  vector<int64_t> expect_output({2, 3, 3, 5});
  GeTensorPtr tensor_out = v_output[0];
  int32_t *data_buf = (int32_t *)tensor_out->GetData().data();
  for (size_t i = 0; i < expect_output.size(); i++) {
    EXPECT_EQ(*(data_buf + i), expect_output[i]);
  }
}

TEST_F(UtestGraphPassesFoldingKernelGatherShapesKernel, InputWithUnknownRankTest) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("GatherShapes", "GatherShapes");
  GeTensorDesc tensor_desc_x1(GeShape({1, 2, 3}), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_x2(GeShape({-2}), FORMAT_NHWC, DT_INT32);
  op_desc_ptr->AddInputDesc(0, tensor_desc_x1);
  op_desc_ptr->AddInputDesc(1, tensor_desc_x2);
  GeTensorDesc tensor_desc_out(GeShape(), FORMAT_NCHW, DT_INT32);
  op_desc_ptr->AddOutputDesc(tensor_desc_out);
  op_desc_ptr->SetAttr("axes",
                       AnyValue::CreateFrom(std::vector<std ::vector<int64_t>>{{0, 1}, {0, 2}, {1, 0}, {1, 2}}));
  shared_ptr<Kernel> gathershapes_kernel = KernelFactory::Instance().Create("GatherShapes");
  ge::ComputeGraphPtr graph_ = std::make_shared<ge::ComputeGraph>("default");
  NodePtr node_ = graph_->AddNode(op_desc_ptr);
  std::vector<GeTensorPtr> v_output;
  ge::Status status = gathershapes_kernel->Compute(node_, v_output);
  EXPECT_EQ(ge::NOT_CHANGED, status);
}

TEST_F(UtestGraphPassesFoldingKernelGatherShapesKernel, InputWithUnknownShapeTest1) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("GatherShapes", "GatherShapes");
  GeTensorDesc tensor_desc_x1(GeShape({1, 2, 3}), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_x2(GeShape({2, 3, -1}), FORMAT_NHWC, DT_INT32);
  op_desc_ptr->AddInputDesc(0, tensor_desc_x1);
  op_desc_ptr->AddInputDesc(1, tensor_desc_x2);
  op_desc_ptr->SetAttr("axes",
                       AnyValue::CreateFrom(std::vector<std ::vector<int64_t>>{{0, 1}, {0, 2}, {1, 0}, {1, 2}}));
  GeTensorDesc tensor_desc_out(GeShape(), FORMAT_NCHW, DT_INT32);
  op_desc_ptr->AddOutputDesc(tensor_desc_out);
  shared_ptr<Kernel> gathershapes_kernel = KernelFactory::Instance().Create("GatherShapes");
  ge::ComputeGraphPtr graph_ = std::make_shared<ge::ComputeGraph>("default");
  NodePtr node_ = graph_->AddNode(op_desc_ptr);
  std::vector<GeTensorPtr> v_output;
  ge::Status status = gathershapes_kernel->Compute(node_, v_output);
  EXPECT_EQ(ge::NOT_CHANGED, status);
}

TEST_F(UtestGraphPassesFoldingKernelGatherShapesKernel, InputWithUnknownShapeTest2) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("GatherShapes", "GatherShapes");
  GeTensorDesc tensor_desc_x1(GeShape({1, 2, 3}), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_x2(GeShape({2, 3, -1}), FORMAT_NHWC, DT_INT32);
  op_desc_ptr->AddInputDesc(0, tensor_desc_x1);
  op_desc_ptr->AddInputDesc(1, tensor_desc_x2);
  GeTensorDesc tensor_desc_out(GeShape(), FORMAT_NCHW, DT_INT32);
  op_desc_ptr->AddOutputDesc(tensor_desc_out);
  shared_ptr<Kernel> gathershapes_kernel = KernelFactory::Instance().Create("GatherShapes");
  ge::ComputeGraphPtr graph_ = std::make_shared<ge::ComputeGraph>("default");
  NodePtr node_ = graph_->AddNode(op_desc_ptr);
  std::vector<GeTensorPtr> v_output;
  ge::Status status = gathershapes_kernel->Compute(node_, v_output);
  EXPECT_EQ(ge::FAILED, status);
}

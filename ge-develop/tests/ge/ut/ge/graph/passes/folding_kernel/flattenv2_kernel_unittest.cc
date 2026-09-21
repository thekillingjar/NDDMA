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
#include "host_kernels/transformation_ops/flattenv2_kernel.h"

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

class UtestGraphPassesFoldingKernelFlattenV2Kernel : public testing::Test {
 protected:
  void SetUp() {
    std::cout << "FlattenV2 kernel test begin." << std::endl;
  }
  void TearDown() {
    std::cout << "FlattenV2 kernel test end." << std::endl;
  }
};

TEST_F(UtestGraphPassesFoldingKernelFlattenV2Kernel, ValidInputTest) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("FlattenV2", FLATTENV2);
  GeTensorDesc tensor_desc_x(GeShape({1, 2, 3}), FORMAT_NCHW, DT_INT32);
  op_desc->AddInputDesc("x",tensor_desc_x);
  op_desc->SetAttr("axis", AnyValue::CreateFrom<int64_t>(1));
  op_desc->SetAttr("end_axis", AnyValue::CreateFrom<int64_t>(-1));
  GeTensorDesc tensor_desc_out(GeShape(), FORMAT_NCHW, DT_INT32);
  op_desc->AddOutputDesc(tensor_desc_out);
  NodePtr _node = graph->AddNode(op_desc);
  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(FLATTENV2);
  std::vector<uint8_t> data = {2,2,2,2,2,2};
  GeTensorPtr data_tensor = std::make_shared<GeTensor>(tensor_desc_x, data);
  std::vector<ConstGeTensorPtr> input = {data_tensor};
  std::vector<GeTensorPtr> outputs;
  Status verify_status = kernel->Compute(_node);
  ASSERT_EQ(ge::SUCCESS, verify_status);

  Status status = kernel->Compute(_node->GetOpDesc(), input, outputs);
  ASSERT_EQ(ge::SUCCESS, status);

  std::vector<int64_t> expect_shape = {1,6};
  std::vector<int64_t> actual_shape = outputs[0]->GetTensorDesc().GetShape().GetDims();
  ASSERT_EQ(expect_shape, actual_shape);
  std::vector<uint8_t> &expect_data = data;
  const uint8_t *actual_data = outputs[0]->GetData().GetData();
  for (size_t i = 0; i < expect_data.size(); i++) {
    ASSERT_EQ(*(actual_data+i), expect_data[i]);
  }
}

TEST_F(UtestGraphPassesFoldingKernelFlattenV2Kernel, InvalidAttrTest) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("FlattenV2", FLATTENV2);
  GeTensorDesc tensor_desc_x(GeShape({1, 2, 3}), FORMAT_NCHW, DT_INT32);
  op_desc->AddInputDesc("x", tensor_desc_x);
  op_desc->SetAttr("axis", AnyValue::CreateFrom<int64_t>(1));
  // end_axis is invalid
  op_desc->SetAttr("end_axis", AnyValue::CreateFrom<int64_t>(-5));
  GeTensorDesc tensor_desc_out(GeShape(), FORMAT_NCHW, DT_INT32);
  op_desc->AddOutputDesc(tensor_desc_out);
  NodePtr _node = graph->AddNode(op_desc);
  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(FLATTENV2);
  std::vector<uint8_t> data = {2,2,2,2,2,2};
  GeTensorPtr data_tensor = std::make_shared<GeTensor>(tensor_desc_x, data);
  std::vector<ConstGeTensorPtr> input = {data_tensor};
  std::vector<GeTensorPtr> outputs;

  Status verify_status = kernel->Compute(_node);
  EXPECT_EQ(NOT_CHANGED, verify_status);

  Status status = kernel->Compute(_node->GetOpDesc(), input, outputs);
  EXPECT_EQ(NOT_CHANGED, status);
}

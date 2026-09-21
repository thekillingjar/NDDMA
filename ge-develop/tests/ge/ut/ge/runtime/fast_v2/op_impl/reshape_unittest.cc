/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include <gtest/gtest.h>
#include "faker/kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tensor.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/tensor_data.h"
namespace gert {
ge::graphStatus InferShapeForReshape(InferShapeContext *context);
}
namespace gert_test {
class ReshapeUT : public testing::Test {};

TEST_F(ReshapeUT, InferShapeOk) {
  ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("Reshape"), nullptr);
  auto infer_shape_func = gert::InferShapeForReshape;

  gert::StorageShape input_shape = {{8, 3, 224, 224}, {8, 3, 224, 224}};
  size_t total_size;
  auto input_tensor_holder = gert::Tensor::CreateFollowing(2, ge::DT_INT64, total_size);
  auto input_tensor = reinterpret_cast<gert::Tensor *>(input_tensor_holder.get());
  input_tensor->SetOriginFormat(ge::FORMAT_ND);
  input_tensor->SetStorageFormat(ge::FORMAT_ND);
  input_tensor->MutableOriginShape() = {2};
  input_tensor->MutableStorageShape() = {2};

  auto tensor_data = reinterpret_cast<int64_t *>(input_tensor->GetAddr());
  input_tensor->SetData(gert::TensorData{tensor_data, nullptr});
  tensor_data[0] = 24;
  tensor_data[1] = -1;

  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(2, 1)
                    .IrInputNum(2)
                    .InputShapes({&input_shape, input_tensor})
                    .OutputShapes({&output_shape})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);

  EXPECT_EQ(output_shape.GetOriginShape().GetDimNum(), 2);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(0), 24);
  EXPECT_EQ(output_shape.GetOriginShape().GetDim(1), 50176);
}

TEST_F(ReshapeUT, DataDependentOk) {
  ASSERT_NE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("Reshape"), nullptr);
  EXPECT_FALSE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("Reshape")->IsInputDataDependency(0));
  EXPECT_TRUE(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("Reshape")->IsInputDataDependency(1));
}
}  // namespace gert_test
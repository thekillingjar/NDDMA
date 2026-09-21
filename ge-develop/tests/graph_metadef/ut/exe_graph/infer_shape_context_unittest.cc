/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/runtime/infer_shape_context.h"
#include "graph/ge_error_codes.h"
#include <gtest/gtest.h>
#include "faker/kernel_run_context_faker.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/kernel_registry.h"
namespace gert {
class InferShapeContextUT : public testing::Test {};
TEST_F(InferShapeContextUT, GetInputShapeOk) {
  gert::StorageShape in_shape1 = {{8, 3, 224, 224}, {8, 1, 224, 224, 16}};
  gert::StorageShape in_shape2 = {{2, 2, 3, 8}, {8, 1, 2, 2, 16}};
  gert::StorageShape out_shape = {{8, 3, 224, 224}, {8, 1, 224, 224, 16}};
  auto context_holder = InferShapeContextFaker()
                            .IrInputNum(2)
                            .NodeIoNum(2, 1)
                            .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                            .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z)
                            .InputShapes({&in_shape1, &in_shape2})
                            .OutputShapes({&out_shape})
                            .Build();
  auto context = context_holder.GetContext<InferShapeContext>();
  ASSERT_NE(context, nullptr);

  ASSERT_NE(context->GetInputShape(0), nullptr);
  EXPECT_EQ(*context->GetInputShape(0), in_shape1.GetOriginShape());

  ASSERT_NE(context->GetInputShape(1), nullptr);
  EXPECT_EQ(*context->GetInputShape(1), in_shape2.GetOriginShape());

  EXPECT_EQ(context->GetInputShape(2), nullptr);
}

TEST_F(InferShapeContextUT, GetDynamicInputShapeOk) {
  gert::StorageShape in_shape1 = {{8, 3, 224, 224}, {8, 1, 224, 224, 16}};
  gert::StorageShape in_shape2 = {{2, 2, 3, 8}, {2, 2, 3, 8}};
  gert::StorageShape in_shape3 = {{3, 2, 3, 8}, {3, 2, 3, 8}};
  gert::StorageShape in_shape4 = {{4, 2, 3, 8}, {4, 2, 3, 8}};
  gert::StorageShape out_shape = {{8, 3, 224, 224}, {8, 1, 224, 224, 16}};
  auto context_holder = InferShapeContextFaker()
                            .IrInstanceNum({1, 2, 0, 1})
                            .NodeIoNum(4, 1)
                            .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                            .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                            .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                            .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                            .InputShapes({&in_shape1, &in_shape2, &in_shape3, &in_shape4})
                            .OutputShapes({&out_shape})
                            .Build();
  auto context = context_holder.GetContext<InferShapeContext>();
  ASSERT_NE(context, nullptr);

  ASSERT_NE(context->GetOptionalInputShape(0), nullptr);
  EXPECT_EQ(*context->GetOptionalInputShape(0), in_shape1.GetOriginShape());

  ASSERT_NE(context->GetDynamicInputShape(1, 0), nullptr);
  EXPECT_EQ(*context->GetDynamicInputShape(1, 0), in_shape2.GetOriginShape());

  ASSERT_NE(context->GetDynamicInputShape(1, 1), nullptr);
  EXPECT_EQ(*context->GetDynamicInputShape(1, 1), in_shape3.GetOriginShape());

  EXPECT_EQ(context->GetOptionalInputShape(2), nullptr);

  ASSERT_NE(context->GetOptionalInputShape(3), nullptr);
  EXPECT_EQ(*context->GetOptionalInputShape(3), in_shape4.GetOriginShape());
}

TEST_F(InferShapeContextUT, GetOutShapeOk) {
  gert::StorageShape in_shape1 = {{8, 3, 224, 224}, {8, 1, 224, 224, 16}};
  gert::StorageShape in_shape2 = {{2, 2, 3, 8}, {8, 1, 2, 2, 16}};
  gert::StorageShape out_shape = {{8, 3, 224, 224}, {8, 1, 224, 224, 16}};
  auto context_holder = InferShapeContextFaker()
                            .IrInputNum(2)
                            .NodeIoNum(2, 1)
                            .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                            .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z)
                            .InputShapes({&in_shape1, &in_shape2})
                            .OutputShapes({&out_shape})
                            .Build();
  auto context = context_holder.GetContext<InferShapeContext>();
  ASSERT_NE(context, nullptr);

  ASSERT_NE(context->GetOutputShape(0), nullptr);
  EXPECT_EQ(*context->GetOutputShape(0), out_shape.GetOriginShape());

  EXPECT_EQ(context->GetOutputShape(1), nullptr);
}

TEST_F(InferShapeContextUT, GetOptionalInputTensorFailed_NotSetOptionalInput) {
  gert::StorageShape in_shape1 = {{8, 3, 224, 224}, {8, 1, 224, 224, 16}};
  auto infer_shape_func_addr = reinterpret_cast<void *>(0x11);

  auto context_holder = KernelRunContextFaker()
                            .IrInputNum(2)
                            .IrInstanceNum({1, 0})
                            .KernelIONum(2, 0)
                            .NodeIoNum(1, 0)
                            .Inputs({&in_shape1, infer_shape_func_addr})
                            .Build();
  auto context = context_holder.GetContext<InferShapeContext>();
  ASSERT_NE(context, nullptr);

  ASSERT_NE(context->GetInputShape(0), nullptr);
  EXPECT_EQ(*context->GetInputShape(0), in_shape1.GetOriginShape());

  EXPECT_EQ(context->GetOptionalInputTensor(1), nullptr);
  EXPECT_NE(context->GetInputTensor(1), nullptr);
}

TEST_F(InferShapeContextUT, GetOptionalInputTensorOK_SetOptionalInput) {
gert::StorageShape in_shape1 = {{8, 3, 224, 224}, {8, 1, 224, 224, 16}};
gert::Tensor in_tensor_2 = {{{1, 16, 256}, {1, 16, 256}},                // shape
                            {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                            kOnDeviceHbm,                                // placement
                            ge::DT_FLOAT16,                              // data type
                            (void *) 0xabc};
auto infer_shape_func_addr = reinterpret_cast<void *>(0x11);

auto context_holder = KernelRunContextFaker()
    .IrInputNum(2)
    .IrInstanceNum({1, 1})
    .KernelIONum(3, 0)
    .NodeIoNum(2, 0)
    .Inputs({&in_shape1, &in_tensor_2, infer_shape_func_addr})
    .Build();
auto context = context_holder.GetContext<InferShapeContext>();
ASSERT_NE(context, nullptr);

ASSERT_NE(context->GetInputShape(0), nullptr);
EXPECT_EQ(*context->GetInputShape(0), in_shape1.GetOriginShape());

EXPECT_NE(context->GetOptionalInputTensor(1), nullptr);
EXPECT_NE(context->GetInputTensor(2), nullptr);
}


TEST_F(InferShapeContextUT, GetRequiredInputTensorOk) {
  gert::Tensor in_tensor_1 = {{{8, 3, 224, 224}, {8, 1, 224, 224, 16}},    // shape
                              {ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, {}},  // format
                              kOnDeviceHbm,                                // placement
                              ge::DT_FLOAT16,                              // data type
                              (void *)0x0};
  gert::Tensor in_tensor_2 = {{{2, 2, 3, 8}, {2, 2, 3, 8}},    // shape
                              {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                              kOnDeviceHbm,                                // placement
                              ge::DT_FLOAT16,                              // data type
                              (void *)0x0};
  gert::Tensor in_tensor_3 = {{{3, 2, 3, 8}, {3, 2, 3, 8}},    // shape
                              {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                              kOnDeviceHbm,                                // placement
                              ge::DT_FLOAT16,                              // data type
                              (void *)0x0};
  gert::Tensor in_tensor_4 = {{{4, 2, 3, 8}, {4, 2, 3, 8}},    // shape
                              {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                              kOnDeviceHbm,                                // placement
                              ge::DT_FLOAT16,                              // data type
                              (void *)0x12345};
  gert::Tensor out_tensor = {{{8, 3, 224, 224}, {8, 1, 224, 224, 16}},    // shape
                              {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                              kOnDeviceHbm,                                // placement
                              ge::DT_FLOAT16,                              // data type
                              (void *)0x0};

  auto context_holder = InferShapeContextFaker()
                            .IrInstanceNum({1, 2, 0, 1})
                            .NodeIoNum(4, 1)
                            .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                            .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                            .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                            .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                            .InputShapes({&in_tensor_1, &in_tensor_2, &in_tensor_3, &in_tensor_4})
                            .OutputShapes({&out_tensor})
                            .Build();
  auto context = context_holder.GetContext<InferShapeContext>();
  ASSERT_NE(context, nullptr);

  ASSERT_NE(context->GetRequiredInputShape(0), nullptr);
  EXPECT_EQ(*context->GetRequiredInputShape(0), in_tensor_1.GetOriginShape());
  ASSERT_NE(context->GetRequiredInputTensor(0), nullptr);
  EXPECT_EQ(context->GetRequiredInputTensor(0)->GetOriginShape(), in_tensor_1.GetOriginShape());
  EXPECT_EQ(context->GetRequiredInputTensor(0)->GetAddr(), nullptr);

  ASSERT_NE(context->GetDynamicInputShape(1, 0), nullptr);
  EXPECT_EQ(*context->GetDynamicInputShape(1, 0), in_tensor_2.GetOriginShape());
  ASSERT_NE(context->GetDynamicInputTensor(1, 0), nullptr);
  EXPECT_EQ(context->GetDynamicInputTensor(1, 0)->GetOriginShape(), in_tensor_2.GetOriginShape());
  EXPECT_EQ(context->GetDynamicInputTensor(1, 0)->GetAddr(), nullptr);

  ASSERT_NE(context->GetDynamicInputShape(1, 1), nullptr);
  EXPECT_EQ(*context->GetDynamicInputShape(1, 1), in_tensor_3.GetOriginShape());
  ASSERT_NE(context->GetDynamicInputTensor(1, 1), nullptr);
  EXPECT_EQ(context->GetDynamicInputTensor(1, 1)->GetOriginShape(), in_tensor_3.GetOriginShape());
  EXPECT_EQ(context->GetDynamicInputTensor(1, 1)->GetAddr(), nullptr);

  EXPECT_EQ(context->GetOptionalInputShape(2), nullptr);

  ASSERT_NE(context->GetRequiredInputShape(3), nullptr);
  EXPECT_EQ(*context->GetRequiredInputShape(3), in_tensor_4.GetOriginShape());
  ASSERT_NE(context->GetRequiredInputTensor(3), nullptr);
  EXPECT_EQ(context->GetRequiredInputTensor(3)->GetOriginShape(), in_tensor_4.GetOriginShape());
  EXPECT_EQ(context->GetRequiredInputTensor(3)->GetAddr(), in_tensor_4.GetAddr());
}
}  // namespace gert

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/runtime/infer_shape_range_context.h"
#include <gtest/gtest.h>
#include "faker/kernel_run_context_faker.h"
#include "exe_graph/runtime/storage_shape.h"
namespace gert {
class InferShapeRangeContextUT : public testing::Test {};
TEST_F(InferShapeRangeContextUT, GetInputShapeRangeOk) {
  Shape same_ele{8, 3, 224, 224};
  gert::Range<Shape> in_shape_range1(&same_ele);
  Shape min1{2, 2, 3, 8};
  Shape max1{2, -1, 3, 8};
  gert::Range<Shape> in_shape_range2(&min1, &max1);
  Shape out_shape1{8, 3, 224, 224};
  Shape out_shape2{8, 224, 224, 224};
  gert::Range<Shape> out_shape_range(&out_shape1, &out_shape2);
  auto context_holder = InferShapeRangeContextFaker()
      .IrInputNum(2)
      .NodeIoNum(2, 1)
      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z)
      .InputShapeRanges({&in_shape_range1, &in_shape_range2})
      .OutputShapeRanges({&out_shape_range})
      .Build();
  auto context = context_holder.GetContext<InferShapeRangeContext>();
  ASSERT_NE(context, nullptr);

  ASSERT_NE(context->GetInputShapeRange(0), nullptr);
  EXPECT_EQ(*context->GetInputShapeRange(0), in_shape_range1);

  ASSERT_NE(context->GetInputShapeRange(1), nullptr);
  EXPECT_EQ(*context->GetInputShapeRange(1), in_shape_range2);

  EXPECT_EQ(context->GetInputShapeRange(2), nullptr);
}

TEST_F(InferShapeRangeContextUT, GetDynamicInputShapeRangeOk) {
  Shape min1{8, 3, 224, 224};
  Shape max1{-1, 3, 224, 224};
  gert::Range<Shape> in_shape_range1(&min1, &max1);
  Shape min2{2, 2, 3, 8};
  Shape max2{2, -1, 3, 8};
  gert::Range<Shape> in_shape_range2(&min2, &max2);
  Shape min3{3, 2, 3, 8};
  Shape max3{3, 2, 9, 8};
  gert::Range<Shape> in_shape_range3(&min3, &max3);
  Shape min4{4, 2, 3, 8};
  Shape max4{4, 2, 3, 16};
  gert::Range<Shape> in_shape_range4(&min4, &max4);
  Shape min5{8, 3, 224, 224};
  Shape max5{-1, 3, 224, 224};
  gert::Range<Shape> out_shape_range(&min5, &max5);
  auto context_holder = InferShapeRangeContextFaker()
      .IrInstanceNum({1, 2, 0, 1})
      .NodeIoNum(4, 1)
      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
      .InputShapeRanges({&in_shape_range1, &in_shape_range2, &in_shape_range3, &in_shape_range4})
      .OutputShapeRanges({&out_shape_range})
      .Build();
  auto context = context_holder.GetContext<InferShapeRangeContext>();
  ASSERT_NE(context, nullptr);

  ASSERT_NE(context->GetOptionalInputShapeRange(0), nullptr);
  EXPECT_EQ(*context->GetOptionalInputShapeRange(0), in_shape_range1);

  ASSERT_NE(context->GetDynamicInputShapeRange(1, 0), nullptr);
  EXPECT_EQ(*context->GetDynamicInputShapeRange(1, 0), in_shape_range2);

  ASSERT_NE(context->GetDynamicInputShapeRange(1, 1), nullptr);
  EXPECT_EQ(*context->GetDynamicInputShapeRange(1, 1), in_shape_range3);

  EXPECT_EQ(context->GetOptionalInputShapeRange(2), nullptr);

  ASSERT_NE(context->GetOptionalInputShapeRange(3), nullptr);
  EXPECT_EQ(*context->GetOptionalInputShapeRange(3), in_shape_range4);
}

TEST_F(InferShapeRangeContextUT, GetOutShapeOk) {
  Shape same_ele{8, 3, 224, 224};
  gert::Range<Shape> in_shape_range1(&same_ele);
  Shape min2{2, 2, 3, 8};
  Shape max2{2, -1, 3, 8};
  gert::Range<Shape> in_shape_range2(&min2, &max2);
  Shape out_min{8, 3, 224, 224};
  Shape out_max{8, 224, 224, 224};
  gert::Range<Shape> out_shape_range(&out_min, &out_max);
  auto context_holder = InferShapeRangeContextFaker()
      .IrInputNum(2)
      .NodeIoNum(2, 1)
      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z)
      .InputShapeRanges({&in_shape_range1, &in_shape_range2})
      .OutputShapeRanges({&out_shape_range})
      .Build();
  auto context = context_holder.GetContext<InferShapeRangeContext>();
  ASSERT_NE(context, nullptr);

  ASSERT_NE(context->GetOutputShapeRange(0), nullptr);
  EXPECT_EQ(*context->GetOutputShapeRange(0), out_shape_range);

  EXPECT_EQ(context->GetOutputShapeRange(1), nullptr);
}

TEST_F(InferShapeRangeContextUT, GetRequiredInputShapeRangeOk) {
  gert::Tensor min_tensor_1 = {{{8, 3, 224, 224}, {8, 3, 224, 224}},    // shape
                              {ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, {}},  // format
                              kOnDeviceHbm,                                // placement
                              ge::DT_FLOAT16,                              // data type
                              (void *)0x0};
  gert::Tensor max_tensor_1 = {{{-1, 3, 224, 224}, {-1, 3, 224, 224}},    // shape
                              {ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, {}},  // format
                              kOnDeviceHbm,                                // placement
                              ge::DT_FLOAT16,                              // data type
                              (void *)0x0};
  gert::Range<Tensor> in_shape_range1(&min_tensor_1, &max_tensor_1);
  gert::Tensor min_tensor_2 = {{{2, 2, 3, 8}, {2, 2, 3, 8}},    // shape
                              {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                              kOnDeviceHbm,                                // placement
                              ge::DT_FLOAT16,                              // data type
                              (void *)0x0};
  gert::Tensor max_tensor_2 = {{{2, -1, 3, 8}, {2, -1, 3, 8}},    // shape
                              {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                              kOnDeviceHbm,                                // placement
                              ge::DT_FLOAT16,                              // data type
                              (void *)0x0};
  gert::Range<Tensor> in_shape_range2(&min_tensor_2, &max_tensor_2);
  gert::Tensor min_tensor_3 = {{{3, 2, 3, 8}, {3, 2, 3, 8}},    // shape
                              {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                              kOnDeviceHbm,                                // placement
                              ge::DT_FLOAT16,                              // data type
                              (void *)0x0};
  gert::Tensor max_tensor_3 = {{{3, 2, 9, 8}, {3, 2, 9, 8}},    // shape
                              {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                              kOnDeviceHbm,                                // placement
                              ge::DT_FLOAT16,                              // data type
                              (void *)0x0};
  gert::Range<Tensor> in_shape_range3(&min_tensor_3, &max_tensor_3);
  gert::Tensor min_tensor_4 = {{{4, 2, 3, 8}, {4, 2, 3, 8}},    // shape
                              {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                              kOnDeviceHbm,                                // placement
                              ge::DT_FLOAT16,                              // data type
                              (void *)0x12345};
  gert::Tensor max_tensor_4 = {{{4, 2, 3, 16}, {4, 2, 3, 16}},    // shape
                              {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                              kOnDeviceHbm,                                // placement
                              ge::DT_FLOAT16,                              // data type
                              (void *)0x12345};
  gert::Range<Tensor> in_shape_range4(&min_tensor_4, &max_tensor_4);
  gert::Tensor min_tensor_5 = {{{8, 3, 224, 224}, {8, 3, 224, 224}},    // shape
                              {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                              kOnDeviceHbm,                                // placement
                              ge::DT_FLOAT16,                              // data type
                              (void *)0x0};
  gert::Tensor max_tensor_5 = {{{-1, 3, 224, 224}, {-1, 3, 224, 224}},    // shape
                              {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                              kOnDeviceHbm,                                // placement
                              ge::DT_FLOAT16,                              // data type
                              (void *)0x0};
  gert::Range<Tensor> out_shape_range(&min_tensor_5, &max_tensor_5);
  auto context_holder = InferShapeRangeContextFaker()
      .IrInstanceNum({1, 2, 0, 1})
      .NodeIoNum(4, 1)
      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
      .InputShapeRanges({&in_shape_range1, &in_shape_range2, &in_shape_range3, &in_shape_range4})
      .OutputShapeRanges({&out_shape_range})
      .Build();
  auto context = context_holder.GetContext<InferShapeRangeContext>();
  ASSERT_NE(context, nullptr);

  ASSERT_NE(context->GetRequiredInputShapeRange(0), nullptr);
  EXPECT_EQ(*context->GetRequiredInputShapeRange(0)->GetMin(), in_shape_range1.GetMin()->GetStorageShape());
  EXPECT_EQ(*context->GetRequiredInputShapeRange(0)->GetMax(), in_shape_range1.GetMax()->GetStorageShape());
  ASSERT_NE(context->GetRequiredInputTensorRange(0), nullptr);
  EXPECT_EQ(context->GetRequiredInputTensorRange(0)->GetMin()->GetOriginShape(), in_shape_range1.GetMin()->GetOriginShape());
  EXPECT_EQ(context->GetRequiredInputTensorRange(0)->GetMin()->GetAddr(), in_shape_range1.GetMin()->GetAddr());
  EXPECT_EQ(context->GetRequiredInputTensorRange(0)->GetMax()->GetOriginShape(), in_shape_range1.GetMax()->GetOriginShape());
  EXPECT_EQ(context->GetRequiredInputTensorRange(0)->GetMax()->GetAddr(), in_shape_range1.GetMax()->GetAddr());

  ASSERT_NE(context->GetDynamicInputShapeRange(1, 0), nullptr);
  EXPECT_EQ(*context->GetDynamicInputShapeRange(1, 0)->GetMin(), in_shape_range2.GetMin()->GetStorageShape());
  EXPECT_EQ(*context->GetDynamicInputShapeRange(1, 0)->GetMax(), in_shape_range2.GetMax()->GetStorageShape());
  ASSERT_NE(context->GetDynamicInputTensorRange(1, 0), nullptr);
  EXPECT_EQ(context->GetDynamicInputTensorRange(1, 0)->GetMin()->GetOriginShape(), in_shape_range2.GetMin()->GetOriginShape());
  EXPECT_EQ(context->GetDynamicInputTensorRange(1, 0)->GetMin()->GetAddr(), in_shape_range2.GetMin()->GetAddr());
  EXPECT_EQ(context->GetDynamicInputTensorRange(1, 0)->GetMax()->GetOriginShape(), in_shape_range2.GetMax()->GetOriginShape());
  EXPECT_EQ(context->GetDynamicInputTensorRange(1, 0)->GetMax()->GetAddr(), in_shape_range2.GetMax()->GetAddr());

  ASSERT_NE(context->GetDynamicInputShapeRange(1, 1), nullptr);
  EXPECT_EQ(*context->GetDynamicInputShapeRange(1, 1)->GetMin(), in_shape_range3.GetMin()->GetStorageShape());
  EXPECT_EQ(*context->GetDynamicInputShapeRange(1, 1)->GetMax(), in_shape_range3.GetMax()->GetStorageShape());
  ASSERT_NE(context->GetDynamicInputTensorRange(1, 1), nullptr);
  EXPECT_EQ(context->GetDynamicInputTensorRange(1, 1)->GetMin()->GetOriginShape(), in_shape_range3.GetMin()->GetOriginShape());
  EXPECT_EQ(context->GetDynamicInputTensorRange(1, 1)->GetMin()->GetAddr(), in_shape_range3.GetMin()->GetAddr());
  EXPECT_EQ(context->GetDynamicInputTensorRange(1, 1)->GetMax()->GetOriginShape(), in_shape_range3.GetMax()->GetOriginShape());
  EXPECT_EQ(context->GetDynamicInputTensorRange(1, 1)->GetMax()->GetAddr(), in_shape_range3.GetMax()->GetAddr());

  EXPECT_EQ(context->GetOptionalInputShapeRange(2), nullptr);

  ASSERT_NE(context->GetRequiredInputShapeRange(3), nullptr);
  EXPECT_EQ(*context->GetRequiredInputShapeRange(3)->GetMin(), in_shape_range4.GetMin()->GetStorageShape());
  EXPECT_EQ(*context->GetRequiredInputShapeRange(3)->GetMax(), in_shape_range4.GetMax()->GetStorageShape());
  ASSERT_NE(context->GetRequiredInputTensorRange(3), nullptr);
  EXPECT_EQ(context->GetRequiredInputTensorRange(3)->GetMin()->GetOriginShape(), in_shape_range4.GetMin()->GetOriginShape());
  EXPECT_EQ(context->GetRequiredInputTensorRange(3)->GetMin()->GetAddr(), in_shape_range4.GetMin()->GetAddr());
  EXPECT_EQ(context->GetRequiredInputTensorRange(3)->GetMax()->GetOriginShape(), in_shape_range4.GetMax()->GetOriginShape());
  EXPECT_EQ(context->GetRequiredInputTensorRange(3)->GetMax()->GetAddr(), in_shape_range4.GetMax()->GetAddr());

  ASSERT_NE(context->GetOutputShapeRange(0), nullptr);
  EXPECT_EQ(*context->GetOutputShapeRange(0)->GetMin(), out_shape_range.GetMin()->GetStorageShape());
  EXPECT_EQ(*context->GetOutputShapeRange(0)->GetMax(), out_shape_range.GetMax()->GetStorageShape());
}
}  // namespace gert

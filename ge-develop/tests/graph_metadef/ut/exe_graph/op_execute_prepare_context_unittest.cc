/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/runtime/op_execute_prepare_context.h"
#include <gtest/gtest.h>
#include "faker/kernel_run_context_faker.h"

namespace gert {
class OpExecutePrepareContextUT : public testing::Test {};
TEST_F(OpExecutePrepareContextUT, GetInputOutputTest) {
  gert::Tensor in_tensor_1 = {{{8, 3, 224, 224}, {8, 1, 224, 224, 16}},    // shape
                              {ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, {}},  // format
                              kOnDeviceHbm,                                // placement
                              ge::DT_FLOAT16,                              // data type
                              (void *)0x12345};
  gert::Tensor in_tensor_2 = {{{2, 2, 3, 8}, {2, 2, 3, 8}},    // shape
                              {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                              kOnDeviceHbm,                                // placement
                              ge::DT_FLOAT16,                              // data type
                              (void *)0x234565};
  gert::Tensor in_tensor_3 = {{{3, 2, 3, 8}, {3, 2, 3, 8}},    // shape
                              {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                              kOnDeviceHbm,                                // placement
                              ge::DT_FLOAT16,                              // data type
                              (void *)0x45678};
  gert::Tensor in_tensor_4 = {{{4, 2, 3, 8}, {4, 2, 3, 8}},    // shape
                              {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                              kOnHost,                                // placement
                              ge::DT_FLOAT16,                              // data type
                              (void *)0x12345};
  gert::Tensor in_tensor_5 = {{{4, 2, 3, 8}, {4, 2, 3, 8}},    // shape
                              {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                              kOnHost,                                // placement
                              ge::DT_FLOAT16,                              // data type
                              (void *)0x123345};
  gert::Tensor out_tensor_1 = {{{8, 3, 224, 224}, {8, 1, 224, 224, 16}},    // shape
                               {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                               kOnDeviceHbm,                                // placement
                               ge::DT_FLOAT16,                              // data type
                               (void *)0x34586};
  gert::Tensor out_tensor_2 = {{{8, 3, 224, 224}, {8, 1, 224, 224, 16}},    // shape
                               {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                               kOnDeviceHbm,                                // placement
                               ge::DT_FLOAT16,                              // data type
                               (void *)0x342354};
  gert::Tensor out_tensor_3 = {{{8, 3, 224, 224}, {8, 1, 224, 224, 16}},    // shape
                               {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                               kOnDeviceHbm,                                // placement
                               ge::DT_FLOAT16,                              // data type
                               (void *)0x642354};
  OpExecuteOptions execute_option;
  execute_option.allow_hf32[0] = '1';
  execute_option.deterministic = 1;
  execute_option.precision_mode = 1;

  auto param = new DummyOpApiParams();
  auto param2 = new DummyOpApiParams();
  auto op_execute_holder = OpExecutePrepareContextFaker()
                               .IrInstanceNum({1, 2, 1, 0, 1})
                               .IrOutputInstanceNum({2, 1})
                               .NodeIoNum(5, 3)
                               .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                               .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                               .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                               .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                               .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                               .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                               .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                               .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                               .InputTensor({&in_tensor_1, &in_tensor_2, &in_tensor_3, &in_tensor_4, &in_tensor_5})
                               .OutputTensor({&out_tensor_1, &out_tensor_2, &out_tensor_3})
                               .ExecuteOption(&execute_option)
                               .Build();
  auto context = op_execute_holder.GetContext<OpExecutePrepareContext>();
  ASSERT_NE(context, nullptr);
  ASSERT_NE(context->GetInputTensor(0), nullptr);
  EXPECT_EQ(context->GetInputTensor(0)->GetOriginShape(), in_tensor_1.GetOriginShape());
  EXPECT_EQ(context->GetInputTensor(0)->GetAddr(), in_tensor_1.GetAddr());
  ASSERT_NE(context->GetRequiredInputTensor(0), nullptr);
  EXPECT_EQ(context->GetRequiredInputTensor(0)->GetOriginShape(), in_tensor_1.GetOriginShape());
  EXPECT_EQ(context->GetRequiredInputTensor(0)->GetAddr(), in_tensor_1.GetAddr());

  ASSERT_NE(context->GetInputTensor(1), nullptr);
  EXPECT_EQ(context->GetInputTensor(1)->GetOriginShape(), in_tensor_2.GetOriginShape());
  EXPECT_EQ(context->GetInputTensor(1)->GetAddr(), in_tensor_2.GetAddr());
  ASSERT_NE(context->GetDynamicInputTensor(1, 0), nullptr);
  EXPECT_EQ(context->GetDynamicInputTensor(1, 0)->GetOriginShape(), in_tensor_2.GetOriginShape());
  EXPECT_EQ(context->GetDynamicInputTensor(1, 0)->GetAddr(), in_tensor_2.GetAddr());

  ASSERT_NE(context->GetInputTensor(2), nullptr);
  EXPECT_EQ(context->GetInputTensor(2)->GetOriginShape(), in_tensor_3.GetOriginShape());
  EXPECT_EQ(context->GetInputTensor(2)->GetAddr(), in_tensor_3.GetAddr());
  ASSERT_NE(context->GetDynamicInputTensor(1, 1), nullptr);
  EXPECT_EQ(context->GetDynamicInputTensor(1, 1)->GetOriginShape(), in_tensor_3.GetOriginShape());
  EXPECT_EQ(context->GetDynamicInputTensor(1, 1)->GetAddr(), in_tensor_3.GetAddr());

  ASSERT_NE(context->GetInputTensor(3), nullptr);
  EXPECT_EQ(context->GetInputTensor(3)->GetOriginShape(), in_tensor_4.GetOriginShape());
  EXPECT_EQ(context->GetInputTensor(3)->GetAddr(), in_tensor_4.GetAddr());
  ASSERT_NE(context->GetOptionalInputTensor(2), nullptr);
  EXPECT_EQ(context->GetOptionalInputTensor(2)->GetOriginShape(), in_tensor_4.GetOriginShape());
  EXPECT_EQ(context->GetOptionalInputTensor(2)->GetAddr(), in_tensor_4.GetAddr());

  EXPECT_EQ(context->GetOptionalInputTensor(3), nullptr);

  ASSERT_NE(context->GetInputTensor(4), nullptr);
  EXPECT_EQ(context->GetInputTensor(4)->GetOriginShape(), in_tensor_5.GetOriginShape());
  EXPECT_EQ(context->GetInputTensor(4)->GetAddr(), in_tensor_5.GetAddr());
  ASSERT_NE(context->GetRequiredInputTensor(4), nullptr);
  EXPECT_EQ(context->GetRequiredInputTensor(4)->GetOriginShape(), in_tensor_5.GetOriginShape());
  EXPECT_EQ(context->GetRequiredInputTensor(4)->GetAddr(), in_tensor_5.GetAddr());

  ASSERT_NE(context->GetOutputTensor(0), nullptr);
  EXPECT_EQ(context->GetOutputTensor(0)->GetOriginShape(), out_tensor_1.GetOriginShape());
  EXPECT_EQ(context->GetOutputTensor(0)->GetAddr(), out_tensor_1.GetAddr());
  ASSERT_NE(context->GetDynamicOutputTensor(0, 0), nullptr);
  EXPECT_EQ(context->GetDynamicOutputTensor(0, 0)->GetOriginShape(), out_tensor_1.GetOriginShape());
  EXPECT_EQ(context->GetDynamicOutputTensor(0, 0)->GetAddr(), out_tensor_1.GetAddr());

  ASSERT_NE(context->GetOutputTensor(1), nullptr);
  EXPECT_EQ(context->GetOutputTensor(1)->GetOriginShape(), out_tensor_2.GetOriginShape());
  EXPECT_EQ(context->GetOutputTensor(1)->GetAddr(), out_tensor_2.GetAddr());
  ASSERT_NE(context->GetDynamicOutputTensor(0, 1), nullptr);
  EXPECT_EQ(context->GetDynamicOutputTensor(0, 1)->GetOriginShape(), out_tensor_2.GetOriginShape());
  EXPECT_EQ(context->GetDynamicOutputTensor(0, 1)->GetAddr(), out_tensor_2.GetAddr());

  ASSERT_NE(context->GetOutputTensor(2), nullptr);
  EXPECT_EQ(context->GetOutputTensor(2)->GetOriginShape(), out_tensor_3.GetOriginShape());
  EXPECT_EQ(context->GetOutputTensor(2)->GetAddr(), out_tensor_3.GetAddr());
  ASSERT_NE(context->GetRequiredOutputTensor(1), nullptr);
  EXPECT_EQ(context->GetRequiredOutputTensor(1)->GetOriginShape(), out_tensor_3.GetOriginShape());
  EXPECT_EQ(context->GetRequiredOutputTensor(1)->GetAddr(), out_tensor_3.GetAddr());
  EXPECT_EQ(context->GetAllowHf32(), execute_option.allow_hf32);
  EXPECT_EQ(context->GetDeterministic(), true);
  EXPECT_EQ(context->GetPrecisionMode(), execute_option.precision_mode);

  size_t kernel_out_start_num = 5 + 3 + 2; // 5 input tensor, 3 output tensor, 2 append input
  auto set_status = context->SetOpApiParamsWithDefaultDeleter<DummyOpApiParams>(param);
  ASSERT_EQ(set_status, ge::GRAPH_SUCCESS);
  set_status = context->SetOpApiParams(param2, nullptr);
  ASSERT_EQ(set_status, ge::GRAPH_FAILED);
  set_status = context->SetOpApiParams(param2, [](void * const data) {
    delete reinterpret_cast<DummyOpApiParams *>(data);
  });
  ASSERT_EQ(set_status, ge::GRAPH_SUCCESS);

  context->SetWorkspaceSizes({64U, 128U, 256U});
  Chain *out1 = reinterpret_cast<Chain *>(
      (reinterpret_cast<KernelRunContext *>(op_execute_holder.holder.context_))->values[kernel_out_start_num + 1]);
  auto ws_size_vec = out1->GetPointer<TypedContinuousVector<size_t>>();
  ASSERT_NE(ws_size_vec, nullptr);
  EXPECT_EQ(ws_size_vec->GetSize(), 3);
  EXPECT_EQ(ws_size_vec->GetData()[0], 64U);
  context->SetWorkspaceSizes({0U});
  EXPECT_EQ(ws_size_vec->GetSize(), 1);
  EXPECT_EQ(ws_size_vec->GetData()[0], 0U);
  EXPECT_NE(context->SetWorkspaceSizes({10U, 10U, 10U, 10U}), ge::GRAPH_SUCCESS);
  EXPECT_EQ(ws_size_vec->GetSize(), 1);
  EXPECT_EQ(ws_size_vec->GetData()[0], 0U);
}
}
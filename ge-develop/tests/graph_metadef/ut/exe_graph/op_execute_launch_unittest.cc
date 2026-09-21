/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "exe_graph/runtime/op_execute_launch_context.h"
#include <gtest/gtest.h>
#include "faker/kernel_run_context_faker.h"

namespace gert {
class OpExecuteLaunchContextUT : public testing::Test {};
TEST_F(OpExecuteLaunchContextUT, GetInputOutputTest) {
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

    void *op_api_params = (void *)0x12;
    auto ws_addr_vector = ContinuousVector::Create<TensorData *>(3);
    auto *ws_addr_cont_vec = reinterpret_cast<TypedContinuousVector<TensorData *> *>(ws_addr_vector.get());
    ws_addr_cont_vec->SetSize(3);
    auto tensor_data_addr = static_cast<TensorData**>(ws_addr_cont_vec->MutableData());
    auto td1 = TensorData((void *)0x34, nullptr);
    auto td2 = TensorData((void *)0x35, nullptr);
    auto td3 = TensorData((void *)0x36, nullptr);
    tensor_data_addr[0] = &td1;
    tensor_data_addr[1] = &td2;
    tensor_data_addr[2] = &td3;
    auto ws_size_vector = ContinuousVector::Create<size_t>(1);
    auto *ws_size_cont_vec = reinterpret_cast<TypedContinuousVector<size_t> *>(ws_size_vector.get());
    ws_size_cont_vec->MutableData()[0] = 32U;
    ws_size_cont_vec->SetSize(1);
    rtStream stream = (void *)0x56;
  auto op_execute_holder = OpExecuteLaunchContextFaker()
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
                               .OpApiParams(op_api_params)
                               .WorkspaceSize(ws_size_vector.get())
                               .WorkspaceAddr(ws_addr_vector.get())
                               .Stream(stream)
                               .Build();
  auto context = op_execute_holder.GetContext<OpExecuteLaunchContext>();
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

  ASSERT_EQ(context->GetOpApiParams(), op_api_params);
  auto workspace_addrs = context->GetWorkspaceAddrs();
  ASSERT_EQ(workspace_addrs->GetData()[0]->GetAddr(), (void *)0x34);
  auto workspace_sizes = context->GetWorkspaceSizes();
  ASSERT_EQ(workspace_sizes->GetData()[0], 32U);
  ASSERT_EQ(context->GetStream(), stream);
}
}
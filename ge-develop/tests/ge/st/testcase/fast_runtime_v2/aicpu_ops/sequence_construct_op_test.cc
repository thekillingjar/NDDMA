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
#include "register/kernel_registry_impl.h"
#include "faker/kernel_run_context_facker.h"
#include "exe_graph/runtime/extended_kernel_context.h"
#include <iostream>
#include "framework/common/debug/ge_log.h"

namespace gert {
struct SequenceConstructComputeTest : public testing::Test {
  SequenceConstructComputeTest() {
    sequenceConstructCompute = KernelRegistry::GetInstance().FindKernelFuncs("SequenceConstructCompute");
  }
  const KernelRegistry::KernelFuncs* sequenceConstructCompute;
};

TEST_F(SequenceConstructComputeTest, failed) {
  uint64_t session_id = 1;
  uint64_t container_id = 1;
  uint32_t input_num = 0;
  auto kernelHolder = gert::KernelRunContextFaker()
    .KernelIONum(3, 0)
    .Inputs({reinterpret_cast<void *>(session_id),
             reinterpret_cast<void *>(container_id),
             reinterpret_cast<void *>(input_num),
              })
    .Build();
  auto context = kernelHolder.GetContext<KernelContext>();
  ASSERT_EQ(sequenceConstructCompute->run_func(context), ge::PARAM_INVALID);
}

TEST_F(SequenceConstructComputeTest, success) {
  uint64_t session_id = 1;
  uint64_t container_id = 1;
  uint32_t input_num = 1;

  // build input shape & input tensor
  size_t total_size = 0;
  auto input_tensor_holder = gert::Tensor::CreateFollowing(2, ge::DT_INT32, total_size);
  auto input_tensor = reinterpret_cast<gert::Tensor*>(input_tensor_holder.get());
  input_tensor->SetOriginFormat(ge::FORMAT_ND);
  input_tensor->SetStorageFormat(ge::FORMAT_ND);
  input_tensor->SetPlacement(gert::kFollowing);
  input_tensor->MutableOriginShape() = {2};
  input_tensor->MutableStorageShape() = {2};

  auto input_tensor_data = reinterpret_cast<int32_t*>(input_tensor->GetAddr());
  input_tensor_data[0] = 1;
  input_tensor_data[1] = 2;
  input_tensor->SetData(gert::TensorData{input_tensor_data, nullptr});

  auto output_tensor_holder = gert::Tensor::CreateFollowing(1, ge::DT_UINT64, total_size);
  auto output_tensor = reinterpret_cast<gert::Tensor*>(output_tensor_holder.get());
  output_tensor->SetOriginFormat(ge::FORMAT_ND);
  output_tensor->SetStorageFormat(ge::FORMAT_ND);
  output_tensor->SetPlacement(gert::kFollowing);
  output_tensor->MutableOriginShape() = {};
  output_tensor->MutableStorageShape() = {};

  auto output_tensor_data = reinterpret_cast<uint64_t*>(output_tensor->GetAddr());
  output_tensor_data[0] = 1;
  output_tensor->SetData(gert::TensorData{output_tensor_data, nullptr});
  auto kernelHolder = gert::KernelRunContextFaker()
                          .KernelIONum(6, 0)
                          .Inputs({reinterpret_cast<void*>(session_id), reinterpret_cast<void*>(container_id),
                                   reinterpret_cast<void*>(input_num), reinterpret_cast<void*>(&(input_tensor->MutableTensorData())),
                                   reinterpret_cast<void*>(&(input_tensor->MutableStorageShape())),
                                   reinterpret_cast<void*>(output_tensor)})
                          .NodeIoNum(1, 1)
                          .IrInstanceNum({1})
                          .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                          .Build();
  auto context = kernelHolder.GetContext<KernelContext>();
  ASSERT_EQ(sequenceConstructCompute->run_func(context), ge::GRAPH_SUCCESS);
}

TEST_F(SequenceConstructComputeTest, input0_desc_null) {
  uint64_t session_id = 1;
  uint64_t container_id = 1;
  uint32_t input_num = 1;

  // build input shape & input tensor
  size_t total_size = 0;
  auto input_tensor_holder = gert::Tensor::CreateFollowing(2, ge::DT_INT32, total_size);
  auto input_tensor = reinterpret_cast<gert::Tensor*>(input_tensor_holder.get());
  input_tensor->SetOriginFormat(ge::FORMAT_ND);
  input_tensor->SetStorageFormat(ge::FORMAT_ND);
  input_tensor->SetPlacement(gert::kFollowing);
  input_tensor->MutableOriginShape() = {2};
  input_tensor->MutableStorageShape() = {2};

  auto input_tensor_data = reinterpret_cast<int32_t*>(input_tensor->GetAddr());
  input_tensor_data[0] = 1;
  input_tensor_data[1] = 2;
  input_tensor->SetData(gert::TensorData{input_tensor_data, nullptr});

  auto output_tensor_holder = gert::Tensor::CreateFollowing(1, ge::DT_UINT64, total_size);
  auto output_tensor = reinterpret_cast<gert::Tensor*>(output_tensor_holder.get());
  output_tensor->SetOriginFormat(ge::FORMAT_ND);
  output_tensor->SetStorageFormat(ge::FORMAT_ND);
  output_tensor->SetPlacement(gert::kFollowing);
  output_tensor->MutableOriginShape() = {};
  output_tensor->MutableStorageShape() = {};

  auto output_tensor_data = reinterpret_cast<uint64_t*>(output_tensor->GetAddr());
  output_tensor_data[0] = 1;
  output_tensor->SetData(gert::TensorData{output_tensor_data, nullptr});
  auto kernelHolder = gert::KernelRunContextFaker()
                          .KernelIONum(6, 0)
                          .Inputs({reinterpret_cast<void*>(session_id), reinterpret_cast<void*>(container_id),
                                   reinterpret_cast<void*>(input_num), nullptr,
                                   reinterpret_cast<void*>(&(input_tensor->MutableStorageShape())),
                                   reinterpret_cast<void*>(output_tensor)})
                          .NodeIoNum(1, 1)
                          .IrInstanceNum({1})
                          .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                          .Build();
  auto context = kernelHolder.GetContext<KernelContext>();
  ASSERT_EQ(sequenceConstructCompute->run_func(context), ge::PARAM_INVALID);
}

TEST_F(SequenceConstructComputeTest, output_null) {
  uint64_t session_id = 1;
  uint64_t container_id = 1;
  uint32_t input_num = 1;

  // build input shape & input tensor
  size_t total_size = 0;
  auto input_tensor_holder = gert::Tensor::CreateFollowing(2, ge::DT_INT32, total_size);
  auto input_tensor = reinterpret_cast<gert::Tensor*>(input_tensor_holder.get());
  input_tensor->SetOriginFormat(ge::FORMAT_ND);
  input_tensor->SetStorageFormat(ge::FORMAT_ND);
  input_tensor->SetPlacement(gert::kFollowing);
  input_tensor->MutableOriginShape() = {2};
  input_tensor->MutableStorageShape() = {2};

  auto input_tensor_data = reinterpret_cast<int32_t*>(input_tensor->GetAddr());
  input_tensor_data[0] = 1;
  input_tensor_data[1] = 2;
  input_tensor->SetData(gert::TensorData{input_tensor_data, nullptr});

  auto output_tensor_holder = gert::Tensor::CreateFollowing(1, ge::DT_UINT64, total_size);
  auto output_tensor = reinterpret_cast<gert::Tensor*>(output_tensor_holder.get());
  output_tensor->SetOriginFormat(ge::FORMAT_ND);
  output_tensor->SetStorageFormat(ge::FORMAT_ND);
  output_tensor->SetPlacement(gert::kFollowing);
  output_tensor->MutableOriginShape() = {};
  output_tensor->MutableStorageShape() = {};

  auto output_tensor_data = reinterpret_cast<uint64_t*>(output_tensor->GetAddr());
  output_tensor_data[0] = 1;
  output_tensor->SetData(gert::TensorData{output_tensor_data, nullptr});
  auto kernelHolder = gert::KernelRunContextFaker()
                          .KernelIONum(6, 0)
                          .Inputs({reinterpret_cast<void*>(session_id), reinterpret_cast<void*>(container_id),
                                   reinterpret_cast<void*>(input_num), reinterpret_cast<void*>(&(input_tensor->MutableTensorData())),
                                   reinterpret_cast<void*>(&(input_tensor->MutableStorageShape())),
                                   nullptr})
                          .NodeIoNum(1, 1)
                          .IrInstanceNum({1})
                          .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                          .Build();
  auto context = kernelHolder.GetContext<KernelContext>();
  ASSERT_EQ(sequenceConstructCompute->run_func(context), ge::PARAM_INVALID);
}
}  // namespace gert
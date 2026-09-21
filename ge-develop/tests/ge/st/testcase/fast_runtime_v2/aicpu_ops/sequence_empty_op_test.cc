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
#include <iostream>
#include "framework/common/debug/ge_log.h"

namespace gert {
struct SequenceEmptyComputeTest : public testing::Test {
  SequenceEmptyComputeTest() {
    SequenceEmptyCompute = KernelRegistry::GetInstance().FindKernelFuncs("SequenceEmptyCompute");
  }
  const KernelRegistry::KernelFuncs* SequenceEmptyCompute;
};

TEST_F(SequenceEmptyComputeTest, success) {
  uint64_t session_id = 1;
  uint64_t container_id = 1;
  uint32_t dtype = 1;

  size_t total_size = 0;
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
                          .KernelIONum(4, 0)
                          .Inputs({reinterpret_cast<void*>(session_id), reinterpret_cast<void*>(container_id),
                                   reinterpret_cast<void*>(dtype), reinterpret_cast<void*>(output_tensor)})
                          .Build();
  auto context = kernelHolder.GetContext<KernelContext>();
  std::cout << "SequenceEmptyCompute->run_func : " << (void*)(SequenceEmptyCompute->run_func) << std::endl;
  ASSERT_EQ(SequenceEmptyCompute->run_func(context), ge::GRAPH_SUCCESS);
}

TEST_F(SequenceEmptyComputeTest, output_null) {
  uint64_t session_id = 1;
  uint64_t container_id = 1;
  uint32_t dtype = 1;

  size_t total_size = 0;
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
                          .KernelIONum(4, 0)
                          .Inputs({reinterpret_cast<void*>(session_id), reinterpret_cast<void*>(container_id),
                                   reinterpret_cast<void*>(dtype), nullptr})
                          .Build();
  auto context = kernelHolder.GetContext<KernelContext>();
  std::cout << "SequenceEmptyCompute->run_func : " << (void*)(SequenceEmptyCompute->run_func) << std::endl;
  ASSERT_EQ(SequenceEmptyCompute->run_func(context), ge::PARAM_INVALID);
}
}  // namespace gert
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel/memory/sink_weight_assigner.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "kernel/memory/single_stream_l2_allocator.h"
#include <gtest/gtest.h>
#include "register/kernel_registry.h"
#include "faker/kernel_run_context_facker.h"
#include "runtime/model_v2_executor.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "core/utils/tensor_utils.h"

namespace gert {
namespace {
KernelRegistry &registry = KernelRegistry::GetInstance();
}
namespace kernel {
class SinkWeightUT : public testing::Test {};

TEST_F(SinkWeightUT, AssignMemoryInputsAddrInvalid) {
  ASSERT_NE(registry.FindKernelFuncs("AssignWeightMemory")->run_func, nullptr);
  auto context_holder = KernelRunContextFaker()
                            .KernelIONum(static_cast<size_t>(AssignWeightMemoryInputs::kNum),
                                         static_cast<size_t>(AssignWeightMemoryOutputs::kNum))
                            .Inputs({
                                nullptr,
                                nullptr,
                            })
                            .Build();
  auto blank_context = context_holder.GetContext<KernelContext>();
  ASSERT_EQ(registry.FindKernelFuncs("AssignWeightMemory")->run_func(blank_context), ge::GRAPH_FAILED);
}

TEST_F(SinkWeightUT, AssignMemoryInputs) {
  size_t size = 100;
  void *device_mem = malloc(size);
  memory::SinkWeightAssigner base_mem(device_mem, size);
  int logic_stream_id = 0;
  TensorData tensor_data1{ge::ValueToPtr(95), nullptr, 10, kTensorPlacementEnd};  // 95 + 10 > 100 ,fail
  auto invalid_holder = KernelRunContextFaker()
                            .KernelIONum(static_cast<size_t>(AssignWeightMemoryInputs::kNum),
                                         static_cast<size_t>(AssignWeightMemoryOutputs::kNum))
                            .Inputs({&tensor_data1, &base_mem, &logic_stream_id})
                            .Build();
  auto invalid_context = invalid_holder.GetContext<KernelContext>();
  ASSERT_EQ(registry.FindKernelFuncs("AssignWeightMemory")->outputs_creator(nullptr, invalid_context),
            ge::GRAPH_SUCCESS);
  ASSERT_NE(registry.FindKernelFuncs("AssignWeightMemory")->run_func(invalid_context), ge::GRAPH_SUCCESS);

  TensorData tensor_data2{ge::ValueToPtr(80), nullptr, 10, kOnDeviceHbm};  // 80 + 10 < 100 ,success
  auto valid_holder = KernelRunContextFaker()
                          .KernelIONum(static_cast<size_t>(AssignWeightMemoryInputs::kNum),
                                       static_cast<size_t>(AssignWeightMemoryOutputs::kNum))
                          .Inputs({&tensor_data2, &base_mem, &logic_stream_id})
                          .Build();
  auto valid_context = valid_holder.GetContext<KernelContext>();
  ASSERT_EQ(registry.FindKernelFuncs("AssignWeightMemory")->outputs_creator(nullptr, valid_context),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("AssignWeightMemory")->run_func(valid_context), ge::GRAPH_SUCCESS);
  auto device_gtd = valid_context->GetOutputPointer<GertTensorData>(0U);
  ASSERT_EQ(device_gtd->GetPlacement(), kOnDeviceHbm);
  free(device_mem);
}
TEST_F(SinkWeightUT, GetWeightMemTest) {
  gert::memory::CachingMemAllocator caching_mem_allocator(0);
  memory::SingleStreamL2Allocator single_stream_allocator(&caching_mem_allocator);

  ASSERT_NE(registry.FindKernelFuncs("GetOrCreateWeightMem")->run_func, nullptr);
  size_t model_weight_size = 100;
  size_t device_mem_size = 200;
  void *device_mem = malloc(device_mem_size);
  OuterWeightMem mem_info = {device_mem, device_mem_size};
  auto context_holder1 = KernelRunContextFaker()
                             .KernelIONum(static_cast<size_t>(GetOrCreateWeightInputs::kNum),
                                          static_cast<size_t>(GetOrCreateWeightOutputs::kNum))
                             .Inputs({(void *)model_weight_size, &mem_info, &single_stream_allocator})
                             .Build();
  auto valid_context = context_holder1.GetContext<KernelContext>();
  ASSERT_EQ(registry.FindKernelFuncs("GetOrCreateWeightMem")->outputs_creator(nullptr, valid_context),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("GetOrCreateWeightMem")->run_func(valid_context), ge::SUCCESS);
  auto real_device_mem = valid_context->GetOutputPointer<GertTensorData>(0);
  ASSERT_EQ(real_device_mem->GetAddr(), device_mem);
  free(device_mem);
}
TEST_F(SinkWeightUT, CreateWeightMemTest) {
  gert::memory::CachingMemAllocator caching_mem_allocator(0);
  memory::SingleStreamL2Allocator single_stream_allocator(&caching_mem_allocator);

  ASSERT_NE(registry.FindKernelFuncs("GetOrCreateWeightMem")->run_func, nullptr);
  size_t model_weight_size = 100;
  size_t device_mem_size = 80;
  void *device_mem = malloc(device_mem_size);
  OuterWeightMem mem_info = {device_mem, device_mem_size};
  auto context_holder1 = KernelRunContextFaker()
                             .KernelIONum(static_cast<size_t>(GetOrCreateWeightInputs::kNum),
                                          static_cast<size_t>(GetOrCreateWeightOutputs::kNum))
                             .Inputs({(void *)model_weight_size, &mem_info, &single_stream_allocator})
                             .Build();
  auto valid_context = context_holder1.GetContext<KernelContext>();
  ASSERT_EQ(registry.FindKernelFuncs("GetOrCreateWeightMem")->outputs_creator(nullptr, valid_context),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("GetOrCreateWeightMem")->run_func(valid_context), ge::SUCCESS);
  auto real_device_mem = valid_context->GetOutputPointer<GertTensorData>(0);
  ASSERT_NE(real_device_mem->GetAddr(), nullptr);
  ASSERT_NE(real_device_mem->GetAddr(), device_mem);
  free(device_mem);
}
}  // namespace kernel
}  // namespace gert
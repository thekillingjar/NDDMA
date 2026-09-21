/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel/outputs/model_outputs.h"
#include <gtest/gtest.h>
#include "faker/kernel_run_context_facker.h"
#include "stub/gert_runtime_stub.h"
#include "register/kernel_registry.h"
#include "kernel/memory/single_stream_l2_allocator.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "kernel/memory/host_mem_allocator.h"
#include "exe_graph/runtime/gert_mem_allocator.h"
#include "kernel/tensor_attr.h"
#include "core/utils/tensor_utils.h"
#include "checker/memory_profiling_log_matcher.h"

namespace gert {
class ModelOutpusUT : public testing::Test {
 public:
  KernelRegistry &registry = KernelRegistry::GetInstance();
  void CheckTensorData(KernelRunContextHolder &run_context) {
    auto tensor = run_context.GetContext<KernelContext>()->GetInputPointer<Tensor>(2);
    auto tensor_data = run_context.GetContext<KernelContext>()->GetOutputPointer<GertTensorData>(0);
    ASSERT_NE(tensor, nullptr);
    ASSERT_NE(tensor_data, nullptr);
    ASSERT_EQ(tensor->GetAddr(), tensor_data->GetAddr());
    ASSERT_NE(tensor->GetAddr(), nullptr);
  }
  void CheckTensor(KernelRunContextHolder &run_context) {
    CheckTensorData(run_context);
    auto tensor = run_context.GetContext<KernelContext>()->GetInputPointer<Tensor>(2);
    ASSERT_EQ(tensor->GetStorageShape(), storage_shape.GetStorageShape());
    ASSERT_EQ(tensor->GetOriginShape(), storage_shape.GetOriginShape());
    ASSERT_EQ(tensor->GetDataType(), attr.data_type);
    ASSERT_EQ(tensor->GetPlacement(), attr.placement);
  }
  StorageShape storage_shape{{4}, {4, 1}};
  kernel::BuildTensorAttr attr{kOnDeviceHbm, ge::DT_FLOAT16, {ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, ExpandDimsType()}};
};
TEST_F(ModelOutpusUT, AllocModelOutTensor_UseAddressOutside_ModelOutputsAllocdOutside) {
  memory::CachingMemAllocator allocator(0, RT_MEMORY_HBM);
  memory::SingleStreamL2Allocator single_stream_l2_allocator(&allocator);

  Tensor tensor_holder{storage_shape, attr.storage_format, attr.placement, attr.data_type, nullptr};
  auto mem_block = allocator.Malloc(1024);
  tensor_holder.MutableTensorData() = TensorUtils::ToTensorData(mem_block, 1024, kOnDeviceHbm);
  {
    TensorData tensor_data_share1;
    ASSERT_EQ(tensor_data_share1.ShareFrom(tensor_holder.MutableTensorData()), ge::GRAPH_SUCCESS);
    ASSERT_EQ(mem_block->GetCount(), 2);
  }
  ASSERT_EQ(mem_block->GetCount(), 1);
  auto addr = tensor_holder.GetAddr();
  auto placement = tensor_holder.GetPlacement();
  GertTensorData data;
  auto run_context = KernelRunContextFaker().Inputs({&single_stream_l2_allocator, (void *)8, &tensor_holder}).Outputs({&data}).Build();
  ASSERT_EQ(kernel::AllocModelOutTensor(run_context.GetContext<KernelContext>()), ge::GRAPH_SUCCESS);

  auto tensor_data = run_context.GetContext<KernelContext>()->GetOutputPointer<GertTensorData>(0);
  ASSERT_NE(tensor_data, nullptr);
  ASSERT_EQ(tensor_data->GetAddr(), addr);
  ASSERT_EQ(tensor_data->GetPlacement(), placement);
  CheckTensorData(run_context);

  // free
  run_context.GetContext<KernelContext>()->GetOutput(0)->Set(nullptr, nullptr);
  tensor_holder.MutableTensorData().Free();
  allocator.Finalize();
}
TEST_F(ModelOutpusUT, AllocModelOutTensor_AllocMemory_PlacementOfOutSideIsWrong) {
  memory::CachingMemAllocator allocator(0, RT_MEMORY_HBM);

  Tensor outside_tensor_holder{storage_shape, attr.storage_format, attr.placement, attr.data_type, nullptr};
  auto mem_block = allocator.Malloc(1024);
  outside_tensor_holder.MutableTensorData() = TensorUtils::ToTensorData(mem_block, 1024, kOnDeviceHbm);
  auto addr = outside_tensor_holder.GetAddr();
  auto placement = outside_tensor_holder.GetPlacement();

  auto host_mem_allocator = std::make_shared<memory::HostMemAllocator>();
  memory::HostGertMemAllocator host_gert_mem_allocator(host_mem_allocator.get());
  GertTensorData data;
  auto run_context = KernelRunContextFaker().Inputs({&host_gert_mem_allocator, (void *)8, &outside_tensor_holder})
      .Outputs({&data}).Build();
  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().NoConsoleOut().SetLevelInfo();
  ASSERT_EQ(kernel::AllocModelOutTensor(run_context.GetContext<KernelContext>()), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(kAllocRe) >= 0);

  auto tensor_data = run_context.GetContext<KernelContext>()->GetOutputPointer<GertTensorData>(0);
  ASSERT_NE(tensor_data, nullptr);
  ASSERT_NE(tensor_data->GetAddr(), nullptr);
  ASSERT_NE(tensor_data->GetAddr(), addr);
  ASSERT_NE(tensor_data->GetPlacement(), placement);
  ASSERT_EQ(tensor_data->GetPlacement(), kOnHost);

  // free
  run_context.GetContext<KernelContext>()->GetOutput(0)->Set(nullptr, nullptr);
  outside_tensor_holder.MutableTensorData().Free();
  allocator.Finalize();
}
TEST_F(ModelOutpusUT, AllocModelOutTensor_AllocMemory_OutTensorIsNull) {
  memory::CachingMemAllocator allocator(0, RT_MEMORY_HBM);

  Tensor outside_tensor_holder{storage_shape, attr.storage_format, attr.placement, attr.data_type, nullptr};
  ge::MemBlock mem_block(allocator, 0, 0);
  outside_tensor_holder.MutableTensorData() = TensorUtils::ToTensorData(&mem_block, 1024, kOnDeviceHbm);
  auto placement = outside_tensor_holder.GetPlacement();

  auto l2_allocators_holder = ContinuousVector::Create<GertAllocator *>(1UL);
  memory::SingleStreamL2Allocator single_stream_l2_allocator(&allocator);
  GertTensorData data;
  auto run_context = KernelRunContextFaker().Inputs({&single_stream_l2_allocator, (void *)8, &outside_tensor_holder})
      .Outputs({&data}).Build();
  ASSERT_EQ(kernel::AllocModelOutTensor(run_context.GetContext<KernelContext>()), ge::GRAPH_SUCCESS);

  auto tensor_data = run_context.GetContext<KernelContext>()->GetOutputPointer<GertTensorData>(0);
  ASSERT_NE(tensor_data, nullptr);
  ASSERT_NE(tensor_data->GetAddr(), nullptr);
  ASSERT_EQ(tensor_data->GetSize(), 8);
  ASSERT_EQ(tensor_data->GetPlacement(), placement);

  // free
  run_context.GetContext<KernelContext>()->GetOutput(0)->Set(nullptr, nullptr);
  outside_tensor_holder.MutableTensorData().Free();
  allocator.Finalize();
}
TEST_F(ModelOutpusUT, AllocModelOutTensor_AllOk_NotAllocAddr) {
  memory::CachingMemAllocator allocator(0, RT_MEMORY_HBM);
  memory::SingleStreamL2Allocator single_stream_l2_allocator(&allocator);
  Tensor tensor_holder{storage_shape, attr.storage_format, attr.placement, attr.data_type, nullptr};
  GertTensorData data;

  auto run_context = KernelRunContextFaker().Inputs({&single_stream_l2_allocator, (void *)8, &tensor_holder}).Outputs({&data}).Build();
  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().NoConsoleOut().SetLevelInfo();
  ASSERT_EQ(kernel::AllocModelOutTensor(run_context.GetContext<KernelContext>()), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(kAllocRe) >= 0);

  CheckTensorData(run_context);

  // free
  run_context.GetContext<KernelContext>()->GetOutput(0)->Set(nullptr, nullptr);
  tensor_holder.MutableTensorData().Free();
  allocator.Finalize();
}
TEST_F(ModelOutpusUT, AllocModelOutTensor_AllOk_NotAllocTensor) {
  memory::CachingMemAllocator allocator(0, RT_MEMORY_HBM);
  memory::SingleStreamL2Allocator single_stream_l2_allocator(&allocator);
  GertTensorData data;

  auto run_context = KernelRunContextFaker().Inputs({&single_stream_l2_allocator, (void *)8, nullptr}).Outputs({&data}).Build();
  ASSERT_EQ(kernel::AllocModelOutTensor(run_context.GetContext<KernelContext>()), ge::GRAPH_SUCCESS);

  CheckTensorData(run_context);

  // free
  run_context.GetContext<KernelContext>()->MutableInput(2)->Set(nullptr, nullptr);
  allocator.Finalize();
}
TEST_F(ModelOutpusUT, AllocModelOutTensor_Failed_InputsInvalid) {
  // inputs num invalid
  auto run_context = KernelRunContextFaker().KernelIONum(1, 2).Build();
  ASSERT_NE(kernel::AllocModelOutTensor(run_context.GetContext<KernelContext>()), ge::GRAPH_SUCCESS);

  // inputs nullptr
  memory::CachingMemAllocator allocator(0, RT_MEMORY_HBM);
  GertTensorData data;

  run_context = KernelRunContextFaker().Inputs({nullptr, nullptr, nullptr}).Outputs({&data}).Build();
  ASSERT_NE(kernel::AllocModelOutTensor(run_context.GetContext<KernelContext>()), ge::GRAPH_SUCCESS);

  allocator.Finalize();
}
TEST_F(ModelOutpusUT, AllocModelOutTensor_Failed_OutputsInvalid) {
  memory::CachingMemAllocator allocator(0, RT_MEMORY_HBM);
  memory::SingleStreamL2Allocator single_stream_l2_allocator(&allocator);
  GertTensorData data;

  auto run_context = KernelRunContextFaker().Inputs({&single_stream_l2_allocator, (void *)8, nullptr}).Outputs({nullptr}).Build();
  ASSERT_NE(kernel::AllocModelOutTensor(run_context.GetContext<KernelContext>()), ge::GRAPH_SUCCESS);

  allocator.Finalize();
}
// todo 确定输出size校验规则后重新放开
#if 0
TEST_F(ModelOutpusUT, AllocModelOutTensor_Failed_AllocOutputTensorButNotEnough) {
  memory::CachingMemAllocator allocator(0, RT_MEMORY_HBM);
  memory::SingleStreamL2Allocator single_stream_l2_allocator(&allocator);
  Tensor tensor{storage_shape, attr.storage_format, attr.placement, attr.data_type, nullptr};
  tensor.MutableTensorData().SetAddr((void *)1, nullptr);
  tensor.MutableTensorData().SetSize(1024);
  auto addr = tensor.GetAddr();
  GertTensorData data;

  auto run_context =
      KernelRunContextFaker().Inputs({&single_stream_l2_allocator, (void *)1025, &tensor}).Outputs({&data}).Build();

  GertRuntimeStub stub;
  ASSERT_NE(kernel::AllocModelOutTensor(run_context.GetContext<KernelContext>()), ge::GRAPH_SUCCESS);
  EXPECT_EQ(stub.GetSlogStub().FindErrorLogEndsWith("Invalid output tensor size 1024, size should be at least 1025"),
            0);
}
#endif
}  // namespace gert
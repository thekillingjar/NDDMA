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
#include <kernel/memory/mem_block.h>
#include "stub/gert_runtime_stub.h"
#include "runtime/rt.h"
#include "faker/kernel_run_context_facker.h"
#include "register/kernel_registry.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "kernel/memory/single_stream_l2_allocator.h"
#include "graph/op_desc.h"
#include "graph/compute_graph.h"
#include "faker/dsacore_taskdef_faker.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "core/utils/tensor_utils.h"

namespace gert {
namespace {
class DsacoreKernelLaunchST : public testing::Test {
 public:
  KernelRegistry &registry = KernelRegistry::GetInstance();
};

TEST_F(DsacoreKernelLaunchST, test_run_launch_dsacore_kernel) {
  auto run_context = BuildKernelRunContext(18, 0);
  rtStarsDsaSqe_t dsa_sqe;
  size_t workspace_size = 2;
  auto workspace_addr_ptr = ContinuousVector::Create<GertTensorData *>(workspace_size);
  auto work_space_vector = reinterpret_cast<ContinuousVector *>(workspace_addr_ptr.get());
  rtStream_t stream = (void *)0x01;
  work_space_vector->SetSize(workspace_size);

  auto fake_hbm_addr = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[100]);
  ScalableConfig config;
  memory::CachingMemAllocator stub_allocator{0, 1, config};
  ge::MemBlock mem_block = {stub_allocator, reinterpret_cast<void *>(fake_hbm_addr.get()), 100};
  memory::SingleStreamL2Allocator single_stream_l2_allocator{&stub_allocator};
  memory::MultiStreamMemBlock ms_block;
  ms_block.ReInit(&single_stream_l2_allocator, &mem_block, memory::BlockAllocType(memory::BlockAllocType::kNorm, 0U));
  auto gtd = TensorUtils::ToGertTensorData(&ms_block, TensorPlacement::kOnDeviceHbm, 0);

  auto work_space_ptr = reinterpret_cast<GertTensorData **>(work_space_vector->MutableData());
  for (size_t i = 0; i < work_space_vector->GetSize(); i++) {
    work_space_ptr[i] = &gtd;
  }

  auto sizes_holder = ContinuousVector::Create<size_t>(8);
  auto sizes = reinterpret_cast<TypedContinuousVector<size_t> *>(sizes_holder.get());
  sizes->SetSize(2);
  sizes->MutableData()[0] = 64;
  sizes->MutableData()[1] = 64;
  auto seed_type = 0;
  auto random_count_type = 0;
  run_context.value_holder[0].Set(reinterpret_cast<void *>(&dsa_sqe), nullptr);
  run_context.value_holder[1].Set(workspace_addr_ptr.get(), nullptr);
  run_context.value_holder[2].Set(sizes_holder.get(), nullptr);

  run_context.value_holder[3].Set(reinterpret_cast<rtStream_t>(stream), nullptr);

  run_context.value_holder[4].Set(reinterpret_cast<void *>(seed_type), nullptr);
  run_context.value_holder[5].Set(reinterpret_cast<void *>(random_count_type), nullptr);

  std::string seed_type_str = "seed";
  std::string random_count_type_str = "random";
  run_context.value_holder[6].Set(const_cast<char *>(seed_type_str.c_str()), nullptr);
  run_context.value_holder[7].Set(const_cast<char *>(random_count_type_str.c_str()), nullptr);

  auto input1_type = 0;
  run_context.value_holder[8].Set(reinterpret_cast<void *>(input1_type), nullptr);
  std::string input1_value_str = "input1";
  std::string input2_value_str = "input2";
  run_context.value_holder[9].Set(const_cast<char *>(input1_value_str.c_str()), nullptr);
  run_context.value_holder[10].Set(const_cast<char *>(input2_value_str.c_str()), nullptr);

  auto input_num = 4;
  auto output_num = 1;
  run_context.value_holder[11].Set(reinterpret_cast<void *>(input_num), nullptr);
  run_context.value_holder[12].Set(reinterpret_cast<void *>(output_num), nullptr);

  uint64_t io_addr = 0x11;
  void *tensor_addr = reinterpret_cast<void *>(io_addr);
  GertTensorData tensor_data = {tensor_addr, 0U, kTensorPlacementEnd, -1};

  run_context.value_holder[13].Set(&tensor_data, nullptr);
  run_context.value_holder[14].Set(&tensor_data, nullptr);
  run_context.value_holder[15].Set(&tensor_data, nullptr);
  run_context.value_holder[16].Set(&tensor_data, nullptr);

  run_context.value_holder[17].Set(&tensor_data, nullptr);

  ASSERT_EQ(registry.FindKernelFuncs("UpdateSqeArg")->run_func(run_context), ge::GRAPH_SUCCESS);
}


TEST_F(DsacoreKernelLaunchST, test_run_launch_dsacore_kernel_dsa_1) {
  auto run_context = BuildKernelRunContext(18, 0);
  rtStarsDsaSqe_t dsa_sqe;
  size_t workspace_size = 1;
  auto workspace_addr_ptr = ContinuousVector::Create<GertTensorData *>(workspace_size);
  auto work_space_vector = reinterpret_cast<ContinuousVector *>(workspace_addr_ptr.get());
  rtStream_t stream = (void *)0x01;
  work_space_vector->SetSize(workspace_size);

  auto fake_hbm_addr = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[100]);
  memory::CachingMemAllocator stub_allocator{0, 1};
  ge::MemBlock mem_block = {stub_allocator, reinterpret_cast<void *>(fake_hbm_addr.get()), 100};
  memory::SingleStreamL2Allocator single_stream_l2_allocator{&stub_allocator};
  memory::MultiStreamMemBlock ms_block;
  ms_block.ReInit(&single_stream_l2_allocator, &mem_block, memory::BlockAllocType(memory::BlockAllocType::kNorm, 0U));
  auto gtd = TensorUtils::ToGertTensorData(&ms_block, TensorPlacement::kOnDeviceHbm, 0);

  auto work_space_ptr = reinterpret_cast<GertTensorData **>(work_space_vector->MutableData());
  for (size_t i = 0; i < work_space_vector->GetSize(); i++) {
    work_space_ptr[i] = &gtd;
  }

  auto sizes_holder = ContinuousVector::Create<size_t>(8);
  auto sizes = reinterpret_cast<TypedContinuousVector<size_t> *>(sizes_holder.get());
  sizes->SetSize(2);
  sizes->MutableData()[0] = 64;
  sizes->MutableData()[1] = 64;
  auto seed_type = 0;
  auto random_count_type = 0;
  run_context.value_holder[0].Set(reinterpret_cast<void *>(&dsa_sqe), nullptr);
  run_context.value_holder[1].Set(workspace_addr_ptr.get(), nullptr);
  run_context.value_holder[2].Set(sizes_holder.get(), nullptr);

  run_context.value_holder[3].Set(reinterpret_cast<rtStream_t>(stream), nullptr);

  run_context.value_holder[4].Set(reinterpret_cast<void *>(seed_type), nullptr);
  run_context.value_holder[5].Set(reinterpret_cast<void *>(random_count_type), nullptr);

  std::string seed_type_str = "seed";
  std::string random_count_type_str = "random";
  run_context.value_holder[6].Set(const_cast<char *>(seed_type_str.c_str()), nullptr);
  run_context.value_holder[7].Set(const_cast<char *>(random_count_type_str.c_str()), nullptr);

  auto input1_type = 0;
  run_context.value_holder[8].Set(reinterpret_cast<void *>(input1_type), nullptr);
  std::string input1_value_str = "input1";
  std::string input2_value_str = "input2";
  run_context.value_holder[9].Set(const_cast<char *>(input1_value_str.c_str()), nullptr);
  run_context.value_holder[10].Set(const_cast<char *>(input2_value_str.c_str()), nullptr);

  auto input_num = 4;
  auto output_num = 1;
  run_context.value_holder[11].Set(reinterpret_cast<void *>(input_num), nullptr);
  run_context.value_holder[12].Set(reinterpret_cast<void *>(output_num), nullptr);

  uint64_t io_addr = 0x11;
  void *tensor_addr = reinterpret_cast<void *>(io_addr);
  GertTensorData tensor_data = {tensor_addr, 0U, kTensorPlacementEnd, -1};

  run_context.value_holder[13].Set(&tensor_data, nullptr);
  run_context.value_holder[14].Set(&tensor_data, nullptr);
  run_context.value_holder[15].Set(&tensor_data, nullptr);
  run_context.value_holder[16].Set(&tensor_data, nullptr);

  run_context.value_holder[17].Set(&tensor_data, nullptr);

  ASSERT_EQ(registry.FindKernelFuncs("UpdateSqeArg")->run_func(run_context), ge::GRAPH_SUCCESS);
}

TEST_F(DsacoreKernelLaunchST, test_run_launch_dsacore_kernel_dsa_2) {
  auto run_context = BuildKernelRunContext(18, 0);
  rtStarsDsaSqe_t dsa_sqe;
  size_t workspace_size = 1;
  auto workspace_addr_ptr = ContinuousVector::Create<GertTensorData *>(workspace_size);
  auto work_space_vector = reinterpret_cast<ContinuousVector *>(workspace_addr_ptr.get());
  rtStream_t stream = (void *)0x01;
  work_space_vector->SetSize(workspace_size);

  auto fake_hbm_addr = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[100]);
  memory::CachingMemAllocator stub_allocator{0, 1};
  ge::MemBlock mem_block = {stub_allocator, reinterpret_cast<void *>(fake_hbm_addr.get()), 100};
  memory::SingleStreamL2Allocator single_stream_l2_allocator{&stub_allocator};
  memory::MultiStreamMemBlock ms_block;
  ms_block.ReInit(&single_stream_l2_allocator, &mem_block, memory::BlockAllocType(memory::BlockAllocType::kNorm, 0U));
  auto gtd = TensorUtils::ToGertTensorData(&ms_block, TensorPlacement::kOnDeviceHbm, 0);

  auto work_space_ptr = reinterpret_cast<GertTensorData **>(work_space_vector->MutableData());
  for (size_t i = 0; i < work_space_vector->GetSize(); i++) {
    work_space_ptr[i] = &gtd;
  }

  auto sizes_holder = ContinuousVector::Create<size_t>(8);
  auto sizes = reinterpret_cast<TypedContinuousVector<size_t> *>(sizes_holder.get());
  sizes->SetSize(2);
  sizes->MutableData()[0] = 64;
  sizes->MutableData()[1] = 64;
  auto seed_type = 0;
  auto random_count_type = 0;
  run_context.value_holder[0].Set(reinterpret_cast<void *>(&dsa_sqe), nullptr);
  run_context.value_holder[1].Set(workspace_addr_ptr.get(), nullptr);
  run_context.value_holder[2].Set(sizes_holder.get(), nullptr);

  run_context.value_holder[3].Set(reinterpret_cast<rtStream_t>(stream), nullptr);

  run_context.value_holder[4].Set(reinterpret_cast<void *>(seed_type), nullptr);
  run_context.value_holder[5].Set(reinterpret_cast<void *>(random_count_type), nullptr);

  std::string seed_type_str = "seed";
  std::string random_count_type_str = "random";
  run_context.value_holder[6].Set(const_cast<char *>(seed_type_str.c_str()), nullptr);
  run_context.value_holder[7].Set(const_cast<char *>(random_count_type_str.c_str()), nullptr);

  auto input1_type = 0;
  run_context.value_holder[8].Set(reinterpret_cast<void *>(input1_type), nullptr);
  std::string input1_value_str = "input1";
  std::string input2_value_str = "input2";
  run_context.value_holder[9].Set(const_cast<char *>(input1_value_str.c_str()), nullptr);
  run_context.value_holder[10].Set(const_cast<char *>(input2_value_str.c_str()), nullptr);

  auto input_num = 4;
  auto output_num = 1;
  run_context.value_holder[11].Set(reinterpret_cast<void *>(input_num), nullptr);
  run_context.value_holder[12].Set(reinterpret_cast<void *>(output_num), nullptr);

  uint64_t io_addr = 0x11;
  void *tensor_addr = reinterpret_cast<void *>(io_addr);
  GertTensorData tensor_data = {tensor_addr, 0U, kTensorPlacementEnd, -1};

  run_context.value_holder[13].Set(&tensor_data, nullptr);
  run_context.value_holder[14].Set(&tensor_data, nullptr);
  run_context.value_holder[15].Set(&tensor_data, nullptr);
  run_context.value_holder[16].Set(&tensor_data, nullptr);

  run_context.value_holder[17].Set(&tensor_data, nullptr);

  ASSERT_EQ(registry.FindKernelFuncs("UpdateSqeArg")->run_func(run_context), ge::GRAPH_SUCCESS);
}

TEST_F(DsacoreKernelLaunchST, test_run_launch_dsacore_kernel_dsa_3) {
  auto run_context = BuildKernelRunContext(18, 0);
  rtStarsDsaSqe_t dsa_sqe;
  size_t workspace_size = 1;
  auto workspace_addr_ptr = ContinuousVector::Create<GertTensorData *>(workspace_size);
  auto work_space_vector = reinterpret_cast<ContinuousVector *>(workspace_addr_ptr.get());
  rtStream_t stream = (void *)0x01;
  work_space_vector->SetSize(workspace_size);

  auto fake_hbm_addr = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[100]);
  memory::CachingMemAllocator stub_allocator{0, 1};
  ge::MemBlock mem_block = {stub_allocator, reinterpret_cast<void *>(fake_hbm_addr.get()), 100};
  memory::SingleStreamL2Allocator single_stream_l2_allocator{&stub_allocator};
  memory::MultiStreamMemBlock ms_block;
  ms_block.ReInit(&single_stream_l2_allocator, &mem_block, memory::BlockAllocType(memory::BlockAllocType::kNorm, 0U));
  auto gtd = TensorUtils::ToGertTensorData(&ms_block, TensorPlacement::kOnDeviceHbm, 0);

  auto work_space_ptr = reinterpret_cast<GertTensorData **>(work_space_vector->MutableData());
  for (size_t i = 0; i < work_space_vector->GetSize(); i++) {
    work_space_ptr[i] = &gtd;
  }

  auto sizes_holder = ContinuousVector::Create<size_t>(8);
  auto sizes = reinterpret_cast<TypedContinuousVector<size_t> *>(sizes_holder.get());
  sizes->SetSize(2);
  sizes->MutableData()[0] = 64;
  sizes->MutableData()[1] = 64;
  auto seed_type = 0;
  auto random_count_type = 0;
  run_context.value_holder[0].Set(reinterpret_cast<void *>(&dsa_sqe), nullptr);
  run_context.value_holder[1].Set(workspace_addr_ptr.get(), nullptr);
  run_context.value_holder[2].Set(sizes_holder.get(), nullptr);

  run_context.value_holder[3].Set(reinterpret_cast<rtStream_t>(stream), nullptr);

  run_context.value_holder[4].Set(reinterpret_cast<void *>(seed_type), nullptr);
  run_context.value_holder[5].Set(reinterpret_cast<void *>(random_count_type), nullptr);

  std::string seed_type_str = "seed";
  std::string random_count_type_str = "random";
  run_context.value_holder[6].Set(const_cast<char *>(seed_type_str.c_str()), nullptr);
  run_context.value_holder[7].Set(const_cast<char *>(random_count_type_str.c_str()), nullptr);

  auto input1_type = 1;
  run_context.value_holder[8].Set(reinterpret_cast<void *>(input1_type), nullptr);
  std::string input1_value_str = "input1";
  std::string input2_value_str = "input2";
  run_context.value_holder[9].Set(const_cast<char *>(input1_value_str.c_str()), nullptr);
  run_context.value_holder[10].Set(const_cast<char *>(input2_value_str.c_str()), nullptr);

  auto input_num = 4;
  auto output_num = 1;
  run_context.value_holder[11].Set(reinterpret_cast<void *>(input_num), nullptr);
  run_context.value_holder[12].Set(reinterpret_cast<void *>(output_num), nullptr);

  uint64_t io_addr = 0x11;
  void *tensor_addr = reinterpret_cast<void *>(io_addr);
  GertTensorData tensor_data = {tensor_addr, 0U, kTensorPlacementEnd, -1};

  run_context.value_holder[13].Set(&tensor_data, nullptr);
  run_context.value_holder[14].Set(&tensor_data, nullptr);
  run_context.value_holder[15].Set(&tensor_data, nullptr);
  run_context.value_holder[16].Set(&tensor_data, nullptr);

  run_context.value_holder[17].Set(&tensor_data, nullptr);

  ASSERT_EQ(registry.FindKernelFuncs("UpdateSqeArg")->run_func(run_context), ge::GRAPH_SUCCESS);
}
}
}  // namespace gert

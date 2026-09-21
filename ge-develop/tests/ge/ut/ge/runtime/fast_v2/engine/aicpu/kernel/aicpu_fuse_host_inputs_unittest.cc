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
#include "faker/kernel_run_context_facker.h"
#include "register/kernel_registry.h"
#include "graph/op_desc.h"
#include "graph/compute_graph.h"
#include "graph/load/model_manager/model_manager.h"
#include "exe_graph/runtime/storage_shape.h"
#include "engine/aicpu/kernel/aicpu_args_handler.h"
#include "exe_graph/runtime/gert_mem_allocator.h"
#include "kernel/memory/single_stream_l2_allocator.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "checker/memory_profiling_log_matcher.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "kernel/memory/single_stream_l2_allocator.h"

namespace gert {
class AicpuFuseHostInputsUT: public testing::Test {
 public:
  memory::CachingMemAllocator caching_mem_allocator{0};
  memory::SingleStreamL2Allocator single_stream_l2_allocator{&caching_mem_allocator};
  KernelRegistry &registry = KernelRegistry::GetInstance();
};

TEST_F(AicpuFuseHostInputsUT, TestTfOneInputSuccess) {
  auto run_context = BuildKernelRunContext(8, 1);
  AicpuTfArgsHandler args_handler("temp", 1, false);
  std::string arg_data = "111";
  std::string task_info = "222";
  size_t ext_info_size = 20;
  args_handler.BuildTfArgs(arg_data, task_info, ext_info_size, 0, 0);
  run_context.value_holder[0].Set(&args_handler, nullptr);

  std::vector<int32_t> indexes(1, 0);
  run_context.value_holder[1].Set(indexes.data(), nullptr);  
  run_context.value_holder[2].Set(reinterpret_cast<void *>(0x11), nullptr); // stream
  auto allocator = memory::CachingMemAllocator(1, RT_MEMORY_TYPE_DEVICE);
  memory::SingleStreamL2Allocator single_stream_l2_allocator(&allocator);

  run_context.value_holder[3].Set(&single_stream_l2_allocator, nullptr);

  // for small host input
  auto temp_addr = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[20]);
  GertTensorData temp_buffer = {temp_addr.get(), 20, kOnHost, -1};
  auto data_type = 1; // DT_FLOAT16
  StorageShape shape({1}, {1});
  auto size = 512;
  run_context.value_holder[4].Set(&temp_buffer, nullptr);
  run_context.value_holder[5].Set(reinterpret_cast<void *>(size), nullptr);
  run_context.value_holder[6].Set(&shape, nullptr);
  run_context.value_holder[7].Set(reinterpret_cast<void *>(data_type), nullptr);

  auto &output_holder = run_context.value_holder[8];
  EXPECT_EQ(registry.FindKernelFuncs("AicpuFuseHost")->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
  auto output_data = output_holder.GetPointer<GertTensorData>();
  EXPECT_NE(output_data, nullptr);
  EXPECT_EQ(output_data->GetAddr(), nullptr);
  EXPECT_EQ(output_data->GetSize(), 0);
  EXPECT_EQ(output_data->GetPlacement(), kOnHost);
  EXPECT_EQ(args_handler.GetHostInputSize(), 0);
  EXPECT_EQ(registry.FindKernelFuncs("AicpuFuseHost")->run_func(run_context), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_data->GetPlacement(), kOnHost);
  EXPECT_EQ(args_handler.GetHostInputSize(), 64);

  // for big host input
  auto temp_addr2 = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[500]);
  GertTensorData temp_buffer2 = {temp_addr2.get(), 500, kOnHost, -1};
  StorageShape shape2({50}, {50});
  auto data_type2 = 10; // DT_UINT64
  run_context.value_holder[4].Set(&temp_buffer2, nullptr);
  run_context.value_holder[6].Set(&shape2, nullptr);
  run_context.value_holder[7].Set(reinterpret_cast<void *>(data_type2), nullptr);
  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().NoConsoleOut().SetLevelInfo();
  EXPECT_EQ(registry.FindKernelFuncs("AicpuFuseHost")->run_func(run_context), ge::GRAPH_SUCCESS);
  EXPECT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(kAllocRe) >= 0);
  // copy & device tensor data
  EXPECT_NE(output_data->GetAddr(), nullptr);
  EXPECT_NE(output_data->GetSize(), 0);
  EXPECT_NE(output_data->GetPlacement(), kOnHost);
  run_context.FreeAll();
}

TEST_F(AicpuFuseHostInputsUT, TestCcOneInputSuccess) {
  auto run_context = BuildKernelRunContext(8, 1);
  AicpuCCArgsHandler args_handler("temp", 1, false);
  std::string arg_data = "111";
  std::string kernel_name = "222";
  std::string so_name = "333";
  size_t ext_info_size = 20;
  args_handler.BuildCCArgs(arg_data, kernel_name, so_name, ext_info_size);
  run_context.value_holder[0].Set(&args_handler, nullptr);

  std::vector<int32_t> indexes(1, 0);
  run_context.value_holder[1].Set(indexes.data(), nullptr);  
  run_context.value_holder[2].Set(reinterpret_cast<void *>(0x11), nullptr); // stream
  run_context.value_holder[3].Set(&single_stream_l2_allocator, nullptr);

  // for small host input
  auto temp_addr = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[20]);
  GertTensorData temp_buffer = {temp_addr.get(), 20, kOnHost, -1};
  auto data_type = 1; // DT_FLOAT16
  StorageShape shape({1}, {1});
  auto size = 512;
  run_context.value_holder[4].Set(&temp_buffer, nullptr);
  run_context.value_holder[5].Set(reinterpret_cast<void *>(size), nullptr);
  run_context.value_holder[6].Set(&shape, nullptr);
  run_context.value_holder[7].Set(reinterpret_cast<void *>(data_type), nullptr);

  auto &output_holder = run_context.value_holder[8];
  EXPECT_EQ(registry.FindKernelFuncs("AicpuFuseHost")->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
  auto output_data = output_holder.GetPointer<GertTensorData>();
  EXPECT_NE(output_data, nullptr);
  EXPECT_EQ(output_data->GetAddr(), nullptr);
  EXPECT_EQ(output_data->GetSize(), 0);
  EXPECT_EQ(output_data->GetPlacement(), kOnHost);
  EXPECT_EQ(args_handler.GetHostInputSize(), 0);
  EXPECT_EQ(registry.FindKernelFuncs("AicpuFuseHost")->run_func(run_context), ge::GRAPH_SUCCESS);
  EXPECT_EQ(output_data->GetPlacement(), kOnHost);
  EXPECT_EQ(args_handler.GetHostInputSize(), 4);
  uint64_t *io_addrs = reinterpret_cast<uint64_t *>(args_handler.GetIoAddr());
  EXPECT_NE(io_addrs[0], 0);
  run_context.FreeAll();
}

TEST_F(AicpuFuseHostInputsUT, TestMultiInputSuccess) {
  auto run_context = BuildKernelRunContext(12, 2);
  AicpuTfArgsHandler args_handler("temp", 3, false); // 3 inputs
  std::string arg_data = "111";
  std::string task_info = "222";
  size_t ext_info_size = 20;
  args_handler.BuildTfArgs(arg_data, task_info, ext_info_size, 0, 0);
  run_context.value_holder[0].Set(&args_handler, nullptr);

  // 0, 1, 2, only fuse 1 & 2
  std::vector<int32_t> indexes{1, 2};
  run_context.value_holder[1].Set(indexes.data(), nullptr);  
  run_context.value_holder[2].Set(reinterpret_cast<void *>(0x11), nullptr); // stream
  run_context.value_holder[3].Set(&single_stream_l2_allocator, nullptr);

  // input 1 device tensor
  auto temp_addr = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[20]);
  GertTensorData temp_buffer = {temp_addr.get(), 20, kOnDeviceHbm, -1};
  auto data_type = 1;
  StorageShape shape({1}, {1});
  auto size = 512;
  run_context.value_holder[4].Set(&temp_buffer, nullptr);
  run_context.value_holder[5].Set(reinterpret_cast<void *>(size), nullptr);
  run_context.value_holder[6].Set(&shape, nullptr);
  run_context.value_holder[7].Set(reinterpret_cast<void *>(data_type), nullptr);

  // input 2 host tensor
  auto temp_addr2 = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[20]);
  GertTensorData temp_buffer2 = {temp_addr2.get(), 20, kOnHost, -1};
  run_context.value_holder[8].Set(&temp_buffer2, nullptr);
  run_context.value_holder[9].Set(reinterpret_cast<void *>(size), nullptr);
  run_context.value_holder[10].Set(&shape, nullptr);
  run_context.value_holder[11].Set(reinterpret_cast<void *>(data_type), nullptr);

  auto &output_holder1 = run_context.value_holder[12];
  auto &output_holder2 = run_context.value_holder[13];
  EXPECT_EQ(registry.FindKernelFuncs("AicpuFuseHost")->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
  auto output_data1 = output_holder1.GetPointer<GertTensorData>();
  EXPECT_NE(output_data1, nullptr);
  EXPECT_EQ(output_data1->GetAddr(), nullptr);
  EXPECT_EQ(output_data1->GetSize(), 0);
  EXPECT_EQ(output_data1->GetPlacement(), kOnHost);

  auto output_data2 = output_holder2.GetPointer<GertTensorData>();
  EXPECT_NE(output_data2, nullptr);
  EXPECT_EQ(output_data2->GetAddr(), nullptr);
  EXPECT_EQ(output_data2->GetSize(), 0);
  EXPECT_EQ(output_data2->GetPlacement(), kOnHost);

  EXPECT_EQ(args_handler.GetHostInputSize(), 0);
  EXPECT_EQ(registry.FindKernelFuncs("AicpuFuseHost")->run_func(run_context), ge::GRAPH_SUCCESS);
  // output1 shared from input
  EXPECT_EQ(output_data1->GetAddr(), temp_addr.get());
  EXPECT_EQ(output_data1->GetSize(), 20);
  EXPECT_EQ(output_data1->GetPlacement(), kOnDeviceHbm);

  // output2 set host on args
  EXPECT_EQ(output_data2->GetAddr(), nullptr);
  EXPECT_EQ(output_data2->GetSize(), 0);
  EXPECT_EQ(output_data2->GetPlacement(), kOnHost);

  // check args
  uint64_t *io_addrs = reinterpret_cast<uint64_t *>(args_handler.GetIoAddr());
  EXPECT_EQ(io_addrs[0], 0);
  EXPECT_EQ(io_addrs[1], 0);
  EXPECT_NE(io_addrs[2], 0);
  EXPECT_EQ(args_handler.GetHostInputSize(), 64);
  run_context.FreeAll();
}
}  // namespace gert

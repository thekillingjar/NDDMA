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
#include "graph/compute_graph.h"
#include "kernel/common_kernel_impl/calc_tenorsize_from_shape.h"
#include "kernel/memory/single_stream_l2_allocator.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "exe_graph/runtime/gert_mem_allocator.h"
#include "faker/fake_value.h"
#include "core/utils/tensor_utils.h"
#include "checker/memory_profiling_log_matcher.h"
#include "exe_graph/lowering/shape_utils.h"

namespace gert {
class AicpuSequenceKernelUT : public testing::Test {
 public:
  memory::CachingMemAllocator caching_mem_allocator{0};
  memory::SingleStreamL2Allocator single_stream_l2_allocator{&caching_mem_allocator};
  KernelRegistry &registry = KernelRegistry::GetInstance();
};

TEST_F(AicpuSequenceKernelUT, test_calc_tensorsize_of_specified_shape) {
  auto run_context = BuildKernelRunContext(2, 1);
  ge::DataType data_type = ge::DT_INT32;
  uint32_t shape_num = 3;
  auto shapes = ContinuousVector::Create<Shape>(shape_num);
  auto shapes_vec = reinterpret_cast<ContinuousVector*>(shapes.get());
  shapes_vec->SetSize(shape_num);
  auto shapes_data = static_cast<Shape*>(shapes_vec->MutableData());
  for (size_t i = 0; i < shape_num; i++) {
    shapes_data[i].SetDimNum(2);
    shapes_data[i].SetDim(0, 2);
    shapes_data[i].SetDim(1, 3);
  }
  run_context.value_holder[0].Set(reinterpret_cast<void *>(data_type), nullptr);
  run_context.value_holder[1].Set(shapes.get(), nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("CalcTensorSizeFromSpecifiedShape")->run_func(run_context), ge::GRAPH_SUCCESS);
  auto alloc_context = run_context.GetContext<KernelContext>();
  auto sizes = alloc_context->GetOutputPointer<TypedContinuousVector<uint64_t>>(0);
  auto sizes_data = sizes->GetData();
  ASSERT_EQ(sizes_data[0], 64);
}

TEST_F(AicpuSequenceKernelUT, test_get_split_vec_with_2_input) {
  auto run_context = KernelRunContextFaker()
                            .IrInputNum(5)
                            .KernelIONum(5,1)
                            .NodeIoNum(5, 1)
                            .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z)
                            .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z)
                            .Build();
  uint32_t axis = 1;
  uint32_t input_num = 2;
  run_context.value_holder[0].Set(reinterpret_cast<void *>(axis), nullptr);
  run_context.value_holder[1].Set(reinterpret_cast<void *>(input_num), nullptr);
  StorageShape x_shape = {{2,6}, {2,6}};
  run_context.value_holder[2].Set(&x_shape, nullptr);
  StorageShape split_shape = {{3}, {3}};
  run_context.value_holder[3].Set(&split_shape, nullptr);
  vector<int32_t> split_vec {2,3,1};
  GertTensorData split_data = {split_vec.data(), 0U, kOnDeviceHbm, -1};
  run_context.value_holder[4].Set(&split_data, nullptr);

  ASSERT_EQ(registry.FindKernelFuncs("GetSplitVecWithInput")->run_func(run_context), ge::GRAPH_SUCCESS);
  auto alloc_context = run_context.GetContext<KernelContext>();
  auto shapes = alloc_context->GetOutputPointer<TypedContinuousVector<Shape>>(0);

  ASSERT_EQ(shapes->GetSize(), 3);
  ASSERT_EQ(shapes->GetData()[0], Shape({2, 2}));
  ASSERT_EQ(shapes->GetData()[1], Shape({2, 3}));
  ASSERT_EQ(shapes->GetData()[2], Shape({2, 1}));
}

TEST_F(AicpuSequenceKernelUT, test_get_split_vec_with_1_input) {
  auto run_context = KernelRunContextFaker()
                            .IrInputNum(3)
                            .KernelIONum(3,1)
                            .NodeIoNum(3, 1)
                            .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z)
                            .Build();
  uint32_t axis = 1;
  uint32_t input_num = 1;
  run_context.value_holder[0].Set(reinterpret_cast<void *>(axis), nullptr);
  run_context.value_holder[1].Set(reinterpret_cast<void *>(input_num), nullptr);

  StorageShape x_shape = {{2,6}, {2,6}};
  run_context.value_holder[2].Set(&x_shape, nullptr);

  ASSERT_EQ(registry.FindKernelFuncs("GetSplitVecWithInput")->run_func(run_context), ge::GRAPH_SUCCESS);
  auto alloc_context = run_context.GetContext<KernelContext>();
  auto shapes = alloc_context->GetOutputPointer<TypedContinuousVector<Shape>>(0);
  ASSERT_EQ(shapes->GetSize(), 6);
  ASSERT_EQ(shapes->GetData()[0], Shape({2,1}));
  ASSERT_EQ(shapes->GetData()[5], Shape({2,1}));
}

TEST_F(AicpuSequenceKernelUT, test_get_split_vec_with_2_input_scalar) {
  auto run_context = KernelRunContextFaker()
                            .IrInputNum(5)
                            .KernelIONum(5,1)
                            .NodeIoNum(5, 1)
                            .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z)
                            .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z)
                            .Build();
  uint32_t axis = 1;
  uint32_t input_num = 2;
  run_context.value_holder[0].Set(reinterpret_cast<void *>(axis), nullptr);
  run_context.value_holder[1].Set(reinterpret_cast<void *>(input_num), nullptr);

  StorageShape x_shape = {{2,6}, {2,6}};
  run_context.value_holder[2].Set(&x_shape, nullptr);
  StorageShape split_shape = {{}, {}};
  run_context.value_holder[3].Set(&split_shape, nullptr);
  uint32_t split_scalar = 4;
  GertTensorData split_data = {&split_scalar, 0U, kOnDeviceHbm, -1};
  run_context.value_holder[4].Set(&split_data, nullptr);

  ASSERT_EQ(registry.FindKernelFuncs("GetSplitVecWithInput")->run_func(run_context), ge::GRAPH_SUCCESS);
  auto alloc_context = run_context.GetContext<KernelContext>();
  auto shapes = alloc_context->GetOutputPointer<TypedContinuousVector<Shape>>(0);
  ASSERT_EQ(shapes->GetSize(), 2);
  ASSERT_EQ(shapes->GetData()[0], Shape({2,4}));
  ASSERT_EQ(shapes->GetData()[1], Shape({2,2}));
}

TEST_F(AicpuSequenceKernelUT, test_split_to_sequence_do_compute) {
  auto run_context = KernelRunContextFaker()
                            .IrInputNum(5)
                            .KernelIONum(5,1)
                            .NodeIoNum(5, 1)
                            .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z)
                            .Build();
  uint32_t axis = 1;
  run_context.value_holder[0].Set(reinterpret_cast<void *>(axis), nullptr);

  StorageShape x_shape = {{2,6}, {2,6}};
  run_context.value_holder[1].Set(&x_shape, nullptr);

  vector<int64_t> x_vec {1,2,3,4,5,6,7,8,9,10,11,12};
  GertTensorData x_data = {x_vec.data(), 0U, kOnDeviceHbm, -1};
  run_context.value_holder[2].Set(&x_data, nullptr);

  uint32_t shape_num = 2;
  auto shapes = ContinuousVector::Create<Shape>(shape_num);
  auto shapes_vec = reinterpret_cast<ContinuousVector*>(shapes.get());
  shapes_vec->SetSize(shape_num);
  memory::CachingMemAllocator allocator(0, RT_MEMORY_HBM);
  uint64_t data_size;
  auto tensor_datas = ContinuousVector::Create<TensorData*>(shape_num);
  auto tensor_data_vec = reinterpret_cast<ContinuousVector*>(tensor_datas.get());
  tensor_data_vec->SetSize(shape_num);
  auto tesor_data_addr = static_cast<TensorData**>(tensor_data_vec->MutableData());
  auto shapes_data = static_cast<Shape*>(shapes_vec->MutableData());
  for (size_t i = 0; i < shape_num; i++) {
    shapes_data[i].SetDimNum(2);
    shapes_data[i].SetDim(0, 2);
    shapes_data[i].SetDim(1, 3);
    CalcAlignedSizeByShape(shapes_data[i], ge::DT_INT64, data_size);
    auto mem_block = caching_mem_allocator.Malloc(data_size);
    tesor_data_addr[i] = new TensorData();
    *(tesor_data_addr[i]) = TensorUtils::ToTensorData(mem_block, data_size, kOnDeviceHbm);
  }
  run_context.value_holder[3].Set(shapes.get(), nullptr);
  run_context.value_holder[4].Set(tensor_datas.get(), nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("SplitToSequenceDoCompute")->run_func(run_context), ge::GRAPH_SUCCESS);
  for (size_t i = 0; i < shape_num; i++) {
    delete tesor_data_addr[i];
  }
  allocator.Finalize();
}

TEST_F(AicpuSequenceKernelUT, test_alloc_specified_mem) {
  auto run_context = BuildKernelRunContext(2, 1);
  run_context.value_holder[0].Set(&single_stream_l2_allocator, nullptr);
  uint32_t shape_num = 3;
  auto shape_sizes = ContinuousVector::Create<uint64_t>(shape_num);
  auto shapes_sizes_vec = reinterpret_cast<ContinuousVector*>(shape_sizes.get());
  shapes_sizes_vec->SetSize(shape_num);
  auto shape_size_data = static_cast<uint64_t*>(shapes_sizes_vec->MutableData());
  for (size_t i = 0; i < shape_num; i++) {
    shape_size_data[i] = i + 1;
  }
  run_context.value_holder[1].Set(shape_sizes.get(), nullptr);
  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().NoConsoleOut().SetLevelInfo();
  ASSERT_EQ(registry.FindKernelFuncs("AllocSpecifiedMem")->run_func(run_context), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(kAllocRe) >= 0);
  auto alloc_context = run_context.GetContext<KernelContext>();
  auto tensor_datas = alloc_context->GetOutputPointer<TypedContinuousVector<TensorData*>>(0);
  for (size_t i = 0; i < tensor_datas->GetSize(); i++) {
    delete tensor_datas->MutableData()[i];
  }
}

TEST_F(AicpuSequenceKernelUT, test_free_specified_mem) {
  auto data_num = 3;
  auto tensor_datas = ContinuousVector::Create<GertTensorData*>(data_num);
  auto tensor_data_vec = reinterpret_cast<ContinuousVector*>(tensor_datas.get());
  tensor_data_vec->SetSize(data_num);
  auto tesor_data_addr = static_cast<TensorData**>(tensor_data_vec->MutableData());
  for (size_t i = 0; i < static_cast<size_t>(data_num); i++) {
    auto mem_block = caching_mem_allocator.Malloc(1U);
    tesor_data_addr[i] = new TensorData();
    *(tesor_data_addr[i]) = TensorUtils::ToTensorData(mem_block, 1U, kOnDeviceHbm);
  }

  auto run_context = BuildKernelRunContext(1, 1);
  run_context.value_holder[0].Set(tensor_datas.get(), nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("FreeSpecifiedMem")->run_func(run_context), ge::GRAPH_SUCCESS);
}

TEST_F(AicpuSequenceKernelUT, test_get_sequence_handle_shape) {
  auto run_context = KernelRunContextFaker().KernelIONum(0, 1).NodeIoNum(1, 1).IrInputNum(1).Build();
  auto kernel_funcs = registry.FindKernelFuncs("GetSequenceHandleShape");
  ASSERT_EQ(kernel_funcs->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(kernel_funcs->run_func(run_context), ge::GRAPH_SUCCESS);
  auto alloc_context = run_context.GetContext<KernelContext>();
  auto shapes = alloc_context->GetOutputPointer<StorageShape>(0);
  ASSERT_EQ(shapes->GetStorageShape().GetDimNum(), 0);
  ASSERT_EQ(shapes->GetOriginShape().GetDimNum(), 0);
}

TEST_F(AicpuSequenceKernelUT, test_BuildInferShapeFromTensorCompute) {
  size_t total_size = 0;
  auto input_tensor_holder = gert::Tensor::CreateFollowing(1, ge::DT_UINT64, total_size);
  auto input_tensor = reinterpret_cast<gert::Tensor*>(input_tensor_holder.get());
  input_tensor->SetOriginFormat(ge::FORMAT_ND);
  input_tensor->SetStorageFormat(ge::FORMAT_ND);
  input_tensor->SetPlacement(gert::kFollowing);
  input_tensor->MutableOriginShape() = {10, 20};
  input_tensor->MutableStorageShape() = {10, 20};

  auto input_tensor_data = reinterpret_cast<uint64_t*>(input_tensor->GetAddr());
  input_tensor_data[0] = 0x55aa;
  input_tensor->SetData(gert::TensorData{input_tensor_data, nullptr});

  auto kernel_holder = gert::KernelRunContextFaker()
                          .KernelIONum(1, 1)
                          .NodeIoNum(1, 1)
                          .Inputs({reinterpret_cast<void*>(input_tensor)})
                          .Build();
  auto context = kernel_holder.GetContext<KernelContext>();
  auto kernel_funcs = registry.FindKernelFuncs("BuildInferShapeFromTensorCompute");
  ASSERT_EQ(kernel_funcs->outputs_creator(nullptr, context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(kernel_funcs->run_func(context), ge::GRAPH_SUCCESS);
  auto shape = context->GetOutputPointer<StorageShape>(0UL);
  ASSERT_NE(shape, nullptr);
  EXPECT_EQ(shape->GetOriginShape(), Shape({10, 20}));
  EXPECT_EQ(shape->GetStorageShape(), Shape({10, 20}));
  auto tensor = context->GetOutputPointer<Tensor>(0UL); 
  ASSERT_NE(tensor, nullptr);
  EXPECT_EQ(tensor->GetOriginShape(), Shape({10, 20}));
  EXPECT_EQ(tensor->GetStorageShape(), Shape({10, 20}));
  EXPECT_EQ(tensor->GetAddr(), nullptr);
  kernel_holder.FreeAll();
}

TEST_F(AicpuSequenceKernelUT, test_BuildAddrFromTensorCompute) {
  size_t total_size = 0;
  auto input_tensor_holder = gert::Tensor::CreateFollowing(1, ge::DT_UINT64, total_size);
  auto input_tensor = reinterpret_cast<gert::Tensor*>(input_tensor_holder.get());
  auto input_tensor_data = reinterpret_cast<uint64_t*>(input_tensor->GetAddr());
  input_tensor_data[0] = 0x55aa;
  input_tensor->SetData(gert::TensorData{input_tensor_data, nullptr});

  input_tensor->SetOriginFormat(ge::FORMAT_ND);
  input_tensor->SetStorageFormat(ge::FORMAT_ND);
  input_tensor->SetPlacement(gert::kOnHost);
  input_tensor->MutableOriginShape() = {};
  input_tensor->MutableStorageShape() = {};

  auto kernelHolder = gert::KernelRunContextFaker()
                          .KernelIONum(2, 1)
                          .Inputs({reinterpret_cast<void*>(input_tensor), &single_stream_l2_allocator})
                          .Build();
  auto context = kernelHolder.GetContext<KernelContext>();
  auto kernel_funcs = registry.FindKernelFuncs("BuildAddrFromTensorCompute");
  ASSERT_EQ(kernel_funcs->outputs_creator(nullptr, context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(kernel_funcs->run_func(context), ge::GRAPH_SUCCESS);
}

}  // namespace gert

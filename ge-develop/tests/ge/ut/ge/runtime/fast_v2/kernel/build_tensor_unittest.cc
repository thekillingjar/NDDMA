/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel/common_kernel_impl/build_tensor.h"
#include <gtest/gtest.h>
#include "faker/kernel_run_context_facker.h"
#include "faker/kernel_outputs_faker.h"
#include "faker/node_faker.h"
#include "register/kernel_registry.h"
#include "kernel/common_kernel_impl/sink_node_bin.h"
#include "kernel/tensor_attr.h"
#include "faker/fake_value.h"
#include "stub/gert_runtime_stub.h"
#include "common/ge_inner_error_codes.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "kernel/memory/host_mem_allocator.h"
#include "kernel/memory/caching_mem_allocator.h"

namespace gert {
struct BuildTensorUT : public testing::Test {
  KernelRegistry &registry = KernelRegistry::GetInstance();
  memory::HostMemAllocator host_mem_allocator;
  memory::HostGertMemAllocator gert_allocator{&host_mem_allocator};
  int64_t logic_stream_id = 0;
};

TEST_F(BuildTensorUT, test_build_tensor_storage_success) {
  kernel::BuildTensorAttr attr{kOnDeviceHbm, ge::DT_FLOAT16};
  auto block = gert_allocator.Malloc(1024U);
  GertTensorData tensor_data = {1024U, kOnHost, 0, block};
  StorageShape storage_shape{{4}, {4}};

  auto run_context = BuildKernelRunContext(3, 1);
  run_context.value_holder[0].Set(&storage_shape, nullptr);
  run_context.value_holder[1].Set(&tensor_data, nullptr);
  run_context.value_holder[2].Set(&attr, nullptr);
  auto funcs = registry.FindKernelFuncs("BuildTensor");
  ASSERT_NE(funcs, nullptr);
  ASSERT_EQ(funcs->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(kernel::BuildTensor(run_context), ge::GRAPH_SUCCESS);
  run_context.FreeValue(3);
}

TEST_F(BuildTensorUT, test_build_ref_tensor_storage_success) {
  kernel::BuildTensorAttr attr{kOnDeviceHbm, ge::DT_FLOAT16};
  auto block = gert_allocator.Malloc(1024U);
  GertTensorData tensor_data = {1024U, kOnHost, 0, block};
  StorageShape storage_shape{{4}, {4}};

  auto run_context = BuildKernelRunContext(3, 1);
  run_context.value_holder[0].Set(&storage_shape, nullptr);
  run_context.value_holder[1].Set(&tensor_data, nullptr);
  run_context.value_holder[2].Set(&attr, nullptr);
  auto funcs = registry.FindKernelFuncs("BuildRefTensor");
  ASSERT_NE(funcs, nullptr);
  ASSERT_EQ(funcs->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(kernel::BuildRefTensor(run_context), ge::GRAPH_SUCCESS);
  auto ref_tensor = run_context.GetContext<KernelContext>()->GetOutputPointer<Tensor>(0U);
  auto addr = block->GetAddr();
  ASSERT_EQ(ref_tensor->MutableTensorData().Free(), ge::GRAPH_SUCCESS);
  ASSERT_EQ(ref_tensor->GetAddr(), addr);
  run_context.FreeValue(3);
}

TEST_F(BuildTensorUT, BuildTensor_NullInput) {
  auto run_context = BuildKernelRunContext(3, 1);
  auto funcs = registry.FindKernelFuncs("BuildTensorPureShape");
  ASSERT_NE(funcs, nullptr);
  ASSERT_EQ(funcs->run_func(run_context), ge::GRAPH_PARAM_INVALID);
}

TEST_F(BuildTensorUT, SplitTensor_Following) {
  size_t total_size;
  auto tensor_holder = Tensor::CreateFollowing(10 * 20, ge::DT_FLOAT, total_size);
  auto tensor = reinterpret_cast<Tensor *>(tensor_holder.get());
  tensor->MutableOriginShape() = {10, 20};
  tensor->MutableStorageShape() = {10, 20};
  tensor->MutableFormat() = {ge::FORMAT_ND, ge::FORMAT_ND, {}};
  memset(tensor->GetAddr(), 1, 10 * 20 * sizeof(float));
  auto context_holder = KernelRunContextFaker()
                            .KernelIONum(2, static_cast<size_t>(kernel::SplitTensorOutputs::kNum))
                            .NodeIoNum(1, 1)
                            .Inputs({tensor, &logic_stream_id})
                            .Build();
  auto run_context = context_holder.GetContext<KernelContext>();
  auto funcs = registry.FindKernelFuncs(kernel::kSplitConstTensor);
  ASSERT_NE(funcs, nullptr);
  ASSERT_EQ(funcs->outputs_creator(FastNodeFaker().Build(), run_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(funcs->run_func(run_context), ge::GRAPH_SUCCESS);

  auto shape = run_context->GetOutputPointer<StorageShape>(static_cast<size_t>(kernel::SplitTensorOutputs::kShape));
  auto output_tensor = run_context->GetOutputPointer<Tensor>(static_cast<size_t>(kernel::SplitTensorOutputs::kShape));
  auto tensor_data =
      run_context->GetOutputPointer<GertTensorData>(static_cast<size_t>(kernel::SplitTensorOutputs::kTensorData));
  ASSERT_NE(shape, nullptr);
  ASSERT_NE(output_tensor, nullptr);
  ASSERT_NE(tensor_data, nullptr);

  EXPECT_EQ(shape->GetOriginShape(), Shape({10, 20}));
  EXPECT_EQ(shape->GetStorageShape(), Shape({10, 20}));
  EXPECT_EQ(output_tensor->GetOriginShape(), Shape({10, 20}));
  EXPECT_EQ(output_tensor->GetStorageShape(), Shape({10, 20}));
  EXPECT_EQ(output_tensor->GetAddr(), nullptr);
  EXPECT_EQ(tensor_data->GetAddr(), tensor->GetAddr());

  context_holder.FreeAll();
}

TEST_F(BuildTensorUT, SplitTensor_Host) {
  auto tensor_holder = TensorFaker().Shape({10, 20}).Format(ge::FORMAT_ND).Placement(kOnHost).Build();
  auto context_holder = KernelRunContextFaker()
                            .KernelIONum(2, static_cast<size_t>(kernel::SplitTensorOutputs::kNum))
                            .NodeIoNum(1, 1)
                            .Inputs({tensor_holder.GetTensor(), &gert_allocator})
                            .Build();

  auto run_context = context_holder.GetContext<KernelContext>();
  auto funcs = registry.FindKernelFuncs(kernel::kSplitDataTensor);
  ASSERT_NE(funcs, nullptr);
  ASSERT_EQ(funcs->outputs_creator(FastNodeFaker().Build(), run_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(funcs->run_func(run_context), ge::GRAPH_SUCCESS);

  // check tensor data
  auto tensor_data_chain = run_context->GetOutput(static_cast<size_t>(kernel::SplitTensorOutputs::kTensorData));
  ASSERT_NE(tensor_data_chain, nullptr);
  EXPECT_TRUE(tensor_data_chain->HasDeleter());
  auto tensor_data = tensor_data_chain->GetPointer<GertTensorData>();
  ASSERT_NE(tensor_data, nullptr);
  EXPECT_EQ(tensor_data->GetAddr(), tensor_holder.GetTensor()->GetAddr());

  // check shape
  auto shape = run_context->GetOutputPointer<StorageShape>(static_cast<size_t>(kernel::SplitTensorOutputs::kShape));
  ASSERT_NE(shape, nullptr);
  EXPECT_EQ(*shape, tensor_holder.GetTensor()->GetShape());

  context_holder.FreeAll();
}

TEST_F(BuildTensorUT, BuildShapeTensorData) {
  StorageShape shape = {{10, 20, 30}, {10, 20, 30}};
  auto context_holder = KernelRunContextFaker()
                            .KernelIONum(1U, 1U)
                            .Inputs({&shape})
                            .NodeAttrs({{"dtype", ge::AnyValue::CreateFrom<int64_t>(ge::DataType::DT_INT64)}})
                            .Build();
  auto context = context_holder.GetContext<KernelContext>();

  auto funcs = registry.FindKernelFuncs(kernel::kBuildShapeTensorData);
  ASSERT_NE(funcs, nullptr);
  ASSERT_EQ(funcs->outputs_creator(nullptr, context), ge::GRAPH_SUCCESS);
  EXPECT_EQ(funcs->run_func(context), ge::GRAPH_SUCCESS);
  auto td = context->GetOutputPointer<GertTensorData>(static_cast<size_t>(0U));
  ASSERT_NE(td, nullptr);
  EXPECT_EQ(td->GetPlacement(), kOnHost);
  EXPECT_EQ(td->GetSize(), Shape::kMaxDimNum * sizeof(int64_t));
  auto shape_data = static_cast<const int64_t *>(td->GetAddr());
  ASSERT_EQ(shape_data, &(shape.GetOriginShape()[0]));
  EXPECT_EQ(shape_data[0], 10);
  EXPECT_EQ(shape_data[1], 20);
  EXPECT_EQ(shape_data[2], 30);

  context_holder.FreeAll();
}

TEST_F(BuildTensorUT, BuildShapeTensorDataMallocOutputSuccess) {
  StorageShape shape = {{10, 20, 30}, {10, 20, 30}};
  auto context_holder = KernelRunContextFaker()
                            .KernelIONum(1U, 1U)
                            .Inputs({&shape})
                            .NodeAttrs({{"dtype", ge::AnyValue::CreateFrom<int64_t>(ge::DataType::DT_INT32)}})
                            .Build();
  auto context = context_holder.GetContext<KernelContext>();

  auto funcs = registry.FindKernelFuncs(kernel::kBuildShapeTensorData);
  ASSERT_NE(funcs, nullptr);
  ASSERT_EQ(funcs->outputs_creator(nullptr, context), ge::GRAPH_SUCCESS);
  EXPECT_EQ(funcs->run_func(context), ge::GRAPH_SUCCESS);
  ASSERT_NE(funcs->trace_printer, nullptr);
  EXPECT_FALSE(funcs->trace_printer(context).empty());

  auto tensor_data = context->GetOutputPointer<GertTensorData>(0U);
  ASSERT_NE(tensor_data, nullptr);
  EXPECT_EQ(tensor_data->GetPlacement(), kOnHost);
  EXPECT_EQ(tensor_data->GetSize(), Shape::kMaxDimNum * sizeof(int32_t));

  auto shape_data = ge::PtrToPtr<void, int32_t>(tensor_data->GetAddr());
  ASSERT_NE(shape_data, nullptr);
  ASSERT_NE(reinterpret_cast<const long int *>(shape_data), &(shape.GetOriginShape()[0]));
  EXPECT_EQ(shape_data[0], 10);
  EXPECT_EQ(shape_data[1], 20);
  EXPECT_EQ(shape_data[2], 30);

  context_holder.FreeAll();
}

TEST_F(BuildTensorUT, BuildShapeTensorDataForShapeNMallocOutputSuccess) {
  StorageShape shape0 = {{10, 20, 30}, {10, 20, 30}};
  StorageShape shape1 = {{1, -1, 512}, {1, -1, 512}};
  auto context_holder = KernelRunContextFaker()
                            .KernelIONum(2U, 2U)
                            .Inputs({&shape0, &shape1})
                            .NodeAttrs({{"dtype", ge::AnyValue::CreateFrom<int64_t>(ge::DataType::DT_INT32)}})
                            .Build();
  auto context = context_holder.GetContext<KernelContext>();

  auto funcs = registry.FindKernelFuncs(kernel::kBuildShapeTensorData);
  ASSERT_NE(funcs, nullptr);
  ASSERT_EQ(funcs->outputs_creator(nullptr, context), ge::GRAPH_SUCCESS);
  EXPECT_EQ(funcs->run_func(context), ge::GRAPH_SUCCESS);

  auto tensor_data0 = context->GetOutputPointer<GertTensorData>(0);
  auto shape_data0 = ge::PtrToPtr<void, int32_t>(tensor_data0->GetAddr());
  ASSERT_NE(shape_data0, nullptr);
  ASSERT_NE(reinterpret_cast<const long int *>(shape_data0), &(shape0.GetOriginShape()[0]));
  EXPECT_EQ(shape_data0[0], 10);
  EXPECT_EQ(shape_data0[1], 20);
  EXPECT_EQ(shape_data0[2], 30);

  auto tensor_data1 = context->GetOutputPointer<GertTensorData>(1);
  auto shape_data1 = ge::PtrToPtr<void, int32_t>(tensor_data1->GetAddr());
  ASSERT_NE(shape_data1, nullptr);
  ASSERT_NE(reinterpret_cast<const long int *>(shape_data1), &(shape1.GetOriginShape()[0]));
  EXPECT_EQ(shape_data1[0], 1);
  EXPECT_EQ(shape_data1[1], -1);
  EXPECT_EQ(shape_data1[2], 512);

  context_holder.FreeAll();
}

TEST_F(BuildTensorUT, BuildShapeTensorDataForShapeNSuccess) {
  StorageShape shape0 = {{10, 20, 30}, {10, 20, 30}};
  StorageShape shape1 = {{1, -1, 512}, {1, -1, 512}};
  auto context_holder = KernelRunContextFaker()
                            .KernelIONum(2U, 2U)
                            .Inputs({&shape0, &shape1})
                            .NodeAttrs({{"dtype", ge::AnyValue::CreateFrom<int64_t>(ge::DataType::DT_INT64)}})
                            .Build();
  auto context = context_holder.GetContext<KernelContext>();

  auto funcs = registry.FindKernelFuncs(kernel::kBuildShapeTensorData);
  ASSERT_NE(funcs, nullptr);
  ASSERT_EQ(funcs->outputs_creator(nullptr, context), ge::GRAPH_SUCCESS);
  EXPECT_EQ(funcs->run_func(context), ge::GRAPH_SUCCESS);

  auto td0 = context->GetOutputPointer<GertTensorData>(0U);
  ASSERT_NE(td0, nullptr);
  auto shape_data0 = static_cast<const int64_t *>(td0->GetAddr());
  ASSERT_EQ(shape_data0, &(shape0.GetOriginShape()[0]));
  EXPECT_EQ(shape_data0[0], 10);
  EXPECT_EQ(shape_data0[1], 20);
  EXPECT_EQ(shape_data0[2], 30);

  auto td1 = context->GetOutputPointer<GertTensorData>(1U);
  ASSERT_NE(td1, nullptr);
  auto shape_data1 = static_cast<const int64_t *>(td1->GetAddr());
  ASSERT_EQ(shape_data1, &(shape1.GetOriginShape()[0]));
  EXPECT_EQ(shape_data1[0], 1);
  EXPECT_EQ(shape_data1[1], -1);
  EXPECT_EQ(shape_data1[2], 512);

  context_holder.FreeAll();
}

TEST_F(BuildTensorUT, BuildShapeTensorDataInvalidDataTypeFailed) {
  StorageShape shape = {{10, 20, 30}, {10, 20, 30}};
  auto context_holder = KernelRunContextFaker()
                            .KernelIONum(1U, 1U)
                            .Inputs({&shape})
                            .NodeAttrs({{"dtype", ge::AnyValue::CreateFrom<int64_t>(0)}})
                            .Build();
  auto context = context_holder.GetContext<KernelContext>();

  auto funcs = registry.FindKernelFuncs(kernel::kBuildShapeTensorData);
  ASSERT_NE(funcs, nullptr);
  ASSERT_NE(funcs->outputs_creator(nullptr, context), ge::GRAPH_SUCCESS);
  context_holder.FreeAll();
}

TEST_F(BuildTensorUT, BuildShapeTensorDataDtypeAttrLostFailed) {
  StorageShape shape = {{10, 20, 30}, {10, 20, 30}};
  auto context_holder = KernelRunContextFaker().KernelIONum(1U, 1U).NodeIoNum(1, 1).Inputs({&shape}).Build();
  auto context = context_holder.GetContext<KernelContext>();

  auto funcs = registry.FindKernelFuncs(kernel::kBuildShapeTensorData);
  ASSERT_NE(funcs, nullptr);
  ASSERT_NE(funcs->outputs_creator(nullptr, context), ge::GRAPH_SUCCESS);
  context_holder.FreeAll();
}
TEST_F(BuildTensorUT, SplitTensorForOutputData_SplitOk_InputCorrect) {
  auto tensor_holder = TensorFaker().DataType(ge::DT_FLOAT).Shape({1, 2, 3, 4}).Build();
  auto context_holder =
      KernelRunContextFaker().KernelIONum(2U, 3U).NodeIoNum(1, 1).Inputs({tensor_holder.GetTensor(), &gert_allocator}).Build();
  auto context = context_holder.GetContext<KernelContext>();

  auto funcs = registry.FindKernelFuncs(kernel::kSplitTensorForOutputData);
  ASSERT_NE(funcs, nullptr);
  ASSERT_NE(funcs->outputs_creator, nullptr);
  ASSERT_EQ(funcs->outputs_creator(nullptr, context), ge::GRAPH_SUCCESS);
  ASSERT_NE(context->GetOutputPointer<Shape>(0), nullptr);
  ASSERT_NE(context->GetOutputPointer<GertTensorData>(1), nullptr);

  ASSERT_NE(funcs->run_func, nullptr);
  ASSERT_EQ(funcs->run_func(context), ge::GRAPH_SUCCESS);
  EXPECT_EQ(context->GetOutputPointer<StorageShape>(0)->GetOriginShape(), Shape({1, 2, 3, 4}));
  EXPECT_EQ(context->GetOutputPointer<StorageShape>(0)->GetStorageShape(), Shape({1, 2, 3, 4}));
  EXPECT_EQ(context->GetOutputPointer<GertTensorData>(1)->GetAddr(), tensor_holder.GetTensor()->GetAddr());
  EXPECT_EQ(context->GetOutputPointer<GertTensorData>(1)->GetSize(), tensor_holder.GetTensor()->GetSize());
  EXPECT_EQ(context->GetOutputPointer<GertTensorData>(1)->GetPlacement(), tensor_holder.GetTensor()->GetPlacement());
}
TEST_F(BuildTensorUT, SplitTensorForOutputData_CheckFailed_NullTensor) {
  auto context_holder = KernelRunContextFaker().KernelIONum(2U, 3U).NodeIoNum(1, 1).Inputs({nullptr, &gert_allocator}).Build();
  auto context = context_holder.GetContext<KernelContext>();

  auto funcs = registry.FindKernelFuncs(kernel::kSplitTensorForOutputData);
  ASSERT_NE(funcs, nullptr);
  ASSERT_NE(funcs->outputs_creator, nullptr);
  ASSERT_EQ(funcs->outputs_creator(nullptr, context), ge::GRAPH_SUCCESS);
  ASSERT_NE(context->GetOutputPointer<Shape>(0), nullptr);
  ASSERT_NE(context->GetOutputPointer<GertTensorData>(1), nullptr);

  ASSERT_NE(funcs->run_func, nullptr);

  GertRuntimeStub stub;
  ASSERT_NE(funcs->run_func(context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(stub.GetSlogStub().FindErrorLogEndsWith(
                "In the `always_zero_copy` mode, the output tensor and tensor data must be allocated by the called"),
            0);
}
TEST_F(BuildTensorUT, SplitTensorForOutputData_CheckFailed_NullTensorData) {
  Tensor tensor;
  auto context_holder = KernelRunContextFaker().KernelIONum(2U, 3U).NodeIoNum(1, 1).Inputs({&tensor, &gert_allocator}).Build();
  auto context = context_holder.GetContext<KernelContext>();

  auto funcs = registry.FindKernelFuncs(kernel::kSplitTensorForOutputData);
  ASSERT_NE(funcs, nullptr);
  ASSERT_NE(funcs->outputs_creator, nullptr);
  ASSERT_EQ(funcs->outputs_creator(nullptr, context), ge::GRAPH_SUCCESS);
  ASSERT_NE(context->GetOutputPointer<Shape>(0), nullptr);
  ASSERT_NE(context->GetOutputPointer<GertTensorData>(1), nullptr);

  ASSERT_NE(funcs->run_func, nullptr);

  GertRuntimeStub stub;
  ASSERT_EQ(funcs->run_func(context), ge::PARAM_INVALID);
  ASSERT_EQ(stub.GetSlogStub().FindErrorLogEndsWith(
                "In the `always_zero_copy` mode, the output tensor and tensor data must be allocated by the called"),
            0);
}

TEST_F(BuildTensorUT, SplitTensorForOutputData_CheckFailed_WhenTensorDataNotEnoughSize) {
  auto tensor_holder = TensorFaker().DataType(ge::DT_FLOAT).Shape({1, 2, 3, 4}).Build();
  tensor_holder.GetTensor()->MutableStorageShape().SetDim(0, 2);
  auto context_holder =
      KernelRunContextFaker().KernelIONum(2U, 3U).NodeIoNum(1, 1).Inputs({tensor_holder.GetTensor(), &gert_allocator}).Build();
  auto context = context_holder.GetContext<KernelContext>();

  auto funcs = registry.FindKernelFuncs(kernel::kSplitTensorForOutputData);
  ASSERT_NE(funcs, nullptr);
  ASSERT_NE(funcs->outputs_creator, nullptr);
  ASSERT_EQ(funcs->outputs_creator(nullptr, context), ge::GRAPH_SUCCESS);
  ASSERT_NE(context->GetOutputPointer<Shape>(0), nullptr);
  ASSERT_NE(context->GetOutputPointer<GertTensorData>(1), nullptr);

  GertRuntimeStub runtime_stub;
  ASSERT_NE(funcs->run_func, nullptr);
  ASSERT_NE(funcs->run_func(context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(runtime_stub.GetSlogStub().GetLogs(DLOG_ERROR).size(), 1);
  EXPECT_TRUE(runtime_stub.GetSlogStub()
                  .GetLogs(DLOG_ERROR)
                  .at(0)
                  .EndsWith("The output tensor memory size[128] is smaller than the actual size[192] required by tensor."));
}
// todo 确定输出size校验规则后重新放开
#if 0
TEST_F(BuildTensorUT, SplitTensorForOutputData_CheckFailed_WhenTensorDataNotEnoughSize) {
  auto tensor_holder = TensorFaker().DataType(ge::DT_FLOAT).Shape({1, 2, 3, 4}).Build();
  tensor_holder.GetTensor()->MutableStorageShape().SetDim(0, 2);

  auto context_holder = KernelRunContextFaker().KernelIONum(1U, 2U).Inputs({tensor_holder.GetTensor()}).Build();
  auto context = context_holder.GetContext<KernelContext>();

  auto funcs = registry.FindKernelFuncs(kernel::kSplitTensorForOutputData);
  ASSERT_NE(funcs, nullptr);
  ASSERT_NE(funcs->outputs_creator, nullptr);
  ASSERT_EQ(funcs->outputs_creator(nullptr, context), ge::GRAPH_SUCCESS);
  ASSERT_NE(context->GetOutputPointer<Shape>(0), nullptr);
  ASSERT_NE(context->GetOutputPointer<TensorData>(1), nullptr);

  GertRuntimeStub runtime_stub;
  ASSERT_NE(funcs->run_func, nullptr);
  ASSERT_NE(funcs->run_func(context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(runtime_stub.GetSlogStub().GetLogs(DLOG_ERROR).size(), 1);
  EXPECT_TRUE(runtime_stub.GetSlogStub()
                  .GetLogs(DLOG_ERROR)
                  .at(0)
                  .EndsWith("Invalid output tensor size 128, expect at least 224"));
}
TEST_F(BuildTensorUT, SplitTensorForOutputDataWithExpectSize_CheckFailed_WhenTensorDataNotEnoughSize) {
  auto tensor_holder = TensorFaker().DataType(ge::DT_FLOAT).Shape({1, 2, 3, 4}).Build();
  tensor_holder.GetTensor()->MutableStorageShape().SetDim(0, 2);

  auto context_holder =
      KernelRunContextFaker().KernelIONum(2U, 2U).Inputs({tensor_holder.GetTensor(), (void *)512}).Build();
  auto context = context_holder.GetContext<KernelContext>();

  auto funcs = registry.FindKernelFuncs(kernel::kSplitTensorForOutputData);
  ASSERT_NE(funcs, nullptr);
  ASSERT_NE(funcs->outputs_creator, nullptr);
  ASSERT_EQ(funcs->outputs_creator(nullptr, context), ge::GRAPH_SUCCESS);
  ASSERT_NE(context->GetOutputPointer<Shape>(0), nullptr);
  ASSERT_NE(context->GetOutputPointer<TensorData>(1), nullptr);

  GertRuntimeStub runtime_stub;
  ASSERT_NE(funcs->run_func, nullptr);
  ASSERT_NE(funcs->run_func(context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(runtime_stub.GetSlogStub().FindErrorLogEndsWith("Invalid output tensor size 128, expect at least 512"), 0);
}
TEST_F(BuildTensorUT, SplitTensorForOutputData_CheckFailed_WhenTensorDataTypeInvalid) {
  auto tensor_holder = TensorFaker().DataType(ge::DT_FLOAT).Shape({1, 2, 3, 4}).Build();

  auto context_holder = KernelRunContextFaker().KernelIONum(1U, 2U).Inputs({tensor_holder.GetTensor()}).Build();
  auto context = context_holder.GetContext<KernelContext>();

  auto funcs = registry.FindKernelFuncs(kernel::kSplitTensorForOutputData);
  ASSERT_NE(funcs, nullptr);
  ASSERT_NE(funcs->outputs_creator, nullptr);
  ASSERT_EQ(funcs->outputs_creator(nullptr, context), ge::GRAPH_SUCCESS);
  ASSERT_NE(context->GetOutputPointer<Shape>(0), nullptr);
  ASSERT_NE(context->GetOutputPointer<TensorData>(1), nullptr);

  GertRuntimeStub runtime_stub;
  ASSERT_NE(funcs->run_func, nullptr);
  // invalid data type
  tensor_holder.GetTensor()->SetDataType(ge::DT_STRING);
  ASSERT_NE(funcs->run_func(context), ge::GRAPH_SUCCESS);
  ASSERT_GE(runtime_stub.GetSlogStub().GetLogs(DLOG_ERROR).size(), 2);
  EXPECT_TRUE(runtime_stub.GetSlogStub().GetLogs(DLOG_ERROR).at(1).EndsWith("[Calc][TensorSizeByShape] shape_size[24] data_type[DT_STRING] failed"));
  // shape(overflow)
  runtime_stub.GetSlogStub().Clear();
  tensor_holder.GetTensor()->SetDataType(ge::DT_FLOAT);
  tensor_holder.GetTensor()->MutableStorageShape().SetDim(0, std::numeric_limits<int64_t>::max() - 10);
  ASSERT_NE(funcs->run_func(context), ge::GRAPH_SUCCESS);
  ASSERT_GE(runtime_stub.GetSlogStub().GetLogs(DLOG_ERROR).size(), 2);
  EXPECT_TRUE(runtime_stub.GetSlogStub()
                  .GetLogs(DLOG_ERROR)
                  .at(1)
                  .EndsWith("Invalid shape 9223372036854775797,2,3,4 or data type 0"));
}
TEST_F(BuildTensorUT, SplitTensorForOutputData_CheckFailed_WhenRoundUpOverflow) {
  auto tensor_holder = TensorFaker().DataType(ge::DT_FLOAT).Shape({1}).Build();

  auto context_holder = KernelRunContextFaker().KernelIONum(1U, 2U).Inputs({tensor_holder.GetTensor()}).Build();
  auto context = context_holder.GetContext<KernelContext>();

  auto funcs = registry.FindKernelFuncs(kernel::kSplitTensorForOutputData);
  ASSERT_NE(funcs, nullptr);
  ASSERT_NE(funcs->outputs_creator, nullptr);
  ASSERT_EQ(funcs->outputs_creator(nullptr, context), ge::GRAPH_SUCCESS);
  ASSERT_NE(context->GetOutputPointer<Shape>(0), nullptr);
  ASSERT_NE(context->GetOutputPointer<TensorData>(1), nullptr);

  ASSERT_NE(funcs->run_func, nullptr);

  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().Clear();
  tensor_holder.GetTensor()->MutableStorageShape().SetDim(0, std::numeric_limits<int64_t>::max() - 10);
  ASSERT_NE(funcs->run_func(context), ge::GRAPH_SUCCESS);
  EXPECT_TRUE(runtime_stub.GetSlogStub()
                  .GetLogs(DLOG_ERROR)
                  .at(1)
                  .EndsWith("Invalid shape 9223372036854775797, overflow after align"));

  runtime_stub.GetSlogStub().Clear();
  tensor_holder.GetTensor()->MutableStorageShape().SetDim(0, std::numeric_limits<int64_t>::max() - 32);
  ASSERT_NE(funcs->run_func(context), ge::GRAPH_SUCCESS);
  EXPECT_TRUE(runtime_stub.GetSlogStub()
                  .GetLogs(DLOG_ERROR)
                  .at(1)
                  .EndsWith("Invalid shape 9223372036854775775, overflow after align"));
}
#endif
}  // namespace gert

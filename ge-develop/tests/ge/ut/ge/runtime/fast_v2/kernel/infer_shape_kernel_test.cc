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

#include "base/registry/op_impl_space_registry_v2.h"
#include "exe_graph/runtime/storage_shape.h"
#include "faker/kernel_run_context_facker.h"
#include "register/kernel_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "expand_dimension.h"
#include "faker/space_registry_faker.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "runtime/v2/engine/aicore/kernel/set_output_shape_kernel.h"

namespace gert {
namespace kernel {
ge::graphStatus InferShape(KernelContext *context);
ge::graphStatus SetOutputShape(KernelContext *context);
}
struct InferShapeKernelTest : public testing::Test {
  KernelRegistry &registry = KernelRegistry::GetInstance();
};

ge::graphStatus CopyInferShape(InferShapeContext* context){
  auto input = context->GetInputShape(0);
  auto output = context->GetOutputShape(0);
  *output = *input;
  return ge::GRAPH_SUCCESS;
}

TEST_F(InferShapeKernelTest, test_infershape_without_padding) {
  gert::Tensor input_shape_tensor = {{{1,2,2,256}, {1,2,2,256}},    // shape
                                     {ge::FORMAT_ND, ge::FORMAT_NCHW, {}},  // format
                                     kOnDeviceHbm,                                // placement
                                     ge::DT_FLOAT16,                              // data type
                                     (void *)0x0};
  Tensor output;
  auto run_context = InferShapeContextFaker().NodeIoNum(1,1)
                     .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_NCHW)
                     .InputShapes({&input_shape_tensor}).OutputShapes({&output}).Build();
  run_context.value_holder[0].Set(&input_shape_tensor, nullptr);
  run_context.value_holder[1].Set((void*)CopyInferShape, nullptr);
  run_context.value_holder[2].Set(&output, nullptr);

  ASSERT_EQ(kernel::InferShape(run_context), ge::GRAPH_SUCCESS);

  ASSERT_EQ(output.GetStorageShape().GetDimNum(), 4);
  ASSERT_EQ(output.GetOriginShape().GetDimNum(), 4);
  ASSERT_EQ(output.GetAddr(), nullptr);
  run_context.FreeAll();
}

// origin shape 为 {8, 512, 5, 5}
// 经过transshape(FORMAT_NCHW -> FORMAT_NC1HWC0)后为 {8, 32, 5, 5, 16}
TEST_F(InferShapeKernelTest, test_infershape_with_transshape_success) {
  // RunTransferShape(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, DT_INT32, true, {8, 512, 5, 5}, {8, 32, 5, 5, 16});
  StorageShape input{{8, 512, 5, 5}, {8, 512, 5, 5}};
  StorageShape output;

  auto run_context = InferShapeContextFaker().NodeIoNum(1,1)
                     .InputShapes({&input}).OutputShapes({&output}).Build();
  auto &compute_info = *run_context.MutableComputeNodeInfo();
  auto output_td = const_cast<CompileTimeTensorDesc*>(compute_info.GetOutputTdInfo(0));
  output_td->SetDataType(ge::DT_INT32);
  output_td->SetStorageFormat(ge::FORMAT_NC1HWC0);
  output_td->SetOriginFormat(ge::FORMAT_NCHW);

  run_context.value_holder[0].Set(&input, nullptr);
  run_context.value_holder[1].Set((void*)CopyInferShape, nullptr);
  run_context.value_holder[2].Set(&output, nullptr);

  ASSERT_EQ(registry.FindKernelFuncs("InferShape")->run_func(run_context), ge::GRAPH_SUCCESS);

  ASSERT_EQ(output.GetOriginShape().GetDimNum(), 4);
  ASSERT_EQ(output.GetOriginShape(), Shape({8, 512, 5, 5}));
  ASSERT_EQ(output.GetStorageShape().GetDimNum(), 5);
  ASSERT_EQ(output.GetStorageShape(), Shape({8, 32, 5, 5, 16}));

  auto ret = registry.FindKernelFuncs("InferShape")->trace_printer(run_context);
  EXPECT_FALSE(ret.empty());
}

// origin shape 为 {8, 512, 5}
// 在最后一位补1, 经过expand dims后为{8, 512, 5, 1}
// 经过transshape(FORMAT_NCHW -> FORMAT_NC1HWC0)后为 {8, 32, 5, 1, 16}
TEST_F(InferShapeKernelTest, test_infershape_with_padding_and_transshape) {
  vector<int64_t> origin_shape = {8, 512, 5};
  StorageShape input{{8, 512, 5}, {8, 512, 5}};
  StorageShape output;

  auto run_context = InferShapeContextFaker().NodeIoNum(1,1).InputShapes({&input}).OutputShapes({&output}).Build();
  auto &compute_info = *run_context.MutableComputeNodeInfo();
  auto output_td = const_cast<CompileTimeTensorDesc*>(compute_info.GetOutputTdInfo(0));
  output_td->SetDataType(ge::DT_INT32);
  // get int_reshape_type
  // get reshape type 此处模拟FE调用transformer中方法获取int类型的reshape type
  int64_t int_reshape_type = transformer::ExpandDimension::GenerateReshapeType(ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, origin_shape.size(),
                                                                    "NCH");
  output_td->SetExpandDimsType(ExpandDimsType(int_reshape_type));
  output_td->SetOriginFormat(ge::FORMAT_NCHW);
  output_td->SetStorageFormat(ge::FORMAT_NC1HWC0);
  run_context.value_holder[0].Set(&input, nullptr);
  run_context.value_holder[1].Set((void*)CopyInferShape, nullptr);
  run_context.value_holder[2].Set(&output, nullptr);

  ASSERT_EQ(registry.FindKernelFuncs("InferShape")->run_func(run_context), ge::GRAPH_SUCCESS);

  ASSERT_EQ(output.GetOriginShape().GetDimNum(), 3);
  ASSERT_EQ(output.GetOriginShape(), Shape({8, 512, 5}));
  ASSERT_EQ(output.GetStorageShape().GetDimNum(), 5);
  ASSERT_EQ(output.GetStorageShape(), Shape({8, 32, 5, 1, 16}));
}

// invalid format, will cause transfershape fail
TEST_F(InferShapeKernelTest, test_infershape_with_transshape_failed) {
  StorageShape input{{8, 512, 5, 5}, {8, 512, 5, 5}};
  StorageShape output;

  auto run_context = InferShapeContextFaker().NodeIoNum(1,1).InputShapes({&input}).OutputShapes({&output}).Build();
  auto &compute_info = *run_context.MutableComputeNodeInfo();
  auto output_td = const_cast<CompileTimeTensorDesc*>(compute_info.GetOutputTdInfo(0));
  output_td->SetDataType(ge::DT_INT32);
  output_td->SetStorageFormat(ge::FORMAT_RESERVED);// invalid format, will cause transfershape fail
  run_context.value_holder[0].Set(&input, nullptr);
  run_context.value_holder[1].Set((void*)CopyInferShape, nullptr);
  run_context.value_holder[2].Set(&output, nullptr);

  ASSERT_EQ(registry.FindKernelFuncs("InferShape")->run_func(run_context), ge::GRAPH_FAILED);
}

TEST_F(InferShapeKernelTest, transfershape_with_the_same_origin_format_and_storage_format) {
  StorageShape input{{8, 512, 5, 5, 4}, {8, 512, 5, 5, 4}};
  StorageShape output;

  auto run_context = InferShapeContextFaker().NodeIoNum(1,1).InputShapes({&input}).OutputShapes({&output}).Build();
  auto &compute_info = *run_context.MutableComputeNodeInfo();
  auto output_td = const_cast<CompileTimeTensorDesc*>(compute_info.GetOutputTdInfo(0));
  output_td->SetDataType(ge::DT_INT32);
  output_td->SetStorageFormat(ge::FORMAT_NHWC);
  output_td->SetOriginFormat(ge::FORMAT_NHWC);
  run_context.value_holder[0].Set(&input, nullptr);
  run_context.value_holder[1].Set((void*)CopyInferShape, nullptr);
  run_context.value_holder[2].Set(&output, nullptr);

  ASSERT_EQ(registry.FindKernelFuncs("InferShape")->run_func(run_context), ge::GRAPH_SUCCESS);
}

UINT32 StubInfershape(InferShapeContext *){
  return ge::GRAPH_SUCCESS;
}

TEST_F(InferShapeKernelTest, test_find_InferShapeFunc_success) {
  auto run_context = BuildKernelRunContext(2, 1);
  const char *node_type = "stub_infershape";
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->CreateOrGetOpImpl(node_type)->infer_shape
      = StubInfershape;

  auto space_registry = std::make_shared<gert::OpImplSpaceRegistryV2>();
  ASSERT_NE(space_registry, nullptr);
  space_registry->CreateOrGetOpImpl(node_type)->infer_shape = StubInfershape;

  run_context.value_holder[0].Set(const_cast<char *>(node_type), nullptr);
  run_context.value_holder[1].Set(space_registry.get(), nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("FindInferShapeFunc")->run_func(run_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(run_context.value_holder[2].GetValue<void *>(), reinterpret_cast<void*>(StubInfershape));
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->CreateOrGetOpImpl(node_type)->infer_shape
      = nullptr;
}

TEST_F(InferShapeKernelTest, test_find_InferShapeFunc_input_empty) {
  auto run_context = BuildKernelRunContext(2, 1);
  auto space_registry = SpaceRegistryFaker().Build();
  ASSERT_NE(space_registry, nullptr);
  run_context.value_holder[1].Set(space_registry.get(), nullptr);
  ASSERT_NE(registry.FindKernelFuncs("FindInferShapeFunc")->run_func(run_context), ge::GRAPH_SUCCESS);
}

TEST_F(InferShapeKernelTest, test_find_InferShapeFunc_find_infershape_func_faild) {
  auto run_context = BuildKernelRunContext(2, 1);
  std::string node_type = "null_infershape";
  auto space_registry = SpaceRegistryFaker().Build();
  ASSERT_NE(space_registry, nullptr);
  run_context.value_holder[1].Set(space_registry.get(), nullptr);
  run_context.value_holder[0].Set(const_cast<char *>(node_type.c_str()), nullptr);
  ASSERT_NE(registry.FindKernelFuncs("FindInferShapeFunc")->run_func(run_context), ge::GRAPH_SUCCESS);
}

TEST_F(InferShapeKernelTest, test_build_infershape_output_success) {
  auto run_context = KernelRunContextFaker().KernelIONum(1, 1).NodeIoNum(1, 1).IrInputNum(1).Build();
  ASSERT_EQ(registry.FindKernelFuncs("InferShape")->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
  ASSERT_NE(run_context.value_holder[1].GetValue<void *>(), nullptr);
  run_context.FreeValue(1);
}

TEST_F(InferShapeKernelTest, test_InferShape_extend_info_is_null) {
  KernelRunContext run_context;
  run_context.compute_node_info = nullptr;
  run_context.kernel_extend_info = nullptr;
  ASSERT_NE(registry.FindKernelFuncs("InferShape")->run_func(reinterpret_cast<KernelContext*>(&run_context)), ge::GRAPH_SUCCESS);
}

UINT32 StubFailedInfershape(InferShapeContext *){
  return 0x01;
}

TEST_F(InferShapeKernelTest, test_InferShape_infer_fun_is_empty) {
  auto run_context = BuildKernelRunContext(2, 1);
  run_context.value_holder[1].Set(nullptr, nullptr);
  ASSERT_NE(registry.FindKernelFuncs("InferShape")->run_func(run_context), ge::GRAPH_SUCCESS);
}

TEST_F(InferShapeKernelTest, test_InferShape_infer_fun_not_exist) {
  auto run_context = BuildKernelRunContext(0, 1);
  ASSERT_NE(registry.FindKernelFuncs("InferShape")->run_func(run_context), ge::GRAPH_SUCCESS);
}

TEST_F(InferShapeKernelTest, test_InferShape_infer_fun_return_failed) {
  auto run_context = BuildKernelRunContext(2, 1);
  run_context.value_holder[1].Set(reinterpret_cast<void*>(StubFailedInfershape), nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("InferShape")->run_func(run_context), 0x01);
}

TEST_F(InferShapeKernelTest, test_set_output_shape_success_001) {
  uint64_t shape_data[18] = {
      2147483652, 5,  6,  7,  8,  1,  1, 1, 1,
      2147483653, 11, 12, 13, 14, 15, 1, 1, 1
  };
  size_t shapebuffer_addr_size = 2 * sizeof(uint64_t) * 26;  
  auto shape_tensor_data = std::make_unique<GertTensorData>(shape_data, shapebuffer_addr_size, kTensorPlacementEnd, -1);
  auto run_context = KernelRunContextFaker().KernelIONum(1, 2).NodeIoNum(1, 2).Build();
  StorageShape output1{{1,2,3,4}, {1,2,3,4}};
  StorageShape output2{{1,2,3,4}, {1,2,3,4}};
  run_context.value_holder[0].Set(shape_tensor_data.get(), nullptr);
  run_context.value_holder[1].Set(&output1, nullptr);
  run_context.value_holder[2].Set(&output2, nullptr);
  ASSERT_EQ(kernel::SetOutputShape(run_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(output1.GetStorageShape().GetDimNum(), 4);
  ASSERT_EQ(output1.GetOriginShape().GetDimNum(), 4);
  ASSERT_EQ(output1.GetStorageShape().GetDim(0), 5);
  ASSERT_EQ(output1.GetStorageShape().GetDim(1), 6);
  ASSERT_EQ(output1.GetStorageShape().GetDim(2), 7);
  ASSERT_EQ(output1.GetStorageShape().GetDim(3), 8);
  ASSERT_EQ(output1.GetOriginShape().GetDim(0), 5);
  ASSERT_EQ(output1.GetOriginShape().GetDim(1), 6);
  ASSERT_EQ(output1.GetOriginShape().GetDim(2), 7);
  ASSERT_EQ(output1.GetOriginShape().GetDim(3), 8);

  ASSERT_EQ(output2.GetStorageShape().GetDimNum(), 5);
  ASSERT_EQ(output2.GetOriginShape().GetDimNum(), 5);
  ASSERT_EQ(output2.GetStorageShape().GetDim(0), 11);
  ASSERT_EQ(output2.GetStorageShape().GetDim(1), 12);
  ASSERT_EQ(output2.GetStorageShape().GetDim(2), 13);
  ASSERT_EQ(output2.GetStorageShape().GetDim(3), 14);
  ASSERT_EQ(output2.GetStorageShape().GetDim(4), 15);
  ASSERT_EQ(output2.GetOriginShape().GetDim(0), 11);
  ASSERT_EQ(output2.GetOriginShape().GetDim(1), 12);
  ASSERT_EQ(output2.GetOriginShape().GetDim(2), 13);
  ASSERT_EQ(output2.GetOriginShape().GetDim(3), 14);
  ASSERT_EQ(output2.GetOriginShape().GetDim(4), 15);
  auto set_output_shape = KernelRegistry::GetInstance().FindKernelFuncs("SetOutputShape");
  auto msgs = set_output_shape->trace_printer(run_context);
  ASSERT_EQ(msgs.size(), 2U);
}

TEST_F(InferShapeKernelTest, test_set_output_shape_success_002) {
  uint32_t shape_data[36] = {
      4, 5,  6,  7,  8,  1,  1, 1, 1,
      5, 11, 12, 13, 14, 15, 1, 1, 1
  };
  size_t shapebuffer_addr_size = 2 * sizeof(uint64_t) * 9;
  auto shape_tensor_data = std::make_unique<GertTensorData>(shape_data, shapebuffer_addr_size, kTensorPlacementEnd, -1);
  auto run_context = KernelRunContextFaker().KernelIONum(1, 2).NodeIoNum(1, 2).Build();
  StorageShape output1{{1,2,3,4}, {1,2,3,4}};
  StorageShape output2{{1,2,3,4}, {1,2,3,4}};
  run_context.value_holder[0].Set(shape_tensor_data.get(), nullptr);
  run_context.value_holder[1].Set(&output1, nullptr);
  run_context.value_holder[2].Set(&output2, nullptr);
  ASSERT_EQ(kernel::SetOutputShape(run_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(output1.GetStorageShape().GetDimNum(), 4);
  ASSERT_EQ(output1.GetOriginShape().GetDimNum(), 4);
  ASSERT_EQ(output1.GetStorageShape().GetDim(0), 5);
  ASSERT_EQ(output1.GetStorageShape().GetDim(1), 6);
  ASSERT_EQ(output1.GetStorageShape().GetDim(2), 7);
  ASSERT_EQ(output1.GetStorageShape().GetDim(3), 8);
  ASSERT_EQ(output1.GetOriginShape().GetDim(0), 5);
  ASSERT_EQ(output1.GetOriginShape().GetDim(1), 6);
  ASSERT_EQ(output1.GetOriginShape().GetDim(2), 7);
  ASSERT_EQ(output1.GetOriginShape().GetDim(3), 8);

  ASSERT_EQ(output2.GetStorageShape().GetDimNum(), 5);
  ASSERT_EQ(output2.GetOriginShape().GetDimNum(), 5);
  ASSERT_EQ(output2.GetStorageShape().GetDim(0), 11);
  ASSERT_EQ(output2.GetStorageShape().GetDim(1), 12);
  ASSERT_EQ(output2.GetStorageShape().GetDim(2), 13);
  ASSERT_EQ(output2.GetStorageShape().GetDim(3), 14);
  ASSERT_EQ(output2.GetStorageShape().GetDim(4), 15);
  ASSERT_EQ(output2.GetOriginShape().GetDim(0), 11);
  ASSERT_EQ(output2.GetOriginShape().GetDim(1), 12);
  ASSERT_EQ(output2.GetOriginShape().GetDim(2), 13);
  ASSERT_EQ(output2.GetOriginShape().GetDim(3), 14);
  ASSERT_EQ(output2.GetOriginShape().GetDim(4), 15);
  auto set_output_shape = KernelRegistry::GetInstance().FindKernelFuncs("SetOutputShape");
  auto msgs = set_output_shape->trace_printer(run_context);
  ASSERT_EQ(msgs.size(), 2U);
}

TEST_F(InferShapeKernelTest, test_set_output_shape_fail_001) {
  uint64_t shape_data[18] = {
      2147483698, 5,  6,  7,  8,  1,  1, 1, 1,
      2147483698, 11, 12, 13, 14, 15, 1, 1, 1
  };
  size_t shapebuffer_addr_size = 2 * sizeof(uint64_t) * 9;
  auto shape_tensor_data = std::make_unique<GertTensorData>(shape_data, shapebuffer_addr_size, kTensorPlacementEnd, -1);
  auto run_context = KernelRunContextFaker().KernelIONum(1, 2).NodeIoNum(1, 2).Build();
  StorageShape output1{{1,2,3,4}, {1,2,3,4}};
  StorageShape output2{{1,2,3,4}, {1,2,3,4}};
  run_context.value_holder[0].Set(shape_tensor_data.get(), nullptr);
  run_context.value_holder[1].Set(&output1, nullptr);
  run_context.value_holder[2].Set(&output2, nullptr);
  ASSERT_EQ(kernel::SetOutputShape(run_context), ge::GRAPH_FAILED);
}

TEST_F(InferShapeKernelTest, test_set_output_shape_fail_002) {
  uint32_t shape_data[36] = {50, 5,  6,  7,  8,  1,  1, 1, 1,
                             50, 11, 12, 13, 14, 15, 1, 1, 1};
  size_t shapebuffer_addr_size = 2 * sizeof(uint64_t) * 9;
  auto shape_tensor_data = std::make_unique<GertTensorData>(shape_data, shapebuffer_addr_size, kTensorPlacementEnd, -1);
  auto run_context = KernelRunContextFaker().KernelIONum(1, 2).NodeIoNum(1, 2).Build();
  StorageShape output1{{1,2,3,4}, {1,2,3,4}};
  StorageShape output2{{1,2,3,4}, {1,2,3,4}};
  run_context.value_holder[0].Set(shape_tensor_data.get(), nullptr);
  run_context.value_holder[1].Set(&output1, nullptr);
  run_context.value_holder[2].Set(&output2, nullptr);
  ASSERT_EQ(kernel::SetOutputShape(run_context), ge::GRAPH_FAILED);
}
}
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
#include "faker/kernel_run_context_facker.h"
#include "register/kernel_registry.h"
#include "exe_graph/runtime/infer_shape_range_context.h"
#include "register/op_impl_registry.h"
#include "mmpa/mmpa_api.h"
#include "faker/space_registry_faker.h"

namespace gert {
namespace kernel {
ge::graphStatus InferShapeRange(KernelContext *context);
ge::graphStatus CreateTensorRangesAndShapeRanges(KernelContext *context);
}
struct InferShapeRangeKernelTest : public testing::Test {
  KernelRegistry& registry = KernelRegistry::GetInstance();
};

ge::graphStatus CopyInferShapeRange(InferShapeRangeContext* context){
  auto input = context->GetInputShapeRange(0);
  auto output = context->GetOutputShapeRange(0);
  *output = *input;
  return ge::GRAPH_SUCCESS;
}

UINT32 StubInfershapeRange(InferShapeRangeContext *context){
  return ge::GRAPH_SUCCESS;
}

TEST_F(InferShapeRangeKernelTest, FindInferShapeRangeFunc_Success) {
  auto run_context = BuildKernelRunContext(2, 3);
  const char *node_type = "stub_infer_shape_range";
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->CreateOrGetOpImpl(node_type)->infer_shape_range
      = StubInfershapeRange;
  auto space_registry = std::make_shared<OpImplSpaceRegistryV2>();
  ASSERT_NE(space_registry, nullptr);
  space_registry->CreateOrGetOpImpl(node_type)->infer_shape_range = StubInfershapeRange;

  run_context.value_holder[0].Set(const_cast<char *>(node_type), nullptr);
  run_context.value_holder[1].Set(space_registry.get(), nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("FindInferShapeRangeFunc")->run_func(run_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(run_context.value_holder[2].GetValue<void *>(), reinterpret_cast<void*>(StubInfershapeRange));
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->CreateOrGetOpImpl(node_type)->infer_shape_range
      = nullptr;
}

TEST_F(InferShapeRangeKernelTest, FindInferShapeRangeFunc_InputEmptyFailed) {
  auto run_context = BuildKernelRunContext(2, 3);
  auto space_registry = SpaceRegistryFaker().Build();
  ASSERT_NE(space_registry, nullptr);
  run_context.value_holder[1].Set(space_registry.get(), nullptr);
  ASSERT_NE(registry.FindKernelFuncs("FindInferShapeRangeFunc")->run_func(run_context), ge::GRAPH_SUCCESS);
}

TEST_F(InferShapeRangeKernelTest, FindInferShapeRangeFunc_Failed) {
  auto run_context = BuildKernelRunContext(2, 1);
  std::string node_type = "null_infer_shape_range";
  run_context.value_holder[0].Set(const_cast<char *>(node_type.c_str()), nullptr);
  auto space_registry = SpaceRegistryFaker().Build();
  ASSERT_NE(space_registry, nullptr);
  run_context.value_holder[1].Set(space_registry.get(), nullptr);
  ASSERT_NE(registry.FindKernelFuncs("FindInferShapeRangeFunc")->run_func(run_context), ge::GRAPH_SUCCESS);
}

UINT32 StubFailedInfershapeRange(InferShapeContext *){
  return 0x01;
}

TEST_F(InferShapeRangeKernelTest, InferShapeRange_InferFuncEmptyFailed) {
  auto run_context = BuildKernelRunContext(2, 3);
  run_context.value_holder[1].Set(nullptr, nullptr);
  ASSERT_NE(registry.FindKernelFuncs("InferShapeRange")->run_func(run_context), ge::GRAPH_SUCCESS);
}

TEST_F(InferShapeRangeKernelTest, InferShapeRange_InferFuncNotExistFailed) {
  auto run_context = BuildKernelRunContext(0, 3);
  ASSERT_NE(registry.FindKernelFuncs("InferShapeRange")->run_func(run_context), ge::GRAPH_SUCCESS);
}

TEST_F(InferShapeRangeKernelTest, InferShapeRange_InferFuncFailed) {
  auto run_context = BuildKernelRunContext(2, 3);
  run_context.value_holder[1].Set(reinterpret_cast<void*>(StubFailedInfershapeRange), nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("InferShapeRange")->run_func(run_context), 0x01);
  registry.FindKernelFuncs("InferShapeRange")->trace_printer(run_context);
}

TEST_F(InferShapeRangeKernelTest, CreateTensorRangesAndShapeRanges_Success) {
  StorageShape input1 = {{2, 8, 224, 224}, {2, 8, 224, 16, 16}};
  StorageShape input2 = {{6, 8}, {2, 4}};
  auto run_context = BuildKernelRunContext(2, 2);
  run_context.value_holder[0].Set(reinterpret_cast<void *>(&input1), nullptr);
  run_context.value_holder[1].Set(reinterpret_cast<void *>(&input2), nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("CreateTensorRangesAndShapeRanges")->outputs_creator(nullptr, run_context),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("CreateTensorRangesAndShapeRanges")->run_func(run_context),
            ge::GRAPH_SUCCESS);

  auto out1 = run_context.value_holder[2].GetPointer<Range<Shape>>();
  auto out2 = run_context.value_holder[3].GetPointer<Range<Shape>>();
  ASSERT_NE(out1, nullptr);
  ASSERT_NE(out2, nullptr);
  auto out_shape1 = out1->GetMin();
  auto out_shape2 = out2->GetMax();
  EXPECT_EQ(*out_shape1, input1.GetOriginShape());
  EXPECT_EQ(*out_shape2, input2.GetOriginShape());
  run_context.FreeValue(2);
  run_context.FreeValue(3);
}

TEST_F(InferShapeRangeKernelTest, InferShapeRange_BuildOutputsSuccess) {
  Shape min{2, 8, 224, 224};
  Shape max{2, 8, 224, 16, 16};
  auto input1 = Range<Shape>(&min, &max);
  auto run_context = KernelRunContextFaker().IrInstanceNum({1}).NodeIoNum(1,1).KernelIONum(2, 3).Build();
  run_context.value_holder[0].Set(reinterpret_cast<void *>(&input1), nullptr);
  run_context.value_holder[1].Set(reinterpret_cast<void *>(StubInfershapeRange), nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("InferShapeRange")->outputs_creator(nullptr, run_context),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("InferShapeRange")->run_func(run_context),
            ge::GRAPH_SUCCESS);
  registry.FindKernelFuncs("InferShapeRange")->trace_printer(run_context);
  auto out1 = run_context.value_holder[2].GetPointer<Range<Shape>>();
  auto out2 = run_context.value_holder[3].GetPointer<Shape>();
  auto out3 = run_context.value_holder[4].GetPointer<Shape>();
  ASSERT_NE(out1, nullptr);
  ASSERT_NE(out2, nullptr);
  ASSERT_NE(out3, nullptr);
  EXPECT_EQ(out1->GetMin(), out2);
  EXPECT_EQ(out1->GetMax(), out3);
  run_context.FreeValue(2);
  run_context.FreeValue(3);
  run_context.FreeValue(4);
}

TEST_F(InferShapeRangeKernelTest, CreateTensorRangesAndShapeRanges_InputChangeSuccess) {
  StorageShape input1 = {{2, 8, 224, 224}, {2, 8, 224, 16, 16}};
  StorageShape input2 = {{6, 8}, {2, 4}};
  auto run_context = BuildKernelRunContext(2, 2);
  run_context.value_holder[0].Set(reinterpret_cast<void *>(&input1), nullptr);
  run_context.value_holder[1].Set(reinterpret_cast<void *>(&input2), nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("CreateTensorRangesAndShapeRanges")->outputs_creator(nullptr, run_context),
            ge::GRAPH_SUCCESS);

  StorageShape actual_input1 = {{2, 8, 224, 224}, {2, 8, 224, 16, 16}};
  StorageShape actual_input2 = {{6, 8}, {2, 4}};
  run_context.value_holder[0].Set(reinterpret_cast<void *>(&actual_input1), nullptr);
  run_context.value_holder[1].Set(reinterpret_cast<void *>(&actual_input2), nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("CreateTensorRangesAndShapeRanges")->run_func(run_context),
            ge::GRAPH_SUCCESS);

  auto out1 = run_context.value_holder[2].GetPointer<Range<Shape>>();
  auto out2 = run_context.value_holder[3].GetPointer<Range<Shape>>();
  ASSERT_NE(out1, nullptr);
  ASSERT_NE(out2, nullptr);
  auto out_shape1 = out1->GetMin();
  auto out_shape2 = out2->GetMax();
  EXPECT_EQ(*out_shape1, actual_input1.GetOriginShape());
  EXPECT_EQ(*out_shape2, actual_input2.GetOriginShape());
  run_context.FreeValue(2);
  run_context.FreeValue(3);
}
}
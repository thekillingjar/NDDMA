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
#include "runtime/model_v2_executor.h"
#include "common/share_graph.h"
#include "lowering/graph_converter.h"
#include "faker/global_data_faker.h"
#include "faker/aicpu_taskdef_faker.h"
#include "stub/gert_runtime_stub.h"
#include "faker/fake_value.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_shape_range_context.h"
#include "register/op_impl_registry.h"
#include "depends/checker/shape_checker.h"
#include "common/bg_test.h"
#include <iostream>

using namespace ge;
namespace gert {
namespace {
ge::graphStatus InferShapeForFakeNode(InferShapeContext *context) {
  std::cout << "fake node infershape" << std::endl;
  Shape expect_shape({16, 2, 2, 1});
  auto input_tensor0 = context->GetInputTensor(0UL);
  EXPECT_EQ(input_tensor0->GetDataType(), ge::DT_FLOAT);
  ShapeChecker::CheckShape(input_tensor0->GetStorageShape(), expect_shape);
  ShapeChecker::CheckShape(input_tensor0->GetOriginShape(), expect_shape);
  EXPECT_EQ(input_tensor0->GetStorageFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(input_tensor0->GetOriginFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(input_tensor0->GetAddr(), nullptr);
  EXPECT_EQ(input_tensor0->GetSize(), 0);
  auto input_tensor1 = context->GetInputTensor(1UL);
  EXPECT_EQ(input_tensor1->GetDataType(), ge::DT_FLOAT);
  ShapeChecker::CheckShape(input_tensor1->GetStorageShape(), expect_shape);
  ShapeChecker::CheckShape(input_tensor1->GetOriginShape(), expect_shape);  
  EXPECT_EQ(input_tensor1->GetStorageFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(input_tensor1->GetOriginFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(input_tensor1->GetAddr(), nullptr);
  EXPECT_EQ(input_tensor1->GetSize(), 0);
  auto output_shape = context->GetOutputShape(0UL);
  *output_shape = input_tensor0->GetStorageShape();
  return ge::GRAPH_SUCCESS;
}

struct FakeNodeCompileInfo {
  int64_t a;
};

ge::graphStatus TilingForFakeNode(TilingContext *context) {
  std::cout << "fake node tiling" << std::endl;
  Shape expect_shape({16, 2, 2, 1});
  auto input_tensor0 = context->GetInputTensor(0UL);
  EXPECT_EQ(input_tensor0->GetDataType(), ge::DT_FLOAT);
  ShapeChecker::CheckShape(input_tensor0->GetStorageShape(), expect_shape);
  ShapeChecker::CheckShape(input_tensor0->GetOriginShape(), expect_shape);
  EXPECT_EQ(input_tensor0->GetStorageFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(input_tensor0->GetOriginFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(input_tensor0->GetAddr(), nullptr);
  EXPECT_EQ(input_tensor0->GetSize(), 0);
  auto input_tensor1 = context->GetInputTensor(1UL);
  EXPECT_EQ(input_tensor1->GetDataType(), ge::DT_FLOAT);
  ShapeChecker::CheckShape(input_tensor1->GetStorageShape(), expect_shape);
  ShapeChecker::CheckShape(input_tensor1->GetOriginShape(), expect_shape);  
  EXPECT_EQ(input_tensor1->GetStorageFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(input_tensor1->GetOriginFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(input_tensor1->GetAddr(), nullptr);
  EXPECT_EQ(input_tensor1->GetSize(), 0);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForFakeNodeWithDeterminic(TilingContext *context) {
  std::cout << "fake node tiling check deterministc" << std::endl;
  // 确定性计算的值为1
  auto deterministic = context->GetDeterministic();
  EXPECT_EQ(deterministic, 1);
  //  强一致性计算紧急需求上库，ge暂时不能依赖metadef，已于BBIT及本地验证DT通过，后续补上
  //  auto deterministic_level = context->GetDeterministicLevel();
  //  EXPECT_EQ(deterministic_level, 2);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingParseForFakeNode(TilingParseContext *context) {
  auto compile_info = context->GetCompiledInfo<FakeNodeCompileInfo>();
  compile_info->a = 10;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferShapeRangeForFakeNode(InferShapeRangeContext *context) {
  std::cout << "fake node infershaperange" << std::endl;
  Shape expect_shape({16, 2, 2, 1});
  auto input_tensor_range0 = context->GetInputTensorRange(0UL);
  auto input_min_tensor0 = input_tensor_range0->GetMin();
  EXPECT_EQ(input_min_tensor0->GetDataType(), ge::DT_FLOAT);
  ShapeChecker::CheckShape(input_min_tensor0->GetStorageShape(), expect_shape);
  ShapeChecker::CheckShape(input_min_tensor0->GetOriginShape(), expect_shape);
  EXPECT_EQ(input_min_tensor0->GetStorageFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(input_min_tensor0->GetOriginFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(input_min_tensor0->GetAddr(), nullptr);
  EXPECT_EQ(input_min_tensor0->GetSize(), 0);
  auto input_max_tensor0 = input_tensor_range0->GetMax();
  EXPECT_EQ(input_max_tensor0->GetDataType(), ge::DT_FLOAT);
  ShapeChecker::CheckShape(input_max_tensor0->GetStorageShape(), expect_shape);
  ShapeChecker::CheckShape(input_max_tensor0->GetOriginShape(), expect_shape);
  EXPECT_EQ(input_max_tensor0->GetStorageFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(input_max_tensor0->GetOriginFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(input_max_tensor0->GetAddr(), nullptr);
  EXPECT_EQ(input_max_tensor0->GetSize(), 0);
  auto input_tensor_range1 = context->GetInputTensorRange(1UL);
  auto input_min_tensor1 = input_tensor_range1->GetMin();
  EXPECT_EQ(input_min_tensor1->GetDataType(), ge::DT_FLOAT);
  ShapeChecker::CheckShape(input_min_tensor1->GetStorageShape(), expect_shape);
  ShapeChecker::CheckShape(input_min_tensor1->GetOriginShape(), expect_shape);

  EXPECT_EQ(input_min_tensor1->GetStorageFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(input_min_tensor1->GetOriginFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(input_min_tensor1->GetAddr(), nullptr);
  EXPECT_EQ(input_min_tensor1->GetSize(), 0);
  auto input_max_tensor1 = input_tensor_range1->GetMax();
  EXPECT_EQ(input_max_tensor1->GetDataType(), ge::DT_FLOAT);
  ShapeChecker::CheckShape(input_max_tensor1->GetStorageShape(), expect_shape);
  ShapeChecker::CheckShape(input_max_tensor1->GetOriginShape(), expect_shape);
  EXPECT_EQ(input_max_tensor1->GetStorageFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(input_max_tensor1->GetOriginFormat(), ge::FORMAT_NCHW);
  EXPECT_EQ(input_max_tensor1->GetAddr(), nullptr);
  EXPECT_EQ(input_max_tensor1->GetSize(), 0);
  auto output_shape_range = context->GetOutputShapeRange(0);
  *output_shape_range->GetMax() = input_max_tensor0->GetStorageShape();
  *output_shape_range->GetMin() = input_min_tensor0->GetStorageShape();
  return ge::GRAPH_SUCCESS;
}

IMPL_OP(FakeAicoreNode).InferShape(InferShapeForFakeNode)
                       .Tiling(TilingForFakeNode)
                       .TilingParse<FakeNodeCompileInfo>(TilingParseForFakeNode);

IMPL_OP(FakeAicoreNodeWithDeterministc).InferShape(InferShapeForFakeNode)
                                       .Tiling(TilingForFakeNodeWithDeterminic)
                                       .TilingParse<FakeNodeCompileInfo>(TilingParseForFakeNode);

IMPL_OP(FakeShapeRangeNode).InferShape(InferShapeForFakeNode)
                           .InferShapeRange(InferShapeRangeForFakeNode);
}

class Runtime2GetTensorKernelSystemTest : public bg::BgTest {
  void SetUp() {
    SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  }
};

/**
 * 用例描述：验证算子交付件（Tiling,infershape,infershaperange）中调用GetInputTensor接口能正常获取到Shape，Format，DataType等信息
 *
 * 预置条件：
 * 1. 写一个fake的aicore算子，在其tiling函数，infershape函数中调用GetInputTensor获取Tensor对象
 * 2. 写一个fake的三类aicpu算子，在其infershaperange函数中调用GetInputTensor接口获取Tensor对象
 * 3. 构造一个包含以上两个算子的图
 *
 * 测试步骤：
 * 1. 按照预制条件构造好两张图
 * 2. lowering、加载计算图
 * 3. 执行计算图，并在fake算子的Tiling，Infershape，InfershapeRange函数中，对GetInputTensor接口获取到的Tensor对象里面的内容进行校验
 *
 * 预期结果：
 * 1. 图执行成功
 * 2. 算子交付件中从GetInputTensor接口中获取到Tensor对象，且从Tensor对象中获取到的Shape信息，Format信息，DataType信息均为实际值且符合预期
 * 3. 算子交付件中从GetInputTensor接口中获取到Tensor对象，且从Tensor对象中获取到的地址信息，Size信息为空
 */
TEST_F(Runtime2GetTensorKernelSystemTest, GetTensorKernelSystemTest) {
  auto graph = ShareGraph::BuildFakeGetTensorNodeGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  AiCpuTfTaskDefFaker aicpu_task_def_faker;
  auto ge_root_model = builder.AddTaskDef("FakeAicoreNode", AiCoreTaskDefFaker("FakeAicoreNodeStubBin").WithHandle())
                       .AddTaskDef("FakeShapeRangeNode", aicpu_task_def_faker.SetNeedMemcpy(true))
                       .BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  GertRuntimeStub fakeRuntime;
  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto outputs = FakeTensors({16, 2, 2, 1}, 1, nullptr, kOnDeviceHbm, ge::FORMAT_NCHW);
  auto inputs = FakeTensors({16, 2, 2, 1}, 4, nullptr, kOnDeviceHbm, ge::FORMAT_NCHW);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                    outputs.GetTensorList(), outputs.size()),
            ge::GRAPH_SUCCESS);

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                    outputs.GetTensorList(), outputs.size()),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

/**
 * 用例描述：验证算子交付件Tiling中调用GetDeterministic接口能正常获取到确定性计算信息
 *
 * 预置条件：
 * 1. 写一个fake的aicore算子，在其tiling函数中调用GetDeterministic获取确定性计算信息
 * 3. 构造一个包含以上算子的图
 *
 * 测试步骤：
 * 1. 按照预制条件构造好两张图
 * 2. lowering、加载计算图
 * 3. 执行计算图，并在fake算子的Tiling函数中，对GetDeterministic接口获取到的确定性计算信息进行校验
 *
 * 预期结果：
 * 1. 图执行成功
 * 2. 算子交付件中从GetDeterministic接口中获取到确定性计算信息且符合预期
 */
TEST_F(Runtime2GetTensorKernelSystemTest, DeterministicSystemTest) {
  auto graph = ShareGraph::BuildFakeDeterministicNodeGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  AiCpuTfTaskDefFaker aicpu_task_def_faker;
  auto ge_root_model = builder.AddTaskDef("FakeAicoreNodeWithDeterministc", AiCoreTaskDefFaker("FakeAicoreNodeStubBin").WithHandle())
                       .BuildGeRootModel();

  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  GertRuntimeStub fakeRuntime;
  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto outputs = FakeTensors({16, 2, 2, 1}, 1, nullptr, kOnDeviceHbm, ge::FORMAT_NCHW);
  auto inputs = FakeTensors({16, 2, 2, 1}, 2, nullptr, kOnDeviceHbm, ge::FORMAT_NCHW);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                    outputs.GetTensorList(), outputs.size()),
            ge::GRAPH_SUCCESS);

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                    outputs.GetTensorList(), outputs.size()),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

/**
 * 用例描述：验证算子交付件（Tiling,infershape,infershaperange）中always_zeros_copy场景调用GetInputTensor接口能正常获取到Shape，Format，DataType等信息
 *
 * 预置条件：
 * 1. 写一个fake的aicore算子，在其tiling函数，infershape函数中调用GetInputTensor获取Tensor对象
 * 2. 写一个fake的三类aicpu算子，在其infershaperange函数中调用GetInputTensor接口获取Tensor对象
 * 3. 构造一个包含以上两个算子的图
 * 4. 打开always_zero_copy开关
 *
 * 测试步骤：
 * 1. 按照预制条件构造好两张图
 * 2. 打开always_zero_copy开关
 * 3. lowering、加载计算图
 * 4. 执行计算图，并在fake算子的Tiling，Infershape，InfershapeRange函数中，对GetInputTensor接口获取到的Tensor对象里面的内容进行校验
 *
 * 预期结果：
 * 1. 图执行成功
 * 2. 算子交付件中从GetInputTensor接口中获取到Tensor对象，且从Tensor对象中获取到的Shape信息，Format信息，DataType信息均为实际值且符合预期
 * 3. 算子交付件中从GetInputTensor接口中获取到Tensor对象，且从Tensor对象中获取到的地址信息，Size信息为空
 */
TEST_F(Runtime2GetTensorKernelSystemTest, GetTensorKernelSystemTestAlwaysZeroCopy) {
  auto graph = ShareGraph::BuildFakeGetTensorNodeZeroCopyGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  AiCpuTfTaskDefFaker aicpu_task_def_faker;
  auto ge_root_model = builder.AddTaskDef("FakeAicoreNode", AiCoreTaskDefFaker("FakeAicoreNodeStubBin").WithHandle())
                       .AddTaskDef("FakeShapeRangeNode", aicpu_task_def_faker.SetNeedMemcpy(true))
                       .BuildGeRootModel();
  LoweringOption option{.trust_shape_on_out_tensor = false, .always_zero_copy = true};
  bg::ValueHolder::PopGraphFrame();  // 不需要BgTest自带的Frame
  ModelConverter::Args args{option, nullptr, nullptr, nullptr, nullptr};
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model, args);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  GertRuntimeStub fakeRuntime;
  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto output0 = TensorFaker().Shape({16, 2, 2, 1}).Format(ge::FORMAT_NCHW).Placement(kOnDeviceHbm).Build();
  auto output1 = TensorFaker().Shape({16, 2, 2, 1}).Format(ge::FORMAT_NCHW).Placement(kOnDeviceHbm).Build();
  std::vector<Tensor *> outputs = {output0.GetTensor(), output1.GetTensor()};
  auto inputs = FakeTensors({16, 2, 2, 1}, 4, nullptr, kOnDeviceHbm, ge::FORMAT_NCHW);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                    outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(),
                                    outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}
}
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
#include "common/share_graph.h"
#include "faker/ge_model_builder.h"
#include "faker/aicore_taskdef_faker.h"
#include "faker/fake_value.h"
#include "stub/gert_runtime_stub.h"
#include "check/executor_statistician.h"

#include "exe_graph/lowering/value_holder.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_dump_utils.h"
#include "graph/utils/execute_graph_utils.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "lowering/model_converter.h"
#include "aicore/launch_kernel/ai_core_launch_kernel.h"
#include "graph/utils/node_utils.h"
namespace gert {
namespace {
class TilingWatcher {
 public:
  static void OnExecuteEvent(int type, void *void_arg, ExecutorEvent event, const void *node, KernelStatus result) {
    static_cast<TilingWatcher *>(void_arg)->OnEvent(event, static_cast<const ::Node *>(node), result);
  }
  void OnEvent(ExecutorEvent event, const ::Node *node, ::KernelStatus result) {
    if (node == nullptr) {
      return;
    }
    auto extend_info = static_cast<const KernelExtendInfo *>(node->context.kernel_extend_info);
    if (extend_info == nullptr) {
      return;
    }
    if (strcmp(extend_info->GetKernelType(), "Tiling") == 0 || strcmp(extend_info->GetKernelType(), "CacheableTiling") == 0) {
      auto tiling_context = reinterpret_cast<const TilingContext *>(&node->context);
      out_shape_ = *tiling_context->GetOutputShape(0);
    }
  }
  void Clear() {
    out_shape_ = {};
  }
  const StorageShape &GetOutShape() const {
    return out_shape_;
  }

 private:
  StorageShape out_shape_;
};

uint32_t g_stub_output_shape[8 + 1] = {4, 1, 2, 3, 1};
ge::graphStatus StubLaunch_UpdateOutputShape(KernelContext *context) {
  auto extend_context = reinterpret_cast<ExtendedKernelContext *>(context);
  if (strcmp(extend_context->GetComputeNodeInfo()->GetNodeType(), "NonZero") != 0) {
    return ge::GRAPH_SUCCESS;
  }
  auto shape_buffer =
      context->GetInputValue<gert::GertTensorData *>(static_cast<size_t>(kernel::InputCommon::kShapeBufferAddr));
  GE_ASSERT_EOK(
      memcpy_s(shape_buffer->GetAddr(), shape_buffer->GetSize(), g_stub_output_shape, sizeof(g_stub_output_shape)));
  return ge::GRAPH_SUCCESS;
}
}  // namespace
class TrustOutputTensorST : public testing::Test {};
/**
 * 用例描述：打开trust_output_tensor开关时，输出节点的InferShape被消除
 * 补充：对应的Findnfershape节点也会被删除
 *
 * 预置条件：
 * 1. Fake Add算子的lowering和对应InferShape、Tiling等执行kernel
 *
 * 测试步骤：
 * 1. 构造一张包含Add算子的单算子图
 * 2. lowering、加载这张图
 * 3. 挂接ess（kernel执行监控）
 * 4. fake输入、输出tensor，origin shape为8，3，224，224；storage shape为8，1，224，224，16
 * 5. 将option trust_output_tensor设置为true，执行图
 *
 * 预期结果：
 * 1. 执行成功
 * 2. 输出tensor的storage shape未变化
 * 3. 通过ess观察，没有执行InferShape kernel
 */
TEST_F(TrustOutputTensorST, TrustOutputTensor_NoInferShape_WhenEnable) {
  auto graph = ShareGraph::BuildSingleNodeGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle()).BuildGeRootModel();

  LoweringOption option{.trust_shape_on_out_tensor = true};
  ModelConverter::Args args(option, nullptr, nullptr, nullptr, nullptr);
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model, args);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  ge::DumpGraph(exe_graph.get(), "E2EAddGraph");

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);
  auto ess = StartExecutorStatistician(model_executor);
  auto tiling_watcher = model_executor->GetSubscribers().AddSubscriber<TilingWatcher>();
  ASSERT_NE(ess, nullptr);
  ASSERT_NE(tiling_watcher, nullptr);
  ess->Clear();

  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto output = TensorFaker().StorageShape({8, 1, 224, 224, 16}).OriginShape({8, 3, 224, 224}).Build();
  output.GetTensor()->SetPlacement(kOnDeviceHbm);
  std::vector<Tensor *> outputs = {output.GetTensor()};
  auto inputs = FakeTensors({8 * 3 * 224 * 224}, 2);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));


  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_watcher->GetOutShape().GetOriginShape(), Shape({8, 3, 224, 224}));
  EXPECT_EQ(tiling_watcher->GetOutShape().GetStorageShape(), Shape({8, 1, 224, 224, 16}));

  tiling_watcher->Clear();
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_watcher->GetOutShape().GetOriginShape(), Shape({8, 3, 224, 224}));
  EXPECT_EQ(tiling_watcher->GetOutShape().GetStorageShape(), Shape({8, 1, 224, 224, 16}));

  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "InferShape"), 0);
  EXPECT_EQ(output.GetTensor()->GetOriginShape(), Shape({8, 3, 224, 224}));
  EXPECT_EQ(output.GetTensor()->GetStorageShape(), Shape({8, 1, 224, 224, 16}));

  auto init_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Init");
  ASSERT_NE(init_node, nullptr);
  auto init_graph = ge::FastNodeUtils::GetSubgraphFromNode(init_node, 0);
  EXPECT_NE(ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_graph, "NoOp"), nullptr);

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

/**
 * 用例描述：打开trust_output_tensor开关但是没有满足零拷贝条件，拷贝输出数据时校验大小报错
 * 补充：对应的Findnfershape节点也会被删除
 *
 * 预置条件：
 * 1. Fake Add算子的lowering和对应InferShape、Tiling等执行kernel
 *
 * 测试步骤：
 * 1. 构造一张包含Add算子的单算子图
 * 2. lowering、加载这张图
 * 3. 挂接ess（kernel执行监控）
 * 4. fake输入、输出tensor，origin shape为8，3，224，224；storage shape为8，1，224，224，16
 * 5. 将option trust_output_tensor设置为true，执行图
 * 6. 输出内存类型为kOnHost（和Add算子输出所需placement不一致，不满足零拷贝条件）
 *
 * 预期结果：
 * 1. 执行失败
 * 2. 校验失败日志
 */
TEST_F(TrustOutputTensorST, TrustOutputTensor_Failed_WhenDisableZeroCopy) {
  auto graph = ShareGraph::BuildSingleNodeGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle()).BuildGeRootModel();

  LoweringOption option{.trust_shape_on_out_tensor = true};
  ModelConverter::Args args(option, nullptr, nullptr, nullptr, nullptr);
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model, args);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  ge::DumpGraph(exe_graph.get(), "E2EAddGraph");

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);
  auto ess = StartExecutorStatistician(model_executor);
  auto tiling_watcher = model_executor->GetSubscribers().AddSubscriber<TilingWatcher>();
  ASSERT_NE(ess, nullptr);
  ASSERT_NE(tiling_watcher, nullptr);
  ess->Clear();

  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto output = TensorFaker().StorageShape({8, 1, 224, 224, 16}).OriginShape({8, 3, 224, 224}).Build();
  output.GetTensor()->SetPlacement(kOnHost);
  std::vector<Tensor *> outputs = {output.GetTensor()};
  auto inputs = FakeTensors({8 * 3 * 224 * 224}, 2);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  runtime_stub.GetSlogStub().Clear();
  ASSERT_NE(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(runtime_stub.GetSlogStub().FindErrorLogEndsWith("Failed to copy output tensor data to the given buffer, output tensor data size 4849664 is less than copy size size 25690112"), 0);

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

/**
 * 用例描述：默认场景trust_output_tensor开关关闭
 *
 * 预置条件：
 * 1. Fake Add算子的lowering和对应InferShape、Tiling等执行kernel
 *
 * 测试步骤：
 * 1. 构造一张包含Add算子的单算子图
 * 2. lowering、加载这张图
 * 3. 挂接ess（kernel执行监控）
 * 4. fake输入tensor，origin shape为8，3，224，224；storage shape为8，1，224，224，16；fake输出tensor，shape填写为任意值
 * 5. 执行图
 *
 * 预期结果：
 * 1. 执行成功
 * 2. 通过ess观察，执行了InferShape kernel
 */
TEST_F(TrustOutputTensorST, TrustOutputTensor_InferShape_WhenDefault) {
  auto graph = ShareGraph::BuildSingleNodeGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle()).BuildGeRootModel();

  LoweringOption option;
  ModelConverter::Args args(option, nullptr, nullptr, nullptr, nullptr);
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model, args);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  ge::DumpGraph(exe_graph.get(), "E2EAddGraph");

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);
  auto ess = StartExecutorStatistician(model_executor);
  auto tiling_watcher = model_executor->GetSubscribers().AddSubscriber<TilingWatcher>();
  ASSERT_NE(ess, nullptr);
  ASSERT_NE(tiling_watcher, nullptr);

  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto output = TensorFaker().StorageShape({8, 1, 224, 224, 16}).OriginShape({8, 3, 224, 224}).Build();
  std::vector<Tensor *> outputs = {output.GetTensor()};
  auto inputs = FakeTensors({8 * 3 * 224 * 224}, 2);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ess->Clear();
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(output.GetTensor()->GetOriginShape(), Shape({8 * 3 * 224 * 224}));
  EXPECT_EQ(output.GetTensor()->GetStorageShape(), Shape({8 * 3 * 224 * 224}));

  tiling_watcher->Clear();
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(output.GetTensor()->GetOriginShape(), Shape({8 * 3 * 224 * 224}));
  EXPECT_EQ(output.GetTensor()->GetStorageShape(), Shape({8 * 3 * 224 * 224}));
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "InferShape"), 2);

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

/**
 * 用例描述：trust_output_tensor开关对第三类算子不生效
 *
 * 预置条件：
 * 1. Fake NonZero算子的lowering和对应InferShapeRange等执行kernel
 *
 * 测试步骤：
 * 1. 构造一张包含NonZero算子的单算子图
 * 2. lowering、加载这张图
 * 3. fake输入tensor，值为{1,2,3,4}；fake输出Tensor，shape为{8,3,224,224}
 * 4. 打桩KernelLaunch，遇到NonZero时，将shape buffer刷新为1，2，3，4
 * 5. 将option trust_output_tensor设置为true，执行图
 *
 * 预期结果：
 * 1. 执行成功
 * 2. 输出Tensor的shape被刷新为1,2,3,4
 */
TEST_F(TrustOutputTensorST, TrustOutputTensor_UpdateOutputShape_WhenType3Node) {
  auto graph = ShareGraph::BuildAiCoreThirdClassNodeGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
                           .AddTaskDef("NonZero", AiCoreTaskDefFaker("NonZeroStubBin").WithHandle())
                           .BuildGeRootModel();

  LoweringOption option{.trust_shape_on_out_tensor = true};
  ModelConverter::Args args(option, nullptr, nullptr, nullptr, nullptr);
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model, args);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  ge::DumpGraph(exe_graph.get(), "E2EType3Graph");

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling().SetUp("LaunchKernelWithHandle", StubLaunch_UpdateOutputShape);

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  ASSERT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto output = TensorFaker().StorageShape({8, 1, 224, 224, 16}).OriginShape({8, 3, 224, 224}).Build();
  output.GetTensor()->SetPlacement(kOnDeviceHbm);
  std::vector<Tensor *> outputs = {output.GetTensor()};
  auto inputs = FakeTensors({1, 2, 3, 4}, 2);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);

  // todo aicore的第三类算子实现有bug，SetOutputShape Node没有添加ordered holders，导致其可能在后续节点的前面执行
  //   因此这里如果添加out tensor的shape检查会失败。
  //   用例TrustOutputTensorAndAlwaysZeroCopy_UpdateOutputShape_WhenType3Node也有同样的问题，待aicore修复后修改此ST

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);

  EXPECT_EQ(output.GetTensor()->GetOriginShape(), Shape({1, 2, 3, 1}));
  EXPECT_EQ(output.GetTensor()->GetStorageShape(), Shape({1, 2, 3, 1}));

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

/**
 * 用例描述：一个节点有部分输出没有连接到NetOutput而是连接到了其他节点时，忽略掉免InferShape能力
 *
 * 预置条件：
 * 1. Fake MinimumGrad、Add算子的lowering和对应InferShape等执行kernel
 *
 * 测试步骤：
 * 1. 构造一张图包含MinimumGrad和Add算子，MinimumGrad输出0连接到NetOutput，输出1连接到Add；Add算子的输出连接到NetOutput
 * 2. lowering、加载这张图
 * 3. fake两个输入Tensor，shape分别为1,2,3,4与4,5,6,7; fake两个输出tensor，shape分别为128与4,5,6,7
 * 4. 将option trust_output_tensor设置为true，执行图
 *
 * 预期结果：
 * 1. 执行成功
 * 2. 输出0的shape被刷新为1,2,3,4
 * 3. 输出1的shape与输入2的shape相同
 */
TEST_F(TrustOutputTensorST, TrustOutputTensor_InferShape_WhenNodeHasMultipleOutputs) {
  auto graph = ShareGraph::BuildMinimumGradAndAddGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
                           .AddTaskDef("MinimumGrad", AiCoreTaskDefFaker("MinimumGradStubBin").WithHandle())
                           .BuildGeRootModel();
  LoweringOption option{.trust_shape_on_out_tensor = true};
  ModelConverter::Args args(option, nullptr, nullptr, nullptr, nullptr);
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model, args);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  ge::DumpGraph(exe_graph.get(), "E2EMinimumGradAndAddGraph");

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  ASSERT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto outputs = FakeTensors({5, 6, 7, 8}, 2);
  auto inputs = FakeTensors({1, 2, 3, 4}, 3);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                    outputs.size()),
            ge::GRAPH_SUCCESS);

  EXPECT_EQ(outputs.GetTensorList()[0]->GetOriginShape(), Shape({1, 2, 3, 4}));
  EXPECT_EQ(outputs.GetTensorList()[0]->GetStorageShape(), Shape({1, 2, 3, 4}));
  EXPECT_EQ(outputs.GetTensorList()[1]->GetOriginShape(), Shape({5, 6, 7, 8}));
  EXPECT_EQ(outputs.GetTensorList()[1]->GetStorageShape(), Shape({5, 6, 7, 8}));

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.GetTensorList(),
                                    outputs.size()),
            ge::GRAPH_SUCCESS);

  EXPECT_EQ(outputs.GetTensorList()[0]->GetOriginShape(), Shape({1, 2, 3, 4}));
  EXPECT_EQ(outputs.GetTensorList()[0]->GetStorageShape(), Shape({1, 2, 3, 4}));
  EXPECT_EQ(outputs.GetTensorList()[1]->GetOriginShape(), Shape({5, 6, 7, 8}));
  EXPECT_EQ(outputs.GetTensorList()[1]->GetStorageShape(), Shape({5, 6, 7, 8}));

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}
/**
 * 用例描述：打开trust_output_tensor和always_zero_copy开关时，执行成功
 *
 * 预置条件：
 * 1. Fake Add算子的lowering和对应InferShape、Tiling等执行kernel
 *
 * 测试步骤：
 * 1. 构造一张包含Add算子的单算子图
 * 2. lowering、加载这张图
 * 3. 挂接ess（kernel执行监控）
 * 4. fake输入、输出tensor，origin shape为8，3，224，224；storage shape为8，1，224，224，16
 * 5. 将option trust_output_tensor和always_zero_copy均设置为true，执行图
 *
 * 预期结果：
 * 1. 执行成功
 * 2. 输出tensor的storage shape未变化
 * 3. 通过ess观察，没有执行InferShape kernel
 * 4. launch观察，输出地址被正确launch
 */
TEST_F(TrustOutputTensorST, TrustOutputTensor_NoInferShapeAndAllocModelOut_WhenEnableNoInferShapeAndAlwaysZeroCopy) {
  auto graph = ShareGraph::BuildSingleNodeGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle()).BuildGeRootModel();

  LoweringOption option{.trust_shape_on_out_tensor = true,.always_zero_copy = true};
  ModelConverter::Args args(option, nullptr, nullptr, nullptr, nullptr);
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model, args);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  ge::DumpGraph(exe_graph.get(), "E2EAddGraph");

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);
  auto ess = StartExecutorStatistician(model_executor);
  auto tiling_watcher = model_executor->GetSubscribers().AddSubscriber<TilingWatcher>();
  ASSERT_NE(ess, nullptr);
  ASSERT_NE(tiling_watcher, nullptr);

  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto output = TensorFaker().StorageShape({8, 1, 224, 224, 16}).OriginShape({8, 3, 224, 224}).Build();
  output.GetTensor()->SetPlacement(kOnDeviceHbm);
  auto out_addr = output.GetTensor()->GetAddr();
  std::vector<Tensor *> outputs = {output.GetTensor()};
  auto inputs = FakeTensors({8 * 3 * 224 * 224}, 2);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ess->Clear();
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_watcher->GetOutShape().GetOriginShape(), Shape({8, 3, 224, 224}));
  EXPECT_EQ(tiling_watcher->GetOutShape().GetStorageShape(), Shape({8, 1, 224, 224, 16}));

  tiling_watcher->Clear();
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_watcher->GetOutShape().GetOriginShape(), Shape({8, 3, 224, 224}));
  EXPECT_EQ(tiling_watcher->GetOutShape().GetStorageShape(), Shape({8, 1, 224, 224, 16}));

  // no infer shape executed
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "InferShape"), 0);
  EXPECT_EQ(output.GetTensor()->GetOriginShape(), Shape({8, 3, 224, 224}));
  EXPECT_EQ(output.GetTensor()->GetStorageShape(), Shape({8, 1, 224, 224, 16}));

  // no AllocModelOut executed
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "AllocMemHbm"), 0);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "AllocMemory"), 0);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "AllocModelOutTensor"), 0);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "AllocModelOutTensorAndUpdateShape"), 0);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "EnsureTensorAtOutMemory"), 0);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "CalcTensorSizeFromShape"), 0);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "CalcTensorSizeFromStorage"), 2);
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "SplitTensorForOutputData"), 2);

  ASSERT_EQ(runtime_stub.GetRtsRuntimeStub().GetLaunchWithHandleArgs().size(), 1U);
  ASSERT_EQ(runtime_stub.GetRtsRuntimeStub().GetLaunchWithHandleArgs().begin()->second.size(), 2U);
  for (const auto &args : runtime_stub.GetRtsRuntimeStub().GetLaunchWithHandleArgs().begin()->second) {
    auto addresses = args->GetLaunchAddresses();
    ASSERT_EQ(addresses[2], out_addr);
  }

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

/**
 * 用例描述：打开trust_output_tensor和always_zero_copy开关时，执行失败
 *
 * 预置条件：
 * 1. Fake Add算子的lowering和对应InferShape、Tiling等执行kernel
 *
 * 测试步骤：
 * 1. 构造一张包含Add算子的单算子图
 * 2. lowering、加载这张图
 * 3. 挂接ess（kernel执行监控）
 * 4. fake输入、输出tensor，origin shape为8，3，224，224；storage shape为8，1，224，224，16，内存比实际所需小
 * 5. 将option trust_output_tensor和always_zero_copy均设置为true，执行图
 *
 * 预期结果：
 * 1. 执行失败
 * 2. 校验失败日志
 */
TEST_F(TrustOutputTensorST, TrustOutputTensorAndAlwaysZeroCopy_CheckOutTensorSizeFailed) {
  auto graph = ShareGraph::BuildSingleNodeGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle()).BuildGeRootModel();

  LoweringOption option{.trust_shape_on_out_tensor = true,.always_zero_copy = true};
  ModelConverter::Args args(option, nullptr, nullptr, nullptr, nullptr);
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model, args);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  ge::DumpGraph(exe_graph.get(), "E2EAddGraph");

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);
  auto ess = StartExecutorStatistician(model_executor);
  auto tiling_watcher = model_executor->GetSubscribers().AddSubscriber<TilingWatcher>();
  ASSERT_NE(ess, nullptr);
  ASSERT_NE(tiling_watcher, nullptr);

  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto output = TensorFaker().StorageShape({8, 1, 224, 224, 1}).OriginShape({8, 3, 224, 224}).Build();
  output.GetTensor()->MutableStorageShape().SetDim(4, 16);
  output.GetTensor()->SetPlacement(kOnDeviceHbm);
  std::vector<Tensor *> outputs = {output.GetTensor()};
  auto inputs = FakeTensors({8 * 3 * 224 * 224}, 2);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ess->Clear();
  runtime_stub.GetSlogStub().Clear();
  ASSERT_NE(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(runtime_stub.GetSlogStub().FindErrorLogEndsWith("The output tensor memory size[1605664] is smaller than the actual size[25690112] required by tensor."), 0);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

/**
 * 用例描述：打开trust_output_tensor和always_zero_copy开关时，第三类算子执行成功
 *
 * 预置条件：
 * 1. Fake NonZero算子的lowering和对应InferShapeRange等执行kernel
 *
 * 测试步骤：
 * 1. 构造一张包含NonZero算子的单算子图
 * 2. lowering、加载这张图
 * 3. fake输入tensor，值为{1,2,3,4}；fake输出Tensor，shape为{8,3,224,224}
 * 4. 打桩KernelLaunch，遇到NonZero时，将shape buffer刷新为1，2，3，4
 * 5. 将option trust_output_tensor设置为true，执行图
 *
 * 预期结果：
 * 1. 执行成功
 * 2. 输出Tensor的shape被刷新为1,2,3,4
 */
TEST_F(TrustOutputTensorST, TrustOutputTensorAndAlwaysZeroCopy_UpdateOutputShape_WhenType3Node) {
  auto graph = ShareGraph::BuildAiCoreThirdClassNodeGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle())
                           .AddTaskDef("NonZero", AiCoreTaskDefFaker("NonZeroStubBin").WithHandle())
                           .BuildGeRootModel();
  LoweringOption option{.trust_shape_on_out_tensor = true, .always_zero_copy = true};
  ModelConverter::Args args(option, nullptr, nullptr, nullptr, nullptr);
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model, args);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());
  ge::DumpGraph(exe_graph.get(), "E2EType3Graph");

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling().SetUp("LaunchKernelWithHandle", StubLaunch_UpdateOutputShape);

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);

  ASSERT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto output = TensorFaker().StorageShape({8, 1, 224, 224, 16}).OriginShape({8, 3, 224, 224}).Build();
  output.GetTensor()->SetPlacement(kOnDeviceHbm);
  std::vector<Tensor *> outputs = {output.GetTensor()};
  auto inputs = FakeTensors({1, 2, 3, 4}, 2);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);
  // todo 在这里检查out tensor的shape会检查失败，原因参考用例TrustOutputTensor_UpdateOutputShape_WhenType3Node

  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);

  EXPECT_EQ(output.GetTensor()->GetOriginShape(), Shape({1, 2, 3, 1}));
  EXPECT_EQ(output.GetTensor()->GetStorageShape(), Shape({1, 2, 3, 1}));

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

/**
 * 用例描述：一个节点的同一个输出作为两个model out的输出，打开trust_output_tensor和always_zero_copy开关时，执行成功
 *
 * 预置条件：
 * 1. Fake Add算子的lowering和对应InferShape、Tiling等执行kernel
 *
 * 测试步骤：
 * 1. 构造一张包含Add算子的单算子图，Add算子的输出作为模型输出0和输出1
 * 2. lowering、加载这张图
 * 3. 挂接ess（kernel执行监控）
 * 4. fake输入、输出tensor，origin shape为8，3，224，224；storage shape为8，1，224，224，16
 * 5. 将option trust_output_tensor和always_zero_copy均设置为true，执行图
 *
 * 预期结果：
 * 1. 执行成功
 * 2. 输出tensor的storage shape未变化
 * 3. 通过ess观察，没有执行InferShape kernel
 * 4. launch观察，输出地址被正确launch
 */


/**
 * Data   Data
 *  \    /
 *   Add   Data
 *    \    /
 *     Add
 *      |
 *   NetOutput
 *
 * 用例描述：
 * 一个图中存在两个相同的节点，打开trust_output_tensor和always_zero_copy开关时，执行成功
 *
 * 补充原因：Lowering优化时对某些节点作了缓存(FindInferShapeFunc\FindTilingFunc之类)，对于做了缓存的节点，相同的算子类型，
 * 在Init图中只存在一个该节点，因此在此场景下，FindInferShapeFunc不能被删除（替换）
 * 预置条件：
 * 1. Fake Add算子的lowering和对应InferShape、Tiling等执行kernel
 *
 * 测试步骤：
 * 1. 构造一张包含Add算子的单算子图，Add算子的输出作为模型输出0和输出1
 * 2. lowering、加载这张图
 * 3. 挂接ess（kernel执行监控）
 * 4. fake输入、输出tensor，origin shape为8，3，224，224；storage shape为8，1，224，224，16
 * 5. 将option trust_output_tensor和always_zero_copy均设置为true，执行图
 *
 * 预期结果：
 * 1. 执行成功
 * 2. 与NetOutput相连接的节点的Infershape可以被删除，另外一个Infershape不能被删除，同时FindInferShapeFunc不能被删除
 * 3. 初始化图中的Add算子FindnfershapeFunc仍然存在
 * 4. 输出shape正确
 */
TEST_F(TrustOutputTensorST, TrustOutputTensor_NoInferShapeButHasFindInferShape_WhenEnableNoInferShapeAndAlwaysZeroCopy) {
  auto graph = ShareGraph::BuildTwoAddNodeGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin").WithHandle()).BuildGeRootModel();

  LoweringOption option{.trust_shape_on_out_tensor = true,.always_zero_copy = true};
  ModelConverter::Args args(option, nullptr, nullptr, nullptr, nullptr);
  auto exe_graph = ModelConverter().ConvertGeModelToExecuteGraph(ge_root_model, args);
  ASSERT_NE(exe_graph, nullptr);
  ASSERT_EQ(3, exe_graph->GetDirectNodesSize());

  GertRuntimeStub runtime_stub;
  runtime_stub.GetKernelStub().StubTiling();

  auto model_executor = ModelV2Executor::Create(exe_graph, ge_root_model);
  ASSERT_NE(model_executor, nullptr);
  auto ess = StartExecutorStatistician(model_executor);
  auto tiling_watcher = model_executor->GetSubscribers().AddSubscriber<TilingWatcher>();
  ASSERT_NE(ess, nullptr);
  ASSERT_NE(tiling_watcher, nullptr);

  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);

  auto output = TensorFaker().StorageShape({8, 1, 224, 224, 16}).OriginShape({8, 3, 224, 224}).Build();
  output.GetTensor()->SetPlacement(kOnDeviceHbm);
  std::vector<Tensor *> outputs = {output.GetTensor()};
  auto inputs = FakeTensors({8 * 3 * 224 * 224}, 2);

  rtStream_t stream;
  ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
  auto i3 = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

  ess->Clear();
  ASSERT_EQ(model_executor->Execute({i3.value}, inputs.GetTensorList(), inputs.size(), outputs.data(), outputs.size()),
            ge::GRAPH_SUCCESS);
  EXPECT_EQ(tiling_watcher->GetOutShape().GetOriginShape(), Shape({8, 3, 224, 224}));
  EXPECT_EQ(tiling_watcher->GetOutShape().GetStorageShape(), Shape({8, 1, 224, 224, 16}));
  tiling_watcher->Clear();

  // no infer shape executed
  EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("Add", "InferShape"), 1);
  EXPECT_EQ(output.GetTensor()->GetOriginShape(), Shape({8, 3, 224, 224}));
  EXPECT_EQ(output.GetTensor()->GetStorageShape(), Shape({8, 1, 224, 224, 16}));

  auto init_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph.get(), "Init");
  ASSERT_NE(init_node, nullptr);
  auto init_graph = ge::FastNodeUtils::GetSubgraphFromNode(init_node, 0);
  EXPECT_NE(ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_graph, "FindInferShapeFunc"), nullptr);

  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
  rtStreamDestroy(stream);
}

}  // namespace gert

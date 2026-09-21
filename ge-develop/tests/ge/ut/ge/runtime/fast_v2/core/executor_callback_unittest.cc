/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "model_v2_executor_unittest.h"
#include <gtest/gtest.h>
#include <mutex>
#include "common/bg_test.h"
#include "faker/fake_value.h"
#include "runtime/model_v2_executor.h"
#include "core/execution_data.h"
#include "exe_graph/runtime/extended_kernel_context.h"
#include "subscriber/profiler/ge_host_profiler.h"
#include "common/global_variables/diagnose_switch.h"
#include "runtime/subscriber/global_profiler.h"
#include "core/builder/model_v2_executor_builder.h"
#include "framework/runtime/executor_option/multi_thread_executor_option.h"
namespace gert {

namespace {
struct TestSubscriber {
  static std::mutex mutex_;
  static void OnExecuteEvent(int32_t sub_exe_graph_type, TestSubscriber *arg, ExecutorEvent event,
                             const void *void_node, KernelStatus result) {
    (void)sub_exe_graph_type;
    auto node = static_cast<const Node *>(void_node);
    auto context = reinterpret_cast<const ExtendedKernelContext *>(&node->context);
    std::lock_guard<std::mutex> lock(mutex_);
    switch (event) {
      case kExecuteStart:
        arg->start_events[context->GetKernelName()]++;
        break;
      case kExecuteEnd:
        if (result == 0) {
          arg->success_events[context->GetKernelName()]++;
        } else {
          arg->failed_events[context->GetKernelName()]++;
        }
        break;
      case kModelStart:
        arg->model_start_events["ModelStart"]++;
        break;
      case kModelEnd:
        arg->model_end_events["ModelEnd"]++;
        break;
      default:
        arg->unknown_events[context->GetKernelName()]++;
        break;
    }
  }
  std::map<std::string, size_t> start_events;
  std::map<std::string, size_t> success_events;
  std::map<std::string, size_t> failed_events;
  std::map<std::string, size_t> unknown_events;
  std::map<std::string, size_t> model_start_events;
  std::map<std::string, size_t> model_end_events;
};
std::mutex TestSubscriber::mutex_;

std::unique_ptr<ModelV2Executor> BuildByType(const ge::ExecuteGraphPtr &graph, const ge::GeRootModelPtr &root_model,
                                             ExecutorType executor_type) {
  return ModelV2ExecutorBuilder(graph).GeRootModel(root_model).Build(executor_type);
}
using ExecutorBuilder = std::function<std::unique_ptr<ModelV2Executor>(const ge::ExecuteGraphPtr &, const ge::GeRootModelPtr &)>;
}  // namespace

class ExecutorCallbackUT : public bg::BgTest {
 public:
  void EXPECT_TestCallbackCalled(ExecutorBuilder builder) {
    auto graph_2_rootmodel = BuildExeGraphFromSingleNode();
    auto graph = graph_2_rootmodel.exe_graph;
    ASSERT_NE(graph, nullptr);

    GertRuntimeStub stub;
    stub.GetKernelStub().AllKernelRegisteredAndSuccess();

    auto model_executor = builder(graph, graph_2_rootmodel.root_model);
    ASSERT_NE(model_executor, nullptr);

    auto arg = model_executor->GetSubscribers().AddSubscriber<TestSubscriber>();
    ASSERT_NE(arg, nullptr);

    ASSERT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
    auto outputs = FakeTensors({2}, 1);
    auto input0 =
        FakeValue<Tensor>(Tensor{{{256}, {256}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, 0});
    auto input1 =
        FakeValue<Tensor>(Tensor{{{256}, {256}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, 0});
    auto input2 = FakeValue<uint64_t>(0);

    ASSERT_EQ(model_executor->Execute({input2.value},
                                      std::vector<Tensor *>({input0.holder.get(), input1.holder.get()}).data(), 2,
                                      reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
              ge::GRAPH_SUCCESS);
    ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);

    EXPECT_FALSE(arg->start_events.empty());
    EXPECT_EQ(arg->start_events.size(), arg->success_events.size());
    EXPECT_TRUE(arg->failed_events.empty());
    EXPECT_TRUE(arg->unknown_events.empty());
    EXPECT_EQ(arg->model_start_events.size(), arg->model_end_events.size());
  }
  void EXPECT_TestBuiltInProfiling(ExecutorBuilder builder) {
    auto graph_2_rootmodel = BuildExeGraphFromSingleNode();
    auto graph = graph_2_rootmodel.exe_graph;
    ASSERT_NE(graph, nullptr);

    GertRuntimeStub stub;
    stub.GetKernelStub().AllKernelRegisteredAndSuccess();

    auto model_executor = builder(graph, graph_2_rootmodel.root_model);
    ASSERT_NE(model_executor, nullptr);
    // profiler init automatically
    auto profiler =
        model_executor->GetSubscribers().MutableBuiltInSubscriber<GeHostProfiler>(BuiltInSubscriberType::kGeProfiling);
    ASSERT_NE(profiler, nullptr);
    ASSERT_FALSE(profiler->IsEnabled(ProfilingType::kGeHost));
    // turn on profiling
    ge::diagnoseSwitch::EnableGeHostProfiling();
    ASSERT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
    auto outputs = FakeTensors({2}, 1);
    auto input0 =
        FakeValue<Tensor>(Tensor{{{256}, {256}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, 0});
    auto input1 =
        FakeValue<Tensor>(Tensor{{{256}, {256}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, 0});
    auto input2 = FakeValue<uint64_t>(0);

    ASSERT_EQ(model_executor->Execute({input2.value},
                                      std::vector<Tensor *>({input0.holder.get(), input1.holder.get()}).data(), 2,
                                      reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
              ge::GRAPH_SUCCESS);
    ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
    EXPECT_NE(GlobalProfilingWrapper::GetInstance()->GetGlobalProfiler()->GetCount(), 0);

    ge::diagnoseSwitch::DisableProfiling();
  }
};

TEST_F(ExecutorCallbackUT, Build_MultiThreadTopo_WithoutExecutorOption_fail) {
  ge::ExecuteGraphPtr graph;
  ge::GeRootModelPtr root_model;
  ASSERT_EQ(BuildByType(graph, root_model, ExecutorType::kTopologicalMultiThread), nullptr);
}

TEST_F(ExecutorCallbackUT, CallbackCalled_Ok_MultiThreadTopo) {
  auto graph_2_rootmodel = BuildExeGraphFromSingleNode();
  auto graph = graph_2_rootmodel.exe_graph;
  ASSERT_NE(graph, nullptr);

  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();

  auto option = MultiThreadExecutorOption(kLeastThreadNumber);
  auto model_executor = ModelV2Executor::Create(graph, option, graph_2_rootmodel.root_model);
  ASSERT_NE(model_executor, nullptr);

  auto arg = model_executor->GetSubscribers().AddSubscriber<TestSubscriber>();
  ASSERT_NE(arg, nullptr);

  ASSERT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  auto outputs = FakeTensors({2}, 1);
  auto input0 =
      FakeValue<Tensor>(Tensor{{{256}, {256}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, 0});
  auto input1 =
      FakeValue<Tensor>(Tensor{{{256}, {256}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, 0});
  auto input2 = FakeValue<uint64_t>(0);

  ASSERT_EQ(model_executor->Execute({input2.value},
                                    std::vector<Tensor *>({input0.holder.get(), input1.holder.get()}).data(), 2,
                                    reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);

  EXPECT_FALSE(arg->start_events.empty());
  EXPECT_EQ(arg->start_events.size(), arg->success_events.size());
  EXPECT_TRUE(arg->failed_events.empty());
  EXPECT_TRUE(arg->unknown_events.empty());
  EXPECT_EQ(arg->model_start_events.size(), arg->model_end_events.size());
}
TEST_F(ExecutorCallbackUT, CallbackCalled_Ok_PriorityTopo) {
  EXPECT_TestCallbackCalled([](const ge::ExecuteGraphPtr &graph, const ge::GeRootModelPtr &root_model) {
    return BuildByType(graph, root_model, ExecutorType::kTopologicalPriority);
  });
}
TEST_F(ExecutorCallbackUT, BuiltInProfiling_Ok_PriorityTopo) {
  EXPECT_TestBuiltInProfiling([](const ge::ExecuteGraphPtr &graph, const ge::GeRootModelPtr &root_model) {
    return BuildByType(graph, root_model, ExecutorType::kTopologicalPriority);
  });
}

TEST_F(ExecutorCallbackUT, CallbackCalled_Ok_Topo) {
  EXPECT_TestCallbackCalled([](const ge::ExecuteGraphPtr &graph, const ge::GeRootModelPtr &root_model) {
    return BuildByType(graph, root_model, ExecutorType::kTopological);
  });
}
TEST_F(ExecutorCallbackUT, BuiltInProfiling_Ok_Topo) {
  EXPECT_TestBuiltInProfiling([](const ge::ExecuteGraphPtr &graph, const ge::GeRootModelPtr &root_model) {
    return BuildByType(graph, root_model, ExecutorType::kTopological);
  });
}

TEST_F(ExecutorCallbackUT, MultipleSubscribers_Ok) {
  auto graph_2_rootmodel = BuildExeGraphFromSingleNode();
  auto graph = graph_2_rootmodel.exe_graph;
  ASSERT_NE(graph, nullptr);

  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();

  auto model_executor = ModelV2Executor::Create(graph, graph_2_rootmodel.root_model);
  ASSERT_NE(model_executor, nullptr);

  auto arg1 = model_executor->GetSubscribers().AddSubscriber<TestSubscriber>();
  ASSERT_NE(arg1, nullptr);

  auto arg2 = model_executor->GetSubscribers().AddSubscriber<TestSubscriber>();
  ASSERT_NE(arg2, nullptr);

  ASSERT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  auto outputs = FakeTensors({2}, 1);
  auto input0 =
      FakeValue<Tensor>(Tensor{{{256}, {256}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, 0});
  auto input1 =
      FakeValue<Tensor>(Tensor{{{256}, {256}}, {ge::FORMAT_ND, ge::FORMAT_ND, {}}, kOnDeviceHbm, ge::DT_FLOAT16, 0});
  auto input2 = FakeValue<uint64_t>(0);

  ASSERT_EQ(
      model_executor->Execute({input2.value}, std::vector<Tensor *>({input0.holder.get(), input1.holder.get()}).data(),
                              2, reinterpret_cast<Tensor **>(outputs.GetAddrList()), outputs.size()),
      ge::GRAPH_SUCCESS);
  ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);

  EXPECT_FALSE(arg1->start_events.empty());
  EXPECT_EQ(arg1->start_events.size(), arg1->success_events.size());
  EXPECT_TRUE(arg1->failed_events.empty());
  EXPECT_TRUE(arg1->unknown_events.empty());

  EXPECT_EQ(arg2->start_events.size(), arg2->success_events.size());
  EXPECT_EQ(arg1->start_events.size(), arg2->start_events.size());
}
}  // namespace gert

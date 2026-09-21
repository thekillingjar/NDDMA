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
#include <gmock/gmock.h>
#include <vector>
#include <iostream>

#include "kernel/common_kernel_impl/infer_shape.h"
#include "core/executor_error_code.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "macro_utils/dt_public_scope.h"
#include "common/global_variables/diagnose_switch.h"
#include "depends/profiler/src/profiling_test_util.h"
#include "common/model_v2_executor_test_helper.h"
#include "runtime/subscriber/global_profiler.h"
#include "common/fake_node_helper.h"
#include "runtime/fast_v2/core/model_v2_executor_unittest.h"
#include "stub/gert_runtime_stub.h"
#include "runtime/model_v2_executor.h"
#include "hybrid/executor/hybrid_model_rt_v2_executor.h"
#include "lowering/model_converter.h"
#include "subscriber/profiler/cann_host_profiler.h"
#include "subscriber/profiler/cann_profiler_v2.h"
#include "subscriber/profiler/cann_memory_profiler.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "exe_graph/runtime/gert_mem_allocator.h"
#include "hybrid/executor/cann_tracing_profiler.h"
#include "common/fake_node_helper.h"
#include "subscriber/profiler/base_executor_profiler.h"
#include "macro_utils/dt_public_unscope.h"

using namespace std;
using namespace testing;

class CannTraceProfilerUT : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {
    ge::diagnoseSwitch::DisableProfiling();
  }
};

TEST_F(CannTraceProfilerUT, CannTracingProfiler_test_ok) {
  ge::diagnoseSwitch::EnableTrainingTrace();
  auto execution_data_holder = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[sizeof(ExecutionData)]);
  auto execution_data = reinterpret_cast<ExecutionData *>(execution_data_holder.get());
  auto infer_shape_node = gert::FakeNodeHelper::FakeNode("test1", "InferShape", 1);

  // build fake ai core launch node
  auto block_dims = 32;
  auto context = gert::KernelRunContextFaker()
      .NodeName("test1")
      .NodeType("test1")
      .KernelType("LaunchKernelWithHandle")
      .KernelName("LaunchKernelWithHandle")
      .KernelIONum(3, 1)
      .Build();

  size_t size = sizeof(Node) + sizeof(AsyncAnyValue *) * 4;
  Node *launch_node = (Node *)malloc(size);
  launch_node->node_id = 0;
  context.value_holder[2].Set(reinterpret_cast<void *>(block_dims), nullptr);
  memcpy(&launch_node->context, context.context, sizeof(KernelRunContext) + 4 * sizeof(AsyncAnyValue *));
  auto streams_holder = gert::ContinuousVector::Create<rtStream_t>(1U);
  auto streams = reinterpret_cast<gert::TypedContinuousVector<rtStream_t> *>(streams_holder.get());
  streams->SetSize(1U);
  AsyncAnyValue streams_value{streams, nullptr};
  std::vector<AsyncAnyValue *> inputs_value{&streams_value};
  execution_data->base_ed.input_values = inputs_value.data();
  // build fake execution data with 2 exe nodes
  execution_data->base_ed.node_num = 2;
  std::vector<Node *> nodes{&infer_shape_node.node, launch_node};
  execution_data->base_ed.nodes = nodes.data();
  execution_data->base_ed.input_num = static_cast<std::size_t>(gert::ExecuteArgIndex::kNum);

  auto model_executor = gert::BuildExecutorTraningTrace();
  gert::ModelV2ExecutorTestHelper::SetExecutionData(execution_data, gert::kMainExeGraph, model_executor.get());
  gert::SubscriberExtendInfo extend_info;
  extend_info.executor = model_executor.get();
  extend_info.node_names_to_attrs["test1"] = {true, false, 20002};
  auto profiler = model_executor->GetSubscribers().AddSubscriber<gert::CannTracingProfiler>(gert::kMainExeGraph,
                                                                                      extend_info);
  gert::CannTracingProfiler::OnExecuteEvent(0, profiler, kExecuteStart, launch_node, 0);
  gert::CannTracingProfiler::OnExecuteEvent(0, profiler, kExecuteEnd, launch_node, 0);
  for (const auto &attr : profiler->node_ids_to_attrs_) {
    EXPECT_EQ(profiler->node_ids_to_attrs_[attr.first].is_fp, true);
    EXPECT_EQ(profiler->node_ids_to_attrs_[attr.first].is_bp, false);
    EXPECT_EQ(profiler->node_ids_to_attrs_[attr.first].start_log_id, 20002);
  }

  extend_info.node_names_to_attrs["test1"] = {true, true, 20000};
  profiler = model_executor->GetSubscribers().AddSubscriber<gert::CannTracingProfiler>(gert::kMainExeGraph,
                                                                                 extend_info);
  gert::CannTracingProfiler::OnExecuteEvent(0, profiler, kExecuteStart, launch_node, 0);
  gert::CannTracingProfiler::OnExecuteEvent(0, profiler, kExecuteEnd, launch_node, 0);
  for (const auto &attr : profiler->node_ids_to_attrs_) {
    EXPECT_EQ(profiler->node_ids_to_attrs_[attr.first].is_fp, true);
    EXPECT_EQ(profiler->node_ids_to_attrs_[attr.first].is_bp, true);
    EXPECT_EQ(profiler->node_ids_to_attrs_[attr.first].start_log_id, 20000);
  }

  extend_info.node_names_to_attrs["test1"] = {true, true, -1};
  profiler = model_executor->GetSubscribers().AddSubscriber<gert::CannTracingProfiler>(gert::kMainExeGraph,
                                                                                 extend_info);
  gert::CannTracingProfiler::OnExecuteEvent(0, profiler, kExecuteStart, launch_node, 0);
  gert::CannTracingProfiler::OnExecuteEvent(0, profiler, kExecuteEnd, launch_node, 0);
  for (const auto &attr : profiler->node_ids_to_attrs_) {
    EXPECT_EQ(profiler->node_ids_to_attrs_[attr.first].is_fp, true);
    EXPECT_EQ(profiler->node_ids_to_attrs_[attr.first].is_bp, true);
    EXPECT_EQ(profiler->node_ids_to_attrs_[attr.first].start_log_id, -1);
  }
  // cover profiler nullptr
  gert::CannTracingProfiler::OnExecuteEvent(0, nullptr, kExecuteStart, launch_node, 0);
  free(launch_node);
}

TEST_F(CannTraceProfilerUT, CannTracingProfiler_test_execution_data_null) {
  ge::diagnoseSwitch::EnableTrainingTrace();
  gert::ModelV2Executor model_executor;
  gert::SubscriberExtendInfo extend_info;
  extend_info.executor = &model_executor;
  extend_info.node_names_to_attrs["test1"] = {true, false, 20002};
  auto profiler = model_executor.GetSubscribers().AddSubscriber<gert::CannTracingProfiler>(gert::kMainExeGraph,
                                                                                            extend_info);
  EXPECT_NE(profiler, nullptr);
}

TEST_F(CannTraceProfilerUT, CannTracingProfiler_test_extend_info_null) {
  ge::diagnoseSwitch::EnableTrainingTrace();
  auto execution_data_holder = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[sizeof(ExecutionData)]);
  auto execution_data = reinterpret_cast<ExecutionData *>(execution_data_holder.get());
  auto infer_shape_node = gert::FakeNodeHelper::FakeNode("test1", "InferShape", 1);

  std::unique_ptr<Node> launch_node = std::make_unique<Node>();
  launch_node->node_id = 0;
  launch_node->context.kernel_extend_info = nullptr;
  // build fake execution data with 2 exe nodes
  execution_data->base_ed.node_num = 2;
  std::vector<Node *> nodes{&infer_shape_node.node, launch_node.get()};
  execution_data->base_ed.nodes = nodes.data();
  execution_data->base_ed.input_num = static_cast<std::size_t>(gert::ExecuteArgIndex::kNum);

  auto model_executor = gert::BuildExecutorTraningTrace();
  gert::ModelV2ExecutorTestHelper::SetExecutionData(execution_data, gert::kMainExeGraph, model_executor.get());
  gert::SubscriberExtendInfo extend_info;
  extend_info.executor = model_executor.get();
  extend_info.node_names_to_attrs["test1"] = {true, false, 20002};
  auto profiler = model_executor->GetSubscribers().AddSubscriber<gert::CannTracingProfiler>(gert::kMainExeGraph,
                                                                                            extend_info);
  EXPECT_NE(profiler, nullptr);
}

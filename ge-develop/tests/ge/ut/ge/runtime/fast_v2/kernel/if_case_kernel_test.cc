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

#include "faker/kernel_run_context_facker.h"
#include "register/kernel_registry.h"
#include "common/ge_inner_attrs.h"
#include "core/execution_data.h"
#include "graph/compute_graph.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/utils/tensor_utils.h"
#include "faker/fake_value.h"
#include "kernel/memory/host_mem_allocator.h"
#include "core/utils/tensor_utils.h"
#include "kernel/control_kernel/control_kernel.h"
#include "graph/utils/execute_graph_utils.h"

namespace gert {
using namespace kernel;
namespace {
class WatcherHolder {
 public:
  Watcher *Get() {
    return reinterpret_cast<Watcher *>(watcher_buffer.data());
  }
  std::vector<uint8_t> watcher_buffer;
};
class WatcherFaker {
 public:
  WatcherFaker &NodeIds(std::vector<NodeIdentity> node_ids) {
    node_ids_ = std::move(node_ids);
    return *this;
  }
  WatcherHolder Build() const {
    WatcherHolder wh;
    wh.watcher_buffer.resize(sizeof(Watcher) + sizeof(NodeIdentity) * node_ids_.size());
    auto watcher = wh.Get();
    watcher->watch_num = node_ids_.size();
    memcpy(watcher->node_ids, node_ids_.data(), sizeof(NodeIdentity) * node_ids_.size());
    return wh;
  }

 private:
  std::vector<NodeIdentity> node_ids_;
};
}  // namespace

class IfOrCaseKernelUT : public testing::Test {
 public:
};

TEST_F(IfOrCaseKernelUT, GenIndexForIf_ReturnTrue_WhenNotScalar) {
  auto tensor = TensorFaker().Shape({1}).Build();
  auto context_holder = KernelRunContextFaker().KernelIONum(1, 1).Inputs({tensor.GetTensor()}).Build();
  auto context = context_holder.GetContext<KernelContext>();
  ASSERT_NE(context, nullptr);
  ASSERT_EQ(GenIndexForIf(context_holder.GetContext<KernelContext>()), ge::GRAPH_SUCCESS);
  auto branch_index = context->GetOutputPointer<uint32_t>(0);
  ASSERT_NE(branch_index, nullptr);
  ASSERT_EQ(*branch_index, 0);
}
TEST_F(IfOrCaseKernelUT, GenIndexForIf_ReturnTrue_WhenZeroLenTensor) {
  auto tensor = TensorFaker().Shape({1, 0}).Build();
  auto context_holder = KernelRunContextFaker().KernelIONum(1, 1).Inputs({tensor.GetTensor()}).Build();
  auto context = context_holder.GetContext<KernelContext>();
  ASSERT_NE(context, nullptr);
  ASSERT_EQ(GenIndexForIf(context_holder.GetContext<KernelContext>()), ge::GRAPH_SUCCESS);
  auto branch_index = context->GetOutputPointer<uint32_t>(0);
  ASSERT_NE(branch_index, nullptr);
  ASSERT_EQ(*branch_index, 1);
}
TEST_F(IfOrCaseKernelUT, GenIndexForIf_ReturnTrue_WhenInt1) {
  auto tensor = TensorFaker().Shape({}).DataType(ge::DT_INT32).Value<int32_t>({1}).Placement(kOnHost).Build();
  auto context_holder = KernelRunContextFaker().KernelIONum(1, 1).Inputs({tensor.GetTensor()}).Build();
  auto context = context_holder.GetContext<KernelContext>();
  ASSERT_NE(context, nullptr);
  ASSERT_EQ(GenIndexForIf(context_holder.GetContext<KernelContext>()), ge::GRAPH_SUCCESS);
  auto branch_index = context->GetOutputPointer<uint32_t>(0);
  ASSERT_NE(branch_index, nullptr);
  ASSERT_EQ(*branch_index, 0);
}
TEST_F(IfOrCaseKernelUT, GenIndexForIf_ReturnFalse_WhenInt0) {
  auto tensor = TensorFaker().Shape({}).DataType(ge::DT_INT32).Value<int32_t>({0}).Placement(kOnHost).Build();
  auto context_holder = KernelRunContextFaker().KernelIONum(1, 1).Inputs({tensor.GetTensor()}).Build();
  auto context = context_holder.GetContext<KernelContext>();
  ASSERT_NE(context, nullptr);
  ASSERT_EQ(GenIndexForIf(context_holder.GetContext<KernelContext>()), ge::GRAPH_SUCCESS);
  auto branch_index = context->GetOutputPointer<uint32_t>(0);
  ASSERT_NE(branch_index, nullptr);
  ASSERT_EQ(*branch_index, 1);
}
TEST_F(IfOrCaseKernelUT, GenIndexForIf_ReturnFalse_WhenFloat0) {
  auto tensor = TensorFaker().Shape({}).DataType(ge::DT_FLOAT).Value<float>({0.0}).Placement(kOnHost).Build();
  auto context_holder = KernelRunContextFaker().KernelIONum(1, 1).Inputs({tensor.GetTensor()}).Build();
  auto context = context_holder.GetContext<KernelContext>();
  ASSERT_NE(context, nullptr);
  ASSERT_EQ(GenIndexForIf(context_holder.GetContext<KernelContext>()), ge::GRAPH_SUCCESS);
  auto branch_index = context->GetOutputPointer<uint32_t>(0);
  ASSERT_NE(branch_index, nullptr);
  ASSERT_EQ(*branch_index, 1);
}
TEST_F(IfOrCaseKernelUT, GenIndexForIf_ReturnTrue_WhenBoolTrue) {
  auto tensor = TensorFaker().Shape({}).DataType(ge::DT_BOOL).Value<uint8_t>({true}).Placement(kOnHost).Build();
  auto context_holder = KernelRunContextFaker().KernelIONum(1, 1).Inputs({tensor.GetTensor()}).Build();
  auto context = context_holder.GetContext<KernelContext>();
  ASSERT_NE(context, nullptr);
  ASSERT_EQ(GenIndexForIf(context_holder.GetContext<KernelContext>()), ge::GRAPH_SUCCESS);
  auto branch_index = context->GetOutputPointer<uint32_t>(0);
  ASSERT_NE(branch_index, nullptr);
  ASSERT_EQ(*branch_index, 0);
}
TEST_F(IfOrCaseKernelUT, GenIndexForIf_Failed_WhenInputNullptr) {
  auto context_holder = KernelRunContextFaker().KernelIONum(1, 1).Inputs({nullptr}).Build();
  auto context = context_holder.GetContext<KernelContext>();
  ASSERT_NE(context, nullptr);
  ASSERT_NE(GenIndexForIf(context_holder.GetContext<KernelContext>()), ge::GRAPH_SUCCESS);
}
TEST_F(IfOrCaseKernelUT, GenIndexForIf_Failed_WhenTensorNullptr) {
  auto tensor = TensorFaker().Shape({}).DataType(ge::DT_INT32).Placement(kOnHost).Build();
  tensor.GetTensor()->SetData(TensorData());
  auto context_holder = KernelRunContextFaker().KernelIONum(1, 1).Inputs({tensor.GetTensor()}).Build();
  auto context = context_holder.GetContext<KernelContext>();
  ASSERT_NE(context, nullptr);
  ASSERT_NE(GenIndexForIf(context_holder.GetContext<KernelContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(IfOrCaseKernelUT, GenIndexForCase_SameOut_WhenBranchIndexLessThanBranchNum) {
  auto tensor = TensorFaker().Shape({}).DataType(ge::DT_INT32).Value<int32_t>({1}).Placement(kOnHost).Build();
  auto context_holder = KernelRunContextFaker().KernelIONum(2, 1).Inputs({tensor.GetTensor(), (void *)3}).Build();
  auto context = context_holder.GetContext<KernelContext>();
  ASSERT_NE(context, nullptr);
  ASSERT_EQ(GenIndexForCase(context_holder.GetContext<KernelContext>()), ge::GRAPH_SUCCESS);
  auto branch_index = context->GetOutputPointer<uint32_t>(0);
  ASSERT_NE(branch_index, nullptr);
  ASSERT_EQ(*branch_index, 1);
}
TEST_F(IfOrCaseKernelUT, GenIndexForCase_DefaultBranch_WhenNegative) {
  auto tensor = TensorFaker().Shape({}).DataType(ge::DT_INT32).Value<int32_t>({-1}).Placement(kOnHost).Build();
  auto context_holder = KernelRunContextFaker().KernelIONum(2, 1).Inputs({tensor.GetTensor(), (void *)3}).Build();
  auto context = context_holder.GetContext<KernelContext>();
  ASSERT_NE(context, nullptr);
  ASSERT_EQ(GenIndexForCase(context_holder.GetContext<KernelContext>()), ge::GRAPH_SUCCESS);
  auto branch_index = context->GetOutputPointer<uint32_t>(0);
  ASSERT_NE(branch_index, nullptr);
  ASSERT_EQ(*branch_index, 2);
}
TEST_F(IfOrCaseKernelUT, GenIndexForCase_DefaultBranch_WhenOutOfRange) {
  auto tensor = TensorFaker().Shape({}).DataType(ge::DT_INT32).Value<int32_t>({3}).Placement(kOnHost).Build();
  auto context_holder = KernelRunContextFaker().KernelIONum(2, 1).Inputs({tensor.GetTensor(), (void *)3}).Build();
  auto context = context_holder.GetContext<KernelContext>();
  ASSERT_NE(context, nullptr);
  ASSERT_EQ(GenIndexForCase(context_holder.GetContext<KernelContext>()), ge::GRAPH_SUCCESS);
  auto branch_index = context->GetOutputPointer<uint32_t>(0);
  ASSERT_NE(branch_index, nullptr);
  ASSERT_EQ(*branch_index, 2);
}
TEST_F(IfOrCaseKernelUT, GenIndexForCase_Failed_WhenZeroBranchNum) {
  auto tensor = TensorFaker().Shape({}).DataType(ge::DT_INT32).Value<int32_t>({0}).Placement(kOnHost).Build();
  auto context_holder = KernelRunContextFaker().KernelIONum(2, 1).Inputs({tensor.GetTensor(), (void *)0}).Build();
  auto context = context_holder.GetContext<KernelContext>();
  ASSERT_NE(context, nullptr);
  ASSERT_NE(GenIndexForCase(context_holder.GetContext<KernelContext>()), ge::GRAPH_SUCCESS);
}
TEST_F(IfOrCaseKernelUT, GenIndexForCase_Failed_WhenTensorNullptr) {
  auto context_holder = KernelRunContextFaker().KernelIONum(2, 1).Inputs({nullptr, (void *)3}).Build();
  auto context = context_holder.GetContext<KernelContext>();
  ASSERT_NE(context, nullptr);
  ASSERT_NE(GenIndexForIf(context_holder.GetContext<KernelContext>()), ge::GRAPH_SUCCESS);
}
TEST_F(IfOrCaseKernelUT, GenIndexForCase_Failed_WhenTensorDataDataNullptr) {
  auto tensor = TensorFaker().Shape({}).DataType(ge::DT_INT32).Placement(kOnHost).Build();
  tensor.GetTensor()->SetData(TensorData());
  auto context_holder = KernelRunContextFaker().KernelIONum(2, 1).Inputs({tensor.GetTensor(), (void *)3}).Build();
  auto context = context_holder.GetContext<KernelContext>();
  ASSERT_NE(context, nullptr);
  ASSERT_NE(GenIndexForIf(context_holder.GetContext<KernelContext>()), ge::GRAPH_SUCCESS);
}

TEST_F(IfOrCaseKernelUT, SwitchNotify_CreateOutputsFailed_NoWatchAttr) {
  const KernelRegistry::KernelFuncs *kernel = KernelRegistry::GetInstance().FindKernelFuncs("SwitchNotify");
  ASSERT_NE(kernel, nullptr);
  auto graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("input1", "Data")->NODE("switch_notify", "SwitchNotify")->NODE("signal", "WatcherPlaceholder"));
      CHAIN(NODE("switch_notify", "SwitchNotify")->EDGE(1, 0)->NODE("bp_true", "BranchPivot"));
      CHAIN(NODE("switch_notify", "SwitchNotify")->EDGE(2, 0)->NODE("bp_false", "BranchPivot"));
    };
    return ge::ToExecuteGraph(g);
  }();
  auto node = ge::ExecuteGraphUtils::FindNodeFromAllNodes(graph.get(), "switch_notify");
  ASSERT_NE(node, nullptr);

  auto run_context = BuildKernelRunContext(1, 3);
  auto kernel_context = run_context.GetContext<KernelContext>();

  ASSERT_NE(kernel->outputs_creator(node, kernel_context), ge::GRAPH_SUCCESS);
}
TEST_F(IfOrCaseKernelUT, SwitchNotify_CreateOutputsFailed_WatcherNumZero) {
  const KernelRegistry::KernelFuncs *kernel = KernelRegistry::GetInstance().FindKernelFuncs("SwitchNotify");
  ASSERT_NE(kernel, nullptr);
  auto graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("input1", "Data")->NODE("switch_notify", "SwitchNotify")->NODE("signal", "WatcherPlaceholder"));
      CHAIN(NODE("switch_notify", "SwitchNotify")->EDGE(1, 0)->NODE("bp_true", "BranchPivot"));
      CHAIN(NODE("switch_notify", "SwitchNotify")->EDGE(2, 0)->NODE("bp_false", "BranchPivot"));
    };
    return ge::ToExecuteGraph(g);
  }();
  auto node = ge::ExecuteGraphUtils::FindNodeFromAllNodes(graph.get(), "switch_notify");
  ASSERT_NE(node, nullptr);
  auto watcher_holder = WatcherFaker().NodeIds({}).Build();
  auto watcher = watcher_holder.Get();
  ASSERT_NE(watcher, nullptr);
  ASSERT_TRUE(ge::AttrUtils::SetInt(node->GetOpDescBarePtr(), ge::kWatcherAddress, (int64_t)watcher));

  auto run_context = BuildKernelRunContext(1, 3);
  (void)run_context.GetContext<KernelContext>();
  ASSERT_NE(kernel->outputs_creator(node, run_context), ge::GRAPH_SUCCESS);
}
TEST_F(IfOrCaseKernelUT, SwitchNotify_CreateOutputsSuccess) {
  const KernelRegistry::KernelFuncs *kernel = KernelRegistry::GetInstance().FindKernelFuncs("SwitchNotify");
  ASSERT_NE(kernel, nullptr);
  auto graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("input1", "Data")->NODE("switch_notify", "SwitchNotify")->NODE("signal", "WatcherPlaceholder"));
      CHAIN(NODE("switch_notify", "SwitchNotify")->EDGE(1, 0)->NODE("bp_true", "BranchPivot"));
      CHAIN(NODE("switch_notify", "SwitchNotify")->EDGE(2, 0)->NODE("bp_false", "BranchPivot"));
    };
    return ge::ToExecuteGraph(g);
  }();
  auto node = ge::ExecuteGraphUtils::FindNodeFromAllNodes(graph.get(), "switch_notify");
  ASSERT_NE(node, nullptr);
  auto watcher_holder = WatcherFaker().NodeIds({0, 1, 2}).Build();
  auto watcher = watcher_holder.Get();
  ASSERT_NE(watcher, nullptr);
  ASSERT_TRUE(ge::AttrUtils::SetInt(node->GetOpDescBarePtr(), ge::kWatcherAddress, (int64_t)watcher));

  auto run_context = BuildKernelRunContext(1, 3);
  auto kernel_context = run_context.GetContext<KernelContext>();

  EXPECT_EQ(kernel->outputs_creator(node, kernel_context), ge::GRAPH_SUCCESS);
  auto signal = kernel_context->GetOutputPointer<NotifySignal>(0);
  EXPECT_NE(signal, nullptr);
  EXPECT_EQ(signal->receiver_num, 2);
  EXPECT_EQ(signal->receivers, &watcher->node_ids[1]);
  EXPECT_EQ(&signal->target_receiver, &watcher->node_ids[0]);
  EXPECT_EQ(watcher->watch_num, 1);

  run_context.FreeAll();
}
TEST_F(IfOrCaseKernelUT, SwitchNotify_SwitchCorrect_0) {
  const KernelRegistry::KernelFuncs *kernel = KernelRegistry::GetInstance().FindKernelFuncs("SwitchNotify");
  ASSERT_NE(kernel, nullptr);
  auto graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("input1", "Data")->NODE("switch_notify", "SwitchNotify")->NODE("signal", "WatcherPlaceholder"));
      CHAIN(NODE("switch_notify", "SwitchNotify")->EDGE(1, 0)->NODE("bp_true", "BranchPivot"));
      CHAIN(NODE("switch_notify", "SwitchNotify")->EDGE(2, 0)->NODE("bp_false", "BranchPivot"));
    };
    return ge::ToExecuteGraph(g);
  }();
  auto node = ge::ExecuteGraphUtils::FindNodeFromAllNodes(graph.get(), "switch_notify");
  ASSERT_NE(node, nullptr);
  auto watcher_holder = WatcherFaker().NodeIds({0, 1, 2}).Build();
  auto watcher = watcher_holder.Get();
  ASSERT_NE(watcher, nullptr);
  ASSERT_TRUE(ge::AttrUtils::SetInt(node->GetOpDescBarePtr(), ge::kWatcherAddress, (int64_t)watcher));

  auto run_context = KernelRunContextFaker().KernelIONum(1, 3).Inputs({(void *)0}).Build();
  auto kernel_context = run_context.GetContext<KernelContext>();

  EXPECT_EQ(kernel->outputs_creator(node, kernel_context), ge::GRAPH_SUCCESS);
  auto signal = kernel_context->GetOutputPointer<NotifySignal>(0);
  ASSERT_NE(signal, nullptr);

  EXPECT_EQ(kernel->run_func(kernel_context), ge::GRAPH_SUCCESS);
  EXPECT_EQ(signal->target_receiver, signal->receivers[0]);  // Cond true

  kernel_context->GetOutput(0)->Set(nullptr, nullptr);
}
TEST_F(IfOrCaseKernelUT, SwitchNotify_SwitchCorrect_1) {
  const KernelRegistry::KernelFuncs *kernel = KernelRegistry::GetInstance().FindKernelFuncs("SwitchNotify");
  ASSERT_NE(kernel, nullptr);
  auto graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("input1", "Data")->NODE("switch_notify", "SwitchNotify")->NODE("signal", "WatcherPlaceholder"));
      CHAIN(NODE("switch_notify", "SwitchNotify")->EDGE(1, 0)->NODE("bp_true", "BranchPivot"));
      CHAIN(NODE("switch_notify", "SwitchNotify")->EDGE(2, 0)->NODE("bp_false", "BranchPivot"));
    };
    return ge::ToExecuteGraph(g);
  }();
  auto node = ge::ExecuteGraphUtils::FindNodeFromAllNodes(graph.get(), "switch_notify");
  ASSERT_NE(node, nullptr);
  auto watcher_holder = WatcherFaker().NodeIds({0, 1, 2}).Build();
  auto watcher = watcher_holder.Get();
  ASSERT_NE(watcher, nullptr);
  ASSERT_TRUE(ge::AttrUtils::SetInt(node->GetOpDescBarePtr(), ge::kWatcherAddress, (int64_t)watcher));

  auto run_context = KernelRunContextFaker().KernelIONum(1, 3).Inputs({(void *)1}).Build();
  auto kernel_context = run_context.GetContext<KernelContext>();

  EXPECT_EQ(kernel->outputs_creator(node, kernel_context), ge::GRAPH_SUCCESS);
  auto signal = kernel_context->GetOutputPointer<NotifySignal>(0);
  ASSERT_NE(signal, nullptr);

  EXPECT_EQ(kernel->run_func(kernel_context), ge::GRAPH_SUCCESS);
  EXPECT_EQ(signal->target_receiver, signal->receivers[1]);  // Cond false

  kernel_context->GetOutput(0)->Set(nullptr, nullptr);
}
TEST_F(IfOrCaseKernelUT, SwitchNotify_Failed_2) {
  const KernelRegistry::KernelFuncs *kernel = KernelRegistry::GetInstance().FindKernelFuncs("SwitchNotify");
  ASSERT_NE(kernel, nullptr);
  auto graph = []() {
    DEF_GRAPH(g) {
      CHAIN(NODE("input1", "Data")->NODE("switch_notify", "SwitchNotify")->NODE("signal", "WatcherPlaceholder"));
      CHAIN(NODE("switch_notify", "SwitchNotify")->EDGE(1, 0)->NODE("bp_true", "BranchPivot"));
      CHAIN(NODE("switch_notify", "SwitchNotify")->EDGE(2, 0)->NODE("bp_false", "BranchPivot"));
    };
    return ge::ToExecuteGraph(g);
  }();
  auto node = ge::ExecuteGraphUtils::FindNodeFromAllNodes(graph.get(), "switch_notify");
  ASSERT_NE(node, nullptr);
  auto watcher_holder = WatcherFaker().NodeIds({0, 1, 2}).Build();
  auto watcher = watcher_holder.Get();
  ASSERT_NE(watcher, nullptr);
  ASSERT_TRUE(ge::AttrUtils::SetInt(node->GetOpDescBarePtr(), ge::kWatcherAddress, (int64_t)watcher));

  auto run_context = KernelRunContextFaker().KernelIONum(1, 3).Inputs({(void *)2}).Build();
  auto kernel_context = run_context.GetContext<KernelContext>();

  EXPECT_EQ(kernel->outputs_creator(node, kernel_context), ge::GRAPH_SUCCESS);
  EXPECT_NE(kernel->run_func(kernel_context), ge::GRAPH_SUCCESS);

  kernel_context->GetOutput(0)->Set(nullptr, nullptr);
}


TEST_F(IfOrCaseKernelUT, GenCondForWhile_bool) {
  const KernelRegistry::KernelFuncs *kernel = KernelRegistry::GetInstance().FindKernelFuncs("GenCondForWhile");
  ASSERT_NE(kernel, nullptr);

  bool value = true;
  auto aligned_size = sizeof(value);
  void *block = malloc(aligned_size);
  memcpy_s(block, aligned_size, &value, aligned_size);
  gert::GertTensorData tensor_data = {block, aligned_size, gert::TensorPlacement::kOnHost, -1};
  StorageShape shape({}, {});
  ge::DataType dt = ge::DT_BOOL;
  auto context_holder =
      KernelRunContextFaker().KernelIONum(3, 1).Inputs({&tensor_data, &shape, (void *)dt}).Build();
  auto context = context_holder.GetContext<KernelContext>();
  ASSERT_NE(context, nullptr);

  EXPECT_EQ(kernel->run_func(context), ge::GRAPH_SUCCESS);
  auto out = context->GetOutputPointer<bool>(0);
  ASSERT_NE(out, nullptr);
  ASSERT_EQ(*out, true);
  free(block);
}

TEST_F(IfOrCaseKernelUT, GenCondForWhile_int32) {
  const KernelRegistry::KernelFuncs *kernel = KernelRegistry::GetInstance().FindKernelFuncs("GenCondForWhile");
  ASSERT_NE(kernel, nullptr);

  int32_t value = 57;
  auto aligned_size = sizeof(value);
  void *block = malloc(aligned_size);
  memcpy_s(block, aligned_size, &value, aligned_size);
  gert::GertTensorData tensor_data = {block, aligned_size, gert::TensorPlacement::kOnHost, -1};
  StorageShape shape({}, {});
  ge::DataType dt = ge::DT_INT32;
  auto context_holder =
      KernelRunContextFaker().KernelIONum(3, 1).Inputs({&tensor_data, &shape, (void *)dt}).Build();
  auto context = context_holder.GetContext<KernelContext>();

  ASSERT_NE(context->GetInputPointer<TensorData>(0U), nullptr);
  ASSERT_NE(context->GetInputPointer<TensorData>(0U)->GetAddr(), nullptr);
  std::cout << *reinterpret_cast<bool *>(context->GetInputPointer<TensorData>(0U)->GetAddr()) << std::endl;
  ASSERT_NE(context, nullptr);

  EXPECT_EQ(kernel->run_func(context), ge::GRAPH_SUCCESS);
  auto out = context->GetOutputPointer<bool>(0);
  ASSERT_NE(out, nullptr);
  ASSERT_EQ(*out, true);
  free(block);
}

TEST_F(IfOrCaseKernelUT, GenCondForWhile_NonScalar) {
  const KernelRegistry::KernelFuncs *kernel = KernelRegistry::GetInstance().FindKernelFuncs("GenCondForWhile");
  ASSERT_NE(kernel, nullptr);

  bool value = false;
  auto aligned_size = sizeof(value);
  void *block = malloc(aligned_size);
  memcpy_s(block, aligned_size, &value, aligned_size);
  gert::GertTensorData tensor_data = {block, aligned_size, gert::TensorPlacement::kOnHost, -1};
  StorageShape shape({1, 1}, {1, 1});
  ge::DataType dt = ge::DT_BOOL;
  auto context_holder =
      KernelRunContextFaker().KernelIONum(3, 1).Inputs({&tensor_data, &shape, (void *)dt}).Build();
  auto context = context_holder.GetContext<KernelContext>();

  ASSERT_NE(context->GetInputPointer<TensorData>(0U), nullptr);
  ASSERT_NE(context->GetInputPointer<TensorData>(0U)->GetAddr(), nullptr);
  std::cout << *reinterpret_cast<bool *>(context->GetInputPointer<TensorData>(0U)->GetAddr()) << std::endl;
  ASSERT_NE(context, nullptr);

  EXPECT_EQ(kernel->run_func(context), ge::GRAPH_SUCCESS);
  auto out = context->GetOutputPointer<bool>(0);
  ASSERT_NE(out, nullptr);
  ASSERT_EQ(*out, true);
  free(block);
}

TEST_F(IfOrCaseKernelUT, GenCondForWhile_fail) {
  const KernelRegistry::KernelFuncs *kernel = KernelRegistry::GetInstance().FindKernelFuncs("GenCondForWhile");
  ASSERT_NE(kernel, nullptr);

  // dtype not support
  gert::GertTensorData tensor_data(0, gert::TensorPlacement::kOnHost, -1, nullptr);
  StorageShape shape({}, {});
  ge::DataType dt = ge::DT_MAX;
  auto context_holder = KernelRunContextFaker().KernelIONum(3, 1).Inputs({&tensor_data, &shape, (void *)dt}).Build();
  auto context = context_holder.GetContext<KernelContext>();
  ASSERT_NE(context, nullptr);
  EXPECT_NE(kernel->run_func(context), ge::GRAPH_SUCCESS);

  // tensor data is null
  ge::DataType dt1 = ge::DT_BOOL;
  auto context_holder1 = KernelRunContextFaker().KernelIONum(3, 1).Inputs({&tensor_data, &shape, (void *)dt1}).Build();
  auto context1 = context_holder1.GetContext<KernelContext>();
  ASSERT_NE(context1, nullptr);
  EXPECT_NE(kernel->run_func(context1), ge::GRAPH_SUCCESS);
}
}  // namespace gert
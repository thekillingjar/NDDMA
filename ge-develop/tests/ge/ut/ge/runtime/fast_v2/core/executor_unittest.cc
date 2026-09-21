/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "core/builder/graph_executor_builder.h"
#include <gtest/gtest.h>
#include "exe_graph/runtime/extended_kernel_context.h"

#include "common/bg_test.h"
#include "common/exe_graph.h"
#include "faker/exe_graph_model_level_data_faker.h"
#include "stub/gert_runtime_stub.h"
#include "core/executor_error_code.h"

namespace gert {
namespace {
std::vector<std::string> g_kernel_trace;
UINT32 FooImpl(gert::KernelContext *context) {
  auto result = 2 * *context->GetInputValue<int64_t *>(0);
  auto output = context->GetOutput(0)->GetValue<int64_t *>();
  *output = result;
  g_kernel_trace.emplace_back(reinterpret_cast<ExtendedKernelContext *>(context)->GetKernelName());
  return kStatusSuccess;
}
UINT32 FooFailImpl(gert::KernelContext *context) {
  return kStatusFailed;
}
bool g_foo_exe_flag = true;
UINT32 AddSuccOrFailImpl(gert::KernelContext *context) {
  if (g_foo_exe_flag) {
    return FooImpl(context);
  } else {
    return FooFailImpl(context);
  }
}
UINT32 AddImpl(gert::KernelContext *context) {
  auto result = *context->GetInputValue<int64_t *>(0) + *context->GetInputValue<int64_t *>(1);
  auto output = context->GetOutput(0)->GetValue<int64_t *>();
  *output = result;
  g_kernel_trace.emplace_back(reinterpret_cast<ExtendedKernelContext *>(context)->GetKernelName());
  return kStatusSuccess;
}
UINT32 MoveToOutputImpl(gert::KernelContext *context) {
  *context->MutableInput(1)->GetValue<int64_t *>() = *context->GetInputValue<int64_t *>(0);
  g_kernel_trace.emplace_back(reinterpret_cast<ExtendedKernelContext *>(context)->GetKernelName());
  return kStatusSuccess;
}
UINT32 OutputCreator(const ge::FastNode *, KernelContext *context) {
  context->GetOutput(0)->SetWithDefaultDeleter(new int64_t);
  return kStatusSuccess;
}
void FakeSubscriber(int type, void *arg, ExecutorEvent event, const void *node, KernelStatus result) {
  (void)type;
  (void)arg;
  (void)event;
  (void)node;
  (void)result;
}
}  // namespace
/**
 * 本UT测试所有graph executor（虽然executor分了不同的策略，但是对于执行行为的观察大部分是一样的）
 */
class ExecutorUT : public bg::BgTestAutoCreate3StageFrame {
 public:
  /*
   * data0
   */
  void TestEmptyGraph(ExecutorType et) {
    auto data = bg::ValueHolder::CreateFeed(0);
    bg::ValueHolder::PopGraphFrame({data}, {}, "NetOutput");
    auto root_frame = bg::ValueHolder::PopGraphFrame();
    auto exe_graph = bg::ExeGraph(root_frame->GetExecuteGraph());

    auto mld = ExeGraphModelLevelDataFaker(exe_graph.GetMainGraph()).Build();
    GertRuntimeStub stub;
    stub.GetKernelStub().AllKernelRegisteredAndSuccess();

    ExeGraphExecutor executor;
    ASSERT_EQ(GraphExecutorBuilder(mld.GetModelLevelData(), exe_graph.GetMainGraph(), &mld.symbols_to_value).Build(et, executor),
              ge::GRAPH_SUCCESS);
    ASSERT_EQ(executor.Load(), ge::GRAPH_SUCCESS);
    ASSERT_EQ(executor.Execute(), ge::GRAPH_SUCCESS);
  }

  /*
   * NetOutput
   *   |
   *  foo
   *   |
   * data0
   */
  void TestOneNodeGraph(ExecutorType et) {
    auto data = bg::ValueHolder::CreateFeed(0);
    auto foo = bg::ValueHolder::CreateSingleDataOutput("Foo", {data});
    bg::ValueHolder::PopGraphFrame({foo}, {}, "NetOutput");
    auto root_frame = bg::ValueHolder::PopGraphFrame();
    auto exe_graph = bg::ExeGraph(root_frame->GetExecuteGraph());

    auto mld = ExeGraphModelLevelDataFaker(exe_graph.GetMainGraph()).Build();
    GertRuntimeStub stub;
    stub.GetKernelStub().SetUp("Foo", FooImpl);

    ExeGraphExecutor executor;
    ASSERT_EQ(GraphExecutorBuilder(mld.GetModelLevelData(), exe_graph.GetMainGraph(), &mld.symbols_to_value).Build(et, executor),
              ge::GRAPH_SUCCESS);
    ASSERT_EQ(executor.Load(), ge::GRAPH_SUCCESS);

    int64_t input = 1024;
    void *inputs[1] = {&input};
    int64_t output = 0;
    void *outputs[1] = {&output};
    ASSERT_EQ(executor.SpecifyInputs(inputs, 0, 1), ge::GRAPH_SUCCESS);
    ASSERT_EQ(executor.SpecifyOutputs(outputs, 1), ge::GRAPH_SUCCESS);
    ASSERT_EQ(executor.Execute(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(output, 2048);

    input = 100;
    ASSERT_EQ(executor.Execute(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(output, 200);
  }
  /*
   *
   *                 NetOutput
   *                 /        \
   *   MoveToOutput   <--- OutputData
   *       |
   *    foo1,2,3  -- priority 1,2,3
   *       |
   *    add1,2,3  -- priority 1,2,3
   *     /    \
   * data0    data1
   */
  bg::ExeGraph BuildGraph(std::vector<std::string> *names = nullptr) {
    auto data0 = bg::ValueHolder::CreateFeed(0);
    ge::AttrUtils::SetInt(data0->GetFastNode()->GetOpDescBarePtr(), "priority", 1);
    auto data1 = bg::ValueHolder::CreateFeed(1);
    ge::AttrUtils::SetInt(data1->GetFastNode()->GetOpDescBarePtr(), "priority", 1);
    std::vector<bg::ValueHolderPtr> add_out;
    for (int32_t i = 0; i < 3; ++i) {
      auto holder = bg::ValueHolder::CreateSingleDataOutput("Add", {data0, data1});
      ge::AttrUtils::SetInt(holder->GetFastNode()->GetOpDescBarePtr(), "priority", i + 1);
      add_out.emplace_back(holder);
    }
    std::vector<bg::ValueHolderPtr> foo_out;
    for (int32_t i = 0; i < 3; ++i) {
      auto holder = bg::ValueHolder::CreateSingleDataOutput("Foo", {add_out[i]});
      ge::AttrUtils::SetInt(holder->GetFastNode()->GetOpDescBarePtr(), "priority", i + 1);
      foo_out.emplace_back(holder);
    }
    auto output_data = bg::ValueHolder::CreateDataOutput("OutputData", {}, 3);
    std::vector<bg::ValueHolderPtr> move_out;
    for (int32_t i = 0; i < 3; ++i) {
      auto holder = bg::ValueHolder::CreateVoid<bg::ValueHolder>("MoveToOutput", {foo_out[i], output_data[i]});
      ge::AttrUtils::SetInt(output_data[i]->GetFastNode()->GetOpDescBarePtr(), "priority", 1);
      ge::AttrUtils::SetInt(holder->GetFastNode()->GetOpDescBarePtr(), "priority", 1);  // 故意设置为1，也不会抢跑
      move_out.emplace_back(holder);
    }
    bg::ValueHolder::PopGraphFrame(output_data, {}, "NetOutput");
    auto root_frame = bg::ValueHolder::PopGraphFrame();

    if (names != nullptr) {
      for (int32_t i = 0; i < 3; ++i) {
        names->push_back(add_out[i]->GetFastNode()->GetName());
        names->push_back(foo_out[i]->GetFastNode()->GetName());
        names->push_back(move_out[i]->GetFastNode()->GetName());
      }
    }

    return bg::ExeGraph(root_frame->GetExecuteGraph());
  }

  void TestMultipleNode(ExecutorType et) {
    auto exe_graph = BuildGraph();
    auto mld = ExeGraphModelLevelDataFaker(exe_graph.GetMainGraph()).Build();
    GertRuntimeStub stub;
    stub.GetKernelStub().SetUp("Foo", {FooImpl, OutputCreator, nullptr, nullptr});
    stub.GetKernelStub().SetUp("Add", {AddImpl, OutputCreator, nullptr, nullptr});
    stub.GetKernelStub().SetUp("MoveToOutput", {MoveToOutputImpl, OutputCreator, nullptr, nullptr});

    ExeGraphExecutor executor;
    ASSERT_EQ(GraphExecutorBuilder(mld.GetModelLevelData(), exe_graph.GetMainGraph(), &mld.symbols_to_value).Build(et, executor),
              ge::GRAPH_SUCCESS);

    int64_t input1 = 1000, input2 = 24;
    void *inputs[] = {&input1, &input2};
    int64_t output1, output2, output3;
    void *outputs[] = {&output1, &output2, &output3};
    ASSERT_EQ(executor.SpecifyInputs(inputs, 0, 2), ge::GRAPH_SUCCESS);
    ASSERT_EQ(executor.SpecifyOutputs(outputs, 3), ge::GRAPH_SUCCESS);
    ASSERT_EQ(executor.Execute(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(output1, 2048);
    EXPECT_EQ(output2, 2048);
    EXPECT_EQ(output3, 2048);

    input1 = 500, input2 = 12;
    ASSERT_EQ(executor.Execute(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(output1, 1024);
    EXPECT_EQ(output2, 1024);
    EXPECT_EQ(output3, 1024);
  }
  void TestMultipleNodePriority(ExecutorType et) {
    std::vector<std::string> expect_names;
    auto exe_graph = BuildGraph(&expect_names);
    auto mld = ExeGraphModelLevelDataFaker(exe_graph.GetMainGraph()).Build();
    GertRuntimeStub stub;
    stub.GetKernelStub().SetUp("Foo", {FooImpl, OutputCreator, nullptr, nullptr});
    stub.GetKernelStub().SetUp("Add", {AddImpl, OutputCreator, nullptr, nullptr});
    stub.GetKernelStub().SetUp("MoveToOutput", {MoveToOutputImpl, OutputCreator, nullptr, nullptr});

    ExeGraphExecutor executor;
    ASSERT_EQ(GraphExecutorBuilder(mld.GetModelLevelData(), exe_graph.GetMainGraph(), &mld.symbols_to_value).Build(et, executor),
              ge::GRAPH_SUCCESS);

    int64_t input1 = 1000, input2 = 24;
    void *inputs[] = {&input1, &input2};
    int64_t output1, output2, output3;
    void *outputs[] = {&output1, &output2, &output3};
    ASSERT_EQ(executor.SpecifyInputs(inputs, 0, 2), ge::GRAPH_SUCCESS);
    ASSERT_EQ(executor.SpecifyOutputs(outputs, 3), ge::GRAPH_SUCCESS);

    g_kernel_trace.clear();
    ASSERT_EQ(executor.Execute(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(g_kernel_trace, expect_names);

    g_kernel_trace.clear();
    ASSERT_EQ(executor.Execute(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(g_kernel_trace, expect_names);
  }
  /*
   *
   *                       NetOutput
   *                       /        \
   *             MoveToOutput   <--- OutputData
   *                   |
   *                add1,2,3  -- priority 1,2,3
   *               /      |
   *     foo_z1,2,3       |   -- priority 1,2,3
   *              |       |
   *     foo_y1,2,3       |   -- priority 1,2,3
   *              \       |
   *               foo_x1,2,3  -- priority 1,2,3
   *                   | 
   *                 data0
   */
  bg::ExeGraph BuildGraph2(std::vector<std::string> *names = nullptr,
                           std::vector<std::string> *priority_names = nullptr) {
    auto data0 = bg::ValueHolder::CreateFeed(0);
    ge::AttrUtils::SetInt(data0->GetFastNode()->GetOpDescBarePtr(), "priority", 1);
    constexpr int32_t node_cnt = 3;
    std::vector<bg::ValueHolderPtr> foo_x_out;
    for (int32_t i = 0; i < node_cnt; ++i) {
      auto holder = bg::ValueHolder::CreateSingleDataOutput("Foo_x", {data0});
      ge::AttrUtils::SetInt(holder->GetFastNode()->GetOpDescBarePtr(), "priority", i + 1);
      foo_x_out.emplace_back(holder);
      if (names != nullptr) {
        names->emplace_back(holder->GetFastNode()->GetName());
      }
    }
    std::vector<bg::ValueHolderPtr> foo_y_out;
    for (int32_t i = 0; i < node_cnt; ++i) {
      auto holder = bg::ValueHolder::CreateSingleDataOutput("Foo_y", {foo_x_out[i]});
      ge::AttrUtils::SetInt(holder->GetFastNode()->GetOpDescBarePtr(), "priority", i + 1);
      foo_y_out.emplace_back(holder);
      if (names != nullptr) {
        names->emplace_back(holder->GetFastNode()->GetName());
      }
    }
    std::vector<bg::ValueHolderPtr> foo_z_out;
    for (int32_t i = 0; i < node_cnt; ++i) {
      auto holder = bg::ValueHolder::CreateSingleDataOutput("Foo_z", {foo_y_out[i]});
      ge::AttrUtils::SetInt(holder->GetFastNode()->GetOpDescBarePtr(), "priority", i + 1);
      foo_z_out.emplace_back(holder);
      if (names != nullptr) {
        names->emplace_back(holder->GetFastNode()->GetName());
      }
    }
    std::vector<bg::ValueHolderPtr> add_out;
    for (int32_t i = 0; i < node_cnt; ++i) {
      auto holder = bg::ValueHolder::CreateSingleDataOutput("Add", {foo_x_out[i], foo_z_out[i]});
      ge::AttrUtils::SetInt(holder->GetFastNode()->GetOpDescBarePtr(), "priority", i + 1);
      add_out.emplace_back(holder);
      if (names != nullptr) {
        names->emplace_back(holder->GetFastNode()->GetName());
      }
    }
    auto output_data = bg::ValueHolder::CreateDataOutput("OutputData", {}, node_cnt);
    std::vector<bg::ValueHolderPtr> move_out;
    for (int32_t i = 0; i < node_cnt; ++i) {
      auto holder = bg::ValueHolder::CreateVoid<bg::ValueHolder>("MoveToOutput", {add_out[i], output_data[i]});
      ge::AttrUtils::SetInt(output_data[i]->GetFastNode()->GetOpDescBarePtr(), "priority", 1);
      ge::AttrUtils::SetInt(holder->GetFastNode()->GetOpDescBarePtr(), "priority", 1);  // 故意设置为1，也不会抢跑
      move_out.emplace_back(holder);
      if (names != nullptr) {
        names->emplace_back(holder->GetFastNode()->GetName());
      }
    }
    bg::ValueHolder::PopGraphFrame(output_data, {}, "NetOutput");
    auto root_frame = bg::ValueHolder::PopGraphFrame();

    if (priority_names != nullptr) {
      for (int32_t i = 0; i < node_cnt; ++i) {
        priority_names->push_back(foo_x_out[i]->GetFastNode()->GetName());
        priority_names->push_back(foo_y_out[i]->GetFastNode()->GetName());
        priority_names->push_back(foo_z_out[i]->GetFastNode()->GetName());
        priority_names->push_back(add_out[i]->GetFastNode()->GetName());
        priority_names->push_back(move_out[i]->GetFastNode()->GetName());
      }
    }
    return bg::ExeGraph(root_frame->GetExecuteGraph());
   }
   
   void TestMultipleNodeFailedThenSuccess(ExecutorType et, bool with_callback = false) {
    std::vector<std::string> expect_names;
    std::vector<std::string> expect_priority_names;
    auto exe_graph = BuildGraph2(&expect_names, &expect_priority_names);
    ge::DumpGraph(exe_graph.GetMainGraph().get(), "TestMultipleNodeFailedThenSuccess");
    auto mld = ExeGraphModelLevelDataFaker(exe_graph.GetMainGraph()).Build();
    GertRuntimeStub stub;
    stub.GetKernelStub().SetUp("Foo_x", {FooImpl, OutputCreator, nullptr, nullptr});
    stub.GetKernelStub().SetUp("Foo_y", {AddSuccOrFailImpl, OutputCreator, nullptr, nullptr});
    stub.GetKernelStub().SetUp("Foo_z", {FooImpl, OutputCreator, nullptr, nullptr});
    stub.GetKernelStub().SetUp("Add", {AddImpl, OutputCreator, nullptr, nullptr});
    stub.GetKernelStub().SetUp("MoveToOutput", {MoveToOutputImpl, OutputCreator, nullptr, nullptr});

    ExeGraphExecutor executor;
    ASSERT_EQ(GraphExecutorBuilder(mld.GetModelLevelData(), exe_graph.GetMainGraph(), &mld.symbols_to_value).Build(et, executor),
              ge::GRAPH_SUCCESS);

    int64_t input1 = 1024;
    void *inputs[] = {&input1};
    int64_t output1, output2, output3;
    void *outputs[] = {&output1, &output2, &output3};
    ASSERT_EQ(executor.SpecifyInputs(inputs, 0, 1), ge::GRAPH_SUCCESS);
    ASSERT_EQ(executor.SpecifyOutputs(outputs, 3), ge::GRAPH_SUCCESS);

    // 此处校验执行失败后入度是否正常恢复，构造场景为：第一次执行失败，第二次执行成功
    // 观测点有两个：1）执行结果是否正常 2）执行的结点名称以及顺序是否符合预期
    g_foo_exe_flag = false;
    if (with_callback) {
      ExecutorSubscriber es = {FakeSubscriber, nullptr};
      ASSERT_EQ(executor.Execute(kMainExeGraph, &es), ge::GRAPH_FAILED);
    } else {
      ASSERT_EQ(executor.Execute(), ge::GRAPH_FAILED);
    }

    g_kernel_trace.clear();
    g_foo_exe_flag = true;
    if (with_callback) {
      ExecutorSubscriber es = {FakeSubscriber, nullptr};
      ASSERT_EQ(executor.Execute(kMainExeGraph, &es), ge::GRAPH_SUCCESS);
    } else {
      ASSERT_EQ(executor.Execute(), ge::GRAPH_SUCCESS);
    }
    EXPECT_EQ(output1, 10240);
    EXPECT_EQ(output2, 10240);
    EXPECT_EQ(output3, 10240);
    if (et == gert::ExecutorType::kTopological) {
      EXPECT_EQ(g_kernel_trace, expect_names);
    } else if (et == gert::ExecutorType::kSequentialPriority || et == gert::ExecutorType::kTopologicalPriority) {
      EXPECT_EQ(g_kernel_trace, expect_priority_names);
    }
  }
};
TEST_F(ExecutorUT, EmptyGraph_Ok_TopoPriority) {
  TestEmptyGraph(ExecutorType::kTopologicalPriority);
}
TEST_F(ExecutorUT, EmptyGraph_Ok_Topo) {
  TestEmptyGraph(ExecutorType::kTopological);
}
TEST_F(ExecutorUT, EmptyGraph_Ok_SeqPriority) {
  TestEmptyGraph(ExecutorType::kSequentialPriority);
}
TEST_F(ExecutorUT, OneNodeGraph_Ok_TopoPriority) {
  TestOneNodeGraph(ExecutorType::kTopologicalPriority);
}
TEST_F(ExecutorUT, OneNodeGraph_Ok_Topo) {
  TestOneNodeGraph(ExecutorType::kTopological);
}
TEST_F(ExecutorUT, OneNodeGraph_Ok_SeqPriority) {
  TestOneNodeGraph(ExecutorType::kSequentialPriority);
}
TEST_F(ExecutorUT, MultiNodeGraph_Ok_TopoPriority) {
  TestMultipleNode(ExecutorType::kTopologicalPriority);
}
TEST_F(ExecutorUT, MultiNodeGraph_Ok_Topo) {
  TestMultipleNode(ExecutorType::kTopological);
}
TEST_F(ExecutorUT, MultiNodeGraph_Ok_SeqPriority) {
  TestMultipleNode(ExecutorType::kSequentialPriority);
}
TEST_F(ExecutorUT, MultipleNodes_ScheduleByPriority_SeqPriority) {
  TestMultipleNodePriority(ExecutorType::kSequentialPriority);
}
TEST_F(ExecutorUT, MultipleNodes_ScheduleByPriority_TopoPriority) {
  TestMultipleNodePriority(ExecutorType::kTopologicalPriority);
}
TEST_F(ExecutorUT, MultiNodeGraph_FailedThenSuccess_TopoPriority) {
  TestMultipleNodeFailedThenSuccess(ExecutorType::kTopologicalPriority);
}
TEST_F(ExecutorUT, MultiNodeGraph_FailedThenSuccess_Topo) {
  TestMultipleNodeFailedThenSuccess(ExecutorType::kTopological);
}
TEST_F(ExecutorUT, MultiNodeGraph_FailedThenSuccess_SeqPriority) {
  TestMultipleNodeFailedThenSuccess(ExecutorType::kSequentialPriority);
}
TEST_F(ExecutorUT, MultiNodeGraphWithCb_FailedThenSuccess_TopoPriority) {
  TestMultipleNodeFailedThenSuccess(ExecutorType::kTopologicalPriority, true);
}
TEST_F(ExecutorUT, MultiNodeGraphWithCb_FailedThenSuccess_Topo) {
  TestMultipleNodeFailedThenSuccess(ExecutorType::kTopological, true);
}
TEST_F(ExecutorUT, MultiNodeGraphWithCb_FailedThenSuccess_SeqPriority) {
  TestMultipleNodeFailedThenSuccess(ExecutorType::kSequentialPriority, true);
}
}  // namespace gert
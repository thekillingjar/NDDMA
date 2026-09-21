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
#include <mmpa/mmpa_api.h>
#include "core/execution_data.h"
#include "common/bg_test.h"
#include "core/model_v2_executor_unittest.h"
#include "stub/gert_runtime_stub.h"
#include "core/builder/model_v2_executor_builder.h"
#include "common/fake_node_helper.h"
#include "common/dump/dump_manager.h"
#include "common/dump/dump_properties.h"
#include "framework/runtime/model_v2_executor.h"
#include "framework/common/types.h"
#include "kernel/tensor_attr.h"
#include "graph/utils/graph_utils.h"
#include "common/share_graph.h"
#include "faker/fake_value.h"
#include <faker/aicpu_taskdef_faker.h>
#include "lowering/model_converter.h"
#include "core/executor_error_code.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "engine/ffts_plus/converter/ffts_plus_proto_transfer.h"
#include "engine/aicore/kernel/aicore_update_kernel.h"
#include "register/ffts_node_calculater_registry.h"
#include "engine/aicore/kernel/aicore_update_kernel.h"
#include "single_op/task/op_task.h"
#include "macro_utils/dt_public_scope.h"
#include "graph/load/model_manager/davinci_model.h"
#include "subscriber/dumper/executor_dumper.h"
#include "subscriber/dumper/host_executor_dumper.h"
#include "common/global_variables/diagnose_switch.h"
#include "stub/gert_runtime_stub.h"
#include "common/debug/ge_log.h"
#include "macro_utils/dt_public_unscope.h"

namespace gert {
namespace {
uint64_t FindFirstNonEmptyId(ExecutorDumper *dumper) {
  for (uint64_t i = 0U; i < dumper->kernel_idxes_to_dump_units_.size(); ++i) {
    if (!dumper->kernel_idxes_to_dump_units_[i].empty()) {
      return i;
    }
  }
  return std::numeric_limits<uint64_t>::max();
}
}  // namespace
class HostExecutorDumperUT : public bg::BgTest {
 protected:
  void SetUp() {
    GlobalDumper::GetInstance()->SetEnableFlags(0UL);
  }
  void TearDown() {
    GlobalDumper::GetInstance()->SetEnableFlags(0UL);
    ge::DumpManager::GetInstance().RemoveDumpProperties(ge::kInferSessionId);
  }
};

bool CheckLogExpected(std::vector<gert::OneLog> &logs, const std::string &expect_log) {
  for (auto &onelog : logs) {
    std::string content = onelog.content;
    std::cout << "test " << content << std::endl;
    if (content.find(expect_log) != std::string::npos) {
      return true;
    }
  }
  return false;
}

TEST_F(HostExecutorDumperUT, EnableHostDump_Ok) {
  auto exe_graph = std::make_shared<ge::ExecuteGraph>("");
  const auto &extend_info = ge::MakeShared<const SubscriberExtendInfo>(nullptr, exe_graph, nullptr, ge::ModelData{},
                                                                       nullptr, SymbolsToValue{}, 0, "", nullptr,
                                                                       std::unordered_map<std::string, TraceAttr>{});
  HostExecutorDumper dumper(extend_info);
  GlobalDumper::GetInstance()->SetEnableFlags(BuiltInSubscriberUtil::BuildEnableFlags<DumpType>({DumpType::kHostDump}));
  EXPECT_TRUE(dumper.IsEnable(DumpType::kHostDump));
  GlobalDumper::GetInstance()->SetEnableFlags(
      BuiltInSubscriberUtil::BuildEnableFlags<DumpType>({DumpType::kExceptionDump}));
  EXPECT_TRUE(dumper.IsEnable(DumpType::kExceptionDump));
  EXPECT_FALSE(dumper.IsEnable(DumpType::kHostDump));
}

TEST_F(HostExecutorDumperUT, HostDumperConstruct_Ok) {
  GertRuntimeStub stub;
  stub.GetKernelStub().AllKernelRegisteredAndSuccess();
  auto executor_2_exe_graph = BuildExecutorFromSingleNode();
  auto executor = std::move(executor_2_exe_graph.executor);
  ASSERT_NE(executor, nullptr);
  const auto &extend_info = ge::MakeShared<const SubscriberExtendInfo>(
      executor.get(), executor_2_exe_graph.exe_graph, nullptr, ge::ModelData{}, nullptr,
      SymbolsToValue{}, 0, "", nullptr, std::unordered_map<std::string, TraceAttr>{});
  HostExecutorDumper dumper(extend_info);
  EXPECT_NE(dumper.extend_info_->executor, nullptr);
  EXPECT_NE(dumper.extend_info_->exe_graph, nullptr);
}

TEST_F(HostExecutorDumperUT, IncreaseUpdataCount_Ok_AfterKernelDone) {
  GlobalDumper::GetInstance()->SetEnableFlags(
      gert::BuiltInSubscriberUtil::EnableBit<gert::DumpType>(gert::DumpType::kHostDump));
  auto executor = BuildExecutorFromSingleNodeForDump();
  ASSERT_NE(executor, nullptr);
  auto dumper =
      executor->GetSubscribers().MutableBuiltInSubscriber<HostExecutorDumper>(BuiltInSubscriberType::kHostDumper);
  ASSERT_NE(dumper, nullptr);
  uint64_t i = FindFirstNonEmptyId(dumper);
  auto fake_node = FakeNodeHelper::FakeNode("add1", "test", i);
  auto dump_unit = dumper->kernel_idxes_to_dump_units_[fake_node.node.node_id][0];
  dump_unit->total_update_count = 2;
  EXPECT_EQ(dump_unit->cur_update_count, 0);
  dumper->OnUpdateDumpUnitForHostDump(fake_node.node);
  EXPECT_EQ(dump_unit->cur_update_count, 1);
  dump_unit->Clear();
  EXPECT_EQ(dump_unit->cur_update_count, 0);
}

TEST_F(HostExecutorDumperUT, Dumper_spaceaddr_empty_ok) {
  GlobalDumper::GetInstance()->SetEnableFlags(
      gert::BuiltInSubscriberUtil::EnableBit<gert::DumpType>(gert::DumpType::kHostDump));
  auto executor = BuildExecutorFromSingleNodeForDump();
  ASSERT_NE(executor, nullptr);
  auto dumper =
      executor->GetSubscribers().MutableBuiltInSubscriber<HostExecutorDumper>(BuiltInSubscriberType::kHostDumper);
  ASSERT_NE(dumper, nullptr);
  auto fake_node = FakeNodeHelper::FakeNode("add1", "AicpuLaunchCCKernel", FindFirstNonEmptyId(dumper));
  auto dump_unit = dumper->kernel_idxes_to_dump_units_[fake_node.node.node_id][0];
  dumper->HostDataDump(&fake_node.node, kExecuteEnd);
  EXPECT_EQ(dump_unit->workspace_info.empty(), true);
  dump_unit->Clear();
}

TEST_F(HostExecutorDumperUT, SaveSessionId_Ok_LoadWithSession) {
  auto executor = BuildExecutorFromSingleNodeForDump();
  auto dumper =
      executor->GetSubscribers().MutableBuiltInSubscriber<HostExecutorDumper>(BuiltInSubscriberType::kHostDumper);
  RtSession rt_session(1);
  ModelLoadArg load_arg{&rt_session, {nullptr, 0}};
  EXPECT_EQ(executor->Load(nullptr, load_arg), ge::SUCCESS);
  EXPECT_EQ(dumper->GetSessionId(), 0);
  GlobalDumper::GetInstance()->SetEnableFlags(BuiltInSubscriberUtil::EnableBit<DumpType>(DumpType::kHostDump));
  HostExecutorDumper::OnExecuteEvent(0, dumper, kModelStart, nullptr, kStatusSuccess);
  GlobalDumper::GetInstance()->SetEnableFlags(0);
  EXPECT_EQ(dumper->GetSessionId(), 1);
  dumper = nullptr;
  HostExecutorDumper::OnExecuteEvent(0, dumper, kModelStart, nullptr, kStatusSuccess);
}

TEST_F(HostExecutorDumperUT, NoInit_WithDumperNull) {
  GertRuntimeStub runtime_stub;
  dlog_setlevel(GE_MODULE_NAME, DLOG_INFO, 0);
  HostExecutorDumper::OnExecuteEvent(0, nullptr, kModelStart, nullptr, kStatusSuccess);
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
  auto logs = runtime_stub.GetSlogStub().GetLogs();
  EXPECT_FALSE(CheckLogExpected(logs, "Start to init dumper."));
}

TEST_F(HostExecutorDumperUT, InitHostDumper_Ok_WithDumpStep) {
  auto executor = BuildExecutorFromSingleNodeForDump();
  auto dumper =
      executor->GetSubscribers().MutableBuiltInSubscriber<HostExecutorDumper>(BuiltInSubscriberType::kHostDumper);
  RtSession rt_session(2);
  ModelLoadArg load_arg{&rt_session, {nullptr, 0}};
  EXPECT_EQ(executor->Load(nullptr, load_arg), ge::SUCCESS);
  EXPECT_EQ(dumper->GetSessionId(), 0);
  GlobalDumper::GetInstance()->SetEnableFlags(BuiltInSubscriberUtil::EnableBit<DumpType>(DumpType::kHostDump));
  ge::DumpProperties dump_properties;
  const std::string dump_step = "0|2-4|8";
  dump_properties.AddPropertyValue("ALL_MODEL_NEED_DUMP_AND_IT_IS_NOT_A_MODEL_NAME", std::set<std::string>());
  dump_properties.SetDumpMode("all");
  dump_properties.SetDumpStep(dump_step);
  ge::DumpManager::GetInstance().AddDumpProperties(2, dump_properties);
  const auto dump_step_set = ge::DumpManager::GetInstance().GetDumpProperties(2).GetDumpStep();
  EXPECT_EQ(dump_step_set, dump_step);
  HostExecutorDumper::OnExecuteEvent(0, dumper, kModelStart, nullptr, kStatusSuccess);
  dlog_setlevel(GE_MODULE_NAME, DLOG_WARN, 0);
  HostExecutorDumper::OnExecuteEvent(0, dumper, kModelStart, nullptr, kStatusSuccess);
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
  GlobalDumper::GetInstance()->SetEnableFlags(0);
  EXPECT_EQ(dumper->GetSessionId(), 2);
  EXPECT_EQ(dumper->step_range_.size(), 1);
  EXPECT_EQ(dumper->step_set_.size(), 2);
  EXPECT_TRUE(dumper->IsInDumpStep(0, dump_step_set));
  EXPECT_TRUE(dumper->IsInDumpStep(2, dump_step_set));
  EXPECT_TRUE(dumper->IsInDumpStep(3, dump_step_set));
  EXPECT_TRUE(dumper->IsInDumpStep(4, dump_step_set));
  EXPECT_TRUE(dumper->IsInDumpStep(8, dump_step_set));
  EXPECT_FALSE(dumper->IsInDumpStep(6, dump_step_set));
  EXPECT_TRUE(dumper->IsInDumpStep(6, ""));
}

TEST_F(HostExecutorDumperUT, DoHostDataDump_WithInvalidPath) {
  GlobalDumper::GetInstance()->SetEnableFlags(
      gert::BuiltInSubscriberUtil::EnableBit<gert::DumpType>(gert::DumpType::kHostDump));
  auto executor = BuildExecutorFromSingleNodeForDump();
  ASSERT_NE(executor, nullptr);
  auto dumper =
      executor->GetSubscribers().MutableBuiltInSubscriber<HostExecutorDumper>(BuiltInSubscriberType::kHostDumper);
  ASSERT_NE(dumper, nullptr);
  StorageShape storage_shape{{2048}, {2048, 1}};
  auto total_size = 0UL;
  auto tensor_holder =
      Tensor::CreateFollowing(storage_shape.GetStorageShape().GetShapeSize(), ge::DT_FLOAT16, total_size);
  auto context_holder_1 =
      KernelRunContextFaker()
          .KernelIONum(1, 2)
          .Outputs({tensor_holder.get(), &reinterpret_cast<Tensor *>(tensor_holder.get())->MutableTensorData()})
          .Build();
  auto &node_add_dump_unit = dumper->node_names_to_dump_units_["add1"];
  node_add_dump_unit.output_addrs[0] = context_holder_1.GetContext<KernelContext>()->GetOutput(1);
  node_add_dump_unit.input_addrs[0] = context_holder_1.GetContext<KernelContext>()->GetOutput(1);
  node_add_dump_unit.input_addrs[1] = context_holder_1.GetContext<KernelContext>()->GetOutput(1);
  gert::StorageShape x2_shape = {{4, 8, 16, 1, 1}, {4, 8, 16, 1, 1, 1, 1}};
  gert::StorageShape x1_shape = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
  auto context_holder_2 = KernelRunContextFaker()
                              .Inputs({&x1_shape})
                              .Outputs({&x2_shape})
                              .KernelIONum(1, 1)
                              .NodeIoNum(1, 1)
                              .IrInstanceNum({1})
                              .Build();
  node_add_dump_unit.output_shapes[0] = context_holder_2.GetContext<KernelContext>()->GetOutput(0);
  node_add_dump_unit.input_shapes[0] = const_cast<Chain *>(context_holder_2.GetContext<KernelContext>()->GetInput(0));
  node_add_dump_unit.input_shapes[1] = const_cast<Chain *>(context_holder_2.GetContext<KernelContext>()->GetInput(0));
  ge::DumpManager::GetInstance().RemoveDumpProperties(ge::kInferSessionId);
  ge::DumpProperties dump_properties;
  dump_properties.AddPropertyValue("ALL_MODEL_NEED_DUMP_AND_IT_IS_NOT_A_MODEL_NAME", {"test"});
  dump_properties.SetDumpMode("all");
  std::string dump_path = "/";
  dump_properties.SetDumpPath(dump_path);
  const std::string dump_step = "0|2-4|8";
  dump_properties.SetDumpStep(dump_step);
  ge::DumpManager::GetInstance().AddDumpProperties(ge::kInferSessionId, dump_properties);
  HostExecutorDumper::OnExecuteEvent(0, dumper, kModelStart, nullptr, kStatusSuccess);
  auto op_desc = dumper->node_names_to_dump_units_["add1"].node->GetOpDesc();
  std::vector<int64_t> tvm_workspace_memory_type = {ge::AicpuWorkSpaceType::CUST_LOG};
  ge::AttrUtils::SetListInt(op_desc, ge::ATTR_NAME_AICPU_WORKSPACE_TYPE, tvm_workspace_memory_type);
  op_desc->SetWorkspaceBytes(vector<int64_t>{32});
  const auto properties = ge::DumpManager::GetInstance().GetDumpProperties(ge::kInferSessionId);
  GertRuntimeStub runtime_stub;
  dlog_setlevel(GE_MODULE_NAME, DLOG_WARN, 0);
  dumper->DoHostDataDump(dumper->node_names_to_dump_units_["add1"], properties);
  auto logs = runtime_stub.GetSlogStub().GetLogs();
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
  EXPECT_FALSE(CheckLogExpected(logs, "not in the dump step"));
}

TEST_F(HostExecutorDumperUT, DoHostDataDump_Ok) {
  GlobalDumper::GetInstance()->SetEnableFlags(
      gert::BuiltInSubscriberUtil::EnableBit<gert::DumpType>(gert::DumpType::kHostDump));
  auto executor = BuildExecutorFromSingleNodeForDump();
  ASSERT_NE(executor, nullptr);
  auto dumper =
      executor->GetSubscribers().MutableBuiltInSubscriber<HostExecutorDumper>(BuiltInSubscriberType::kHostDumper);
  ASSERT_NE(dumper, nullptr);
  StorageShape storage_shape{{2048}, {2048, 1}};
  auto total_size = 0UL;
  auto tensor_holder =
      Tensor::CreateFollowing(storage_shape.GetStorageShape().GetShapeSize(), ge::DT_FLOAT16, total_size);
  auto context_holder_1 =
      KernelRunContextFaker()
          .KernelIONum(1, 2)
          .Outputs({tensor_holder.get(), &reinterpret_cast<Tensor *>(tensor_holder.get())->MutableTensorData()})
          .Build();
  auto &node_add_dump_unit = dumper->node_names_to_dump_units_["add1"];
  node_add_dump_unit.output_addrs[0] = context_holder_1.GetContext<KernelContext>()->GetOutput(1);
  node_add_dump_unit.input_addrs[0] = context_holder_1.GetContext<KernelContext>()->GetOutput(1);
  node_add_dump_unit.input_addrs[1] = context_holder_1.GetContext<KernelContext>()->GetOutput(1);
  gert::StorageShape x2_shape = {{4, 8, 16, 1, 1}, {4, 8, 16, 1, 1, 1, 1}};
  gert::StorageShape x1_shape = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
  auto context_holder_2 = KernelRunContextFaker()
                              .Inputs({&x1_shape})
                              .Outputs({&x2_shape})
                              .KernelIONum(1, 1)
                              .NodeIoNum(1, 1)
                              .IrInstanceNum({1})
                              .Build();
  node_add_dump_unit.output_shapes[0] = context_holder_2.GetContext<KernelContext>()->GetOutput(0);
  node_add_dump_unit.input_shapes[0] = const_cast<Chain *>(context_holder_2.GetContext<KernelContext>()->GetInput(0));
  node_add_dump_unit.input_shapes[1] = const_cast<Chain *>(context_holder_2.GetContext<KernelContext>()->GetInput(0));
  ge::DumpManager::GetInstance().RemoveDumpProperties(ge::kInferSessionId);
  ge::DumpProperties dump_properties;
  dump_properties.AddPropertyValue("ALL_MODEL_NEED_DUMP_AND_IT_IS_NOT_A_MODEL_NAME", {"test"});
  dump_properties.SetDumpMode("all");
  std::string dump_path = "./dump/";
  dump_properties.SetDumpPath(dump_path);
  const std::string dump_step = "0|2-4|8";
  dump_properties.SetDumpStep(dump_step);
  ge::DumpManager::GetInstance().AddDumpProperties(ge::kInferSessionId, dump_properties);
  HostExecutorDumper::OnExecuteEvent(0, dumper, kModelStart, nullptr, kStatusSuccess);
  auto op_desc = dumper->node_names_to_dump_units_["add1"].node->GetOpDesc();
  std::vector<int64_t> tvm_workspace_memory_type = {ge::AicpuWorkSpaceType::CUST_LOG};
  ge::AttrUtils::SetListInt(op_desc, ge::ATTR_NAME_AICPU_WORKSPACE_TYPE, tvm_workspace_memory_type);
  op_desc->SetWorkspaceBytes(vector<int64_t>{32});
  const auto properties = ge::DumpManager::GetInstance().GetDumpProperties(ge::kInferSessionId);
  ASSERT_TRUE(ge::CreateDirectory(dump_path) == 0);
  GertRuntimeStub runtime_stub;
  dlog_setlevel(GE_MODULE_NAME, DLOG_INFO, 0);
  dumper->DoHostDataDump(dumper->node_names_to_dump_units_["add1"], properties);
  auto logs = runtime_stub.GetSlogStub().GetLogs();
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
  EXPECT_FALSE(CheckLogExpected(logs, "not in the dump step"));

  HostExecutorDumper::OnExecuteEvent(0, dumper, kModelEnd, nullptr, kStatusSuccess);
  EXPECT_EQ(dumper->GetIterationNum(), 1);
  dlog_setlevel(GE_MODULE_NAME, DLOG_WARN, 0);
  dumper->DoHostDataDump(dumper->node_names_to_dump_units_["add1"], properties);
  logs = runtime_stub.GetSlogStub().GetLogs();
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
  EXPECT_TRUE(CheckLogExpected(logs, "not in the dump step"));
  ASSERT_TRUE(mmRmdir(dump_path.c_str()) == 0);
}

TEST_F(HostExecutorDumperUT, DoHostDump_WithWrongInputSize) {
  GlobalDumper::GetInstance()->SetEnableFlags(
      gert::BuiltInSubscriberUtil::EnableBit<gert::DumpType>(gert::DumpType::kHostDump));
  auto graph = std::make_shared<ge::ComputeGraph>("test");
  auto op_data = ge::OpDescBuilder("fake", "Fake").AddInput("x_0").AddInput("x_1").AddOutput("y").Build();
  auto fake_node = graph->AddNode(op_data);
  auto executor = BuildExecutorFromSingleNodeForDump();
  auto dumper = executor->GetSubscribers().MutableBuiltInSubscriber<HostExecutorDumper>(BuiltInSubscriberType::kHostDumper);
  ge::DumpManager::GetInstance().RemoveDumpProperties(ge::kInferSessionId);
  ge::DumpProperties dump_properties;
  dump_properties.AddPropertyValue("ALL_MODEL_NEED_DUMP_AND_IT_IS_NOT_A_MODEL_NAME", {"test"});
  dump_properties.SetDumpMode("all");
  ge::DumpManager::GetInstance().AddDumpProperties(ge::kInferSessionId, dump_properties);
  NodeDumpUnit dump_unit{};
  dump_unit.node = fake_node;
  dump_unit.input_addrs.resize(0, nullptr);
  dump_unit.output_addrs.resize(1, nullptr);
  dump_unit.input_shapes.resize(2, nullptr);
  dump_unit.output_shapes.resize(1, nullptr);
  const auto properties = ge::DumpManager::GetInstance().GetDumpProperties(ge::kInferSessionId);
  GertRuntimeStub runtime_stub;
  dlog_setlevel(GE_MODULE_NAME, DLOG_WARN, 0);
  dumper->DoHostDataDump(dump_unit, properties);
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
  auto logs = runtime_stub.GetSlogStub().GetLogs();
  EXPECT_TRUE(CheckLogExpected(logs, "input addr or output addr size is invalid"));
}

TEST_F(HostExecutorDumperUT, DoHostDump_WithNoDumpLayer) {
  GlobalDumper::GetInstance()->SetEnableFlags(
      gert::BuiltInSubscriberUtil::EnableBit<gert::DumpType>(gert::DumpType::kHostDump));
  auto graph = std::make_shared<ge::ComputeGraph>("test");
  auto op_data = ge::OpDescBuilder("fake", "Fake").AddInput("x_0").AddInput("x_1").AddOutput("y").Build();
  auto fake_node = graph->AddNode(op_data);
  auto executor = BuildExecutorFromSingleNodeForDump();
  auto dumper = executor->GetSubscribers().MutableBuiltInSubscriber<HostExecutorDumper>(BuiltInSubscriberType::kHostDumper);
  ge::DumpManager::GetInstance().RemoveDumpProperties(ge::kInferSessionId);
  ge::DumpProperties dump_properties;
  dump_properties.AddPropertyValue("WRONG_NAME", {"test"});
  dump_properties.ClearOpDebugFlag();
  dump_properties.SetDumpOpSwitch("off");
  ge::DumpManager::GetInstance().AddDumpProperties(ge::kInferSessionId, dump_properties);
  NodeDumpUnit dump_unit{};
  dump_unit.node = fake_node;
  dump_unit.input_addrs.resize(2, nullptr);
  dump_unit.output_addrs.resize(1, nullptr);
  dump_unit.input_shapes.resize(2, nullptr);
  dump_unit.output_shapes.resize(1, nullptr);
  const auto properties = ge::DumpManager::GetInstance().GetDumpProperties(ge::kInferSessionId);
  EXPECT_FALSE(properties.IsOpDebugOpen());
  EXPECT_FALSE(properties.IsSingleOpNeedDump());
  GertRuntimeStub runtime_stub;
  dlog_setlevel(GE_MODULE_NAME, DLOG_INFO, 0);
  dumper->DoHostDataDump(dump_unit, properties);
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
  auto logs = runtime_stub.GetSlogStub().GetLogs();
  EXPECT_TRUE(CheckLogExpected(logs, "no need to dump"));
}

// op with 2 inputs, addr is null
TEST_F(HostExecutorDumperUT, DoHostDump_WithInputAddrIsNull) {
  GlobalDumper::GetInstance()->SetEnableFlags(
      gert::BuiltInSubscriberUtil::EnableBit<gert::DumpType>(gert::DumpType::kHostDump));
  auto graph = std::make_shared<ge::ComputeGraph>("test");
  auto op_data = ge::OpDescBuilder("fake", "Fake").AddInput("x_0").AddInput("x_1").AddOutput("y").Build();
  auto fake_node = graph->AddNode(op_data);
  auto executor = BuildExecutorFromSingleNodeForDump();
  auto dumper =
      executor->GetSubscribers().MutableBuiltInSubscriber<HostExecutorDumper>(BuiltInSubscriberType::kHostDumper);
  ge::DumpManager::GetInstance().RemoveDumpProperties(ge::kInferSessionId);
  ge::DumpProperties dump_properties;
  dump_properties.AddPropertyValue("ALL_MODEL_NEED_DUMP_AND_IT_IS_NOT_A_MODEL_NAME", {"test"});
  dump_properties.SetDumpMode("all");
  std::string dump_path = "./dump/";
  dump_properties.SetDumpPath(dump_path);
  ge::DumpManager::GetInstance().AddDumpProperties(ge::kInferSessionId, dump_properties);
  ASSERT_TRUE(ge::CreateDirectory(dump_path) == 0);

  NodeDumpUnit dump_unit{};
  dump_unit.node = fake_node;
  dump_unit.input_addrs.resize(2, nullptr);  // 2 input
  dump_unit.output_addrs.resize(1, nullptr);
  dump_unit.input_shapes.resize(2, nullptr);
  dump_unit.output_shapes.resize(1, nullptr);

  GertRuntimeStub runtime_stub;
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
  EXPECT_EQ(dumper->DoHostDataDump(dump_unit, dump_properties), ge::SUCCESS);
  ASSERT_TRUE(mmRmdir(dump_path.c_str()) == 0);
  auto logs = runtime_stub.GetSlogStub().GetLogs(DLOG_ERROR);  // no error logs
  EXPECT_EQ(logs.size(), 0U);
}
}  // namespace gert
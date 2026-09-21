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

#include "macro_utils/dt_public_scope.h"

#include "graph/load/model_manager/davinci_model.h"
#include "graph/load/model_manager/task_info/rts/memcpy_addr_async_task_info.h"
#include "ge/ut/ge/ffts_plus_proto_tools.h"
#include "exec_runtime/execution_runtime_utils.h"

namespace ge {
class UtestMemcpyAddrAsyncTaskInfo : public testing::Test {
protected:
  void SetUp() override {
    ExecutionRuntimeUtils::in_heterogeneous_executor_ = false;
    ModelInit();
    TaskDefInit();
    ArgsInit();
  }

  void TearDown() override {
    ExecutionRuntimeUtils::in_heterogeneous_executor_ = false;
  }

  DavinciModel model_{0, nullptr};
  domi::TaskDef task_def_;

  PisToArgs args_;
  uint8_t dev_ptr_[static_cast<size_t>(ArgsPlacement::kEnd)][128];
  uint8_t host_ptr_[static_cast<size_t>(ArgsPlacement::kEnd)][128];

private:
  void ModelInit() {
    rtStream_t stream = nullptr;
    model_.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
    model_.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
    model_.stream_list_.push_back(stream);

    model_.runtime_param_.logic_mem_base = 0x8003000;
    model_.runtime_param_.logic_weight_base = 0x8008000;
    model_.runtime_param_.logic_var_base = 0x800e000;
    model_.runtime_param_.mem_size = 0x5000;
    model_.runtime_param_.weight_size = 0x6000;
    model_.runtime_param_.var_size = 0x1000;
    model_.logical_mem_allocations_.push_back({0, 0U, UINT64_MAX, MemAllocation::Type::ABSOLUTE, 0U});

    model_.op_list_[0] = CreateOpDesc("dummy_op", MEMCPYADDRASYNC, 1, 1);
  }

  void TaskDefInit() {
    task_def_.set_stream_id(0);
    auto *memcpy_async = task_def_.mutable_memcpy_async();
    memcpy_async->set_dst(0x8003000);
    memcpy_async->set_dst_max(512);
    memcpy_async->set_src(0x8003100);
    memcpy_async->set_count(1);
    memcpy_async->set_kind(RT_MEMCPY_ADDR_DEVICE_TO_DEVICE);
    memcpy_async->set_op_index(0);
  }

  void ArgsInit() {
    for (size_t n = 0; n < args_.size(); n++) {
      args_[n].dev_addr = reinterpret_cast<uint64_t>(dev_ptr_[n]);
      args_[n].host_addr = reinterpret_cast<void *>(host_ptr_[n]);
    }
  }
};

TEST_F(UtestMemcpyAddrAsyncTaskInfo, memcpy_addr_async_task_init_failed) {
  MemcpyAddrAsyncTaskInfo task_info;
  DavinciModel model(0, nullptr);
  domi::TaskDef task_def;
  TaskRunParam param;

  // DavinciModel is null
  EXPECT_NE(task_info.ParseTaskRunParam(task_def, nullptr, param), SUCCESS);

  // GetOpByIndex failed
  EXPECT_NE(task_info.ParseTaskRunParam(task_def, &model, param), SUCCESS);

  // ArgsFormat parse failed
  model.op_list_[0] = CreateOpDesc("dummy_op", MEMCPYADDRASYNC, 1, 1);
  auto *memcpy_async = task_def.mutable_memcpy_async();
  memcpy_async->set_op_index(0);
  memcpy_async->set_args_format("{bad_format}");
  EXPECT_NE(task_info.ParseTaskRunParam(task_def, &model, param), SUCCESS);

  // GetRtAddress failed
  memcpy_async->set_src(0x10);
  memcpy_async->set_args_format("{}{}{i_instance0*}{o_instance0*}");
  EXPECT_NE(task_info.ParseTaskRunParam(task_def, &model_, param), SUCCESS);
}

TEST_F(UtestMemcpyAddrAsyncTaskInfo, memcpy_async_task_init_success) {
  MemcpyAddrAsyncTaskInfo task_info;
  TaskRunParam param;
  task_def_.mutable_memcpy_async()->set_args_format("{}{}{i_instance0*}{o_instance0*}");
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def_, &model_, param), SUCCESS);

  IowAddrs iow_addrs = {std::move(param.parsed_input_addrs), std::move(param.parsed_output_addrs)};
  const PisToPersistentWorkspace workspace = {};
  EXPECT_EQ(task_info.Init(task_def_, &model_, args_, workspace, iow_addrs), SUCCESS);

  std::vector<TaskArgsRefreshInfo> infos;
  EXPECT_EQ(task_info.GetTaskArgsRefreshInfos(infos), SUCCESS);
  // src, dst
  EXPECT_EQ(infos.size(), 2UL);
}

TEST_F(UtestMemcpyAddrAsyncTaskInfo, memcpy_async_task_init_success_david) {
  MemcpyAddrAsyncTaskInfo task_info;
  TaskRunParam param;
  task_def_.mutable_memcpy_async()->set_args_format("{}{}{}{}{i_instance0*}{o_instance0*}{#.32b1}{.32b}{}");
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def_, &model_, param), SUCCESS);

  IowAddrs iow_addrs = {std::move(param.parsed_input_addrs), std::move(param.parsed_output_addrs)};
  const PisToPersistentWorkspace workspace = {};
  EXPECT_EQ(task_info.Init(task_def_, &model_, args_, workspace, iow_addrs), SUCCESS);

  std::vector<TaskArgsRefreshInfo> infos;
  EXPECT_EQ(task_info.GetTaskArgsRefreshInfos(infos), SUCCESS);
  // src, dst
  EXPECT_EQ(infos.size(), 2UL);
}

TEST_F(UtestMemcpyAddrAsyncTaskInfo, success_memcpy_async_task_init_disable_zero_copy_when_Heterogeneous) {
  ExecutionRuntimeUtils::EnableInHeterogeneousExecutor();
  MemcpyAddrAsyncTaskInfo task_info;
  TaskRunParam param;
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def_, &model_, param), SUCCESS);

  IowAddrs iow_addrs = {std::move(param.parsed_input_addrs), std::move(param.parsed_output_addrs)};
  const PisToPersistentWorkspace workspace = {};
  EXPECT_EQ(task_info.Init(task_def_, &model_, args_, workspace, iow_addrs), SUCCESS);
}

TEST_F(UtestMemcpyAddrAsyncTaskInfo, distribute_non_david) {
  MemcpyAddrAsyncTaskInfo task_info;
  TaskRunParam param;
  task_def_.mutable_memcpy_async()->set_args_format("{}{}{i_instance0*}{o_instance0*}");
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def_, &model_, param), SUCCESS);

  IowAddrs iow_addrs = {std::move(param.parsed_input_addrs), std::move(param.parsed_output_addrs)};
  const PisToPersistentWorkspace workspace = {};
  EXPECT_EQ(task_info.Init(task_def_, &model_, args_, workspace, iow_addrs), SUCCESS);
  domi::GetContext().is_online_model = true;
  EXPECT_EQ(task_info.Distribute(), SUCCESS);

  EXPECT_TRUE(task_info.IsSupportReDistribute());
  EXPECT_EQ(task_info.Distribute(), SUCCESS);
  domi::GetContext().is_online_model = false;
}

TEST_F(UtestMemcpyAddrAsyncTaskInfo, distribute_david) {
  MemcpyAddrAsyncTaskInfo task_info;
  TaskRunParam param;
  task_def_.mutable_memcpy_async()->set_args_format("{}{}{}{}{i_instance0*}{o_instance0*}{#.32b1}{.32b}{}");
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def_, &model_, param), SUCCESS);

  IowAddrs iow_addrs = {std::move(param.parsed_input_addrs), std::move(param.parsed_output_addrs)};
  const PisToPersistentWorkspace workspace = {};
  EXPECT_EQ(task_info.Init(task_def_, &model_, args_, workspace, iow_addrs), SUCCESS);
  EXPECT_EQ(task_info.Distribute(), SUCCESS);
}
}  // namespace ge

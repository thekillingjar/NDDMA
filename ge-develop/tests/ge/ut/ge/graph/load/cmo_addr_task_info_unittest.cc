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
#include "exec_runtime/execution_runtime_utils.h"

#include "graph/compute_graph.h"
#include "graph/load/model_manager/task_info/rts/cmo_addr_task_info.h"
#include "graph/load/model_manager/davinci_model.h"
#include "proto/task.pb.h"
#include "ge/ut/ge/ffts_plus_proto_tools.h"

namespace ge {
class UtestCmoAddrTaskInfo : public testing::Test {
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

  DavinciModel davinci_model{0, nullptr};
  domi::TaskDef task_def;
  OpDescPtr op_desc;
  PisToArgs args_;
  uint8_t dev_ptr_[static_cast<size_t>(ArgsPlacement::kEnd)][128];
  uint8_t host_ptr_[static_cast<size_t>(ArgsPlacement::kEnd)][128];

private:
  void ModelInit() {
    rtStream_t stream = nullptr;
    davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
    davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
    davinci_model.stream_list_= {stream};

    davinci_model.runtime_param_.logic_mem_base = 0x8003000;
    davinci_model.runtime_param_.logic_weight_base = 0x8008000;
    davinci_model.runtime_param_.logic_var_base = 0x800e000;
    davinci_model.runtime_param_.mem_size = 0x5000;
    davinci_model.runtime_param_.weight_size = 0x6000;
    davinci_model.runtime_param_.var_size = 0x1000;
    davinci_model.logical_mem_allocations_.push_back({0, 0x1000UL, 64, ge::MemAllocation::Type::FEATURE_MAP, 0U});

    op_desc = MakeShared<OpDesc>("cmo","Cmo");
    op_desc->SetId(0);
    davinci_model.op_list_[op_desc->GetId()] = op_desc;
    GeTensorDesc tensor(GeShape({4, 4, 4, 4}), FORMAT_ND, DT_INT64);
    TensorUtils::SetSize(tensor, 2048);
    op_desc->AddInputDesc(tensor);
    op_desc->SetInputOffset({0x1000});
  }

  void TaskDefInit() {
    task_def.set_stream_id(0);
    domi::CmoAddrTaskDef *cmo_addr_task = task_def.mutable_cmo_addr_task();
    cmo_addr_task->set_op_index(op_desc->GetId());
    cmo_addr_task->set_cmo_op_code(6);
    cmo_addr_task->set_num_inner(0);
    cmo_addr_task->set_num_outer(0);
    cmo_addr_task->set_length_inner(1024);
    cmo_addr_task->set_src(0x1000);
    cmo_addr_task->set_stride_inner(0);
    cmo_addr_task->set_stride_outer(0);
  }

  void ArgsInit() {
    for (size_t n = 0; n < args_.size(); n++) {
      args_[n].dev_addr = reinterpret_cast<uint64_t>(dev_ptr_[n]);
      args_[n].host_addr = reinterpret_cast<void *>(host_ptr_[n]);
    }
  }
};

TEST_F(UtestCmoAddrTaskInfo, parse_and_init_success) {
  CmoAddrTaskInfo task_info;
  TaskRunParam task_run_param = {};

  task_def.mutable_cmo_addr_task()->set_args_format("{#1}{#1}{#1}{#.32b1}{#.32b1}{i_instance0*}{#1}{#.32b1}{.32b}{}");
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);
  EXPECT_EQ(task_run_param.parsed_input_addrs.size(), 1UL);
  EXPECT_EQ(task_run_param.parsed_output_addrs.size(), 0UL);
  EXPECT_EQ(task_run_param.parsed_workspace_addrs.size(), 0UL);

  EXPECT_EQ(task_info.ParseOpIndex(task_def), 0);

  PisToArgs args;
  uint8_t device_args[1024];
  uint8_t host_args[1024];
  args[static_cast<uint32_t>(task_info.args_placement_)].dev_addr = reinterpret_cast<uint64_t>(device_args);
  args[static_cast<uint32_t>(task_info.args_placement_)].host_addr = host_args;
  args[static_cast<uint32_t>(task_info.args_placement_)].len = 1024;
  const PisToPersistentWorkspace persistant_workspace = {};
  IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                        std::move(task_run_param.parsed_workspace_addrs)};
  EXPECT_EQ(task_info.Init(task_def, &davinci_model, args, persistant_workspace, iow_addrs), SUCCESS);
  EXPECT_EQ(task_info.io_addrs_.size(), 1UL);
  EXPECT_EQ(task_info.io_addrs_[0], 0x1000);
  EXPECT_TRUE(reinterpret_cast<uint64_t>(task_info.args_) % 64 == 0);
  std::vector<TaskArgsRefreshInfo> infos;
  EXPECT_EQ(task_info.GetTaskArgsRefreshInfos(infos), SUCCESS);
  EXPECT_EQ(infos.size(), 1UL);
}

TEST_F(UtestCmoAddrTaskInfo, parse_and_init_with_size_success) {
  AttrUtils::SetInt(op_desc, "max_size", 10);
  CmoAddrTaskInfo task_info;
  TaskRunParam task_run_param = {};
  task_def.mutable_cmo_addr_task()->set_args_format("{#1}{#1}{#1}{#.32b1}{#.32b1}{i_instance0*}{#1}{#.32b1}{.32b}{}");
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);
  EXPECT_EQ(task_run_param.parsed_input_addrs.size(), 1UL);
  EXPECT_EQ(task_run_param.parsed_output_addrs.size(), 0UL);
  EXPECT_EQ(task_run_param.parsed_workspace_addrs.size(), 0UL);

  EXPECT_EQ(task_info.ParseOpIndex(task_def), 0);

  PisToArgs args;
  uint8_t device_args[1024];
  uint8_t host_args[1024];
  args[static_cast<uint32_t>(task_info.args_placement_)].dev_addr = reinterpret_cast<uint64_t>(device_args);
  args[static_cast<uint32_t>(task_info.args_placement_)].host_addr = host_args;
  args[static_cast<uint32_t>(task_info.args_placement_)].len = 1024;
  const PisToPersistentWorkspace persistant_workspace = {};
  IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                        std::move(task_run_param.parsed_workspace_addrs)};
  EXPECT_EQ(task_info.Init(task_def, &davinci_model, args, persistant_workspace, iow_addrs), SUCCESS);
  EXPECT_EQ(task_info.io_addrs_.size(), 1UL);
  EXPECT_EQ(task_info.io_addrs_[0], 0x1000);
  EXPECT_TRUE(reinterpret_cast<uint64_t>(task_info.args_) % 64 == 0);
  std::vector<TaskArgsRefreshInfo> infos;
  EXPECT_EQ(task_info.GetTaskArgsRefreshInfos(infos), SUCCESS);
  EXPECT_EQ(infos.size(), 1UL);
}

TEST_F(UtestCmoAddrTaskInfo, init_with_disable_zcpy_success) {
  task_def.mutable_cmo_addr_task()->set_args_format("{#1}{#1}{#1}{#.32b1}{#.32b1}{i_instance0*}{#1}{#.32b1}{.32b}{}");
  davinci_model.runtime_param_.mem_size = 10000UL;
  std::vector<uint8_t> memory_holder(davinci_model.runtime_param_.mem_size);
  davinci_model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(memory_holder.data());
  MemAllocation fm_mem_allocation = {0, static_cast<uint64_t>(davinci_model.runtime_param_.mem_base),
                                     davinci_model.runtime_param_.mem_size, ge::MemAllocation::Type::FEATURE_MAP, 0U};
  davinci_model.logical_mem_allocations_.emplace_back(fm_mem_allocation);

  CmoAddrTaskInfo task_info;
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);
  EXPECT_EQ(task_run_param.parsed_input_addrs.size(), 1UL);
  EXPECT_EQ(task_run_param.parsed_output_addrs.size(), 0UL);
  EXPECT_EQ(task_run_param.parsed_workspace_addrs.size(), 0UL);

  PisToArgs args;
  uint8_t device_args[1024];
  uint8_t host_args[2048];
  args[static_cast<uint32_t>(task_info.args_placement_)].dev_addr = reinterpret_cast<uint64_t>(device_args);
  args[static_cast<uint32_t>(task_info.args_placement_)].host_addr = host_args;
  args[static_cast<uint32_t>(task_info.args_placement_)].len = 2048;
  const PisToPersistentWorkspace persistant_workspace = {};
  ExecutionRuntimeUtils::EnableInHeterogeneousExecutor();
  IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                        std::move(task_run_param.parsed_workspace_addrs)};
  EXPECT_EQ(task_info.Init(task_def, &davinci_model, args, persistant_workspace, iow_addrs), SUCCESS);
  EXPECT_EQ(task_info.io_addrs_.size(), 1UL);
  EXPECT_EQ(task_info.io_addrs_[0], davinci_model.runtime_param_.mem_base + 0x1000);
  EXPECT_TRUE(reinterpret_cast<uint64_t>(task_info.args_) % 64 == 0);
  domi::GetContext().is_online_model = true;
  EXPECT_EQ(task_info.Distribute(), SUCCESS);
  EXPECT_TRUE(task_info.IsSupportReDistribute());
  EXPECT_EQ(task_info.Distribute(), SUCCESS);
  domi::GetContext().is_online_model = false;
}

TEST_F(UtestCmoAddrTaskInfo, kernel_task_info_update_args_aicpu) {
  CmoAddrTaskInfo task_info;
  task_def.mutable_cmo_addr_task()->set_args_format("{#1}{#1}{#1}{#.32b1}{#.32b1}{i_instance0*}{#1}{#.32b1}{.32b}{}");
  task_info.davinci_model_ = &davinci_model;
  task_info.op_desc_ = op_desc;
  task_info.io_addrs_ = {PtrToValue((void *)0x12345678)};
  uint8_t device_args[1024];
  task_info.args_ = device_args;
  task_info.args_io_addrs_updater_.v_mem_allocation_id_and_offset_.push_back({0, 0});
   uint64_t* active_base_addr = (uint64_t*)malloc(8);
  davinci_model.allocation_ids_to_active_base_addr_ = active_base_addr;
  std::vector<uint64_t> allocation_ids_to_active_base_addr;
  allocation_ids_to_active_base_addr.emplace_back(100);
  std::vector<uint64_t> args(1U, 0U);
  EXPECT_EQ(task_info.UpdateHostArgs(allocation_ids_to_active_base_addr, args.data(), 128), SUCCESS);
  EXPECT_EQ(args[0], 100);
  free(active_base_addr);
}

TEST_F(UtestCmoAddrTaskInfo, parse_and_init_with_max_size_success) {
  AttrUtils::SetInt(op_desc, "max_size", 512);
  CmoAddrTaskInfo task_info;
  TaskRunParam task_run_param = {};
  task_def.mutable_cmo_addr_task()->set_args_format("{#1}{#1}{#1}{#.32b1}{#.32b1}{i_instance0*}{#1}{#.32b1}{.32b}{}");
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);
  EXPECT_EQ(task_run_param.parsed_input_addrs.size(), 1UL);
  EXPECT_EQ(task_run_param.parsed_output_addrs.size(), 0UL);
  EXPECT_EQ(task_run_param.parsed_workspace_addrs.size(), 0UL);

  EXPECT_EQ(task_info.ParseOpIndex(task_def), 0);

  PisToArgs args;
  uint8_t device_args[1024];
  uint8_t host_args[1024];
  args[static_cast<uint32_t>(task_info.args_placement_)].dev_addr = reinterpret_cast<uint64_t>(device_args);
  args[static_cast<uint32_t>(task_info.args_placement_)].host_addr = host_args;
  args[static_cast<uint32_t>(task_info.args_placement_)].len = 1024;
  const PisToPersistentWorkspace persistant_workspace = {};
  IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                        std::move(task_run_param.parsed_workspace_addrs)};
  EXPECT_EQ(task_info.Init(task_def, &davinci_model, args, persistant_workspace, iow_addrs), SUCCESS);
  EXPECT_EQ(task_info.io_addrs_.size(), 1UL);
  EXPECT_TRUE(reinterpret_cast<uint64_t>(task_info.args_) % 64 == 0);
  std::vector<TaskArgsRefreshInfo> infos;
  EXPECT_EQ(task_info.GetTaskArgsRefreshInfos(infos), SUCCESS);
  EXPECT_EQ(infos.size(), 1UL);
}

TEST_F(UtestCmoAddrTaskInfo, parse_and_init_with_max_size_offset_invalid) {
  AttrUtils::SetInt(op_desc, "max_size", 512);
  AttrUtils::SetInt(op_desc, "offset", 2048);
  CmoAddrTaskInfo task_info;
  TaskRunParam task_run_param = {};
  EXPECT_NE(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);
}

TEST_F(UtestCmoAddrTaskInfo, ParseTaskRunParam_EmptyArgsFormat) {
  CmoAddrTaskInfo task_info;
  TaskRunParam task_run_param = {};
  task_def.mutable_cmo_addr_task()->set_args_format("");
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);
  EXPECT_EQ(task_run_param.parsed_input_addrs.size(), 1UL);
  EXPECT_EQ(task_run_param.parsed_output_addrs.size(), 0UL);
  EXPECT_EQ(task_run_param.parsed_workspace_addrs.size(), 0UL);
}

TEST_F(UtestCmoAddrTaskInfo, ParseTaskRunParam_InvalidOffset) {
  AttrUtils::SetInt(op_desc, "offset", 2048);
  CmoAddrTaskInfo task_info;
  TaskRunParam task_run_param = {};
  EXPECT_NE(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);
}

TEST_F(UtestCmoAddrTaskInfo, ParseTaskRunParam_ValidMaxSize) {
  AttrUtils::SetInt(op_desc, "max_size", 512);
  CmoAddrTaskInfo task_info;
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);
  EXPECT_EQ(task_run_param.parsed_input_addrs.size(), 1UL);
  EXPECT_EQ(task_run_param.parsed_output_addrs.size(), 0UL);
  EXPECT_EQ(task_run_param.parsed_workspace_addrs.size(), 0UL);
}

TEST_F(UtestCmoAddrTaskInfo, Init_ValidArgsPlacement) {
  CmoAddrTaskInfo task_info;
  task_info.davinci_model_ = &davinci_model;
  task_info.op_desc_ = op_desc;
  task_def.mutable_cmo_addr_task()->set_args_format("{#1}{#1}{#1}{#.32b1}{#.32b1}{i_instance0*}{#1}{#.32b1}{.32b}{}");
  PisToArgs args;
  TaskRunParam task_run_param = {};
  uint8_t device_args[1024];
  uint8_t host_args[1024];
  args[static_cast<uint32_t>(task_info.args_placement_)].dev_addr = reinterpret_cast<uint64_t>(device_args);
  args[static_cast<uint32_t>(task_info.args_placement_)].host_addr = host_args;
  args[static_cast<uint32_t>(task_info.args_placement_)].len = 1024;
  const PisToPersistentWorkspace persistent_workspace = {};
  IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                        std::move(task_run_param.parsed_workspace_addrs)};
  EXPECT_EQ(task_info.Init(task_def, &davinci_model, args, persistent_workspace, iow_addrs), SUCCESS);
  EXPECT_EQ(task_info.io_addrs_.size(), 0UL);
}

TEST_F(UtestCmoAddrTaskInfo, Distribute_Success) {
  CmoAddrTaskInfo task_info;
  task_info.davinci_model_ = &davinci_model;
  task_info.op_desc_ = op_desc;
  PisToArgs args;
  TaskRunParam task_run_param = {};
  task_def.mutable_cmo_addr_task()->set_args_format("{#1}{#1}{#1}{#.32b1}{#.32b1}{i_instance0*}{#1}{#.32b1}{.32b}{}");
  uint8_t device_args[1024];
  uint8_t host_args[1024];
  args[static_cast<uint32_t>(task_info.args_placement_)].dev_addr = reinterpret_cast<uint64_t>(device_args);
  args[static_cast<uint32_t>(task_info.args_placement_)].host_addr = host_args;
  args[static_cast<uint32_t>(task_info.args_placement_)].len = 1024;
  const PisToPersistentWorkspace persistent_workspace = {};
  IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                        std::move(task_run_param.parsed_workspace_addrs)};
  EXPECT_EQ(task_info.Init(task_def, &davinci_model, args, persistent_workspace, iow_addrs), SUCCESS);
  EXPECT_EQ(task_info.Distribute(), SUCCESS);
}
}  // namespace ge

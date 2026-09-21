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
#include <iostream>
#include "common/opskernel/ops_kernel_info_store.h"
#include "graph/ge_local_context.h"
#include "depends/mmpa/src/mmpa_stub.h"

#include "macro_utils/dt_public_scope.h"
#include "graph/load/model_manager/davinci_model.h"
#include "graph/load/model_manager/task_info/hccl/hccl_task_info.h"
#include "graph/load/model_manager/task_info/fe/dsa_task_info.h"
#include "ge/ut/ge/ffts_plus_proto_tools.h"
#include "opskernel_executor/ops_kernel_executor_manager.h"
#include "macro_utils/dt_public_unscope.h"


namespace ge {
namespace {
class HcclOpsKernelInfoStore : public OpsKernelInfoStore {
 public:
  HcclOpsKernelInfoStore() = default;
  Status Initialize(const std::map<std::string, std::string> &options) override { return SUCCESS; }
  // close opsKernelInfoStore
  Status Finalize() override { return SUCCESS; }
  // get all opsKernelInfo
  void GetAllOpsKernelInfo(std::map<std::string, OpInfo> &infos) const override {}
  // whether the opsKernelInfoStore is supported based on the operator attribute
  bool CheckSupported(const OpDescPtr &opDescPtr, std::string &un_supported_reason) const override { return true; }
  Status UnloadTask(GETaskInfo &task) {
    return SUCCESS;
  }
};
}  // namespace

class HcclTaskInfoTest : public testing::Test {
 protected:
  void SetUp() {
  }
  void TearDown() {
  }
};

TEST_F(HcclTaskInfoTest, Calculate_Update_Args) {
  DavinciModel model(0, nullptr);
  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = {stream};
  OpsKernelExecutorManager::GetInstance().Initialize({});
  OpsKernelExecutorManager::GetInstance().executors_[kEngineNameHccl] = std::make_shared<HcclOpsKernelInfoStore>();

  domi::TaskDef task_def;
  domi::KernelHcclDef *kernel_hccl_def = task_def.mutable_kernel_hccl();
  kernel_hccl_def->set_op_index(0);
  kernel_hccl_def->set_hccl_type("HcomBroadcast");

  GeTensorDesc desc;
  auto op_desc = std::make_shared<OpDesc>("hcom_reduce", HCOMREDUCE);
  AttrUtils::SetInt(op_desc, HCOM_ATTR_ROOT_RANK, 0);
  AttrUtils::SetStr(op_desc, HCOM_ATTR_REDUCE_TYPE, "min");
  AttrUtils::SetBool(op_desc, ATTR_NAME_IS_UNKNOWN_SHAPE, true);
  op_desc->SetStreamId(0);
  op_desc->SetId(0);
  op_desc->AddInputDesc(desc);
  op_desc->AddOutputDesc(desc);
  op_desc->SetInputOffset({8});
  op_desc->SetWorkspace({800});
  op_desc->SetWorkspaceBytes({150});
  op_desc->SetOutputOffset({8});
  op_desc->SetOpKernelLibName(kEngineNameHccl);

  (void)AttrUtils::SetListInt(op_desc, TVM_ATTR_NAME_WORKSPACE_TYPE, {kRtMemoryUB});

  model.runtime_param_.mem_size = 1024;
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(malloc(1024));
  MemAllocation not_change_mem_item = {0, 0U, UINT64_MAX, ge::MemAllocation::Type::ABSOLUTE, 0U};
  model.logical_mem_allocations_.emplace_back(not_change_mem_item);
  model.feature_base_refreshable_ = true;
  model.op_list_[op_desc->GetId()] = op_desc;

  PisToArgs args;
  const PisToPersistentWorkspace persistant_workspace = {};
  IowAddrs iow_addrs;
  args[0].dev_addr = 3;
  args[1].dev_addr = 3;
  iow_addrs.input_logic_addrs = {{1, 0}};
  iow_addrs.output_logic_addrs = {{2, 0}};
  iow_addrs.workspace_logic_addrs = {{1, 0}};

  HcclTaskInfo hccl_task_info;
  TaskRunParam task_run_param = {};
  auto ret = hccl_task_info.ParseTaskRunParam(task_def, &model, task_run_param);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(hccl_task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);
  hccl_task_info.io_addrs_.clear();
  hccl_task_info.io_addrs_.push_back(12405000);
  hccl_task_info.io_addrs_.push_back(12405100);
  int64_t op_index = hccl_task_info.ParseOpIndex(task_def);
  EXPECT_EQ(op_index, 0);
  DumpProperties dump_properties;
  dump_properties.SetDumpMode("all");
  dump_properties.AddPropertyValue(DUMP_ALL_MODEL, {});
  model.SetDumpProperties(dump_properties);
  std::vector<int64_t> host_args2({0,0,0});
  // model.allocation_ids_to_active_base_addr_.clear();
  // model.allocation_ids_to_active_base_addr_.push_back(12405233);
  // model.allocation_ids_to_active_base_addr_.push_back(12406543);
  // model.allocation_ids_to_active_base_addr_.push_back(12405409);
  std::vector<uint64_t> allocation_ids_to_active_base_addr_vec;
  allocation_ids_to_active_base_addr_vec.push_back(12405233);
  allocation_ids_to_active_base_addr_vec.push_back(12406543);
  allocation_ids_to_active_base_addr_vec.push_back(12405409);
  ret = hccl_task_info.UpdateHostArgs(allocation_ids_to_active_base_addr_vec, static_cast<void *>(host_args2.data()), 3 * sizeof(int64_t));
  EXPECT_EQ(ret, SUCCESS);
  hccl_task_info.InsertDumpOp("input");
  hccl_task_info.InsertDumpOp("output");
  MemAllocationAndOffset v1 = {0, 120};
  MemAllocationAndOffset v2 = {2, 0};
  hccl_task_info.args_io_addrs_updater_.v_mem_allocation_id_and_offset_.clear();
  hccl_task_info.args_io_addrs_updater_.v_mem_allocation_id_and_offset_.push_back(v1);
  hccl_task_info.args_io_addrs_updater_.v_mem_allocation_id_and_offset_.push_back(v2);
  std::vector<int64_t> host_args({0,0});
  ret = hccl_task_info.UpdateHostArgs(allocation_ids_to_active_base_addr_vec, static_cast<void *>(host_args.data()), 2 * sizeof(int64_t));
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(host_args[0], 12405353);
  EXPECT_EQ(host_args[1], 12405409);
  task_def.clear_kernel_hccl();
  free(reinterpret_cast<uint8_t *>(model.runtime_param_.mem_base));
}

TEST_F(HcclTaskInfoTest, dsa_dump_test) {
  domi::TaskDef task_def;
  DSATaskInfo task_info;
  EXPECT_EQ(task_info.Init(task_def, nullptr, {}, {}, {{},{},{}}), PARAM_INVALID);

  DavinciModel model(0, nullptr);
  task_def.set_stream_id(0);
  EXPECT_EQ(task_info.Init(task_def, &model, {}, {}, {{},{},{}}), FAILED);

  model.stream_list_.push_back((void *)0x12345);
  model.runtime_param_.mem_size = 10240;
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(new uint8_t[model.runtime_param_.mem_size]);
  MemAllocation fm_mem_allocation = {0, static_cast<uint64_t>(model.runtime_param_.mem_base),
                                     model.runtime_param_.mem_size, ge::MemAllocation::Type::FEATURE_MAP, 0U};
  model.logical_mem_allocations_.emplace_back(fm_mem_allocation);

  MemAllocation absolut_mem_allocation = {1, 0, 0xffffffffffff, ge::MemAllocation::Type::ABSOLUTE, 0U};
  model.logical_mem_allocations_.emplace_back(absolut_mem_allocation);

  domi::DSATaskDef *dsa_task = task_def.mutable_dsa_task();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr add_node = CreateNode(*graph, "add", ADD, 3, 1);
  auto op_desc = add_node->GetOpDesc();
  EXPECT_NE(op_desc, nullptr);

  const vector<int64_t> v_memory_type{RT_MEMORY_L1};
  AttrUtils::SetListInt(op_desc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, v_memory_type);
  op_desc->SetOutputOffset({2048});
  {  // RT_MEMORY_L1
    auto tensor_desc = op_desc->MutableOutputDesc(0);
    EXPECT_NE(tensor_desc, nullptr);
    TensorUtils::SetSize(*tensor_desc, 64);
    TensorUtils::SetDataOffset(*tensor_desc, 2048);
  }

  const vector<int64_t> v_memory_type1{RT_MEMORY_L1, RT_MEMORY_L1, RT_MEMORY_L1};
  AttrUtils::SetListInt(op_desc, ATTR_NAME_INPUT_MEM_TYPE_LIST, v_memory_type1);
  op_desc->SetInputOffset({2048, 2048, 2048});

  {  // RT_MEMORY_L1
    for (int i = 0; i < 3; ++i) {
      auto tensor_desc = op_desc->MutableOutputDesc(0);
      EXPECT_NE(tensor_desc, nullptr);
      TensorUtils::SetSize(*tensor_desc, 64);
      TensorUtils::SetDataOffset(*tensor_desc, 2048);
    }
  }

  const vector<int64_t> v_memory_type2{RT_MEMORY_HBM, RT_MEMORY_HBM};
  op_desc->SetWorkspace({1308, 1458});
  op_desc->SetWorkspaceBytes({150, 150});
  AttrUtils::SetListInt(op_desc, TVM_ATTR_NAME_WORKSPACE_TYPE, v_memory_type2);
  AttrUtils::SetListInt(op_desc, ATTR_NAME_WORKSPACE_TYPE_LIST, v_memory_type2);
  model.op_list_[0] = op_desc;

  dsa_task->set_op_index(0);
  dsa_task->set_start(1);
  dsa_task->set_sqe_type(1);
  dsa_task->set_distribution_type(1);
  dsa_task->set_data_type(1);
  dsa_task->set_alg_type(1);
  dsa_task->set_input_vld(1);
  dsa_task->set_input_value_addr_flag(1);
  dsa_task->set_input1_value_or_ptr(0);
  dsa_task->set_input2_value_or_ptr(0);
  dsa_task->set_seed_value_or_ptr(0);
  dsa_task->set_random_count_value_or_ptr(0);
  domi::DSATaskArgsDef *dsa_task_args = dsa_task->mutable_args();
  dsa_task_args->set_output_addr(0x12);
  dsa_task_args->set_workspace_philox_count_addr(0x24);
  dsa_task_args->set_workspace_input_addr(0x457);
  dsa_task_args->set_seed_value_or_addr("5");
  dsa_task_args->set_input1_value_or_addr("1");
  dsa_task_args->set_input2_value_or_addr("2");

  model.is_op_debug_reg_ = true;

  std::vector<uint64_t> active_mem_base_addr;
  std::vector<HostArg> host_args;
  host_args.emplace_back(HostArg{ValueToPtr(0xa123), 64, ArgsPlacement::kArgsPlacementHbm});
  host_args.emplace_back(HostArg{ValueToPtr(0xa224), 64, ArgsPlacement::kArgsPlacementSqe});
  EXPECT_EQ(task_info.UpdateHostArgs(active_mem_base_addr, host_args), SUCCESS);

  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);

  PisToArgs args = {};
  args[0].dev_addr = PtrToValue(malloc(1024));
  args[1].dev_addr = PtrToValue(malloc(1024));
  const PisToPersistentWorkspace persistant_workspace = {};
  IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                        std::move(task_run_param.parsed_workspace_addrs)};

  EXPECT_EQ(task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);
  task_info.support_refresh_ = false;
  task_info.PostDumpProcess(task_def);
  task_info.support_refresh_ = true;
  task_info.PostDumpProcess(task_def);
  EXPECT_EQ(task_info.Distribute(), SUCCESS);
  delete [] reinterpret_cast<uint8_t *>(model.runtime_param_.mem_base);
  model.runtime_param_.mem_base = 0U;
  model.stream_list_.clear();
  free(ValueToPtr(args[0].dev_addr));
  free(ValueToPtr(args[1].dev_addr));
}


TEST_F(HcclTaskInfoTest, dsa_dump_on_watcher_test) {
  domi::TaskDef task_def;
  DSATaskInfo task_info;
  EXPECT_EQ(task_info.Init(task_def, nullptr, {}, {}, {{},{},{}}), PARAM_INVALID);

  DavinciModel model(0, nullptr);
  task_def.set_stream_id(0);
  EXPECT_EQ(task_info.Init(task_def, &model, {}, {}, {{},{},{}}), FAILED);

  model.stream_list_.push_back((void *)0x12345);
  model.runtime_param_.mem_size = 10240;
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(new uint8_t[model.runtime_param_.mem_size]);
  MemAllocation fm_mem_allocation = {0, static_cast<uint64_t>(model.runtime_param_.mem_base),
                                     model.runtime_param_.mem_size, ge::MemAllocation::Type::FEATURE_MAP, 0U};
  model.logical_mem_allocations_.emplace_back(fm_mem_allocation);

  MemAllocation absolut_mem_allocation = {1, 0, 0xffffffffffff, ge::MemAllocation::Type::ABSOLUTE, 0U};
  model.logical_mem_allocations_.emplace_back(absolut_mem_allocation);

  domi::DSATaskDef *dsa_task = task_def.mutable_dsa_task();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr add_node = CreateNode(*graph, "add", ADD, 3, 1);
  auto op_desc = add_node->GetOpDesc();
  EXPECT_NE(op_desc, nullptr);

  const vector<int64_t> v_memory_type{RT_MEMORY_L1};
  AttrUtils::SetListInt(op_desc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, v_memory_type);
  op_desc->SetOutputOffset({2048});
  {  // RT_MEMORY_L1
    auto tensor_desc = op_desc->MutableOutputDesc(0);
    EXPECT_NE(tensor_desc, nullptr);
    TensorUtils::SetSize(*tensor_desc, 64);
    TensorUtils::SetDataOffset(*tensor_desc, 2048);
  }

  const vector<int64_t> v_memory_type1{RT_MEMORY_L1, RT_MEMORY_L1, RT_MEMORY_L1};
  AttrUtils::SetListInt(op_desc, ATTR_NAME_INPUT_MEM_TYPE_LIST, v_memory_type1);
  op_desc->SetInputOffset({2048, 2048, 2048});

  {  // RT_MEMORY_L1
    for (int i = 0; i < 3; ++i) {
      auto tensor_desc = op_desc->MutableOutputDesc(0);
      EXPECT_NE(tensor_desc, nullptr);
      TensorUtils::SetSize(*tensor_desc, 64);
      TensorUtils::SetDataOffset(*tensor_desc, 2048);
    }
  }

  const vector<int64_t> v_memory_type2{RT_MEMORY_HBM, RT_MEMORY_HBM};
  op_desc->SetWorkspace({1308, 1458});
  op_desc->SetWorkspaceBytes({150, 150});
  AttrUtils::SetListInt(op_desc, TVM_ATTR_NAME_WORKSPACE_TYPE, v_memory_type2);
  AttrUtils::SetListInt(op_desc, ATTR_NAME_WORKSPACE_TYPE_LIST, v_memory_type2);
  model.op_list_[0] = op_desc;

  dsa_task->set_op_index(0);
  dsa_task->set_start(1);
  dsa_task->set_sqe_type(1);
  dsa_task->set_distribution_type(1);
  dsa_task->set_data_type(1);
  dsa_task->set_alg_type(1);
  dsa_task->set_input_vld(1);
  dsa_task->set_input_value_addr_flag(1);
  dsa_task->set_input1_value_or_ptr(0);
  dsa_task->set_input2_value_or_ptr(0);
  dsa_task->set_seed_value_or_ptr(0);
  dsa_task->set_random_count_value_or_ptr(0);
  domi::DSATaskArgsDef *dsa_task_args = dsa_task->mutable_args();
  dsa_task_args->set_output_addr(0x12);
  dsa_task_args->set_workspace_philox_count_addr(0x24);
  dsa_task_args->set_workspace_input_addr(0x457);
  dsa_task_args->set_seed_value_or_addr("5");
  dsa_task_args->set_input1_value_or_addr("1");
  dsa_task_args->set_input2_value_or_addr("2");

  DumpProperties dump_properties;
  dump_properties.SetDumpMode("output");
  dump_properties.AddPropertyValue(DUMP_WATCHER_MODEL, {"test"}); // dsa is layer op on watcher mode
  model.SetDumpProperties(dump_properties);

  std::vector<uint64_t> active_mem_base_addr;
  std::vector<HostArg> host_args;
  host_args.emplace_back(HostArg{ValueToPtr(0xa123), 64, ArgsPlacement::kArgsPlacementHbm});
  host_args.emplace_back(HostArg{ValueToPtr(0xa224), 64, ArgsPlacement::kArgsPlacementSqe});
  EXPECT_EQ(task_info.UpdateHostArgs(active_mem_base_addr, host_args), SUCCESS);

  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);

  PisToArgs args = {};
  args[0].dev_addr = PtrToValue(malloc(1024));
  args[1].dev_addr = PtrToValue(malloc(1024));
  const PisToPersistentWorkspace persistant_workspace = {};
  IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                        std::move(task_run_param.parsed_workspace_addrs)};

  EXPECT_EQ(task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);
  task_info.support_refresh_ = false;
  task_info.PostDumpProcess(task_def);
  task_info.support_refresh_ = true;
  task_info.PostDumpProcess(task_def);
  EXPECT_EQ(task_info.Distribute(), SUCCESS);
  delete [] reinterpret_cast<uint8_t *>(model.runtime_param_.mem_base);
  model.runtime_param_.mem_base = 0U;
  model.stream_list_.clear();
  free(ValueToPtr(args[0].dev_addr));
  free(ValueToPtr(args[1].dev_addr));
}

TEST_F(HcclTaskInfoTest, dsa_dump_addr_refresh_test_not_support_fresh) {
  domi::TaskDef task_def;
  DSATaskInfo task_info;
  EXPECT_EQ(task_info.Init(task_def, nullptr, {}, {}, {{},{},{}}), PARAM_INVALID);

  DavinciModel model(0, nullptr);
  task_def.set_stream_id(0);
  EXPECT_EQ(task_info.Init(task_def, &model, {}, {}, {{},{},{}}), FAILED);

  model.stream_list_.push_back((void *)0x12345);
  model.runtime_param_.mem_size = 10240;
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(new uint8_t[model.runtime_param_.mem_size]);
  MemAllocation fm_mem_allocation = {0, static_cast<uint64_t>(model.runtime_param_.mem_base),
                                     model.runtime_param_.mem_size, ge::MemAllocation::Type::FEATURE_MAP, 0U};
  model.logical_mem_allocations_.emplace_back(fm_mem_allocation);

  MemAllocation absolut_mem_allocation = {1, 0, 0xffffffffffff, ge::MemAllocation::Type::ABSOLUTE, 0U};
  model.logical_mem_allocations_.emplace_back(absolut_mem_allocation);

  domi::DSATaskDef *dsa_task = task_def.mutable_dsa_task();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr add_node = CreateNode(*graph, "add", ADD, 3, 1);
  auto op_desc = add_node->GetOpDesc();
  EXPECT_NE(op_desc, nullptr);

  const vector<int64_t> v_memory_type{RT_MEMORY_L1};
  AttrUtils::SetListInt(op_desc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, v_memory_type);
  op_desc->SetOutputOffset({2048});
  {  // RT_MEMORY_L1
    auto tensor_desc = op_desc->MutableOutputDesc(0);
    EXPECT_NE(tensor_desc, nullptr);
    TensorUtils::SetSize(*tensor_desc, 64);
    TensorUtils::SetDataOffset(*tensor_desc, 2048);
  }

  const vector<int64_t> v_memory_type1{RT_MEMORY_L1, RT_MEMORY_L1, RT_MEMORY_L1};
  AttrUtils::SetListInt(op_desc, ATTR_NAME_INPUT_MEM_TYPE_LIST, v_memory_type1);
  op_desc->SetInputOffset({2048, 2048, 2048});

  {  // RT_MEMORY_L1
    for (int i = 0; i < 3; ++i) {
      auto tensor_desc = op_desc->MutableOutputDesc(0);
      EXPECT_NE(tensor_desc, nullptr);
      TensorUtils::SetSize(*tensor_desc, 64);
      TensorUtils::SetDataOffset(*tensor_desc, 2048);
    }
  }

  const vector<int64_t> v_memory_type2{RT_MEMORY_HBM, RT_MEMORY_HBM};
  op_desc->SetWorkspace({1308, 1458});
  op_desc->SetWorkspaceBytes({150, 150});
  AttrUtils::SetListInt(op_desc, TVM_ATTR_NAME_WORKSPACE_TYPE, v_memory_type2);
  AttrUtils::SetListInt(op_desc, ATTR_NAME_WORKSPACE_TYPE_LIST, v_memory_type2);
  model.op_list_[0] = op_desc;

  dsa_task->set_op_index(0);
  dsa_task->set_start(1);
  dsa_task->set_sqe_type(1);
  dsa_task->set_distribution_type(1);
  dsa_task->set_data_type(1);
  dsa_task->set_alg_type(1);
  dsa_task->set_input_vld(1);
  dsa_task->set_input_value_addr_flag(1);
  dsa_task->set_input1_value_or_ptr(0);
  dsa_task->set_input2_value_or_ptr(0);
  dsa_task->set_seed_value_or_ptr(0);
  dsa_task->set_random_count_value_or_ptr(0);
  domi::DSATaskArgsDef *dsa_task_args = dsa_task->mutable_args();
  dsa_task_args->set_output_addr(0x12);
  dsa_task_args->set_workspace_philox_count_addr(0x24);
  dsa_task_args->set_workspace_input_addr(0x457);
  dsa_task_args->set_seed_value_or_addr("5");
  dsa_task_args->set_input1_value_or_addr("1");
  dsa_task_args->set_input2_value_or_addr("2");

  model.is_op_debug_reg_ = true;
  
  task_info.support_refresh_ = true;
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);
  EXPECT_EQ(task_info.dump_args_, 0);

  PisToArgs args = {};
  args[0].dev_addr = PtrToValue(malloc(1024));
  args[1].dev_addr = PtrToValue(malloc(1024));
  const PisToPersistentWorkspace persistant_workspace = {};
  IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                        std::move(task_run_param.parsed_workspace_addrs)};

  EXPECT_EQ(task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);

  std::vector<uint64_t> active_mem_base_addr;
  std::vector<HostArg> host_args;
  host_args.emplace_back(HostArg{ValueToPtr(0xa123), 64, ArgsPlacement::kArgsPlacementHbm});
  host_args.emplace_back(HostArg{ValueToPtr(0xa224), 64, ArgsPlacement::kArgsPlacementSqe});
  EXPECT_EQ(task_info.UpdateHostArgs(active_mem_base_addr, host_args), SUCCESS);

  task_info.PostDumpProcess(task_def);
  EXPECT_EQ(task_info.dump_args_, 0);

  EXPECT_EQ(task_info.Distribute(), SUCCESS);
  delete [] reinterpret_cast<uint8_t *>(model.runtime_param_.mem_base);
  model.runtime_param_.mem_base = 0U;
  model.stream_list_.clear();
  free(ValueToPtr(args[0].dev_addr));
  free(ValueToPtr(args[1].dev_addr));
}

}  // namespace ge

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

#include "macro_utils/dt_public_scope.h"
#include "graph/load/model_manager/task_info/fe/dsa_task_info.h"
#include "graph/load/model_manager/davinci_model.h"
#include "ge/ut/ge/ffts_plus_proto_tools.h"
#include "framework/common/types.h"

extern std::string g_runtime_stub_mock;

namespace ge {
class UtestDSATask : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

// test Init_DSATaskInfo
TEST_F(UtestDSATask, init_dsa_task_info) {
  const char_t * const kEnvValue = "SET_CAPA_VALUE";
  char_t npu_collect_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &npu_collect_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&npu_collect_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvValue, fail_collect_path.c_str(), 1);
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

  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);
  EXPECT_EQ(task_run_param.parsed_input_addrs.size(), 3);
  EXPECT_EQ(task_run_param.parsed_output_addrs.size(), 1);
  EXPECT_EQ(task_run_param.parsed_workspace_addrs.size(), 2);

  PisToArgs args = {};
  args[0].dev_addr = PtrToValue(malloc(1024));
  args[1].dev_addr = PtrToValue(malloc(1024));
  const PisToPersistentWorkspace persistant_workspace = {};
  IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                        std::move(task_run_param.parsed_workspace_addrs)};

  EXPECT_EQ(task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);

  dsa_task->set_input1_value_or_ptr(1);
  EXPECT_EQ(task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);

  std::vector<TaskArgsRefreshInfo> infos;
  EXPECT_EQ(task_info.GetTaskArgsRefreshInfos(infos), SUCCESS);

  std::vector<IowPaRemapInfo> remap_infos;
  EXPECT_EQ(task_info.GetTaskIowPaRemapInfos(remap_infos), SUCCESS);
  ASSERT_EQ(remap_infos.size(), 1UL);
  EXPECT_EQ(remap_infos[0U].policy, PaRemapPolicy::KNoSupport);
  EXPECT_EQ(remap_infos[0U].tensor_size, 150);
  EXPECT_EQ(remap_infos[0U].allocation_offset, 1308);

  int64_t op_index = task_info.ParseOpIndex(task_def);
  EXPECT_EQ(op_index, 0);
  EXPECT_EQ(task_info.Release(), SUCCESS);
  delete [] reinterpret_cast<uint8_t *>(model.runtime_param_.mem_base);
  model.runtime_param_.mem_base = 0U;
  model.stream_list_.clear();
  free(ValueToPtr(args[0].dev_addr));
  free(ValueToPtr(args[1].dev_addr));
}

TEST_F(UtestDSATask, init_stateless_dsa_task_info) {
  const char_t * const kEnvValue = "SET_CAPA_VALUE";
  char_t npu_collect_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &npu_collect_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&npu_collect_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvValue, fail_collect_path.c_str(), 1);
  domi::TaskDef task_def;
  task_def.set_stream_id(0);

  DavinciModel model(0, nullptr);
  model.stream_list_.push_back((void *)0x12345);
  model.runtime_param_.mem_size = 10240;
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(new uint8_t[model.runtime_param_.mem_size]);
  model.feature_base_refreshable_ = true;
  MemAllocation fm_mem_allocation = {0, static_cast<uint64_t>(model.runtime_param_.mem_base),
                                     model.runtime_param_.mem_size, ge::MemAllocation::Type::FEATURE_MAP, 0U};
  model.logical_mem_allocations_.emplace_back(fm_mem_allocation);

  MemAllocation absolut_mem_allocation = {1, 0, 0xffffffffffff, ge::MemAllocation::Type::ABSOLUTE, 0U};
  model.logical_mem_allocations_.emplace_back(absolut_mem_allocation);

  domi::DSATaskDef *dsa_task = task_def.mutable_dsa_task();
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr add_node = CreateNode(*graph, "add", ADD, 5, 1);
  auto op_desc = add_node->GetOpDesc();
  EXPECT_NE(op_desc, nullptr);

  op_desc->SetOutputOffset({2048});
  {
    auto tensor_desc = op_desc->MutableOutputDesc(0);
    EXPECT_NE(tensor_desc, nullptr);
    TensorUtils::SetSize(*tensor_desc, 64);
    TensorUtils::SetDataOffset(*tensor_desc, 2048);
  }

  op_desc->SetInputOffset({2048, 2048, 2048, 2048, 2048});
  {
    for (int i = 0; i < 5; ++i) {
      auto tensor_desc = op_desc->MutableInputDesc(i);
      EXPECT_NE(tensor_desc, nullptr);
      TensorUtils::SetSize(*tensor_desc, 64);
      TensorUtils::SetDataOffset(*tensor_desc, 2048);
    }
  }

  op_desc->SetWorkspace({1308});
  op_desc->SetWorkspaceBytes({150});
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
  dsa_task_args->set_seed_value_or_addr("5");
  dsa_task_args->set_input1_value_or_addr("1");
  dsa_task_args->set_input2_value_or_addr("2");

  DSATaskInfo task_info;
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);

  PisToArgs args = {};
  args[0].dev_addr = PtrToValue(malloc(1024));
  args[1].dev_addr = PtrToValue(malloc(1024));
  const PisToPersistentWorkspace persistant_workspace = {};
  IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                        std::move(task_run_param.parsed_workspace_addrs)};

  EXPECT_EQ(task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);

  dsa_task->set_input1_value_or_ptr(1);
  EXPECT_EQ(task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);

  std::vector<IowPaRemapInfo> remap_infos;
  EXPECT_EQ(task_info.GetTaskIowPaRemapInfos(remap_infos), SUCCESS);
  EXPECT_TRUE(remap_infos.empty());

  delete [] reinterpret_cast<uint8_t *>(model.runtime_param_.mem_base);
  model.runtime_param_.mem_base = 0U;
  model.stream_list_.clear();
  free(ValueToPtr(args[0].dev_addr));
  free(ValueToPtr(args[1].dev_addr));
  task_info.Release();
}

TEST_F(UtestDSATask, testParseTaskRunParam) {
  domi::TaskDef task_def;
  DSATaskInfo task_info;

  DavinciModel model(0, nullptr);
  task_def.set_stream_id(0);

  model.stream_list_.push_back((void *)0x12345);
  model.runtime_param_.mem_size = 10240;
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(new uint8_t[model.runtime_param_.mem_size]);
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

  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);

  delete [] reinterpret_cast<uint8_t *>(model.runtime_param_.mem_base);
  model.runtime_param_.mem_base = 0U;
  model.stream_list_.clear();
}

TEST_F(UtestDSATask, testDistributeWithFlag) {
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

  // model.is_op_debug_reg_ = true;
  std::set<std::string> temp;
  model.data_dumper_.dump_properties_.model_dump_properties_map_.emplace(DUMP_ALL_MODEL, temp);

  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);

  PisToArgs args = {};
  args[0].dev_addr = PtrToValue(malloc(1024));
  args[1].dev_addr = PtrToValue(malloc(1024));
  const PisToPersistentWorkspace persistant_workspace = {};
  IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                        std::move(task_run_param.parsed_workspace_addrs)};

  EXPECT_EQ(task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);
  EXPECT_EQ(task_info.dump_flag_, RT_KERNEL_DUMPFLAG);

  domi::GetContext().is_online_model = true;
  auto ret = task_info.Distribute();
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_TRUE(task_info.IsSupportReDistribute());
  EXPECT_EQ(task_info.Distribute(), SUCCESS);
  delete [] reinterpret_cast<uint8_t *>(model.runtime_param_.mem_base);
  model.runtime_param_.mem_base = 0U;
  model.stream_list_.clear();
  free(ValueToPtr(args[0].dev_addr));
  free(ValueToPtr(args[1].dev_addr));
  domi::GetContext().is_online_model = false;
}

TEST_F(UtestDSATask, testPostProcess) {
  domi::TaskDef task_def;
  DSATaskInfo task_info;
  EXPECT_EQ(task_info.Init(task_def, nullptr, {}, {}, {{},{},{}}), PARAM_INVALID);

  DavinciModel model(0, nullptr);
  DumpProperties dump_properties;
  dump_properties.SetDumpMode("output");
  dump_properties.AddPropertyValue(DUMP_LAYER_OP_MODEL, {});
  dump_properties.AddPropertyValue(DUMP_WATCHER_MODEL, {"add"});
  model.SetDumpProperties(dump_properties);

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

  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);

  PisToArgs args = {};
  args[0].dev_addr = PtrToValue(malloc(1024));
  args[1].dev_addr = PtrToValue(malloc(1024));
  const PisToPersistentWorkspace persistant_workspace = {};
  IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                        std::move(task_run_param.parsed_workspace_addrs)};

  EXPECT_EQ(task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);

  task_info.PostDumpProcess(task_def);

  const auto workspace_size = ModelUtils::GetWorkspaceSize(op_desc);
  const uint64_t dev_size = MemSizeAlign(workspace_size[workspace_size.size() - 1U], sizeof(uint64_t));
  EXPECT_EQ(task_info.workspace_io_addrs_.size(),
      ((dev_size / sizeof(uint64_t)) + task_info.input_data_addrs_.size() + task_info.output_data_addrs_.size()));
  EXPECT_EQ(task_info.hbm_args_refresh_flags_.size(),
      ((dev_size / sizeof(uint64_t)) + task_info.input_addr_refresh_.size() + task_info.output_addr_refresh_.size()));

  EXPECT_EQ(task_info.Distribute(), SUCCESS);

  dump_properties.AddPropertyValue(DUMP_WATCHER_MODEL, {"test"});
  model.SetDumpProperties(dump_properties);

  task_info.PostProcess(task_def);
  task_info.support_refresh_ = true;
  task_info.PostProcess(task_def);
  delete [] reinterpret_cast<uint8_t *>(model.runtime_param_.mem_base);
  model.runtime_param_.mem_base = 0U;
  model.stream_list_.clear();
  free(ValueToPtr(args[0].dev_addr));
  free(ValueToPtr(args[1].dev_addr));
}

TEST_F(UtestDSATask, update_host_args_failed) {
  const char_t * const kEnvValue = "SET_CAPA_VALUE";
  char_t npu_collect_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &npu_collect_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&npu_collect_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvValue, fail_collect_path.c_str(), 1);
  domi::TaskDef task_def;
  DSATaskInfo task_info;

  std::vector<uint64_t> active_mem_base_addr;
  std::vector<HostArg> host_args;

  host_args.emplace_back(HostArg{ValueToPtr(0xa123), 64, ArgsPlacement::kArgsPlacementHbm});
  host_args.emplace_back(HostArg{ValueToPtr(0xa224), 64, ArgsPlacement::kArgsPlacementHbm});
  host_args.emplace_back(HostArg{ValueToPtr(0xa323), 64, ArgsPlacement::kArgsPlacementHbm});
  EXPECT_NE(task_info.UpdateHostArgs(active_mem_base_addr, host_args), SUCCESS);

  host_args.clear();
  host_args.emplace_back(HostArg{ValueToPtr(0xa123), 64, ArgsPlacement::kArgsPlacementHbm});
  host_args.emplace_back(HostArg{ValueToPtr(0xa224), 64, ArgsPlacement::kArgsPlacementSqe});
  EXPECT_EQ(task_info.UpdateHostArgs(active_mem_base_addr, host_args), SUCCESS);
}
}  // namespace ge

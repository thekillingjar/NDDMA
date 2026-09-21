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
#include "graph/load/model_manager/task_info/rts/memcpy_async_task_info.h"
#include "ge/ut/ge/ffts_plus_proto_tools.h"
#include "graph/load/model_manager/memory_app_type_classifier.h"
#include "common/opskernel/ops_kernel_info_types.h"

namespace ge {
class UtestMemcpyAsyncTaskInfo : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestMemcpyAsyncTaskInfo, success_memcpy_async_task_init) {
  PisToArgs args;
  const PisToPersistentWorkspace persistant_workspace = {};
  IowAddrs iow_addrs;
  args[0].dev_addr = 0xacd3;
  args[1].dev_addr = 0x1b43;
  iow_addrs.input_logic_addrs = {{0x1a23, (uint64_t)ge::MemoryAppType::kMemoryTypeFix}};
  iow_addrs.output_logic_addrs = {{0xc212, (uint64_t)ge::MemoryAppType::kMemoryTypeFeatureMap, true}};

  DavinciModel model(0, nullptr);
  domi::TaskDef task_def;
  task_def.set_stream_id(0);

  domi::MemcpyAsyncDef *memcpy_async = task_def.mutable_memcpy_async();
  memcpy_async->set_dst(10);
  memcpy_async->set_dst_max(512);
  memcpy_async->set_src(10);
  memcpy_async->set_count(1);
  memcpy_async->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
  memcpy_async->set_op_index(6);

  model.feature_base_refreshable_ = false;
  model.runtime_param_.logic_mem_base = 0x8003000;
  model.runtime_param_.logic_weight_base = 0x8008000;
  model.runtime_param_.logic_var_base = 0x800e000;
  model.runtime_param_.mem_size = 0x5000;
  model.runtime_param_.weight_size = 0x6000;
  model.runtime_param_.var_size = 0x1000;

  MemcpyAsyncTaskInfo memcpy_async_task_info;

  // GetOpByIndex src failed
  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_.push_back(stream);
  EXPECT_EQ(memcpy_async_task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), INTERNAL_ERROR);

  model.op_list_[6] = CreateOpDesc("memcpyasync", MEMCPYASYNC);
  memcpy_async->set_src(0x08008000);

  MemAllocation not_change_mem_item = {static_cast<uint32_t>(model.logical_mem_allocations_.size()), 0U, UINT64_MAX,
                                       ge::MemAllocation::Type::ABSOLUTE, 0U};
  model.logical_mem_allocations_.emplace_back(not_change_mem_item);

  // set OpDesc attr
  AttrUtils::SetStr(model.op_list_[6], ATTR_DYNAMIC_SHAPE_FIXED_ADDR, "Hello Mr Tree");
  AttrUtils::SetStr(model.op_list_[6], "_alloc_input_fixed_addr", "dsa out 0");
  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  model.op_list_[6]->AddInputDesc(tensor);
  model.op_list_[6]->AddOutputDesc(tensor);
  EXPECT_EQ(memcpy_async_task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);
  EXPECT_EQ(memcpy_async_task_info.logical_src_addr_, 0x1a23);
  EXPECT_EQ(PtrToValue(memcpy_async_task_info.src_), 0xacd3);

  model.feature_base_refreshable_ = true;
  memcpy_async_task_info.args_mem_type_ = 0;
  EXPECT_EQ(memcpy_async_task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);
  EXPECT_EQ(PtrToValue(memcpy_async_task_info.src_), 0xacd3);
  EXPECT_EQ(memcpy_async_task_info.logical_src_addr_, 0x1a23);
  EXPECT_EQ(memcpy_async_task_info.io_addr_mem_types_[0], (uint64_t)ge::MemoryAppType::kMemoryTypeFix);
  EXPECT_EQ(memcpy_async_task_info.logical_dst_addr_, 0xc212);
  EXPECT_EQ(memcpy_async_task_info.io_addr_mem_types_[1], (uint64_t)ge::MemoryAppType::kMemoryTypeFeatureMap);

  MemcpyAsyncTaskInfo memcpy_async_task_info1;
  model.logical_mem_allocations_.clear();
  model.logical_mem_allocations_.emplace_back(not_change_mem_item);
  memcpy_async_task_info1.args_mem_type_ = 64;
  EXPECT_EQ(memcpy_async_task_info1.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);
  EXPECT_EQ(PtrToValue(memcpy_async_task_info1.src_), 0x1b43);

  // set OpDesc attr
  std::vector<int64_t> memory_type = { RT_MEMORY_TS };
  AttrUtils::SetListInt(model.op_list_[6], ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memory_type);
  model.SetFeatureBaseRefreshable(false);
  memcpy_async->set_dst_max(0);
  memcpy_async->set_dst(0x8003001);
  EXPECT_EQ(memcpy_async_task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);

  memcpy_async->set_dst_max(0);
  model.op_list_[6]->SetInputOffset({1024});
  model.op_list_[6]->SetOutputOffset({5120});
  memcpy_async->set_dst(0x8003001);
  EXPECT_EQ(memcpy_async_task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);

  task_def.clear_memcpy_async();
}

TEST_F(UtestMemcpyAsyncTaskInfo, success_memcpy_async_task_init_failed) {
  PisToArgs args;
  const PisToPersistentWorkspace persistant_workspace = {};
  IowAddrs iow_addrs;
  args[0].dev_addr = 0xacd3;
  args[1].dev_addr = 0x1b43;
  iow_addrs.input_logic_addrs = {{0x1a23, (uint64_t)ge::MemoryAppType::kMemoryTypeFix}};
  iow_addrs.output_logic_addrs = {{0xc212, (uint64_t)ge::MemoryAppType::kMemoryTypeFeatureMap}};

  DavinciModel model(0, nullptr);
  domi::TaskDef task_def;
  task_def.set_stream_id(0);

  domi::MemcpyAsyncDef *memcpy_async = task_def.mutable_memcpy_async();
  memcpy_async->set_dst(10);
  memcpy_async->set_dst_max(512);
  memcpy_async->set_src(10);
  memcpy_async->set_count(1);
  memcpy_async->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
  memcpy_async->set_op_index(6);

  model.runtime_param_.logic_mem_base = 0x8003000;
  model.runtime_param_.logic_weight_base = 0x8008000;
  model.runtime_param_.logic_var_base = 0x800e000;
  model.runtime_param_.mem_size = 0x5000;
  model.runtime_param_.weight_size = 0x6000;
  model.runtime_param_.var_size = 0x1000;
  MemAllocation fm_mem_allocation = {0, static_cast<uint64_t>(model.runtime_param_.mem_base),
                                     model.runtime_param_.mem_size, ge::MemAllocation::Type::FEATURE_MAP, 0U};
  model.logical_mem_allocations_.emplace_back(fm_mem_allocation);

  // DavinciModel is null
  MemcpyAsyncTaskInfo memcpy_async_task_info;
  EXPECT_EQ(memcpy_async_task_info.Init(task_def, nullptr, args, persistant_workspace, iow_addrs), PARAM_INVALID);

  // SetStream failed
  EXPECT_EQ(memcpy_async_task_info.Init(task_def, nullptr, args, persistant_workspace, iow_addrs), PARAM_INVALID);

  // GetOpByIndex failed
  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_.push_back(stream);
  EXPECT_EQ(memcpy_async_task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), INTERNAL_ERROR);

  model.op_list_[6] = CreateOpDesc("memcpyasync", MEMCPYASYNC);
  EXPECT_EQ(memcpy_async_task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);
  memcpy_async->set_src(0x08008000);

  EXPECT_EQ(memcpy_async_task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);
  memcpy_async->set_dst(0x08003000);

  // set OpDesc attr
  std::vector<int64_t> memory_type = { RT_MEMORY_TS };
  AttrUtils::SetListInt(model.op_list_[6], ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memory_type);
  memcpy_async->set_dst_max(0);
  EXPECT_EQ(memcpy_async_task_info.Init(task_def, nullptr, args, persistant_workspace, iow_addrs), PARAM_INVALID);
  memcpy_async->set_dst_max(512);


  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  model.op_list_[6]->AddInputDesc(tensor);
  model.op_list_[6]->AddOutputDesc(tensor);
  model.op_list_[6]->SetInputOffset({1024});
  model.op_list_[6]->SetOutputOffset({5120});
  EXPECT_EQ(memcpy_async_task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);

  memcpy_async->set_dst(0x08009000);
  EXPECT_EQ(memcpy_async_task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);

  task_def.clear_memcpy_async();
}

TEST_F(UtestMemcpyAsyncTaskInfo, success_parse_task_run_param_dst_ts_mem) {
  PisToArgs args;
  IowAddrs iow_addrs;
  args[0].dev_addr = 0xacd3;
  args[1].dev_addr = 0x1b43;
  iow_addrs.input_logic_addrs = {{0x1a23, (uint64_t)ge::MemoryAppType::kMemoryTypeFix}};
  iow_addrs.output_logic_addrs = {{0xc212, (uint64_t)ge::MemoryAppType::kMemoryTypeFeatureMap}};

  DavinciModel model(0, nullptr);
  domi::TaskDef task_def;
  task_def.set_stream_id(0);

  domi::MemcpyAsyncDef *memcpy_async = task_def.mutable_memcpy_async();
  memcpy_async->set_dst(10);
  memcpy_async->set_dst_max(512);
  memcpy_async->set_src(10);
  memcpy_async->set_count(1);
  memcpy_async->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
  memcpy_async->set_op_index(6);

  model.runtime_param_.logic_mem_base = 0x8003000;
  model.runtime_param_.logic_weight_base = 0x8008000;
  model.runtime_param_.logic_var_base = 0x800e000;
  model.runtime_param_.mem_size = 0x5000;
  model.runtime_param_.weight_size = 0x6000;
  model.runtime_param_.var_size = 0x1000;
  MemAllocation fm_mem_allocation = {0, static_cast<uint64_t>(model.runtime_param_.mem_base),
                                     model.runtime_param_.mem_size, ge::MemAllocation::Type::FEATURE_MAP, 0U};
  model.logical_mem_allocations_.emplace_back(fm_mem_allocation);


  MemcpyAsyncTaskInfo memcpy_async_task_info;
  model.op_list_[6] = CreateOpDesc("memcpyasync", MEMCPYASYNC);
  memcpy_async->set_src(0x08008000);
  memcpy_async->set_dst(0x08003000);
  std::vector<int64_t> memory_type = { RT_MEMORY_TS };
  AttrUtils::SetListInt(model.op_list_[6], ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memory_type);
  memcpy_async->set_dst_max(0);
  memcpy_async->set_dst_max(512);

  TaskRunParam task_run_param = {};
  EXPECT_EQ(memcpy_async_task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);

  task_def.clear_memcpy_async();
}

TEST_F(UtestMemcpyAsyncTaskInfo, success_memcpy_async_task_distribute) {
  PisToArgs args;
  const PisToPersistentWorkspace persistant_workspace = {};
  IowAddrs iow_addrs;
  args[0].dev_addr = 0xacd3;
  args[1].dev_addr = 0x1b43;
  iow_addrs.input_logic_addrs = {{0x1a23, (uint64_t)ge::MemoryAppType::kMemoryTypeFix}};
  iow_addrs.output_logic_addrs = {{0xc212, (uint64_t)ge::MemoryAppType::kMemoryTypeFeatureMap}};

  DavinciModel model(0, nullptr);
  model.SetFeatureBaseRefreshable(true);
  domi::TaskDef task_def;
  task_def.set_stream_id(0);

  domi::MemcpyAsyncDef *memcpy_async = task_def.mutable_memcpy_async();
  memcpy_async->set_dst(10);
  memcpy_async->set_dst_max(512);
  memcpy_async->set_src(10);
  memcpy_async->set_count(1);
  memcpy_async->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
  memcpy_async->set_op_index(6);

  model.runtime_param_.logic_mem_base = 0x8003000;
  model.runtime_param_.logic_weight_base = 0x8008000;
  model.runtime_param_.logic_var_base = 0x800e000;
  model.runtime_param_.mem_size = 0x5000;
  model.runtime_param_.weight_size = 0x6000;
  model.runtime_param_.var_size = 0x1000;
  MemAllocation not_change_mem_item = {static_cast<uint32_t>(model.logical_mem_allocations_.size()), 0U, UINT64_MAX,
                                       ge::MemAllocation::Type::ABSOLUTE, 0U};
  model.logical_mem_allocations_.emplace_back(not_change_mem_item);

  MemcpyAsyncTaskInfo memcpy_async_task_info;

  // GetOpByIndex src failed
  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_.push_back(stream);
  EXPECT_EQ(memcpy_async_task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), INTERNAL_ERROR);

  model.op_list_[6] = CreateOpDesc("memcpyasync", MEMCPYASYNC);
  memcpy_async->set_src(0x08008000);
  EXPECT_EQ(memcpy_async_task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);

  // set OpDesc attr
  AttrUtils::SetStr(model.op_list_[6], ATTR_DYNAMIC_SHAPE_FIXED_ADDR, "Hello Mr Tree");
  AttrUtils::SetStr(model.op_list_[6], "_alloc_input_fixed_addr", "dsa out 0");
  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  model.op_list_[6]->AddInputDesc(tensor);
  model.op_list_[6]->AddOutputDesc(tensor);
  memcpy_async->set_dst_max(0);
  EXPECT_EQ(memcpy_async_task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);

  memcpy_async->set_dst_max(0);
  model.op_list_[6]->SetInputOffset({1024});
  model.op_list_[6]->SetOutputOffset({5120});
  EXPECT_EQ(memcpy_async_task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);

  task_def.clear_memcpy_async();
}

TEST_F(UtestMemcpyAsyncTaskInfo, success_distribute) {
  DavinciModel model(0, nullptr);
  model.ge_model_ = MakeShared<GeModel>();
  model.op_list_[0] = CreateOpDesc("memcpyasync", MEMCPYASYNC);
  model.SetFeatureBaseRefreshable(true);
  MemAllocation fm_mem_allocation = {0, static_cast<uint64_t>(model.runtime_param_.mem_base),
                                     512, ge::MemAllocation::Type::FEATURE_MAP, 0U};
  model.logical_mem_allocations_.emplace_back(fm_mem_allocation);

  const auto model_task_def = MakeShared<domi::ModelTaskDef>();
  domi::TaskDef *task_def = model_task_def->add_task();
  task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC));
  task_def->mutable_memcpy_async()->set_op_index(0);
  task_def->set_stream_id(0);
  task_def->mutable_memcpy_async()->set_dst_max(512);
  task_def->mutable_memcpy_async()->set_count(1);
  task_def->mutable_memcpy_async()->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);

  TaskInfoPtr task_info = TaskInfoFactory::Instance().Create(static_cast<ModelTaskType>(task_def->type()));
  EXPECT_EQ(task_info->Init(*task_def, &model, {}, {}, {{}, {}, {}}), FAILED);  // stream empty

  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_.push_back(stream);

  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info->ParseTaskRunParam(*task_def, &model, task_run_param), SUCCESS);
  PisToArgs args;
  args[0].dev_addr = 0xacd3;
  args[1].dev_addr = 0x1b43;
  const PisToPersistentWorkspace persistant_workspace = {};
  IowAddrs iow_addrs = {std::move(task_run_param.parsed_input_addrs), std::move(task_run_param.parsed_output_addrs),
                        std::move(task_run_param.parsed_workspace_addrs)};

  EXPECT_EQ(task_info->Init(*task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);

  model.task_list_ = { task_info };
  model.ge_model_->SetModelTaskDef(model_task_def);
  domi::GetContext().is_online_model = true;
  EXPECT_EQ(model.DistributeTask(*model_task_def), SUCCESS);
  EXPECT_EQ(task_info->Distribute(), SUCCESS);

  EXPECT_TRUE(task_info->IsSupportReDistribute());
  EXPECT_EQ(task_info->Distribute(), SUCCESS);
  domi::GetContext().is_online_model = false;

  task_info->Release();
}

TEST_F(UtestMemcpyAsyncTaskInfo, success_memcpy_async_calculate_args) {
  DavinciModel model(0, nullptr);
  model.SetFeatureBaseRefreshable(true);
  domi::TaskDef task_def;

  domi::MemcpyAsyncDef *memcpy_async = task_def.mutable_memcpy_async();
  memcpy_async->set_dst(0x08003000);
  memcpy_async->set_dst_max(512);
  memcpy_async->set_src(0x08008000);
  memcpy_async->set_count(1);
  memcpy_async->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
  memcpy_async->set_op_index(0);

  model.op_list_[0] = CreateOpDesc("memcpyasync", MEMCPYASYNC);
  AttrUtils::SetStr(model.op_list_[0], ATTR_DYNAMIC_SHAPE_FIXED_ADDR, "Hello Mr Tree");
  AttrUtils::SetStr(model.op_list_[0], "_alloc_input_fixed_addr", "dsa out 0");

  model.runtime_param_.logic_mem_base = 0x08003000;
  model.runtime_param_.logic_weight_base = 0x08008000;
  model.runtime_param_.logic_var_base = 0x0800e000;
  model.runtime_param_.mem_size = 0x5000;
  model.runtime_param_.mem_base = 0x12345678;
  model.runtime_param_.weight_base = 0x22345678;
  model.runtime_param_.weight_size = 0x6000;
  model.runtime_param_.var_size = 0x1000;
  MemAllocation not_change_mem_item = {static_cast<uint32_t>(model.logical_mem_allocations_.size()), 0U, UINT64_MAX,
                                       ge::MemAllocation::Type::ABSOLUTE, 0U};
  model.logical_mem_allocations_.emplace_back(not_change_mem_item);

  // DavinciModel is null
  MemcpyAsyncTaskInfo memcpy_async_task_info;
  TaskRunParam task_run_param = {};
  EXPECT_EQ(memcpy_async_task_info.ParseTaskRunParam(task_def, &model, task_run_param), SUCCESS);
  EXPECT_EQ(task_run_param.args_descs.size(), 1);
  EXPECT_EQ(task_run_param.args_descs[0].args_len, sizeof(void *) * 2U);
  EXPECT_EQ(task_run_param.parsed_input_addrs.size(), 1U);
  EXPECT_EQ(task_run_param.parsed_output_addrs.size(), 1U);
  EXPECT_EQ(task_run_param.parsed_input_addrs[0].logic_addr, model.runtime_param_.weight_base);
  EXPECT_EQ(task_run_param.parsed_output_addrs[0].logic_addr, model.runtime_param_.mem_base);
  EXPECT_EQ(task_run_param.parsed_input_addrs[0].memory_type, kWeightMemType);
  EXPECT_EQ(task_run_param.parsed_output_addrs[0].memory_type, kFmMemType);
  EXPECT_EQ(task_run_param.parsed_input_addrs[0].support_refresh, true);
  EXPECT_EQ(task_run_param.parsed_output_addrs[0].support_refresh, true);

  int64_t op_index = memcpy_async_task_info.ParseOpIndex(task_def);
  EXPECT_EQ(op_index, 0);
}

TEST_F(UtestMemcpyAsyncTaskInfo, memcpy_async_update_args) {
  DavinciModel model(0, nullptr);

  MemcpyAsyncTaskInfo memcpy_async_task_info;
  memcpy_async_task_info.davinci_model_ = &model;

  EXPECT_EQ(memcpy_async_task_info.UpdateArgs(), SUCCESS);
}
}  // namespace ge

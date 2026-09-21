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
#include "graph/load/model_manager/task_info/rts/npu_get_float_status_task_info.h"
#include "graph/load/model_manager/davinci_model.h"
#include "depends/runtime/src/runtime_stub.h"

namespace ge {
class UtestNpuGetFloatStatusTask : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestNpuGetFloatStatusTask, init_npu_get_float_status_task_info) {
  PisToArgs args;
  const PisToPersistentWorkspace persistant_workspace = {};
  IowAddrs iow_addrs;
  args[0].dev_addr = PtrToValue(malloc(1024));;
  uint64_t output_addr = (uint64_t)malloc(1024);
  iow_addrs.output_logic_addrs = {{output_addr, (uint64_t)ge::MemoryAppType::kMemoryTypeFeatureMap}};

  domi::TaskDef task_def;
  NpuGetFloatStatusTaskInfo task_info;
  EXPECT_EQ(task_info.Init(task_def, nullptr), PARAM_INVALID);

  DavinciModel model(0, nullptr);
  task_def.set_stream_id(0);
  EXPECT_EQ(task_info.Init(task_def, &model), FAILED);

  auto op_desc = std::make_shared<OpDesc>("test", "test");
  op_desc->SetId(0);
  model.op_list_[op_desc->GetId()] = op_desc;

  auto kernel_def = task_def.mutable_npu_get_float_status();
  kernel_def->set_op_index(op_desc->GetId());

  model.stream_list_.push_back((void *)0x12345);
  model.runtime_param_.mem_size = 10240;
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(new uint8_t[model.runtime_param_.mem_size]);
  MemAllocation not_change_mem_item = {0, 0U, UINT64_MAX, ge::MemAllocation::Type::ABSOLUTE, 0U};
  model.logical_mem_allocations_.emplace_back(not_change_mem_item);

  domi::NpuGetFloatStatusDef *npu_get_float_status_task = task_def.mutable_npu_get_float_status();
  npu_get_float_status_task->set_mode(1);
  npu_get_float_status_task->set_output_addr(0x12);
  npu_get_float_status_task->set_output_size(2048);
  npu_get_float_status_task->set_op_index(0);

  int64_t op_index = task_info.ParseOpIndex(task_def);
  EXPECT_EQ(op_index, 0);

  TaskRunParam task_run_param = {};
  auto ret = task_info.ParseTaskRunParam(task_def, &model, task_run_param);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(task_run_param.args_descs.size(), 1);
  EXPECT_EQ(task_run_param.args_descs[0].args_len, sizeof(void *));
  EXPECT_EQ(task_run_param.parsed_output_addrs.size(), 1U);
  EXPECT_EQ(task_run_param.parsed_output_addrs[0].logic_addr, model.runtime_param_.mem_base + 0x12);
  EXPECT_EQ(task_run_param.parsed_output_addrs[0].memory_type, kFmMemType);
  EXPECT_EQ(task_run_param.parsed_output_addrs[0].support_refresh, true);

  EXPECT_EQ(task_info.Init(task_def, &model, args, persistant_workspace, iow_addrs), SUCCESS);
  EXPECT_EQ(PtrToValue(task_info.args_), args[0].dev_addr);
  EXPECT_EQ(task_info.output_addr_mem_type_, (uint64_t)ge::MemoryAppType::kMemoryTypeFeatureMap);
  EXPECT_EQ(PtrToValue(task_info.output_addr_), output_addr);

  std::vector<TaskArgsRefreshInfo> infos;
  EXPECT_EQ(task_info.GetTaskArgsRefreshInfos(infos), SUCCESS);
  EXPECT_EQ(infos.size(), 1UL);

  ret = task_info.UpdateArgs();
  EXPECT_EQ(ret, SUCCESS);

  delete [] reinterpret_cast<uint8_t *>(model.runtime_param_.mem_base);
  model.runtime_param_.mem_base = 0U;
  model.stream_list_.clear();
  free(ValueToPtr(args[0].dev_addr));
  free((void*)output_addr);
}

TEST_F(UtestNpuGetFloatStatusTask, testDistribute) {
  NpuGetFloatStatusTaskInfo task_info;

  domi::TaskDef task_def;
  task_def.set_stream_id(0);

  DavinciModel model(0, nullptr);
  model.stream_list_.push_back((void *)0x12345);

  auto op_desc = std::make_shared<OpDesc>("test", "test");
  op_desc->SetId(0);
  model.op_list_[op_desc->GetId()] = op_desc;

  auto kernel_def = task_def.mutable_npu_get_float_status();
  kernel_def->set_op_index(op_desc->GetId());
  task_info.Init(task_def, &model);
  domi::GetContext().is_online_model = true;
  auto ret = task_info.Distribute();
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_TRUE(task_info.IsSupportReDistribute());
  EXPECT_EQ(task_info.Distribute(), SUCCESS);
  domi::GetContext().is_online_model = false;
  model.stream_list_.clear();
}
}  // namespace ge

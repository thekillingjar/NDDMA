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
#include "graph/load/model_manager/task_info/rts/npu_clear_float_status_task_info.h"
#include "graph/load/model_manager/davinci_model.h"
#include "depends/runtime/src/runtime_stub.h"

namespace ge {
class UtestNpuClearFloatStatusTask : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestNpuClearFloatStatusTask, init_npu_clear_float_status_task_info) {
  domi::TaskDef task_def;
  NpuClearFloatStatusTaskInfo task_info;
  EXPECT_EQ(task_info.Init(task_def, nullptr), PARAM_INVALID);

  DavinciModel model(0, nullptr);
  task_def.set_stream_id(0);
  EXPECT_EQ(task_info.Init(task_def, &model), FAILED);

  model.stream_list_.push_back((void *)0x12345);
  model.runtime_param_.mem_size = 10240;
  model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(new uint8_t[model.runtime_param_.mem_size]);
  domi::NpuClearFloatStatusDef *npu_clear_float_status_task = task_def.mutable_npu_clear_float_status();
  npu_clear_float_status_task->set_mode(1);

  EXPECT_EQ(task_info.Init(task_def, &model), SUCCESS);

  delete [] reinterpret_cast<uint8_t *>(model.runtime_param_.mem_base);
  model.runtime_param_.mem_base = 0U;
  model.stream_list_.clear();
}

TEST_F(UtestNpuClearFloatStatusTask, testDistribute) {
  NpuClearFloatStatusTaskInfo task_info;

  domi::TaskDef task_def;
  task_def.set_stream_id(0);

  DavinciModel model(0, nullptr);
  model.stream_list_.push_back((void *)0x12345);
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

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
#include "graph/load/model_manager/task_info/rts/end_graph_task_info.h"
#include "graph/load/model_manager/davinci_model.h"
#include "macro_utils/dt_public_unscope.h"

using namespace std;

namespace ge {
class UtestEndGraphTask : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

// test Init_EndGraphTaskInfo_failed
TEST_F(UtestEndGraphTask, init_end_graph_task_info) {
  domi::TaskDef task_def;
  EndGraphTaskInfo task_info;
  EXPECT_EQ(task_info.Init(task_def, nullptr), PARAM_INVALID);

  DavinciModel model(0, nullptr);
  task_def.set_stream_id(0);
  EXPECT_EQ(task_info.Init(task_def, &model), FAILED);

  model.stream_list_.push_back((void *)0x12345);
  EXPECT_EQ(task_info.Init(task_def, &model), SUCCESS);
  model.stream_list_.clear();
}

TEST_F(UtestEndGraphTask, testDistribute_Invalid) {
  EndGraphTaskInfo task_info;

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
  model.stream_list_.clear();
  domi::GetContext().is_online_model = false;
}
}  // namespace ge

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
#include "graph/load/model_manager/task_info/rts/event_wait_task_info.h"
#include "graph/load/model_manager/davinci_model.h"
#include "ge/ut/ge/ffts_plus_proto_tools.h"

using namespace std;

namespace ge {
class UtestNotifyWaitTask : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

// test Init_EventWaitTaskInfo
TEST_F(UtestNotifyWaitTask, init_and_distribute_event_wait_task_info) {
  domi::TaskDef task_def;
  task_def.set_stream_id(0);
  task_def.set_event_id(0);
  task_def.mutable_event_ex()->set_op_index(0);
  task_def.mutable_event_ex()->set_event_type(0);

  DavinciModel model(0, nullptr);
  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = { stream };

  rtEvent_t event = nullptr;
  rtEventCreate(&event);
  model.event_list_ = { event };

  model.op_list_[0] = CreateOpDesc("op_name", "op_type");

  EventWaitTaskInfo task_info;
  EXPECT_EQ(task_info.Init(task_def, &model), SUCCESS);
  EXPECT_EQ(task_info.davinci_model_->GetOpByIndex(task_info.op_index_)->GetName(), "op_name");

  domi::GetContext().is_online_model = true;
  EXPECT_EQ(task_info.Distribute(), SUCCESS);

  EXPECT_TRUE(task_info.IsSupportReDistribute());
  EXPECT_EQ(task_info.Distribute(), SUCCESS);
  domi::GetContext().is_online_model = false;
}

}  // namespace ge

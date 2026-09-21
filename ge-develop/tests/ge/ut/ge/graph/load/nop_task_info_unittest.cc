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
#include "graph/load/model_manager/task_info/rts/nop_task_info.h"
#include "graph/load/model_manager/davinci_model.h"

namespace ge {
class UtestNopTaskInfo : public testing::Test {
protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestNopTaskInfo, init_and_distribute_nop_success) {
  PisToArgs args;
  const PisToPersistentWorkspace persistant_workspace = {};
  IowAddrs iow_addrs;
  NopTaskInfo task_info;
  DavinciModel davinci_model(0, nullptr);
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  domi::TaskDef task_def;
  task_def.set_stream_id(0U);
  EXPECT_EQ(task_info.Init(task_def, &davinci_model, args, persistant_workspace, iow_addrs), SUCCESS);
  // distribute
  domi::GetContext().is_online_model = true;
  EXPECT_EQ(task_info.Distribute(), SUCCESS);
  EXPECT_TRUE(task_info.IsSupportReDistribute());
  EXPECT_EQ(task_info.Distribute(), SUCCESS);
  domi::GetContext().is_online_model = false;
}
} // namespace ge

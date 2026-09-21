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
#include "graph/load/model_manager/task_info/fe/cmo_barrier_task_info.h"
#include "graph/load/model_manager/davinci_model.h"

namespace ge {
class UtestCmoBarrierTaskInfo : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}

 public:
  void CreateCmoBarrierTaskDef(DavinciModel &davinci_model, domi::TaskDef &task_def) {
    rtStream_t stream = nullptr;
    davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
    davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
    davinci_model.stream_list_ = {stream};

    task_def.set_stream_id(0);
    domi::CmoBarrierTaskDef *cmo_barrier_task_def = task_def.mutable_cmo_barrier_task();
    cmo_barrier_task_def->set_logic_id_num(0);
    domi::CmoBarrierInfoDef *cmo_barrier_info_def = cmo_barrier_task_def->add_barrier_info();
    cmo_barrier_info_def->set_cmo_type(0);
    cmo_barrier_info_def->set_logic_id(0);
  }
};

TEST_F(UtestCmoBarrierTaskInfo, cmo_barrier_task_init_success) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  CreateCmoBarrierTaskDef(davinci_model, task_def);
  CmoBarrierTaskInfo cmo_barrier_task_info;
  EXPECT_EQ(cmo_barrier_task_info.Init(task_def, &davinci_model), SUCCESS);
}

TEST_F(UtestCmoBarrierTaskInfo, cmo_barrier_task_distribute_success) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  CreateCmoBarrierTaskDef(davinci_model, task_def);
  CmoBarrierTaskInfo cmo_barrier_task_info;
  domi::GetContext().is_online_model = true;
  cmo_barrier_task_info.Init(task_def, &davinci_model);
  EXPECT_EQ(cmo_barrier_task_info.Distribute(), SUCCESS);
  EXPECT_TRUE(cmo_barrier_task_info.IsSupportReDistribute());
  EXPECT_EQ(cmo_barrier_task_info.Distribute(), SUCCESS);
  domi::GetContext().is_online_model = false;
}
}

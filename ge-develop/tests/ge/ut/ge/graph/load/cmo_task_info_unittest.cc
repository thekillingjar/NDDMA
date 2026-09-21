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
#include "graph/load/model_manager/task_info/fe/cmo_task_info.h"
#include "graph/load/model_manager/davinci_model.h"

namespace ge {
class UtestCmoTaskInfo : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}

 public:
  void CreateCmoTaskDef(DavinciModel &davinci_model, domi::TaskDef &task_def) {
    rtStream_t stream = nullptr;
    davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
    davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
    davinci_model.stream_list_ = { stream };

    task_def.set_stream_id(0);
    domi::CmoTaskDef *cmo_task_def = task_def.mutable_cmo_task();
    cmo_task_def->set_cmo_type(1);
    cmo_task_def->set_logic_id(0);
    cmo_task_def->set_qos(0);
    cmo_task_def->set_part_id(0);
    cmo_task_def->set_pmg(0);
    cmo_task_def->set_op_code(0);
    cmo_task_def->set_num_inner(0);
    cmo_task_def->set_num_outer(0);
    cmo_task_def->set_length_inner(0);
    cmo_task_def->set_source_addr(0);
    cmo_task_def->set_strider_outer(0);
    cmo_task_def->set_strider_inner(0);

    davinci_model.runtime_param_.logic_mem_base = 0x8003000;
    davinci_model.runtime_param_.logic_weight_base = 0x8008000;
    davinci_model.runtime_param_.logic_var_base = 0x800e000;
    davinci_model.runtime_param_.mem_size = 0x5000;
    davinci_model.runtime_param_.weight_size = 0x6000;
    davinci_model.runtime_param_.var_size = 0x1000;
  }
};

TEST_F(UtestCmoTaskInfo, cmo_task_init_failed) {
  domi::TaskDef task_def;
  task_def.set_stream_id(0);
  CmoTaskInfo cmo_task_info;
  EXPECT_EQ(cmo_task_info.Init(task_def, nullptr), PARAM_INVALID);
}

TEST_F(UtestCmoTaskInfo, cmo_task_init_success) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  CreateCmoTaskDef(davinci_model, task_def);
  CmoTaskInfo cmo_task_info;
  EXPECT_EQ(cmo_task_info.Init(task_def, &davinci_model), SUCCESS);
}

TEST_F(UtestCmoTaskInfo, success_cmo_task_distribute) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  CreateCmoTaskDef(davinci_model, task_def);
  CmoTaskInfo cmo_task_info;
  domi::GetContext().is_online_model = true;
  cmo_task_info.Init(task_def, &davinci_model);
  EXPECT_EQ(cmo_task_info.Distribute(), SUCCESS);
  EXPECT_TRUE(cmo_task_info.IsSupportReDistribute());
  EXPECT_EQ(cmo_task_info.Distribute(), SUCCESS);
  domi::GetContext().is_online_model = false;
}
}

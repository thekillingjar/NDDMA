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
#include "graph/load/model_manager/task_info/ge/profiler_trace_task_info.h"
#include "common/opskernel/ops_kernel_info_types.h"

namespace ge {

class UtestProfilerTraceTaskInfo : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

// test ProfilerTraceTaskInfo Init.
TEST_F(UtestProfilerTraceTaskInfo, task_init_infer) {
  DavinciModel model(0, nullptr);
  model.SetId(1);
  domi::ModelTaskDef model_task_def;
  domi::TaskDef *task = model_task_def.add_task();
  task->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_PROFILER_TRACE));
  ProfilerTraceTaskInfo task_info;
  task->_impl_.stream_id_ = 0;
  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = { stream };

  EXPECT_EQ(task_info.Init(*task, &model), SUCCESS);
}

TEST_F(UtestProfilerTraceTaskInfo, test_init_train) {
  DavinciModel model(0, nullptr);
  model.SetId(1);
  domi::ModelTaskDef model_task_def;
  domi::TaskDef *task = model_task_def.add_task();
  task->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_PROFILER_TRACE));
  ProfilerTraceTaskInfo task_info;
  domi::GetContext().is_online_model = true;
  task->_impl_.stream_id_ = 0;
  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = { stream };
  EXPECT_EQ(task_info.Init(*task, &model), SUCCESS);
}

// test ProfilerTraceTaskInfo Distribute.
TEST_F(UtestProfilerTraceTaskInfo, test_distribute) {
  DavinciModel model(0, nullptr);
  model.SetId(1);
  domi::ModelTaskDef model_task_def;
  domi::TaskDef *task = model_task_def.add_task();
  task->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_PROFILER_TRACE));
  ProfilerTraceTaskInfo task_info;
  domi::GetContext().is_online_model = false;
  task->_impl_.stream_id_ = 0;
  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = { stream };
  task_info.Init(*task, &model);
  domi::GetContext().is_online_model = true;
  EXPECT_EQ(task_info.Distribute(), SUCCESS);
  EXPECT_TRUE(task_info.IsSupportReDistribute());
  EXPECT_EQ(task_info.Distribute(), SUCCESS);
  domi::GetContext().is_online_model = false;
}

TEST_F(UtestProfilerTraceTaskInfo, test_distribute_out_range) {
  DavinciModel model(0, nullptr);
  model.SetId(1);
  domi::ModelTaskDef model_task_def;
  domi::TaskDef *task = model_task_def.add_task();
  task->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_PROFILER_TRACE));
  ProfilerTraceTaskInfo task_info;
  task->_impl_.stream_id_ = 0;
  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = { stream };
  task_info.Init(*task, &model);
  task_info.log_id_ = 30000U;
  EXPECT_EQ(task_info.Distribute(), SUCCESS);
}

TEST_F(UtestProfilerTraceTaskInfo, TestDistribute_MockFail) {
  DavinciModel model(0, nullptr);
  model.SetId(0xffffffffU);
  domi::ModelTaskDef model_task_def;
  domi::TaskDef *task = model_task_def.add_task();
  task->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_PROFILER_TRACE));
  ProfilerTraceTaskInfo task_info;
  task->_impl_.stream_id_ = 0;
  rtStream_t stream = nullptr;
  model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  model.stream_list_ = { stream };
  task_info.Init(*task, &model);
  task_info.log_id_ = 10001;
  mmSetEnv("CONSTANT_FOLDING_PASS", "mock_fail", 1);
  EXPECT_EQ(task_info.Distribute(), FAILED);
  unsetenv("CONSTANT_FOLDING_PASS");
}
}  // namespace ge

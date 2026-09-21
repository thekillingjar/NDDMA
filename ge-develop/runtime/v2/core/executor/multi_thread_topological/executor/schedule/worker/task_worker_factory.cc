/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_worker_factory.h"
#include "task_worker_impl.h"
#include "core/executor/multi_thread_topological/executor/schedule/queue/lock_free/spsc_queue.h"
#include "core/executor/multi_thread_topological/executor/schedule/queue/lock_free/mpmc_queue.h"

namespace gert {
namespace {
using SpscTaskWorker = TaskWorkerImpl<SpscQueue<ExecTask *>>;
using MpmcTaskWorker = TaskWorkerImpl<MpmcQueue<ExecTask *>>;
}  // namespace

TaskWorker *TaskWorkerFactory::Create(const TaskWorkerConfig &cfg) const {
  GE_ASSERT_TRUE(cfg.thread_count > 0);
  GE_ASSERT_TRUE(cfg.pending_queue_size_log2 > 1);
  GE_ASSERT_TRUE(cfg.completed_queue_size_log2 > 1);

  if (cfg.thread_count == 1) {
    return new (std::nothrow) SpscTaskWorker(cfg);
  }
  return new (std::nothrow) MpmcTaskWorker(cfg);
}
}  // namespace gert

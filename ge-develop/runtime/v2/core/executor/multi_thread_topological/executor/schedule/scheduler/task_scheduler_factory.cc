/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_scheduler_factory.h"
#include <thread>
#include "core/executor/multi_thread_topological/executor/schedule/producer/task_producer_factory.h"
#include "core/executor/multi_thread_topological/executor/schedule/worker/task_worker_factory.h"
#include "task_scheduler.h"
#include "core/executor/multi_thread_topological/executor/schedule/producer/task_producer.h"
#include "common/checker.h"

namespace gert {
namespace {
void ConfigWorkers(TaskScheduler *scheduler, const std::vector<TaskWorkerConfig> &workerConfigs) {
  size_t totalThreadCount = 1U;  // Main Thread

  for (auto &workerCfg : workerConfigs) {
    auto worker = TaskWorkerFactory::GetInstance().Create(workerCfg);
    if (worker != nullptr) {
      scheduler->AddWorker(*worker, workerCfg.bind_task_type);
      totalThreadCount += workerCfg.thread_count;
    }
  }

  if (totalThreadCount > std::thread::hardware_concurrency()) {
    GELOGW("Thread count %ld exceeds the hardware concurrentcy %d", totalThreadCount,
           std::thread::hardware_concurrency());
  }
}
}  // namespace

TaskScheduler *TaskSchedulerFactory::Create(const TaskSchedulerConfig &cfg) {
  auto producer = TaskProducerFactory::GetInstance().Create(cfg.producer_cfg);
  GE_ASSERT_NOTNULL(producer);

  auto scheduler = new (std::nothrow) TaskScheduler(*producer);
  if (!scheduler) {
    delete scheduler;
    return nullptr;
  }

  ConfigWorkers(scheduler, cfg.worker_cfgs);

  return scheduler;
}
}  // namespace gert
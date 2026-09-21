/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_TASK_SCHEDULER_CONFIG_H
#define AIR_CXX_RUNTIME_V2_TASK_SCHEDULER_CONFIG_H

#include <vector>
#include "task_worker_config.h"
#include "task_producer_config.h"

namespace gert {
struct TaskSchedulerConfig {
  void AddWorkers(size_t workerCount, ExecTaskType type, TaskThreadMode mode, size_t threadCount) {
    TaskWorkerConfig workerCfg;
    workerCfg.bind_task_type = type;
    workerCfg.thread_mode = mode;
    workerCfg.thread_count = threadCount;
    producer_cfg.thread_num += threadCount;

    for (size_t i = 0U; i < workerCount; i++) {
      worker_cfgs.emplace_back(workerCfg);
    }
  }

  TaskProducerConfig producer_cfg;
  std::vector<TaskWorkerConfig> worker_cfgs;
};
}  // namespace gert

#endif // AIR_CXX_RUNTIME_V2_TASK_SCHEDULER_CONFIG_H
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "executor/cpu_sched_model.h"
#include "framework/common/debug/log.h"

namespace ge {
void CpuSchedModel::LogModelDesc() const {
  for (int32_t i = 0; i < static_cast<int32_t>(model_info_.queueNum); ++i) {
    GELOGD("queue[%d]: queue_flag = %u, queue_id = %u", i, model_info_.queues[i].flag, model_info_.queues[i].queueId);
  }
  int32_t total_tasks_num = 0;
  int32_t total_streams_num = static_cast<int32_t>(model_info_.aicpuStreamNum);
  for (int32_t i = 0; i < total_streams_num; i++) {
    auto &stream = model_info_.streams[i];
    int32_t tasks_num = static_cast<int32_t>(stream.taskNum);
    for (int32_t j = 0; j < tasks_num; ++j) {
      auto &task = stream.tasks[j];
      GELOGD("Tasks[%d], task_id = %d, stream_id = %d, kernel_name = %s",
             j, task.taskId, stream.streamId, reinterpret_cast<const char_t *>(task.kernelName));
    }
    total_tasks_num += tasks_num;
  }
  GELOGD("model_id = %u, queues_num = %d, streams_num = %d, tasks_num = %d",
         model_info_.modelId,
         static_cast<int32_t>(model_info_.queueNum),
         total_streams_num,
         total_tasks_num);
}
}  // namespace ge

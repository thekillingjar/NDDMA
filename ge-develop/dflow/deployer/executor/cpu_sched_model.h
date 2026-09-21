/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_DEPLOY_EXECUTOR_CPU_SCHED_MODEL_H_
#define AIR_RUNTIME_DEPLOY_EXECUTOR_CPU_SCHED_MODEL_H_

#include <map>
#include <vector>
#include "aicpu/aicpu_schedule/aicpusd_info.h"
#include "framework/common/ge_types.h"
#include "hccl/base.h"

namespace ge {
class CpuSchedModel {
 public:
  void LogModelDesc() const;

 private:
  friend class CpuSchedModelBuilder;
  friend class DynamicModelExecutor;
  ::ModelInfo model_info_{};
  ::ModelCfgInfo model_cfg_{};
  // key: stream_id, value: tasks on stream
  std::map<uint32_t, std::vector<::ModelTaskInfo>> tasks_;
  std::vector<::ModelQueueInfo> queues_;
  std::vector<::ModelStreamInfo> streams_;
  std::vector<std::vector<uint8_t>> task_params_;
  // define for event relation
  std::vector<HcomP2pOpInfo> inputs_events_;
  std::vector<HcomP2pOpInfo> outputs_events_;
};
}  // namespace ge

#endif  // AIR_RUNTIME_DEPLOY_EXECUTOR_CPU_SCHED_MODEL_H_

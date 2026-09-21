/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_DEPLOY_EXECUTOR_NPU_SCHED_MODEL_CONFIGURATOR_H
#define AIR_RUNTIME_DEPLOY_EXECUTOR_NPU_SCHED_MODEL_CONFIGURATOR_H

#include "common/plugin/ge_make_unique_util.h"
#include "ge/ge_api_types.h"

namespace ge {
class NpuSchedModelConfigurator {
 public:
  NpuSchedModelConfigurator() = default;
  ~NpuSchedModelConfigurator() noexcept = default;
  GE_DELETE_ASSIGN_AND_COPY(NpuSchedModelConfigurator);

  struct AicpuModelConfig {
    int32_t version;
    uint32_t model_id;
    uint32_t runtime_model_id;
    int32_t abnormal_exist;
    int32_t abnormal_enqueue;
    int32_t req_msg_queue;
    int32_t resp_msg_queue;
    int8_t rsv[36];
  };
  static Status SetModelConfig(const AicpuModelConfig &config);

 private:
  static Status BuildModelConfigTask(const AicpuModelConfig &config, std::vector<uint8_t> &task_args);
  static Status ExecuteKernel(const std::string &kernel_name, const std::vector<uint8_t> &task_args);
};
}
#endif  // AIR_RUNTIME_DEPLOY_EXECUTOR_NPU_SCHED_MODEL_CONFIGURATOR_H

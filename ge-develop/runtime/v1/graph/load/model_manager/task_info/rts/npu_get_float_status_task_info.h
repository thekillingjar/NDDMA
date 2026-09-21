/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_LOAD_MODEL_MANAGER_TASK_INFO_NPU_GET_FLOAT_STATUS_TASK_INFO_H_
#define GE_GRAPH_LOAD_MODEL_MANAGER_TASK_INFO_NPU_GET_FLOAT_STATUS_TASK_INFO_H_

#include "graph/load/model_manager/task_info/args_io_addrs_updater.h"
#include "graph/load/model_manager/task_info/task_info.h"

namespace ge {
class NpuGetFloatStatusTaskInfo : public TaskInfo {
 public:
  NpuGetFloatStatusTaskInfo() = default;

  ~NpuGetFloatStatusTaskInfo() override = default;

  Status Init(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
              const PisToArgs &args = {}, const PisToPersistentWorkspace &persistent_workspace = {},
              const IowAddrs &iow_addrs = {{}, {}, {}}) override;

  Status Distribute() override;

  Status ParseTaskRunParam(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                           TaskRunParam &task_run_param) override;

  Status GetTaskArgsRefreshInfos(std::vector<TaskArgsRefreshInfo> &infos) override;

  Status UpdateHostArgs(const std::vector<uint64_t> &active_mem_base_addr,
                        void *const host_args,
                        const size_t host_args_max_len) override;

  int64_t ParseOpIndex(const domi::TaskDef &task_def) const override;

 private:
  DavinciModel *davinci_model_{nullptr};
  std::vector<void *> io_addrs_;
  std::vector<uint64_t> io_addr_mem_types_;
  uint8_t *output_addr_{nullptr};
  uint64_t output_addr_mem_type_{0U};
  void *args_{nullptr};
  uint64_t output_size_{0U};
  uint32_t mode_{0U};
  ArgsIoAddrsUpdater args_io_addrs_updater_;
};
}  // namespace ge
#endif  // GE_GRAPH_LOAD_MODEL_MANAGER_TASK_INFO_NPU_GET_FLOAT_STATUS_TASK_INFO_H_

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_MEMCPY_ASYNC_TASK_INFO_H_
#define GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_MEMCPY_ASYNC_TASK_INFO_H_

#include "graph/load/model_manager/task_info/args_io_addrs_updater.h"
#include "graph/load/model_manager/task_info/task_info.h"
#include "graph/op_desc.h"

namespace ge {
class MemcpyAsyncTaskInfo : public TaskInfo {
 public:
  MemcpyAsyncTaskInfo() = default;

  ~MemcpyAsyncTaskInfo() override {
    src_ = nullptr;
    dst_ = nullptr;
  }

  Status Init(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
              const PisToArgs &args = {}, const PisToPersistentWorkspace &persistent_workspace = {},
              const IowAddrs &iow_addrs = {{}, {}, {}}) override;

  Status Distribute() override;

  Status UpdateHostArgs(const std::vector<uint64_t> &active_mem_base_addr,
                        void *const host_args, const size_t host_args_max_len) override;

  Status ParseTaskRunParam(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                           TaskRunParam &task_run_param) override;

  Status GetTaskArgsRefreshInfos(std::vector<TaskArgsRefreshInfo> &infos) override;

  uint32_t GetMemType() override;

  int64_t ParseOpIndex(const domi::TaskDef &task_def) const override;

 private:
  Status SetIoAddrs(const IowAddrs &iow_addrs);

  uint8_t *dst_{nullptr};
  uint64_t dst_max_{0U};
  uint8_t *src_{nullptr};
  uint64_t count_{0U};
  uint64_t logical_src_addr_{0U};
  uint64_t logical_dst_addr_{0U};
  uint64_t logical_src_mem_type_{0U};
  uint64_t logical_dst_mem_type_{0U};
  uint32_t kind_{RT_MEMCPY_RESERVED};
  std::vector<void *> io_addrs_;
  std::vector<uint64_t> io_addr_mem_types_;
  DavinciModel *davinci_model_{nullptr};
  uint32_t op_index_{0U};
  OpDescPtr op_desc_{nullptr};
  uint32_t args_mem_type_{0U};
  uint32_t qosCfg_ = 0U;
  ArgsIoAddrsUpdater args_io_addrs_updater_;
  ArgsPlacement pls_{ArgsPlacement::kArgsPlacementHbm};
};
}  // namespace ge
#endif  // GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_MEMCPY_ASYNC_TASK_INFO_H_

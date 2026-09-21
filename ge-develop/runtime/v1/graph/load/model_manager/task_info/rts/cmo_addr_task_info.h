/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_CMO_ADDR_TASK_INFO_H_
#define GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_CMO_ADDR_TASK_INFO_H_

#include "graph/load/model_manager/task_info/args_io_addrs_updater.h"
#include "graph/load/model_manager/task_info/task_info.h"
#include "graph/op_desc.h"
#include "graph/args_format_desc.h"

namespace ge {
class CmoAddrTaskInfo : public TaskInfo {
 public:
  CmoAddrTaskInfo() = default;

  ~CmoAddrTaskInfo() override = default;

  Status Init(const domi::TaskDef &task_def, DavinciModel *const davinci_model, const PisToArgs &args = {},
              const PisToPersistentWorkspace &persistent_workspace = {},
              const IowAddrs &iow_addrs = {{}, {}, {}}) override;

  Status Distribute() override;

  Status ParseTaskRunParam(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                           TaskRunParam &task_run_param) override;

  Status UpdateHostArgs(const std::vector<uint64_t> &active_mem_base_addr, void *const host_args,
                        const size_t host_args_max_len) override;

  Status GetTaskArgsRefreshInfos(std::vector<TaskArgsRefreshInfo> &infos) override;

  int64_t ParseOpIndex(const domi::TaskDef &task_def) const override;

 private:
  DavinciModel *davinci_model_{nullptr};
  OpDescPtr op_desc_;
  void *args_{nullptr};
  void *host_args_{nullptr};
  ArgsFormatDesc format_;
  uint32_t task_id_{0U};
  uint32_t stream_id_{0U};
  size_t args_size_{0UL};
  size_t format_args_size_{0UL};
  std::vector<uint64_t> io_addrs_;
  std::vector<uint64_t> io_addr_mem_types_;
  size_t io_align_offset_{0UL};
  ArgsPlacement args_placement_{ArgsPlacement::kArgsPlacementHbm};
  rtCmoOpCode_t cmo_op_code_;
  ArgsIoAddrsUpdater args_io_addrs_updater_;
};
}  // namespace ge
#endif  // GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_CMO_ADDR_TASK_INFO_H_

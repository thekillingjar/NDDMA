/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_MEMCPY_ADDR_ASYNC_TASK_INFO_H_
#define GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_MEMCPY_ADDR_ASYNC_TASK_INFO_H_

#include "runtime/mem.h"
#include "graph/op_desc.h"
#include "graph/args_format_desc.h"
#include "graph/load/model_manager/task_info/args_io_addrs_updater.h"
#include "graph/load/model_manager/task_info/task_info.h"

namespace ge {
class MemcpyAddrAsyncTaskInfo : public TaskInfo {
 public:
  Status Init(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
              const PisToArgs &args, const PisToPersistentWorkspace &persistent_workspace,
              const IowAddrs &iow_addrs) override;

  Status Distribute() override;

  Status ParseTaskRunParam(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                           TaskRunParam &task_run_param) override;

  Status GetTaskArgsRefreshInfos(std::vector<TaskArgsRefreshInfo> &infos) override;

  int64_t ParseOpIndex(const domi::TaskDef &task_def) const override;

 private:
  Status HandleZeroCopy(DavinciModel *const davinci_model, uintptr_t base, uint64_t logic_addr);

  OpDescPtr op_desc_{nullptr};
  ArgsFormatDesc format_;
  size_t args_size_{0U};

  uint64_t dst_max_{0U};
  uint64_t count_{0U};
  rtMemcpyKind_t kind_{RT_MEMCPY_ADDR_DEVICE_TO_DEVICE};
  ArgsPlacement pls_{ArgsPlacement::kArgsPlacementHbm};
  uint32_t qosCfg_{0U};

  uint64_t device_args_aligned_{0U};
  void *host_args_aligned_{nullptr};
  size_t aligned_io_offset_{0U};
  ArgsIoAddrsUpdater args_io_addrs_updater_;
};
}  // namespace ge
#endif  // GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_MEMCPY_ADDR_ASYNC_TASK_INFO_H_

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_LOAD_MODEL_MANAGER_TASK_INFO_DSA_TASK_INFO_H_
#define GE_GRAPH_LOAD_MODEL_MANAGER_TASK_INFO_DSA_TASK_INFO_H_

#include "graph/load/model_manager/task_info/args_io_addrs_updater.h"
#include "graph/load/model_manager/task_info/task_info.h"
#include "common/opskernel/ge_task_info.h"

namespace ge {
class DSATaskInfo : public TaskInfo {
 public:
  DSATaskInfo() = default;

  ~DSATaskInfo() override = default;

  Status Init(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
              const PisToArgs &args = {}, const PisToPersistentWorkspace &persistent_workspace = {},
              const IowAddrs &iow_addrs = {{}, {}, {}}) override;

  Status Distribute() override;

  Status Release() override;

  uint32_t GetTaskID() const override { return task_id_; }

  uint32_t GetStreamId() const override { return stream_id_; }

  void PostProcess(const domi::TaskDef &task_def) override;

  Status GetTaskIowPaRemapInfos(std::vector<IowPaRemapInfo> &infos) override;

  Status ParseTaskRunParam(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                           TaskRunParam &task_run_param) override;

  int64_t ParseOpIndex(const domi::TaskDef &task_def) const override;

  Status UpdateHostArgs(const std::vector<uint64_t> &active_mem_base_addr,
                        const std::vector<HostArg> &host_args) override;

  Status GetTaskArgsRefreshInfos(std::vector<TaskArgsRefreshInfo> &infos) override;

  uintptr_t GetDumpArgs() const override {
    return static_cast<uintptr_t>(dump_args_);
  }

 private:
  rtStarsDsaSqe_t dsa_sqe_;
  uint32_t task_id_{0U};
  uint32_t stream_id_{0U};
  uint32_t op_index_{0U};
  OpDescPtr op_desc_{nullptr};
  DavinciModel *davinci_model_{nullptr};
  uint32_t dump_flag_{RT_KERNEL_DEFAULT};
  uint64_t hbm_workspace_args_{0UL};
  uint64_t dump_args_{0UL};
  int64_t hbm_args_len_{0};

  std::vector<uint64_t> input_data_addrs_;
  std::vector<uint64_t> output_data_addrs_;
  std::vector<uint64_t> workspace_data_addrs_;

  std::vector<uint8_t> input_addr_refresh_;
  std::vector<uint8_t> output_addr_refresh_;
  std::vector<uint8_t> workspace_addr_refresh_;
  bool support_refresh_{false};

  std::vector<uint64_t> sqe_io_addrs_;
  size_t stateful_workspace_idx_{0UL};
  std::vector<uint64_t> workspace_io_addrs_;
  ArgsIoAddrsUpdater sqe_args_updater_;         // args placement is sqe
  ArgsIoAddrsUpdater workspace_args_updater_;   // args placement is hbm
  std::vector<uint8_t> sqe_args_refresh_flags_;
  std::vector<uint8_t> hbm_args_refresh_flags_;

  // todo: dsa 独立申请dsa workspace地址时
  std::vector<int64_t> workspace_offset_;
  void *self_wkspace_base_ = nullptr;

  void InitDsaDumpInfo(const OpDescPtr &op_desc);
  void PostDumpProcess(const domi::TaskDef &task_def);
  void PostProfilingProcess(const domi::TaskDef &task_def);
  void GetAddrs(const IowAddrs &iow_addrs);
  Status InitWorkspace(const OpDescPtr &op_desc, const domi::DSATaskDef &dsa_task);
  Status InitSqe(const OpDescPtr &op_desc, const domi::DSATaskDef &dsa_task);
  uint8_t BoolToUint8(const bool val) const { return (val ? 1U : 0U); }
  Status UpdateHostArgsWithSqePlacement(const std::vector<uint64_t> &active_mem_base_addr,
                                        void *const host_args,
                                        const size_t host_args_len) const;

  Status UpdateHostArgsWithHbmPlacement(const std::vector<uint64_t> &active_mem_base_addr,
                                        void *const host_args,
                                        const size_t host_args_len) const;
};
}  // namespace ge
#endif  // GE_GRAPH_LOAD_MODEL_MANAGER_TASK_INFO_DSA_TASK_INFO_H_

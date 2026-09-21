/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_FUSION_TASK_INFO_H_
#define GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_FUSION_TASK_INFO_H_

#include "graph/load/model_manager/task_info/args_io_addrs_updater.h"
#include "graph/args_format_desc.h"
#include "graph/op_desc.h"
#include "graph/def_types.h"
#include "graph/load/model_manager/task_info/task_info.h"
#include "graph/load/model_manager/task_info/args_format/args_format_utils.h"
#include "graph/ge_context.h"
#include "graph/utils/attr_utils.h"
#include "framework/omg/parser/parser_types.h"
#include "framework/common/types.h"
#include "register/op_tiling_registry.h"
#include "common/dump/kernel_tracing_utils.h"

namespace ge {
class FusionTaskInfo : public TaskInfo {
 public:
  FusionTaskInfo() : TaskInfo() {}

  ~FusionTaskInfo() override {
    davinci_model_ = nullptr;
    args_ = nullptr;
    stub_func_ = nullptr;
    tiling_data_addr_ = nullptr;
    dump_args_ = nullptr;
  }

  Status ParseTaskRunParam(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                           TaskRunParam &task_run_param) override;

  Status Init(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
              const PisToArgs &args = {}, const PisToPersistentWorkspace &persistent_workspace = {},
              const IowAddrs &iow_addrs = {{}, {}, {}}) override;
  Status Distribute() override;
  Status GetTaskArgsRefreshInfos(std::vector<TaskArgsRefreshInfo> &infos) override;

  uint32_t GetTaskID() const override { return task_id_; }
  uint32_t GetStreamId() const override { return stream_id_; }
  int64_t ParseOpIndex(const domi::TaskDef &task_def) const override;

  void GetTilingKeyAndData(uint32_t &tiling_key, std::string &tiling_data) const override;

  // dfx func
  void ResetArgsEx() { rt_args_ex_ = rtFusionArgsEx_t{}; }
  void PostProcess(const domi::TaskDef &task_def) override;
  bool CallSaveDumpInfo() const override  { return call_save_dump_; }
  uintptr_t GetDumpArgs() const override { return static_cast<uintptr_t>(PtrToValue(dump_args_)); }
  uintptr_t GetArgs() const override { return static_cast<uintptr_t>(PtrToValue(rt_args_ex_.args) ); }
  size_t GetArgSize() const override { return static_cast<size_t>(rt_args_ex_.argsSize); }
  std::map<uint64_t, uint64_t> GetCustToRelevantOffset() const override {
    return cust_to_relevant_offset_;
  }

 private:
  struct ArgsFormatInfo {
    std::map<size_t, std::pair<size_t, size_t>> ir_input_2_range;
    std::map<size_t, std::pair<size_t, size_t>> ir_output_2_range;
    std::vector<ArgDesc> arg_descs;
    std::vector<std::vector<int64_t>> shape_infos;
    size_t level1_addr_cnt{0UL};
  };

  Status CopyTilingDataIfNeeded();
  void UpdateIoAndWorkspaceAddrs(const IowAddrs &iow_addrs);
  Status InitArgs(const PisToArgs &args);
  Status InitKernel(const domi::TaskDef &task_def, const PisToArgs &args);

  Status ParseArgsFormat(ArgsFormatInfo &args_format_info);
  size_t GetArgsSizeByFormat(ArgsFormatInfo &args_format_info) const;
  bool HasOverflowAddr(const OpDescPtr &op_desc) const;
  Status AssembleIoByArgsFormat(const ArgsFormatInfo &args_format_info);
  Status AssembleShapeInfoAddrs(const std::vector<ArgDesc> &dynamic_args_desc,
                                const std::vector<size_t> &level2_addr_idx,
                                const ArgsFormatInfo &args_format_info);

  Status SetAicoreTaskHandle(rtAicoreFusionInfo_t &aicore_fusion_info);
  Status SetAicoreTaskStubFunc(rtAicoreFusionInfo_t &aicore_fusion_info);

  void AppendIoAddr(const uint64_t addr, const uint64_t addr_type);
  Status AppendWorkspaceAddr(int32_t ir_idx);
  Status AppendInputOutputAddrByInstanceIndex(size_t ins_idx, bool is_input);
  Status AppendInputOutputAddr(size_t ir_idx, bool is_input, const ArgsFormatInfo &args_format_info);

  Status SetAIcoreLaunchAttrs(const domi::LaunchConfig& cfg, rtAicoreFusionInfo_t &aicore_fusion_info);

  Status SetTvmTaskZeroCopy(const OpDescPtr &op_desc, const std::vector<uint64_t> &virtual_io_addrs, void *args) const;

  void UpdateTaskId();

  uint32_t task_id_{0U};
  uint32_t stream_id_{0U};
  ModelTaskType task_type_ = ModelTaskType::MODEL_TASK_KERNEL;
  OpDescPtr op_desc_;   // Clear after distribute.
  DavinciModel *davinci_model_{nullptr};

  bool is_all_kernel_{false};
  void *args_{nullptr};
  uint32_t args_size_{};
  ArgsFormatInfo args_format_info_;
  std::vector<uint64_t> io_addrs_;
  std::vector<uint64_t> io_addr_mem_types_;
  std::vector<uint64_t> input_data_addrs_;
  std::vector<uint64_t> output_data_addrs_;
  std::vector<uint64_t> workspace_addrs_;
  std::vector<uint64_t> input_mem_types_;
  std::vector<uint64_t> output_mem_types_;
  std::vector<uint64_t> workspace_mem_types_;
  ArgsPlacement args_placement_{ArgsPlacement::kArgsPlacementHbm};
  ArgsIoAddrsUpdater args_io_addrs_updater_;

  rtFunsionTaskInfo_t rt_fusion_task_{};
  rtFusionArgsEx_t rt_args_ex_{};

  rtLaunchConfig_t rt_launch_config_{};
  std::vector<rtLaunchAttribute_t> launch_attr_list_;

  // 静态
  void *stub_func_{nullptr};
  // 动态
  uint64_t tiling_key_ = 0U;
  void *tiling_data_addr_{nullptr};
  size_t tiling_data_size_ = 0UL;
  bool has_tiling_{false};
  std::string tiling_data_host_;

  // dfx
  uint32_t dump_flag_{RT_KERNEL_DEFAULT};
  void *dump_args_{nullptr};
  bool call_save_dump_{false};
  std::vector<uint64_t> l0_dump_list_;
  std::map<uint64_t, uint64_t> cust_to_relevant_offset_;
};
}  // namespace ge
#endif  // GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TASK_INFO_FUSION_TASK_INFO_H_
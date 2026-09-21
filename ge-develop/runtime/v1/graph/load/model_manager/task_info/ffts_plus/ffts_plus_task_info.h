/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_LOAD_MODEL_MANAGER_TASK_INFO_FFTS_PLUS_TASK_INFO_H_
#define GE_GRAPH_LOAD_MODEL_MANAGER_TASK_INFO_FFTS_PLUS_TASK_INFO_H_

#include "graph/op_desc.h"
#include "graph/load/model_manager/task_info/task_info.h"
#include "graph/load/model_manager/task_info/ffts_plus/ffts_plus_proto_transfer.h"
#include "graph/load/model_manager/task_info/ffts_plus/ffts_plus_args_helper.h"
#include "hybrid/node_executor/aicpu/aicpu_ext_info_handler.h"
#include "common/op_tiling/op_tiling_rt2.h"

namespace ge {
class FftsPlusTaskInfo : public TaskInfo {
 public:
  FftsPlusTaskInfo() = default;
  ~FftsPlusTaskInfo() override = default;

  Status Init(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
              const PisToArgs &args = {}, const PisToPersistentWorkspace &persistent_workspace = {},
              const IowAddrs &iow_addrs = {{}, {}, {}}) override;
  Status Distribute() override;
  Status Release() override;
  Status UpdateHostArgs(const std::vector<uint64_t> &active_mem_base_addr,
                        void *const host_args, const size_t host_args_max_len) override;
  Status ParseTaskRunParam(const domi::TaskDef &task_def, DavinciModel *const davinci_model,
                           TaskRunParam &task_run_param) override;
  Status GetTaskArgsRefreshInfos(std::vector<TaskArgsRefreshInfo> &infos) override;
  uint32_t GetTaskID() const override { return task_id_; }
  uint32_t GetStreamId() const override { return stream_id_; }
  Status CalculateTilingDataSize(const OpDescPtr &op_desc, const bool is_atomic_op_type);
  Status HandleSoftSyncOp(const uint32_t op_index, const OpDescPtr &op_desc) const;

  bool CallSaveDumpInfo() const override { return true; }

  void PostProcess(const domi::TaskDef &task_def) override;

  const std::vector<FusionOpInfo> &GetAllFusionOpInfo() const override { return fusion_op_info_; }
  int64_t ParseOpIndex(const domi::TaskDef &task_def) const override;

  std::map<uint64_t, uint64_t> GetCustToRelevantOffset() const override {
    if (ffts_flus_args_helper_ != nullptr) {
      return ffts_flus_args_helper_->GetCustToRelevantOffset();
    }
    return {};
  }

 private:
  void SetTransferCallback(FftsPlusProtoTransfer &transfer);
  Status SetCachePersistentWay(const OpDescPtr &op_desc) const;
  void InitDumpArgs(const OpDescPtr &op_desc, const uintptr_t op_args, const size_t args_offset,
                    const std::vector<uintptr_t> &first_level_args);
  uintptr_t FindDumpArgs(const std::string &op_name) const;
  bool OpNeedDump(const OpDescPtr &op_desc) const;
  void SavePrintOrDumpTask(const OpDescPtr &op_desc, const domi::FftsPlusCtxDef &ctx_def, const uintptr_t &dump_args,
                           const ModelTaskType task_type);
  Status CreateAicpuSession(STR_FWK_OP_KERNEL &fwk_op_kernel) const;
  Status LoadCustAicpuSo(const OpDescPtr &op_desc, const domi::FftsPlusAicpuCtxDef &ctx_def) const;
  Status InitShapeInfosArgs(const domi::FftsPlusCtxDef &ctx_def) const;
  Status ParseArgsFormat(const domi::FftsPlusCtxDef &ctx_def) const;
  Status TilingDataHandle(const domi::FftsPlusCtxDef &ctx_def,
                          const OpDescPtr &op_desc) const;
  Status InitTilingInfo();
  Status InitArgsBaseInfo(const PisToArgs &args);
  void InitDescBufInfo();
  Status AssembleOtherArgsByArgsBase(const domi::FftsPlusTaskDef &ffts_plus_task_def);
  Status PrePareForTransfer(const domi::TaskDef &task_def);
  void CalculateAscendAicpuKernelSize(const domi::FftsPlusCtxDef &ctx_dex);
  DavinciModel *davinci_model_{nullptr};
  rtFftsPlusTaskInfo_t ffts_plus_task_info_{};

  // |--dynamic_io_shape_desc_buffer--|--aligned_mem--|--ctx_desc_buffer--|
  // |--aligned_mem--|--args_for_aicaiv--|--args_for_ascend_aicpu--|
  size_t pis_args_size_{0UL};
  uint8_t *pis_args_dev_base_{nullptr};
  uint8_t *pis_args_host_base_{nullptr};

  // ioaddr_base
  uint8_t *args_{nullptr}; // runtime args memory
  uint8_t *args_host_{nullptr};
  size_t args_size_{0U}; // runtime args memory length
  size_t dsa_workspace_size_{0U};
  // ascend aicpu
  size_t bin_args_size_{0UL};
  // for desc_buf
  uint8_t *desc_buf_host_{nullptr};
  size_t desc_buffer_len_{0UL};
  size_t desc_buf_aligned_size_{0UL};

  uint32_t dump_flag_{RT_KERNEL_DEFAULT};
  std::map<std::string, uintptr_t> dump_op_args_;
  std::map<std::string, std::vector<uintptr_t>> dump_op_2_first_level_args_;
  std::map<std::string, size_t> dump_args_offset_;
  uint32_t task_id_{0xFFFFFFFFU};
  uint32_t stream_id_{0xFFFFFFFFU};

  std::vector<void *> ext_info_addrs_;
  std::vector<FusionOpInfo> fusion_op_info_;
  OpDescPtr op_desc_{nullptr};
  std::unique_ptr<FftsPlusArgsHelper> ffts_flus_args_helper_{nullptr};
  std::unique_ptr<FftsPlusProtoTransfer> ffts_proto_transfer_{nullptr};

  // for tiling
  size_t tiling_data_len_{0U};
  ArgsPlacement args_placement_{ArgsPlacement::kArgsPlacementHbm};

  std::vector<uint64_t> l0_dump_list_; // 保存task内所有context的size info
};
}  // namespace ge
#endif  // GE_GRAPH_LOAD_MODEL_MANAGER_TASK_INFO_FFTS_PLUS_TASK_INFO_H_

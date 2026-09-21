/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_LOAD_MODEL_MANAGER_TASK_INFO_FFTS_PLUS_PROTO_TRANSFER_H_
#define GE_GRAPH_LOAD_MODEL_MANAGER_TASK_INFO_FFTS_PLUS_PROTO_TRANSFER_H_

#include <vector>
#include <functional>
#include <string>
#include <utility>

#include "ge/ge_api_error_codes.h"
#include "graph/op_desc.h"
#include "runtime/rt.h"
#include "proto/task.pb.h"
#include "graph/load/model_manager/task_info/task_info.h"
#include "graph/load/model_manager/task_info/ffts_plus/ffts_plus_args_helper.h"
#include "graph/load/model_manager/davinci_model.h"
#include "hybrid/node_executor/aicpu/aicpu_ext_info_handler.h"
#include "common/op_tiling/op_tiling_rt2.h"

namespace ge {
void CleanRtFftsPlusTask(rtFftsPlusTaskInfo_t &ffts_plus_task_info) noexcept;
using FftsRunAddrHandle = std::function<Status(const uintptr_t logic_addr, uint8_t *&addr, uint64_t &mem_type)>;
using FftsAddrPrefHandle =
    std::function<Status(const OpDescPtr &op_desc, const std::string &kernel_name, const std::string &prefix,
                         std::vector<std::pair<void *, uint32_t>> &addr_pref_cnt)>;
using FftsFindNodeHandle = std::function<OpDescPtr(const uint32_t op_index)>;
using FftsSaveCtxArgsHandle =
    std::function<void(const OpDescPtr &op_desc, const uintptr_t op_args, const size_t args_offset,
                       const std::vector<uintptr_t> &first_level_args)>;
using FftsCreateAicpuSession = std::function<Status(STR_FWK_OP_KERNEL &fwk_op_kernel)>;
using FftsLoadCustAiCpuSo = std::function<Status(const OpDescPtr &op_desc, const domi::FftsPlusAicpuCtxDef &ctx_def)>;
using FftsSaveAicpuCtxHandle = std::function<Status(const OpDescPtr &op_desc, const domi::aicpuKernelDef &kernel_def)>;
using FftsGetEventIdHandle = std::function<ge::Status(const ge::OpDescPtr &op_desc, uint32_t &event_id)>;
using FftsIsDumpHandle = std::function<bool(const ge::OpDescPtr &op_desc)>;
using FftsSaveL0DumpInfoHandle = std::function<void(const std::vector<uint64_t> &l0_dump_list)>;

struct DsaUpdateCtxHelper {
  std::vector<uint64_t> input_data_addrs;
  std::vector<uint64_t> output_data_addrs;
  std::vector<uint64_t> workspace_data_addrs;
  std::vector<TaskArgsRefreshInfo> input_data_refresh_infos;
  std::vector<TaskArgsRefreshInfo> output_data_refresh_infos;
  std::vector<TaskArgsRefreshInfo> workspace_data_refresh_infos;
  void Refresh() {
    input_data_addrs.clear();
    output_data_addrs.clear();
    workspace_data_addrs.clear();
    input_data_refresh_infos.clear();
    output_data_refresh_infos.clear();
    workspace_data_refresh_infos.clear();
  }
  bool IsAllRealAddrReady() const {
    // dsa的地址排布是输入，workspace，输出，如果输出的个数已经达到指定个数，则dsa的所有地址已经ready，可以进行ctx的刷新流程
    return (output_data_addrs.size() == kDSAOutputAddrSize) || (output_data_refresh_infos.size() == kDSAOutputAddrSize);
  }
};

class FftsPlusProtoTransfer {
 public:
  FftsPlusProtoTransfer(const uintptr_t args_base,
                        FftsPlusArgsHelper *const ffts_plus_args_helper,
                        const RuntimeParam &rts_param, std::vector<void *> &ext_args)
    : args_base_(args_base),
      ffts_plus_args_helper_(ffts_plus_args_helper), ext_info_addrs_(ext_args),
      runtime_param_(rts_param) {
  }
  ~FftsPlusProtoTransfer() = default;

  Status Transfer(const OpDescPtr &op_desc, const domi::FftsPlusTaskDef &ffts_plus_task_def,
                  rtFftsPlusTaskInfo_t &ffts_plus_task_info, uint8_t *const desc_buf = nullptr,
                  const size_t desc_buf_len = 0UL);

  void SetRunAddrHandle(const FftsRunAddrHandle &handle) { run_addr_handle_ = handle; }
  void SetAddrPrefHandle(const FftsAddrPrefHandle &handle) { addr_pref_handle_ = handle; }
  void SetFindNodeHandle(const FftsFindNodeHandle &handle) { find_node_handle_ = handle; }
  void SetSaveCtxArgsHandle(const FftsSaveCtxArgsHandle &handle) { save_ctx_args_handle_ = handle; }
  void SetGetSessionId(const std::function<uint64_t(void)> &handle) { aicpu_get_session_id_ = handle; }
  void SetCreateAicpuSession(const FftsCreateAicpuSession &handle) { create_aicpu_session_ = handle; }
  void SetLoadCustAicpuSo(const FftsLoadCustAiCpuSo &handle) { load_cust_aicpu_so_ = handle; }
  void SetSaveAicpuCtxHandle(const FftsSaveAicpuCtxHandle &handle) { save_aicpu_ctx_handle_ = handle; }
  void SetGetEventIdHandle(const FftsGetEventIdHandle &handle) { get_event_id_ = handle; }
  void SetIsDumpHandle(const FftsIsDumpHandle &handle) { is_dump_ = handle; }
  void SetSaveL0DumpInfoHandle(const FftsSaveL0DumpInfoHandle &handle) { save_l0_dump_info_handle_ = handle; }
  void SetDavinciModel(DavinciModel *davinci_model) { davinci_model_ = davinci_model; }
  const std::vector<FusionOpInfo> &GetAllFusionOpInfo() const { return fusion_op_info_; }
  Status GenCtxLevel1RefreshInfo(const AddrType2CtxAddrInfo &addr_type_2_ctx_addr_infos,
                                 const std::vector<TaskArgsRefreshInfo> &args_fresh_infos,
                                 const size_t start_idx,
                                 const size_t real_addr_size,
                                 std::vector<TaskArgsRefreshInfo> &ctx_level1_args_fresh_infos);
  Status UpdateCtxLevel1Addrs(const AddrType2CtxAddrInfo &addr_type_2_ctx_addr_infos,
                              const std::vector<uint64_t> &real_addrs, const size_t start_idx,
                              const size_t real_addr_size);
  /**
   * addr_pref_cnt.size()为1代表1:0或者0:1的mix_enhanced的only aic 或者 only aiv场景，
   * GE的代码里面感知1到底是aic还是aiv的这个细节不合适，无脑复制一份，aic和aiv刷成一样的
   * @param op_desc
   * @param addr_pref_cnt
   */
  static void ExpandMixOnlyInfos(const OpDesc &op_desc, std::vector<std::pair<void *, uint32_t>> &addr_pref_cnt);
  void SetPisArgsHostBase(const uintptr_t pis_args_host_base) { pis_args_host_base_ = pis_args_host_base; }
 private:
  Status UpdateEventIdForAicpuBlockingOp(const OpDescPtr &op_desc,
                                         const std::shared_ptr<ge::hybrid::AicpuExtInfoHandler> &ext_handle) const;
  Status CheckDeviceSupportBlockingAicpuOpProcess(bool &is_support) const;
  void InitFftsPlusArgs(const domi::FftsPlusCtxDef &ctx_def);
  Status InitFftsPlusCtx(const domi::FftsPlusTaskDef &task_def, uint8_t *const ctx, const int32_t num);

  Status InitAicAivCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx);
  Status InitManualAicAivCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusAicAivCtx_t &ctx);
  Status InitAutoAicAivCtx(const domi::FftsPlusAicAivCtxDef &ctx_def, rtFftsPlusAicAivCtx_t &ctx) const;

  Status InitWriteValueCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx) const;

  Status InitMixAicAivCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx);
  Status InitManualMixAicAivCtx(const domi::FftsPlusMixAicAivCtxDef &ctx_def,
                                const std::vector<std::string> &kernel_name_prefixes,
                                rtFftsPlusMixAicAivCtx_t &ctx, const int32_t start_idx = 0);
  Status InitAutoMixAicAivCtx(const domi::FftsPlusMixAicAivCtxDef &ctx_def,
                              rtFftsPlusMixAicAivCtx_t &ctx, const int32_t start_idx = 0) const;

  Status InitSdmaCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx) const;

  Status InitDataCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx) const;

  Status InitAicpuCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx) const;

  Status InitAicpuFwkAddrInfo(const OpDescPtr &op_desc, uint8_t *const ori_args_addr, const size_t args_size) const;
  Status InitAicpuInfo(const OpDescPtr &op_desc, const domi::FftsPlusAicpuCtxDef &ctx_def, void *&addr) const;
  Status InitAicpuFwkExtInfo(const OpDescPtr &op_desc, const domi::FftsPlusAicpuCtxDef &ctx_def, void *&addr) const;
  Status InitAicpuExtInfo(const OpDescPtr &op_desc, const domi::FftsPlusAicpuCtxDef &ctx_def, void *&addr) const;
  Status InitAicpuTaskExtInfo(const OpDescPtr &op_desc, const std::string &ext_info,
                              std::shared_ptr<hybrid::AicpuExtInfoHandler> &ext_handle) const;
  Status CopyTaskInfoToWorkspace(const OpDescPtr &op_desc, const void *const task_info_addr,
                                 const size_t task_info_addr_size) const;
  Status InitAicpuIoAddrs(const OpDescPtr &op_desc, const uint64_t io_addr, const size_t io_size,
                          const size_t addr_inner_offset) const;

  Status InitCondSwitchCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx) const;
  Status InitCaseCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx) const;

  void GetScheduleModeFromRunInfo(uint32_t &schedule_mode) const;
  Status InitDsaCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx) const;

  void AppendCtxLevel1RrefreshInfo(const TaskArgsRefreshInfo &info,
                                   const uint32_t *addr_high,
                                   const uint32_t *addr_low,
                                   const OpDescPtr &op_desc,
                                   const Level1AddrType addr_type);

  void AppendDsaCtxLevel1RrefreshInfo(const OpDescPtr &op_desc,
                                      const domi::FftsPlusDsaCtxDef &ctx_def,
                                      rtFftsPlusDsaCtx_t *const ctx);

  void AssembleDsaCtxByRealAddr(const OpDescPtr &op_desc, const domi::FftsPlusDsaCtxDef &ctx_def,
                                const std::vector<uint64_t> &input_data_addrs,
                                const std::vector<uint64_t> &output_data_addrs,
                                const std::vector<uint64_t> &workspace_data_addrs, rtFftsPlusDsaCtx_t *const ctx) const;

  Status InitOpRunInfo(size_t &tiling_data_pos, uint8_t *tiling_addr);

  Status SetTilingDataAddr(size_t &tiling_data_len);

  Status GetAndCheckDsaAddr(const OpDescPtr &op_desc, std::vector<std::pair<uint64_t, uint64_t>> &input_data_addrs,
                            std::vector<std::pair<uint64_t, uint64_t>> &output_data_addrs,
                            std::vector<std::pair<uint64_t, uint64_t>> &workspace_data_addrs) const;

  void SaveFirstLevelAddressDumpInfo(const OpDescPtr &op_desc, const std::vector<uint64_t> &input_data_addrs,
                                     const std::vector<uint64_t> &output_data_addrs) const;

  Status InitDsaWorkSpace(const OpDescPtr &op_desc,
                          const domi::FftsPlusDsaCtxDef &ctx_def,
                          const std::vector<std::pair<uint64_t, uint64_t>> &input_datas) const;

  Status AssembleDsaWorkSpaceByInput(const OpDescPtr &op_desc, const domi::FftsPlusDsaCtxDef &ctx_def,
                                     const vector<uint64_t> &input_data_addrs,
                                     const uint64_t workspace_input_addr) const;

  void InitAdditionalData(const domi::FftsPlusTaskDef &task_def);

  template<typename T>
  Status InitThreadIoAddrs(const T &ctx_def, const uint32_t thread_dim, const int32_t start_idx = 0) const;

  template<typename T>
  Status GetThreadIoAddr(ge::AttrHolder &obj, const T &ctx_def, std::vector<uint64_t> &addr_offset,
                         const uint32_t thread_id, const int32_t idx, const int32_t start_idx) const;

  Status ConstructArgsFromFormat(const domi::FftsPlusMixAicAivCtxDef &ctx_def, const ArgsFormatHolder &format_holder);

  Status AssembleTilingContextArgs(const ArgDesc &arg_desc);

  ge::OpDescPtr op_desc_;
  uintptr_t pis_args_host_base_{0U};
  uintptr_t args_base_{0U};     // runtime args memory
  FftsPlusArgsHelper *ffts_plus_args_helper_{nullptr};

  // for static shape reuse tiling data
  bool is_atomic_op_type_{false};
  std::vector<uint8_t> tiling_addr_base_;
  uint8_t *tiling_addr_base_dev_{nullptr};
  uint32_t block_dim_ = 0U;
  std::unordered_map<std::shared_ptr<optiling::OpRunInfoV2>,
      std::unordered_map<std::string, size_t>> tiling_info_addr_pos_;
  int32_t device_id_{0};

  std::vector<void *> &ext_info_addrs_;
  const RuntimeParam &runtime_param_;
  DavinciModel *davinci_model_{nullptr};
  uint32_t logic_stream_id_{0xFFFFFFFFU};
  std::vector<FusionOpInfo> fusion_op_info_;
  std::map<uint32_t, std::set<uint32_t>> ctx_additional_data_;
  FftsFindNodeHandle find_node_handle_{nullptr};
  FftsRunAddrHandle run_addr_handle_{nullptr};
  FftsAddrPrefHandle addr_pref_handle_{nullptr};
  FftsSaveCtxArgsHandle save_ctx_args_handle_{nullptr};
  FftsGetEventIdHandle get_event_id_{nullptr};
  FftsIsDumpHandle is_dump_{nullptr};
  FftsSaveL0DumpInfoHandle save_l0_dump_info_handle_{nullptr};
  DsaUpdateCtxHelper dsa_update_ctx_helper_;
  std::vector<TaskArgsRefreshInfo> ctx_level1_refresh_info_list_;
  std::function<uint64_t(void)> aicpu_get_session_id_ = []()->uint64_t { return 0U; };
  FftsCreateAicpuSession create_aicpu_session_ = [](STR_FWK_OP_KERNEL &kernel_def)->uint64_t {
    (void)kernel_def;
    return 0U;
  };
  FftsLoadCustAiCpuSo load_cust_aicpu_so_ =
      [](const OpDescPtr &op_desc, const domi::FftsPlusAicpuCtxDef &kernel_def)->uint64_t {
        (void)op_desc;
        (void)kernel_def;
        return 0U;
      };
  FftsSaveAicpuCtxHandle save_aicpu_ctx_handle_ =
      [](const OpDescPtr &op_desc, const domi::aicpuKernelDef &kernel_def)->uint64_t {
        (void)op_desc;
        (void)kernel_def;
        return 0U;
      };

  using CtxHandle = std::function<Status(FftsPlusProtoTransfer *, const domi::FftsPlusCtxDef &,
                                         rtFftsPlusComCtx_t *const)>;
  static std::map<rtFftsPlusContextType_t, CtxHandle> init_ctx_fun_;

  std::vector<uint64_t> l0_dump_list_; // 保存当前处理的context的size info
};
}  // namespace ge
#endif  // GE_GRAPH_LOAD_MODEL_MANAGER_TASK_INFO_FFTS_PLUS_PROTO_TRANSFER_H_

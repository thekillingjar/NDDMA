/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_ENGINE_FFTS_PLUS_FFTS_PLUS_PROTO_TRANSFER_H_
#define AIR_CXX_RUNTIME_V2_ENGINE_FFTS_PLUS_FFTS_PLUS_PROTO_TRANSFER_H_

#include <vector>
#include <functional>
#include <string>
#include "ge/ge_api_error_codes.h"
#include "graph/op_desc.h"
#include "runtime/rt.h"
#include "proto/task.pb.h"
#include "graph/load/model_manager/task_info/task_info.h"
#include "hybrid/node_executor/aicpu/aicpu_ext_info_handler.h"

namespace gert {
void CleanRtFftsPlusTask(rtFftsPlusTaskInfo_t &ffts_plus_task_info);
using FftsRunAddrHandle = std::function<ge::Status(const uintptr_t logic_addr, uint8_t *&addr)>;
using FftsAddrPrefHandle = std::function<ge::Status(const std::string &kernel_name, void *&addr, uint32_t &pref_cnt)>;
using FftsFindNodeHandle = std::function<ge::OpDescPtr(const uint32_t op_index)>;
using FftsSaveCtxArgsHandle = std::function<void(const ge::OpDescPtr &op_desc, const size_t args_offset)>;
using FftsCreateAicpuSession = std::function<ge::Status(STR_FWK_OP_KERNEL &fwk_op_kernel)>;
using FftsLoadCustAiCpuSo = std::function<ge::Status(const ge::OpDescPtr &op_desc,
                                                     const domi::FftsPlusAicpuCtxDef &ctx_def)>;
using FftsSaveAicpuCtxHandle = std::function<ge::Status(const ge::OpDescPtr &op_desc,
                                                        const domi::aicpuKernelDef &kernel_def)>;
class FftsPlusProtoTransfer {
 public:
  FftsPlusProtoTransfer(const uintptr_t args_base, std::vector<uintptr_t> &io_addrs, const ge::RuntimeParam &rts_param,
                        std::vector<void *> &ext_args, std::vector<size_t> &mode_addr_idx)
    : args_base_(args_base), io_addrs_(io_addrs), ext_info_addrs_(ext_args), mode_addr_idx_(mode_addr_idx),
      runtime_param_(rts_param) {
  }
  ~FftsPlusProtoTransfer();

  std::unique_ptr<uint8_t[]> Transfer(const ge::OpDescPtr &op_desc, const domi::FftsPlusTaskDef &ffts_plus_task_def,
                      size_t &total_size);

  void SetFindNodeHandle(const FftsFindNodeHandle &handle) { find_node_handle_ = handle; }
  // for aicpu dynamic shape
  void SetSaveAicpuCtxHandle(const FftsSaveAicpuCtxHandle &handle) { save_aicpu_ctx_handle_ = handle; }

 private:
  void InitFftsPlusSqe(const domi::FftsPlusSqeDef &sqe_def, rtFftsPlusSqe_t *const sqe) const;
  void InitFftsPlusSqeHeader(const domi::StarsSqeHeaderDef &sqe_header_def, rtStarsSqeHeader_t &sqe_header) const;
  ge::Status InitFftsPlusCtx(const domi::FftsPlusTaskDef &task_def, uint8_t *const ctx, const int32_t num);
  void FusionOpPreProc(const ge::OpDescPtr op_desc, const domi::FftsPlusCtxDef &ctx_def,
                       rtFftsPlusContextType_t ctx_type);
  ge::Status InitPersistentCacheCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx) const;
  ge::Status InitAicAivCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx) const;
  ge::Status InitManualAicAivCtx(const domi::FftsPlusAicAivCtxDef &ctx_def, rtFftsPlusAicAivCtx_t &ctx) const;
  ge::Status InitAutoAicAivCtx(const domi::FftsPlusAicAivCtxDef &ctx_def, rtFftsPlusAicAivCtx_t &ctx) const;

  ge::Status InitNotifyCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx) const;

  ge::Status InitWriteValueCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx) const;

  ge::Status InitMixAicAivCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx) const;
  ge::Status InitManualMixAicAivCtx(const domi::FftsPlusMixAicAivCtxDef &ctx_def,
                                const std::vector<std::string> &kernel_name_prefixes,
                                rtFftsPlusMixAicAivCtx_t &ctx, const int32_t start_idx = 0) const;
  ge::Status InitAutoMixAicAivCtx(const domi::FftsPlusMixAicAivCtxDef &ctx_def,
                              rtFftsPlusMixAicAivCtx_t &ctx, const int32_t start_idx = 0) const;

  ge::Status InitSdmaCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx) const;

  ge::Status InitDataCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx) const;

  ge::Status InitAicpuCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx) const;
  ge::Status InitAicpuCtxUserData(const ge::OpDescPtr &op_desc, const domi::FftsPlusAicpuCtxDef &ctx_def,
                              rtFftsPlusAiCpuCtx_t &ctx) const;

  ge::Status InitAicpuInfo(const ge::OpDescPtr &op_desc, const domi::FftsPlusAicpuCtxDef &ctx_def, void *&addr) const;
  ge::Status InitAicpuExtInfo(const ge::OpDescPtr &op_desc, const domi::FftsPlusAicpuCtxDef &ctx_def,
                              const void* const& addr) const;
  ge::Status InitAicpuTaskExtInfo(const ge::OpDescPtr &op_desc, const std::string &ext_info,
                              std::shared_ptr<ge::hybrid::AicpuExtInfoHandler> &ext_handle) const;
  ge::Status InitAicpuIoAddrs(const ge::OpDescPtr &op_desc, const uintptr_t &io_addr, const size_t io_size) const;

  ge::Status InitCondSwitchCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx) const;

  ge::Status InitCaseSwitchCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx) const;
  ge::Status InitCaseDefaultCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx) const;
  ge::Status InitCaseCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx) const;

  ge::Status InitAtStartCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx) const;

  ge::Status InitAtEndCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx) const;

  ge::Status InitLabelCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx) const;

  void InitAdditionalData(const domi::FftsPlusTaskDef &task_def);

  template<typename T>
  ge::Status InitThrdIoAddrs(const T &ctx_def, const uint16_t thread_id, const int32_t addr_count,
                       const int32_t start_idx = 0) const;

  uintptr_t args_base_{0U};     // runtime args memory
  std::vector<uintptr_t> &io_addrs_;
  std::vector<void *> &ext_info_addrs_;
  std::vector<size_t> &mode_addr_idx_;
  const ge::RuntimeParam &runtime_param_;
  uint32_t logic_stream_id_{0xFFFFFFFFU};
  std::vector<ge::FusionOpInfo> fusion_op_info_;
  std::map<uint32_t, std::set<uint32_t>> ctx_additional_data_;
  uint8_t sub_type_{std::numeric_limits<uint8_t>::max()};
  FftsFindNodeHandle find_node_handle_{nullptr};
  FftsRunAddrHandle run_addr_handle_{nullptr};
  FftsAddrPrefHandle addr_pref_handle_{nullptr};
  FftsSaveCtxArgsHandle save_ctx_args_handle_{nullptr};
  FftsSaveAicpuCtxHandle save_aicpu_ctx_handle_ =
      [](const ge::OpDescPtr &, const domi::aicpuKernelDef &) { return 0U; };

  using CtxHandle = std::function<ge::Status(FftsPlusProtoTransfer *, const domi::FftsPlusCtxDef &,
                                         rtFftsPlusComCtx_t *const)>;
  static std::map<rtFftsPlusContextType_t, CtxHandle> init_ctx_fun_;
  rtFftsPlusTaskInfo_t *ffts_plus_task_info_{nullptr};
};

enum class InfoStType : int16_t {
  kDescBuf,
  kInfoStTypeEnd
};
/*
 * ————————————————Memory---------------------
 * TransTaskInfo || rtFftsPlusSqe_t || descBuf
 *
 * */
struct TransTaskInfo {
  size_t offsets[static_cast<size_t>(InfoStType::kInfoStTypeEnd)];
  rtFftsPlusTaskInfo_t rt_task_info{};
  uint8_t args[1];
};
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_ENGINE_FFTS_PLUS_FFTS_PLUS_PROTO_TRANSFER_H_

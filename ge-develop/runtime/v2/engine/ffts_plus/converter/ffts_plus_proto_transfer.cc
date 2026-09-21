/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/checker.h"
#include "securec.h"
#include "ffts_plus_common.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/load/model_manager/model_utils.h"
#include "mmpa/mmpa_api.h"
#include "aicpu_task_struct.h"
#include "engine/aicore/fe_rt2_common.h"
#include "framework/common/ge_inner_error_codes.h"
#include "ffts_plus_proto_transfer.h"

namespace {
constexpr int32_t kSrcSlotNum = 4;
constexpr int32_t kNotifyIdNum = 16;
constexpr int32_t kWriteValueNum = 4;

constexpr int32_t kRequiredUserDataNum = 6;
constexpr int32_t kArgsAddrLIndex = 2;
constexpr int32_t kArgsAddrHIndex = 3;
constexpr size_t kHostPidIndex = 6UL;

constexpr int32_t kManualIndex = 0;
constexpr int32_t kManualAicAivCtxPcNum = 1;
constexpr int32_t kAutoNonTailIndex = 0;
constexpr int32_t kAutoTailIndex = 1;
constexpr int32_t kAutoAicAivCtxPcNum = 2;
constexpr int32_t kAutoNonTailAicCtxIndexVal0 = 0;
constexpr int32_t kAutoTailAicCtxIndex = 1;
constexpr int32_t kAutoNonTailAivCtxIndexVal2 = 2;
constexpr int32_t kAutoTailAivCtxIndex = 3;
constexpr int32_t kAutoMixAicAivCtxPcNum = 4;

constexpr uint32_t k1BitMask = 0x00000001U;    // 1  bit , 0000,0001
constexpr uint32_t k2BitsMask = 0x00000003U;   // 2  bits, 0000,0011
constexpr uint32_t k3BitsMask = 0x00000007U;   // 3  bits, 0000,0111
constexpr uint32_t k4BitsMask = 0x0000000FU;   // 4  bits, 0000,1111
constexpr uint32_t k5BitsMask = 0x0000001FU;   // 5  bits, 0001,1111
constexpr uint32_t k6BitsMask = 0x0000003FU;   // 6  bits, 0011,1111
constexpr uint32_t k7BitsMask = 0x0000007FU;   // 7  bits, 0111,1111
constexpr uint32_t k8BitsMask = 0x000000FFU;   // 8  bits, 1111,1111

constexpr uint32_t k12BitsMask = 0x00000FFFU;  // 12 bits, 0000,1111,1111,1111
constexpr uint32_t k16BitsMask = 0x0000FFFFU;  // 16 bits, 1111,1111,1111,1111

constexpr uint32_t k17BitsMask = 0x0001FFFFU;  // 17 bits, 0000,0000,0000,0001,1111,1111,1111,1111
constexpr uint32_t k32BitsMask = 0xFFFFFFFFU;  // 32 bits, 1111,1111,1111,1111,1111,1111,1111,1111

const std::string kAicpuAllshape = "_AllShape";
constexpr uint32_t kCustomAicpuKernelType = 4U;
const std::string kMixl2PrefixMixAic = "_mix_aic";
const std::string kMixl2PrefixMixAiv = "_mix_aiv";

const std::set<rtFftsPlusContextType_t> kSaveArgsCtxType = {
    RT_CTX_TYPE_AICORE,
    RT_CTX_TYPE_AIV,
    RT_CTX_TYPE_MIX_AIC,
    RT_CTX_TYPE_MIX_AIV,
    RT_CTX_TYPE_AICPU
};
constexpr uint32_t kModeInArgsFirstFieldVal0 = 0U; // mode addr at args field
constexpr uint32_t kArgsSkipFirstField = 1U; // mix ctx args first addr is not input/output addr
constexpr uint32_t kSaveTaskAddr = 1U;
}

namespace gert {
std::map<rtFftsPlusContextType_t, FftsPlusProtoTransfer::CtxHandle> FftsPlusProtoTransfer::init_ctx_fun_ {
    { RT_CTX_TYPE_AICORE, &FftsPlusProtoTransfer::InitAicAivCtx },
    { RT_CTX_TYPE_AIV, &FftsPlusProtoTransfer::InitAicAivCtx },
    { RT_CTX_TYPE_PERSISTENT_CACHE, &FftsPlusProtoTransfer::InitPersistentCacheCtx },
    { RT_CTX_TYPE_NOTIFY_WAIT, &FftsPlusProtoTransfer::InitNotifyCtx },
    { RT_CTX_TYPE_NOTIFY_RECORD, &FftsPlusProtoTransfer::InitNotifyCtx },
    { RT_CTX_TYPE_WRITE_VALUE, &FftsPlusProtoTransfer::InitWriteValueCtx },
    { RT_CTX_TYPE_MIX_AIC, &FftsPlusProtoTransfer::InitMixAicAivCtx },
    { RT_CTX_TYPE_MIX_AIV, &FftsPlusProtoTransfer::InitMixAicAivCtx },
    { RT_CTX_TYPE_SDMA, &FftsPlusProtoTransfer::InitSdmaCtx },
    { RT_CTX_TYPE_FLUSH_DATA, &FftsPlusProtoTransfer::InitDataCtx },
    { RT_CTX_TYPE_INVALIDATE_DATA, &FftsPlusProtoTransfer::InitDataCtx },
    { RT_CTX_TYPE_WRITEBACK_DATA, &FftsPlusProtoTransfer::InitDataCtx },
    { RT_CTX_TYPE_AICPU, &FftsPlusProtoTransfer::InitAicpuCtx },
    { RT_CTX_TYPE_COND_SWITCH, &FftsPlusProtoTransfer::InitCondSwitchCtx },
    { RT_CTX_TYPE_CASE_SWITCH, &FftsPlusProtoTransfer::InitCaseCtx },
    { RT_CTX_TYPE_AT_START, &FftsPlusProtoTransfer::InitAtStartCtx },
    { RT_CTX_TYPE_AT_END, &FftsPlusProtoTransfer::InitAtEndCtx },
    { RT_CTX_TYPE_LABEL, &FftsPlusProtoTransfer::InitLabelCtx }
};

FftsPlusProtoTransfer::~FftsPlusProtoTransfer() {
  for (auto &ext_info_addr : ext_info_addrs_) {
    GE_FREE_RT_LOG(ext_info_addr);
  }
}

void InitMixSqeSubType(rtFftsPlusSqe_t * const sqe, uint8_t sub_type) {
  if (sub_type != std::numeric_limits<uint8_t>::max()) {
    sqe->subType = sub_type;
  }
  return;
}

std::unique_ptr<uint8_t[]> FftsPlusProtoTransfer::Transfer(const ge::OpDescPtr &op_desc,
    const domi::FftsPlusTaskDef &ffts_plus_task_def, size_t &total_size) {
  FE_ASSERT_NOTNULL(find_node_handle_);
  FE_ASSERT_NOTNULL(op_desc);
  logic_stream_id_ = static_cast<uint32_t>(op_desc->GetStreamId());
  const int32_t ctx_num = ffts_plus_task_def.ffts_plus_ctx_size();
  // +1 because descbuf need to be 128byte align
  size_t descBufLen = sizeof(rtFftsPlusComCtx_t) * static_cast<size_t>(ctx_num);
  total_size = sizeof(TransTaskInfo) + sizeof(rtFftsPlusSqe_t) + descBufLen + ADDR_ALIGN_NUM_128;
  GELOGD("Alloc task info buf len: %zu, total size: %zu.", descBufLen, total_size);
  auto holder = ge::MakeUnique<uint8_t[]>(total_size);
  FE_ASSERT_NOTNULL(holder);
  TransTaskInfo *task_info_ptr = reinterpret_cast<TransTaskInfo*>(holder.get());
  task_info_ptr->rt_task_info.descBufLen = descBufLen;
  uint8_t *sqe_base = task_info_ptr->args;
  auto *const ffts_plus_sqe = ge::PtrToPtr<uint8_t, rtFftsPlusSqe_t>(sqe_base);
  InitFftsPlusSqe(ffts_plus_task_def.ffts_plus_sqe(), ffts_plus_sqe);

  size_t buf_offset = sizeof(rtFftsPlusSqe_t);
  task_info_ptr->offsets[static_cast<size_t>(InfoStType::kDescBuf)] = buf_offset;
  auto *const buff_ptr = &task_info_ptr->args[buf_offset];
  GELOGD("Init node[%s] ctx begin, context num=%d", op_desc->GetName().c_str(), ctx_num);

  if (InitFftsPlusCtx(ffts_plus_task_def, buff_ptr, ctx_num) != ge::SUCCESS) {
    return nullptr;
  }
  InitMixSqeSubType(ffts_plus_sqe, sub_type_);
  return holder;
}

void FftsPlusProtoTransfer::InitFftsPlusSqe(const domi::FftsPlusSqeDef &sqe_def, rtFftsPlusSqe_t * const sqe) const {
  InitFftsPlusSqeHeader(sqe_def.sqe_header(), sqe->sqeHeader);

  sqe->wrrRatio = static_cast<uint16_t>(sqe_def.wrr_ratio() & k4BitsMask);
  sqe->sqeIndex = static_cast<uint16_t>(sqe_def.sqe_index());
  sqe->totalContextNum = static_cast<uint16_t>(sqe_def.total_context_num());
  GELOGD("Trans total context num: %u", sqe->totalContextNum);
  sqe->readyContextNum = static_cast<uint16_t>(sqe_def.ready_context_num());
  sqe->preloadContextNum = static_cast<uint16_t>(sqe_def.preload_context_num());

  sqe->prefetchOstNum = static_cast<uint16_t>(sqe_def.prefetch_ost_num() & k5BitsMask);
  sqe->cmaintOstNum = static_cast<uint16_t>(sqe_def.cmaint_ost_num() & k5BitsMask);

  sqe->aicPrefetchLower = static_cast<uint16_t>(sqe_def.aic_prefetch_lower() & k5BitsMask);
  sqe->aicPrefetchUpper = static_cast<uint16_t>(sqe_def.aic_prefetch_upper() & k5BitsMask);
  sqe->aivPrefetchLower = static_cast<uint16_t>(sqe_def.aiv_prefetch_lower() & k5BitsMask);
  sqe->aivPrefetchUpper = static_cast<uint16_t>(sqe_def.aiv_prefetch_upper() & k5BitsMask);
}

void FftsPlusProtoTransfer::InitFftsPlusSqeHeader(const domi::StarsSqeHeaderDef &sqe_header_def,
                                                  rtStarsSqeHeader_t &sqe_header) const {
  sqe_header.l1Lock = static_cast<uint8_t>(sqe_header_def.l1_lock());
  sqe_header.l1Unlock = static_cast<uint8_t>(sqe_header_def.l1_unlock());
  sqe_header.blockDim = static_cast<uint16_t>(sqe_header_def.block_dim());
}

void FftsPlusProtoTransfer::FusionOpPreProc(const ge::OpDescPtr op_desc, const domi::FftsPlusCtxDef &ctx_def,
                                            rtFftsPlusContextType_t ctx_type) {
  if (op_desc == nullptr) {
    return;
  }
  uint32_t prof_ctx_type = static_cast<uint32_t>(ctx_type);
  (void)ge::AttrUtils::SetInt(op_desc, kFFTSProfCtxType, prof_ctx_type);
  GELOGD("Set op [%s] profiling context type %u.", op_desc->GetNamePtr(), prof_ctx_type);
  std::vector<std::string> orig_names;
  if (ge::AttrUtils::GetListStr(op_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, orig_names) &&
      (!orig_names.empty())) {
    ge::FusionOpInfo fusion_op_info;
    fusion_op_info.stream_id = logic_stream_id_;
    fusion_op_info.op_index = ctx_def.op_index();
    fusion_op_info.original_op_names = orig_names;
    fusion_op_info.op_name = op_desc->GetName();
    fusion_op_info_.emplace_back(fusion_op_info);
  }
  if ((save_ctx_args_handle_ != nullptr) && (kSaveArgsCtxType.count(ctx_type) > 0U)) {
    size_t dump_args_offset = io_addrs_.size();
    const auto it = ctx_additional_data_.find(kModeInArgsFirstFieldVal0);
    if ((it != ctx_additional_data_.cend()) && (it->second.count(ctx_def.context_id()) > 0U)) {
      dump_args_offset += kArgsSkipFirstField;
    }
    GELOGD("save ctx args, op idx: %u, ctx type: %u, ctx id: %u", ctx_def.op_index(),
           ctx_def.context_type(), ctx_def.context_id());
    save_ctx_args_handle_(op_desc, dump_args_offset);
  }
  return;
}

ge::Status FftsPlusProtoTransfer::InitFftsPlusCtx(const domi::FftsPlusTaskDef &task_def,
    uint8_t *const ctx, const int32_t num) {
  InitAdditionalData(task_def);
  rtFftsPlusComCtx_t *const com_ctx = ge::PtrToPtr<uint8_t, rtFftsPlusComCtx_t>(ctx);
  FE_ASSERT_NOTNULL(com_ctx);
  for (auto i = 0; i < num; ++i) {
    const domi::FftsPlusCtxDef &ctx_def = task_def.ffts_plus_ctx(i);
    const auto ctx_type = static_cast<rtFftsPlusContextType_t>(ctx_def.context_type());
    if ((num == 1) && (ctx_type == RT_CTX_TYPE_MIX_AIC || ctx_type == RT_CTX_TYPE_MIX_AIV)) {
      sub_type_ = ctx_type;
    }
    const auto op_desc = find_node_handle_(ctx_def.op_index());
    FusionOpPreProc(op_desc, ctx_def, ctx_type);
    GELOGI("Init ctx %d in FftsPlusTask, context_type=%u, op_index=%u", i, ctx_type, ctx_def.op_index());
    const std::map<rtFftsPlusContextType_t, CtxHandle>::const_iterator it = init_ctx_fun_.find(ctx_type);
    if (it != init_ctx_fun_.cend()) {
      auto *const temp_ctx = ge::PtrAdd<rtFftsPlusComCtx_t>(com_ctx, static_cast<size_t>(num), static_cast<size_t>(i));
      temp_ctx->contextType = ctx_type;
      GE_CHK_STATUS_RET(it->second(this, ctx_def, temp_ctx), "Init fftsplusCtx failed, ctx_index:%d", i);
    } else {
      REPORT_INNER_ERR_MSG("E19999", "Unsupported ctx type %d", ctx_type);
      GELOGE(ge::FAILED, "[Check][CtxType] Unsupported ctx type %u", ctx_type);
      return ge::FAILED;
    }
  }
  return ge::SUCCESS;
}

ge::Status FftsPlusProtoTransfer::InitPersistentCacheCtx(const domi::FftsPlusCtxDef &task_def,
    rtFftsPlusComCtx_t *const com_ctx) const {
  rtFftsPlusPersistentCacheCtx_t *const ctx = ge::PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusPersistentCacheCtx_t>(com_ctx);
  FE_ASSERT_NOTNULL(ctx);
  const domi::FftsPlusCachePersistCtxDef &ctx_def = task_def.cache_persist_ctx();
  ctx->contextType = static_cast<rtFftsPlusContextType_t>(task_def.context_type());
  ctx->successorNum = static_cast<uint8_t>(ctx_def.successor_num());
  ctx->aten = static_cast<uint8_t>(ctx_def.aten() & k1BitMask);
  ctx->predCntInit = static_cast<uint8_t>(ctx_def.pred_cnt_init());
  ctx->predCnt = static_cast<uint8_t>(ctx_def.pred_cnt());

  if (ctx_def.successor_list_size() > RT_CTX_SUCCESSOR_NUM) {
    REPORT_INNER_ERR_MSG("E19999",
                       "Size of successor_list in FftsPlusPersistentCacheCtxDef should not > %d, it is %d actually",
                       RT_CTX_SUCCESSOR_NUM, ctx_def.successor_list_size());
    GELOGE(ge::FAILED,
           "[Check][Param] Size of successor_list in FftsPlusPersistentCacheCtxDef should not > %d, it is %d actually",
           RT_CTX_SUCCESSOR_NUM, ctx_def.successor_list_size());
    return ge::FAILED;
  }
  for (auto i = 0; i < ctx_def.successor_list_size(); i++) {
    ctx->successorList[i] = static_cast<uint16_t>(ctx_def.successor_list(i));
  }

  ctx->persistentEnable = static_cast<uint8_t>(ctx_def.persistent_en() & k1BitMask);
  ctx->persistentSize = static_cast<uint16_t>(ctx_def.persistent_size());
  ctx->persistentId = static_cast<uint32_t>(ctx_def.persistent_id());
  return ge::SUCCESS;
}

ge::Status FftsPlusProtoTransfer::InitAicAivCtx(const domi::FftsPlusCtxDef &task_def,
    rtFftsPlusComCtx_t *const com_ctx) const {
  rtFftsPlusAicAivCtx_t *const ctx = ge::PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusAicAivCtx_t>(com_ctx);
  FE_ASSERT_NOTNULL(ctx);
  const domi::FftsPlusAicAivCtxDef &ctx_def = task_def.aic_aiv_ctx();
  ctx->successorNum = static_cast<uint8_t>(ctx_def.successor_num());
  ctx->aten = static_cast<uint8_t>(ctx_def.aten() & k1BitMask);
  ctx->prefetchConfig = static_cast<uint8_t>(ctx_def.prefetch_config());
  ctx->predCntInit = static_cast<uint8_t>(ctx_def.pred_cnt_init());
  ctx->predCnt = static_cast<uint8_t>(ctx_def.pred_cnt());
  if (ctx_def.successor_list_size() > RT_CTX_SUCCESSOR_NUM) {
    REPORT_INNER_ERR_MSG("E19999", "Size of successor_list in FftsPlusAicAivCtxDef should not > %d, but %d exactly",
                       RT_CTX_SUCCESSOR_NUM, ctx_def.successor_list_size());
    GELOGE(ge::FAILED, "[Check][Param] Size of successor_list in FftsPlusAicAivCtxDef should not > %d, but %d exactly",
           RT_CTX_SUCCESSOR_NUM, ctx_def.successor_list_size());
    return ge::FAILED;
  }
  for (auto i = 0; i < ctx_def.successor_list_size(); i++) {
    ctx->successorList[i] = static_cast<uint16_t>(ctx_def.successor_list(i));
  }

  ctx->schem = static_cast<uint16_t>(ctx_def.schem() & k2BitsMask);
  GELOGD("Set schedule mode[%u] to context of rts.", static_cast<uint32_t>(ctx->schem));
  ctx->atm = static_cast<uint16_t>(ctx_def.atm() & k1BitMask);
  ctx->prefetchEnableBitmap = static_cast<uint16_t>(ctx_def.prefetch_enable_bitmap() & k4BitsMask);
  ctx->prefetchOnceBitmap = static_cast<uint16_t>(ctx_def.prefetch_once_bitmap() & k4BitsMask);

  ctx->pmg = static_cast<uint16_t>(ctx_def.pmg() & k2BitsMask);
  ctx->ns = static_cast<uint16_t>(ctx_def.ns() & k1BitMask);
  ctx->partId = static_cast<uint16_t>(ctx_def.part_id() & k8BitsMask);
  ctx->qos = static_cast<uint16_t>(ctx_def.qos() & k4BitsMask);

  ctx->threadId = static_cast<uint16_t>(ctx_def.thread_id());
  ctx->threadDim = static_cast<uint16_t>(ctx_def.thread_dim());
  ctx->nonTailBlockdim = static_cast<uint16_t>(ctx_def.non_tail_block_dim());
  ctx->tailBlockdim = static_cast<uint16_t>(ctx_def.tail_block_dim());

  if (ctx_def.src_slot_size() > kSrcSlotNum) {
    REPORT_INNER_ERR_MSG("E19999", "Size of src_slot in FftsPlusAicAivCtxDef should not > %d, but %d exactly",
                       kSrcSlotNum, ctx_def.src_slot_size());
    GELOGE(ge::FAILED, "[Check][Param] Size of src_slot in FftsPlusAicAivCtxDef should not > %d, but %d exactly",
           kSrcSlotNum, ctx_def.src_slot_size());
    return ge::FAILED;
  }
  for (auto i = 0; i < ctx_def.src_slot_size(); ++i) {
    ctx->srcSlot[i] = static_cast<uint16_t>(ctx_def.src_slot(i));
  }

  const uint64_t task_param_ptr_base = args_base_ + (sizeof(void *) * io_addrs_.size());
  GELOGD("FftsPlusAicAivCtxDef: task param addr is %lu.", task_param_ptr_base);
  ctx->taskParamPtrBaseL = static_cast<uint32_t>(task_param_ptr_base & k32BitsMask);
  ctx->taskParamPtrBaseH = static_cast<uint16_t>((task_param_ptr_base >> 32U) & k16BitsMask);
  ctx->taskParamPtrOffset = static_cast<uint16_t>(ctx_def.task_param_ptr_offset());

  if (ctx->threadDim == 0U) {
    GELOGD("Context thread dim is zero, Dynamic shape mode.");
    return ge::SUCCESS;
  }

  return (ctx->atm == 0U) ? InitManualAicAivCtx(ctx_def, *ctx) : InitAutoAicAivCtx(ctx_def, *ctx);
}

ge::Status FftsPlusProtoTransfer::InitManualAicAivCtx(const domi::FftsPlusAicAivCtxDef &ctx_def,
    rtFftsPlusAicAivCtx_t &ctx) const {
  for (auto i = 0; i < ctx_def.task_addr_size(); ++i) {
    GELOGD("index %d, task addr is 0x%lx", i, ctx_def.task_addr(i));
    io_addrs_.emplace_back(ctx_def.task_addr(i));
  }

  // PcL for low 32 bits of pc, PcH for high 16 bits of pc
  if (ctx_def.kernel_name_size() != kManualAicAivCtxPcNum) {
    REPORT_INNER_ERR_MSG("E19999", "Size of kernel_name in FftsPlusAicAivCtxDef should be %d, but %d exactly",
                       kManualAicAivCtxPcNum, ctx_def.kernel_name_size());
    GELOGE(ge::FAILED, "[Check][Param] Size of kernel_name in FftsPlusAicAivCtxDef should be %d, but %d exactly",
           kManualAicAivCtxPcNum, ctx_def.kernel_name_size());
    return ge::FAILED;
  }
  uint32_t i_cache_prefetch_cnt = 0U;
  void *task_start_pc = nullptr;
  if (addr_pref_handle_ != nullptr) {
    GE_CHK_STATUS_RET(addr_pref_handle_(ctx_def.kernel_name(kManualIndex), task_start_pc, i_cache_prefetch_cnt),
                      "Get addr and pref cnt failed, kernel_name=%s", ctx_def.kernel_name(kManualIndex).c_str());
  }
  ctx.nonTailTaskStartPcL = static_cast<uint32_t>(ge::PtrToValue(task_start_pc) & k32BitsMask);
  ctx.nonTailTaskStartPcH = static_cast<uint16_t>((ge::PtrToValue(task_start_pc) >> 32U) & k16BitsMask);
  ctx.tailTaskStartPcL = static_cast<uint32_t>(ge::PtrToValue(task_start_pc) & k32BitsMask);
  ctx.tailTaskStartPcH = static_cast<uint16_t>((ge::PtrToValue(task_start_pc) >> 32U) & k16BitsMask);
  ctx.icachePrefetchCnt = static_cast<uint16_t>(i_cache_prefetch_cnt & k5BitsMask);

  return ge::SUCCESS;
}

ge::Status FftsPlusProtoTransfer::InitAutoAicAivCtx(const domi::FftsPlusAicAivCtxDef &ctx_def,
    rtFftsPlusAicAivCtx_t &ctx) const {
  if (ctx_def.save_task_addr() == kSaveTaskAddr) {
    for (uint16_t i = 0U; i < (ctx.threadDim - 1U); ++i) {
      GE_RETURN_IF_ERROR(InitThrdIoAddrs(ctx_def, i, ctx_def.task_addr_offset_size()));
    }
    GE_RETURN_IF_ERROR(
        InitThrdIoAddrs(ctx_def, ctx.threadDim - 1U,
                        static_cast<int32_t>(ctx_def.input_output_count())));
    for (auto k = 0; k < (ctx_def.task_addr_size() - ctx_def.task_addr_offset_size()); ++k) {
      auto logic_addr = static_cast<uintptr_t>(ctx_def.task_addr(ctx_def.task_addr_offset_size() + k));
      io_addrs_.emplace_back(logic_addr);
    }
  }

  // PcL for low 32 bits of pc, PcH for high 16 bits of pc
  if (ctx_def.kernel_name_size() != kAutoAicAivCtxPcNum) {
    REPORT_INNER_ERR_MSG("E19999", "Size of kernel_name in FftsPlusAicAivCtxDef should be %d, but %d exactly",
                       kAutoAicAivCtxPcNum, ctx_def.kernel_name_size());
    GELOGE(ge::FAILED, "[Check][Param] Size of kernel_name in FftsPlusAicAivCtxDef should be %d, but %d exactly",
           kAutoAicAivCtxPcNum, ctx_def.kernel_name_size());
    return ge::FAILED;
  }
  uint32_t non_tail_i_cache_prefetch_cnt = 0U;
  void *non_tail_task_start_pc = nullptr;
  if (addr_pref_handle_ != nullptr) {
    GE_CHK_STATUS_RET(addr_pref_handle_(ctx_def.kernel_name(kAutoNonTailIndex), non_tail_task_start_pc,
                                        non_tail_i_cache_prefetch_cnt),
                      "Get addr and pref cnt failed, kernel_name=%s", ctx_def.kernel_name(kAutoNonTailIndex).c_str());
  }
  ctx.nonTailTaskStartPcL = static_cast<uint32_t>(ge::PtrToValue(non_tail_task_start_pc) & k32BitsMask);
  ctx.nonTailTaskStartPcH = static_cast<uint16_t>((ge::PtrToValue(non_tail_task_start_pc) >> 32U) & k16BitsMask);
  uint32_t tail_i_cache_prefetch_cnt = 0U;
  void *tail_task_start_pc = nullptr;
  if (addr_pref_handle_ != nullptr) {
    GE_CHK_STATUS_RET(
        addr_pref_handle_(ctx_def.kernel_name(kAutoTailIndex), tail_task_start_pc, tail_i_cache_prefetch_cnt),
        "Get addr and pref cnt failed, kernel_name=%s", ctx_def.kernel_name(kAutoTailIndex).c_str());
  }

  ctx.tailTaskStartPcL = static_cast<uint32_t>(ge::PtrToValue(tail_task_start_pc) & k32BitsMask);
  ctx.tailTaskStartPcH = static_cast<uint16_t>((ge::PtrToValue(tail_task_start_pc) >> 32U) & k16BitsMask);
  const uint32_t i_cache_prefetch_cnt = std::min(non_tail_i_cache_prefetch_cnt, tail_i_cache_prefetch_cnt);
  ctx.icachePrefetchCnt = static_cast<uint16_t>(i_cache_prefetch_cnt & k5BitsMask);

  return ge::SUCCESS;
}

ge::Status FftsPlusProtoTransfer::InitNotifyCtx(const domi::FftsPlusCtxDef &task_def,
    rtFftsPlusComCtx_t *const com_ctx) const {
  rtFftsPlusNotifyCtx_t *const ctx = ge::PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusNotifyCtx_t>(com_ctx);
  FE_ASSERT_NOTNULL(ctx);
  const domi::FftsPlusNotifyCtxDef &ctx_def = task_def.notify_ctx();
  ctx->successorNum = static_cast<uint8_t>(ctx_def.successor_num());
  ctx->aten = static_cast<uint8_t>(ctx_def.aten() & k1BitMask);
  ctx->predCntInit = static_cast<uint8_t>(ctx_def.pred_cnt_init());
  ctx->predCnt = static_cast<uint8_t>(ctx_def.pred_cnt());

  if (ctx_def.successor_list_size() > RT_CTX_SUCCESSOR_NUM) {
    REPORT_INNER_ERR_MSG("E19999", "Size of successor_list in FftsPlusNotifyCtxDef should not > %d, but %d exactly",
                       RT_CTX_SUCCESSOR_NUM, ctx_def.successor_list_size());
    GELOGE(ge::FAILED, "[Check][Param] Size of successor_list in FftsPlusNotifyCtxDef should not > %d, but %d exactly",
           RT_CTX_SUCCESSOR_NUM, ctx_def.successor_list_size());
    return ge::FAILED;
  }
  for (auto i = 0; i < ctx_def.successor_list_size(); i++) {
    ctx->successorList[i] = static_cast<uint16_t>(ctx_def.successor_list(i));
  }

  ctx->satm = static_cast<uint16_t>(ctx_def.satm() & k1BitMask);
  ctx->atm = static_cast<uint16_t>(ctx_def.atm() & k1BitMask);
  ctx->threadId = static_cast<uint16_t>(ctx_def.thread_id());
  ctx->threadDim = static_cast<uint16_t>(ctx_def.thread_dim());
  ctx->notifyIdBase = static_cast<uint16_t>(ctx_def.notify_id_base());
  ctx->autoWindow = static_cast<uint8_t>(ctx_def.auto_window());

  if (ctx_def.notify_id_size() > kNotifyIdNum) {
    REPORT_INNER_ERR_MSG("E19999", "Size of notify_id in FftsPlusNotifyCtxDef should not > %d, but %d exactly",
                       kNotifyIdNum, ctx_def.notify_id_size());
    GELOGE(ge::FAILED, "[Check][Param] Size of notify_id in FftsPlusNotifyCtxDef should not > %d, but %d exactly",
           kNotifyIdNum, ctx_def.notify_id_size());
    return ge::FAILED;
  }
  for (auto i = 0; i < ctx_def.notify_id_size(); i++) {
    ctx->notifyId[i] = static_cast<uint16_t>(ctx_def.notify_id(i));
  }

  return ge::SUCCESS;
}

ge::Status FftsPlusProtoTransfer::InitWriteValueCtx(const domi::FftsPlusCtxDef &task_def,
    rtFftsPlusComCtx_t *const com_ctx) const {
  rtFftsPlusWriteValueCtx_t *const ctx = ge::PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusWriteValueCtx_t>(com_ctx);
  FE_ASSERT_NOTNULL(ctx);
  const domi::FftsPlusWriteValueCtxDef &ctx_def = task_def.write_value_ctx();
  ctx->successorNum = static_cast<uint8_t>(ctx_def.successor_num());
  ctx->aten = static_cast<uint8_t>(ctx_def.aten() & k1BitMask);
  ctx->predCntInit = static_cast<uint8_t>(ctx_def.pred_cnt_init());
  ctx->predCnt = static_cast<uint8_t>(ctx_def.pred_cnt());

  if (ctx_def.successor_list_size() > RT_CTX_SUCCESSOR_NUM) {
    REPORT_INNER_ERR_MSG("E19999", "Size of successor_list in FftsPlusWriteValueCtxDef should not > %d, but %d exactly",
                       RT_CTX_SUCCESSOR_NUM, ctx_def.successor_list_size());
    GELOGE(ge::FAILED, "[Check][Param] Size of list in FftsPlusWriteValueCtxDef should not > %d, but %d exactly",
           RT_CTX_SUCCESSOR_NUM, ctx_def.successor_list_size());
    return ge::FAILED;
  }
  for (auto i = 0; i < ctx_def.successor_list_size(); i++) {
    ctx->successorList[i] = static_cast<uint16_t>(ctx_def.successor_list(i));
  }

  ctx->atm = static_cast<uint16_t>(ctx_def.atm() & k1BitMask);
  ctx->threadId = static_cast<uint16_t>(ctx_def.thread_id());
  ctx->threadDim = static_cast<uint16_t>(ctx_def.thread_dim());

  ctx->awSize = static_cast<uint8_t>(ctx_def.aw_size() & k3BitsMask);
  ctx->awSnoop = static_cast<uint8_t>(ctx_def.aw_snoop() & k1BitMask);
  ctx->awCache = static_cast<uint8_t>(ctx_def.aw_cache() & k4BitsMask);
  ctx->awProt = static_cast<uint8_t>(ctx_def.aw_prot() & k3BitsMask);
  ctx->awVa = static_cast<uint8_t>(ctx_def.aw_va() & k1BitMask);

  ctx->arSize = static_cast<uint8_t>(ctx_def.ar_size() & k3BitsMask);
  ctx->arSnoop = static_cast<uint8_t>(ctx_def.ar_snoop() & k1BitMask);
  ctx->arCache = static_cast<uint8_t>(ctx_def.ar_cache() & k4BitsMask);
  ctx->arProt = static_cast<uint8_t>(ctx_def.ar_prot() & k3BitsMask);
  ctx->arVa = static_cast<uint8_t>(ctx_def.ar_va() & k1BitMask);

  uint8_t *write_addr_base = nullptr;
  if ((run_addr_handle_ != nullptr) && (run_addr_handle_(ctx_def.write_addr_base(), write_addr_base) != ge::SUCCESS)) {
    GELOGE(ge::INTERNAL_ERROR, "[Check][GetRtAddress]logic write addr base is 0x%lx.", ctx_def.write_addr_base());
    return ge::INTERNAL_ERROR;
  }
  ctx->writeAddressBaseL = static_cast<uint32_t>(ge::PtrToValue(write_addr_base) & k32BitsMask);
  ctx->writeAddressBaseH = static_cast<uint32_t>((ge::PtrToValue(write_addr_base) >> 32U) & k17BitsMask);
  ctx->writeAddressOffset = ctx_def.write_addr_offset();

  if (ctx_def.write_value_size() > kWriteValueNum) {
    REPORT_INNER_ERR_MSG("E19999", "Size of write_value in FftsPlusWriteValueCtxDef should not > %d, but %d exactly",
                       kWriteValueNum, ctx_def.write_value_size());
    GELOGE(ge::FAILED, "[Check][Param] Size of write_value in FftsPlusWriteValueCtxDef should not > %d, but %d exactly",
           kWriteValueNum, ctx_def.write_value_size());
    return ge::FAILED;
  }
  for (auto i = 0; i < ctx_def.write_value_size(); i++) {
    ctx->writeValue[i] = static_cast<uint16_t>(ctx_def.write_value(i));
  }

  return ge::SUCCESS;
}

ge::Status FftsPlusProtoTransfer::InitMixAicAivCtx(const domi::FftsPlusCtxDef &task_def,
    rtFftsPlusComCtx_t *const com_ctx) const {
  rtFftsPlusMixAicAivCtx_t *const ctx = ge::PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusMixAicAivCtx_t>(com_ctx);
  FE_ASSERT_NOTNULL(ctx);
  const domi::FftsPlusMixAicAivCtxDef &ctx_def = task_def.mix_aic_aiv_ctx();
  ctx->successorNum = static_cast<uint8_t>(ctx_def.successor_num());
  ctx->aten = static_cast<uint8_t>(ctx_def.aten() & k1BitMask);
  ctx->prefetchConfig = static_cast<uint8_t>(ctx_def.prefetch_config());
  ctx->predCntInit = static_cast<uint8_t>(ctx_def.pred_cnt_init());
  ctx->predCnt = static_cast<uint8_t>(ctx_def.pred_cnt());
  if (ctx_def.successor_list_size() > RT_CTX_SUCCESSOR_NUM) {
    REPORT_INNER_ERR_MSG("E19999", "Size of successor_list in FftsPlusMixAicAivCtxDef should not > %d, but %d exactly",
                       RT_CTX_SUCCESSOR_NUM, ctx_def.successor_list_size());
    GELOGE(ge::FAILED, "[Check][Param] Size of list in FftsPlusMixAicAivCtxDef should not > %d, but %d exactly",
           RT_CTX_SUCCESSOR_NUM, ctx_def.successor_list_size());
    return ge::FAILED;
  }
  for (auto i = 0; i < ctx_def.successor_list_size(); i++) {
    ctx->successorList[i] = static_cast<uint16_t>(ctx_def.successor_list(i));
  }

  ctx->schem = static_cast<uint16_t>(ctx_def.schem() & k2BitsMask);
  GELOGD("Set schedule mode[%u] for rts context.", static_cast<uint32_t>(ctx->schem));
  ctx->atm = static_cast<uint16_t>(ctx_def.atm() & k1BitMask);
  ctx->prefetchEnableBitmap = static_cast<uint16_t>(ctx_def.prefetch_enable_bitmap() & k4BitsMask);
  ctx->prefetchOnceBitmap = static_cast<uint16_t>(ctx_def.prefetch_once_bitmap() & k4BitsMask);

  ctx->pmg = static_cast<uint16_t>(ctx_def.pmg() & k2BitsMask);
  ctx->ns = static_cast<uint16_t>(ctx_def.ns() & k1BitMask);
  ctx->partId = static_cast<uint16_t>(ctx_def.part_id() & k8BitsMask);
  ctx->qos = static_cast<uint16_t>(ctx_def.qos() & k4BitsMask);

  ctx->threadId = static_cast<uint16_t>(ctx_def.thread_id());
  ctx->threadDim = static_cast<uint16_t>(ctx_def.thread_dim());

  ctx->nonTailBlockRatioN = static_cast<uint8_t>(ctx_def.non_tail_block_ratio_n());
  ctx->tailBlockRatioN = static_cast<uint8_t>(ctx_def.tail_block_ratio_n());

  ctx->nonTailBlockdim = static_cast<uint16_t>(ctx_def.non_tail_block_dim());
  ctx->tailBlockdim = static_cast<uint16_t>(ctx_def.tail_block_dim());

  const uint64_t task_param_ptr_base = args_base_ + (sizeof(void *) * io_addrs_.size());
  GELOGD("FftsPlusMixAicAivCtxDef: task param addr is %lu.", task_param_ptr_base);
  ctx->aicTaskParamPtrL = static_cast<uint32_t>(task_param_ptr_base & k32BitsMask);
  ctx->aicTaskParamPtrH = static_cast<uint16_t>((task_param_ptr_base >> 32U) & k16BitsMask);
  ctx->aivTaskParamPtrL = static_cast<uint32_t>(task_param_ptr_base & k32BitsMask);
  ctx->aivTaskParamPtrH = static_cast<uint16_t>((task_param_ptr_base >> 32U) & k16BitsMask);
  ctx->aicTaskParamPtrOffset = static_cast<uint16_t>(ctx_def.aic_task_param_ptr_offset());
  ctx->aivTaskParamPtrOffset = static_cast<uint16_t>(ctx_def.aiv_task_param_ptr_offset());
  int32_t start_addr = 0;
  const auto &op_desc = find_node_handle_(task_def.op_index());
  FE_ASSERT_NOTNULL(op_desc);
  const auto it = ctx_additional_data_.find(kModeInArgsFirstFieldVal0);
  if ((it != ctx_additional_data_.cend()) && (it->second.count(task_def.context_id()) > 0U)) {
    start_addr = 1;
    uint32_t need_mode = 1U;
    (void)ge::AttrUtils::SetInt(op_desc, kNeedModeAddr, need_mode);
    GELOGD("Set need mode addr to mix_aic/mix_aiv args.");
  }

  if (ctx_def.src_slot_size() > kSrcSlotNum) {
    REPORT_INNER_ERR_MSG("E19999", "Size of src_slot in FftsPlusMixAicAivCtxDef should not > %d, but %d exactly",
                       kSrcSlotNum, ctx_def.src_slot_size());
    GELOGE(ge::FAILED, "[Check][Param] Size of src_slot in FftsPlusMixAicAivCtxDef should not > %d, but %d exactly",
           kSrcSlotNum, ctx_def.src_slot_size());
    return ge::FAILED;
  }
  for (auto i = 0; i < ctx_def.src_slot_size(); i++) {
    ctx->srcSlot[i] = static_cast<uint16_t>(ctx_def.src_slot(i));
  }

  if (ctx->threadDim == 0U) {
    GELOGD("Context thread dim is zero, Dynamic shape mode.");
    return ge::SUCCESS;
  }

  if (ctx->atm == 0U) {
    std::vector<std::string> name_prefixes;
    (void)ge::AttrUtils::GetListStr(op_desc, ge::ATTR_NAME_KERNEL_NAMES_PREFIX, name_prefixes);
    GE_CHK_STATUS_RET(InitManualMixAicAivCtx(ctx_def, name_prefixes, *ctx, start_addr),
                      "Init MixAicAivCtx in manual mode failed, node:[%s].", op_desc->GetName().c_str());
  } else {
    GE_CHK_STATUS_RET(InitAutoMixAicAivCtx(ctx_def, *ctx, start_addr), "Init MixAicAivCtx in auto mode failed");
  }

  return ge::SUCCESS;
}

ge::Status FftsPlusProtoTransfer::InitManualMixAicAivCtx(const domi::FftsPlusMixAicAivCtxDef &ctx_def,
    const std::vector<std::string> &kernel_name_prefixes,
    rtFftsPlusMixAicAivCtx_t &ctx, const int32_t start_idx) const {
  for (int32_t i = start_idx; i < ctx_def.task_addr_size(); ++i) {
    GELOGD("index %u, task addr is 0x%lx", i, ctx_def.task_addr(i));
    io_addrs_.emplace_back(ctx_def.task_addr(i));
  }

  const size_t prefix_size = kernel_name_prefixes.size();
  // PcL for low 32 bits of pc, PcH for high 16 bits of pc
  if (static_cast<size_t>(ctx_def.kernel_name_size()) != prefix_size) {
    REPORT_INNER_ERR_MSG("E19999", "Size of kernel_name in FftsPlusMixAicAivCtxDef should be %zu, but %d exactly",
                       prefix_size, ctx_def.kernel_name_size());
    GELOGE(ge::FAILED, "[Check][Param] Size of kernel_name in FftsPlusMixAicAivCtxDef should be %zu, but %d exactly",
           prefix_size, ctx_def.kernel_name_size());
    return ge::FAILED;
  }
  uint32_t aic_i_cache_prefetch_cnt = 0U;
  void *aic_task_start_pc = nullptr;
  uint32_t aiv_i_cache_prefetch_cnt = 0U;
  void *aiv_task_start_pc = nullptr;

  if (addr_pref_handle_ != nullptr) {
    for (size_t i = 0UL; i < prefix_size; ++i) {
      if (kernel_name_prefixes[i] == kMixl2PrefixMixAic) {
        GE_CHK_STATUS_RET(addr_pref_handle_(ctx_def.kernel_name(i), aic_task_start_pc, aic_i_cache_prefetch_cnt),
                          "Get addr and pref cnt failed, kernel_name=%s", ctx_def.kernel_name(i).c_str());
      } else if (kernel_name_prefixes[i] == kMixl2PrefixMixAiv) {
        GE_CHK_STATUS_RET(addr_pref_handle_(ctx_def.kernel_name(i), aiv_task_start_pc, aiv_i_cache_prefetch_cnt),
                          "Get addr and pref cnt failed, kernel_name=%s", ctx_def.kernel_name(i).c_str());
      } else {
        // Only for static check
      }
    }
  }

  ctx.nonTailAicTaskStartPcL = static_cast<uint32_t>(ge::PtrToValue(aic_task_start_pc) & k32BitsMask);
  ctx.nonTailAicTaskStartPcH = static_cast<uint16_t>((ge::PtrToValue(aic_task_start_pc) >> 32U) & k16BitsMask);
  ctx.tailAicTaskStartPcL = static_cast<uint32_t>(ge::PtrToValue(aic_task_start_pc) & k32BitsMask);
  ctx.tailAicTaskStartPcH = static_cast<uint16_t>((ge::PtrToValue(aic_task_start_pc) >> 32U) & k16BitsMask);
  ctx.aicIcachePrefetchCnt = static_cast<uint16_t>(aic_i_cache_prefetch_cnt & k5BitsMask);

  ctx.nonTailAivTaskStartPcL = static_cast<uint32_t>(ge::PtrToValue(aiv_task_start_pc) & k32BitsMask);
  ctx.nonTailAivTaskStartPcH = static_cast<uint16_t>((ge::PtrToValue(aiv_task_start_pc) >> 32U) & k16BitsMask);
  ctx.tailAivTaskStartPcL = static_cast<uint32_t>(ge::PtrToValue(aiv_task_start_pc) & k32BitsMask);
  ctx.tailAivTaskStartPcH = static_cast<uint16_t>((ge::PtrToValue(aiv_task_start_pc) >> 32U) & k16BitsMask);
  ctx.aivIcachePrefetchCnt = static_cast<uint16_t>(aiv_i_cache_prefetch_cnt & k5BitsMask);

  return ge::SUCCESS;
}

ge::Status FftsPlusProtoTransfer::InitAutoMixAicAivCtx(const domi::FftsPlusMixAicAivCtxDef &ctx_def,
    rtFftsPlusMixAicAivCtx_t &ctx, const int32_t start_idx) const {
  if (ctx_def.save_task_addr() == kSaveTaskAddr) {
    for (uint16_t i = 0U; i < (ctx.threadDim - 1U); ++i) {
      GE_RETURN_IF_ERROR(InitThrdIoAddrs(ctx_def, i, ctx_def.task_addr_offset_size(), start_idx));
    }
    GE_RETURN_IF_ERROR(InitThrdIoAddrs(
        ctx_def, ctx.threadDim - 1U,
        static_cast<int32_t>(ctx_def.input_output_count()), start_idx));
    const int32_t last_thread_workspace_size = ctx_def.task_addr_size() - ctx_def.task_addr_offset_size() - start_idx;
    for (auto k = 0; k < last_thread_workspace_size; ++k) {
      uintptr_t logic_addr = ctx_def.task_addr(ctx_def.task_addr_offset_size() + k);
      io_addrs_.emplace_back(logic_addr);
    }
  }

  // PcL for low 32 bits of pc, PcH for high 16 bits of pc
  if (ctx_def.kernel_name_size() != kAutoMixAicAivCtxPcNum) {
    REPORT_INNER_ERR_MSG("E19999", "Size of kernel_name in FftsPlusMixAicAivCtxDef should be %d, but %d exactly",
                       kAutoMixAicAivCtxPcNum, ctx_def.kernel_name_size());
    GELOGE(ge::FAILED, "[Check][Param] Size of kernel_name in FftsPlusMixAicAivCtxDef should be %d, but %d exactly",
           kAutoMixAicAivCtxPcNum, ctx_def.kernel_name_size());
    return ge::FAILED;
  }
  uint32_t non_tail_aic_i_cache_prefetch_cnt = 0U;
  void *non_tail_aic_task_start_pc = nullptr;
  if (addr_pref_handle_ != nullptr) {
    GE_CHK_STATUS_RET(addr_pref_handle_(ctx_def.kernel_name(kAutoNonTailAicCtxIndexVal0), non_tail_aic_task_start_pc,
                                        non_tail_aic_i_cache_prefetch_cnt),
                      "Get addr and pref cnt failed, kernel_name=%s",
                      ctx_def.kernel_name(kAutoNonTailAicCtxIndexVal0).c_str());
  }

  ctx.nonTailAicTaskStartPcL = static_cast<uint32_t>(ge::PtrToValue(non_tail_aic_task_start_pc) & k32BitsMask);
  ctx.nonTailAicTaskStartPcH = static_cast<uint16_t>((ge::PtrToValue(non_tail_aic_task_start_pc) >> 32U) & k16BitsMask);
  uint32_t tail_aic_i_cache_prefetch_cnt = 0U;
  void *tail_aic_task_start_pc = nullptr;
  if (addr_pref_handle_ != nullptr) {
    GE_CHK_STATUS_RET(addr_pref_handle_(ctx_def.kernel_name(kAutoTailAicCtxIndex), tail_aic_task_start_pc,
                                        tail_aic_i_cache_prefetch_cnt),
                      "Get addr and pref cnt failed, kernel_name=%s",
                      ctx_def.kernel_name(kAutoTailAicCtxIndex).c_str());
  }

  ctx.tailAicTaskStartPcL = static_cast<uint32_t>(ge::PtrToValue(tail_aic_task_start_pc) & k32BitsMask);
  ctx.tailAicTaskStartPcH = static_cast<uint16_t>((ge::PtrToValue(tail_aic_task_start_pc) >> 32U) & k16BitsMask);
  const uint32_t aic_i_cache_prefetch_cnt = std::min(non_tail_aic_i_cache_prefetch_cnt, tail_aic_i_cache_prefetch_cnt);
  ctx.aicIcachePrefetchCnt = static_cast<uint16_t>(aic_i_cache_prefetch_cnt & k5BitsMask);

  uint32_t non_tail_aiv_i_cache_prefetch_cnt = 0U;
  void *non_tail_aiv_task_start_pc = nullptr;
  if (addr_pref_handle_ != nullptr) {
    GE_CHK_STATUS_RET(addr_pref_handle_(ctx_def.kernel_name(kAutoNonTailAivCtxIndexVal2), non_tail_aiv_task_start_pc,
                                        non_tail_aiv_i_cache_prefetch_cnt),
                      "Get addr and pref cnt failed, kernel_name=%s",
                      ctx_def.kernel_name(kAutoNonTailAivCtxIndexVal2).c_str());
  }

  ctx.nonTailAivTaskStartPcL = static_cast<uint32_t>(ge::PtrToValue(non_tail_aiv_task_start_pc) & k32BitsMask);
  ctx.nonTailAivTaskStartPcH = static_cast<uint16_t>((ge::PtrToValue(non_tail_aiv_task_start_pc) >> 32U) & k16BitsMask);
  uint32_t tail_aiv_i_cache_prefetch_cnt = 0U;
  void *tail_aiv_task_start_pc = nullptr;
  if (addr_pref_handle_ != nullptr) {
    GE_CHK_STATUS_RET(addr_pref_handle_(ctx_def.kernel_name(kAutoTailAivCtxIndex), tail_aiv_task_start_pc,
                                        tail_aiv_i_cache_prefetch_cnt),
                      "Get addr and pref cnt failed, kernel_name=%s",
                      ctx_def.kernel_name(kAutoTailAivCtxIndex).c_str());
  }

  ctx.tailAivTaskStartPcL = static_cast<uint32_t>(ge::PtrToValue(tail_aiv_task_start_pc) & k32BitsMask);
  ctx.tailAivTaskStartPcH = static_cast<uint16_t>((ge::PtrToValue(tail_aiv_task_start_pc) >> 32U) & k16BitsMask);
  const uint32_t aiv_i_cache_prefetch_cnt = std::min(non_tail_aiv_i_cache_prefetch_cnt, tail_aiv_i_cache_prefetch_cnt);
  ctx.aivIcachePrefetchCnt = static_cast<uint16_t>(aiv_i_cache_prefetch_cnt & k5BitsMask);

  return ge::SUCCESS;
}

ge::Status FftsPlusProtoTransfer::InitSdmaCtx(const domi::FftsPlusCtxDef &task_def,
    rtFftsPlusComCtx_t *const com_ctx) const {
  rtFftsPlusSdmaCtx_t *const ctx = ge::PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusSdmaCtx_t>(com_ctx);
  FE_ASSERT_NOTNULL(ctx);
  const domi::FftsPlusSdmaCtxDef &ctx_def = task_def.sdma_ctx();
  ctx->successorNum = static_cast<uint8_t>(ctx_def.successor_num());
  ctx->aten = static_cast<uint8_t>(ctx_def.aten() & k1BitMask);
  ctx->predCntInit = static_cast<uint8_t>(ctx_def.pred_cnt_init());
  ctx->predCnt = static_cast<uint8_t>(ctx_def.pred_cnt());
  if (ctx_def.successor_list_size() > RT_CTX_SUCCESSOR_NUM) {
    REPORT_INNER_ERR_MSG("E19999", "Size of successor_list in FftsPlusSdmaCtxDef should not > %d, but %d exactly",
                       RT_CTX_SUCCESSOR_NUM, ctx_def.successor_list_size());
    GELOGE(ge::FAILED, "[Check][Param] Size of successor_list in FftsPlusSdmaCtxDef should not > %d, but %d exactly",
           RT_CTX_SUCCESSOR_NUM, ctx_def.successor_list_size());
    return ge::FAILED;
  }
  for (auto i = 0; i < ctx_def.successor_list_size(); i++) {
    ctx->successorList[i] = static_cast<uint16_t>(ctx_def.successor_list(i));
  }

  ctx->atm = static_cast<uint8_t>(ctx_def.atm() & k1BitMask);
  ctx->pmg = static_cast<uint16_t>(ctx_def.pmg() & k2BitsMask);
  ctx->ns = static_cast<uint16_t>(ctx_def.ns() & k1BitMask);
  ctx->partId = static_cast<uint16_t>(ctx_def.part_id() & k8BitsMask);
  ctx->qos = static_cast<uint16_t>(ctx_def.qos() & k4BitsMask);

  ctx->threadId = static_cast<uint16_t>(ctx_def.thread_id());
  ctx->threadDim = static_cast<uint16_t>(ctx_def.thread_dim());

  ctx->sdmaSqeHeader = ctx_def.sdma_sqe_header();

  ctx->sourceStreamId = static_cast<uint16_t>(ctx_def.src_stream_id());
  ctx->sourceSubstreamId = static_cast<uint16_t>(ctx_def.src_sub_stream_id());

  ctx->destinationStreamId = static_cast<uint16_t>(ctx_def.dst_stream_id());
  ctx->destinationSubstreamId = static_cast<uint16_t>(ctx_def.dst_sub_stream_id());

  uint8_t *src_addr_base = nullptr;
  if ((run_addr_handle_ != nullptr) && (run_addr_handle_(ctx_def.src_addr_base(), src_addr_base) != ge::SUCCESS)) {
    GELOGE(ge::INTERNAL_ERROR, "[Check][GetRtAddress] failed, logic src addr is 0x%lx.", ctx_def.src_addr_base());
    return ge::INTERNAL_ERROR;
  }
  ctx->sourceAddressBaseL = static_cast<uint32_t>(ge::PtrToValue(src_addr_base) & k32BitsMask);
  ctx->sourceAddressBaseH = static_cast<uint32_t>(ge::PtrToValue(src_addr_base) >> 32U);
  ctx->sourceAddressOffset = ctx_def.src_addr_offset();

  uint8_t *dst_addr_base = nullptr;
  if ((run_addr_handle_ != nullptr) && (run_addr_handle_(ctx_def.dst_addr_base(), dst_addr_base) != ge::SUCCESS)) {
    GELOGE(ge::INTERNAL_ERROR, "[Check][GetRtAddress] failed, logic dst addr is 0x%lx.", ctx_def.dst_addr_base());
    return ge::INTERNAL_ERROR;
  }
  ctx->destinationAddressBaseL = static_cast<uint32_t>(ge::PtrToValue(dst_addr_base) & k32BitsMask);
  ctx->destinationAddressBaseH = static_cast<uint32_t>(ge::PtrToValue(dst_addr_base) >> 32U);
  ctx->destinationAddressOffset = ctx_def.dst_addr_offset();

  ctx->nonTailDataLength = ctx_def.non_tail_data_len();
  ctx->tailDataLength = ctx_def.tail_data_len();
  return ge::SUCCESS;
}

ge::Status FftsPlusProtoTransfer::InitDataCtx(const domi::FftsPlusCtxDef &task_def,
    rtFftsPlusComCtx_t *const com_ctx) const {
  rtFftsPlusDataCtx_t *const ctx = ge::PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusDataCtx_t>(com_ctx);
  FE_ASSERT_NOTNULL(ctx);
  const domi::FftsPlusDataCtxDef &ctx_def = task_def.data_ctx();
  ctx->successorNum = static_cast<uint8_t>(ctx_def.successor_num());
  ctx->aten = static_cast<uint8_t>(ctx_def.aten() & k1BitMask);
  ctx->cntInit = static_cast<uint8_t>(ctx_def.cnt_init());
  ctx->cnt = static_cast<uint8_t>(ctx_def.cnt());
  if (ctx_def.successor_list_size() > RT_CTX_SUCCESSOR_NUM) {
    REPORT_INNER_ERR_MSG("E19999", "Size of successor_list in FftsPlusDataCtxDef should not > %d, but %d exactly",
                       RT_CTX_SUCCESSOR_NUM, ctx_def.successor_list_size());
    GELOGE(ge::FAILED, "[Check][Param] Size of successor_list in FftsPlusDataCtxDef should not > %d, but %d exactly",
           RT_CTX_SUCCESSOR_NUM, ctx_def.successor_list_size());
    return ge::FAILED;
  }
  for (auto i = 0; i < ctx_def.successor_list_size(); i++) {
    ctx->successorList[i] = static_cast<uint16_t>(ctx_def.successor_list(i));
  }

  ctx->atm = static_cast<uint8_t>(ctx_def.atm() & k1BitMask);
  ctx->pmg = static_cast<uint16_t>(ctx_def.pmg() & k2BitsMask);
  ctx->ns = static_cast<uint16_t>(ctx_def.ns() & k1BitMask);
  ctx->partId = static_cast<uint16_t>(ctx_def.part_id() & k8BitsMask);
  ctx->qos = static_cast<uint16_t>(ctx_def.qos() & k4BitsMask);

  ctx->threadId = static_cast<uint16_t>(ctx_def.thread_id());
  ctx->threadDim = static_cast<uint16_t>(ctx_def.thread_dim());
  ctx->origConsumerCounter = static_cast<uint16_t>(ctx_def.orig_consumer_counter());
  ctx->runConsumerCounter = static_cast<uint16_t>(ctx_def.run_consumer_counter());

  uint8_t *addr_base = nullptr;
  if ((run_addr_handle_ != nullptr) && (run_addr_handle_(ctx_def.addr_base(), addr_base) != ge::SUCCESS)) {
    GELOGE(ge::INTERNAL_ERROR, "[Check][GetRtAddress] failed, logic addr base is 0x%lx.", ctx_def.addr_base());
    return ge::INTERNAL_ERROR;
  }
  ctx->addressBaseL = static_cast<uint32_t>(ge::PtrToValue(addr_base) & k32BitsMask);
  ctx->addressBaseH = static_cast<uint32_t>(ge::PtrToValue(addr_base) >> 32U);
  ctx->addressOffset = ctx_def.addr_offset();

  ctx->nonTailNumOutter = static_cast<uint16_t>(ctx_def.non_tail_num_outter());
  ctx->nonTailNumInner = static_cast<uint16_t>(ctx_def.non_tail_num_inner());
  ctx->nonTailLengthInner = ctx_def.non_tail_len_inner();
  ctx->nonTailStrideOutter = ctx_def.non_tail_stride_outter();
  ctx->nonTailStrideInner = ctx_def.non_tail_stride_inner();

  ctx->tailNumOutter = static_cast<uint16_t>(ctx_def.tail_num_outter());
  ctx->tailNumInner = static_cast<uint16_t>(ctx_def.tail_num_inner());
  ctx->tailLengthInner = ctx_def.tail_len_inner();
  ctx->tailStrideOutter = ctx_def.tail_stride_outter();
  ctx->tailStrideInner = ctx_def.tail_stride_inner();

  return ge::SUCCESS;
}

ge::Status FftsPlusProtoTransfer::InitAicpuCtx(const domi::FftsPlusCtxDef &task_def,
    rtFftsPlusComCtx_t *const com_ctx) const {
  rtFftsPlusAiCpuCtx_t *const ctx = ge::PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusAiCpuCtx_t>(com_ctx);
  FE_ASSERT_NOTNULL(ctx);
  const domi::FftsPlusAicpuCtxDef &ctx_def = task_def.aicpu_ctx();
  ctx->successorNum = static_cast<uint8_t>(ctx_def.successor_num());
  ctx->aten = static_cast<uint8_t>(ctx_def.aten() & k1BitMask);
  ctx->predCntInit = static_cast<uint8_t>(ctx_def.pred_cnt_init());
  ctx->predCnt = static_cast<uint8_t>(ctx_def.pred_cnt());

  if (ctx_def.successor_list_size() > RT_CTX_SUCCESSOR_NUM) {
    REPORT_INNER_ERR_MSG("E19999", "Size of successor_list in FftsPlusAicpuCtxDef should not > %d, but %d exactly",
                       RT_CTX_SUCCESSOR_NUM, ctx_def.successor_list_size());
    GELOGE(ge::FAILED, "[Check][Param] Size of successor_list in FftsPlusAicpuCtxDef should not > %d, but %d exactly",
           RT_CTX_SUCCESSOR_NUM, ctx_def.successor_list_size());
    return ge::FAILED;
  }
  for (auto i = 0; i < ctx_def.successor_list_size(); i++) {
    ctx->successorContextID[i] = static_cast<uint16_t>(ctx_def.successor_list(i));
  }

  ctx->atm = static_cast<uint16_t>(ctx_def.atm() & k1BitMask);
  ctx->sqeIndex = static_cast<uint16_t>(ctx_def.sqe_index());
  ctx->kernelType = static_cast<uint8_t>(ctx_def.kernel_type() & k7BitsMask);
  ctx->bm = static_cast<uint8_t>(ctx_def.bm() & k1BitMask);
  ctx->topicType = static_cast<uint8_t>(ctx_def.topic_type() & k4BitsMask);
  ctx->qos = static_cast<uint8_t>(ctx_def.qos() & k3BitsMask);

  ctx->threadId = static_cast<uint16_t>(ctx_def.thread_id());
  ctx->threadDim = static_cast<uint16_t>(ctx_def.thread_dim());
  ctx->nonTailBlockdim = static_cast<uint16_t>(ctx_def.non_tail_block_dim());
  ctx->tailBlockdim = static_cast<uint16_t>(ctx_def.tail_block_dim());

  ctx->subtopicId = static_cast<uint32_t>(ctx_def.sub_topic_id() & k12BitsMask);
  ctx->topicId = static_cast<uint32_t>(ctx_def.topic_id() & k6BitsMask);
  ctx->groupId = static_cast<uint32_t>(ctx_def.group_id() & k6BitsMask);
  ctx->taskParamOffset = ctx_def.task_param_offset();
  const auto op_desc = find_node_handle_(task_def.op_index());
  GE_CHK_STATUS_RET(InitAicpuCtxUserData(op_desc, ctx_def, *ctx), "Init user data for aicpu ctx failed");

  return ge::SUCCESS;
}

ge::Status FftsPlusProtoTransfer::InitAicpuCtxUserData(const ge::OpDescPtr &op_desc,
    const domi::FftsPlusAicpuCtxDef &ctx_def, rtFftsPlusAiCpuCtx_t &ctx) const {
  FE_ASSERT_NOTNULL(op_desc);
  const auto user_data_len = sizeof(ctx.usrData) / sizeof(uint32_t);
  if (user_data_len < static_cast<uint64_t>(kRequiredUserDataNum)) {
    REPORT_INNER_ERR_MSG("E19999", "Length of user_data in rtFftsPlusAiCpuCtx_t should not < %d, but %" PRIu64 " exactly",
        kRequiredUserDataNum, static_cast<uint64_t>(user_data_len));
    GELOGE(ge::FAILED, "[Check][Param] Length of user_data in rtFftsPlusAiCpuCtx_t should not < %d, but %lu exactly",
           kRequiredUserDataNum, user_data_len);
    return ge::FAILED;
  }
  ctx.usrDataLength = static_cast<uint32_t>(kRequiredUserDataNum);

  // get args addr
  void *args_addr = nullptr;
  GE_CHK_STATUS_RET(InitAicpuInfo(op_desc, ctx_def, args_addr));
  ctx.usrData[kArgsAddrLIndex] = static_cast<uint32_t>(ge::PtrToValue(args_addr) & k32BitsMask);
  ctx.usrData[kArgsAddrHIndex] = static_cast<uint32_t>(ge::PtrToValue(args_addr) >> 32U);
  // 适配aicpu vf场景，把userdata6刷新成hostpid
  ctx.usrData[kHostPidIndex] = static_cast<uint32_t>(mmGetPid());

  GELOGD("Init aicpu ctx user data success, node is [%s], type is [%s].",
         op_desc->GetName().c_str(), op_desc->GetType().c_str());
  return ge::SUCCESS;
}

ge::Status FftsPlusProtoTransfer::InitAicpuInfo(const ge::OpDescPtr &op_desc, const domi::FftsPlusAicpuCtxDef &ctx_def,
    void *&addr) const {
  if (ctx_def.kernel_type() == kCustomAicpuKernelType) {
    GELOGE(ge::FAILED, "Not support custom aicpu op.");
    // load custom aicpu so need move to other position
    return ge::FAILED;
  }
  return InitAicpuExtInfo(op_desc, ctx_def, addr);
}

ge::Status FftsPlusProtoTransfer::InitAicpuExtInfo(const ge::OpDescPtr &op_desc,
    const domi::FftsPlusAicpuCtxDef &ctx_def, const void* const& addr) const {
  (void)addr;
  GELOGI("Begin to initialize aicpu op extra info, thread dim is %u.", ctx_def.thread_dim());
  if (ctx_def.thread_dim() == 0U) {   // Zero mark for dynamic.
    return save_aicpu_ctx_handle_(op_desc, ctx_def.kernel());
  }
  GELOGI("Init aicpu op context info success");
  return ge::SUCCESS;
}


ge::Status FftsPlusProtoTransfer::InitCondSwitchCtx(const domi::FftsPlusCtxDef &task_def,
    rtFftsPlusComCtx_t *const com_ctx) const {
  rtFftsPlusCondSwitchCtx_t *const ctx = ge::PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusCondSwitchCtx_t>(com_ctx);
  FE_ASSERT_NOTNULL(ctx);
  const domi::FftsPlusCondSwitchCtxDef &ctx_def = task_def.cond_switch_ctx();
  ctx->trueSuccessorNum = static_cast<uint8_t>(ctx_def.true_successor_num());
  ctx->falseSuccessorNum = static_cast<uint8_t>(ctx_def.false_successor_num() & k7BitsMask);
  ctx->aten = static_cast<uint8_t>(ctx_def.aten() & k1BitMask);

  if (ctx_def.condition() == RT_COND_TYPE_MAX) {
    REPORT_INNER_ERR_MSG("E19999", "Unsupported cond type %u", ctx_def.condition());
    GELOGE(ge::FAILED, "[Check][CtxType] Unsupported cond type %u", ctx_def.condition());
    return ge::FAILED;
  }
  ctx->condition = static_cast<rtFftsPlusCondType_t>(ctx_def.condition());
  ctx->predCntInit = static_cast<uint8_t>(ctx_def.pred_cnt_init());
  ctx->predCnt = static_cast<uint8_t>(ctx_def.pred_cnt());

  if (ctx_def.true_successor_list_size() > RT_CTX_TRUE_SUCCESSOR_NUM) {
    REPORT_INNER_ERR_MSG("E19999",
                       "Size of true_successor_list in FftsPlusCondSwitchCtxDef should not > %d, but %d exactly",
                       RT_CTX_TRUE_SUCCESSOR_NUM, ctx_def.true_successor_list_size());
    GELOGE(ge::FAILED,
           "[Check][Param] Size of true_successor_list in FftsPlusCondSwitchCtxDef should not > %d, but %d exactly",
           RT_CTX_TRUE_SUCCESSOR_NUM, ctx_def.true_successor_list_size());
    return ge::FAILED;
  }
  for (auto i = 0; i < ctx_def.true_successor_list_size(); i++) {
    ctx->trueSuccessorList[i] = static_cast<uint16_t>(ctx_def.true_successor_list(i));
  }

  if (ctx_def.false_successor_list_size() > RT_CTX_FALSE_SUCCESSOR_NUM) {
    REPORT_INNER_ERR_MSG("E19999",
                       "Size of false_successor_list in FftsPlusCondSwitchCtxDef should not > %d, but %d exactly",
                       RT_CTX_FALSE_SUCCESSOR_NUM, ctx_def.false_successor_list_size());
    GELOGE(ge::FAILED,
           "[Check][Param] Size of false_successor_list in FftsPlusCondSwitchCtxDef should not > %d, but %d exactly",
           RT_CTX_FALSE_SUCCESSOR_NUM, ctx_def.false_successor_list_size());
    return ge::FAILED;
  }
  for (auto i = 0; i < ctx_def.false_successor_list_size(); i++) {
    ctx->falseSuccessorList[i] = static_cast<uint16_t>(ctx_def.false_successor_list(i));
  }

  ctx->atm = static_cast<uint16_t>(ctx_def.atm() & k1BitMask);
  ctx->threadId = static_cast<uint16_t>(ctx_def.thread_id());
  ctx->threadDim = static_cast<uint16_t>(ctx_def.thread_dim());

  ctx->arSize = static_cast<uint8_t>(ctx_def.ar_size() & k3BitsMask);
  ctx->snoop = static_cast<uint8_t>(ctx_def.snoop() & k1BitMask);
  ctx->arCache = static_cast<uint8_t>(ctx_def.ar_cache() & k4BitsMask);
  ctx->arProt = static_cast<uint8_t>(ctx_def.ar_prot() & k3BitsMask);
  ctx->va = static_cast<uint8_t>(ctx_def.va() & k1BitMask);

  uint8_t *addr_base_0 = nullptr;
  if ((run_addr_handle_ != nullptr) && (run_addr_handle_(ctx_def.load_addr0_base(), addr_base_0) != ge::SUCCESS)) {
    GELOGE(ge::INTERNAL_ERROR, "[Check][GetRtAddress] logic load addr0 base is 0x%lx.", ctx_def.load_addr0_base());
    return ge::INTERNAL_ERROR;
  }
  ctx->loadAddress0BaseL = static_cast<uint32_t>(ge::PtrToValue(addr_base_0) & k32BitsMask);
  ctx->loadAddress0BaseH = static_cast<uint32_t>((ge::PtrToValue(addr_base_0) >> 32U) & k17BitsMask);
  ctx->ld0En = static_cast<uint32_t>(ctx_def.ld0_en() & k1BitMask);
  ctx->loadAddress0Offset = ctx_def.load_addr0_offset();

  uint8_t *addr_base_1 = nullptr;
  if ((run_addr_handle_ != nullptr) && (run_addr_handle_(ctx_def.load_addr1_base(), addr_base_1) != ge::SUCCESS)) {
    GELOGE(ge::INTERNAL_ERROR, "[Check][GetRtAddress]logic load addr1 base is 0x%lx.", ctx_def.load_addr1_base());
    return ge::INTERNAL_ERROR;
  }
  ctx->loadAddress1BaseL = static_cast<uint32_t>(ge::PtrToValue(addr_base_1) & k32BitsMask);
  ctx->loadAddress1BaseH = static_cast<uint32_t>((ge::PtrToValue(addr_base_1) >> 32U) & k17BitsMask);
  ctx->ld1En = static_cast<uint32_t>(ctx_def.ld1_en() & k1BitMask);
  ctx->loadAddress1Offset = ctx_def.load_addr1_offset();

  ctx->cmpValue1 = ctx_def.cmp_value_1();
  ctx->cmpValue2 = ctx_def.cmp_value_2();

  return ge::SUCCESS;
}

ge::Status FftsPlusProtoTransfer::InitCaseSwitchCtx(const domi::FftsPlusCtxDef &task_def,
    rtFftsPlusComCtx_t *const com_ctx) const {
  rtFftsPlusCaseSwitchCtx_t *const ctx = ge::PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusCaseSwitchCtx_t>(com_ctx);
  FE_ASSERT_NOTNULL(ctx);
  const domi::FftsPlusCaseSwitchCtxDef &ctx_def = task_def.case_switch_ctx();
  ctx->successorNum = static_cast<uint8_t>(ctx_def.successor_num());
  ctx->aten = static_cast<uint8_t>(ctx_def.aten() & k1BitMask);

  ctx->startLabelId = static_cast<uint8_t>(ctx_def.start_label_id());
  ctx->labelListLen = static_cast<uint8_t>(ctx_def.label_list_len());
  ctx->predCntInit = static_cast<uint8_t>(ctx_def.pred_cnt_init());
  ctx->predCnt = static_cast<uint8_t>(ctx_def.pred_cnt());

  if (ctx_def.successor_list_size() > RT_CTX_SUCCESSOR_NUM) {
    REPORT_INNER_ERR_MSG("E19999", "Size of successor_list in FftsPlusCaseDefaultCtxDef should not > %d, but %d exactly",
                       RT_CTX_SUCCESSOR_NUM, ctx_def.successor_list_size());
    GELOGE(ge::FAILED, "[Check][Param] Size of list in FftsPlusCaseDefaultCtxDef should not > %d, but %d exactly",
           RT_CTX_SUCCESSOR_NUM, ctx_def.successor_list_size());
    return ge::FAILED;
  }
  for (auto i = 0; i < ctx_def.successor_list_size(); i++) {
    ctx->successorList[i] = static_cast<uint16_t>(ctx_def.successor_list(i));
  }

  ctx->atm = static_cast<uint8_t>(ctx_def.atm() & k1BitMask);
  ctx->threadId = static_cast<uint16_t>(ctx_def.thread_id());
  ctx->threadDim = static_cast<uint16_t>(ctx_def.thread_dim());

  ctx->arSize = static_cast<uint8_t>(ctx_def.ar_size() & k3BitsMask);
  ctx->snoop = static_cast<uint8_t>(ctx_def.snoop() & k1BitMask);
  ctx->arCache = static_cast<uint8_t>(ctx_def.ar_cache() & k4BitsMask);
  ctx->arProt = static_cast<uint8_t>(ctx_def.ar_prot() & k3BitsMask);
  ctx->va = static_cast<uint8_t>(ctx_def.va() & k1BitMask);

  uint8_t *addr_base_0 = nullptr;
  if ((run_addr_handle_ != nullptr) && (run_addr_handle_(ctx_def.load_addr0_base(), addr_base_0) != ge::SUCCESS)) {
    GELOGE(ge::INTERNAL_ERROR, "[Check][GetRtAddress] logic load addr0 base is 0x%lx.", ctx_def.load_addr0_base());
    return ge::INTERNAL_ERROR;
  }
  ctx->loadAddress0BaseL = static_cast<uint32_t>(ge::PtrToValue(addr_base_0) & k32BitsMask);
  ctx->loadAddress0BaseH = static_cast<uint32_t>((ge::PtrToValue(addr_base_0) >> 32U) & k17BitsMask);
  ctx->ld0En = static_cast<uint32_t>(ctx_def.ld0_en() & k1BitMask);
  ctx->loadAddress0Offset = ctx_def.load_addr0_offset();

  uint8_t *addr_base_1 = nullptr;
  if ((run_addr_handle_ != nullptr) && (run_addr_handle_(ctx_def.load_addr1_base(), addr_base_1) != ge::SUCCESS)) {
    GELOGE(ge::INTERNAL_ERROR, "[Check][GetRtAddress] logic load addr1 base is 0x%lx.", ctx_def.load_addr1_base());
    return ge::INTERNAL_ERROR;
  }
  ctx->loadAddress1BaseL = static_cast<uint32_t>(ge::PtrToValue(addr_base_1) & k32BitsMask);
  ctx->loadAddress1BaseH = static_cast<uint32_t>((ge::PtrToValue(addr_base_1) >> 32U) & k17BitsMask);
  ctx->ld1En = static_cast<uint32_t>(ctx_def.ld1_en() & k1BitMask);
  ctx->loadAddress1Offset = ctx_def.load_addr1_offset();

  return ge::SUCCESS;
}

ge::Status FftsPlusProtoTransfer::InitCaseDefaultCtx(const domi::FftsPlusCtxDef &task_def,
    rtFftsPlusComCtx_t *const com_ctx) const {
  rtFftsPlusCaseDefCtx_t *const ctx = ge::PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusCaseDefCtx_t>(com_ctx);
  FE_ASSERT_NOTNULL(ctx);
  const domi::FftsPlusCaseDefaultCtxDef &ctx_def = task_def.case_default_ctx();
  ctx->successorNum = static_cast<uint8_t>(ctx_def.successor_num());
  ctx->aten = static_cast<uint8_t>(ctx_def.aten() & k1BitMask);

  ctx->startLabelId = static_cast<uint8_t>(ctx_def.successor_num());
  ctx->labelListLen = static_cast<uint8_t>(ctx_def.label_list_len());
  ctx->predCntInit = static_cast<uint8_t>(ctx_def.pred_cnt_init());
  ctx->predCnt = static_cast<uint8_t>(ctx_def.pred_cnt());

  if (ctx_def.successor_list_size() > RT_CTX_SUCCESSOR_NUM) {
    REPORT_INNER_ERR_MSG("E19999", "Size of successor_list in FftsPlusCaseDefaultCtxDef should not > %d, but %d exactly",
                       RT_CTX_SUCCESSOR_NUM, ctx_def.successor_list_size());
    GELOGE(ge::FAILED, "[Check][Param] Size of list in FftsPlusCaseDefaultCtxDef should not > %d, but %d exactly",
           RT_CTX_SUCCESSOR_NUM, ctx_def.successor_list_size());
    return ge::FAILED;
  }
  for (auto i = 0; i < ctx_def.successor_list_size(); i++) {
    ctx->successorList[i] = static_cast<uint16_t>(ctx_def.successor_list(i));
  }

  return ge::SUCCESS;
}

ge::Status FftsPlusProtoTransfer::InitCaseCtx(const domi::FftsPlusCtxDef &task_def,
    rtFftsPlusComCtx_t *const com_ctx) const {
  if (static_cast<int32_t>(task_def.has_case_switch_ctx()) == static_cast<int32_t>(task_def.has_case_default_ctx())) {
    REPORT_INNER_ERR_MSG("E19999", "case_switch_ctx %s and case_default_ctx %s when software ctx type is case",
                       task_def.has_case_switch_ctx() ? "exist" : "does not exist",
                       task_def.has_case_default_ctx() ? "exist" : "does not exist");
    GELOGE(ge::FAILED, "[Check][Ctx] case_switch_ctx %s and case_default_ctx %s when software ctx type is case",
           task_def.has_case_switch_ctx() ? "exist" : "does not exist",
           task_def.has_case_default_ctx() ? "exist" : "does not exist");
    return ge::FAILED;
  }

  if (task_def.has_case_switch_ctx()) {
    GE_CHK_STATUS_RET(InitCaseSwitchCtx(task_def, com_ctx), "Init CaseSwitchCtx failed");
  }
  if (task_def.has_case_default_ctx()) {
    GE_CHK_STATUS_RET(InitCaseDefaultCtx(task_def, com_ctx), "Init CaseDefaultCtx failed");
  }
  return ge::SUCCESS;
}

ge::Status FftsPlusProtoTransfer::InitAtStartCtx(const domi::FftsPlusCtxDef &task_def,
    rtFftsPlusComCtx_t *const com_ctx) const {
  rtFftsPlusAtStartCtx_t *const ctx = ge::PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusAtStartCtx_t>(com_ctx);
  FE_ASSERT_NOTNULL(ctx);
  const domi::FftsPlusAtStartCtxDef &ctx_def = task_def.at_start_ctx();
  ctx->successorNum = static_cast<uint8_t>(ctx_def.successor_num());
  ctx->aten = static_cast<uint8_t>(ctx_def.aten() & k1BitMask);
  ctx->predCntInit = static_cast<uint8_t>(ctx_def.pred_cnt_init());
  ctx->predCnt = static_cast<uint8_t>(ctx_def.pred_cnt());

  if (ctx_def.successor_list_size() > RT_CTX_SUCCESSOR_NUM) {
    REPORT_INNER_ERR_MSG("E19999", "Size of successor_list in FftsPlusAtStartCtxDef should not > %d, but %d exactly",
                       RT_CTX_SUCCESSOR_NUM, ctx_def.successor_list_size());
    GELOGE(ge::FAILED, "[Check][Param] Size of successor_list in FftsPlusAtStartCtxDef should not > %d, but %d exactly",
           RT_CTX_SUCCESSOR_NUM, ctx_def.successor_list_size());
    return ge::FAILED;
  }
  for (auto i = 0; i < ctx_def.successor_list_size(); i++) {
    ctx->successorList[i] = static_cast<uint16_t>(ctx_def.successor_list(i));
  }

  ctx->threadId = static_cast<uint16_t>(ctx_def.thread_id());
  ctx->threadDim = static_cast<uint16_t>(ctx_def.thread_dim());

  ctx->threadIdInit = static_cast<uint16_t>(ctx_def.thread_id_init());
  ctx->threadWindowSize = static_cast<uint16_t>(ctx_def.thread_window_size());

  return ge::SUCCESS;
}

ge::Status FftsPlusProtoTransfer::InitAtEndCtx(const domi::FftsPlusCtxDef &task_def,
    rtFftsPlusComCtx_t *const com_ctx) const {
  rtFftsPlusAtEndCtx_t *const ctx = ge::PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusAtEndCtx_t>(com_ctx);
  FE_ASSERT_NOTNULL(ctx);
  const domi::FftsPlusAtEndCtxDef &ctx_def = task_def.at_end_ctx();
  ctx->atStartSlotNumber = static_cast<uint8_t>(ctx_def.at_start_slot_num());
  ctx->outLabelSlotNumber = static_cast<uint8_t>(ctx_def.out_label_slot_num() & k7BitsMask);

  ctx->aten = static_cast<uint8_t>(ctx_def.aten() & k1BitMask);
  ctx->predCntInit = static_cast<uint8_t>(ctx_def.pred_cnt_init());
  ctx->predCnt = static_cast<uint8_t>(ctx_def.pred_cnt());

  if (ctx_def.succ_at_start_slot_size() > RT_CTX_SUCC_AT_START_SLOT_NUM) {
    REPORT_INNER_ERR_MSG("E19999", "Size of succ_at_start_slot in FftsPlusAtEndCtxDef should not > %d, but %d exactly",
                       RT_CTX_SUCC_AT_START_SLOT_NUM, ctx_def.succ_at_start_slot_size());
    GELOGE(ge::FAILED, "[Check][Param] Size of succ in FftsPlusAtStartCtxDef should not > %d, but %d exactly",
           RT_CTX_SUCC_AT_START_SLOT_NUM, ctx_def.succ_at_start_slot_size());
    return ge::FAILED;
  }
  for (auto i = 0; i < ctx_def.succ_at_start_slot_size(); i++) {
    ctx->succAtStartSlot[i] = static_cast<uint16_t>(ctx_def.succ_at_start_slot(i));
  }

  if (ctx_def.succ_out_label_slot_size() > RT_CTX_SUCC_OUT_LABEL_SLOT_NUM) {
    REPORT_INNER_ERR_MSG("E19999", "Size of succ_out_label_slot in FftsPlusAtEndCtxDef should not > %d, but %d exactly",
                       RT_CTX_SUCC_OUT_LABEL_SLOT_NUM, ctx_def.succ_out_label_slot_size());
    GELOGE(ge::FAILED,
           "[Check][Param] Size of succ_out_label_slot in FftsPlusAtStartCtxDef should not > %d, but %d exactly",
           RT_CTX_SUCC_OUT_LABEL_SLOT_NUM, ctx_def.succ_out_label_slot_size());
    return ge::FAILED;
  }
  for (auto i = 0; i < ctx_def.succ_out_label_slot_size(); i++) {
    ctx->succOutLabelSlot[i] = static_cast<uint16_t>(ctx_def.succ_out_label_slot(i));
  }

  ctx->threadId = static_cast<uint16_t>(ctx_def.thread_id());

  return ge::SUCCESS;
}

ge::Status FftsPlusProtoTransfer::InitLabelCtx(const domi::FftsPlusCtxDef &task_def,
    rtFftsPlusComCtx_t *const com_ctx) const {
  rtFftsPlusLabelCtx_t *const ctx = ge::PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusLabelCtx_t>(com_ctx);
  FE_ASSERT_NOTNULL(ctx);
  const domi::FftsPlusLabelCtxDef &ctx_def = task_def.label_ctx();
  ctx->successorNum = static_cast<uint8_t>(ctx_def.successor_num());
  ctx->predCntInit = static_cast<uint8_t>(ctx_def.pred_cnt_init());
  ctx->predCnt = static_cast<uint8_t>(ctx_def.pred_cnt());

  ctx->aten = static_cast<uint8_t>(ctx_def.aten() & k1BitMask);
  ctx->threadId = static_cast<uint16_t>(ctx_def.thread_id());
  ctx->threadDim = static_cast<uint16_t>(ctx_def.thread_dim());
  if (ctx_def.successor_list_size() > RT_CTX_SUCCESSOR_NUM) {
    REPORT_INNER_ERR_MSG("E19999", "Size of successor_list in FftsPlusLabelCtxDef should not > %d, but %d exactly",
                       RT_CTX_SUCCESSOR_NUM, ctx_def.successor_list_size());
    GELOGE(ge::FAILED, "[Check][Param] Size of successor_list in FftsPlusLabelCtxDef should not > %d, but %d exactly",
           RT_CTX_SUCCESSOR_NUM, ctx_def.successor_list_size());
    return ge::FAILED;
  }
  for (auto i = 0; i < ctx_def.successor_list_size(); i++) {
    ctx->successorList[i] = static_cast<uint16_t>(ctx_def.successor_list(i));
  }

  return ge::SUCCESS;
}

void FftsPlusProtoTransfer::InitAdditionalData(const domi::FftsPlusTaskDef &task_def) {
  GELOGD("init additional data start, size: %d bytes", task_def.additional_data_size());
  for (int32_t i = 0; i < task_def.additional_data_size(); ++i) {
    const domi::AdditionalDataDef &additional_data = task_def.additional_data(i);
    auto &additional_context = ctx_additional_data_[additional_data.data_type()];
    for (int32_t j = 0; j < additional_data.context_id_size(); ++j) {
      (void)additional_context.emplace(additional_data.context_id(j));
      GELOGD("additional data type:%u, context id:%u", additional_data.data_type(), additional_data.context_id(j));
    }
  }
}

template <typename T>
ge::Status FftsPlusProtoTransfer::InitThrdIoAddrs(const T &ctx_def, const uint16_t thread_id, const int32_t addr_count,
                                                  const int32_t start_idx) const {
  int32_t addr_count_start = 0;
  GE_ASSERT_TRUE(!ge::AddOverflow(addr_count, start_idx, addr_count_start));
  if ((ctx_def.task_addr_size() < (addr_count_start)) || (ctx_def.task_addr_offset_size() < addr_count)) {
    GELOGE(ge::FAILED,
           "task_addr start_idx[%i], size[%i], task_addr_offset size[%i], but "
           "need read count[%i]",
           start_idx, ctx_def.task_addr_size(), ctx_def.task_addr_offset_size(),
           addr_count);
    return ge::FAILED;
  }
  for (int32_t i = 0; i < addr_count; ++i) {
    uintptr_t logic_addr = ctx_def.task_addr(start_idx + i) + (thread_id * ctx_def.task_addr_offset(i));
    GELOGD("task base addr is %lu, offset is %lu, thread id is %u, logic addr is 0x%lx",
           ctx_def.task_addr(i), ctx_def.task_addr_offset(i), static_cast<uint32_t>(thread_id), logic_addr);
    io_addrs_.emplace_back(logic_addr);
  }
  return ge::SUCCESS;
}
}  // namespace ge

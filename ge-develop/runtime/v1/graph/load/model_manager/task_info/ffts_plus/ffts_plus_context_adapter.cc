/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ffts_plus_context_adapter.h"

namespace {
constexpr int32_t kSrcSlotNum = 4;
constexpr int32_t kNotifyIdNum = 16;
constexpr int32_t kWriteValueNum = 4;

constexpr uint32_t k1BitMask = 0x00000001U;   // 1  bit , 0000,0001
constexpr uint32_t k2BitsMask = 0x00000003U;  // 2  bits, 0000,0011
constexpr uint32_t k3BitsMask = 0x00000007U;  // 3  bits, 0000,0111
constexpr uint32_t k4BitsMask = 0x0000000FU;  // 4  bits, 0000,1111
constexpr uint32_t k5BitsMask = 0x0000001FU;  // 5  bits, 0001,1111
constexpr uint32_t k6BitsMask = 0x0000003FU;  // 6  bits, 0011,1111
constexpr uint32_t k7BitsMask = 0x0000007FU;  // 7  bits, 0111,1111
constexpr uint32_t k8BitsMask = 0x000000FFU;  // 8  bits, 1111,1111

constexpr uint32_t k12BitsMask = 0x00000FFFU;  // 12 bits, 0000,1111,1111,1111
constexpr uint32_t k32BitsMask = 0xFFFFFFFFU;  // 32 bits, 1111,1111,1111,1111,1111,1111,1111,1111
}  // namespace
namespace ge {

using CtxParserFunc = std::function<Status(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx)>;

Status FftsPlusContextAdapter::ParseCtxByType(rtFftsPlusContextType_t ctx_type, const domi::FftsPlusCtxDef &task_def,
                                              rtFftsPlusComCtx_t *const com_ctx) {
  static std::map<rtFftsPlusContextType_t, CtxParserFunc> context_parser_func{
      {RT_CTX_TYPE_AICORE, &FftsPlusContextAdapter::ParseAicAivCtx},
      {RT_CTX_TYPE_AIV, &FftsPlusContextAdapter::ParseAicAivCtx},
      {RT_CTX_TYPE_PERSISTENT_CACHE, &FftsPlusContextAdapter::ParsePersistentCacheCtx},
      {RT_CTX_TYPE_NOTIFY_WAIT, &FftsPlusContextAdapter::ParseNotifyCtx},
      {RT_CTX_TYPE_NOTIFY_RECORD, &FftsPlusContextAdapter::ParseNotifyCtx},
      {RT_CTX_TYPE_WRITE_VALUE, &FftsPlusContextAdapter::ParseWriteValueCtx},
      {RT_CTX_TYPE_MIX_AIC, &FftsPlusContextAdapter::ParseMixAicAivCtx},
      {RT_CTX_TYPE_MIX_AIV, &FftsPlusContextAdapter::ParseMixAicAivCtx},
      {RT_CTX_TYPE_SDMA, &FftsPlusContextAdapter::ParseSdmaCtx},
      {RT_CTX_TYPE_FLUSH_DATA, &FftsPlusContextAdapter::ParseDataCtx},
      {RT_CTX_TYPE_INVALIDATE_DATA, &FftsPlusContextAdapter::ParseDataCtx},
      {RT_CTX_TYPE_WRITEBACK_DATA, &FftsPlusContextAdapter::ParseDataCtx},
      {RT_CTX_TYPE_AICPU, &FftsPlusContextAdapter::ParseAicpuCtx},
      {RT_CTX_TYPE_COND_SWITCH, &FftsPlusContextAdapter::ParseCondSwitchCtx},
      {RT_CTX_TYPE_CASE_SWITCH, &FftsPlusContextAdapter::ParseCaseCtx},
      {RT_CTX_TYPE_AT_START, &FftsPlusContextAdapter::ParseAtStartCtx},
      {RT_CTX_TYPE_AT_END, &FftsPlusContextAdapter::ParseAtEndCtx},
      {RT_CTX_TYPE_LABEL, &FftsPlusContextAdapter::ParseLabelCtx},
      {RT_CTX_TYPE_DSA, &FftsPlusContextAdapter::ParseDsaCtx}};

  const auto iter = context_parser_func.find(ctx_type);
  if (iter != context_parser_func.cend()) {
    com_ctx->contextType = ctx_type;
    return iter->second(task_def, com_ctx);
  } else {
    GELOGE(FAILED, "Context type:[%d] is unsupported.", static_cast<int32_t>(ctx_type));
    return FAILED;
  }
}

void FftsPlusContextAdapter::InitFftsPlusSqe(const domi::FftsPlusSqeDef &sqe_def, rtFftsPlusSqe_t &sqe) {
  sqe.sqeHeader.l1Lock = static_cast<uint8_t>(sqe_def.sqe_header().l1_lock());
  sqe.sqeHeader.l1Unlock = static_cast<uint8_t>(sqe_def.sqe_header().l1_unlock());
  sqe.sqeHeader.blockDim = static_cast<uint16_t>(sqe_def.sqe_header().block_dim());

  sqe.wrrRatio = static_cast<uint16_t>(sqe_def.wrr_ratio() & k4BitsMask);
  sqe.sqeIndex = static_cast<uint16_t>(sqe_def.sqe_index());

  sqe.totalContextNum = static_cast<uint16_t>(sqe_def.total_context_num());
  sqe.readyContextNum = static_cast<uint16_t>(sqe_def.ready_context_num());
  sqe.preloadContextNum = static_cast<uint16_t>(sqe_def.preload_context_num());

  sqe.prefetchOstNum = static_cast<uint16_t>(sqe_def.prefetch_ost_num() & k5BitsMask);
  sqe.cmaintOstNum = static_cast<uint16_t>(sqe_def.cmaint_ost_num() & k5BitsMask);

  sqe.aicPrefetchLower = static_cast<uint16_t>(sqe_def.aic_prefetch_lower() & k5BitsMask);
  sqe.aicPrefetchUpper = static_cast<uint16_t>(sqe_def.aic_prefetch_upper() & k5BitsMask);
  sqe.aivPrefetchLower = static_cast<uint16_t>(sqe_def.aiv_prefetch_lower() & k5BitsMask);
  sqe.aivPrefetchUpper = static_cast<uint16_t>(sqe_def.aiv_prefetch_upper() & k5BitsMask);
}

Status FftsPlusContextAdapter::ParseAicAivCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx) {
  rtFftsPlusAicAivCtx_t *const ctx = PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusAicAivCtx_t>(com_ctx);
  GE_CHECK_NOTNULL(ctx);
  const domi::FftsPlusAicAivCtxDef &ctx_def = task_def.aic_aiv_ctx();
  ctx->successorNum = static_cast<uint8_t>(ctx_def.successor_num());
  ctx->aten = static_cast<uint8_t>(ctx_def.aten() & k1BitMask);
  ctx->prefetchConfig = static_cast<uint8_t>(ctx_def.prefetch_config());
  ctx->predCntInit = static_cast<uint8_t>(ctx_def.pred_cnt_init());
  ctx->predCnt = static_cast<uint8_t>(ctx_def.pred_cnt());

  GE_CHECK_LE(ctx_def.successor_list_size(), RT_CTX_SUCCESSOR_NUM);
  for (int32_t i = 0; i < ctx_def.successor_list_size(); ++i) {
    ctx->successorList[i] = static_cast<uint16_t>(ctx_def.successor_list(i));
  }

  ctx->taskParamPtrOffset = static_cast<uint16_t>(ctx_def.task_param_ptr_offset());
  uint32_t schedule_mode = ctx_def.schem();
  ctx->schem = static_cast<uint16_t>(schedule_mode & k2BitsMask);
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
  ctx->policyPri = static_cast<uint16_t>(ctx_def.policy_pri());

  GE_CHECK_LE(ctx_def.src_slot_size(), kSrcSlotNum);
  for (int32_t i = 0; i < ctx_def.src_slot_size(); ++i) {
    ctx->srcSlot[i] = static_cast<uint16_t>(ctx_def.src_slot(i));
  }

  return SUCCESS;
}

Status FftsPlusContextAdapter::ParseMixAicAivCtx(const domi::FftsPlusCtxDef &task_def,
                                                 rtFftsPlusComCtx_t *const com_ctx) {
  rtFftsPlusMixAicAivCtx_t *const ctx = PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusMixAicAivCtx_t>(com_ctx);
  GE_CHECK_NOTNULL(ctx);
  const domi::FftsPlusMixAicAivCtxDef &ctx_def = task_def.mix_aic_aiv_ctx();
  ctx->successorNum = static_cast<uint8_t>(ctx_def.successor_num());
  ctx->aten = static_cast<uint8_t>(ctx_def.aten() & k1BitMask);
  ctx->prefetchConfig = static_cast<uint8_t>(ctx_def.prefetch_config());
  ctx->predCntInit = static_cast<uint8_t>(ctx_def.pred_cnt_init());
  ctx->predCnt = static_cast<uint8_t>(ctx_def.pred_cnt());

  GE_CHECK_LE(ctx_def.successor_list_size(), RT_CTX_SUCCESSOR_NUM);
  for (int32_t i = 0; i < ctx_def.successor_list_size(); ++i) {
    ctx->successorList[i] = static_cast<uint16_t>(ctx_def.successor_list(i));
  }
  uint32_t schedule_mode = ctx_def.schem();
  ctx->schem = static_cast<uint16_t>(schedule_mode & k2BitsMask);
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
  ctx->policyPri = static_cast<uint16_t>(ctx_def.policy_pri());
  ctx->aicTaskParamPtrOffset = static_cast<uint16_t>(ctx_def.aic_task_param_ptr_offset());
  ctx->aivTaskParamPtrOffset = static_cast<uint16_t>(ctx_def.aiv_task_param_ptr_offset());

  GE_CHECK_LE(ctx_def.src_slot_size(), kSrcSlotNum);
  for (int32_t i = 0; i < ctx_def.src_slot_size(); ++i) {
    ctx->srcSlot[i] = static_cast<uint16_t>(ctx_def.src_slot(i));
  }

  return SUCCESS;
}

Status FftsPlusContextAdapter::ParseSdmaCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx) {
  rtFftsPlusSdmaCtx_t *const ctx = PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusSdmaCtx_t>(com_ctx);
  GE_CHECK_NOTNULL(ctx);
  const domi::FftsPlusSdmaCtxDef &ctx_def = task_def.sdma_ctx();
  ctx->successorNum = static_cast<uint8_t>(ctx_def.successor_num());
  ctx->aten = static_cast<uint8_t>(ctx_def.aten() & k1BitMask);
  ctx->predCntInit = static_cast<uint8_t>(ctx_def.pred_cnt_init());
  ctx->predCnt = static_cast<uint8_t>(ctx_def.pred_cnt());

  GE_CHECK_LE(ctx_def.successor_list_size(), RT_CTX_SUCCESSOR_NUM);
  for (int32_t i = 0; i < ctx_def.successor_list_size(); i++) {
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
  ctx->sourceAddressOffset = ctx_def.src_addr_offset();
  ctx->destinationAddressOffset = ctx_def.dst_addr_offset();

  ctx->nonTailDataLength = ctx_def.non_tail_data_len();
  ctx->tailDataLength = ctx_def.tail_data_len();

  return SUCCESS;
}

Status FftsPlusContextAdapter::ParseDataCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx) {
  rtFftsPlusDataCtx_t *const ctx = PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusDataCtx_t>(com_ctx);
  GE_CHECK_NOTNULL(ctx);
  const domi::FftsPlusDataCtxDef &ctx_def = task_def.data_ctx();
  ctx->successorNum = static_cast<uint8_t>(ctx_def.successor_num());
  ctx->aten = static_cast<uint8_t>(ctx_def.aten() & k1BitMask);
  ctx->cntInit = static_cast<uint8_t>(ctx_def.cnt_init());
  ctx->cnt = static_cast<uint8_t>(ctx_def.cnt());

  GE_CHECK_LE(ctx_def.successor_list_size(), RT_CTX_SUCCESSOR_NUM);
  for (int32_t i = 0; i < ctx_def.successor_list_size(); i++) {
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

  return SUCCESS;
}

Status FftsPlusContextAdapter::ParseAicpuCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx) {
  rtFftsPlusAiCpuCtx_t *const ctx = PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusAiCpuCtx_t>(com_ctx);
  GE_CHECK_NOTNULL(ctx);
  const domi::FftsPlusAicpuCtxDef &ctx_def = task_def.aicpu_ctx();
  ctx->successorNum = static_cast<uint8_t>(ctx_def.successor_num());
  ctx->aten = static_cast<uint8_t>(ctx_def.aten() & k1BitMask);
  ctx->predCntInit = static_cast<uint8_t>(ctx_def.pred_cnt_init());
  ctx->predCnt = static_cast<uint8_t>(ctx_def.pred_cnt());

  GE_CHECK_LE(ctx_def.successor_list_size(), RT_CTX_SUCCESSOR_NUM);
  for (int32_t i = 0; i < ctx_def.successor_list_size(); i++) {
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

  return SUCCESS;
}

Status FftsPlusContextAdapter::ParseDsaCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx) {
  rtFftsPlusDsaCtx_t *const ctx = PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusDsaCtx_t>(com_ctx);
  GE_CHECK_NOTNULL(ctx);
  const domi::FftsPlusDsaCtxDef &ctx_def = task_def.dsa_ctx();
  ctx->contextType = static_cast<rtFftsPlusContextType_t>(task_def.context_type());
  ctx->successorNum = static_cast<uint8_t>(ctx_def.successor_num());
  ctx->aten = static_cast<uint8_t>(ctx_def.aten());
  ctx->predCntInit = static_cast<uint8_t>(ctx_def.pred_cnt_init());
  ctx->predCnt = static_cast<uint8_t>(ctx_def.pred_cnt());
  GE_CHECK_LE(ctx_def.successor_list_size(), RT_CTX_SUCCESSOR_NUM);
  for (int32_t i = 0; i < ctx_def.successor_list_size(); i++) {
    ctx->successorList[i] = static_cast<uint16_t>(ctx_def.successor_list(i));
  }
  ctx->addressOffset = static_cast<uint16_t>(ctx_def.address_offset());
  ctx->threadId = static_cast<uint16_t>(ctx_def.thread_id());
  ctx->threadDim = static_cast<uint16_t>(ctx_def.thread_dim());
  ctx->start = static_cast<uint32_t>(ctx_def.start());
  ctx->functionType = static_cast<uint32_t>(ctx_def.distribution_type());
  ctx->dataType = static_cast<uint32_t>(ctx_def.data_type());
  ctx->algoType = static_cast<uint32_t>(ctx_def.alg_type());
  ctx->paramVldBitmap = static_cast<uint32_t>(ctx_def.input_vld());
  ctx->paramAddrValBitmap = static_cast<uint32_t>(ctx_def.input_value_addr_flag());

  if (ctx_def.seed_value_or_ptr() != kDSASetInputAddr) {
    const uint64_t seed_value_or_addr = *(PtrToPtr<char_t, uint64_t>(ctx_def.args().seed_value_or_addr().c_str()));
    ctx->dsaCfgSeedLow = static_cast<uint32_t>(seed_value_or_addr & k32BitsMask);
    ctx->dsaCfgSeedHigh = static_cast<uint32_t>(seed_value_or_addr >> k32Bits);
  }

  if (ctx_def.random_count_value_or_ptr() != kDSASetInputAddr) {
    const uint64_t random_count_value_or_addr =
        *(PtrToPtr<char_t, uint64_t>(ctx_def.args().random_count_value_or_addr().c_str()));
    ctx->dsaCfgNumberLow = static_cast<uint32_t>(random_count_value_or_addr & k32BitsMask);
    ctx->dsaCfgNumberHigh = static_cast<uint32_t>(random_count_value_or_addr >> k32Bits);
  }

  return SUCCESS;
}

Status FftsPlusContextAdapter::ParsePersistentCacheCtx(const domi::FftsPlusCtxDef &task_def,
                                                       rtFftsPlusComCtx_t *const com_ctx) {
  rtFftsPlusPersistentCacheCtx_t *const ctx = PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusPersistentCacheCtx_t>(com_ctx);
  GE_CHECK_NOTNULL(ctx);
  const domi::FftsPlusCachePersistCtxDef &ctx_def = task_def.cache_persist_ctx();
  ctx->contextType = static_cast<rtFftsPlusContextType_t>(task_def.context_type());
  ctx->successorNum = static_cast<uint8_t>(ctx_def.successor_num());
  ctx->aten = static_cast<uint8_t>(ctx_def.aten() & k1BitMask);
  ctx->predCntInit = static_cast<uint8_t>(ctx_def.pred_cnt_init());
  ctx->predCnt = static_cast<uint8_t>(ctx_def.pred_cnt());

  GE_CHECK_LE(ctx_def.successor_list_size(), RT_CTX_SUCCESSOR_NUM);
  for (int32_t i = 0; i < ctx_def.successor_list_size(); ++i) {
    ctx->successorList[i] = static_cast<uint16_t>(ctx_def.successor_list(i));
  }

  ctx->persistentEnable = static_cast<uint8_t>(ctx_def.persistent_en() & k1BitMask);
  ctx->persistentSize = static_cast<uint16_t>(ctx_def.persistent_size());
  ctx->persistentId = static_cast<uint32_t>(ctx_def.persistent_id());
  return SUCCESS;
}

Status FftsPlusContextAdapter::ParseNotifyCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx) {
  rtFftsPlusNotifyCtx_t *const ctx = PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusNotifyCtx_t>(com_ctx);
  GE_CHECK_NOTNULL(ctx);
  const domi::FftsPlusNotifyCtxDef &ctx_def = task_def.notify_ctx();
  ctx->successorNum = static_cast<uint8_t>(ctx_def.successor_num());
  ctx->aten = static_cast<uint8_t>(ctx_def.aten() & k1BitMask);
  ctx->predCntInit = static_cast<uint8_t>(ctx_def.pred_cnt_init());
  ctx->predCnt = static_cast<uint8_t>(ctx_def.pred_cnt());

  GE_CHECK_LE(ctx_def.successor_list_size(), RT_CTX_SUCCESSOR_NUM);
  for (int32_t i = 0; i < ctx_def.successor_list_size(); ++i) {
    ctx->successorList[i] = static_cast<uint16_t>(ctx_def.successor_list(i));
  }

  ctx->satm = static_cast<uint16_t>(ctx_def.satm() & k1BitMask);
  ctx->atm = static_cast<uint16_t>(ctx_def.atm() & k1BitMask);
  ctx->threadId = static_cast<uint16_t>(ctx_def.thread_id());
  ctx->threadDim = static_cast<uint16_t>(ctx_def.thread_dim());
  ctx->notifyIdBase = static_cast<uint16_t>(ctx_def.notify_id_base());
  ctx->autoWindow = static_cast<uint8_t>(ctx_def.auto_window());

  GE_CHECK_LE(ctx_def.notify_id_size(), kNotifyIdNum);
  for (int32_t i = 0; i < ctx_def.notify_id_size(); ++i) {
    ctx->notifyId[i] = static_cast<uint16_t>(ctx_def.notify_id(i));
  }

  return SUCCESS;
}

Status FftsPlusContextAdapter::ParseLabelCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx) {
  rtFftsPlusLabelCtx_t *const ctx = PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusLabelCtx_t>(com_ctx);
  GE_CHECK_NOTNULL(ctx);
  const domi::FftsPlusLabelCtxDef &ctx_def = task_def.label_ctx();
  ctx->successorNum = static_cast<uint8_t>(ctx_def.successor_num());
  ctx->predCntInit = static_cast<uint8_t>(ctx_def.pred_cnt_init());
  ctx->predCnt = static_cast<uint8_t>(ctx_def.pred_cnt());

  GE_CHECK_LE(ctx_def.successor_list_size(), RT_CTX_SUCCESSOR_NUM);
  for (int32_t i = 0; i < ctx_def.successor_list_size(); i++) {
    ctx->successorList[i] = static_cast<uint16_t>(ctx_def.successor_list(i));
  }

  return SUCCESS;
}

Status FftsPlusContextAdapter::ParseAtStartCtx(const domi::FftsPlusCtxDef &task_def,
                                               rtFftsPlusComCtx_t *const com_ctx) {
  rtFftsPlusAtStartCtx_t *const ctx = PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusAtStartCtx_t>(com_ctx);
  GE_CHECK_NOTNULL(ctx);
  const domi::FftsPlusAtStartCtxDef &ctx_def = task_def.at_start_ctx();
  ctx->successorNum = static_cast<uint8_t>(ctx_def.successor_num());
  ctx->aten = static_cast<uint8_t>(ctx_def.aten() & k1BitMask);
  ctx->predCntInit = static_cast<uint8_t>(ctx_def.pred_cnt_init());
  ctx->predCnt = static_cast<uint8_t>(ctx_def.pred_cnt());

  GE_CHECK_LE(ctx_def.successor_list_size(), RT_CTX_SUCCESSOR_NUM);
  for (int32_t i = 0; i < ctx_def.successor_list_size(); i++) {
    ctx->successorList[i] = static_cast<uint16_t>(ctx_def.successor_list(i));
  }

  ctx->threadId = static_cast<uint16_t>(ctx_def.thread_id());
  ctx->threadDim = static_cast<uint16_t>(ctx_def.thread_dim());

  ctx->threadIdInit = static_cast<uint16_t>(ctx_def.thread_id_init());
  ctx->threadWindowSize = static_cast<uint16_t>(ctx_def.thread_window_size());

  return SUCCESS;
}

Status FftsPlusContextAdapter::ParseAtEndCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx) {
  rtFftsPlusAtEndCtx_t *const ctx = PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusAtEndCtx_t>(com_ctx);
  GE_CHECK_NOTNULL(ctx);
  const domi::FftsPlusAtEndCtxDef &ctx_def = task_def.at_end_ctx();
  ctx->atStartSlotNumber = static_cast<uint8_t>(ctx_def.at_start_slot_num());
  ctx->outLabelSlotNumber = static_cast<uint8_t>(ctx_def.out_label_slot_num() & k7BitsMask);

  ctx->aten = static_cast<uint8_t>(ctx_def.aten() & k1BitMask);
  ctx->predCntInit = static_cast<uint8_t>(ctx_def.pred_cnt_init());
  ctx->predCnt = static_cast<uint8_t>(ctx_def.pred_cnt());

  GE_CHECK_LE(ctx_def.succ_at_start_slot_size(), RT_CTX_SUCC_AT_START_SLOT_NUM);
  for (int32_t i = 0; i < ctx_def.succ_at_start_slot_size(); i++) {
    ctx->succAtStartSlot[i] = static_cast<uint16_t>(ctx_def.succ_at_start_slot(i));
  }

  GE_CHECK_LE(ctx_def.succ_out_label_slot_size(), RT_CTX_SUCC_OUT_LABEL_SLOT_NUM);
  for (int32_t i = 0; i < ctx_def.succ_out_label_slot_size(); i++) {
    ctx->succOutLabelSlot[i] = static_cast<uint16_t>(ctx_def.succ_out_label_slot(i));
  }

  ctx->threadId = static_cast<uint16_t>(ctx_def.thread_id());

  return SUCCESS;
}

Status FftsPlusContextAdapter::ParseWriteValueCtx(const domi::FftsPlusCtxDef &task_def,
                                                  rtFftsPlusComCtx_t *const com_ctx) {
  rtFftsPlusWriteValueCtx_t *const ctx = PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusWriteValueCtx_t>(com_ctx);
  GE_CHECK_NOTNULL(ctx);
  const domi::FftsPlusWriteValueCtxDef &ctx_def = task_def.write_value_ctx();
  ctx->successorNum = static_cast<uint8_t>(ctx_def.successor_num());
  ctx->aten = static_cast<uint8_t>(ctx_def.aten() & k1BitMask);
  ctx->predCntInit = static_cast<uint8_t>(ctx_def.pred_cnt_init());
  ctx->predCnt = static_cast<uint8_t>(ctx_def.pred_cnt());

  GE_CHECK_LE(ctx_def.successor_list_size(), RT_CTX_SUCCESSOR_NUM);
  for (int32_t i = 0; i < ctx_def.successor_list_size(); ++i) {
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
  ctx->writeAddressOffset = ctx_def.write_addr_offset();

  GE_CHECK_LE(ctx_def.write_value_size(), kWriteValueNum);
  for (int32_t i = 0; i < ctx_def.write_value_size(); ++i) {
    ctx->writeValue[i] = static_cast<uint16_t>(ctx_def.write_value(i));
  }

  return SUCCESS;
}

Status FftsPlusContextAdapter::ParseCondSwitchCtx(const domi::FftsPlusCtxDef &task_def,
                                                  rtFftsPlusComCtx_t *const com_ctx) {
  rtFftsPlusCondSwitchCtx_t *const ctx = PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusCondSwitchCtx_t>(com_ctx);
  GE_CHECK_NOTNULL(ctx);
  const domi::FftsPlusCondSwitchCtxDef &ctx_def = task_def.cond_switch_ctx();
  ctx->trueSuccessorNum = static_cast<uint8_t>(ctx_def.true_successor_num());
  ctx->falseSuccessorNum = static_cast<uint8_t>(ctx_def.false_successor_num() & k7BitsMask);
  ctx->aten = static_cast<uint8_t>(ctx_def.aten() & k1BitMask);

  if (ctx_def.condition() == RT_COND_TYPE_MAX) {
    REPORT_INNER_ERR_MSG("E19999", "Unsupported cond type %u", ctx_def.condition());
    GELOGE(FAILED, "[Check][CtxType] Unsupported cond type %u", ctx_def.condition());
    return FAILED;
  }
  ctx->condition = static_cast<rtFftsPlusCondType_t>(ctx_def.condition());
  ctx->predCntInit = static_cast<uint8_t>(ctx_def.pred_cnt_init());
  ctx->predCnt = static_cast<uint8_t>(ctx_def.pred_cnt());

  GE_CHECK_LE(ctx_def.true_successor_list_size(), RT_CTX_TRUE_SUCCESSOR_NUM);
  for (int32_t i = 0; i < ctx_def.true_successor_list_size(); i++) {
    ctx->trueSuccessorList[i] = static_cast<uint16_t>(ctx_def.true_successor_list(i));
  }

  GE_CHECK_LE(ctx_def.false_successor_list_size(), RT_CTX_FALSE_SUCCESSOR_NUM);
  for (int32_t i = 0; i < ctx_def.false_successor_list_size(); i++) {
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

  ctx->ld0En = static_cast<uint32_t>(ctx_def.ld0_en() & k1BitMask);
  ctx->loadAddress0Offset = ctx_def.load_addr0_offset();
  ctx->ld1En = static_cast<uint32_t>(ctx_def.ld1_en() & k1BitMask);
  ctx->loadAddress1Offset = ctx_def.load_addr1_offset();

  ctx->cmpValue1 = ctx_def.cmp_value_1();
  ctx->cmpValue2 = ctx_def.cmp_value_2();

  return SUCCESS;
}

Status FftsPlusContextAdapter::ParseCaseSwitchCtx(const domi::FftsPlusCtxDef &task_def,
                                                  rtFftsPlusComCtx_t *const com_ctx) {
  rtFftsPlusCaseSwitchCtx_t *const ctx = PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusCaseSwitchCtx_t>(com_ctx);
  GE_CHECK_NOTNULL(ctx);
  const domi::FftsPlusCaseSwitchCtxDef &ctx_def = task_def.case_switch_ctx();
  ctx->successorNum = static_cast<uint8_t>(ctx_def.successor_num());
  ctx->aten = static_cast<uint8_t>(ctx_def.aten() & k1BitMask);

  ctx->startLabelId = static_cast<uint8_t>(ctx_def.start_label_id());
  ctx->labelListLen = static_cast<uint8_t>(ctx_def.label_list_len());
  ctx->predCntInit = static_cast<uint8_t>(ctx_def.pred_cnt_init());
  ctx->predCnt = static_cast<uint8_t>(ctx_def.pred_cnt());

  GE_CHECK_LE(ctx_def.successor_list_size(), RT_CTX_SUCCESSOR_NUM);
  for (int32_t i = 0; i < ctx_def.successor_list_size(); i++) {
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

  ctx->ld0En = static_cast<uint32_t>(ctx_def.ld0_en() & k1BitMask);
  ctx->loadAddress0Offset = ctx_def.load_addr0_offset();
  ctx->ld1En = static_cast<uint32_t>(ctx_def.ld1_en() & k1BitMask);
  ctx->loadAddress1Offset = ctx_def.load_addr1_offset();

  return SUCCESS;
}

Status FftsPlusContextAdapter::ParseCaseDefaultCtx(const domi::FftsPlusCtxDef &task_def,
                                                   rtFftsPlusComCtx_t *const com_ctx) {
  rtFftsPlusCaseDefCtx_t *const ctx = PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusCaseDefCtx_t>(com_ctx);
  GE_CHECK_NOTNULL(ctx);
  const domi::FftsPlusCaseDefaultCtxDef &ctx_def = task_def.case_default_ctx();
  ctx->successorNum = static_cast<uint8_t>(ctx_def.successor_num());
  ctx->aten = static_cast<uint8_t>(ctx_def.aten() & k1BitMask);

  ctx->startLabelId = static_cast<uint8_t>(ctx_def.successor_num());
  ctx->labelListLen = static_cast<uint8_t>(ctx_def.label_list_len());
  ctx->predCntInit = static_cast<uint8_t>(ctx_def.pred_cnt_init());
  ctx->predCnt = static_cast<uint8_t>(ctx_def.pred_cnt());

  GE_CHECK_LE(ctx_def.successor_list_size(), RT_CTX_SUCCESSOR_NUM);
  for (int32_t i = 0; i < ctx_def.successor_list_size(); i++) {
    ctx->successorList[i] = static_cast<uint16_t>(ctx_def.successor_list(i));
  }

  return SUCCESS;
}

Status FftsPlusContextAdapter::ParseCaseCtx(const domi::FftsPlusCtxDef &task_def, rtFftsPlusComCtx_t *const com_ctx) {
  if (static_cast<int32_t>(task_def.has_case_switch_ctx()) == static_cast<int32_t>(task_def.has_case_default_ctx())) {
    REPORT_INNER_ERR_MSG("E19999", "case_switch_ctx %s and case_default_ctx %s when software ctx type is case",
                       task_def.has_case_switch_ctx() ? "exist" : "does not exist",
                       task_def.has_case_default_ctx() ? "exist" : "does not exist");
    GELOGE(FAILED, "[Check][Ctx] case_switch_ctx %s and case_default_ctx %s when software ctx type is case",
           task_def.has_case_switch_ctx() ? "exist" : "does not exist",
           task_def.has_case_default_ctx() ? "exist" : "does not exist");
    return FAILED;
  }

  if (task_def.has_case_switch_ctx()) {
    GE_CHK_STATUS_RET(ParseCaseSwitchCtx(task_def, com_ctx), "Init CaseSwitchCtx failed");
  }
  if (task_def.has_case_default_ctx()) {
    GE_CHK_STATUS_RET(ParseCaseDefaultCtx(task_def, com_ctx), "Init CaseDefaultCtx failed");
  }
  return SUCCESS;
}

}  // namespace ge

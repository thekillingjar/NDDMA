/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mixl2_update_kernel.h"
#include "common/math/math_util.h"
#include "common/sgt_slice_type.h"
#include "aicore_update_kernel.h"
#include "register/op_tiling.h"
#include "adump_api.h"
#include "kernel/kernel_log.h"
#include "kernel/memory/mem_block.h"
#include "kernel/memory/multi_stream_mem_block.h"
#include "register/kernel_registry_impl.h"
#include "aicore/launch_kernel/ai_core_launch_kernel.h"
#include "common/dump/kernel_tracing_utils.h"
#include "exe_graph/runtime/tiling_data.h"
#include "runtime/rt_ffts_plus_define.h"
#include "runtime/rt_ffts_plus.h"
#include "runtime/mem.h"
#include "engine/aicore/kernel/rt_ffts_plus_launch_args.h"
#include "engine/ffts_plus/converter/ffts_plus_proto_transfer.h"
#include "common/dump/exception_dumper.h"
#include "framework/runtime/subscriber/global_dumper.h"
#include "graph/small_vector.h"
namespace gert {
namespace kernel {
namespace {
constexpr uint32_t kBitLen16Max = 0xFFFFU;
constexpr size_t kBitOffset = 16U;
constexpr size_t kFFTSDumpExtNum = 3U;

Status UpdateMixL2AicAivCtx(rtFftsPlusMixAicAivCtx_t *ctx, const AICoreSubTaskFlush *flush_data,
                            const uint32_t block_dim, const uint32_t schedule_mode) {
  const uint64_t paraBase = reinterpret_cast<uintptr_t>(flush_data->args_base);
  SetLow32FromSrc(ctx->aicTaskParamPtrL, paraBase);
  SetHigh16FromSrc(ctx->aicTaskParamPtrH, paraBase);
  ctx->aicTaskParamPtrOffset = 0U;
  ctx->aivTaskParamPtrH = ctx->aicTaskParamPtrH;
  ctx->aivTaskParamPtrL = ctx->aicTaskParamPtrL;
  ctx->aivTaskParamPtrOffset = 0U;
  ctx->nonTailBlockdim = static_cast<uint16_t>(block_dim);
  ctx->tailBlockdim = static_cast<uint16_t>(block_dim);
  ctx->schem = static_cast<uint16_t>(schedule_mode);
  GELOGD("MixL2 nonTail_Block_dim:%u, tail_Block_dim:%u, schedule mode:%u", ctx->nonTailBlockdim, ctx->tailBlockdim,
         schedule_mode);
  ctx->aicIcachePrefetchCnt = static_cast<uint16_t>(flush_data->aic_icache_prefetch_cnt);
  SetLow32FromSrc(ctx->nonTailAicTaskStartPcL, flush_data->aic_non_tail_task_start_pc);
  SetHigh16FromSrc(ctx->nonTailAicTaskStartPcH, flush_data->aic_non_tail_task_start_pc);
  SetLow32FromSrc(ctx->tailAicTaskStartPcL, flush_data->aic_non_tail_task_start_pc);
  SetHigh16FromSrc(ctx->tailAicTaskStartPcH, flush_data->aic_non_tail_task_start_pc);
  ctx->aivIcachePrefetchCnt = flush_data->aiv_icache_prefetch_cnt;
  SetLow32FromSrc(ctx->nonTailAivTaskStartPcL, flush_data->aiv_non_tail_task_start_pc);
  SetHigh16FromSrc(ctx->nonTailAivTaskStartPcH, flush_data->aiv_non_tail_task_start_pc);
  SetLow32FromSrc(ctx->tailAivTaskStartPcL, flush_data->aiv_non_tail_task_start_pc);
  SetHigh16FromSrc(ctx->tailAivTaskStartPcH, flush_data->aiv_non_tail_task_start_pc);
  GELOGD("MixL2 %lu|0x%lx|%lu|0x%lx", flush_data->aic_icache_prefetch_cnt, flush_data->aic_non_tail_task_start_pc,
         flush_data->aiv_icache_prefetch_cnt, flush_data->aiv_non_tail_task_start_pc);
  GELOGD("MixL2 context update success");
  return ge::SUCCESS;
}

Status UpdateMixL2RatioCtx(rtFftsPlusMixAicAivCtx_t *ctx, const int64_t c_ratio, const int64_t v_ratio) {
  uint16_t core_type = RT_CTX_TYPE_MIX_AIC;
  int32_t ratio = 0;
  if (c_ratio == 0) {
    core_type = RT_CTX_TYPE_MIX_AIV;
  } else if (v_ratio == 0) {
    core_type = RT_CTX_TYPE_MIX_AIC;
  } else if (c_ratio > v_ratio) {
    ratio = static_cast<int32_t>(c_ratio / v_ratio);
    core_type = RT_CTX_TYPE_MIX_AIV;
  } else {
    ratio = static_cast<int32_t>(v_ratio / c_ratio);
    core_type = RT_CTX_TYPE_MIX_AIC;
  }
  ctx->contextType = core_type;
  ctx->nonTailBlockRatioN = static_cast<uint8_t>(ratio);
  ctx->tailBlockRatioN = static_cast<uint8_t>(ratio);
  GELOGD("Dyn Ratio context update: contextType:%u, nonTailBlockRatioN:%u, tailBlockRatioN:%u", ctx->contextType,
         ctx->nonTailBlockRatioN, ctx->tailBlockRatioN);
  return ge::SUCCESS;
}

ge::graphStatus GetCtxFromKernel(const KernelContext *context, rtFftsPlusMixAicAivCtx_t *&ctx) {
  auto ctx_id = context->GetInputValue<uint32_t>(static_cast<size_t>(MixL2UpdateKey::CTX_ID));
  auto task_info = context->GetInputValue<rtFftsPlusTaskInfo_t *>(static_cast<size_t>(MixL2UpdateKey::TASK_INFO));
  FE_ASSERT_NOTNULL(task_info);
  FE_ASSERT_NOTNULL(task_info->descBuf);
  FE_ASSERT_NOTNULL(task_info->fftsPlusSqe);
  auto *context_head = reinterpret_cast<rtFftsPlusComCtx_t *>(const_cast<void *>(task_info->descBuf));
  FE_ASSERT_NOTNULL(context_head);
  if (ctx_id >= task_info->fftsPlusSqe->totalContextNum) {
    GELOGE(ge::GRAPH_FAILED, "Context Id(%u) overflow.", ctx_id);
    return ge::GRAPH_FAILED;
  }
  ctx = reinterpret_cast<rtFftsPlusMixAicAivCtx_t *>(context_head + ctx_id);
  FE_ASSERT_NOTNULL(ctx);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MixL2UpdateContext(KernelContext *context) {
  GELOGD("Update MixL2 node begin.");
  auto flush_data = context->GetInputPointer<AICoreSubTaskFlush>(static_cast<size_t>(MixL2UpdateKey::FLUSH_DATA));
  FE_ASSERT_NOTNULL(flush_data);
  auto block_dim = context->GetInputValue<uint32_t>(static_cast<size_t>(MixL2UpdateKey::BLOCK_DIM));
  auto schedule_mode = context->GetInputValue<uint32_t>(static_cast<size_t>(MixL2UpdateKey::SCHEDULE_MODE));
  rtFftsPlusMixAicAivCtx_t *ctx = nullptr;
  if (GetCtxFromKernel(context, ctx) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  UpdateMixL2AicAivCtx(ctx, flush_data, block_dim, schedule_mode);
  auto is_mix_ratio = context->GetInputValue<bool>(static_cast<size_t>(MixL2UpdateKey::MIX_RATIO));
  if (!is_mix_ratio) {
    return ge::GRAPH_SUCCESS;
  }

  auto tiling_key = context->GetInputValue<size_t>(static_cast<size_t>(MixL2UpdateKey::TILING_KEY));

  auto tiling_keys = context->GetInputPointer<gert::ContinuousVector>(static_cast<size_t>(MixL2UpdateKey::TILING_VEC));
  FE_ASSERT_NOTNULL(tiling_keys);
  auto tiling_key_vec = reinterpret_cast<const uint64_t *>(tiling_keys->GetData());

  auto c_ratios = context->GetInputPointer<gert::ContinuousVector>(static_cast<size_t>(MixL2UpdateKey::CRATIO_VEC));
  FE_ASSERT_NOTNULL(c_ratios);
  auto c_ratio_vec = reinterpret_cast<const int64_t *>(c_ratios->GetData());

  auto v_ratios = context->GetInputPointer<gert::ContinuousVector>(static_cast<size_t>(MixL2UpdateKey::VRATIO_VEC));
  FE_ASSERT_NOTNULL(v_ratios);
  auto v_ratio_vec = reinterpret_cast<const int64_t *>(v_ratios->GetData());

  const size_t kernel_num = tiling_keys->GetSize();
  int64_t c_ratio;
  int64_t v_ratio;
  size_t i = 0;
  for (; i < kernel_num; ++i) {
    if (tiling_key_vec[i] == tiling_key) {
      c_ratio = c_ratio_vec[i];
      v_ratio = v_ratio_vec[i];
      break;
    }
  }
  if (i == kernel_num) {
    GELOGE(ge::FAILED, "Dyn Ratio tilingKey match failed.");
    return ge::GRAPH_FAILED;
  }
  UpdateMixL2RatioCtx(ctx, c_ratio, v_ratio);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FillMixL2ProfilingInfo(const KernelContext *context, ProfilingInfoWrapper &prof_info) {
  rtFftsPlusMixAicAivCtx_t *ctx = nullptr;
  if (GetCtxFromKernel(context, ctx) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  prof_info.SetBlockDim((static_cast<uint32_t>(ctx->nonTailBlockdim) & kBitLen16Max) |
                        (static_cast<uint32_t>(ctx->nonTailBlockRatioN) << kBitOffset));
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(MixL2UpdateContext).RunFunc(MixL2UpdateContext).ProfilingInfoFiller(FillMixL2ProfilingInfo);

inline void InitMixL2Addrs(size_t index, const void *addr, uintptr_t *args_host_data) {
  const uintptr_t data_base = ge::PtrToValue(addr);
  GELOGD("Addr base: 0x%lx, args index: %zu", data_base, index);
  args_host_data[index] = data_base;
}

ge::graphStatus GetMixL2FlushData(const KernelContext *context, AICoreSubTaskFlush *flush_data) {
  auto sink_ret = context->GetInputPointer<AICoreSinkRet>(static_cast<size_t>(MixL2ArgsInKey::SINK_RET));
  FE_ASSERT_NOTNULL(sink_ret);
  flush_data->aic_non_tail_task_start_pc = sink_ret->aic_non_tail_task_start_pc;
  flush_data->aic_tail_task_start_pc = sink_ret->aic_tail_task_start_pc;
  flush_data->aic_icache_prefetch_cnt = sink_ret->aic_icache_prefetch_cnt;
  flush_data->aiv_non_tail_task_start_pc = sink_ret->aiv_non_tail_task_start_pc;
  flush_data->aiv_tail_task_start_pc = sink_ret->aiv_tail_task_start_pc;
  flush_data->aiv_icache_prefetch_cnt = sink_ret->aiv_icache_prefetch_cnt;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SaveMixL0ExceptionDump(KernelContext *context, RtFFTSKernelLaunchArgs *rt_args, size_t io_addr_num,
                                       const ContinuousVector *workspace, const gert::GertTensorData *shape_buffer) {
  auto need_assert = context->GetInputValue<uint32_t>(static_cast<size_t>(MixL2ArgsInKey::HAS_ASSERT));
  if (!GlobalDumper::GetInstance()->IsEnable(gert::DumpType::kLiteExceptionDump) && need_assert != 1) {
    return ge::GRAPH_SUCCESS;
  }
  ge::SmallVector<uint64_t, 8U> in_outputs_size;
  uint32_t size = 0U;
  for (size_t io_index = 0; io_index < io_addr_num; ++io_index) {
    auto tensor_data = context->GetInputValue<gert::GertTensorData *>(static_cast<size_t>(MixL2ArgsInKey::IO_START) +
                                                                      io_addr_num + io_index);
    FE_ASSERT_NOTNULL(tensor_data);
    if (tensor_data->GetAddr() == nullptr) {
      in_outputs_size.emplace_back(0UL);
    } else {
      const auto dy_desc = rt_args->GetDyDescByIoIndex(io_index);
      if (dy_desc != nullptr && dy_desc->is_group_first) {
        uint64_t in_val = (static_cast<uint64_t>(L0DumpType::kFoldedWithDesc) << kDumpTypeBitNum) | dy_desc->dyn_num;
        GELOGD("Dynamic index: %d with number %u, reported value: %lx.", dy_desc->io_index, dy_desc->dyn_num, in_val);
        in_outputs_size.emplace_back(in_val);
        ++size;
      }
      in_outputs_size.emplace_back(tensor_data->GetSize());
    }
    ++size;
  }
  if (shape_buffer != nullptr) {
    in_outputs_size.emplace_back(shape_buffer->GetSize());
    ++size;
  }
  size_t work_size = workspace->GetSize();
  GELOGD("need_assert value: %u, work_size value: %zu.", need_assert, work_size);
  auto work_addrs = reinterpret_cast<GertTensorData *const *>(workspace->GetData());
  FE_ASSERT_NOTNULL(work_addrs);
  for (size_t i = 0; i < work_size; i++) {
    uint64_t w_size = work_addrs[i]->GetSize();
    if (need_assert == 1 && i == 0) {
      GELOGD("work_0_size: %lu.", w_size);
      w_size = (kAssertWorkFlag << kDumpTypeBitNum) | w_size;
    }
    in_outputs_size.emplace_back(w_size);
    ++size;
  }

  rtArgsSizeInfo_t rt_args_size{};
  auto &addr = rt_args_size.infoAddr;
  const size_t limit_adx_size = (Adx::MAX_TENSOR_NUM - Adx::ADUMP_ARGS_EXCEPTION_HEAD - kFFTSDumpExtNum);
  size = (size > limit_adx_size) ? limit_adx_size : size;
  size_t report_size = Adx::ADUMP_ARGS_EXCEPTION_HEAD + size + kFFTSDumpExtNum;
  addr = Adx::AdumpGetSizeInfoAddr(report_size, rt_args_size.atomicIndex);
  FE_ASSERT_NOTNULL(addr);
  size_t rep_num = 0;
  static_cast<uint64_t *>(addr)[rep_num++] = static_cast<uint64_t>(rt_args_size.atomicIndex);
  static_cast<uint64_t *>(addr)[rep_num++] = size + kFFTSDumpExtNum;
  uint64_t ctx_id = rt_args->NeedAtomicClean() ? 1UL : 0UL;
  static_cast<uint64_t *>(addr)[rep_num++] = ctx_id;
  uint64_t args_table_size = rt_args->GetArgsPointer<uint8_t>(RtFFTSKernelLaunchArgs::kArgsTypeEnd) -
                             rt_args->GetArgsPointer<uint8_t>(RtFFTSKernelLaunchArgs::kArgsHostAddr);
  static_cast<uint64_t *>(addr)[rep_num++] = args_table_size;
  static_cast<uint64_t *>(addr)[rep_num++] = (kDumpSkipAddrNum << kDumpOffsetBitNum) | static_cast<uint64_t>(size);
  KLOGD("Report l0 dump with io: %zu, size: %zu, ctx_id: %lu, args_size: %lu.", io_addr_num, size, ctx_id, args_table_size);
  for (size_t i = 0UL; i < static_cast<size_t>(size); ++i) {
    GELOGD("Report info idx:%zu val:%lx.", i, in_outputs_size[i]);
    static_cast<uint64_t *>(addr)[rep_num + i] = in_outputs_size[i];
  }
  (void)rtSetExceptionExtInfo(&rt_args_size);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FFTSUpdateMixL2Args(KernelContext *context) {
  auto workspace = context->GetInputPointer<ContinuousVector>(static_cast<size_t>(MixL2ArgsInKey::WORKSPACE));
  FE_ASSERT_NOTNULL(workspace);
  auto need_mode_addr = context->GetInputValue<uint32_t>(static_cast<size_t>(MixL2ArgsInKey::NEED_MODE_ADDR));
  auto io_addr_num = context->GetInputValue<size_t>(static_cast<size_t>(MixL2ArgsInKey::IO_ADDR_NUM));
  auto flush_data = context->GetOutputPointer<AICoreSubTaskFlush>(static_cast<size_t>(MixL2ArgsOutKey::FLUSH_DATA));
  FE_ASSERT_NOTNULL(flush_data);
  auto args_para = context->GetOutputPointer<NodeMemPara>(static_cast<size_t>(MixL2ArgsOutKey::ARGS_PARA));
  FE_ASSERT_NOTNULL(args_para);
  FE_ASSERT_NOTNULL(args_para->host_addr);
  FE_ASSERT_NOTNULL(args_para->dev_addr);
  auto rt_args = reinterpret_cast<RtFFTSKernelLaunchArgs *>(args_para->host_addr);
  if (GetMixL2FlushData(context, flush_data) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  size_t work_size = workspace->GetSize();
  FE_ASSERT_TRUE(work_size <= rt_args->GetWorkspaceCap());
  size_t add_addr_size = (need_mode_addr == 1U);
  GELOGD("Update args size: add addr[%zu] io[%zu], workspace[%zu].", add_addr_size, io_addr_num, work_size);
  uintptr_t *args_base_addr = static_cast<uintptr_t *>(rt_args->GetArgBase());
  FE_ASSERT_NOTNULL(args_base_addr);
  size_t args_pos = rt_args->GetArgsPos();
  uintptr_t *args_host_data = &args_base_addr[args_pos];
  size_t args_abs_pos = rt_args->GetArgsAbsPos();
  flush_data->args_base = static_cast<void *>(&(static_cast<uint8_t *>(args_para->dev_addr)[args_abs_pos]));
  GELOGD("Args: host_pos[%zu], abs_pos[%zu], dev_base[%lx], dev_args_addr[%lx].", args_pos, args_abs_pos,
         args_para->host_addr, flush_data->args_base);
  size_t arg_index = 0;
  if (need_mode_addr == 1U) {
    uint64_t mode_addr = 0U;
    uint32_t len = 0U;
    GE_CHK_RT_RET(rtGetC2cCtrlAddr(&mode_addr, &len));
    InitMixL2Addrs(arg_index++, reinterpret_cast<void *>(mode_addr), args_host_data);
  }

  for (size_t io_index = 0; io_index < io_addr_num; ++io_index) {
    auto io_shape = context->GetInputValue<StorageShape *>(static_cast<size_t>(MixL2ArgsInKey::IO_START) + io_index);
    FE_ASSERT_NOTNULL(io_shape);
    auto tensor_data = context->GetInputValue<gert::GertTensorData *>(static_cast<size_t>(MixL2ArgsInKey::IO_START) +
                                                                      io_addr_num + io_index);
    FE_ASSERT_NOTNULL(tensor_data);
    const uintptr_t data_base = ge::PtrToValue(tensor_data->GetAddr());
    GELOGD("Addr base:0x%lx, io index:%zu, args index:%zu", data_base, io_index, arg_index);
    rt_args->SetIoAddr(io_index, arg_index, data_base, io_shape->GetStorageShape(), args_para->dev_addr);
  }

  // set shape buffer addr
  auto shape_buffer = context->GetInputValue<gert::GertTensorData *>(static_cast<size_t>(MixL2ArgsInKey::SHAPE_BUFFER));
  if (shape_buffer != nullptr) {
    GELOGD("Node add shape buffer addr.");
    InitMixL2Addrs(arg_index++, shape_buffer->GetAddr(), args_host_data);
  }
  FE_ASSERT_GRAPH_SUCCESS(SaveMixL0ExceptionDump(context, rt_args, io_addr_num, workspace, shape_buffer));
  auto work_addr = reinterpret_cast<GertTensorData *const *>(workspace->GetData());
  for (size_t i = 0; i < work_size; ++i) {
    InitMixL2Addrs(arg_index++, work_addr[i]->GetAddr(), args_host_data);
  }
  size_t tiling_pos = rt_args->GetTilingAbsPos();
  auto tiling_base = static_cast<void *>(&(static_cast<uint8_t *>(args_para->dev_addr)[tiling_pos]));
  GELOGD("Tiling database: %lx, position: %zu", tiling_base, tiling_pos);
  InitMixL2Addrs(arg_index, tiling_base, args_host_data);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MixL2UpdateGeDataDumpInfo(const KernelContext *context, gert::DataDumpInfoWrapper &wrapper) {
  size_t in_num = context->GetInputValue<size_t>(static_cast<size_t>(MixL2DataDumpKey::IN_NUM));
  size_t out_num = context->GetInputValue<size_t>(static_cast<size_t>(MixL2DataDumpKey::OUT_NUM));
  if (in_num > kMaxIndexNum || out_num > kMaxIndexNum) {
    GELOGE(ge::FAILED, "In/Out io num:%zu %zu are over max num.", in_num, out_num);
    return ge::GRAPH_FAILED;
  }

  auto ctx_id = context->GetInputValue<uint32_t>(static_cast<size_t>(MixL2DataDumpKey::CONTEXT_ID));
  size_t io_num = in_num + out_num;
  uint32_t thread_id = 0;
  if (wrapper.CreateFftsCtxInfo(thread_id, ctx_id) != ge::GRAPH_SUCCESS) {
    GELOGE(ge::FAILED, "CreateFftsCtxInfo failed, thread_id:%u, ctx_id:%u.", thread_id, ctx_id);
    return ge::GRAPH_FAILED;
  }

  // input + output
  for (size_t input_i = 0U; input_i < io_num; ++input_i) {
    bool is_input = input_i < in_num;
    auto tensor_data =
        context->GetInputValue<gert::GertTensorData *>(static_cast<size_t>(MixL2DataDumpKey::IO_START) + input_i);
    FE_ASSERT_NOTNULL(tensor_data);
    FE_ASSERT_NOTNULL(tensor_data->GetAddr());
    uint64_t addr = static_cast<uint64_t>(ge::PtrToValue(tensor_data->GetAddr()));
    if (wrapper.AddFftsCtxAddr(thread_id, is_input, addr, tensor_data->GetSize()) != ge::GRAPH_SUCCESS) {
      GELOGE(ge::FAILED, "AddFftsCtxAddr input failed, thread_id:%u, ctx_id:%u, input_addr:%lu, input_size:%lu.",
             thread_id, ctx_id, addr, tensor_data->GetSize());
      return ge::GRAPH_FAILED;
    }
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MixL2UpdateDataDumpInfo(KernelContext *context) {
  (void)context;
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(MixL2UpdateDataDumpInfo).RunFunc(MixL2UpdateDataDumpInfo).DataDumpInfoFiller(MixL2UpdateGeDataDumpInfo);

ge::graphStatus MixL2UpdateGeExceptionDumpInfo(const KernelContext *context, gert::ExceptionDumpInfoWrapper &wrapper) {
  auto workspace = context->GetInputPointer<ContinuousVector>(static_cast<size_t>(MixL2ExceptionDumpKey::WORKSPACE));
  FE_ASSERT_NOTNULL(workspace);
  // workspace
  size_t work_size = workspace->GetSize();
  auto workspace_addrs_data = reinterpret_cast<GertTensorData *const *>(workspace->GetData());
  for (size_t i = 0U; i < work_size; ++i) {
    FE_ASSERT_NOTNULL(workspace_addrs_data[i]);
    uintptr_t workspace_addr = static_cast<uintptr_t>(ge::PtrToValue(workspace_addrs_data[i]->GetAddr()));
    int64_t addr_size = static_cast<int64_t>(workspace_addrs_data[i]->GetSize());
    GELOGD("Add workspace, index: %zu, workspace_addr: %lu, addr_size: %ld, work_size: %zu.", i,
           static_cast<uint64_t>(workspace_addr), addr_size, work_size);
    wrapper.AddWorkspace(workspace_addr, addr_size);
  }

  // args
  auto args_para = context->GetInputValue<NodeMemPara *>(static_cast<size_t>(MixL2ExceptionDumpKey::ARGS_PARA));
  FE_ASSERT_NOTNULL(args_para);
  FE_ASSERT_NOTNULL(args_para->host_addr);
  auto rt_args_host = reinterpret_cast<RtFFTSKernelLaunchArgs *>(args_para->host_addr);
  uintptr_t args_addr = static_cast<uintptr_t>(
      ge::PtrToValue(rt_args_host->GetArgsPointer<uint8_t>(RtFFTSKernelLaunchArgs::kArgsHostAddr)));
  size_t args_size = rt_args_host->GetArgsCap(RtFFTSKernelLaunchArgs::kArgsHostAddr);
  GELOGD("Add host arguments, args_addr: %lu, args_size: %zu.", static_cast<uint64_t>(args_addr), args_size);
  wrapper.SetHostArgs(args_addr, args_size);

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus MixL2UpdateExceptionDumpInfo(KernelContext *context) {
  (void)context;
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(MixL2UpdateExceptionDumpInfo)
    .RunFunc(MixL2UpdateExceptionDumpInfo)
    .ExceptionDumpInfoFiller(MixL2UpdateGeExceptionDumpInfo);

ge::graphStatus CreateMixL2FlushData(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  auto task_flush = context->GetOutput(static_cast<size_t>(MixL2ArgsOutKey::FLUSH_DATA));
  FE_ASSERT_NOTNULL(task_flush);
  task_flush->SetWithDefaultDeleter(new (std::nothrow) AICoreSubTaskFlush());
  return ge::GRAPH_SUCCESS;
}

void PrintArgsProc(const KernelContext *context, std::vector<std::string> &args_str) {
  auto args_para = context->GetOutputPointer<NodeMemPara>(static_cast<size_t>(MixL2ArgsOutKey::ARGS_PARA));
  if (args_para == nullptr) {
    args_str.emplace_back("Launch args pointer is null.");
    return;
  }
  auto rt_args = reinterpret_cast<RtFFTSKernelLaunchArgs *>(args_para->host_addr);
  if (rt_args == nullptr) {
    args_str.emplace_back("Launch rt args pointer is null.");
    return;
  }
  std::stringstream ss;
  uintptr_t *args_base_addr = static_cast<uintptr_t *>(rt_args->GetArgBase());
  ss << "Tiling data: ";
  size_t print_size = (rt_args->GetArgsAbsPos() - rt_args->GetTilingAbsPos()) / sizeof(TensorAddress);
  PrintHex(args_base_addr, print_size, ss);
  args_str.emplace_back(ss.str());
  ss.str("");
  ss << "All args: ";
  print_size = rt_args->GetArgsCap(RtFFTSKernelLaunchArgs::kArgsHostAddr) / sizeof(TensorAddress);
  PrintHex(args_base_addr + rt_args->GetArgsPos(), print_size, ss);
  args_str.emplace_back(ss.str());

  ss.str("");
  ss << "Dynamic in args: ";
  auto all_size = rt_args->GetArgsCap(RtFFTSKernelLaunchArgs::kDyInputsHostAddr) / sizeof(TensorAddress);
  size_t offset = 0;
  while (all_size != 0) {
    print_size = all_size >= MAX_PRINT_SIZE ? MAX_PRINT_SIZE : all_size;
    all_size -= print_size;
    PrintHex(rt_args->GetArgsPointer<uintptr_t>(RtFFTSKernelLaunchArgs::kDyInputsHostAddr) + offset, print_size, ss);
    offset += print_size;
    args_str.emplace_back(ss.str());
    ss.str("");
  }
  ss << "Dynamic out args: ";
  all_size = rt_args->GetArgsCap(RtFFTSKernelLaunchArgs::kDyOutputsHostAddr) / sizeof(TensorAddress);
  offset = 0;
  while (all_size != 0) {
    print_size = all_size >= MAX_PRINT_SIZE ? MAX_PRINT_SIZE : all_size;
    all_size -= print_size;
    PrintHex(rt_args->GetArgsPointer<uintptr_t>(RtFFTSKernelLaunchArgs::kDyOutputsHostAddr) + offset, print_size, ss);
    offset += print_size;
    args_str.emplace_back(ss.str());
    ss.str("");
  }
  return;
}

std::vector<std::string> PrintMixL2Args(const KernelContext *context) {
  std::stringstream ss;
  ss << "FFTS Plus launch MixL2 args: ";
  std::vector<std::string> msgs;
  msgs.emplace_back(ss.str());
  std::vector<std::string> args_str;
  PrintArgsProc(context, args_str);
  msgs.insert(msgs.cend(), args_str.cbegin(), args_str.cend());
  return msgs;
}
REGISTER_KERNEL(FFTSUpdateMixL2Args)
    .RunFunc(FFTSUpdateMixL2Args)
    .OutputsCreator(CreateMixL2FlushData)
    .TracePrinter(PrintMixL2Args);

ge::graphStatus CopyTilingdata(KernelContext *context) {
  errno_t ret;
  GELOGD("Begin copying tiling data");
  auto args_para = context->GetInputPointer<NodeMemPara>(0);
  auto data_str = context->GetInputStrPointer(1);
  auto data_size = context->GetInputValue<size_t>(2);
  FE_ASSERT_NOTNULL(data_str);
  FE_ASSERT_NOTNULL(args_para);
  FE_ASSERT_NOTNULL(args_para->host_addr);
  auto rt_args = reinterpret_cast<RtFFTSKernelLaunchArgs *>(args_para->host_addr);
  auto &tilingData = rt_args->GetTilingData();
  auto des_data_ptr = tilingData.GetData();
  FE_ASSERT_NOTNULL(des_data_ptr);
  GELOGI("Des cap: %zu with src size: %zu.", tilingData.GetCapacity(), data_size);
  ret = memcpy_s(des_data_ptr, tilingData.GetCapacity(), data_str, data_size);
  FE_ASSERT_EOK(ret, "Copy tilingdata failed.");
  tilingData.SetDataSize(data_size);
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(CopyTilingdata).RunFunc(CopyTilingdata);

ge::graphStatus CopyAtomicTilingdata(KernelContext *context) {
  errno_t ret;
  GELOGD("Copy atomic tilingdata begin");
  auto args_para = context->GetInputPointer<NodeMemPara>(0);
  auto data_str = context->GetInputStrPointer(1);
  auto data_size = context->GetInputValue<size_t>(2);
  FE_ASSERT_NOTNULL(args_para);
  FE_ASSERT_NOTNULL(args_para->host_addr);
  auto rt_args = reinterpret_cast<RtFFTSKernelLaunchArgs *>(args_para->host_addr);
  auto &tilingData = rt_args->GetAtomTilingData();
  auto des_data_ptr = tilingData.GetData();
  GELOGI("Des cap: %zu with src size: %zu.", tilingData.GetCapacity(), data_size);
  ret = memcpy_s(des_data_ptr, tilingData.GetCapacity(), data_str, data_size);
  FE_ASSERT_EOK(ret, "Copy atomic tilingdata failed.");
  tilingData.SetDataSize(data_size);
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(CopyAtomicTilingdata).RunFunc(CopyAtomicTilingdata);
}  // namespace
}  // namespace kernel
}  // namespace gert
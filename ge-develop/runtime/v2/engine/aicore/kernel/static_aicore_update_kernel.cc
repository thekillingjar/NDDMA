/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicore_update_kernel.h"
#include "register/ffts_node_calculater_registry.h"
#include "kernel/kernel_log.h"
#include "kernel/memory/mem_block.h"
#include "kernel/memory/multi_stream_mem_block.h"
#include "engine/aicore/kernel/rt_ffts_plus_launch_args.h"
#include "common/dump/kernel_tracing_utils.h"
#include "exe_graph/runtime/tiling_data.h"
#include "register/kernel_registry_impl.h"
#include "runtime/rt_ffts_plus_define.h"
#include "runtime/rt_ffts_plus.h"
#include "common/math/math_util.h"
#include "common/sgt_slice_type.h"
#include "exe_graph/runtime/tensor.h"
#include "engine/ffts_plus/converter/ffts_plus_proto_transfer.h"
#include "core/executor/multi_thread_topological/executor/schedule/producer/producers/kernel_tags/critical_section_config.h"
namespace gert {
namespace kernel {
namespace {
ge::graphStatus UpdateManualCmoCtxProc(const rtFftsPlusTaskInfo_t *task_info, const gert::ContinuousVector *cmo_idx,
    const gert::ContinuousVector *ctx_id_vec, const AICoreSubTaskFlush *flush_data, size_t type) {
  const size_t idx_num = cmo_idx->GetSize();
  auto idx_vec = reinterpret_cast<const int32_t *>(cmo_idx->GetData());
  auto ctx_vec = reinterpret_cast<const uint32_t *>(ctx_id_vec->GetData());
  rtFftsPlusComCtx_t *context_head = reinterpret_cast<rtFftsPlusComCtx_t*>(const_cast<void*>(task_info->descBuf));
  uint16_t total_num = task_info->fftsPlusSqe->totalContextNum;
  for (size_t j = 0U; j < idx_num; ++j) {
    int32_t io_index = idx_vec[j];
    uint32_t ctx_id = ctx_vec[j];
    if (ctx_id >= total_num) {
      GELOGE(ge::FAILED, "Data ctx[%u] over total num[%u].", ctx_id, total_num);
      return ge::GRAPH_FAILED;
    }
    const auto &addr_base_vv = (type == 0U) ? flush_data->input_addr_vv : flush_data->output_addr_vv;
    if (static_cast<size_t>(io_index) >= kMaxIndexNum) {
      GELOGE(ge::FAILED, "[FillDataCtx]Index(%d) over size.", io_index);
      return ge::GRAPH_FAILED;
    }
    uint64_t para_base = addr_base_vv[io_index][0U];
    GELOGD("Update data context [%u] for io index: %d with address: %lx.", ctx_id, io_index, para_base);
    FE_ASSERT_TRUE(para_base != 0U);
    auto data_ctx = reinterpret_cast<rtFftsPlusDataCtx_t*>(context_head + ctx_id);
    SetLow32FromSrc(data_ctx->addressBaseL, para_base);
    SetHigh32FromSrc(data_ctx->addressBaseH, para_base);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus UpdateStaticCmoProc(const KernelContext *context, const rtFftsPlusTaskInfo_t *task_info,
    const AICoreSubTaskFlush *flush_data) {
  for (size_t i = 0U; i < kMaxCmoType; ++i) {
    auto cmo_idx = context->GetInputPointer<gert::ContinuousVector>(
        static_cast<size_t>(StaUpdateKey::PREFETCH_IDX) + (i << 1));
    FE_ASSERT_NOTNULL(cmo_idx);
    const size_t idx_num = cmo_idx->GetSize();
    if (idx_num == 0U) {
      continue;
    }
    auto ctx_id_vec = context->GetInputPointer<gert::ContinuousVector>(
        static_cast<size_t>(StaUpdateKey::PREFETCH_CTX) + (i << 1));
    FE_ASSERT_NOTNULL(ctx_id_vec);
    if (ctx_id_vec->GetSize() != idx_num) {
      GELOGE(ge::FAILED, "Cmo index num[%zu] not equal with ctx id num[%zu].", idx_num, ctx_id_vec->GetSize());
      return ge::GRAPH_FAILED;
    }
    if (UpdateManualCmoCtxProc(task_info, cmo_idx, ctx_id_vec, flush_data, i) != ge::GRAPH_SUCCESS) {
      return ge::GRAPH_FAILED;
    }
  }
  return ge::GRAPH_SUCCESS;
}

inline void UpdateStaticAicAivCtx(rtFftsPlusAicAivCtx_t *ctx, const AICoreSubTaskFlush *flush_data) {
  uint64_t paraBase = reinterpret_cast<uintptr_t>(flush_data->args_base);
  SetLow32FromSrc(ctx->taskParamPtrBaseL, paraBase);
  SetHigh16FromSrc(ctx->taskParamPtrBaseH, paraBase);
  SetLow32FromSrc(ctx->nonTailTaskStartPcL, flush_data->aic_non_tail_task_start_pc);
  SetHigh16FromSrc(ctx->nonTailTaskStartPcH, flush_data->aic_non_tail_task_start_pc);
  SetLow32FromSrc(ctx->tailTaskStartPcL, flush_data->aic_tail_task_start_pc);
  SetHigh16FromSrc(ctx->tailTaskStartPcH, flush_data->aic_tail_task_start_pc);
  ctx->icachePrefetchCnt = flush_data->aic_icache_prefetch_cnt;
  ctx->taskParamPtrOffset = flush_data->param_ptr_offset * sizeof(uintptr_t);
  GELOGD("Update context with para base:%lx.", paraBase);
  return;
}

struct AutoArgProc {
  uintptr_t *args_host_data{nullptr};
  size_t in_num{0U};
};

ge::graphStatus StaManualUpdateContext(KernelContext *context) {
  KLOGD("Update AICore static node begin.");
  auto flush_data = context->GetInputPointer<AICoreSubTaskFlush>(static_cast<size_t>(StaUpdateKey::FLUSH_DATA));
  FE_ASSERT_NOTNULL(flush_data);
  auto task_info = context->GetInputValue<rtFftsPlusTaskInfo_t*>(static_cast<size_t>(StaUpdateKey::TASK_INFO));
  FE_ASSERT_NOTNULL(task_info);
  FE_ASSERT_NOTNULL(task_info->descBuf);
  FE_ASSERT_NOTNULL(task_info->fftsPlusSqe);
  if (UpdateStaticCmoProc(context, task_info, flush_data) != ge::GRAPH_SUCCESS) {
    KLOGE("Update data context failed.");
    return ge::GRAPH_FAILED;
  }
  auto context_id = context->GetInputValue<uint32_t>(static_cast<size_t>(StaUpdateKey::AICORE_CTX));
  auto *context_head = reinterpret_cast<rtFftsPlusComCtx_t*>(const_cast<void*>(task_info->descBuf));
  uint16_t total_num = task_info->fftsPlusSqe->totalContextNum;
  if (context_id >= total_num) {
    KLOGE("Context Id(%u) overflow with %u.", context_id, total_num);
    return ge::GRAPH_FAILED;
  }
  auto ctx = reinterpret_cast<rtFftsPlusAicAivCtx_t*>(context_head + context_id);
  FE_ASSERT_NOTNULL(ctx);
  UpdateStaticAicAivCtx(ctx, flush_data);
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(StaManualUpdateContext).RunFunc(StaManualUpdateContext);

inline void InitCtxIoAddrs(size_t index, const void *addr, size_t in_num,
    AICoreSubTaskFlush *flush_data, uintptr_t *args_host_data) {
  const uintptr_t data_base = ge::PtrToValue(addr);
  if (index < in_num) {
    flush_data->input_addr_vv[index][0U] = data_base;
  } else {
    flush_data->output_addr_vv[index - in_num][0U] = data_base;
  }
  GELOGD("Addr_base: 0x%lx, addr_index: %zu.", data_base, index);
  args_host_data[index] = data_base;
}
inline void InitCtxWorkAddrs(size_t index, const void *addr, uintptr_t *args_host_data) {
  const uintptr_t data_base = ge::PtrToValue(addr);
  GELOGD("Addr_base: 0x%lx, addr_index: %zu.", data_base, index);
  args_host_data[index] = data_base;
}
ge::graphStatus FFTSUpdateStaAICoreArgs(KernelContext *context) {
  auto workspace = context->GetInputPointer<ContinuousVector>(static_cast<size_t>(StaArgsInKey::WORKSPACE));
  FE_ASSERT_NOTNULL(workspace);
  auto sink_ret = context->GetInputPointer<AICoreSinkRet>(static_cast<size_t>(StaArgsInKey::SINK_RET));
  FE_ASSERT_NOTNULL(sink_ret);
  auto args_para = context->GetInputValue<NodeMemPara*>(static_cast<size_t>(StaArgsInKey::ARGS_PARA));
  FE_ASSERT_NOTNULL(args_para);
  auto in_num = context->GetInputValue<size_t>(static_cast<size_t>(StaArgsInKey::IN_NUM));
  auto out_num = context->GetInputValue<size_t>(static_cast<size_t>(StaArgsInKey::OUT_NUM));
  if (in_num > kMaxIndexNum || out_num > kMaxIndexNum) {
    KLOGE("In/Out io num: %zu/%zu exceeds maximum allowed number.", in_num, out_num);
    return ge::GRAPH_FAILED;
  }
  auto flush_data = context->GetOutputPointer<AICoreSubTaskFlush>(static_cast<size_t>(StaArgsOutKey::FLUSH_DATA));
  FE_ASSERT_NOTNULL(flush_data);
  flush_data->aic_non_tail_task_start_pc = sink_ret->aic_non_tail_task_start_pc;
  flush_data->aic_tail_task_start_pc = sink_ret->aic_tail_task_start_pc;
  flush_data->aic_icache_prefetch_cnt = sink_ret->aic_icache_prefetch_cnt;

  size_t work_size = workspace->GetSize();
  size_t io_num = in_num + out_num;
  flush_data->param_ptr_offset = io_num + work_size;
  GELOGD("Update args: io=%zu, workspace=%zu.", io_num, work_size);
  auto rt_args = reinterpret_cast<RtFFTSKernelLaunchArgs *>(args_para->host_addr);
  size_t real_size = flush_data->param_ptr_offset * sizeof(uintptr_t);
  FE_ASSERT_TRUE(real_size <= rt_args->GetArgsCap(RtFFTSKernelLaunchArgs::kArgsHostAddr));
  GELOGD("Host real size is [%zu], with a maximum size of [%zu].", real_size, args_para->size);
  uintptr_t *args_base_addr = static_cast<uintptr_t*>(rt_args->GetArgBase());
  FE_ASSERT_NOTNULL(args_base_addr);
  size_t args_pos = rt_args->GetArgsPos();
  uintptr_t *args_host_data = &args_base_addr[args_pos];
  size_t args_abs_pos = rt_args->GetArgsAbsPos();
  flush_data->args_base = static_cast<void*>(&(static_cast<uint8_t*>(args_para->dev_addr)[args_abs_pos]));
  GELOGD("Args: host_pos[%zu], absolute_pos[%zu], rtArgs: base[%lx], device_args_addr[%lx].", args_pos, args_abs_pos,
         args_para->host_addr, flush_data->args_base);
  size_t arg_index = 0U;
  // input + output
  for (size_t input_i = 0U; input_i < io_num; ++input_i) {
    auto tensor_data = context->GetInputValue<gert::GertTensorData *>(
        static_cast<size_t>(StaArgsInKey::IO_START) + input_i);
    FE_ASSERT_NOTNULL(tensor_data);
    InitCtxIoAddrs(arg_index, tensor_data->GetAddr(), in_num, flush_data, args_host_data);
    arg_index++;
  }
  // workspace
  auto work_addr = reinterpret_cast<GertTensorData *const *>(workspace->GetData());
  for (size_t i = 0U; i < work_size; ++i) {
    InitCtxWorkAddrs(arg_index, work_addr[i]->GetAddr(), args_host_data);
    arg_index++;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus StaticUpdateManualGeDataDumpInfo(const KernelContext *context, gert::DataDumpInfoWrapper &wrapper) {
  size_t in_num = context->GetInputValue<size_t>(static_cast<size_t>(ManualDataDumpKey::IN_NUM));
  size_t out_num = context->GetInputValue<size_t>(static_cast<size_t>(ManualDataDumpKey::OUT_NUM));
  if (in_num > kMaxIndexNum || out_num > kMaxIndexNum) {
    GELOGE(ge::FAILED, "In/Out io num:%zu %zu over max num.", in_num, out_num);
    return ge::GRAPH_FAILED;
  }

  auto ctx_id = context->GetInputValue<uint32_t>(static_cast<size_t>(ManualDataDumpKey::CONTEXT_ID));
  auto thread_id = context->GetInputValue<uint32_t>(static_cast<size_t>(ManualDataDumpKey::THREAD_ID));

  size_t io_num = in_num + out_num;
  std::vector<uint64_t> input_shapes_size_vec;
  std::vector<uint64_t> output_shapes_size_vec;
  for (size_t i = 0; i < io_num; ++i) {
    uint64_t shape_size = static_cast<uint64_t>(
        context->GetInputValue<size_t>(static_cast<size_t>(ManualDataDumpKey::IO_START) + i));
    if (i < in_num) {
      input_shapes_size_vec.emplace_back(shape_size);
    } else {
      output_shapes_size_vec.emplace_back(shape_size);
    }
  }

  if (wrapper.CreateFftsCtxInfo(thread_id, ctx_id) != ge::GRAPH_SUCCESS) {
    GELOGE(ge::FAILED, "CreateFftsCtxInfo failed, thread_id:%u, ctx_id:%u.", thread_id, ctx_id);
    return ge::GRAPH_FAILED;
  }

  // input + output
  for (size_t i = 0U; i < io_num; ++i) {
    bool is_input = i < in_num;
    size_t index = is_input ? i : (i - in_num);
    uint64_t addr_size = is_input ? input_shapes_size_vec[index] : output_shapes_size_vec[index];
    auto tensor_data =
        context->GetInputValue<gert::GertTensorData *>(static_cast<size_t>(ManualDataDumpKey::IO_START) + io_num + i);
    FE_ASSERT_NOTNULL(tensor_data);
    FE_ASSERT_NOTNULL(tensor_data->GetAddr());
    uint64_t ctx_addr = static_cast<uint64_t>(ge::PtrToValue(tensor_data->GetAddr()));
    if (wrapper.AddFftsCtxAddr(thread_id, is_input, ctx_addr, addr_size) != ge::GRAPH_SUCCESS) {
      GELOGE(ge::FAILED, "AddFftsCtxAddr input failed, thread_id:%u, ctx_id:%u, input_addr:%lu, input_size:%lu.",
             thread_id, ctx_id, ctx_addr, addr_size);
      return ge::GRAPH_FAILED;
    }
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus StaticUpdateManualDataDumpInfo(KernelContext *context) {
  (void)context;
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(StaticUpdateManualDataDumpInfo).RunFunc(StaticUpdateManualDataDumpInfo).
    DataDumpInfoFiller(StaticUpdateManualGeDataDumpInfo);

ge::graphStatus StaticManualUpdateGeExceptionDumpInfo(const KernelContext *context,
                                                      gert::ExceptionDumpInfoWrapper &wrapper) {
  auto workspace = context->GetInputPointer<ContinuousVector>(static_cast<size_t>(ManualExceptionDumpKey::WORKSPACE));
  FE_ASSERT_NOTNULL(workspace);
  // workspace
  size_t work_size = workspace->GetSize();
  auto workspace_addrs_data = reinterpret_cast<GertTensorData *const *>(workspace->GetData());
  for (size_t i = 0U; i < work_size; ++i) {
    FE_ASSERT_NOTNULL(workspace_addrs_data[i]);
    uintptr_t workspace_addr = static_cast<uintptr_t>(ge::PtrToValue(workspace_addrs_data[i]->GetAddr()));
    int64_t addr_size = static_cast<int64_t>(workspace_addrs_data[i]->GetSize());
    GELOGD("Add workspace, index: %zu, workspace address: %lu, address size: %ld, work size: %zu.",
           i, static_cast<uint64_t>(workspace_addr), addr_size, work_size);
    wrapper.AddWorkspace(workspace_addr, addr_size);
  }

  // args
  auto args_para = context->GetInputValue<NodeMemPara*>(static_cast<size_t>(ManualExceptionDumpKey::ARGS_PARA));
  FE_ASSERT_NOTNULL(args_para);
  FE_ASSERT_NOTNULL(args_para->host_addr);
  auto rt_args_host_addr = reinterpret_cast<RtFFTSKernelLaunchArgs *>(args_para->host_addr);
  uintptr_t args_addr = static_cast<uintptr_t>(
      ge::PtrToValue(rt_args_host_addr->GetArgsPointer<uint8_t>(RtFFTSKernelLaunchArgs::kArgsHostAddr)));
  size_t args_size = rt_args_host_addr->GetArgsCap(RtFFTSKernelLaunchArgs::kArgsHostAddr);
  GELOGD("Add host arguments, args_addr: %lu, args_size: %zu.", static_cast<uint64_t>(args_addr), args_size);
  wrapper.SetHostArgs(args_addr, args_size);

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus StaticManualUpdateExceptionDumpInfo(KernelContext *context) {
  (void)context;
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(StaticManualUpdateExceptionDumpInfo).RunFunc(StaticManualUpdateExceptionDumpInfo).
    ExceptionDumpInfoFiller(StaticManualUpdateGeExceptionDumpInfo);

ge::graphStatus CreateStaticFlushData(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  auto flush_data = context->GetOutput(static_cast<size_t>(StaArgsOutKey::FLUSH_DATA));
  FE_ASSERT_NOTNULL(flush_data);
  flush_data->SetWithDefaultDeleter(new (std::nothrow) AICoreSubTaskFlush());
  return ge::GRAPH_SUCCESS;
}

std::string PrintStaticArgs(const KernelContext *context) {
  auto args_para = context->GetInputValue<NodeMemPara*>(static_cast<size_t>(StaArgsInKey::ARGS_PARA));
  if (args_para == nullptr) {
    return "Launch manual args pointer is null.";
  }
  auto rt_args = reinterpret_cast<RtFFTSKernelLaunchArgs *>(args_para->host_addr);
  std::stringstream ss;
  ss << "All args: ";
  uintptr_t *args_base_addr = static_cast<uintptr_t*>(rt_args->GetArgBase());
  size_t print_size = (rt_args->GetAtomTilingAbsPos() - rt_args->GetArgsAbsPos()) / sizeof(TensorAddress);
  PrintHex(args_base_addr + rt_args->GetArgsPos(), print_size, ss);
  return ss.str();
}

std::vector<std::string> PrintStaticAICoreArgs(const KernelContext *context) {
  std::stringstream ss;
  ss << "FFTS Plus launch static aicore args: ";
  std::vector<std::string> msgs;
  msgs.emplace_back(ss.str());
  msgs.emplace_back(PrintStaticArgs(context));
  return msgs;
}
REGISTER_KERNEL(FFTSUpdateStaAICoreArgs).RunFunc(FFTSUpdateStaAICoreArgs).OutputsCreator(CreateStaticFlushData).
    TracePrinter(PrintStaticAICoreArgs).ConcurrentCriticalSectionKey(kKernelUseMemory);

void InitAutoIoAddrs(size_t index, const void *addr, const uint64_t *offset_vec,
                     AICoreSubTaskFlush *flush_data, AutoArgProc &proc_arg) {
  const uintptr_t data_base = ge::PtrToValue(addr);
  for (uint32_t i = 0U; i < flush_data->thread_dim; ++i) {
    const size_t ctx_io_idx = (flush_data->param_ptr_offset * i) + index;
    GELOGD("Addr_base: 0x%lx, addr_index: %zu, thread index: %u, thread addr offset: 0x%lx, args index: %zu",
           data_base, index, i, offset_vec[index], ctx_io_idx);
    proc_arg.args_host_data[ctx_io_idx] = (data_base + (offset_vec[index] * i));
    if (index < proc_arg.in_num) {
      flush_data->input_addr_vv[index][i] = data_base;
    } else {
      flush_data->output_addr_vv[index - proc_arg.in_num][i] = data_base;
    }
  }
}

bool InitAutoWorkAddrs(size_t index, const memory::FftsMemBlock *ffts_mem, AICoreSubTaskFlush *flush_data,
                                  uintptr_t *args_host_data) {
  for (uint32_t i = 0U; i < flush_data->thread_dim; ++i) {
    const size_t ctx_io_idx = (flush_data->param_ptr_offset * i) + index;
    FE_ASSERT_NOTNULL(ffts_mem->Addr(i));
    args_host_data[ctx_io_idx] = ge::PtrToValue(ffts_mem->Addr(i));
    GELOGD("Addr_base: 0x%lx, addr_index: %zu, thread index: %u, args index: %zu",
           args_host_data[ctx_io_idx], index, i, ctx_io_idx);
  }
  return true;
}

ge::graphStatus FFTSUpdateAutoAICoreArgs(KernelContext *context) {
  auto workspace = context->GetInputPointer<ContinuousVector>(static_cast<size_t>(AutoArgsInKey::WORKSPACE));
  FE_ASSERT_NOTNULL(workspace);
  auto sink_ret = context->GetInputPointer<AICoreSinkRet>(static_cast<size_t>(AutoArgsInKey::SINK_RET));
  FE_ASSERT_NOTNULL(sink_ret);
  auto args_para = context->GetInputValue<NodeMemPara*>(static_cast<size_t>(AutoArgsInKey::ARGS_PARA));
  FE_ASSERT_NOTNULL(args_para);
  auto thread_dim = context->GetInputValue<uint32_t>(static_cast<size_t>(AutoArgsInKey::THREAD_DIM));
  auto window_size = context->GetInputValue<uint32_t>(static_cast<size_t>(AutoArgsInKey::WINDOW_SIZE));
  auto offset = context->GetInputPointer<ContinuousVector>(static_cast<size_t>(AutoArgsInKey::THREAD_OFFSET));
  FE_ASSERT_NOTNULL(offset);
  AutoArgProc proc_arg;
  proc_arg.in_num = context->GetInputValue<size_t>(static_cast<size_t>(AutoArgsInKey::IN_NUM));
  auto out_num = context->GetInputValue<size_t>(static_cast<size_t>(AutoArgsInKey::OUT_NUM));
  if (proc_arg.in_num > kMaxIndexNum || out_num > kMaxIndexNum) {
    KLOGE("In/Out io num: %zu/%zu exceeds maximum allowed number.", proc_arg.in_num, out_num);
    return ge::GRAPH_FAILED;
  }
  auto flush_data = context->GetOutputPointer<AICoreSubTaskFlush>(static_cast<size_t>(StaArgsOutKey::FLUSH_DATA));
  FE_ASSERT_NOTNULL(flush_data);
  flush_data->thread_dim = thread_dim;
  flush_data->window_size = window_size;
  flush_data->aic_non_tail_task_start_pc = sink_ret->aic_non_tail_task_start_pc;
  flush_data->aic_tail_task_start_pc = sink_ret->aic_tail_task_start_pc;
  flush_data->aic_icache_prefetch_cnt = sink_ret->aic_icache_prefetch_cnt;

  size_t work_size = workspace->GetSize();
  size_t io_num = proc_arg.in_num + out_num;
  flush_data->param_ptr_offset = io_num + work_size;
  if (flush_data->param_ptr_offset != offset->GetSize()) {
    KLOGE("Args num[%u] not equal offset[%zu].", flush_data->param_ptr_offset, offset->GetSize());
    return ge::GRAPH_FAILED;
  }
  GELOGD("Update args in num:%zu, out num:%zu, work num:%zu.", proc_arg.in_num, out_num, work_size);
  auto rt_args = reinterpret_cast<RtFFTSKernelLaunchArgs *>(args_para->host_addr);
  size_t real_size = flush_data->param_ptr_offset * thread_dim * sizeof(uintptr_t);
  FE_ASSERT_TRUE(real_size <= rt_args->GetArgsCap(RtFFTSKernelLaunchArgs::kArgsHostAddr));
  GELOGD("Host real size is [%zu], with a maximum size of [%zu].", real_size, args_para->size);
  uintptr_t *args_base_addr = static_cast<uintptr_t*>(rt_args->GetArgBase());
  FE_ASSERT_NOTNULL(args_base_addr);
  size_t args_pos = rt_args->GetArgsPos();
  proc_arg.args_host_data = &args_base_addr[args_pos];
  size_t args_abs_pos = rt_args->GetArgsAbsPos();
  flush_data->args_base = static_cast<void*>(&(static_cast<uint8_t*>(args_para->dev_addr)[args_abs_pos]));
  GELOGD("Args: host_pos[%zu], absolute_pos[%zu], rtArgs: base[%lx], device_args_addr[%lx].", args_pos, args_abs_pos,
         args_para->host_addr, flush_data->args_base);
  auto offset_vec = reinterpret_cast<const uint64_t *>(offset->GetData());
  // input + output
  for (size_t input_i = 0U; input_i < io_num; ++input_i) {
    auto tensor_data = context->GetInputValue<gert::GertTensorData *>(
        static_cast<size_t>(AutoArgsInKey::IO_START) + input_i);
    FE_ASSERT_NOTNULL(tensor_data);
    FE_ASSERT_NOTNULL(tensor_data->GetAddr());
    InitAutoIoAddrs(input_i, tensor_data->GetAddr(), offset_vec, flush_data, proc_arg);
  }
  // workspace
  auto work_addr = reinterpret_cast<memory::FftsMemBlock *const *>(workspace->GetData());
  for (size_t i = 0U; i < work_size; ++i) {
    FE_ASSERT_NOTNULL(work_addr[i]);
    FE_ASSERT_TRUE(InitAutoWorkAddrs(io_num + i, work_addr[i], flush_data, proc_arg.args_host_data));
  }
  return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus GetAddrSizeAndOffset(size_t input_i, bool is_input, size_t index, uint32_t thread_dim,
                                            uint32_t thread_id, const std::vector<uint64_t> &input_orishapes_size_vec,
                                            const std::vector<uint64_t> &output_orishapes_size_vec,
                                            const uint64_t *offset_vec,
                                            uint64_t &addr_size, uint64_t &offset) {
  if (offset_vec[input_i] == 0) {
    // no slice, addr_size is ori_shape size
    addr_size = is_input ? input_orishapes_size_vec[index] : output_orishapes_size_vec[index];
    offset = addr_size;
  } else {
    // slice, get non_tail addr_size from offset_vec, tail addr_size = orishape_size - non_tail_size
    if (thread_id == thread_dim - 1) {
      uint64_t orishape_size = is_input ? input_orishapes_size_vec[index] : output_orishapes_size_vec[index];
      uint64_t non_tail_shapes_size = (thread_dim - 1) * offset_vec[input_i];
      if (orishape_size < non_tail_shapes_size) {
        GELOGE(ge::FAILED, "Orishape_size[%lu] smaller than non_tail_shapes_size[%lu].",
               orishape_size, non_tail_shapes_size);
        return ge::GRAPH_FAILED;
      }
      addr_size = orishape_size - non_tail_shapes_size;
    } else {
      addr_size = offset_vec[input_i];
    }
    offset = offset_vec[input_i];
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus StaticUpdateAutoGeDataDumpInfo(const KernelContext *context, gert::DataDumpInfoWrapper &wrapper) {
  uint32_t thread_dim = context->GetInputValue<uint32_t>(static_cast<size_t>(AutoDataDumpKey::THREAD_DIM));
  if (thread_dim == 0U) {
    GELOGE(ge::FAILED, "Thread dim is zero.");
    return ge::GRAPH_FAILED;
  }
  uint32_t window_size = context->GetInputValue<uint32_t>(static_cast<size_t>(AutoDataDumpKey::WINDOW_SIZE));
  if (window_size == 0U) {
    GELOGE(ge::FAILED, "Window size is zero.");
    return ge::GRAPH_FAILED;
  }
  auto offset = context->GetInputPointer<ContinuousVector>(static_cast<size_t>(AutoDataDumpKey::THREAD_ADDR_OFFSET));
  FE_ASSERT_NOTNULL(offset);
  auto offset_vec = reinterpret_cast<const uint64_t *>(offset->GetData());
  size_t in_num = context->GetInputValue<size_t>(static_cast<size_t>(AutoDataDumpKey::IN_NUM));
  size_t out_num = context->GetInputValue<size_t>(static_cast<size_t>(AutoDataDumpKey::OUT_NUM));
  if (in_num > kMaxIndexNum || out_num > kMaxIndexNum) {
    GELOGE(ge::FAILED, "In/Out io num:%zu/%zu over max num.", in_num, out_num);
    return ge::GRAPH_FAILED;
  }

  auto ctx_ids = context->GetInputPointer<gert::ContinuousVector>(static_cast<size_t>(AutoDataDumpKey::AICORE_CTX));
  FE_ASSERT_NOTNULL(ctx_ids);
  auto ctx_id_vec = reinterpret_cast<const int32_t*>(ctx_ids->GetData());

  size_t io_num = in_num + out_num;
  std::vector<uint64_t> input_orishapes_size_vec;
  std::vector<uint64_t> output_orishapes_size_vec;
  for (size_t i = 0; i < io_num; ++i) {
    uint64_t shape_size = static_cast<uint64_t>(
        context->GetInputValue<size_t>(static_cast<size_t>(AutoDataDumpKey::IO_START) + i));
    if (i < in_num) {
      input_orishapes_size_vec.emplace_back(shape_size);
    } else {
      output_orishapes_size_vec.emplace_back(shape_size);
    }
  }
  size_t  ctx_id_vec_size = ctx_ids->GetSize();
  if (ctx_id_vec_size <= (static_cast<size_t>((thread_dim - 1) % window_size))) {
    GELOGE(ge::FAILED, "Ctx_id_vec size(%zu) error, thread_dim is %u, window_size is %u.",
           ctx_id_vec_size, thread_dim, window_size);
    return ge::GRAPH_FAILED;
  }
  for (uint32_t thread_id = 0; thread_id < thread_dim; ++thread_id) {
    uint32_t ctxid = ctx_id_vec[thread_id % window_size];
    if (wrapper.CreateFftsCtxInfo(thread_id, ctxid) != ge::GRAPH_SUCCESS) {
      GELOGE(ge::FAILED, "CreateFftsCtxInfo failed, thread_id:%u, ctxid:%u.", thread_id, ctxid);
      return ge::GRAPH_FAILED;
    }
    // input + output
    for (size_t i = 0U; i < io_num; ++i) {
      bool is_input = i < in_num;
      size_t index = is_input ? i : (i - in_num);
      uint64_t addr_size = 0;
      uint64_t io_offset = 0;
      if (GetAddrSizeAndOffset(i, is_input, index, thread_dim, thread_id, input_orishapes_size_vec,
                               output_orishapes_size_vec, offset_vec, addr_size, io_offset) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
      }
      auto tensor_data =
          context->GetInputValue<gert::GertTensorData *>(static_cast<size_t>(AutoDataDumpKey::IO_START) + io_num + i);
      FE_ASSERT_NOTNULL(tensor_data);
      FE_ASSERT_NOTNULL(tensor_data->GetAddr());
      const uintptr_t data_base = static_cast<uintptr_t>(ge::PtrToValue(tensor_data->GetAddr()));
      uint64_t addr = static_cast<uint64_t>(data_base + thread_id * io_offset);
      if (wrapper.AddFftsCtxAddr(thread_id, is_input, addr, addr_size) != ge::GRAPH_SUCCESS) {
        GELOGE(ge::FAILED, "AddFftsCtxAddr input failed, thread_id:%u, ctxid:%u, input_addr:%lu, input_size:%lu.",
               thread_id, ctxid, addr, addr_size);
        return ge::GRAPH_FAILED;
      }
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus StaticUpdateAutoDataDumpInfo(KernelContext *context) {
  (void)context;
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(StaticUpdateAutoDataDumpInfo).RunFunc(StaticUpdateAutoDataDumpInfo).
    DataDumpInfoFiller(StaticUpdateAutoGeDataDumpInfo);

ge::graphStatus StaticAutoUpdateGeExceptionDumpInfo(const KernelContext *context,
                                                    gert::ExceptionDumpInfoWrapper &wrapper) {
  uint32_t thread_dim = context->GetInputValue<uint32_t>(static_cast<size_t>(AutoExceptionDumpKey::THREAD_DIM));
  if (thread_dim == 0U) {
    GELOGE(ge::FAILED, "Thread dim is zero.");
    return ge::GRAPH_FAILED;
  }
  auto workspace = context->GetInputPointer<ContinuousVector>(static_cast<size_t>(AutoExceptionDumpKey::WORKSPACE));
  FE_ASSERT_NOTNULL(workspace);
  // workspace
  size_t work_size = workspace->GetSize();
  auto addrs_data = reinterpret_cast<memory::FftsMemBlock *const *>(workspace->GetData());
  for (size_t i = 0U; i < work_size; ++i) {
    FE_ASSERT_NOTNULL(addrs_data[i]);
    int64_t addr_size = static_cast<int64_t>(addrs_data[i]->GetSize());
    GELOGD("Add workspace, index: %zu, address size: %ld, work size: %zu.", i, addr_size, work_size);
    // worksapce addr and addr_size need to process for each thread id
    for (size_t j = 0; j < thread_dim; ++j) {
      uintptr_t workspace_addr = static_cast<uintptr_t>(ge::PtrToValue(addrs_data[i]->Addr(j)));
      GELOGD("Add thread[%zu] workspace addr[%lu]", j, static_cast<uint64_t>(workspace_addr));
      wrapper.AddWorkspace(workspace_addr, addr_size);
    }
  }

  // args
  auto args_para = context->GetInputValue<NodeMemPara*>(static_cast<size_t>(AutoExceptionDumpKey::ARGS_PARA));
  FE_ASSERT_NOTNULL(args_para);
  FE_ASSERT_NOTNULL(args_para->host_addr);
  auto rt_args = reinterpret_cast<RtFFTSKernelLaunchArgs *>(args_para->host_addr);
  uintptr_t args_addr = static_cast<uintptr_t>(
      ge::PtrToValue(rt_args->GetArgsPointer<uintptr_t>(RtFFTSKernelLaunchArgs::kArgsHostAddr)));
  size_t args_size = rt_args->GetArgsCap(RtFFTSKernelLaunchArgs::kArgsHostAddr);
  GELOGD("Add host arguments, args_addr: %lu, args_size: %zu.", static_cast<uint64_t>(args_addr), args_size);
  wrapper.SetHostArgs(args_addr, args_size);

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus StaticAutoUpdateExceptionDumpInfo(KernelContext *context) {
  (void)context;
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(StaticAutoUpdateExceptionDumpInfo).RunFunc(StaticAutoUpdateExceptionDumpInfo).
    ExceptionDumpInfoFiller(StaticAutoUpdateGeExceptionDumpInfo);

std::string PrintAutoStaticArgs(const KernelContext *context) {
  auto args_para = context->GetInputValue<NodeMemPara*>(static_cast<size_t>(AutoArgsInKey::ARGS_PARA));
  if (args_para == nullptr) {
    return "Launch auto args pointer is null.";
  }
  auto rt_args = reinterpret_cast<RtFFTSKernelLaunchArgs *>(args_para->host_addr);
  std::stringstream ss;
  ss << "All args: ";
  uintptr_t *args_base_addr = static_cast<uintptr_t*>(rt_args->GetArgBase());
  PrintHex(args_base_addr, rt_args->GetArgsSize() / sizeof(TensorAddress), ss);
  return ss.str();
}

std::vector<std::string> PrintStaticAutoAICoreArgs(const KernelContext *context) {
  std::stringstream ss;
  ss << "FFTS Plus launch static aicore args: ";
  std::vector<std::string> msgs;
  msgs.emplace_back(ss.str());
  msgs.emplace_back(PrintAutoStaticArgs(context));
  return msgs;
}
REGISTER_KERNEL(FFTSUpdateAutoAICoreArgs).RunFunc(FFTSUpdateAutoAICoreArgs).OutputsCreator(CreateStaticFlushData).
    TracePrinter(PrintStaticAutoAICoreArgs);

ge::graphStatus UpdateAutoCmoCtxProc(const rtFftsPlusTaskInfo_t *task_info, const gert::ContinuousVector *cmo_idx,
    const gert::ContinuousVector *ctx_id_vec, const AICoreSubTaskFlush *flush_data, size_t type) {
  const size_t idx_num = cmo_idx->GetSize();
  auto idx_vec = reinterpret_cast<const int32_t *>(cmo_idx->GetData());
  auto ctx_vec = reinterpret_cast<const uint32_t *>(ctx_id_vec->GetData());
  rtFftsPlusComCtx_t *context_head = reinterpret_cast<rtFftsPlusComCtx_t*>(const_cast<void*>(task_info->descBuf));
  uint16_t total_num = task_info->fftsPlusSqe->totalContextNum;
  for (size_t j = 0U; j < idx_num; ++j) {
    int32_t io_index = idx_vec[j];
    if (static_cast<size_t>(io_index) >= kMaxIndexNum) {
      GELOGE(ge::FAILED, "[FillDataCtx]Index(%d) over size.", io_index);
      return ge::GRAPH_FAILED;
    }
    const auto &addr_base_vv = (type == 0U) ? flush_data->input_addr_vv : flush_data->output_addr_vv;
    uint64_t para_base = addr_base_vv[io_index][0U];
    FE_ASSERT_TRUE(para_base != 0U);
    for (size_t i = 0U; i < static_cast<size_t>(flush_data->window_size); ++i) {
      uint32_t ctx_id = ctx_vec[j * flush_data->window_size + i];
      GELOGD("Update data context [%u] for io index: %d with address: %lx.", ctx_id, io_index, para_base);
      if (ctx_id >= total_num) {
        GELOGE(ge::FAILED, "Data ctx[%u] over total num[%u].", ctx_id, total_num);
        return ge::GRAPH_FAILED;
      }
      auto data_ctx = reinterpret_cast<rtFftsPlusDataCtx_t*>(context_head + ctx_id);
      SetLow32FromSrc(data_ctx->addressBaseL, para_base);
      SetHigh32FromSrc(data_ctx->addressBaseH, para_base);
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus UpdateStaticAutoCmoProc(const KernelContext *context, const rtFftsPlusTaskInfo_t *task_info,
    const AICoreSubTaskFlush *flush_data) {
  for (size_t i = 0U; i < kMaxCmoType; ++i) {
    auto cmo_idx = context->GetInputPointer<gert::ContinuousVector>(
        static_cast<size_t>(AutoUpdateKey::PREFETCH_IDX) + (i << 1));
    FE_ASSERT_NOTNULL(cmo_idx);
    const size_t idx_num = cmo_idx->GetSize();
    if (idx_num == 0U) {
      continue;
    }
    auto ctx_id_vec = context->GetInputPointer<gert::ContinuousVector>(
        static_cast<size_t>(AutoUpdateKey::PREFETCH_CTX) + (i << 1));
    FE_ASSERT_NOTNULL(ctx_id_vec);
    if (ctx_id_vec->GetSize() != (idx_num * flush_data->window_size)) {
      GELOGE(ge::FAILED, "Cmo index num[%zu] not equal with ctx id num[%zu].", idx_num, ctx_id_vec->GetSize());
      return ge::GRAPH_FAILED;
    }
    if (UpdateAutoCmoCtxProc(task_info, cmo_idx, ctx_id_vec, flush_data, i) != ge::GRAPH_SUCCESS) {
      return ge::GRAPH_FAILED;
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus StaAutoUpdateContext(KernelContext *context) {
  KLOGD("Update AICore static auto node begin.");
  auto flush_data = context->GetInputPointer<AICoreSubTaskFlush>(static_cast<size_t>(AutoUpdateKey::FLUSH_DATA));
  FE_ASSERT_NOTNULL(flush_data);
  auto task_info = context->GetInputValue<rtFftsPlusTaskInfo_t*>(static_cast<size_t>(AutoUpdateKey::TASK_INFO));
  FE_ASSERT_NOTNULL(task_info);
  FE_ASSERT_NOTNULL(task_info->descBuf);
  FE_ASSERT_NOTNULL(task_info->fftsPlusSqe);
  if (UpdateStaticAutoCmoProc(context, task_info, flush_data) != ge::GRAPH_SUCCESS) {
    KLOGE("Update data context failed.");
    return ge::GRAPH_FAILED;
  }
  auto ctx_ids = context->GetInputPointer<gert::ContinuousVector>(static_cast<size_t>(AutoUpdateKey::AICORE_CTX));
  FE_ASSERT_NOTNULL(ctx_ids);
  auto ctx_id_vec = reinterpret_cast<const int32_t*>(ctx_ids->GetData());
  const size_t ctx_num = ctx_ids->GetSize();
  uint64_t paraBase = reinterpret_cast<uintptr_t>(flush_data->args_base);
  auto *context_head = reinterpret_cast<rtFftsPlusComCtx_t*>(const_cast<void*>(task_info->descBuf));
  uint16_t total_num = task_info->fftsPlusSqe->totalContextNum;
  for (size_t idx = 0U; idx < ctx_num; ++idx) {
    if (ctx_id_vec[idx] >= total_num) {
      KLOGE("Context Id(%d) overflow.", ctx_id_vec[idx]);
      return ge::GRAPH_FAILED;
    }
    auto ctx = reinterpret_cast<rtFftsPlusAicAivCtx_t*>(context_head + ctx_id_vec[idx]);
    FE_ASSERT_NOTNULL(ctx);
    GELOGD("Update context id %d", ctx_id_vec[idx]);
    UpdateStaAicAivCtx(ctx, flush_data, paraBase);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FillStaAutoProfilingInfo(const KernelContext *context, ProfilingInfoWrapper &prof_info) {
  const auto block_dim = context->GetInputValue<uint32_t>(static_cast<size_t>(AutoUpdateKey::BLOCK_DIM));
  prof_info.SetBlockDim(block_dim);
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(StaAutoUpdateContext).RunFunc(StaAutoUpdateContext).ProfilingInfoFiller(FillStaAutoProfilingInfo);
}  // namespace
}  // namespace kernel
}  // namespace gert

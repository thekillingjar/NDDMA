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
#include "kernel/memory/ffts_mem_allocator.h"
#include "kernel/common_kernel_impl/calc_tenorsize_from_shape.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "engine/aicore/kernel/rt_ffts_plus_launch_args.h"
#include "common/dump/kernel_tracing_utils.h"
#include "exe_graph/runtime/tiling_data.h"
#include "register/kernel_registry_impl.h"
#include "runtime/rt_ffts_plus_define.h"
#include "runtime/rt_ffts_plus.h"
#include "runtime/mem.h"
#include "register/op_tiling.h"
#include "common/math/math_util.h"
#include "exe_graph/runtime/tensor.h"
#include "engine/ffts_plus/converter/ffts_plus_proto_transfer.h"
#include "exe_graph/lowering/shape_utils.h"
namespace gert {
namespace kernel {
namespace {
ge::graphStatus AtomicUpdateContext(KernelContext *context) {
  GELOGD("Update atomic node begin.");
  auto flush_data = context->GetInputValue<AICoreSubTaskFlush*>(static_cast<size_t>(AtomUpdateKey::FLUSH_DATA));
  FE_ASSERT_NOTNULL(flush_data);
  auto ctx_ids = context->GetInputPointer<gert::ContinuousVector>(static_cast<size_t>(AtomUpdateKey::AICORE_CTX));
  FE_ASSERT_NOTNULL(ctx_ids);
  flush_data->blk_dim = context->GetInputValue<uint32_t>(static_cast<size_t>(AtomUpdateKey::BLOCK_DIM));
  flush_data->tail_blk_dim = context->GetInputValue<uint32_t>(static_cast<size_t>(AtomUpdateKey::TAIL_BLOCK_DIM));
  auto task_info = context->GetInputValue<rtFftsPlusTaskInfo_t*>(static_cast<size_t>(AtomUpdateKey::TASK_INFO));
  FE_ASSERT_NOTNULL(task_info);
  FE_ASSERT_NOTNULL(task_info->descBuf);
  FE_ASSERT_NOTNULL(task_info->fftsPlusSqe);
  auto *context_head = reinterpret_cast<rtFftsPlusComCtx_t*>(const_cast<void*>(task_info->descBuf));
  auto ctx_id_vec = reinterpret_cast<const int32_t*>(ctx_ids->GetData());
  const size_t ctx_num = ctx_ids->GetSize();
  uint64_t paraBase = reinterpret_cast<uintptr_t>(flush_data->args_base);
  KLOGD("Update %zu ctx Base:0x%lx, block dim:%d/%d, thread dim:%u.", ctx_num, paraBase, flush_data->blk_dim,
        flush_data->tail_blk_dim, flush_data->thread_dim);
  uint16_t total_num = task_info->fftsPlusSqe->totalContextNum;
  for (size_t idx = 0U; idx < ctx_num; ++idx) {
    if (ctx_id_vec[idx] >= total_num) {
      KLOGE("Atomic context Id(%d) overflow.", ctx_id_vec[idx]);
      return ge::GRAPH_FAILED;
    }
    auto ctx = reinterpret_cast<rtFftsPlusAicAivCtx_t*>(context_head + ctx_id_vec[idx]);
    if (ctx == nullptr) {
      KLOGE("Atomic context[%d] is nullptr.", ctx_id_vec[idx]);
      return ge::GRAPH_FAILED;
    }
    GELOGD("Update context id [%d].", ctx_id_vec[idx]);
    if (flush_data->atom_proc_type != static_cast<uint32_t>(AtomProcType::STATIC_OP)) {
      UpdateDyAicAivCtx(ctx, flush_data, paraBase);
    } else {
      UpdateStaAicAivCtx(ctx, flush_data, paraBase);
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FillAtomicProfilingInfo(const KernelContext *context, ProfilingInfoWrapper &prof_info) {
  const auto block_dim = context->GetInputValue<uint32_t>(static_cast<size_t>(AtomUpdateKey::BLOCK_DIM));
  prof_info.SetBlockDim(block_dim);
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(AtomicUpdateContext).RunFunc(AtomicUpdateContext).ProfilingInfoFiller(FillAtomicProfilingInfo);

ge::Status FFTSCalcAtomicOutputShapeSize(KernelContext *context) {
  auto extend_context = reinterpret_cast<ExtendedKernelContext *>(context);
  auto compute_node_info = extend_context->GetComputeNodeInfo();
  FE_ASSERT_NOTNULL(compute_node_info);
  auto output_clean_index = context->GetInputPointer<gert::ContinuousVector>(0U);
  FE_ASSERT_NOTNULL(output_clean_index);
  auto slice_shape = context->GetInputPointer<gert::TypedContinuousVector<size_t>>(1U);
  FE_ASSERT_NOTNULL(slice_shape);
  size_t out_size = slice_shape->GetSize();
  size_t out_clean_size = output_clean_index->GetSize();
  const auto out_clean_vec = reinterpret_cast<const int64_t *>(output_clean_index->GetData());
  const auto slice_vec = reinterpret_cast<const Shape *>(slice_shape->GetData());
  for (size_t i = 0; i < out_clean_size; ++i) {
    auto output_index = out_clean_vec[i];
    if (static_cast<size_t>(output_index) >= out_size) {
      KLOGE("Output index (%ld) is over then slice size(%zu).", output_index, out_size);
      return ge::GRAPH_FAILED;
    }
    auto tensor = compute_node_info->GetOutputTdInfo(output_index);
    FE_ASSERT_NOTNULL(tensor);
    auto tensor_size_ptr = context->GetOutputPointer<uint64_t>(i);
    FE_ASSERT_NOTNULL(tensor_size_ptr);
    CalcAlignedSizeByShape(slice_vec[output_index], tensor->GetDataType(), *tensor_size_ptr);
    KLOGD("Out index[%ld] has size[%lu] with type[%d].", output_index, *tensor_size_ptr, tensor->GetDataType());
  }
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(FFTSCalcAtomicOutputShapeSize).RunFunc(FFTSCalcAtomicOutputShapeSize);

ge::graphStatus GetAtomFlushDataVal(const KernelContext *context, AICoreSubTaskFlush* flush_data) {
  auto thread_dim = context->GetInputValue<uint32_t>(static_cast<size_t>(AtomArgsInKey::THREAD_DIM));
  auto window_size = context->GetInputValue<uint32_t>(static_cast<size_t>(AtomArgsInKey::WINDOW_SIZE));
  auto proc_type = context->GetInputValue<uint32_t>(static_cast<size_t>(AtomArgsInKey::PROC_TYPE));
  auto sink_ret = context->GetInputPointer<AICoreSinkRet>(static_cast<size_t>(AtomArgsInKey::SINK_RET));
  FE_ASSERT_NOTNULL(sink_ret);
  if (thread_dim > kFFTSMaxThreadNum) {
    GELOGE(ge::FAILED, "[GetAtomFlushDataVal]Thread dim[%u] is invalid.", thread_dim);
    return ge::GRAPH_FAILED;
  }
  flush_data->thread_dim = thread_dim;
  flush_data->window_size = window_size;
  flush_data->atom_proc_type = proc_type;
  flush_data->aic_non_tail_task_start_pc = sink_ret->aic_non_tail_task_start_pc;
  flush_data->aic_tail_task_start_pc = sink_ret->aic_tail_task_start_pc;
  flush_data->aic_icache_prefetch_cnt = sink_ret->aic_icache_prefetch_cnt;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FFTSUpdateAtomicArgs(KernelContext *context) {
  auto workspace = context->GetInputPointer<ContinuousVector>(static_cast<size_t>(AtomArgsInKey::WORK_ADDR));
  FE_ASSERT_NOTNULL(workspace);
  auto work_index = context->GetInputPointer<ContinuousVector>(static_cast<size_t>(AtomArgsInKey::WORK_CLEAR_IDX));
  FE_ASSERT_NOTNULL(work_index);
  auto out_index = context->GetInputPointer<ContinuousVector>(static_cast<size_t>(AtomArgsInKey::OUT_CLEAR_IDX));
  FE_ASSERT_NOTNULL(out_index);
  auto out_mem_type = context->GetInputPointer<ContinuousVector>(static_cast<size_t>(AtomArgsInKey::OUT_MEM_TYPE));
  FE_ASSERT_NOTNULL(out_mem_type);
  auto args_para = context->GetInputValue<NodeMemPara*>(static_cast<size_t>(AtomArgsInKey::ARGS_PARA));
  FE_ASSERT_NOTNULL(args_para);
  FE_ASSERT_NOTNULL(args_para->host_addr);
  FE_ASSERT_NOTNULL(args_para->dev_addr);
  auto thread_para = context->GetInputPointer<AICoreThreadParam>(static_cast<size_t>(AtomArgsInKey::THREAD_PARAM));
  FE_ASSERT_NOTNULL(thread_para);
  auto flush_data = context->GetOutputPointer<AICoreSubTaskFlush>(static_cast<size_t>(0));
  FE_ASSERT_NOTNULL(flush_data);
  if (GetAtomFlushDataVal(context, flush_data) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  size_t clear_work_size = work_index->GetSize();
  auto clear_work_index_v = reinterpret_cast<const int64_t*>(work_index->GetData());
  size_t clear_out_size = out_index->GetSize();
  auto out_index_vec = reinterpret_cast<const int64_t*>(out_index->GetData());
  size_t mem_type_size = out_mem_type->GetSize();
  auto mem_type_vec = reinterpret_cast<const uint32_t*>(out_mem_type->GetData());

  flush_data->param_ptr_offset = clear_work_size + clear_out_size;
  if (flush_data->atom_proc_type != static_cast<uint32_t>(AtomProcType::STATIC_OP)) {
    // dynamic need tiling addr
    flush_data->param_ptr_offset++;
  }
  GELOGD("Update atomic args cleared; out num: %zu, work num: %zu.", clear_out_size, clear_work_size);
  auto rt_args = reinterpret_cast<RtFFTSKernelLaunchArgs *>(args_para->host_addr);
  size_t real_size = flush_data->param_ptr_offset * sizeof(uintptr_t);
  FE_ASSERT_TRUE(real_size <= rt_args->GetArgsCap(RtFFTSKernelLaunchArgs::kAtomArgsHostAddr));
  uintptr_t *args_base_addr = static_cast<uintptr_t*>(rt_args->GetArgBase());
  FE_ASSERT_NOTNULL(args_base_addr);
  size_t args_pos = rt_args->GetAtomArgsPos();
  uintptr_t *args_host_data = &args_base_addr[args_pos];
  size_t args_abs_pos = rt_args->GetAtomArgsAbsPos();
  flush_data->args_base = static_cast<void*>(&(static_cast<uint8_t*>(args_para->dev_addr)[args_abs_pos]));
  GELOGD("Args: host pos[%zu], absolute pos[%zu], rtArgs: base[%lx], dev args addr[%lx], proc_type: %u.", args_pos,
         args_abs_pos, args_para->host_addr, flush_data->args_base, flush_data->atom_proc_type);
  // only output
  for (size_t i = 0U; i < clear_out_size; ++i) {
    if (static_cast<size_t>(out_index_vec[i]) >= mem_type_size) {
      return ge::GRAPH_FAILED;
    }
    // mem pool type after input and output
    if (mem_type_vec[out_index_vec[i]] == 0) {
      auto tensor_data =
        context->GetInputValue<gert::GertTensorData *>(static_cast<size_t>(AtomArgsInKey::IO_START) + i);
      GE_CHECK_NOTNULL(tensor_data);
      FE_ASSERT_TRUE(InitCtxIoAddrs(i, tensor_data, thread_para, flush_data, args_host_data));
    } else {
      auto ffts_mem =
        context->GetInputValue<memory::FftsMemBlock *>(static_cast<size_t>(AtomArgsInKey::IO_START) + i);
      GE_CHECK_NOTNULL(ffts_mem);
      FE_ASSERT_TRUE(InitCtxIoAddrs(i, ffts_mem, thread_para, flush_data, args_host_data));
    }
  }
  size_t io_index = clear_out_size;
  // workspace
  size_t tiling_work_size = workspace->GetSize();
  auto addrs_data = reinterpret_cast<void *const *>(workspace->GetData());
  for (size_t i = 0; i < clear_work_size; ++i) {
    size_t clear_work_idx = clear_work_index_v[i];
    if (clear_work_idx >= tiling_work_size) {
      GELOGE(ge::FAILED, "Clean work index[%zu] over size[%zu].", clear_work_idx, tiling_work_size);
      return ge::GRAPH_FAILED;
    }
    if (flush_data->atom_proc_type == static_cast<uint32_t>(AtomProcType::DY_SLICE_OP)) {
      InitCtxIoAddrs(io_index, reinterpret_cast<memory::FftsMemBlock *>(addrs_data[clear_work_idx]), thread_para,
                     flush_data, args_host_data);
    } else {
      InitL1WorkAddrs(io_index, reinterpret_cast<GertTensorData *>(addrs_data[clear_work_idx])->GetAddr(),
                      flush_data, args_host_data);
    }
    io_index++;
  }
  if (flush_data->atom_proc_type == static_cast<uint32_t>(AtomProcType::STATIC_OP)) {
    return ge::GRAPH_SUCCESS;
  }
  // tiling data address
  size_t tiling_pos = rt_args->GetAtomTilingAbsPos();
  size_t tiling_offset = rt_args->GetAtomTilingOffset();
  auto tiling_base = static_cast<void*>(&(static_cast<uint8_t*>(args_para->dev_addr)[tiling_pos]));
  InitOpTiling(io_index, flush_data, args_host_data, tiling_offset, tiling_base);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CreateAtomicFlushData(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  auto task_flush = context->GetOutput(0);
  FE_ASSERT_NOTNULL(task_flush);
  task_flush->SetWithDefaultDeleter(new (std::nothrow) AICoreSubTaskFlush());
  return ge::GRAPH_SUCCESS;
}

void PrintAtomArgs(const KernelContext *context, std::vector<std::string> &args_str) {
  auto args_para = context->GetInputValue<NodeMemPara*>(static_cast<size_t>(AtomArgsInKey::ARGS_PARA));
  if (args_para == nullptr) {
    args_str.emplace_back("Launch args pointer is null.");
    return;
  }
  auto proc_type = context->GetInputValue<uint32_t>(static_cast<size_t>(AtomArgsInKey::PROC_TYPE));
  auto rt_args = reinterpret_cast<RtFFTSKernelLaunchArgs *>(args_para->host_addr);
  std::stringstream ss;
  if (proc_type != static_cast<uint32_t>(AtomProcType::STATIC_OP)) {
    size_t tiling_pos = rt_args->GetAtomTilingAbsPos();
    auto print_base = reinterpret_cast<uintptr_t*>(&(static_cast<uint8_t*>(args_para->host_addr)[tiling_pos]));
    ss << "Tiling data: ";
    size_t print_size = (rt_args->GetAtomArgsAbsPos() - tiling_pos) / sizeof(TensorAddress);
    PrintHex(print_base, print_size, ss);
    args_str.emplace_back(ss.str());
    ss.str("");
  }
  ss << "All args: ";
  auto args_pos = rt_args->GetAtomArgsAbsPos();
  auto print_base = reinterpret_cast<uintptr_t*>(&(static_cast<uint8_t*>(args_para->host_addr)[args_pos]));
  size_t print_size = rt_args->GetArgsCap(RtFFTSKernelLaunchArgs::kAtomArgsHostAddr) / sizeof(TensorAddress);
  PrintHex(print_base, print_size, ss);
  args_str.emplace_back(ss.str());
  return;
}

std::vector<std::string> PrintAtomicArgs(const KernelContext *context) {
  std::stringstream ss;
  ss << "FFTS Plus launch AICore args: ";
  std::vector<std::string> msgs;
  msgs.emplace_back(ss.str());
  std::vector<std::string> args_str;
  PrintAtomArgs(context, args_str);
  msgs.insert(msgs.cend(), args_str.cbegin(), args_str.cend());
  return msgs;
}
REGISTER_KERNEL(FFTSUpdateAtomicArgs).RunFunc(FFTSUpdateAtomicArgs).OutputsCreator(CreateAtomicFlushData).
    TracePrinter(PrintAtomicArgs);
}  // namespace
}  // namespace kernel
}  // namespace gert

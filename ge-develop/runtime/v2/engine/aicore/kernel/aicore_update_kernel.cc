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
#include "engine/node_converter_utils.h"
#include "register/ffts_node_calculater_registry.h"
#include "engine/aicore/kernel/rt_ffts_plus_launch_args.h"
#include "common/dump/kernel_tracing_utils.h"
#include "exe_graph/runtime/tiling_data.h"
#include "register/kernel_registry_impl.h"
#include "register/op_tiling.h"
#include "common/math/math_util.h"
#include "utils/extern_math_util.h"
#include "exe_graph/runtime/tensor.h"
#include "exe_graph/lowering/shape_utils.h"
#include "engine/ffts_plus/converter/ffts_plus_proto_transfer.h"
namespace gert {
namespace kernel {
namespace {
ge::graphStatus ReportRollbackError(KernelContext *context) {
  (void)context;
  KLOGE("AiCore tiling failed, and the rollback for aicpu also failed.");
  return ge::GRAPH_FAILED;
}
REGISTER_KERNEL(ReportRollbackError).RunFunc(ReportRollbackError);

ge::graphStatus ExpandDfxWorkspaceSize(KernelContext *context) {
  auto workspace = context->MutableInputPointer<gert::TypedContinuousVector<size_t>>(0U);
  FE_ASSERT_NOTNULL(workspace);
  auto buffer_size = context->GetInputValue<size_t>(1U);
  size_t work_size = workspace->GetSize();
  if (work_size == 0 || buffer_size == 0) {
    KLOGW("Work/Buffer size %zu/%ld is zero.", work_size, buffer_size);
    return ge::GRAPH_SUCCESS;
  }
  auto work_vec = reinterpret_cast<size_t *>(workspace->MutableData());
  KLOGD("Node expands dfx workspace size to %zu with %ld.", work_vec[0], buffer_size);
  FE_ASSERT_TRUE(!ge::AddOverflow(work_vec[0], buffer_size, work_vec[0]));
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(ExpandDfxWorkspaceSize).RunFunc(ExpandDfxWorkspaceSize);
}  // namespace

bool InitCtxIoAddrs(size_t index, const gert::GertTensorData *tensor_data, const AICoreThreadParam *task_param,
                    AICoreSubTaskFlush *flush_data, uintptr_t *args_host_data) {
  FE_ASSERT_NOTNULL(tensor_data->GetAddr());
  const uintptr_t data_base = ge::PtrToValue(tensor_data->GetAddr());
  size_t tensor_size = tensor_data->GetSize();
  size_t io_index = flush_data->need_mode_addr ? (index - 1) : index;
  for (uint32_t i = 0U; i < flush_data->thread_dim; ++i) {
    const size_t ctx_io_idx = (flush_data->param_ptr_offset * i) + index;
    GELOGD("Addr base:0x%lx, io index:%zu, thread:%u, thread_offset:%lu, args index:%zu, tensor_size:%zu.", data_base,
           io_index, i, task_param->offset_vec[io_index], ctx_io_idx, tensor_size);
    uint64_t offset = task_param->offset_vec[io_index] * i;
    FE_ASSERT_TRUE(offset < tensor_size);
    args_host_data[ctx_io_idx] = (data_base + offset);
    if (io_index < task_param->input_num) {
      flush_data->input_addr_vv[io_index][i] = data_base;
      flush_data->input_offset[io_index] = task_param->offset_vec[io_index];
    } else if (io_index < task_param->input_output_num) {
      flush_data->output_addr_vv[io_index - task_param->input_num][i] = data_base;
      flush_data->output_offset[io_index] = task_param->offset_vec[io_index];
    }
  }
  return true;
}

bool InitCtxIoAddrs(const size_t index, const memory::FftsMemBlock *ffts_mem, const AICoreThreadParam *task_param,
                    AICoreSubTaskFlush *flush_data, uintptr_t *args_host_data) {
  for (uint32_t i = 0U; i < flush_data->thread_dim; ++i) {
    const size_t ctx_io_idx = (flush_data->param_ptr_offset * i) + index;
    FE_ASSERT_NOTNULL(ffts_mem->Addr(i));
    args_host_data[ctx_io_idx] = ge::PtrToValue(ffts_mem->Addr(i));  // l2 pool get thread [i]'s addr
    size_t io_index = flush_data->need_mode_addr ? (index - 1) : index;
    GELOGD("Addr base: 0x%lx, addr index: %zu, thread index: %u, args index: %zu", args_host_data[ctx_io_idx], io_index,
           i, ctx_io_idx);
    if (io_index < task_param->input_num) {
      flush_data->input_addr_vv[io_index][i] = args_host_data[ctx_io_idx];
    } else if (io_index < task_param->input_output_num) {
      flush_data->output_addr_vv[io_index - task_param->input_num][i] = args_host_data[ctx_io_idx];
    }
  }
  return true;
}

void InitL1WorkAddrs(const size_t index, const void *addr, AICoreSubTaskFlush *flush_data, uintptr_t *args_host_data) {
  for (uint32_t i = 0U; i < flush_data->thread_dim; ++i) {
    const size_t ctx_io_idx = (flush_data->param_ptr_offset * i) + index;
    args_host_data[ctx_io_idx] = ge::PtrToValue(addr);
    GELOGD("Addr base: 0x%lx, addr index: %zu, thread index: %u, args index: %zu", args_host_data[ctx_io_idx], index, i,
           ctx_io_idx);
  }
}

void InitOpTiling(const size_t index, const AICoreSubTaskFlush *flush_data, uintptr_t *args_host_data,
                  size_t tiling_offset, const void *args_addr) {
  const uintptr_t data_base = ge::PtrToValue(args_addr);
  for (uint32_t i = 0U; i < flush_data->thread_dim; ++i) {
    const auto data_pos = (i < (flush_data->thread_dim - 1U)) ? 0U : tiling_offset;
    GELOGD("Addr base: 0x%lx, addr index: %zu, thread index: %u, thread addr offset: 0x%lx, args index: %zu", data_base,
           index, i, data_pos, (flush_data->param_ptr_offset * i) + index);
    const size_t ctx_io_idx = (flush_data->param_ptr_offset * i) + index;
    args_host_data[ctx_io_idx] = (data_base + data_pos);
  }
}

}  // namespace kernel
}  // namespace gert

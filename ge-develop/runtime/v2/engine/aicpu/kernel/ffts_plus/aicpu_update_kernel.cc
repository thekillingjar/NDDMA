/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_update_kernel.h"
#include "exe_graph/runtime/shape.h"
#include "graph/types.h"
#include "runtime/rt_ffts_plus.h"
#include "runtime/rt_ffts_plus_define.h"
#include "register/kernel_registry_impl.h"
#include "engine/ffts_plus/converter/ffts_plus_proto_transfer.h"
#include "graph/ge_error_codes.h"
#include "graph/types.h"
#include "common/checker.h"
#include "exe_graph/runtime/gert_tensor_data.h"

namespace {
constexpr size_t kArgsAddrLIndex = 2UL;
constexpr size_t kArgsAddrHIndex = 3UL;
constexpr uint32_t k32BitsMask = 0xFFFFFFFFU;
}  // namespace

namespace gert {
namespace kernel {
ge::graphStatus AicpuCalcOutputMaxThreadSize(gert::KernelContext *context) {
  auto extend_context = reinterpret_cast<ExtendedKernelContext *>(context);
  auto compute_node_info = extend_context->GetComputeNodeInfo();
  size_t output_id = context->GetInputValue<size_t>(static_cast<size_t>(MaxThreadSizeInputIndex::kOutputId));
  auto not_tail_slice_out =
      context->GetInputPointer<gert::ContinuousVector>(static_cast<size_t>(MaxThreadSizeInputIndex::kNotLastOutSlice));
  auto tail_slice_out =
      context->GetInputPointer<gert::ContinuousVector>(static_cast<size_t>(MaxThreadSizeInputIndex::kLastOutSlice));
  auto max_thread_size = context->GetOutputPointer<uint64_t>(static_cast<size_t>(0));

  GE_ASSERT_NOTNULL(compute_node_info);
  GE_ASSERT_NOTNULL(not_tail_slice_out);
  GE_ASSERT_NOTNULL(tail_slice_out);
  GE_ASSERT_NOTNULL(max_thread_size);

  size_t tail_slice_out_size = tail_slice_out->GetSize();
  GE_ASSERT_EQ(not_tail_slice_out->GetSize(), tail_slice_out_size);

  auto not_tail_slice_out_vec = reinterpret_cast<const Shape *>(not_tail_slice_out->GetData());
  auto tail_slice_out_vec = reinterpret_cast<const Shape *>(tail_slice_out->GetData());

  auto tensor = compute_node_info->GetOutputTdInfo(output_id);
  GE_ASSERT_NOTNULL(tensor);
  GE_ASSERT_TRUE(tail_slice_out_size > output_id, "output_id[%zu] must be less than tail_slice_out_size[%zu]",
                 output_id, tail_slice_out_size);
  int64_t tail_size = 1;
  const size_t tail_dim_num = tail_slice_out_vec[output_id].GetDimNum();
  for (size_t i = 0; i < tail_dim_num; ++i) {
    if (ge::MulOverflow(tail_size, tail_slice_out_vec[output_id][i], tail_size)) {
      GELOGE(ge::FAILED, "[Kernel]tail_size[%ld] calculate err with range:%lu.", tail_size,
             tail_slice_out_vec[output_id][i]);
      return ge::GRAPH_FAILED;
    }
  }
  int64_t not_tail_size = 1;
  const size_t not_tail_dim_num = not_tail_slice_out_vec[output_id].GetDimNum();
  for (size_t j = 0; j < not_tail_dim_num; ++j) {
    if (ge::MulOverflow(not_tail_size, not_tail_slice_out_vec[output_id][j], not_tail_size)) {
      GELOGE(ge::FAILED, "[Kernel]not_tail_size[%ld] calculate err with range:%lu.", not_tail_size,
             not_tail_slice_out_vec[output_id][j]);
      return ge::GRAPH_FAILED;
    }
  }
  auto max_size = (tail_size > not_tail_size) ? tail_size : not_tail_size;
  auto max_byte_size = ge::GetSizeInBytes(max_size, tensor->GetDataType());
  *max_thread_size = static_cast<uint64_t>(max_byte_size);
  GELOGD("Output anchor index: %d, max_thread_size: %lu", output_id, *max_thread_size);
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(AicpuCalcOutputMaxThreadSize).RunFunc(AicpuCalcOutputMaxThreadSize);

ge::graphStatus CalcInputPara(gert::KernelContext *context, const ComputeNodeInfo *compute_node_info,
                              AICpuThreadParam *args_para, size_t &count) {
  auto input_indexes =
      context->GetInputPointer<gert::ContinuousVector>(static_cast<size_t>(ThreadParamInputIndex::kInputIndexes));
  GE_ASSERT_NOTNULL(input_indexes);
  auto input_indexes_vec = reinterpret_cast<const uint32_t *>(input_indexes->GetData());
  auto slice_in =
      context->GetInputPointer<gert::ContinuousVector>(static_cast<size_t>(ThreadParamInputIndex::kNotLastInSlice));
  GE_ASSERT_NOTNULL(slice_in);
  size_t input_num = compute_node_info->GetInputsNum();
  GE_ASSERT_EQ(input_num, slice_in->GetSize());
  auto slice_in_vec = reinterpret_cast<const Shape*>(slice_in->GetData());
  const size_t idx_num = input_indexes->GetSize();
  GELOGI("Cut input index num:%zu, input num:%zu", idx_num, input_num);
  for (size_t i = 0; i < idx_num; ++i) {
    uint32_t index = input_indexes_vec[i];
    auto tensor = compute_node_info->GetInputTdInfo(index);
    GE_ASSERT_NOTNULL(tensor);
    int64_t thread_offset = 1;
    const size_t dim_num = slice_in_vec[i].GetDimNum();
    for (size_t j = 0; j < dim_num; ++j) {
      if (ge::MulOverflow(thread_offset, slice_in_vec[i][j], thread_offset)) {
        GELOGE(ge::FAILED, "[Kernel]Offset[%ld] calculate err with range:%lu.", thread_offset, slice_in_vec[i][j]);
        return ge::GRAPH_FAILED;
      }
    }
    thread_offset = ge::GetSizeInBytes(thread_offset, tensor->GetDataType());
    GELOGI("Input anchorIndex: %u, thread_offset: %ld.", index, thread_offset);
    GE_ASSERT_TRUE(index < MAX_OFFSET_NUM, "index[%zu] must less than MAX_OFFSET_NUM[%zu]", index, MAX_OFFSET_NUM);
    args_para->task_addr_offset[index] = static_cast<uint64_t>(thread_offset);
  }
  count = input_num;
  args_para->input_num = input_num;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CalcOutputPara(gert::KernelContext *context, const ComputeNodeInfo *compute_node_info,
                               AICpuThreadParam *args_para, size_t &count) {
  auto output_indexes =
      context->GetInputPointer<gert::ContinuousVector>(static_cast<size_t>(ThreadParamInputIndex::kOutputIndexes));
  GE_ASSERT_NOTNULL(output_indexes);
  auto output_indexes_vec = reinterpret_cast<const uint32_t *>(output_indexes->GetData());
  auto slice_out =
      context->GetInputPointer<gert::ContinuousVector>(static_cast<size_t>(ThreadParamInputIndex::kNotLastOutSlice));
  GE_ASSERT_NOTNULL(slice_out);
  size_t output_num = compute_node_info->GetOutputsNum();
  GE_ASSERT_EQ(output_num, slice_out->GetSize());
  auto slice_out_vec = reinterpret_cast<const Shape*>(slice_out->GetData());
  const size_t idx_num = output_indexes->GetSize();
  GELOGI("Cut output index num:%zu, out num:%zu", idx_num, output_num);
  for (size_t i = 0; i < idx_num; ++i) {
    uint32_t index = output_indexes_vec[i];
    auto tensor = compute_node_info->GetOutputTdInfo(index);
    GE_ASSERT_NOTNULL(tensor);
    int64_t thread_offset = 1;
    const size_t dim_num = slice_out_vec[i].GetDimNum();
    for (size_t j = 0; j < dim_num; ++j) {
      if (ge::MulOverflow(thread_offset, slice_out_vec[i][j], thread_offset)) {
        GELOGE(ge::FAILED, "[Kernel]Offset[%ld] calculate err with range:%lu.", thread_offset, slice_out_vec[i][j]);
        return ge::GRAPH_FAILED;
      }
    }
    thread_offset = ge::GetSizeInBytes(thread_offset, tensor->GetDataType());
    GELOGI("Output anchor index: %d, thread_offset: %lu.", index, thread_offset);
    size_t offset_index = count + index;
    GE_ASSERT_TRUE(offset_index < MAX_OFFSET_NUM, "offset_index[%zu] must be less than MAX_OFFSET_NUM[%zu]",
                   offset_index, MAX_OFFSET_NUM);
    args_para->task_addr_offset[offset_index] = static_cast<uint64_t>(thread_offset);
  }
  count += output_num;
  args_para->output_num = output_num;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus AICpuGetAutoThreadParam(gert::KernelContext *context) {
  GELOGD("AICpuGetAutoThreadParam begin.");
  auto extend_context = reinterpret_cast<ExtendedKernelContext *>(context);
  auto compute_node_info = extend_context->GetComputeNodeInfo();
  auto thread_dim = context->GetInputValue<uint32_t>(static_cast<size_t>(ThreadParamInputIndex::kCutDim));
  auto auto_thread_param = context->GetOutputPointer<AICpuThreadParam>(static_cast<size_t>(0));

  GE_ASSERT_NOTNULL(compute_node_info);
  GE_ASSERT_NOTNULL(auto_thread_param);

  if (thread_dim == 0) {
    GELOGE(ge::PARAM_INVALID, "Thread dim is zero.");
  }
  size_t count = 0;
  if (CalcInputPara(context, compute_node_info, auto_thread_param, count) != ge::GRAPH_SUCCESS) {
    GELOGE(ge::FAILED, "Calculate thread input offset failed.");
    return ge::GRAPH_FAILED;
  }
  if (CalcOutputPara(context, compute_node_info, auto_thread_param, count) != ge::GRAPH_SUCCESS) {
    GELOGE(ge::FAILED, "Calculate thread output offset failed.");
    return ge::GRAPH_FAILED;
  }

  auto_thread_param->input_output_num = count;
  auto_thread_param->thread_dim = thread_dim;
  GELOGD("[Kernel] AICpuGetAutoThreadParam Success.");
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus AicpuCreateThreadParam(const ge::FastNode *node, gert::KernelContext *context) {
  (void)node;
  auto thread_param = new (std::nothrow) AICpuThreadParam();
  GE_ASSERT_NOTNULL(thread_param);
  context->GetOutput(0)->SetWithDefaultDeleter(thread_param);
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(AICpuGetAutoThreadParam).RunFunc(AICpuGetAutoThreadParam).OutputsCreator(AicpuCreateThreadParam);

ge::graphStatus AICpuUpdateContext(gert::KernelContext *context) {
  GELOGD("AICpuUpdateContext begin.");
  auto flush_data =
      context->GetInputPointer<AICpuSubTaskFlush>(static_cast<size_t>(UpdateContextInputIndex::kFlushData));
  auto ctx_ids =
      context->GetInputPointer<gert::ContinuousVector>(static_cast<size_t>(UpdateContextInputIndex::kCtxIds));
  auto thread_dim = context->GetInputValue<uint32_t>(static_cast<size_t>(UpdateContextInputIndex::kThreadDim));
  auto task_info = context->GetOutputPointer<rtFftsPlusTaskInfo_t>(static_cast<int32_t>(0));

  GE_ASSERT_NOTNULL(flush_data);
  GE_ASSERT_NOTNULL(task_info);
  GE_ASSERT_NOTNULL(task_info->descBuf);
  GE_ASSERT_NOTNULL(task_info->fftsPlusSqe);
  GE_ASSERT_NOTNULL(ctx_ids);
  if (thread_dim == 0) {
    GELOGE(ge::PARAM_INVALID, "Thread dim is zero.");
  }

  rtFftsPlusAiCpuCtx_t *ctx_head =
      reinterpret_cast<rtFftsPlusAiCpuCtx_t *>(const_cast<void *>(task_info->descBuf));
  auto ctx_id_vec = reinterpret_cast<const int32_t*>(ctx_ids->GetData());
  const size_t ctx_num = ctx_ids->GetSize();
  const uint64_t address_offset = 32UL;
  uint16_t total_num = task_info->fftsPlusSqe->totalContextNum;
  auto parse_base = reinterpret_cast<uintptr_t>(flush_data->args_base_addr_dev);
  for (size_t idx = 0; idx < ctx_num; ++idx) {
    if (ctx_id_vec[idx] > total_num) {
      GELOGE(ge::FAILED, "Context Id(%d) over flow.", ctx_id_vec[idx]);
      return ge::GRAPH_FAILED;
    }
    GELOGI("ctx_num is %d, ctx[%d] is %d", ctx_num, idx, ctx_id_vec[idx]);
    auto ctx = reinterpret_cast<rtFftsPlusAiCpuCtx_t *>(ctx_head + ctx_id_vec[idx]);
    ctx->threadDim = static_cast<uint16_t>(thread_dim);
    ctx->usrData[kArgsAddrLIndex] = static_cast<uint32_t>(parse_base & k32BitsMask);
    ctx->usrData[kArgsAddrHIndex] = static_cast<uint32_t>(static_cast<uint64_t>(parse_base) >> address_offset);
  }
  GELOGD("AICpuUpdateContext Success.");
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(AICpuUpdateContext).RunFunc(AICpuUpdateContext);
}  // namespace kernel
}  // namespace gert

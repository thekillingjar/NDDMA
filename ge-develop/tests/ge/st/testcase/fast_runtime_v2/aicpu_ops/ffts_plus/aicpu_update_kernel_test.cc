/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "faker/kernel_run_context_facker.h"
#include "common/math/math_util.h"
#include "faker/aicpu_ext_info_faker.h"
#include "register/kernel_registry.h"
#include "common/plugin/ge_make_unique_util.h"
#include <kernel/memory/mem_block.h>
#include "aicpu_task_struct.h"
#include "engine/aicpu/kernel/ffts_plus/aicpu_ffts_args.h"
#include "engine/aicpu/kernel/ffts_plus/aicpu_update_kernel.h"
#include "engine/aicpu/converter/aicpu_ffts_node_converter.h"
#include "engine/ffts_plus/converter/ffts_plus_proto_transfer.h"
#include "kernel/memory/single_stream_l2_allocator.h"
#include "exe_graph/runtime/gert_tensor_data.h"

namespace gert {
using namespace kernel;
namespace {
struct AicpuTaskStruct {
  aicpu::AicpuParamHead head;
  uint64_t io_addrp[6];
}__attribute__((packed));
} // namespace

class AicpuFFTSUpdateKernelST : public testing::Test {
 public:
  KernelRegistry &registry = KernelRegistry::GetInstance();
};

TEST_F(AicpuFFTSUpdateKernelST, test_ffts_aicpu_cal_thread_param) {
  uint32_t thread_dim = 2;
  Shape shape({4, 2, 3});

  auto in_index = ContinuousVector::Create<uint32_t>(3);
  auto in_index_vec = reinterpret_cast<ContinuousVector *>(in_index.get());
  in_index_vec->SetSize(2);
  auto in_index_ptr = reinterpret_cast<uint32_t *>(in_index_vec->MutableData());
  for (size_t i = 0; i < in_index_vec->GetSize(); i++) {
    in_index_ptr[i] = i;
  }

  auto out_index = ContinuousVector::Create<uint32_t>(3);
  auto out_index_vec = reinterpret_cast<ContinuousVector *>(out_index.get());
  out_index_vec->SetSize(2);
  auto out_index_ptr = reinterpret_cast<uint32_t *>(out_index_vec->MutableData());
  for (size_t i = 0; i < out_index_vec->GetSize(); i++) {
    out_index_ptr[i] = i;
  }

  auto in_slice = ContinuousVector::Create<Shape>(65);
  auto in_slice_vec = reinterpret_cast<ContinuousVector *>(in_slice.get());
  in_slice_vec->SetSize(2);
  auto in_slice_ptr = reinterpret_cast<Shape *>(in_slice_vec->MutableData());
  for (size_t i = 0; i < in_slice_vec->GetSize(); i++) {
    in_slice_ptr[i] = shape;
  }

  auto out_slice = ContinuousVector::Create<Shape>(65);
  auto out_slice_vec = reinterpret_cast<ContinuousVector *>(out_slice.get());
  out_slice_vec->SetSize(2);
  auto out_slice_ptr = reinterpret_cast<Shape *>(out_slice_vec->MutableData());
  for (size_t i = 0; i < out_slice_vec->GetSize(); i++) {
    out_slice_ptr[i] = shape;
  }
  
  auto run_context = KernelRunContextFaker().KernelIONum(5, 1).NodeIoNum(2,2).IrInputNum(2)
      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .Build();
  run_context.value_holder[0].Set(reinterpret_cast<void *>(thread_dim), nullptr);
  run_context.value_holder[1].Set(in_index_vec, nullptr);
  run_context.value_holder[2].Set(out_index_vec, nullptr);
  run_context.value_holder[3].Set(in_slice_vec, nullptr);
  run_context.value_holder[4].Set(out_slice_vec, nullptr);

  AICpuThreadParam thread_param;
  run_context.value_holder[5].Set(&thread_param, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("AICpuGetAutoThreadParam")->outputs_creator(nullptr, run_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("AICpuGetAutoThreadParam")->run_func(run_context), ge::GRAPH_SUCCESS);
}

TEST_F(AicpuFFTSUpdateKernelST, test_ffts_aicpu_update_context) {
  AICpuSubTaskFlush flush_data;
  auto ctx_ids = ContinuousVector::Create<size_t>(65);
  auto ctx_ids_vec = reinterpret_cast<ContinuousVector *>(ctx_ids.get());
  ctx_ids_vec->SetSize(3);
  auto ctx_ids_ptr = reinterpret_cast<size_t *>(ctx_ids_vec->MutableData());
  for (size_t i = 0; i < ctx_ids_vec->GetSize(); i++) {
    ctx_ids_ptr[i] = i + 10;
  }
  uint32_t thread_dim = 2;

  auto run_context = BuildKernelRunContext(3, 1);
  run_context.value_holder[0].Set(&flush_data, nullptr);
  run_context.value_holder[1].Set(ctx_ids_vec, nullptr);
  run_context.value_holder[2].Set(reinterpret_cast<void *>(thread_dim), nullptr);

  size_t descBufLen = sizeof(rtFftsPlusComCtx_t) * static_cast<size_t>(16);
  size_t total_size = sizeof(TransTaskInfo) + descBufLen + sizeof(rtFftsPlusSqe_t) ;
  auto holder = ge::MakeUnique<uint8_t[]>(total_size);
  TransTaskInfo *task_info_ptr = reinterpret_cast<TransTaskInfo*>(holder.get());
  size_t buf_offset = sizeof(rtFftsPlusSqe_t);
  task_info_ptr->offsets[static_cast<size_t>(InfoStType::kDescBuf)] = buf_offset;
  task_info_ptr->rt_task_info.descBufLen = descBufLen;
  auto *buff_ptr = &task_info_ptr->args[buf_offset];
  for (int i = 0; i < 4; ++i) {
    buff_ptr += sizeof(rtFftsPlusComCtx_t);
    auto context = reinterpret_cast<rtFftsPlusAiCpuCtx_t*>(buff_ptr);
    context->contextType = RT_CTX_TYPE_AICPU;
  }
  for (int i = 0; i < 4; ++i) {
    buff_ptr += sizeof(rtFftsPlusComCtx_t);
    auto context = reinterpret_cast<rtFftsPlusDataCtx_t*>(buff_ptr);
    context->contextType = RT_CTX_TYPE_FLUSH_DATA;
  }
  for (int i = 0; i < 4; ++i) {
    buff_ptr += sizeof(rtFftsPlusComCtx_t);
    auto context = reinterpret_cast<rtFftsPlusDataCtx_t*>(buff_ptr);
    context->contextType = RT_CTX_TYPE_INVALIDATE_DATA;
  }
  for (int i = 0; i < 4; ++i) {
    buff_ptr += sizeof(rtFftsPlusComCtx_t);
    auto context = reinterpret_cast<rtFftsPlusDataCtx_t*>(buff_ptr);
    context->contextType = RT_CTX_TYPE_WRITEBACK_DATA;
  }

  auto *const ffts_plus_sqe = ge::PtrToPtr<uint8_t, rtFftsPlusSqe_t>(task_info_ptr->args);
  task_info_ptr->rt_task_info.fftsPlusSqe = ffts_plus_sqe;
  task_info_ptr->rt_task_info.descBuf = &task_info_ptr->args[buf_offset];
  ffts_plus_sqe->totalContextNum = 16;
  run_context.value_holder[3].Set(&task_info_ptr->rt_task_info, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("AICpuUpdateContext")->run_func(run_context), ge::GRAPH_SUCCESS);
}

TEST_F(AicpuFFTSUpdateKernelST, test_aicpu_cal_output_max_threadSize) {
  Shape shape({4, 2, 3});
  auto not_last_out_slice = ContinuousVector::Create<Shape>(65);
  auto not_last_out_slice_vec = reinterpret_cast<ContinuousVector *>(not_last_out_slice.get());
  not_last_out_slice_vec->SetSize(2);
  auto not_last_out_slice_ptr = reinterpret_cast<Shape *>(not_last_out_slice_vec->MutableData());
  for (size_t i = 0; i < not_last_out_slice_vec->GetSize(); i++) {
    not_last_out_slice_ptr[i] = shape;
  }

  Shape last_shape({1, 2, 3});
  auto last_out_slice = ContinuousVector::Create<Shape>(65);
  auto last_out_slice_vec = reinterpret_cast<ContinuousVector *>(last_out_slice.get());
  last_out_slice_vec->SetSize(2);
  auto last_out_slice_ptr = reinterpret_cast<Shape *>(last_out_slice_vec->MutableData());
  for (size_t i = 0; i < last_out_slice_vec->GetSize(); i++) {
    last_out_slice_ptr[i] = last_shape;
  }
  auto run_context = KernelRunContextFaker().KernelIONum(3, 1).NodeIoNum(2,2).IrInputNum(2)
      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .Build();
  size_t index = 1UL;
  run_context.value_holder[0].Set((void*)index, nullptr);
  run_context.value_holder[1].Set(not_last_out_slice_vec, nullptr);
  run_context.value_holder[2].Set(last_out_slice_vec, nullptr);

  ASSERT_EQ(registry.FindKernelFuncs("AicpuCalcOutputMaxThreadSize")->run_func(run_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(run_context.value_holder[3].GetValue<uint64_t>(), 48);
}

}  // namespace gert

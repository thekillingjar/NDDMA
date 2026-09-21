/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/ge_error_codes.h"
#include "register/kernel_registry_impl.h"
#include "common/checker.h"
#include "exe_graph/runtime/tensor.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "core/debug/kernel_tracing.h"
#include "kernel/common_kernel_impl/calc_tenorsize_from_shape.h"
#include "runtime/rt_ffts_plus.h"
#include "runtime/rt_ffts_plus_define.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "kernel/memory/ffts_mem_allocator.h"

namespace gert {
namespace kernel {
struct SdmaDataLenHolder {
  size_t tail_len;
  size_t non_tail_len;
};

enum class CalcLenKey { NON_TAIL_SLICE = 0, TAIL_SLICE = 1, DTYPE = 2, SLICE_FLAG = 3, THREAD_DUM = 4, RESERVED };

enum class SdmaUpdateKey {
  CTX_ID = 0,
  THREAD_DIM = 1,
  WINDOW_SIZE = 2,
  SDMA_LEN = 3,
  INPUT_MEM_TYPE = 4,
  OUTPUT_MEM_TYPE = 5,
  INPUT_TENSOR = 6,
  OUTPUT_TENSOR = 7,
  RESERVED
};

ge::graphStatus SdmaUpdateContext(KernelContext *const context) {
  GELOGD("SdmaUpdateContext begin.");
  auto ctx_ids = context->GetInputPointer<gert::ContinuousVector>(static_cast<size_t>(SdmaUpdateKey::CTX_ID));
  auto thread_dim = context->GetInputValue<uint32_t>(static_cast<size_t>(SdmaUpdateKey::THREAD_DIM));
  auto window_size = context->GetInputValue<uint32_t>(static_cast<size_t>(SdmaUpdateKey::WINDOW_SIZE));
  auto sdma_len = context->GetInputPointer<SdmaDataLenHolder>(static_cast<size_t>(SdmaUpdateKey::SDMA_LEN));
  uint32_t input_mem_type = context->GetInputValue<uint32_t>(static_cast<uint32_t>(SdmaUpdateKey::INPUT_MEM_TYPE));
  uint32_t output_mem_type = context->GetInputValue<uint32_t>(static_cast<uint32_t>(SdmaUpdateKey::OUTPUT_MEM_TYPE));

  auto task_info = context->GetOutputPointer<rtFftsPlusTaskInfo_t>(0UL);

  GE_ASSERT_NOTNULL(sdma_len);
  GE_ASSERT_NOTNULL(task_info);
  GE_ASSERT_NOTNULL(task_info->descBuf);
  GE_ASSERT_NOTNULL(task_info->fftsPlusSqe);
  GE_ASSERT_NOTNULL(ctx_ids);

  GE_ASSERT_TRUE(thread_dim != 0, "Thread dim val [0] is invalid");

  rtFftsPlusSdmaCtx_t *const ctx_head = reinterpret_cast<rtFftsPlusSdmaCtx_t *>(const_cast<void *>(task_info->descBuf));
  auto ctx_id_vec = reinterpret_cast<const int32_t *>(ctx_ids->GetData());
  const size_t ctx_num = ctx_ids->GetSize();
  uint16_t total_num = task_info->fftsPlusSqe->totalContextNum;
  GE_ASSERT_TRUE(window_size <= ctx_num);
  for (uint32_t idx = 0U; idx < window_size; ++idx) {
    if (ctx_id_vec[idx] > total_num) {
      GELOGE(ge::FAILED, "Context id [%d] is invalid.", ctx_id_vec[idx]);
      return ge::FAILED;
    }
    auto ctx = reinterpret_cast<rtFftsPlusSdmaCtx_t *>(ctx_head + ctx_id_vec[idx]);
    GE_ASSERT_NOTNULL(ctx);
    ctx->threadDim = thread_dim;
    ctx->tailDataLength = sdma_len->tail_len;
    ctx->nonTailDataLength = sdma_len->non_tail_len;

    if (input_mem_type == 0U) {
      auto tensor_data = context->GetInputPointer<GertTensorData>(static_cast<uint32_t>(SdmaUpdateKey::INPUT_TENSOR));
      GE_ASSERT_NOTNULL(tensor_data);
      auto addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tensor_data->GetAddr()));
      ctx->sourceAddressBaseL = static_cast<uint32_t>(addr & 0xFFFFFFFFFU);
      ctx->sourceAddressBaseH = static_cast<uint32_t>(addr >> 32U);
      ctx->sourceAddressOffset = ctx->nonTailDataLength;
    } else {
      auto ffts_mem =
          context->GetInputPointer<memory::FftsMemBlock>(static_cast<uint32_t>(SdmaUpdateKey::INPUT_TENSOR));
      GE_ASSERT_NOTNULL(ffts_mem);
      auto addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ffts_mem->Addr(idx)));
      ctx->sourceAddressBaseL = static_cast<uint32_t>(addr & 0xFFFFFFFFFU);
      ctx->sourceAddressBaseH = static_cast<uint32_t>(addr >> 32U);
      ctx->sourceAddressOffset = 0U;
    }

    if (output_mem_type == 0U) {
      auto tensor_data = context->GetInputPointer<GertTensorData>(static_cast<uint32_t>(SdmaUpdateKey::OUTPUT_TENSOR));
      GE_ASSERT_NOTNULL(tensor_data);
      auto addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tensor_data->GetAddr()));
      ctx->destinationAddressBaseL = static_cast<uint32_t>(addr & 0xFFFFFFFFFU);
      ctx->destinationAddressBaseH = static_cast<uint32_t>(addr >> 32U);
      ctx->destinationAddressOffset = ctx->nonTailDataLength;
    } else {
      auto ffts_mem =
          context->GetInputPointer<memory::FftsMemBlock>(static_cast<uint32_t>(SdmaUpdateKey::OUTPUT_TENSOR));
      GE_ASSERT_NOTNULL(ffts_mem);
      auto addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ffts_mem->Addr(idx)));
      ctx->destinationAddressBaseL = static_cast<uint32_t>(addr & 0xFFFFFFFFFU);
      ctx->destinationAddressBaseH = static_cast<uint32_t>(addr >> 32U);
      ctx->destinationAddressOffset = 0U;
    }
  }
  GELOGD("SdmaUpdateContext end.");

  return ge::GRAPH_SUCCESS;
}

std::vector<std::string> SdmaContextTracer(const KernelContext *context) {
  auto window_size = context->GetInputValue<uint32_t>(static_cast<size_t>(SdmaUpdateKey::WINDOW_SIZE));
  auto task_info = context->GetOutputPointer<rtFftsPlusTaskInfo_t>(0UL);
  if (task_info == nullptr) {
    return {"task info is nullptr"};
  }
  rtFftsPlusSdmaCtx_t *const ctx_head = reinterpret_cast<rtFftsPlusSdmaCtx_t *>(const_cast<void *>(task_info->descBuf));
  if (ctx_head == nullptr) {
    return {"ctx_head is nullptr"};
  }

  uint16_t total_num = task_info->fftsPlusSqe->totalContextNum;
  auto ctx_ids = context->GetInputPointer<gert::ContinuousVector>(static_cast<size_t>(SdmaUpdateKey::CTX_ID));
  if (ctx_ids == nullptr) {
    return {"context ids is nullptr"};
  }
  auto ctx_id_vec = reinterpret_cast<const int32_t *>(ctx_ids->GetData());
  const size_t ctx_num = ctx_ids->GetSize();
  if (window_size > ctx_num) {
    return {"window size is invalid"};
  }

  std::vector<std::string> strs;
  std::stringstream ss;
  for (uint32_t idx = 0U; idx < window_size; ++idx) {
    if (ctx_id_vec[idx] > total_num) {
      return {"ctx id is out of range"};
    }
    auto ctx = reinterpret_cast<rtFftsPlusSdmaCtx_t *>(ctx_head + ctx_id_vec[idx]);
    if (ctx != nullptr) {
      ss << "idx:" << std::dec << idx << " src_offset:" << ctx->sourceAddressOffset
         << " dst_offset:" << ctx->destinationAddressOffset << std::hex << " src_addr:" << ctx->sourceAddressBaseH
         << " " << ctx->sourceAddressBaseL << " dst_addr:" << ctx->destinationAddressBaseH << " "
         << ctx->destinationAddressBaseL;
      strs.emplace_back(ss.str().c_str());
      ss.clear();
      ss.str("");
    }
  }

  return strs;
}

REGISTER_KERNEL(SdmaUpdateContext).RunFunc(SdmaUpdateContext).TracePrinter(SdmaContextTracer);

ge::graphStatus CalcFftsThreadDataLen(KernelContext *const context) {
  auto non_tail_slice =
      context->GetInputPointer<gert::ContinuousVector>(static_cast<size_t>(CalcLenKey::NON_TAIL_SLICE));
  GE_ASSERT_NOTNULL(non_tail_slice);
  auto tail_slice = context->GetInputPointer<gert::ContinuousVector>(static_cast<size_t>(CalcLenKey::TAIL_SLICE));
  GE_ASSERT_NOTNULL(tail_slice);
  auto dtype = context->GetInputValue<ge::DataType>(static_cast<size_t>(CalcLenKey::DTYPE));
  auto slice_flag = context->GetInputValue<bool>(static_cast<size_t>(CalcLenKey::SLICE_FLAG));
  auto thread_dim = context->GetInputValue<uint32_t>(static_cast<size_t>(CalcLenKey::THREAD_DUM));

  auto mem_size = context->GetOutputPointer<size_t>(0UL);
  GE_ASSERT_NOTNULL(mem_size);
  auto data_len_holder = context->GetOutputPointer<SdmaDataLenHolder>(1UL);
  GE_ASSERT_NOTNULL(data_len_holder);

  const auto non_tail_shape = reinterpret_cast<const Shape *>(non_tail_slice->GetData());
  GE_ASSERT_NOTNULL(non_tail_shape);
  uint64_t non_tail_size{0UL};
  GE_ASSERT_GRAPH_SUCCESS(CalcUnalignedTensorSizeByShape(*non_tail_shape, dtype, non_tail_size));
  data_len_holder->non_tail_len = non_tail_size;

  if (!slice_flag) {
    data_len_holder->non_tail_len /= thread_dim;
    data_len_holder->tail_len = data_len_holder->non_tail_len;
    *mem_size = data_len_holder->non_tail_len;
  } else {
    GE_ASSERT_EQ(non_tail_slice->GetSize(), tail_slice->GetSize());
    GE_ASSERT_TRUE(non_tail_slice->GetSize() > 0UL);

    const auto tail_shape = reinterpret_cast<const Shape *>(tail_slice->GetData());
    GE_ASSERT_NOTNULL(tail_shape);
    uint64_t tail_size{0UL};
    GE_ASSERT_GRAPH_SUCCESS(CalcUnalignedTensorSizeByShape(*tail_shape, dtype, tail_size));
    data_len_holder->tail_len = tail_size;
    *mem_size = std::max(data_len_holder->non_tail_len, data_len_holder->tail_len);
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus BuildSdmaDataLenHolder(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  auto *addr_holder = new (std::nothrow) SdmaDataLenHolder();
  GE_ASSERT_NOTNULL(addr_holder);
  auto chain = context->GetOutput(1UL);
  GE_ASSERT_NOTNULL(chain);
  chain->SetWithDefaultDeleter(addr_holder);
  return ge::GRAPH_SUCCESS;
}

std::vector<std::string> SdmaCalcTracer(const KernelContext *context) {
  auto slice_flag = context->GetInputValue<bool>(static_cast<size_t>(CalcLenKey::SLICE_FLAG));
  auto thread_dim = context->GetInputValue<uint32_t>(static_cast<size_t>(CalcLenKey::THREAD_DUM));
  auto data_len_holder = context->GetOutputPointer<SdmaDataLenHolder>(1UL);
  std::stringstream ss;
  if (data_len_holder == nullptr) {
    return {"SdmaDataLenHolder is nullptr"};
  }
  ss << "slice_flag:" << slice_flag << "thread_dim:" << thread_dim << "non_tail_len:" << data_len_holder->non_tail_len
     << data_len_holder->tail_len;
  return {ss.str()};
}

REGISTER_KERNEL(CalcFftsThreadDataLen)
    .RunFunc(CalcFftsThreadDataLen)
    .OutputsCreator(BuildSdmaDataLenHolder)
    .TracePrinter(SdmaCalcTracer);
}  // namespace kernel
}  // namespace gert

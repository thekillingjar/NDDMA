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
#include "register/kernel_registry.h"
#include "engine/aicore/kernel/aicore_update_kernel.h"
#include "kernel/common_kernel_impl/sink_node_bin.h"
#include "stub/gert_runtime_stub.h"
#include "exe_graph/runtime/tensor.h"
#include "faker/kernel_outputs_faker.h"
#include "faker/node_faker.h"
#include "engine/aicore/kernel/rt_ffts_plus_launch_args.h"
#include "engine/aicore/kernel/aicore_update_kernel.h"
#include "stub/gert_runtime_stub.h"
#include "exe_graph/runtime/tensor.h"
#include "faker/kernel_run_context_facker.h"
#include "register/kernel_registry.h"
#include "register/ffts_node_calculater_registry.h"
#include "common/plugin/ge_make_unique_util.h"
#include <kernel/memory/mem_block.h>
#include "kernel/memory/ffts_mem_allocator.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "graph_builder/bg_memory.h"
#include "engine/aicore/kernel/aicore_update_kernel.h"
#include "kernel/common_kernel_impl/sink_node_bin.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "stub/gert_runtime_stub.h"
#include "exe_graph/runtime/tensor.h"
#include "engine/aicore/kernel/rt_ffts_plus_launch_args.h"
#include "engine/ffts_plus/converter/ffts_plus_proto_transfer.h"

namespace gert {
namespace kernel {}
using namespace kernel;

struct SdmaDataLenHolder {
  size_t tail_len;
  size_t non_tail_len;
};

class RtsFFTSPlusKernelTestUT : public testing::Test {
 public:
  KernelRegistryImpl &registry = KernelRegistryImpl::GetInstance();
};

TEST_F(RtsFFTSPlusKernelTestUT, calc_mem_len) {
  auto funcs = registry.FindKernelFuncs("CalcFftsThreadDataLen");
  ASSERT_NE(funcs, nullptr);
  ASSERT_NE(funcs->outputs_creator, nullptr);

  Shape shape({4, 2, 3});
  auto in_slice = ContinuousVector::Create<Shape>(12);
  auto in_slice_vec = reinterpret_cast<ContinuousVector *>(in_slice.get());
  in_slice_vec->SetSize(4);
  auto in_slice_ptr = reinterpret_cast<Shape *>(in_slice_vec->MutableData());
  for (size_t i = 0; i < in_slice_vec->GetSize(); i++) {
    in_slice_ptr[i] = shape;
  }

  bool slice_flag{true};
  uint32_t thread_dim{4};
  ge::DataType dt = ge::DT_FLOAT;
  auto context_holder = KernelRunContextFaker()
                            .KernelIONum(5U, 2U)
                            .Inputs({in_slice_vec, in_slice_vec, reinterpret_cast<void *>(dt),
                                     reinterpret_cast<void *>(slice_flag), reinterpret_cast<void *>(thread_dim)})
                            .Build();

  auto node = FastNodeFaker().Build();
  auto context = context_holder.GetContext<KernelContext>();
  EXPECT_EQ(funcs->outputs_creator(node, context), ge::GRAPH_SUCCESS);
  EXPECT_EQ(funcs->run_func(context), ge::GRAPH_SUCCESS);
  EXPECT_FALSE(funcs->trace_printer(context).empty());

  auto mem_size = context->GetOutputPointer<size_t>(0UL);
  EXPECT_NE(mem_size, nullptr);
  EXPECT_EQ(*mem_size, 96);

  auto data_len_holder = context->GetOutputPointer<SdmaDataLenHolder>(1UL);
  EXPECT_NE(data_len_holder, nullptr);
  EXPECT_EQ(data_len_holder->non_tail_len, 96);
  EXPECT_EQ(data_len_holder->tail_len, 96);
}

TEST_F(RtsFFTSPlusKernelTestUT, sdma_update_with_level1addr) {
  auto funcs = registry.FindKernelFuncs("SdmaUpdateContext");
  ASSERT_NE(funcs, nullptr);

  auto ctx_ids = ContinuousVector::Create<int32_t>(4);
  auto ctx_ids_vec = reinterpret_cast<ContinuousVector *>(ctx_ids.get());
  ctx_ids_vec->SetSize(4);
  auto ctx_ids_ptr = reinterpret_cast<int32_t *>(ctx_ids_vec->MutableData());
  for (size_t i = 0; i < ctx_ids_vec->GetSize(); i++) {
    ctx_ids_ptr[i] = i;
  }

  uint32_t thread_dim{4};
  uint32_t window_size{4};
  struct SdmaDataLenHolder sdma_holder {
    10, 10
  };

  size_t descBufLen = sizeof(rtFftsPlusSdmaCtx_t) * 4UL;
  size_t total_size = sizeof(TransTaskInfo) + descBufLen + sizeof(rtFftsPlusSqe_t);
  auto holder = ge::MakeUnique<uint8_t[]>(total_size);
  TransTaskInfo *task_info_ptr = reinterpret_cast<TransTaskInfo *>(holder.get());
  size_t buf_offset = sizeof(rtFftsPlusSqe_t);
  task_info_ptr->offsets[static_cast<size_t>(InfoStType::kDescBuf)] = buf_offset;
  task_info_ptr->rt_task_info.descBufLen = descBufLen;
  auto *buff_ptr = &task_info_ptr->args[buf_offset];
  for (int i = 0; i < 4; ++i) {
    auto context = reinterpret_cast<rtFftsPlusSdmaCtx_t *>(buff_ptr);
    context->contextType = RT_FFTS_SUB_TASK_TYPE_SDMA;
    buff_ptr += sizeof(rtFftsPlusSdmaCtx_t);
  }

  rtFftsPlusTaskInfo_t task_inf;
  auto *const ffts_plus_sqe = ge::PtrToPtr<uint8_t, rtFftsPlusSqe_t>(task_info_ptr->args);
  task_inf.fftsPlusSqe = ffts_plus_sqe;
  task_inf.descBuf = &task_info_ptr->args[buf_offset];
  ffts_plus_sqe->totalContextNum = 4;

  // level1
  uint32_t mem_type{0};
  auto device_data = std::vector<int8_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  GertTensorData in_data = {(uint8_t *)device_data.data(), 0U, kTensorPlacementEnd, -1};
  GertTensorData out_data = {(uint8_t *)device_data.data(), 0U, kTensorPlacementEnd, -1};
  auto context_holder =
      KernelRunContextFaker()
          .KernelIONum(8U, 1U)
          .Inputs({ctx_ids_vec, reinterpret_cast<void *>(thread_dim), reinterpret_cast<void *>(window_size),
                   &sdma_holder, reinterpret_cast<void *>(mem_type), reinterpret_cast<void *>(mem_type), &in_data,
                   &out_data})
          .Build();

  auto context = context_holder.GetContext<KernelContext>();
  auto task_info_holder = context->GetOutput(0UL);
  task_info_holder->Set(&task_inf, nullptr);
  EXPECT_EQ(funcs->run_func(context), ge::GRAPH_SUCCESS);
  EXPECT_FALSE(funcs->trace_printer(context).empty());

  auto *res = &task_info_ptr->args[buf_offset];
  for (int i = 0; i < 4; ++i) {
    auto sdma_ctx = reinterpret_cast<rtFftsPlusSdmaCtx_t *>(res);
    EXPECT_EQ(sdma_ctx->sourceAddressOffset, 10);
    res += sizeof(rtFftsPlusSdmaCtx_t);
  }

  // level2
  uint32_t l2_mem_type{1};
  auto level_1_allocator = memory::CachingMemAllocator::GetAllocator();
  ASSERT_NE(level_1_allocator, nullptr);
  auto level_2_allocator = memory::FftsMemAllocator::GetAllocator(*level_1_allocator, 4U);
  ASSERT_NE(level_2_allocator, nullptr);
  memory::FftsMemBlock *block = level_2_allocator->Malloc(10);

  auto l2_ctx_holder =
      KernelRunContextFaker()
          .KernelIONum(8U, 1U)
          .Inputs({ctx_ids_vec, reinterpret_cast<void *>(thread_dim), reinterpret_cast<void *>(window_size),
                   &sdma_holder, reinterpret_cast<void *>(l2_mem_type), reinterpret_cast<void *>(l2_mem_type), block,
                   block})
          .Build();

  auto l2_ctx = l2_ctx_holder.GetContext<KernelContext>();
  auto task_info = l2_ctx->GetOutput(0UL);
  task_info->Set(&task_inf, nullptr);
  EXPECT_EQ(funcs->run_func(l2_ctx), ge::GRAPH_SUCCESS);
  EXPECT_FALSE(funcs->trace_printer(l2_ctx).empty());

  auto *l2_res = &task_info_ptr->args[buf_offset];
  for (int i = 0; i < 4; ++i) {
    auto sdma_ctx = reinterpret_cast<rtFftsPlusSdmaCtx_t *>(l2_res);
    EXPECT_EQ(sdma_ctx->sourceAddressOffset, 0);
    EXPECT_EQ(sdma_ctx->threadDim, 4);
    l2_res += sizeof(rtFftsPlusSdmaCtx_t);
  }
  block->Free();
  level_2_allocator->Recycle();
  level_1_allocator->Finalize();
}
}  // namespace gert
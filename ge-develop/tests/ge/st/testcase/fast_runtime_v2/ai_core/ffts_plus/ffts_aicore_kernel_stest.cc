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

#include "macro_utils/dt_public_scope.h"
#include "graph/node.h"
#include "faker/kernel_run_context_facker.h"
#include "register/kernel_registry.h"
#include "register/ffts_node_calculater_registry.h"
#include "common/plugin/ge_make_unique_util.h"
#include <kernel/memory/mem_block.h>
#include "kernel/memory/ffts_mem_allocator.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "graph_builder/bg_memory.h"
#include "engine/aicore/kernel/aicore_update_kernel.h"
#include "engine/aicore/fe_rt2_common.h"
#include "kernel/common_kernel_impl/sink_node_bin.h"
#include "stub/gert_runtime_stub.h"
#include "engine/aicore/kernel/rt_ffts_plus_launch_args.h"
#include "engine/ffts_plus/converter/ffts_plus_proto_transfer.h"
#include "kernel/memory/single_stream_l2_allocator.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "kernel/memory/ffts_mem_allocator.h"
#include "common/dump/exception_dumper.h"
#include "subscriber/dumper/executor_dumper.h"
#include "subscriber/profiler/cann_profiler_v2.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "macro_utils/dt_public_unscope.h"

namespace gert {
namespace kernel {
}
using namespace kernel;
using namespace ge;

class FFTSAICoreKernelTestST : public testing::Test {
 public:
  KernelRegistryImpl &registry = KernelRegistryImpl::GetInstance();
};

TEST_F(FFTSAICoreKernelTestST, test_ffts_args_memory_success) {
  NodeMemPara node_para;
  RtFFTSKernelLaunchArgs::ComputeNodeDesc node_desc;
  node_desc.max_tiling_data = 63;
  node_desc.max_tail_tiling_data = 63;
  node_desc.addr_num = 1;
  node_desc.input_num = 2;
  node_desc.output_num = 1;
  node_desc.workspace_cap = 8;
  node_desc.thread_num_max = 16;
  size_t total_size = 0;
  ge::NodePtr tmp_node = std::make_shared<ge::Node>();
  auto rt_arg = RtFFTSKernelLaunchArgs::Create(tmp_node, node_desc, total_size);
  node_para.host_addr = rt_arg.get();
  auto run_context1 = BuildKernelRunContext(1, static_cast<size_t>(AllocFFTSArgOutputs::kNum));
  run_context1.value_holder[0].Set(&node_para, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("RedirectLaunchArgs")->run_func(run_context1),
            ge::GRAPH_SUCCESS);
  //ASSERT_EQ(run_context1.value_holder[4].GetValue<size_t>(), 3264);
}

TEST_F(FFTSAICoreKernelTestST, AllocateFftsMem) {
  // fake allocator
  memory::CachingMemAllocator allocator(0, RT_MEMORY_HBM);
  memory::SingleStreamL2Allocator single_stream_l2_allocator(&allocator);
  auto create_context_holder = KernelRunContextFaker().KernelIONum(2, 1)
      .Inputs({&single_stream_l2_allocator, reinterpret_cast<void *>(0x2)}).Build();
  auto create_context = create_context_holder.GetContext<KernelContext>();

  ASSERT_EQ(registry.FindKernelFuncs("CreateFftsMemAllocator")->run_func(create_context), ge::GRAPH_SUCCESS);

  auto ffts_holder = create_context->GetOutputPointer<memory::FftsMemAllocator *>(0U);
  ASSERT_NE(ffts_holder, nullptr);
  ASSERT_NE(*ffts_holder, nullptr);

  // allocate & free
  auto allocate_context_holder = KernelRunContextFaker().KernelIONum(2, 1)
      .Inputs({*ffts_holder, reinterpret_cast<void *>(0x10)}).Build();
  auto allocate_context = allocate_context_holder.GetContext<KernelContext>();
  ASSERT_EQ(registry.FindKernelFuncs("AllocateFftsMem")->run_func(allocate_context), ge::GRAPH_SUCCESS);

  auto block_holder = allocate_context->GetOutputPointer<memory::FftsMemBlock *>(0UL);
  ASSERT_NE(block_holder, nullptr);
  ASSERT_NE(*block_holder, nullptr);

  auto free_context_holder = KernelRunContextFaker().KernelIONum(1, 0)
      .Inputs({*block_holder}).Build();
  auto free_context = free_context_holder.GetContext<KernelContext>();
  ASSERT_EQ(registry.FindKernelFuncs("FreeFftsMem")->run_func(free_context), ge::GRAPH_SUCCESS);

  // allocate & free batch
  auto sizes_holder = ContinuousVector::Create<size_t>(8UL);
  auto sizes = reinterpret_cast<TypedContinuousVector<size_t> *>(sizes_holder.get());
  sizes->SetSize(2UL);
  sizes->MutableData()[0UL] = 128UL;
  sizes->MutableData()[1UL] = 128UL;

  auto batch_alloc_context_holder = KernelRunContextFaker().KernelIONum(2, 1)
      .Inputs({*ffts_holder, sizes_holder.get()}).Build();
  auto batch_alloc_context = batch_alloc_context_holder.GetContext<KernelContext>();
  ASSERT_EQ(registry.FindKernelFuncs("AllocateBatchFftsMems")->outputs_creator(
      nullptr, batch_alloc_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("AllocateBatchFftsMems")->run_func(batch_alloc_context), ge::GRAPH_SUCCESS);

  auto blocks = batch_alloc_context->GetOutputPointer<TypedContinuousVector<memory::FftsMemBlock *>>(0UL);
  ASSERT_NE(blocks, nullptr);
  ASSERT_GT(blocks->GetCapacity(), 2);
  ASSERT_NE(blocks->MutableData()[0], nullptr);
  ASSERT_NE(blocks->MutableData()[1], nullptr);

  auto batch_free_context_holder = KernelRunContextFaker().KernelIONum(1, 0)
      .Inputs({blocks}).Build();
  auto batch_free_context = batch_free_context_holder.GetContext<KernelContext>();
  ASSERT_EQ(registry.FindKernelFuncs("FreeBatchFftsMems")->run_func(batch_free_context), ge::GRAPH_SUCCESS);

  // recycle
  auto recycle_context_holder = KernelRunContextFaker().KernelIONum(1, 1).Inputs({*ffts_holder}).Build();
  auto recycle_context = recycle_context_holder.GetContext<KernelContext>();
  ASSERT_EQ(registry.FindKernelFuncs("RecycleFftsMems")->run_func(recycle_context), ge::GRAPH_SUCCESS);
}

TEST_F(FFTSAICoreKernelTestST, test_ffts_update_auto_args) {
  // update args
  auto level_1_allocator = memory::CachingMemAllocator::GetAllocator();
  ASSERT_NE(level_1_allocator, nullptr);
  auto level_2_allocator = memory::FftsMemAllocator::GetAllocator(*level_1_allocator, 2U);
  ASSERT_NE(level_2_allocator, nullptr);
  auto work_space = ContinuousVector::Create<memory::FftsMemBlock *>(3);
  auto work_space_vector = reinterpret_cast<ContinuousVector *>(work_space.get());
  work_space_vector->SetSize(3);
  auto work_space_ptr = reinterpret_cast<memory::FftsMemBlock **>(work_space_vector->MutableData());
  for (size_t i = 0; i < work_space_vector->GetSize(); i++) {
    work_space_ptr[i] = level_2_allocator->Malloc(2);
  }

  uint32_t thread_dim = 2;
  uint32_t window_size = 2;
  gert::GertTensorData data;
  AICoreSinkRet sink_ret;

  auto run_context = BuildKernelRunContext(static_cast<size_t>(AutoArgsInKey::kNUM) + 3, static_cast<size_t>(kernel::ArgsOutKey::kNUM));
  run_context.value_holder[static_cast<size_t>(AutoArgsInKey::WORKSPACE)].Set(work_space_vector, nullptr);
  run_context.value_holder[static_cast<size_t>(AutoArgsInKey::SINK_RET)].Set(&sink_ret, nullptr);
  run_context.value_holder[static_cast<size_t>(AutoArgsInKey::THREAD_DIM)]
      .Set(reinterpret_cast<void *>(thread_dim), nullptr);
  run_context.value_holder[static_cast<size_t>(AutoArgsInKey::WINDOW_SIZE)]
      .Set(reinterpret_cast<void *>(window_size), nullptr);

  auto offset = ContinuousVector::Create<uint64_t>(6);
  auto offset_vector = reinterpret_cast<ContinuousVector *>(offset.get());
  offset_vector->SetSize(5);
  auto offset_vector_ptr = reinterpret_cast<uint64_t*>(offset_vector->MutableData());
  for (size_t i = 0; i < offset_vector->GetSize(); i++) {
  offset_vector_ptr[i] = i;
  }
  run_context.value_holder[static_cast<size_t>(AutoArgsInKey::THREAD_OFFSET)].Set((void*)offset_vector, nullptr);
  size_t io_num = 1;
  run_context.value_holder[static_cast<size_t>(AutoArgsInKey::IN_NUM)].Set((void*)io_num, nullptr);
  run_context.value_holder[static_cast<size_t>(AutoArgsInKey::OUT_NUM)].Set((void*)io_num, nullptr);
  auto device_data = std::vector<int8_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  GertTensorData tensor_data = {(uint8_t *)device_data.data(), 0U, kTensorPlacementEnd, -1};
  GertTensorData out_data = {(uint8_t *)device_data.data(), 0U, kTensorPlacementEnd, -1};
  run_context.value_holder[static_cast<size_t>(AutoArgsInKey::IO_START)].Set(&tensor_data, nullptr);
  run_context.value_holder[static_cast<size_t>(AutoArgsInKey::IO_START) + 1].Set(&out_data, nullptr);

  NodeMemPara node_para;
  RtFFTSKernelLaunchArgs::ComputeNodeDesc node_desc;
  node_desc.max_tiling_data = 63;
  node_desc.max_tail_tiling_data = 63;
  node_desc.addr_num = 1;
  node_desc.input_num = 1;
  node_desc.output_num = 1;
  node_desc.workspace_cap = 8;
  node_desc.thread_num_max = 16;
  size_t total_size = 0;
  ge::NodePtr tmp_node = std::make_shared<ge::Node>();
  auto rt_arg = RtFFTSKernelLaunchArgs::Create(tmp_node, node_desc, total_size);
  node_para.host_addr = rt_arg.get();
  char *mem_2 = new char[total_size];
  node_para.dev_addr = (void*)mem_2;
  AICoreSubTaskFlush flush_data;
  run_context.value_holder[static_cast<size_t>(AutoArgsInKey::IO_START) + 2].Set(&flush_data, nullptr);
  run_context.value_holder[static_cast<size_t>(AutoArgsInKey::ARGS_PARA)].Set(&node_para, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("FFTSUpdateAutoAICoreArgs")->outputs_creator(nullptr, run_context),
  ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("FFTSUpdateAutoAICoreArgs")->run_func(run_context),
      ge::GRAPH_SUCCESS);
  EXPECT_FALSE(registry.FindKernelFuncs("FFTSUpdateAutoAICoreArgs")->trace_printer(run_context).empty());
  delete[] mem_2;
  for (size_t i = 0; i < work_space_vector->GetSize(); i++) {
    level_2_allocator->Free(work_space_ptr[i]);
  }
  level_2_allocator->Recycle();
  level_1_allocator->Finalize();
}

// TEST_F(FFTSAICoreKernelTestST, test_AICoreUpdateGeExceptionDumpInfo) {
//   auto context = BuildKernelRunContext(static_cast<size_t>(ExceptionDumpKey::RESERVED), 0);
//   uint32_t thread_dim = 1;
//   context.value_holder[static_cast<size_t>(ExceptionDumpKey::THREAD_DIM)]
//       .Set(reinterpret_cast<void *>(thread_dim), nullptr);

//   auto level_1_allocator = memory::CachingMemAllocator::GetAllocator();
//   ASSERT_NE(level_1_allocator, nullptr);
//   auto level_2_allocator = memory::FftsMemAllocator::GetAllocator(*level_1_allocator, 2U);
//   ASSERT_NE(level_2_allocator, nullptr);
//   size_t ws_size = 2;
//   auto work_space = ContinuousVector::Create<memory::FftsMemBlock *>(ws_size);
//   auto work_space_vector = reinterpret_cast<ContinuousVector *>(work_space.get());
//   work_space_vector->SetSize(ws_size);
//   auto work_space_ptr = reinterpret_cast<memory::FftsMemBlock **>(work_space_vector->MutableData());
//   for (size_t i = 0; i < work_space_vector->GetSize(); i++) {
//     work_space_ptr[i] = level_2_allocator->Malloc(ws_size);
//   }
//   context.value_holder[static_cast<size_t>(ExceptionDumpKey::WORKSPACE)].Set(work_space_vector, nullptr);

//   RtFFTSKernelLaunchArgs::ComputeNodeDesc node_desc;
//   node_desc.max_tiling_data = 63;
//   node_desc.max_tail_tiling_data = 0;
//   node_desc.addr_num = 0;
//   node_desc.input_num = 2;
//   node_desc.output_num = 1;
//   node_desc.workspace_cap = 8;
//   node_desc.thread_num_max = 1;
//   size_t size = 0;
//   ge::NodePtr tmp_node = std::make_shared<ge::Node>();
//   auto rt_arg = RtFFTSKernelLaunchArgs::Create(tmp_node, node_desc, size);

//   NodeMemPara node_para;
//   char *mem_1 = new char[size];
//   char *mem_2 = new char[size];
//   node_para.host_addr = (void*)mem_1;
//   node_para.dev_addr = (void*)mem_2;
//   (void)memcpy_s(node_para.host_addr, node_para.size, rt_arg.get(), node_para.size);
//   context.value_holder[static_cast<size_t>(ExceptionDumpKey::ARGS_PARA)].Set(&node_para, nullptr);

//   ExtraOpInfo dump_unit;
//   ExecutorExceptionDumpInfoWrapper wrapper(&dump_unit);
//   ASSERT_NE(registry.FindKernelFuncs("AICoreUpdateExceptionDumpInfo"), nullptr);
//   auto ret = registry.FindKernelFuncs("AICoreUpdateExceptionDumpInfo")->exception_dump_info_filler(context, wrapper);
//   ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
//   ASSERT_EQ(registry.FindKernelFuncs("AICoreUpdateExceptionDumpInfo")->run_func(context),
//       ge::GRAPH_SUCCESS);
//   delete[] mem_1;
//   delete[] mem_2;
//   for (size_t i = 0; i < work_space_vector->GetSize(); i++) {
//     level_2_allocator->Free(work_space_ptr[i]);
//   }
//   level_2_allocator->Recycle();
//   level_1_allocator->Finalize();
// }

TEST_F(FFTSAICoreKernelTestST, test_auto_aicore_update_context) {
  AICoreSubTaskFlush flush_data;
  flush_data.input_addr_vv[0][0] = 0x1000;
  flush_data.input_addr_vv[0][1] = 0x1001;
  flush_data.output_addr_vv[0][0] = 0x2000;
  flush_data.output_addr_vv[0][1] = 0x2001;
  flush_data.thread_dim = 2;
  flush_data.window_size = 2;

  auto prefetch_idx = ContinuousVector::Create<uint32_t>(2);
  auto prefetch_idx_vec = reinterpret_cast<ContinuousVector *>(prefetch_idx.get());
  prefetch_idx_vec->SetSize(1);
  auto prefetch_idx_ptr = reinterpret_cast<uint32_t *>(prefetch_idx_vec->MutableData());
  for (size_t i = 0; i < prefetch_idx_vec->GetSize(); i++) {
  prefetch_idx_ptr[i] = 0;
  }
  auto empty_idx = ContinuousVector::Create<uint32_t>(2);
  auto empty_idx_vec = reinterpret_cast<ContinuousVector *>(empty_idx.get());
  empty_idx_vec->SetSize(0);

  // context id: 0~3
  auto ctx_ids = ContinuousVector::Create<int32_t>(4);
  auto ctx_ids_vec = reinterpret_cast<ContinuousVector *>(ctx_ids.get());
  ctx_ids_vec->SetSize(4);
  auto ctx_ids_ptr = reinterpret_cast<int32_t *>(ctx_ids_vec->MutableData());
  for (size_t i = 0; i < ctx_ids_vec->GetSize(); i++) {
  ctx_ids_ptr[i] = i;
  }

  // prefetch context id: 4~5
  auto prefetch_ctx = ContinuousVector::Create<uint32_t>(4);
  auto prefetch_ctx_vec = reinterpret_cast<ContinuousVector *>(prefetch_ctx.get());
  prefetch_ctx_vec->SetSize(2);
  auto prefetch_ctx_ptr = reinterpret_cast<uint32_t *>(prefetch_ctx_vec->MutableData());
  for (size_t i = 0; i < prefetch_ctx_vec->GetSize(); i++) {
  prefetch_ctx_ptr[i] = i + 4;
  }
  // writeback context id: 6~7
  auto writ_ctx = ContinuousVector::Create<uint32_t>(4);
  auto writ_ctx_vec = reinterpret_cast<ContinuousVector *>(writ_ctx.get());
  writ_ctx_vec->SetSize(2);
  auto writ_ctx_ptr = reinterpret_cast<uint32_t *>(writ_ctx_vec->MutableData());
  for (size_t i = 0; i < writ_ctx_vec->GetSize(); i++) {
  writ_ctx_ptr[i] = i + 6;
  }
  // invalid context id: 8~9
  auto invalid_ctx = ContinuousVector::Create<uint32_t>(4);
  auto invalid_ctx_vec = reinterpret_cast<ContinuousVector *>(invalid_ctx.get());
  invalid_ctx_vec->SetSize(2);
  auto invalid_ctx_ptr = reinterpret_cast<uint32_t *>(invalid_ctx_vec->MutableData());
  for (size_t i = 0; i < invalid_ctx_vec->GetSize(); i++) {
  invalid_ctx_ptr[i] = i + 8;
  }
  auto run_context = KernelRunContextFaker().KernelIONum(static_cast<size_t>(AutoUpdateKey::RESERVED), 0).
      NodeIoNum(2,2).IrInputNum(2)
      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .Build();
  run_context.value_holder[static_cast<size_t>(AutoUpdateKey::FLUSH_DATA)].Set(&flush_data, nullptr);
  run_context.value_holder[static_cast<size_t>(AutoUpdateKey::AICORE_CTX)].Set(ctx_ids_vec, nullptr);
  run_context.value_holder[static_cast<size_t>(AutoUpdateKey::PREFETCH_IDX)].Set(prefetch_idx_vec, nullptr);
  run_context.value_holder[static_cast<size_t>(AutoUpdateKey::WRITEBACK_IDX)].Set(prefetch_idx_vec, nullptr);
  run_context.value_holder[static_cast<size_t>(AutoUpdateKey::INVALID_IDX)].Set(empty_idx_vec, nullptr);
  run_context.value_holder[static_cast<size_t>(AutoUpdateKey::PREFETCH_CTX)].Set(prefetch_ctx_vec, nullptr);
  run_context.value_holder[static_cast<size_t>(AutoUpdateKey::WRITEBACK_CTX)].Set(writ_ctx_vec, nullptr);
  run_context.value_holder[static_cast<size_t>(AutoUpdateKey::INVALID_CTX)].Set(invalid_ctx_vec, nullptr);

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
  auto context = reinterpret_cast<rtFftsPlusAicAivCtx_t*>(buff_ptr);
  context->contextType = RT_CTX_TYPE_AICORE;
  }
  for (int i = 0; i < 2; ++i) {
  buff_ptr += sizeof(rtFftsPlusComCtx_t);
  auto context = reinterpret_cast<rtFftsPlusDataCtx_t*>(buff_ptr);
  context->contextType = RT_CTX_TYPE_FLUSH_DATA;
  }
  for (int i = 0; i < 2; ++i) {
  buff_ptr += sizeof(rtFftsPlusComCtx_t);
  auto context = reinterpret_cast<rtFftsPlusDataCtx_t*>(buff_ptr);
  context->contextType = RT_CTX_TYPE_WRITEBACK_DATA;
  }
  for (int i = 0; i < 2; ++i) {
  buff_ptr += sizeof(rtFftsPlusComCtx_t);
  auto context = reinterpret_cast<rtFftsPlusDataCtx_t*>(buff_ptr);
  context->contextType = RT_CTX_TYPE_INVALIDATE_DATA;
  }
  rtFftsPlusTaskInfo_t task_inf;
  auto *const ffts_plus_sqe = ge::PtrToPtr<uint8_t, rtFftsPlusSqe_t>(task_info_ptr->args);
  task_inf.fftsPlusSqe = ffts_plus_sqe;
  task_inf.descBuf = &task_info_ptr->args[buf_offset];
  ffts_plus_sqe->totalContextNum = 16;
  run_context.value_holder[static_cast<size_t>(AutoUpdateKey::TASK_INFO)].Set(&task_inf, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("StaAutoUpdateContext")->run_func(run_context), ge::GRAPH_SUCCESS);

  flush_data.window_size = 5;
  ASSERT_NE(registry.FindKernelFuncs("StaAutoUpdateContext")->run_func(run_context), ge::GRAPH_SUCCESS);
  flush_data.window_size = 2;

  prefetch_idx_ptr[0] = 155;
  ASSERT_NE(registry.FindKernelFuncs("StaAutoUpdateContext")->run_func(run_context), ge::GRAPH_SUCCESS);
  prefetch_idx_ptr[0] = 0;

  prefetch_ctx_ptr[0] = 24;
  ASSERT_NE(registry.FindKernelFuncs("StaAutoUpdateContext")->run_func(run_context), ge::GRAPH_SUCCESS);
  prefetch_ctx_ptr[0] = 4;

  ctx_ids_ptr[0] = 21;
  ASSERT_NE(registry.FindKernelFuncs("StaAutoUpdateContext")->run_func(run_context), ge::GRAPH_SUCCESS);
  ctx_ids_ptr[0] = 0;

  // profiling
  uint32_t blk = 4;
  run_context.value_holder[static_cast<size_t>(AutoUpdateKey::BLOCK_DIM)].Set(reinterpret_cast<void *>(blk), nullptr);
  ProfNodeAdditionInfo info;
  CannProfilingInfoWrapper prof_info(&info);
  ASSERT_EQ(registry.FindKernelFuncs("StaAutoUpdateContext")->profiling_info_filler(run_context, prof_info),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(prof_info.add_infos_[0]->node_basic_info.data.nodeBasicInfo.blockDim, 4);
}

TEST_F(FFTSAICoreKernelTestST, test_StaticUpdateManualGeDataDumpInfo) {
  auto context = BuildKernelRunContext(static_cast<size_t>(ManualDataDumpKey::RESERVED) + 3, 0);
  uint32_t thread_id = 2;
  uint32_t ctx_id = 2;
  uint32_t in_num = 1;
  uint32_t out_num = 1;
  context.value_holder[static_cast<size_t>(ManualDataDumpKey::THREAD_ID)]
      .Set(reinterpret_cast<void *>(thread_id), nullptr);
  context.value_holder[static_cast<size_t>(ManualDataDumpKey::CONTEXT_ID)]
      .Set(reinterpret_cast<void *>(ctx_id), nullptr);
  context.value_holder[static_cast<size_t>(ManualDataDumpKey::IN_NUM)].Set(reinterpret_cast<void *>(in_num), nullptr);
  context.value_holder[static_cast<size_t>(ManualDataDumpKey::OUT_NUM)]
      .Set(reinterpret_cast<void *>(out_num), nullptr);

  auto input_shapes_size = std::vector<uint64_t>{1};
  auto output_shapes_size = std::vector<uint64_t>{1};
  context.value_holder[static_cast<size_t>(ManualDataDumpKey::IO_START)].Set(&input_shapes_size, nullptr);
  context.value_holder[static_cast<size_t>(ManualDataDumpKey::IO_START) + 1].Set(&output_shapes_size, nullptr);
  auto device_data1 = std::vector<int8_t>{10};

  GertTensorData in_data = {(uint8_t *)device_data1.data(), 0U, kTensorPlacementEnd, -1};
  GertTensorData out_data = {(uint8_t *)device_data1.data(), 0U, kTensorPlacementEnd, -1};
  context.value_holder[static_cast<size_t>(ManualDataDumpKey::IO_START) + 2].Set(&in_data, nullptr);
  context.value_holder[static_cast<size_t>(ManualDataDumpKey::IO_START) + 3].Set(&out_data, nullptr);

  ASSERT_NE(registry.FindKernelFuncs("StaticUpdateManualDataDumpInfo"), nullptr);
  NodeDumpUnit dump_unit;
  gert::ExecutorDataDumpInfoWrapper wrapper(&dump_unit);
  auto ret = registry.FindKernelFuncs("StaticUpdateManualDataDumpInfo")->data_dump_info_filler(context, wrapper);
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("StaticUpdateManualDataDumpInfo")->run_func(context),
      ge::GRAPH_SUCCESS);
}

// TEST_F(FFTSAICoreKernelTestST, test_StaticManualUpdateGeExceptionDumpInfo) {
//   auto context = BuildKernelRunContext(static_cast<size_t>(ManualExceptionDumpKey::RESERVED), 0);

//   size_t ws_size = 2;
//   auto work_space = ContinuousVector::Create<memory::MultiStreamMemBlock *>(ws_size);
//   auto work_space_vector = reinterpret_cast<ContinuousVector *>(work_space.get());
//   work_space_vector->SetSize(ws_size);
//   auto work_space_ptr = reinterpret_cast<memory::MultiStreamMemBlock **>(work_space_vector->MutableData());
//   constexpr uint64_t stub_mem_hbm_addr = 0x22;
//   memory::CachingMemAllocator stub_allocator{0, 1};
//   memory::SingleStreamL2Allocator single_stream_l2_allocator{&stub_allocator};
//   ge::MemBlock mem_block(stub_allocator, reinterpret_cast<void *>(stub_mem_hbm_addr), 100);
//   memory::MultiStreamMemBlock ms_block;
//   ms_block.ReInit(&single_stream_l2_allocator, &mem_block, memory::BlockAllocType(memory::BlockAllocType::kNorm, 0U));
//   for (size_t i = 0; i < work_space_vector->GetSize(); i++) {
//     work_space_ptr[i] = &ms_block;
//   }
//   context.value_holder[static_cast<size_t>(ManualExceptionDumpKey::WORKSPACE)].Set(work_space_vector, nullptr);

//   NodeMemPara node_para;
//   RtFFTSKernelLaunchArgs::ComputeNodeDesc node_desc;
//   node_desc.max_tiling_data = 63;
//   node_desc.max_tail_tiling_data = 0;
//   node_desc.addr_num = 0;
//   node_desc.input_num = 2;
//   node_desc.output_num = 1;
//   node_desc.workspace_cap = 8;
//   node_desc.thread_num_max = 1;
//   size_t total_size = 0;
//   ge::NodePtr tmp_node = std::make_shared<ge::Node>();
//   auto rt_arg = RtFFTSKernelLaunchArgs::Create(tmp_node, node_desc, total_size);
//   char *mem_1 = new char[total_size];
//   char *mem_2 = new char[total_size];
//   node_para.host_addr = (void*)mem_1;
//   node_para.dev_addr = (void*)mem_2;
//   (void)memcpy_s(node_para.host_addr, node_para.size, rt_arg.get(), node_para.size);
//   context.value_holder[static_cast<size_t>(ManualExceptionDumpKey::ARGS_PARA)].Set(&node_para, nullptr);

//   ExtraOpInfo dump_unit;
//   ExecutorExceptionDumpInfoWrapper wrapper(&dump_unit);
//   ASSERT_NE(registry.FindKernelFuncs("StaticManualUpdateExceptionDumpInfo"), nullptr);
//   auto ret = registry.FindKernelFuncs("StaticManualUpdateExceptionDumpInfo")->exception_dump_info_filler(context, wrapper);
//   ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
//   ASSERT_EQ(registry.FindKernelFuncs("StaticManualUpdateExceptionDumpInfo")->run_func(context),
//       ge::GRAPH_SUCCESS);
//   delete[] mem_1;
//   delete[] mem_2;
// }

TEST_F(FFTSAICoreKernelTestST, test_StaticUpdateAutoDataDumpInfo) {
  auto context = BuildKernelRunContext(static_cast<size_t>(AutoDataDumpKey::RESERVED) + 3, 0);
  uint32_t thread_dim = 2;
  uint32_t window_size = 2;
  uint32_t in_num = 1;
  uint32_t out_num = 1;
  context.value_holder[static_cast<size_t>(AutoDataDumpKey::THREAD_DIM)]
      .Set(reinterpret_cast<void *>(thread_dim), nullptr);
  context.value_holder[static_cast<size_t>(AutoDataDumpKey::WINDOW_SIZE)]
      .Set(reinterpret_cast<void *>(window_size), nullptr);
  context.value_holder[static_cast<size_t>(AutoDataDumpKey::IN_NUM)].Set(reinterpret_cast<void *>(in_num), nullptr);
  context.value_holder[static_cast<size_t>(AutoDataDumpKey::OUT_NUM)].Set(reinterpret_cast<void *>(out_num), nullptr);
  auto ctx_ids = ContinuousVector::Create<uint64_t>(4);
  auto ctx_ids_vec = reinterpret_cast<ContinuousVector *>(ctx_ids.get());
  ctx_ids_vec->SetSize(4);
  auto ctx_ids_vec_ptr = reinterpret_cast<uint64_t*>(ctx_ids_vec->MutableData());
  for (size_t i = 0; i < ctx_ids_vec->GetSize(); ++i) {
    ctx_ids_vec_ptr[i] = i;
  }
  context.value_holder[static_cast<size_t>(AutoDataDumpKey::AICORE_CTX)].Set((void*)ctx_ids_vec, nullptr);
  auto offset = ContinuousVector::Create<uint64_t>(6);
  auto offset_vector = reinterpret_cast<ContinuousVector *>(offset.get());
  offset_vector->SetSize(5);
  auto offset_vector_ptr = reinterpret_cast<uint64_t*>(offset_vector->MutableData());
  for (size_t i = 0; i < offset_vector->GetSize(); ++i) {
    offset_vector_ptr[i] = i;
  }
  context.value_holder[static_cast<size_t>(AutoDataDumpKey::THREAD_ADDR_OFFSET)].Set((void*)offset_vector, nullptr);

  auto input_shapes_size = std::vector<uint64_t>{1};
  auto output_shapes_size = std::vector<uint64_t>{1};
  context.value_holder[static_cast<size_t>(AutoDataDumpKey::IO_START)].Set(&input_shapes_size, nullptr);
  context.value_holder[static_cast<size_t>(AutoDataDumpKey::IO_START) + 1].Set(&output_shapes_size, nullptr);
  auto device_data1 = std::vector<int8_t>{10};
  GertTensorData in_data = {(uint8_t *)device_data1.data(), 0U, kTensorPlacementEnd, -1};
  GertTensorData out_data = {(uint8_t *)device_data1.data(), 0U, kTensorPlacementEnd, -1};
  context.value_holder[static_cast<size_t>(AutoDataDumpKey::IO_START) + 2].Set(&in_data, nullptr);
  context.value_holder[static_cast<size_t>(AutoDataDumpKey::IO_START) + 3].Set(&out_data, nullptr);

  ASSERT_NE(registry.FindKernelFuncs("StaticUpdateAutoDataDumpInfo"), nullptr);
  NodeDumpUnit dump_unit;
  gert::ExecutorDataDumpInfoWrapper wrapper(&dump_unit);
  auto ret = registry.FindKernelFuncs("StaticUpdateAutoDataDumpInfo")->data_dump_info_filler(context, wrapper);
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("StaticUpdateAutoDataDumpInfo")->run_func(context),
      ge::GRAPH_SUCCESS);
}

// TEST_F(FFTSAICoreKernelTestST, test_StaticAutoUpdateGeExceptionDumpInfo) {
//   auto context = BuildKernelRunContext(static_cast<size_t>(AutoExceptionDumpKey::RESERVED), 0);
//   uint32_t thread_dim = 1;
//   context.value_holder[static_cast<size_t>(AutoExceptionDumpKey::THREAD_DIM)]
//       .Set(reinterpret_cast<void *>(thread_dim), nullptr);
//   auto level_1_allocator = memory::CachingMemAllocator::GetAllocator();
//   ASSERT_NE(level_1_allocator, nullptr);
//   auto level_2_allocator = memory::FftsMemAllocator::GetAllocator(*level_1_allocator, 2U);
//   ASSERT_NE(level_2_allocator, nullptr);
//   size_t ws_size = 2;
//   auto work_space = ContinuousVector::Create<memory::FftsMemBlock *>(ws_size);
//   auto work_space_vector = reinterpret_cast<ContinuousVector *>(work_space.get());
//   work_space_vector->SetSize(ws_size);
//   auto work_space_ptr = reinterpret_cast<memory::FftsMemBlock **>(work_space_vector->MutableData());
//   for (size_t i = 0; i < work_space_vector->GetSize(); i++) {
//     work_space_ptr[i] = level_2_allocator->Malloc(ws_size);
//   }
//   context.value_holder[static_cast<size_t>(AutoExceptionDumpKey::WORKSPACE)].Set(work_space_vector, nullptr);

//   NodeMemPara node_para;
//   RtFFTSKernelLaunchArgs::ComputeNodeDesc node_desc;
//   node_desc.max_tiling_data = 63;
//   node_desc.max_tail_tiling_data = 0;
//   node_desc.addr_num = 0;
//   node_desc.input_num = 2;
//   node_desc.output_num = 1;
//   node_desc.workspace_cap = 8;
//   node_desc.thread_num_max = 1;
//   size_t total_size = 0;
//   ge::NodePtr tmp_node = std::make_shared<ge::Node>();
//   auto rt_arg = RtFFTSKernelLaunchArgs::Create(tmp_node, node_desc, total_size);
//   char *mem_1 = new char[total_size];
//   char *mem_2 = new char[total_size];
//   node_para.host_addr = (void*)mem_1;
//   node_para.dev_addr = (void*)mem_2;
//   (void)memcpy_s(node_para.host_addr, node_para.size, rt_arg.get(), node_para.size);
//   context.value_holder[static_cast<size_t>(AutoExceptionDumpKey::ARGS_PARA)].Set(&node_para, nullptr);

//   ExtraOpInfo dump_unit;
//   ExecutorExceptionDumpInfoWrapper wrapper(&dump_unit);
//   ASSERT_NE(registry.FindKernelFuncs("StaticAutoUpdateExceptionDumpInfo"), nullptr);
//   auto ret = registry.FindKernelFuncs("StaticAutoUpdateExceptionDumpInfo")->exception_dump_info_filler(context, wrapper);
//   ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
//   ASSERT_EQ(registry.FindKernelFuncs("StaticAutoUpdateExceptionDumpInfo")->run_func(context),
//       ge::GRAPH_SUCCESS);
//   delete[] mem_1;
//   delete[] mem_2;
//   for (size_t i = 0; i < work_space_vector->GetSize(); i++) {
//     level_2_allocator->Free(work_space_ptr[i]);
//   }
//   level_2_allocator->Recycle();
//   level_1_allocator->Finalize();
// }

TEST_F(FFTSAICoreKernelTestST, test_atomic_aicore_update_context) {
  auto run_context = KernelRunContextFaker().KernelIONum(static_cast<size_t>(AutoUpdateKey::RESERVED), 0).
      NodeIoNum(2,2).IrInputNum(2)
      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .Build();

  // profiling
  uint32_t blk = 4;
  run_context.value_holder[2].Set((void *)&blk, nullptr);
  ProfNodeAdditionInfo info;
  CannProfilingInfoWrapper prof_info(&info);
  ASSERT_EQ(registry.FindKernelFuncs("AtomicLaunchKernelWithFlag")->profiling_info_filler(run_context, prof_info),
            ge::GRAPH_SUCCESS);
}


}  // namespace gert

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
#include "faker/kernel_run_context_facker.h"
#include "register/kernel_registry.h"
#include "register/ffts_node_calculater_registry.h"
#include "common/plugin/ge_make_unique_util.h"
#include <kernel/memory/mem_block.h>
#include "kernel/memory/ffts_mem_allocator.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "kernel/memory/single_stream_l2_allocator.h"
#include "graph_builder/bg_memory.h"
#include "engine/aicore/kernel/aicore_update_kernel.h"
#include "kernel/common_kernel_impl/sink_node_bin.h"
#include "stub/gert_runtime_stub.h"
#include "exe_graph/runtime/tensor.h"
#include "engine/aicore/kernel/rt_ffts_plus_launch_args.h"
#include "engine/ffts_plus/converter/ffts_plus_proto_transfer.h"
#include "subscriber/profiler/cann_profiler_v2.h"
#include "macro_utils/dt_public_unscope.h"
namespace gert {
namespace kernel {
}
using namespace kernel;

class FFTSAtomicKernelTestUT : public testing::Test {
 public:
  KernelRegistryImpl &registry = KernelRegistryImpl::GetInstance();
};

TEST_F(FFTSAtomicKernelTestUT, test_ffts_atomic_update_args) {
  // update args
  auto level_1_allocator = memory::CachingMemAllocator::GetAllocator();
  ASSERT_NE(level_1_allocator, nullptr);
  auto level_2_allocator = memory::FftsMemAllocator::GetAllocator(*level_1_allocator, 2U);
  ASSERT_NE(level_2_allocator, nullptr);
  auto work_space = ContinuousVector::Create<memory::FftsMemBlock *>(4);
  auto work_space_vector = reinterpret_cast<ContinuousVector *>(work_space.get());
  work_space_vector->SetSize(4);
  auto work_space_ptr = reinterpret_cast<memory::FftsMemBlock **>(work_space_vector->MutableData());
    for (size_t i = 0; i < work_space_vector->GetSize(); i++) {
    work_space_ptr[i] = level_2_allocator->Malloc(2);
  }

  auto l1_work_space = ContinuousVector::Create<GertTensorData *>(4);
  auto l1_work_space_vector = reinterpret_cast<ContinuousVector *>(l1_work_space.get());
  l1_work_space_vector->SetSize(4);
  constexpr uint64_t stub_mem_hbm_addr = 0x22;
  memory::CachingMemAllocator stub_allocator{0, 1};
  ge::MemBlock mem_block(stub_allocator, reinterpret_cast<void *>(stub_mem_hbm_addr), 3);
  memory::SingleStreamL2Allocator single_stream_l2_allocator{&stub_allocator};
  memory::MultiStreamMemBlock ms_block;
  ms_block.ReInit(&single_stream_l2_allocator, &mem_block, memory::BlockAllocType(memory::BlockAllocType::kNorm, 0U));
  GertTensorData workspace_gtd = GertTensorData(ms_block.GetAddr(), ms_block.GetSize(), TensorPlacement::kOnDeviceHbm, 0);
  auto l1_work_space_ptr = reinterpret_cast<GertTensorData **>(l1_work_space_vector->MutableData());
  for (size_t i = 0; i < l1_work_space_vector->GetSize(); i++) {
    l1_work_space_ptr[i] = &workspace_gtd;
  }

  AICoreThreadParam thread_param{0};
  thread_param.input_output_num = 2;
  uint32_t thread_dim = 2;
  uint32_t window_size = 2;
  AICoreSinkRet sink_ret;

  auto run_context = BuildKernelRunContext(static_cast<size_t>(AtomArgsInKey::kNUM) + 3, 1);
  run_context.value_holder[static_cast<size_t>(AtomArgsInKey::THREAD_DIM)].Set(reinterpret_cast<void *>(thread_dim), nullptr);
  run_context.value_holder[static_cast<size_t>(AtomArgsInKey::WINDOW_SIZE)].Set(reinterpret_cast<void *>(window_size), nullptr);
  run_context.value_holder[static_cast<size_t>(AtomArgsInKey::SINK_RET)].Set(&sink_ret, nullptr);
  run_context.value_holder[static_cast<size_t>(AtomArgsInKey::WORK_ADDR)].Set(work_space_vector, nullptr);
  run_context.value_holder[static_cast<size_t>(AtomArgsInKey::THREAD_PARAM)].Set(&thread_param, nullptr);
  uint32_t proc_type = static_cast<uint32_t>(AtomProcType::DY_SLICE_OP);
  run_context.value_holder[static_cast<size_t>(AtomArgsInKey::PROC_TYPE)].Set(reinterpret_cast<void *>(proc_type), nullptr);
  auto work_clear = ContinuousVector::Create<int64_t>(2);
  auto work_clear_vec = reinterpret_cast<ContinuousVector *>(work_clear.get());
  work_clear_vec->SetSize(2);
  auto work_clear_ptr = reinterpret_cast<int64_t*>(work_clear_vec->MutableData());
  for (size_t i = 0; i < work_clear_vec->GetSize(); i++) {
    work_clear_ptr[i] = i + 1;
  }
  run_context.value_holder[static_cast<size_t>(AtomArgsInKey::WORK_CLEAR_IDX)].Set(work_clear_vec, nullptr);

  auto out_clear = ContinuousVector::Create<int64_t>(2);
  auto out_clear_vec = reinterpret_cast<ContinuousVector *>(out_clear.get());
  out_clear_vec->SetSize(2);
  auto out_clear_ptr = reinterpret_cast<int64_t*>(out_clear_vec->MutableData());
  for (size_t i = 0; i < out_clear_vec->GetSize(); i++) {
    out_clear_ptr[i] = i;
  }
  run_context.value_holder[static_cast<size_t>(AtomArgsInKey::OUT_CLEAR_IDX)].Set(out_clear_vec, nullptr);

  auto out_type = ContinuousVector::Create<uint32_t>(2);
  auto out_type_vec = reinterpret_cast<ContinuousVector *>(out_type.get());
  out_type_vec->SetSize(2);
  auto out_type_ptr = reinterpret_cast<uint32_t*>(out_type_vec->MutableData());
  out_type_ptr[0] = 0;
  out_type_ptr[1] = 1;
  run_context.value_holder[static_cast<size_t>(AtomArgsInKey::OUT_MEM_TYPE)].Set(out_type_vec, nullptr);

  auto device_data1 = std::vector<int8_t>{10};
  GertTensorData out_data = {(uint8_t *)device_data1.data(), 0U, kTensorPlacementEnd, -1};
  out_data.SetSize(device_data1.size());
  run_context.value_holder[static_cast<size_t>(AtomArgsInKey::IO_START)].Set(&out_data, nullptr);

  memory::FftsMemBlock *out_data1 = level_2_allocator->Malloc(20);
  run_context.value_holder[static_cast<size_t>(AtomArgsInKey::IO_START) + 1].Set(out_data1, nullptr);

  NodeMemPara node_para;
  RtFFTSKernelLaunchArgs::ComputeNodeDesc node_desc;
  node_desc.max_tiling_data = 63;
  node_desc.max_tail_tiling_data = 63;
  node_desc.addr_num = 1;
  node_desc.input_num = 1;
  node_desc.output_num = 1;
  node_desc.workspace_cap = 8;
  node_desc.thread_num_max = 16;
  node_desc.need_atomic = true;
  node_desc.max_atom_tiling_data = 64;
  node_desc.max_atom_tail_tiling_data = 64;
  size_t total_size = 0;
  ge::NodePtr tmp_node = std::make_shared<ge::Node>();
  auto rt_arg = RtFFTSKernelLaunchArgs::Create(tmp_node, node_desc, total_size);
  node_para.host_addr = rt_arg.get();
  char *mem_2 = new char[total_size];
  node_para.dev_addr = (void*)mem_2;
  AICoreSubTaskFlush flush_data;
  run_context.value_holder[static_cast<size_t>(AtomArgsInKey::IO_START) + 2].Set(&flush_data, nullptr);
  run_context.value_holder[static_cast<size_t>(AtomArgsInKey::ARGS_PARA)].Set(&node_para, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("FFTSUpdateAtomicArgs")->outputs_creator(nullptr, run_context),
  ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("FFTSUpdateAtomicArgs")->run_func(run_context),
      ge::GRAPH_SUCCESS);
  EXPECT_FALSE(registry.FindKernelFuncs("FFTSUpdateAtomicArgs")->trace_printer(run_context).empty());
  auto context = run_context.GetContext<KernelContext>();
  auto args_para = context->GetInputValue<NodeMemPara*>(static_cast<size_t>(AtomArgsInKey::ARGS_PARA));
  auto rt_args = reinterpret_cast<RtFFTSKernelLaunchArgs *>(args_para->host_addr);
  uintptr_t *args_base_addr = static_cast<uintptr_t*>(rt_args->GetArgBase());
  size_t args_pos = rt_args->GetAtomArgsPos();
  uintptr_t *args_host_data = &args_base_addr[args_pos];
  uintptr_t a = args_host_data[2];
  uintptr_t b = ge::PtrToValue(work_space_ptr[1]->Addr(0));
  ASSERT_EQ(a, b);

  proc_type = static_cast<uint32_t>(AtomProcType::DY_OP);
  run_context.value_holder[static_cast<size_t>(AtomArgsInKey::PROC_TYPE)].Set(reinterpret_cast<void *>(proc_type), nullptr);
  run_context.value_holder[static_cast<size_t>(AtomArgsInKey::WORK_ADDR)].Set(l1_work_space_vector, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("FFTSUpdateAtomicArgs")->run_func(run_context),
      ge::GRAPH_SUCCESS);
  a = args_host_data[2];
  b = ge::PtrToValue(l1_work_space_ptr[1]->GetAddr());
  ASSERT_EQ(a, b);

  for (size_t i = 0; i < work_space_vector->GetSize(); i++) {
    level_2_allocator->Free(work_space_ptr[i]);
  }
  level_2_allocator->Free(out_data1);
  level_2_allocator->Recycle();
  level_1_allocator->Finalize();
  delete[] mem_2;
}

TEST_F(FFTSAtomicKernelTestUT, test_atomic_update_context) {
  AICoreSubTaskFlush flush_data;
  flush_data.input_addr_vv[0][0] = 0x1000;
  flush_data.input_addr_vv[0][1] = 0x1001;
  flush_data.output_addr_vv[0][0] = 0x2000;
  flush_data.output_addr_vv[0][1] = 0x2001;
  flush_data.thread_dim = 2;
  flush_data.window_size = 2;

  // context id: 0~3
  auto ctx_ids = ContinuousVector::Create<int32_t>(4);
  auto ctx_ids_vec = reinterpret_cast<ContinuousVector *>(ctx_ids.get());
  ctx_ids_vec->SetSize(4);
  auto ctx_ids_ptr = reinterpret_cast<int32_t *>(ctx_ids_vec->MutableData());
  for (size_t i = 0; i < ctx_ids_vec->GetSize(); i++) {
    ctx_ids_ptr[i] = i;
  }

  auto run_context = KernelRunContextFaker().KernelIONum(static_cast<size_t>(AtomUpdateKey::RESERVED), 0).
      NodeIoNum(2,2).IrInputNum(2)
      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .Build();
  run_context.value_holder[static_cast<size_t>(AtomUpdateKey::FLUSH_DATA)].Set(&flush_data, nullptr);
  run_context.value_holder[static_cast<size_t>(AtomUpdateKey::AICORE_CTX)].Set(ctx_ids_vec, nullptr);
  uint32_t block_dim = 48;
  run_context.value_holder[static_cast<size_t>(AtomUpdateKey::BLOCK_DIM)].Set(reinterpret_cast<void *>(block_dim), nullptr);
  run_context.value_holder[static_cast<size_t>(AtomUpdateKey::TAIL_BLOCK_DIM)].Set(reinterpret_cast<void *>(block_dim), nullptr);

  size_t descBufLen = sizeof(rtFftsPlusComCtx_t) * static_cast<size_t>(4);
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
    context->contextType = RT_CTX_TYPE_AIV;
  }
  rtFftsPlusTaskInfo_t task_inf;
  auto *const ffts_plus_sqe = ge::PtrToPtr<uint8_t, rtFftsPlusSqe_t>(task_info_ptr->args);
  task_inf.fftsPlusSqe = ffts_plus_sqe;
  task_inf.descBuf = &task_info_ptr->args[buf_offset];
  ffts_plus_sqe->totalContextNum = 4;
  run_context.value_holder[static_cast<size_t>(AtomUpdateKey::TASK_INFO)].Set(&task_inf, nullptr);

  ASSERT_EQ(registry.FindKernelFuncs("AtomicUpdateContext")->run_func(run_context), ge::GRAPH_SUCCESS);

  ctx_ids_ptr[1] = 5;
  run_context.value_holder[static_cast<size_t>(AtomUpdateKey::AICORE_CTX)].Set(ctx_ids_vec, nullptr);
  ASSERT_NE(registry.FindKernelFuncs("AtomicUpdateContext")->run_func(run_context), ge::GRAPH_SUCCESS);

  // profiling
  ProfNodeAdditionInfo info;
  CannProfilingInfoWrapper prof_info(&info);
  ASSERT_EQ(registry.FindKernelFuncs("AtomicUpdateContext")->profiling_info_filler(run_context, prof_info),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(prof_info.add_infos_[static_cast<size_t>(NodeProfInfoType::kOriginalNode)]->
            node_basic_info.data.nodeBasicInfo.blockDim, 48);
}

TEST_F(FFTSAtomicKernelTestUT, test_calc_atomic_out_shape_size) {
auto run_context = KernelRunContextFaker().KernelIONum(2, 2).
        NodeIoNum(2,3).IrInputNum(2)
    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
    .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
    .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
    .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
    .Build();

  auto clear_index = ContinuousVector::Create<int64_t>(2);
  auto clear_index_vec = reinterpret_cast<ContinuousVector *>(clear_index.get());
  clear_index_vec->SetSize(2);
  auto clear_index_ptr = reinterpret_cast<int64_t *>(clear_index_vec->MutableData());
  clear_index_ptr[0] = 0;
  clear_index_ptr[1] = 2;

  auto slice_shape = ContinuousVector::Create<Shape>(3);
  auto slice_shape_vec = reinterpret_cast<ContinuousVector *>(slice_shape.get());
  slice_shape_vec->SetSize(3);
  auto slice_shape_ptr = reinterpret_cast<Shape *>(slice_shape_vec->MutableData());
  Shape shape({3, 2, 2});
  Shape shape1({4, 4, 5});
  slice_shape_ptr[0] = shape;
  slice_shape_ptr[1] = shape;
  slice_shape_ptr[2] = shape1;
  run_context.value_holder[0].Set(clear_index_vec, nullptr);
  run_context.value_holder[1].Set(slice_shape_vec, nullptr);

  ASSERT_EQ(registry.FindKernelFuncs("FFTSCalcAtomicOutputShapeSize")->run_func(run_context), ge::GRAPH_SUCCESS);
  auto context = run_context.GetContext<KernelContext>();
  auto tensor_size_ptr = context->GetOutputPointer<uint64_t>(0);
  ASSERT_EQ(*tensor_size_ptr, 64);
  tensor_size_ptr = context->GetOutputPointer<uint64_t>(1);
  ASSERT_EQ(*tensor_size_ptr, 352);
}
}  // namespace gert

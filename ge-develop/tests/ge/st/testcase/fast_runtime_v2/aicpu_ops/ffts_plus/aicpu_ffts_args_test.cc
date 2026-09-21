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
#include "faker/aicpu_ext_info_faker.h"
#include "stub/gert_runtime_stub.h"
#include "register/kernel_registry.h"
#include "common/plugin/ge_make_unique_util.h"
#include "common/math/math_util.h"
#include <kernel/memory/mem_block.h>
#include "aicpu_task_struct.h"
#include "engine/aicpu/kernel/ffts_plus/aicpu_ffts_args.h"
#include "engine/aicpu/kernel/ffts_plus/aicpu_update_kernel.h"
#include "engine/aicpu/converter/aicpu_ffts_node_converter.h"
#include "engine/ffts_plus/converter/ffts_plus_proto_transfer.h"
#include "register/ffts_node_calculater_registry.h"
#include "graph_builder/bg_memory.h"
#include "core/utils/tensor_utils.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "kernel/memory/ffts_mem_allocator.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "subscriber/dumper/executor_dumper.h"

namespace gert {
using namespace kernel;
namespace {
struct AicpuTaskStruct {
  aicpu::AicpuParamHead head;
  uint64_t io_addrp[6];
}__attribute__((packed));
} // namespace
class AicpuFFTSArgsST : public testing::Test {
 public:
  KernelRegistry &registry = KernelRegistry::GetInstance();
};

TEST_F(AicpuFFTSArgsST, test_ffts_update_aicpu_cc_args) { 
  auto ext_data = GetFakeExtInfo();
  auto ext_len = ext_data.size();
  AicpuTaskStruct args;
  args.head.length = sizeof(args);
  std::string node_name = "Add";
  auto unknown_shape_type = ge::DEPEND_COMPUTE;

  AICpuThreadParam thread_param;
  thread_param.input_num = 2;
  thread_param.output_num = 1;
  thread_param.thread_dim = 2;
  thread_param.input_output_num = 3;

  uint32_t in_shape_num = 2;
  auto in_shapes = ContinuousVector::Create<Shape>(in_shape_num);
  auto in_shapes_vec = reinterpret_cast<ContinuousVector*>(in_shapes.get());
  in_shapes_vec->SetSize(in_shape_num);
  auto in_shapes_data = static_cast<Shape*>(in_shapes_vec->MutableData());
  for (size_t i = 0; i < in_shape_num; i++) {
    in_shapes_data[i].SetDimNum(2);
    in_shapes_data[i].SetDim(0, 2);
    in_shapes_data[i].SetDim(1, 3);
  }

  uint32_t out_shape_num = 1;
  auto out_shapes = ContinuousVector::Create<Shape>(out_shape_num);
  auto out_shapes_vec = reinterpret_cast<ContinuousVector*>(out_shapes.get());
  out_shapes_vec->SetSize(out_shape_num);
  auto out_shapes_data = static_cast<Shape*>(out_shapes_vec->MutableData());
  for (size_t i = 0; i < out_shape_num; i++) {
    out_shapes_data[i].SetDimNum(2);
    out_shapes_data[i].SetDim(0, 2);
    out_shapes_data[i].SetDim(1, 3);
  }

  auto device_data = std::vector<int8_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  GertTensorData tensor_data = {(uint8_t *)device_data.data(), 0U, kTensorPlacementEnd, -1};
  auto device_data1 = std::vector<int8_t>{10};
  GertTensorData in_data = {(uint8_t *)device_data1.data(), 0U, kTensorPlacementEnd, -1};
  GertTensorData out_data = {(uint8_t *)device_data1.data(), 0U, kTensorPlacementEnd, -1};
  auto out_type = ContinuousVector::Create<uint32_t>(1);
  auto out_type_vec = reinterpret_cast<ContinuousVector *>(out_type.get());
  out_type_vec->SetSize(1);
  auto out_type_ptr = reinterpret_cast<uint32_t*>(out_type_vec->MutableData());
  out_type_ptr[0] = 0;

  auto in_type = ContinuousVector::Create<uint32_t>(2);
  auto in_type_vec = reinterpret_cast<ContinuousVector *>(in_type.get());
  in_type_vec->SetSize(2);
  auto in_type_ptr = reinterpret_cast<uint32_t*>(in_type_vec->MutableData());
  in_type_ptr[0] = 0;
  in_type_ptr[1] = 0;

  uint64_t session_id = 0;
  
  auto run_context = BuildKernelRunContext(18, 2);
  run_context.value_holder[0].Set(in_type_vec, nullptr);
  run_context.value_holder[1].Set(out_type_vec, nullptr);
  run_context.value_holder[2].Set(reinterpret_cast<void *>(session_id), nullptr);
  run_context.value_holder[3].Set(const_cast<char *>(ext_data.c_str()), nullptr);
  run_context.value_holder[4].Set(reinterpret_cast<void *>(ext_len), nullptr);
  run_context.value_holder[5].Set(&args, nullptr);
  run_context.value_holder[6].Set(reinterpret_cast<void *>(args.head.length), nullptr);
  run_context.value_holder[7].Set(const_cast<char *>(node_name.c_str()), nullptr);
  run_context.value_holder[8].Set(reinterpret_cast<void *>(unknown_shape_type), nullptr);
  run_context.value_holder[9].Set(&thread_param, nullptr);
  run_context.value_holder[10].Set(in_shapes.get(), nullptr);
  run_context.value_holder[11].Set(out_shapes.get(), nullptr);
  run_context.value_holder[12].Set(in_shapes.get(), nullptr);
  run_context.value_holder[13].Set(out_shapes.get(), nullptr);

  auto ctx_ids = ContinuousVector::Create<size_t>(5);
  auto ctx_ids_vec = reinterpret_cast<ContinuousVector *>(ctx_ids.get());
  ctx_ids_vec->SetSize(2);
  auto ctx_ids_ptr = reinterpret_cast<size_t *>(ctx_ids_vec->MutableData());
  for (size_t i = 0; i < ctx_ids_vec->GetSize(); ++i) {
    ctx_ids_ptr[i] = i + 10;
  }

  run_context.value_holder[14].Set(ctx_ids_vec, nullptr);
  run_context.value_holder[15].Set(&tensor_data, nullptr);
  run_context.value_holder[16].Set(&in_data, nullptr);
  run_context.value_holder[17].Set(&out_data, nullptr);

  AICpuSubTaskFlush flush_data;
  run_context.value_holder[18].Set(&flush_data, nullptr);
  NodeMemPara arg_addr;
  size_t size = 6144;
  arg_addr.size = size;
  char *mem  = new char[size];
  arg_addr.dev_addr = (void*)mem;
  auto holder = ge::MakeUnique<uint8_t[]>(size);
  arg_addr.host_addr = holder.get();

  run_context.value_holder[19].Set(&arg_addr, nullptr);

  ASSERT_EQ(registry.FindKernelFuncs("FFTSUpdateAICpuCCArgs")->outputs_creator(nullptr, run_context),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("FFTSUpdateAICpuCCArgs")->run_func(run_context), ge::GRAPH_SUCCESS);
  EXPECT_FALSE(registry.FindKernelFuncs("FFTSUpdateAICpuCCArgs")->trace_printer(run_context).empty());
  delete[] mem;

  NodeMemPara arg_addr_null;
  run_context.value_holder[19].Set(&arg_addr_null, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("FFTSUpdateAICpuCCArgs")->outputs_creator(nullptr, run_context),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("FFTSUpdateAICpuCCArgs")->run_func(run_context), ge::GRAPH_FAILED);
}

TEST_F(AicpuFFTSArgsST, test_ffts_update_aicpu_tf_args) { 
  auto ext_data = GetFakeExtInfo();
  auto ext_len = ext_data.size();
  AicpuTaskStruct args;
  args.head.length = sizeof(args);
  std::string node_name = "Add";
  auto unknown_shape_type = ge::DEPEND_COMPUTE;

  AICpuThreadParam thread_param;
  thread_param.input_num = 2;
  thread_param.output_num = 1;
  thread_param.thread_dim = 2;
  thread_param.input_output_num = 3;

  uint32_t in_shape_num = 2;
  auto in_shapes = ContinuousVector::Create<Shape>(in_shape_num);
  auto in_shapes_vec = reinterpret_cast<ContinuousVector*>(in_shapes.get());
  in_shapes_vec->SetSize(in_shape_num);
  auto in_shapes_data = static_cast<Shape*>(in_shapes_vec->MutableData());
  for (size_t i = 0; i < in_shape_num; i++) {
    in_shapes_data[i].SetDimNum(2);
    in_shapes_data[i].SetDim(0, 2);
    in_shapes_data[i].SetDim(1, 3);
  }

  uint32_t out_shape_num = 1;
  auto out_shapes = ContinuousVector::Create<Shape>(out_shape_num);
  auto out_shapes_vec = reinterpret_cast<ContinuousVector*>(out_shapes.get());
  out_shapes_vec->SetSize(out_shape_num);
  auto out_shapes_data = static_cast<Shape*>(out_shapes_vec->MutableData());
  for (size_t i = 0; i < out_shape_num; i++) {
    out_shapes_data[i].SetDimNum(2);
    out_shapes_data[i].SetDim(0, 2);
    out_shapes_data[i].SetDim(1, 3);
  }

  auto device_data = std::vector<int8_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  GertTensorData tensor_data = {(uint8_t *)device_data.data(), 0U, kTensorPlacementEnd, -1};
  auto device_data1 = std::vector<int8_t>{10};
  GertTensorData in_data = {(uint8_t *)device_data1.data(), 0U, kTensorPlacementEnd, -1};
  GertTensorData out_data = {(uint8_t *)device_data1.data(), 0U, kTensorPlacementEnd, -1};
  auto out_type = ContinuousVector::Create<uint32_t>(1);
  auto out_type_vec = reinterpret_cast<ContinuousVector *>(out_type.get());
  out_type_vec->SetSize(1);
  auto out_type_ptr = reinterpret_cast<uint32_t*>(out_type_vec->MutableData());
  out_type_ptr[0] = 0;

  auto in_type = ContinuousVector::Create<uint32_t>(2);
  auto in_type_vec = reinterpret_cast<ContinuousVector *>(in_type.get());
  in_type_vec->SetSize(2);
  auto in_type_ptr = reinterpret_cast<uint32_t*>(in_type_vec->MutableData());
  in_type_ptr[0] = 0;
  in_type_ptr[1] = 0;

  uint64_t session_id = 0;
  uint64_t step_id = 3;
  
  auto run_context = BuildKernelRunContext(19, 2);
  run_context.value_holder[0].Set(in_type_vec, nullptr);
  run_context.value_holder[1].Set(out_type_vec, nullptr);
  run_context.value_holder[2].Set(reinterpret_cast<void *>(session_id), nullptr);
  run_context.value_holder[3].Set(const_cast<char *>(ext_data.c_str()), nullptr);
  run_context.value_holder[4].Set(reinterpret_cast<void *>(ext_len), nullptr);
  run_context.value_holder[5].Set(&args, nullptr);
  run_context.value_holder[6].Set(reinterpret_cast<void *>(args.head.length), nullptr);
  run_context.value_holder[7].Set(const_cast<char *>(node_name.c_str()), nullptr);
  run_context.value_holder[8].Set(reinterpret_cast<void *>(unknown_shape_type), nullptr);
  run_context.value_holder[9].Set(&thread_param, nullptr);
  run_context.value_holder[10].Set(in_shapes.get(), nullptr);
  run_context.value_holder[11].Set(out_shapes.get(), nullptr);
  run_context.value_holder[12].Set(in_shapes.get(), nullptr);
  run_context.value_holder[13].Set(out_shapes.get(), nullptr);

  auto ctx_ids = ContinuousVector::Create<size_t>(5);
  auto ctx_ids_vec = reinterpret_cast<ContinuousVector *>(ctx_ids.get());
  ctx_ids_vec->SetSize(2);
  auto ctx_ids_ptr = reinterpret_cast<size_t *>(ctx_ids_vec->MutableData());
  for (size_t i = 0; i < ctx_ids_vec->GetSize(); ++i) {
    ctx_ids_ptr[i] = i + 10;
  }
  run_context.value_holder[14].Set(ctx_ids_vec, nullptr);

  run_context.value_holder[15].Set(&tensor_data, nullptr);
  run_context.value_holder[16].Set(&in_data, nullptr);
  run_context.value_holder[17].Set(&out_data, nullptr);
  run_context.value_holder[18].Set(reinterpret_cast<void *>(step_id), nullptr);

  AICpuSubTaskFlush flush_data;
  run_context.value_holder[19].Set(&flush_data, nullptr);
  NodeMemPara arg_addr;
  size_t size = 6144;
  arg_addr.size = size;
  char *mem  = new char[size];
  arg_addr.dev_addr = (void*)mem;
  auto holder = ge::MakeUnique<uint8_t[]>(size);
  arg_addr.host_addr = holder.get();

  run_context.value_holder[20].Set(&arg_addr, nullptr);

  ASSERT_EQ(registry.FindKernelFuncs("FFTSUpdateAICpuTfArgs")->outputs_creator(nullptr, run_context),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("FFTSUpdateAICpuTfArgs")->run_func(run_context), ge::GRAPH_SUCCESS);
  EXPECT_FALSE(registry.FindKernelFuncs("FFTSUpdateAICpuTfArgs")->trace_printer(run_context).empty());
  delete[] mem;

  NodeMemPara arg_addr_null;
  run_context.value_holder[20].Set(&arg_addr_null, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("FFTSUpdateAICpuTfArgs")->outputs_creator(nullptr, run_context),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("FFTSUpdateAICpuTfArgs")->run_func(run_context), ge::GRAPH_FAILED);
}

TEST_F(AicpuFFTSArgsST, test_init_aicpu_ffts_ctx_user) {
  auto ctx_ids = ContinuousVector::Create<size_t>(5);
  auto ctx_ids_vec = reinterpret_cast<ContinuousVector *>(ctx_ids.get());
  ctx_ids_vec->SetSize(2);
  auto ctx_ids_ptr = reinterpret_cast<size_t *>(ctx_ids_vec->MutableData());
  for (size_t i = 0; i < ctx_ids_vec->GetSize(); ++i) {
    ctx_ids_ptr[i] = i + 10;
  }

  std::string so_name = "libcpu_kernels.so";
  std::string kernel_name = "RunCpuKernel";
  const size_t so_name_len = so_name.size() + 1U;
  const size_t kernel_name_len = kernel_name.size() + 1U;
  GertTensorData so_name_dev;
  GertTensorData kernel_name_dev;
  rtStream_t stream = nullptr;
  
  auto run_context = BuildKernelRunContext(9, 1);
  run_context.value_holder[0].Set(ctx_ids_vec, nullptr);
  run_context.value_holder[1].Set(reinterpret_cast<void *>(0), nullptr);
  run_context.value_holder[2].Set(&so_name_dev, nullptr);
  run_context.value_holder[3].Set(reinterpret_cast<void *>(so_name_len), nullptr);
  run_context.value_holder[4].Set(const_cast<char *>(so_name.c_str()), nullptr);
  run_context.value_holder[5].Set(&kernel_name_dev, nullptr);
  run_context.value_holder[6].Set(reinterpret_cast<void *>(kernel_name_len), nullptr);
  run_context.value_holder[7].Set(const_cast<char *>(kernel_name.c_str()), nullptr);
  run_context.value_holder[8].Set(stream, nullptr);

  size_t descBufLen = sizeof(rtFftsPlusComCtx_t) * static_cast<size_t>(16);
  size_t total_size = sizeof(TransTaskInfo) + descBufLen + sizeof(rtFftsPlusSqe_t) + sizeof(rtFftsPlusComCtx_t);
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
  run_context.value_holder[9].Set(&task_info_ptr->rt_task_info, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("FFTSInitAicpuCtxUserData")->run_func(run_context), ge::GRAPH_SUCCESS);
}

TEST_F(AicpuFFTSArgsST, test_FillAICpuDataDumpInfo) {
  auto ext_data = GetFakeExtInfo();
  auto ext_len = ext_data.size();
  AicpuTaskStruct args;
  args.head.length = sizeof(args);
  std::string node_name = "Add";
  auto unknown_shape_type = ge::DEPEND_COMPUTE;

  AICpuThreadParam thread_param;
  thread_param.input_num = 2;
  thread_param.output_num = 1;
  thread_param.thread_dim = 2;
  thread_param.input_output_num = 3;

  uint32_t in_shape_num = 2;
  auto in_shapes = ContinuousVector::Create<Shape>(in_shape_num);
  auto in_shapes_vec = reinterpret_cast<ContinuousVector*>(in_shapes.get());
  in_shapes_vec->SetSize(in_shape_num);
  auto in_shapes_data = static_cast<Shape*>(in_shapes_vec->MutableData());
  for (size_t i = 0; i < in_shape_num; i++) {
    in_shapes_data[i].SetDimNum(2);
    in_shapes_data[i].SetDim(0, 2);
    in_shapes_data[i].SetDim(1, 3);
  }

  uint32_t out_shape_num = 1;
  auto out_shapes = ContinuousVector::Create<Shape>(out_shape_num);
  auto out_shapes_vec = reinterpret_cast<ContinuousVector*>(out_shapes.get());
  out_shapes_vec->SetSize(out_shape_num);
  auto out_shapes_data = static_cast<Shape*>(out_shapes_vec->MutableData());
  for (size_t i = 0; i < out_shape_num; i++) {
    out_shapes_data[i].SetDimNum(2);
    out_shapes_data[i].SetDim(0, 2);
    out_shapes_data[i].SetDim(1, 3);
  }

  auto device_data = std::vector<int8_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  GertTensorData tensor_data = {(uint8_t *)device_data.data(), 0U, kTensorPlacementEnd, -1};

  auto mem_block = ContinuousVector::Create<memory::FftsMemBlock *>(10);
  auto mem_block_vector = reinterpret_cast<ContinuousVector *>(mem_block.get());
  mem_block_vector->SetSize(10);
  auto mem_block_ptr = reinterpret_cast<memory::FftsMemBlock **>(mem_block_vector->MutableData());
  auto level_1_allocator = memory::CachingMemAllocator::GetAllocator();
  ASSERT_NE(level_1_allocator, nullptr);
  auto level_2_allocator = memory::FftsMemAllocator::GetAllocator(*level_1_allocator, 2U);
  ASSERT_NE(level_2_allocator, nullptr);
  for (size_t i = 0; i < mem_block_vector->GetSize(); i++) {
    mem_block_ptr[i] = level_2_allocator->Malloc(2);
  }

  auto device_data1 = std::vector<int8_t>{10};
  GertTensorData out_data = {(uint8_t *)device_data1.data(), 0U, kTensorPlacementEnd, -1};
  auto out_type = ContinuousVector::Create<uint32_t>(1);
  auto out_type_vec = reinterpret_cast<ContinuousVector *>(out_type.get());
  out_type_vec->SetSize(1);
  auto out_type_ptr = reinterpret_cast<uint32_t*>(out_type_vec->MutableData());
  out_type_ptr[0] = 0;

  auto in_type = ContinuousVector::Create<uint32_t>(2);
  auto in_type_vec = reinterpret_cast<ContinuousVector *>(in_type.get());
  in_type_vec->SetSize(2);
  auto in_type_ptr = reinterpret_cast<uint32_t*>(in_type_vec->MutableData());
  in_type_ptr[0] = 0;
  in_type_ptr[1] = 1;

  uint64_t session_id = 0;

  auto run_context = KernelRunContextFaker().KernelIONum(20, 2).NodeIoNum(2, 1).IrInputNum(2)
                                            .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                                            .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                                            .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                                            .Build();
  run_context.value_holder[0].Set(in_type_vec, nullptr);
  run_context.value_holder[1].Set(out_type_vec, nullptr);
  run_context.value_holder[2].Set(reinterpret_cast<void *>(session_id), nullptr);
  run_context.value_holder[3].Set(const_cast<char *>(ext_data.c_str()), nullptr);
  run_context.value_holder[4].Set(reinterpret_cast<void *>(ext_len), nullptr);
  run_context.value_holder[5].Set(&args, nullptr);
  run_context.value_holder[6].Set(reinterpret_cast<void *>(args.head.length), nullptr);
  run_context.value_holder[7].Set(const_cast<char *>(node_name.c_str()), nullptr);
  run_context.value_holder[8].Set(reinterpret_cast<void *>(unknown_shape_type), nullptr);
  run_context.value_holder[9].Set(&thread_param, nullptr);
  run_context.value_holder[10].Set(in_shapes.get(), nullptr);
  run_context.value_holder[11].Set(out_shapes.get(), nullptr);
  run_context.value_holder[12].Set(in_shapes.get(), nullptr);
  run_context.value_holder[13].Set(out_shapes.get(), nullptr);
  run_context.value_holder[15].Set(&tensor_data, nullptr);
  run_context.value_holder[16].Set(reinterpret_cast<void *>(mem_block_ptr[0]), nullptr);
  run_context.value_holder[17].Set(&out_data, nullptr);

  AICpuSubTaskFlush flush_data;
  run_context.value_holder[18].Set(&flush_data, nullptr);

  NodeDumpUnit dump_unit;
  gert::ExecutorDataDumpInfoWrapper wrapper(&dump_unit);
  ASSERT_NE(registry.FindKernelFuncs("FFTSUpdateAICpuCCArgs")->data_dump_info_filler(run_context, wrapper),
            ge::GRAPH_SUCCESS);

  auto ctx_ids = ContinuousVector::Create<size_t>(5);
  auto ctx_ids_vec = reinterpret_cast<ContinuousVector *>(ctx_ids.get());
  ctx_ids_vec->SetSize(2);
  auto ctx_ids_ptr = reinterpret_cast<size_t *>(ctx_ids_vec->MutableData());
  for (size_t i = 0; i < ctx_ids_vec->GetSize(); ++i) {
    ctx_ids_ptr[i] = i + 10;
  }
  run_context.value_holder[14].Set(ctx_ids_vec, nullptr);

  NodeDumpUnit dump_unit_suc;
  gert::ExecutorDataDumpInfoWrapper wrapper_suc(&dump_unit_suc);
  ASSERT_EQ(registry.FindKernelFuncs("FFTSUpdateAICpuCCArgs")->data_dump_info_filler(run_context, wrapper_suc),
            ge::GRAPH_SUCCESS);
  ASSERT_NE(registry.FindKernelFuncs("FFTSUpdateAICpuCCArgs")->data_dump_info_filler(run_context, wrapper_suc),
            ge::GRAPH_SUCCESS);

  for (size_t i = 0; i < mem_block_vector->GetSize(); i++) {
    level_2_allocator->Free(mem_block_ptr[i]);
  }
}
}  // namespace gert

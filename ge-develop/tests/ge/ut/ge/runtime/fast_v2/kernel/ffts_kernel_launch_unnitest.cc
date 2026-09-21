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
#include "stub/gert_runtime_stub.h"
#include "faker/kernel_run_context_facker.h"
#include "register/kernel_registry.h"
#include "graph/op_desc.h"
#include "mmpa/mmpa_api.h"
#include "core/debug/kernel_tracing.h"
#include "engine/aicore/fe_rt2_common.h"
#include "engine/ffts_plus/converter/ffts_plus_proto_transfer.h"
#include "register/ffts_node_calculater_registry.h"
#include "kernel/memory/mem_block.h"
#include "kernel/memory/multi_stream_mem_block.h"
namespace gert {

class FFTSKernelLaunchUT : public testing::Test {
 public:
  KernelRegistry &registry = KernelRegistry::GetInstance();
};

TEST_F(FFTSKernelLaunchUT, test_ffts_launch_task_no_copy) {
  rtStream_t stream_ = reinterpret_cast<void *>(0x12);
  NodeMemPara node_para;
  size_t descBufLen = sizeof(rtFftsPlusComCtx_t) * static_cast<size_t>(3);
  size_t total_size = sizeof(TransTaskInfo) + descBufLen + sizeof(rtFftsPlusSqe_t) + sizeof(rtFftsPlusComCtx_t);
  auto holder = ge::MakeUnique<uint8_t[]>(total_size);
  TransTaskInfo *pre_data_ptr = reinterpret_cast<TransTaskInfo*>(holder.get());
  size_t buf_offset = sizeof(rtFftsPlusSqe_t);
  pre_data_ptr->offsets[static_cast<size_t>(InfoStType::kDescBuf)] = buf_offset;
  pre_data_ptr->rt_task_info.descBufLen = descBufLen;
  pre_data_ptr->rt_task_info.descBuf = holder.get() + sizeof(TransTaskInfo);
  node_para.host_addr = pre_data_ptr;
  node_para.dev_addr = pre_data_ptr;
  auto context = BuildKernelRunContext(5, 0);
  context.value_holder[0].Set(reinterpret_cast<void *>(stream_), nullptr);
  context.value_holder[1].Set(&node_para, nullptr);
  uint32_t need_launch = 0;
  context.value_holder[2].Set(reinterpret_cast<void *>(need_launch), nullptr);

  gert::DfxExeArg exe_arg;
  exe_arg.need_assert = false;
  exe_arg.need_print = true;
  exe_arg.buffer_size = 1234;
  context.value_holder[3].Set(&exe_arg, nullptr);

  auto work_space = ContinuousVector::Create<memory::MultiStreamMemBlock *>(4);
  auto work_space_vector = reinterpret_cast<ContinuousVector *>(work_space.get());
  work_space_vector->SetSize(4);
  auto work_space_ptr = reinterpret_cast<memory::MultiStreamMemBlock **>(work_space_vector->MutableData());
  memory::MultiStreamMemBlock mem_block;
  for (size_t i = 0; i < work_space_vector->GetSize(); i++) {
    work_space_ptr[i] = &mem_block;
  }
  dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 0);
  context.value_holder[4].Set(work_space.get(), nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("LaunchFFTSPlusTaskNoCopy")->run_func(context), ge::GRAPH_SUCCESS);
  need_launch = 1;
  context.value_holder[2].Set(reinterpret_cast<void *>(need_launch), nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("LaunchFFTSPlusTaskNoCopy")->run_func(context), ge::GRAPH_SUCCESS);
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
}
}  // namespace gert
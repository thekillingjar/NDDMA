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
#include "faker/continuous_vector_builder.h"
#include "runtime/base.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "common/ge_inner_error_codes.h"
#include "register/kernel_registry.h"
#include "checker/memory_profiling_log_matcher.h"
#include "stub/gert_runtime_stub.h"

namespace gert {
static const int32_t END_OF_SEQUENCE = 507005; // success
class StreamKernelUT : public testing::Test {
 public:
  KernelRegistry &registry = KernelRegistry::GetInstance();
};

TEST_F(StreamKernelUT, TEST_STREAM_SUCCESS) {
  auto run_context = BuildKernelRunContext(1, 0);
  rtStream_t stream = nullptr;
  run_context.value_holder[0].Set(stream, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("SyncStream")->run_func(run_context), ge::GRAPH_SUCCESS);
}

TEST_F(StreamKernelUT, TEST_END_OF_SEQUENCE) {
  const char_t * const kEnvPath = "END_OF_SEQUENCE";
  char_t path[MMPA_MAX_PATH] = "end";
  mmSetEnv(kEnvPath, &path[0U], MMPA_MAX_PATH);

  auto run_context = BuildKernelRunContext(1, 0);
  rtStream_t stream = nullptr;
  run_context.value_holder[0].Set(stream, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("SyncStream")->run_func(run_context), ge::END_OF_SEQUENCE);
  unsetenv(kEnvPath);
}

TEST_F(StreamKernelUT, SplitRtStreams_Success_GetMainStream) {
  auto streams = ContinuousVectorBuilder::Create<rtStream_t>(
      {(rtStream_t)0x100, (rtStream_t)0x200, (rtStream_t)0x300, (rtStream_t)0x400});
  auto context = KernelRunContextFaker().Inputs({streams.get(), (void *)4}).Outputs({nullptr, nullptr, nullptr, nullptr}).Build();
  GertRuntimeStub runtime_stub;
  runtime_stub.GetSlogStub().NoConsoleOut().SetLevelInfo();
  ASSERT_EQ(registry.FindKernelFuncs("SplitRtStreams")->run_func(context), ge::GRAPH_SUCCESS);
  ASSERT_TRUE(runtime_stub.GetSlogStub().FindInfoLogRegex(kMirrorStream, {{2, ToHex(0x100)}, {6, "0"}}) >= 0);
  ASSERT_EQ(*context.GetContext<KernelContext>()->GetOutputPointer<rtStream_t>(0), (rtStream_t)0x100);
}
TEST_F(StreamKernelUT, SplitRtStreams_Success_GetSecondaryStream) {
  auto streams = ContinuousVectorBuilder::Create<rtStream_t>(
      {(rtStream_t)0x100, (rtStream_t)0x200, (rtStream_t)0x300, (rtStream_t)0x400});
  auto context = KernelRunContextFaker().Inputs({streams.get(), (void *)4}).Outputs({nullptr, nullptr, nullptr, nullptr}).Build();
  ASSERT_EQ(registry.FindKernelFuncs("SplitRtStreams")->run_func(context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(*context.GetContext<KernelContext>()->GetOutputPointer<rtStream_t>(2), (rtStream_t)0x300);
}
TEST_F(StreamKernelUT, SplitRtStreams_Failed_InvalidStreamNum) {
  auto streams = ContinuousVectorBuilder::Create<rtStream_t>(
      {(rtStream_t)0x100, (rtStream_t)0x200, (rtStream_t)0x300, (rtStream_t)0x400});
  auto context = KernelRunContextFaker().Inputs({streams.get(), (void *)5}).Outputs({nullptr, nullptr, nullptr, nullptr}).Build();
  ASSERT_NE(registry.FindKernelFuncs("SplitRtStreams")->run_func(context), ge::GRAPH_SUCCESS);

  context = KernelRunContextFaker().Inputs({streams.get(), (void *)3}).Outputs({nullptr, nullptr}).Build();
  ASSERT_NE(registry.FindKernelFuncs("SplitRtStreams")->run_func(context), ge::GRAPH_SUCCESS);
}
TEST_F(StreamKernelUT, SplitRtStreams_Failed_InvalidStreams) {
  auto context = KernelRunContextFaker().Inputs({ nullptr, (void *)2}).Outputs({nullptr, nullptr}).Build();
  ASSERT_NE(registry.FindKernelFuncs("SplitRtStreams")->run_func(context), ge::GRAPH_SUCCESS);
}
} // namespace gert
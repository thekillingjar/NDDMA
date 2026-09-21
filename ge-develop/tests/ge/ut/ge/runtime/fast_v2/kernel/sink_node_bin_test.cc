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
#include "kernel/common_kernel_impl/sink_node_bin.h"
#include "stub/gert_runtime_stub.h"
#include "kernel/common_kernel_impl/sink_node_bin.h"
#include "engine/aicore/kernel/aicore_update_kernel.h"

namespace gert {
namespace kernel {
ge::graphStatus SinkNodeBinWithHandle(KernelContext *context);
}
struct SinkNodeBinTest : public testing::Test {
  KernelRegistry &registry = KernelRegistry::GetInstance();
};

TEST_F(SinkNodeBinTest, test_sink_node_bin_with_hanle_success) {
  kernel::BinData bin_data;
  const char *bin_key1 = "key1";
  const char *bin_key2 = "key2";
  const char *empty_key = "";

  struct FakeRuntime : RuntimeStubImpl {
    rtError_t rtRegisterAllKernel(const rtDevBinary_t *bin, void **handle) {
      static size_t registed_num = 0x10;
      *handle = (void *)(registed_num++);
      return RT_ERROR_NONE;
    }
  };
  GertRuntimeStub runtime(std::unique_ptr<RuntimeStubImpl>(new FakeRuntime()));

  auto run_context = BuildKernelRunContext(2, 1);
  run_context.value_holder[0].Set((void *)&bin_data, nullptr);
  run_context.value_holder[1].Set((void *)bin_key1, nullptr);
  ASSERT_EQ(kernel::SinkNodeBinWithHandle(run_context.GetContext<KernelContext>()), ge::GRAPH_SUCCESS);
  ASSERT_EQ(run_context.value_holder[2].GetValue<uint64_t>(), 0x10);

  run_context = BuildKernelRunContext(2, 1);
  run_context.value_holder[0].Set((void *)&bin_data, nullptr);
  run_context.value_holder[1].Set((void *)bin_key2, nullptr);
  ASSERT_EQ(kernel::SinkNodeBinWithHandle(run_context.GetContext<KernelContext>()), ge::GRAPH_SUCCESS);
  ASSERT_EQ(run_context.value_holder[2].GetValue<uint64_t>(), 0x10 + 1);

  run_context = BuildKernelRunContext(2, 1);
  run_context.value_holder[0].Set((void *)&bin_data, nullptr);
  run_context.value_holder[1].Set((void *)bin_key1, nullptr);
  ASSERT_EQ(kernel::SinkNodeBinWithHandle(run_context.GetContext<KernelContext>()), ge::GRAPH_SUCCESS);
  ASSERT_EQ(run_context.value_holder[2].GetValue<uint64_t>(), 0x10); // cached

  run_context = BuildKernelRunContext(2, 1);
  run_context.value_holder[0].Set((void *)&bin_data, nullptr);
  run_context.value_holder[1].Set((void *)empty_key, nullptr);
  ASSERT_EQ(kernel::SinkNodeBinWithHandle(run_context.GetContext<KernelContext>()), ge::GRAPH_SUCCESS);
  ASSERT_EQ(run_context.value_holder[2].GetValue<uint64_t>(), 0x10 + 2);

  run_context = BuildKernelRunContext(2, 1);
  run_context.value_holder[0].Set((void *)&bin_data, nullptr);
  run_context.value_holder[1].Set((void *)empty_key, nullptr);
  ASSERT_EQ(kernel::SinkNodeBinWithHandle(run_context.GetContext<KernelContext>()), ge::GRAPH_SUCCESS);
  ASSERT_EQ(run_context.value_holder[2].GetValue<uint64_t>(), 0x10 + 3);
}

TEST_F(SinkNodeBinTest, test_sink_node_bin_with_out_hanle_success) {
  kernel::BinData bin_data;

  RTS_STUB_RETURN_VALUE(rtQueryFunctionRegistered, rtError_t, 0x1);
  auto run_context = BuildKernelRunContext(4, 1);
  run_context.value_holder[0].Set(&bin_data, nullptr);
  run_context.value_holder[1].Set((void *)"stub", nullptr);
  run_context.value_holder[2].Set((void *)"metadata", nullptr);
  run_context.value_holder[3].Set((void *)"kernenname", nullptr);

  struct FakeRuntime : RuntimeStubImpl {
    rtError_t rtDevBinaryRegister(const rtDevBinary_t *bin, void **handle) {
      *handle = reinterpret_cast<void *>(0x12345678);
      return RT_ERROR_NONE;
    }
    rtError_t rtGetFunctionByName(const char *stub_name, void **stub_func) {
      *stub_func = reinterpret_cast<void *>(0x20);
      return RT_ERROR_NONE;
    }
  };
  GertRuntimeStub runtime(std::unique_ptr<RuntimeStubImpl>(new FakeRuntime()));

  ASSERT_EQ(registry.FindKernelFuncs("SinkNodeBinWithoutHandle")->run_func(run_context.GetContext<KernelContext>()),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(run_context.value_holder[4].GetValue<uint64_t>(), 0x20);
}

TEST_F(SinkNodeBinTest, test_sink_node_bin_with_out_handle_failed) {
  auto run_context = BuildKernelRunContext(4, 1);
  RTS_STUB_RETURN_VALUE(rtQueryFunctionRegistered, rtError_t, 0x1);
  ASSERT_NE(registry.FindKernelFuncs("SinkNodeBinWithoutHandle")->run_func(run_context), ge::GRAPH_SUCCESS);
}

TEST_F(SinkNodeBinTest, test_sink_node_bin_with_handle_failed) {
  auto run_context = BuildKernelRunContext(4, 1);
  ASSERT_NE(registry.FindKernelFuncs("SinkNodeBinWithHandle")->run_func(run_context), ge::GRAPH_SUCCESS);
}

TEST_F(SinkNodeBinTest, test_ffts_sink_node_bin) {
  auto run_context = BuildKernelRunContext(1, 1);
  kernel::BinData bin_data;
  run_context.value_holder[0].Set(&bin_data, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("SinkFFTSAICoreNodeBin")->run_func(run_context), ge::GRAPH_SUCCESS);
  run_context.value_holder[0].Set(nullptr, nullptr);
  ASSERT_NE(registry.FindKernelFuncs("SinkFFTSAICoreNodeBin")->run_func(run_context), ge::GRAPH_SUCCESS);

  auto run_context1 = BuildKernelRunContext(3, 1);
  uint32_t bin_handle = 2;
  uint32_t tiling_key = 32;
  run_context1.value_holder[0].Set(&bin_handle, nullptr);
  run_context1.value_holder[1].Set(&tiling_key, nullptr);
  run_context1.value_holder[2].Set(&tiling_key, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("GetFFTSAICorePcAndPref")->outputs_creator(nullptr, run_context1),
  ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("GetFFTSAICorePcAndPref")->run_func(run_context1), ge::GRAPH_SUCCESS);
}
TEST_F(SinkNodeBinTest, test_ffts_static_sink_node_bin) {
  auto run_context = BuildKernelRunContext(4, 1);
  kernel::BinData bin_data;
  run_context.value_holder[0].Set((void *)"stub", nullptr);
  run_context.value_holder[1].Set((void *)"stub", nullptr);
  run_context.value_holder[2].Set((void *)"stub", nullptr);
  run_context.value_holder[3].Set(&bin_data, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("SinkFFTSStaManualNodeBin")->outputs_creator(nullptr, run_context),
    ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("SinkFFTSStaManualNodeBin")->run_func(run_context), ge::GRAPH_SUCCESS);
}

TEST_F(SinkNodeBinTest, test_ffts_auto_static_sink_node_bin) {
  size_t kernel_num = 2;
  auto run_context = BuildKernelRunContext(9, 1);
  kernel::BinData bin_data;
  run_context.value_holder[0].Set(reinterpret_cast<void *>(kernel_num), nullptr);
  run_context.value_holder[1].Set((void *)"stub", nullptr);
  run_context.value_holder[2].Set((void *)"stub", nullptr);
  run_context.value_holder[3].Set((void *)"stub", nullptr);
  run_context.value_holder[4].Set(&bin_data, nullptr);
  run_context.value_holder[5].Set((void *)"stub", nullptr);
  run_context.value_holder[6].Set((void *)"stub", nullptr);
  run_context.value_holder[7].Set((void *)"stub", nullptr);
  run_context.value_holder[8].Set(&bin_data, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("SinkFFTSStaAutoNodeBin")->outputs_creator(nullptr, run_context),
  ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("SinkFFTSStaAutoNodeBin")->run_func(run_context), ge::GRAPH_SUCCESS);
}

TEST_F(SinkNodeBinTest, test_sink_mix_dy_node_bin) {
  auto run_context = BuildKernelRunContext(2, 1);
  kernel::BinData bin_data;
  run_context.value_holder[0].Set(&bin_data, nullptr);
  run_context.value_holder[1].Set((void *)"kernel_bin_id", nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("SinkMixDyNodeBin")->run_func(run_context), ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("SinkMixDyNodeBin")->run_func(run_context), ge::GRAPH_SUCCESS);
  run_context.value_holder[1].Set((void *)"", nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("SinkMixDyNodeBin")->run_func(run_context), ge::GRAPH_SUCCESS);
  run_context.value_holder[0].Set(nullptr, nullptr);
  ASSERT_NE(registry.FindKernelFuncs("SinkMixDyNodeBin")->run_func(run_context), ge::GRAPH_SUCCESS);
}
TEST_F(SinkNodeBinTest, test_get_mix_dy_node_pc) {
  auto run_context = BuildKernelRunContext(6, 1);
  size_t kernel_num = 1;
  uint32_t tail_key = 2;
  uint32_t kernel_type = 1;
  ASSERT_NE(registry.FindKernelFuncs("GetMixAddrAndPrefCnt")->run_func(run_context), ge::GRAPH_SUCCESS);
  run_context.value_holder[0].Set(reinterpret_cast<void *>(kernel_num), nullptr);
  run_context.value_holder[1].Set(reinterpret_cast<void *>(kernel_type), nullptr);
  run_context.value_holder[2].Set(reinterpret_cast<void *>(tail_key), nullptr);
  run_context.value_holder[3].Set(reinterpret_cast<void *>(tail_key), nullptr);
  run_context.value_holder[4].Set((void*)"kernel_num", nullptr);
  run_context.value_holder[5].Set(reinterpret_cast<void *>(tail_key), nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("GetMixAddrAndPrefCnt")->outputs_creator(nullptr, run_context),
  ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("GetMixAddrAndPrefCnt")->run_func(run_context), ge::GRAPH_SUCCESS);
}
TEST_F(SinkNodeBinTest, test_sink_mix_static_node_bin) {
  auto run_context = BuildKernelRunContext(7, 1);
  kernel::BinData bin_data;
  size_t num = 1;
  uint32_t kernel_type = 0;
  run_context.value_holder[0].Set(reinterpret_cast<void *>(num), nullptr);
  run_context.value_holder[1].Set(reinterpret_cast<void *>(kernel_type), nullptr);
  run_context.value_holder[2].Set((void *)"stub", nullptr);
  run_context.value_holder[3].Set((void *)"stub", nullptr);
  run_context.value_holder[4].Set((void *)"stub", nullptr);
  run_context.value_holder[5].Set((void *)"stub", nullptr);
  run_context.value_holder[6].Set(&bin_data, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("SinkMixStaticNodeBin")->outputs_creator(nullptr, run_context),
  ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("SinkMixStaticNodeBin")->run_func(run_context), ge::GRAPH_SUCCESS);
}
TEST_F(SinkNodeBinTest, test_sink_enhanced_dy_mix) {
  auto run_context = BuildKernelRunContext(7, 1);
  kernel::BinData bin_data;
  size_t num = 1;
  uint32_t kernel_type = 3;
  run_context.value_holder[0].Set(reinterpret_cast<void *>(num), nullptr);
  run_context.value_holder[1].Set(reinterpret_cast<void *>(kernel_type), nullptr);
  run_context.value_holder[2].Set((void *)"stub", nullptr);
  run_context.value_holder[3].Set((void *)"stub", nullptr);
  run_context.value_holder[4].Set((void *)"stub", nullptr);
  run_context.value_holder[5].Set((void *)"stub", nullptr);
  run_context.value_holder[6].Set(&bin_data, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("SinkMixStaticNodeBin")->outputs_creator(nullptr, run_context),
  ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("SinkMixStaticNodeBin")->run_func(run_context), ge::GRAPH_SUCCESS);
}
TEST_F(SinkNodeBinTest, test_parse_kernel_info) {
  gert::kernel::AICoreSinkRet sink_ret;
  rtKernelDetailInfo_t rt_kernel_info_1;
  rt_kernel_info_1.functionInfoNum = 2;
  rt_kernel_info_1.functionInfo[0].pcAddr = &rt_kernel_info_1;
  rt_kernel_info_1.functionInfo[0].prefetchCnt = 10;
  rt_kernel_info_1.functionInfo[0].mixType = 3;
  rt_kernel_info_1.functionInfo[1].pcAddr = &rt_kernel_info_1;
  rt_kernel_info_1.functionInfo[1].prefetchCnt = 10;
  rt_kernel_info_1.functionInfo[1].mixType = 3;
  EXPECT_EQ(kernel::ParseKernelInfo(rt_kernel_info_1, rt_kernel_info_1, &sink_ret), ge::SUCCESS);
  EXPECT_EQ(sink_ret.aic_icache_prefetch_cnt, 10);
  EXPECT_EQ(sink_ret.aiv_icache_prefetch_cnt, 10);
  rtKernelDetailInfo_t rt_kernel_info_2;
  rt_kernel_info_2.functionInfoNum = 1;
  rt_kernel_info_2.functionInfo[0].pcAddr = &rt_kernel_info_2;
  rt_kernel_info_2.functionInfo[0].prefetchCnt = 10;
  rt_kernel_info_2.functionInfo[0].mixType = 2;

  EXPECT_EQ(kernel::ParseKernelInfo(rt_kernel_info_2, rt_kernel_info_2, &sink_ret), ge::SUCCESS);
  EXPECT_EQ(sink_ret.aiv_icache_prefetch_cnt, 10);

  rtKernelDetailInfo_t rt_kernel_info_3;
  rt_kernel_info_3.functionInfoNum = 1;
  rt_kernel_info_3.functionInfo[0].pcAddr = &rt_kernel_info_2;
  rt_kernel_info_3.functionInfo[0].prefetchCnt = 10;
  rt_kernel_info_3.functionInfo[0].mixType = 10;
  EXPECT_EQ(kernel::ParseKernelInfo(rt_kernel_info_3, rt_kernel_info_3, &sink_ret), ge::SUCCESS);

  rtKernelDetailInfo_t rt_kernel_info_4;
  rt_kernel_info_4.functionInfoNum = 3;
  EXPECT_EQ(kernel::CheckKernelInfoValid(rt_kernel_info_4), false);
}
TEST_F(SinkNodeBinTest, test_get_kernel_type) {
  gert::kernel::AICoreSinkRet sink_ret;
  gert::kernel::KernelFunctionCtx ctx_1(0, RT_DYNAMIC_SHAPE_KERNEL, 0, 0, (void*)"stub", nullptr);
  auto ret = kernel::GetKernelFuncInfoByKernelType(ctx_1, "mix_aic", &sink_ret);
  EXPECT_EQ(ret, ge::SUCCESS);

  ret = kernel::GetKernelFuncInfoByKernelType(ctx_1, "mix_aiv", &sink_ret);
  EXPECT_EQ(ret, ge::SUCCESS);
}
}  // namespace gert
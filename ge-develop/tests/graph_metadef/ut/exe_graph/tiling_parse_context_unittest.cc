/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/runtime/tiling_parse_context.h"
#include <gtest/gtest.h>
#include <vector>
#include "faker/kernel_run_context_faker.h"
#include "platform/platform_info.h"
namespace gert {
class TilingParseContextUT : public testing::Test {};
struct CompiledInfo1 {
  uint64_t a;
  uint64_t b;
};
struct CompileDInfo2 {
  uint32_t a;
  uint32_t b;
};
struct CompiledInfo3 {
  int32_t core_num;
};

TEST_F(TilingParseContextUT, GetIoOk) {
  char *json_str = const_cast<char *>("{}");
  CompiledInfo1 ci = {10, 20};
  auto context_holder = KernelRunContextFaker().KernelIONum(1, 1).Inputs({json_str}).Outputs({&ci}).Build();
  auto context = context_holder.GetContext<TilingParseContext>();
  ASSERT_NE(context, nullptr);
  EXPECT_STREQ(context->GetCompiledJson(), "{}");
  ASSERT_NE(context->GetCompiledInfo<CompiledInfo1>(), nullptr);
  EXPECT_EQ(context->GetCompiledInfo<CompiledInfo1>()->a, 10);
  EXPECT_EQ(context->GetCompiledInfo<CompiledInfo1>()->b, 20);
}
TEST_F(TilingParseContextUT, SetCompiledInfoOk) {
  char *json_str = const_cast<char *>("{}");
  fe::PlatFormInfos platform_info;
  auto context_holder = KernelRunContextFaker().KernelIONum(1, 1)
          .Inputs({json_str, reinterpret_cast<char *>(&platform_info)})
          .Outputs({nullptr}).Build();
  auto context = context_holder.GetContext<TilingParseContext>();
  EXPECT_EQ(context->GetPlatformInfo()->GetCoreNum(), 8);
}

TEST_F(TilingParseContextUT, CompiledInfoLessThan8Bytes) {
  char *json_str = const_cast<char *>("{}");
  CompiledInfo3 ci = {2};
  auto context_holder = KernelRunContextFaker().KernelIONum(1, 1).Inputs({json_str}).Outputs({&ci}).Build();
  auto context = context_holder.GetContext<TilingParseContext>();
  ASSERT_NE(context, nullptr);
  EXPECT_STREQ(context->GetCompiledJson(), "{}");
  ASSERT_NE(context->GetCompiledInfo<CompiledInfo3>(), nullptr);
  EXPECT_EQ(context->GetCompiledInfo<CompiledInfo3>()->core_num, 2);
}
}  // namespace gert

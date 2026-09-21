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
#include "framework/runtime/model_desc.h"
#include "register/kernel_registry.h"

namespace gert {
class RtModelDescUT : public testing::Test {
 public:
  KernelRegistry &registry = KernelRegistry::GetInstance();
};

TEST_F(RtModelDescUT, GetFileConstantWeightDir_GetFromModelDescOK) {
  auto run_context = BuildKernelRunContext(1, 1);
  auto kernel_funcs = registry.FindKernelFuncs("GetFileConstantWeightDir");
  ASSERT_NE(kernel_funcs, nullptr);
  size_t total_size = sizeof(ModelDesc) + 0 * sizeof(ModelIoDesc);
  auto model_desc_holder = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[total_size]);
  auto &model_desc = *(reinterpret_cast<ModelDesc *>(model_desc_holder.get()));
  std::string file_constant_weight_dir("/home/bob/model/weight/");
  model_desc.SetFileConstantWeightDir(file_constant_weight_dir.c_str());

  run_context.value_holder[0].Set(&model_desc, nullptr);

  ASSERT_EQ(kernel_funcs->run_func(run_context), ge::GRAPH_SUCCESS);
  auto file_constant_weight_dir_ptr = run_context.GetContext<KernelContext>()->GetOutputPointer<const ge::char_t *>(0);
  ASSERT_NE(file_constant_weight_dir_ptr, nullptr);
  ASSERT_EQ(std::string(*file_constant_weight_dir_ptr), file_constant_weight_dir);
}

TEST_F(RtModelDescUT, GetOmPath_GetFailed_ModelDescIsNullptr) {
  auto run_context = BuildKernelRunContext(1, 1);
  auto kernel_funcs = registry.FindKernelFuncs("GetFileConstantWeightDir");
  ASSERT_NE(kernel_funcs, nullptr);

  run_context.value_holder[0].Set(nullptr, nullptr);
  EXPECT_NE(kernel_funcs->run_func(run_context), ge::GRAPH_SUCCESS);
  auto file_constant_weight_dir_ptr = run_context.GetContext<KernelContext>()->GetOutputPointer<const ge::char_t *>(0);
  EXPECT_EQ(*file_constant_weight_dir_ptr, nullptr);
}
}  // namespace gert

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
#include "faker/kernel_outputs_faker.h"
#include "faker/node_faker.h"
#include "register/kernel_registry.h"
#include "kernel/tensor_attr.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "faker/fake_value.h"
#include "kernel/ge_local_kernel/rank_kernel.h"

namespace gert {
namespace {
const std::string kKernelName = "RankKernel";
const size_t kOutputRankSize = 1U;
}
struct RankKernelUt : public testing::Test {
  KernelRegistry &registry = KernelRegistry::GetInstance();
};

TEST_F(RankKernelUt, RankKernelInt32Ok) {
  StorageShape shape = {{2,2,3}, {2,2,3}};
  Shape kOutputData{3};
  Shape output_shape{};

  auto context_holder = KernelRunContextFaker()
                            .KernelIONum(static_cast<size_t>(kernel::RankKernelInputs::kEnd),
                                         static_cast<size_t>(kernel::RankKernelOutputs::kEnd))
                            .NodeIoNum(1U, 1U)
                            .Inputs({&shape})
                            .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                            .Build();
  auto context = context_holder.GetContext<KernelContext>();

  auto funcs = registry.FindKernelFuncs(kKernelName);
  ASSERT_NE(funcs, nullptr);
  FastNodeFaker faker;
  auto node = faker.Build();
  (void)ge::AttrUtils::SetInt(node->GetOpDescBarePtr(), "dtype", ge::DT_INT32);
  ASSERT_EQ(funcs->outputs_creator(node, context), ge::GRAPH_SUCCESS);
  EXPECT_EQ(funcs->run_func(context), ge::GRAPH_SUCCESS);

  auto td = context->GetOutputPointer<GertTensorData>(static_cast<size_t>(0));
  ASSERT_NE(td, nullptr);
  EXPECT_EQ(td->GetPlacement(), kOnHost);
  EXPECT_EQ(td->GetSize(), kOutputRankSize * sizeof(int32_t));
  auto shape_data = static_cast<const int32_t *>(td->GetAddr());
  EXPECT_EQ(shape_data[0], kOutputData[0]);

  context_holder.FreeAll();
}

}  // namespace gert
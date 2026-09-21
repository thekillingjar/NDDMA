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
#include "faker/fake_value.h"
#include "task.pb.h"
#include "runtime/mem.h"
#include "common/runtime_api_wrapper.h"

namespace gert {
namespace kernel {}  // namespace kernel

struct CmoLaunchKernelUt : public testing::Test {
  KernelRegistry &registry = KernelRegistry::GetInstance();
};

TEST_F(CmoLaunchKernelUt, LaunchPrefetchSuccess) {
  uint8_t src[1024];
  GertTensorData in_data = {(uint8_t *)src, 0U, kTensorPlacementEnd, -1};
  uint32_t op_code = 6U;
  auto faker = FastNodeFaker();
  auto node = faker.Build();
  ge::AttrUtils::SetInt(node->GetOpDescBarePtr(), "type", op_code);

  auto init_func = registry.FindKernelFuncs("InitCmoArgs");
  ASSERT_NE(init_func, nullptr);
  auto init_ctx_holder =
      KernelRunContextFaker().KernelIONum(1U, 1U).Inputs({reinterpret_cast<void *>(op_code)}).Build();
  auto init_ctx = init_ctx_holder.GetContext<KernelContext>();
  EXPECT_EQ(init_func->outputs_creator(node, init_ctx), ge::GRAPH_SUCCESS);
  EXPECT_EQ(init_func->run_func(init_ctx_holder), ge::GRAPH_SUCCESS);

  auto task_info = init_ctx->GetOutputPointer<rtCmoTaskInfo_t>(0UL);
  EXPECT_NE(task_info, nullptr);
  EXPECT_EQ(task_info->opCode, op_code);

  StorageShape storage_shape({1, 1024, 1024, 20}, {1, 2024, 1024, 20});
  ge::DataType dt = ge::DT_FLOAT;
  uint32_t max_size = 120 * 1024 * 1024;
  int64_t offset = 1024;

  auto update_func = registry.FindKernelFuncs("UpdatePrefetchTaskInfo");
  ASSERT_NE(update_func, nullptr);
  auto update_holder = KernelRunContextFaker()
                           .KernelIONum(6U, 0U)
                           .Inputs({task_info, &in_data, &storage_shape, reinterpret_cast<void *>(dt),
                                    reinterpret_cast<void *>(max_size), reinterpret_cast<void *>(offset)})
                           .Build();
  auto update_context = update_holder.GetContext<KernelContext>();
  EXPECT_EQ(update_func->run_func(update_context), ge::GRAPH_SUCCESS);

  void *stream;
  auto launch_func = registry.FindKernelFuncs("LaunchCmoTask");
  ASSERT_NE(launch_func, nullptr);
  auto context_holder = KernelRunContextFaker().KernelIONum(2U, 0U).Inputs({task_info, &stream}).Build();
  auto context = context_holder.GetContext<KernelContext>();
  EXPECT_EQ(launch_func->run_func(context), ge::GRAPH_SUCCESS);
  EXPECT_FALSE(launch_func->trace_printer(context).empty());
  EXPECT_EQ(task_info->lengthInner, max_size);
  EXPECT_EQ(task_info->sourceAddr, reinterpret_cast<uint64_t>(src) + offset);
}

TEST_F(CmoLaunchKernelUt, UpdatePrefetchOffsetOverflow) {
  uint8_t src[1024];
  GertTensorData in_data = {(uint8_t *)src, 0U, kTensorPlacementEnd, -1};
  uint32_t op_code = 6U;
  auto faker = FastNodeFaker();
  auto node = faker.Build();
  ge::AttrUtils::SetInt(node->GetOpDescBarePtr(), "type", op_code);

  auto init_func = registry.FindKernelFuncs("InitCmoArgs");
  ASSERT_NE(init_func, nullptr);
  auto init_ctx_holder =
      KernelRunContextFaker().KernelIONum(1U, 1U).Inputs({reinterpret_cast<void *>(op_code)}).Build();
  auto init_ctx = init_ctx_holder.GetContext<KernelContext>();
  EXPECT_EQ(init_func->outputs_creator(node, init_ctx), ge::GRAPH_SUCCESS);
  EXPECT_EQ(init_func->run_func(init_ctx_holder), ge::GRAPH_SUCCESS);

  auto task_info = init_ctx->GetOutputPointer<rtCmoTaskInfo_t>(0UL);
  EXPECT_NE(task_info, nullptr);
  EXPECT_EQ(task_info->opCode, op_code);

  StorageShape storage_shape({1, 1, 1, 20}, {1, 1, 1, 20});
  ge::DataType dt = ge::DT_FLOAT;
  uint32_t max_size = 120 * 1024 * 1024;
  int64_t offset = 1024;

  auto update_func = registry.FindKernelFuncs("UpdatePrefetchTaskInfo");
  ASSERT_NE(update_func, nullptr);
  auto update_holder = KernelRunContextFaker()
                           .KernelIONum(6U, 0U)
                           .Inputs({task_info, &in_data, &storage_shape, reinterpret_cast<void *>(dt),
                                    reinterpret_cast<void *>(max_size), reinterpret_cast<void *>(offset)})
                           .Build();
  auto update_context = update_holder.GetContext<KernelContext>();
  EXPECT_NE(update_func->run_func(update_context), ge::GRAPH_SUCCESS);
}
}  // namespace gert

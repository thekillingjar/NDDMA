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
#include "kernel/ge_local_kernel/multi_batch_shape_data_kernel.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "common/plugin/ge_make_unique_util.h"

namespace gert {
namespace {
std::unique_ptr<uint8_t[]> GetContinuousVector2DByVector2D(const std::vector<std::vector<int32_t>> &vector_2d,
                                                           size_t &total_size) {
  total_size = ContinuousVectorVector::GetOverHeadLength(vector_2d.size());
  for (const auto &inner_vec : vector_2d) {
    size_t inner_vec_length = 0U;
    if (ge::MulOverflow(inner_vec.size(), sizeof(int32_t), inner_vec_length)) {
        return nullptr;
    }
    if (ge::AddOverflow(inner_vec_length, sizeof(ContinuousVector), inner_vec_length)) {
        return nullptr;
    }
    if (ge::AddOverflow(total_size, inner_vec_length, total_size)) {
        return nullptr;
    }
  }
  auto holder = ge::MakeUnique<uint8_t[]>(total_size);
  auto cvv = new (holder.get()) ContinuousVectorVector();
  cvv->Init(vector_2d.size());

  for (const auto &inner_vec : vector_2d) {
    auto cv = cvv->Add<int32_t>(inner_vec.size());
    if (!inner_vec.empty()) {
      const size_t copy_size = inner_vec.size() * sizeof(int32_t);
      memcpy_s(cv->MutableData(), cv->GetCapacity() * sizeof(int32_t), inner_vec.data(), copy_size);
    }
  }
  return holder;
}
}
  struct GetCurDynamicShapeKernelUt : public testing::Test {
    KernelRegistry &registry = KernelRegistry::GetInstance();
  };
  TEST_F(GetCurDynamicShapeKernelUt, GetCurDynamicShapeKernelExecuteOk) {
    StorageShape shape = {{10,20,30}, {10,20,30}};
    StorageShape shape1 = {{5}, {5}};
    std::vector<int32_t> kOutputData{10, 30, 20, 20, 30};
    std::vector<std::vector<int32_t>> shape_index = {{0, 2}, {1}, {1, 2}};
    size_t cvv_total_size = 0U;
    auto const_index_data = GetContinuousVector2DByVector2D(shape_index, cvv_total_size);
    auto context_holder = KernelRunContextFaker()
        .KernelIONum(5U, 1U)
        .Inputs({&shape1, reinterpret_cast<void *>(const_index_data.get()), &shape, &shape, &shape})
        .Build();
    auto context = context_holder.GetContext<KernelContext>();

    auto funcs = registry.FindKernelFuncs("GetCurDynamicShape");
    ASSERT_NE(funcs, nullptr);
    auto node = FastNodeFaker().Build();
    ASSERT_EQ(funcs->outputs_creator(node, context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(funcs->run_func(context), ge::GRAPH_SUCCESS);
    funcs->trace_printer(context);
    auto td = context->GetOutputPointer<GertTensorData>(static_cast<size_t>(0));
    ASSERT_NE(td, nullptr);
    EXPECT_EQ(td->GetPlacement(), kOnHost);
    EXPECT_EQ(td->GetSize(), 5 * sizeof(int32_t));
    auto shape_data = static_cast<const int32_t *>(td->GetAddr());
    for (size_t i = 0UL; i < 5UL; i++) {
      EXPECT_EQ(shape_data[i], kOutputData[i]);
    }
    context_holder.FreeAll();
  }
}
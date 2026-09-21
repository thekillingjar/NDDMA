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
#include <kernel/memory/mem_block.h>
#include "faker/kernel_run_context_facker.h"
#include "faker/aicpu_ext_info_faker.h"
#include "register/kernel_registry_impl.h"
#include "aicpu_engine_struct.h"
#include "graph/op_desc.h"
#include "graph/def_types.h"
#include "graph/compute_graph.h"
#include "exe_graph/runtime/gert_tensor_data.h"

namespace gert {
class AicpuMemcpyUT : public testing::Test {
 public:
  KernelRegistryImpl &registry = KernelRegistryImpl::GetInstance();
};

TEST_F(AicpuMemcpyUT, test_parse_and_update) {
  auto prepare_copy_inputs_context = BuildKernelRunContext(8, 0);
  size_t output_num = 1U;
  aicpu::FWKAdapter::ResultSummary summary;
  auto temp_addr = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[20]);
  GertTensorData temp_buffer = {temp_addr.get(), 0U, kTensorPlacementEnd, -1};

  prepare_copy_inputs_context.value_holder[0].Set(reinterpret_cast<void *>(output_num), nullptr);
  prepare_copy_inputs_context.value_holder[1].Set(&temp_buffer, nullptr);
  prepare_copy_inputs_context.value_holder[2].Set(&temp_buffer, nullptr);
  prepare_copy_inputs_context.value_holder[3].Set(&temp_buffer, nullptr);
  prepare_copy_inputs_context.value_holder[4].Set(&temp_buffer, nullptr);
  prepare_copy_inputs_context.value_holder[5].Set(&summary, nullptr);
  prepare_copy_inputs_context.value_holder[6].Set(&temp_buffer, nullptr);
  prepare_copy_inputs_context.value_holder[7].Set(&temp_buffer, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("PrepareCopyInputs")->run_func(prepare_copy_inputs_context), ge::GRAPH_SUCCESS);

  auto get_output_shape_ctx = BuildKernelRunContext(2, 1);
  summary.shape_data_size = 8;
  get_output_shape_ctx.value_holder[0].Set(&summary, nullptr);
  get_output_shape_ctx.value_holder[1].Set(&temp_buffer, nullptr);

  ASSERT_EQ(registry.FindKernelFuncs("GetOutputShapeFromHbmBuffer")->outputs_creator(nullptr, get_output_shape_ctx),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("GetOutputShapeFromHbmBuffer")->run_func(get_output_shape_ctx),
            ge::GRAPH_SUCCESS);

  auto get_host_summary_ctx = BuildKernelRunContext(2, 1);
  temp_buffer = {&summary, 0U, kTensorPlacementEnd, -1};
  get_host_summary_ctx.value_holder[0].Set(&temp_buffer, nullptr);
  get_host_summary_ctx.value_holder[1].Set(&summary, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("GetHostSummary")->outputs_creator(nullptr, get_host_summary_ctx),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("GetHostSummary")->run_func(get_host_summary_ctx), ge::GRAPH_SUCCESS);

  auto get_summary_data_sizes_ctx = BuildKernelRunContext(1, 1);
  get_summary_data_sizes_ctx.value_holder[0].Set(&summary, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("GetSummaryDataSizes")->run_func(get_summary_data_sizes_ctx), ge::GRAPH_SUCCESS);

  auto get_summary_shape_sizes_ctx = BuildKernelRunContext(1, 1);
  get_summary_shape_sizes_ctx.value_holder[0].Set(&summary, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("GetSummaryShapeSizes")->run_func(get_summary_shape_sizes_ctx), ge::GRAPH_SUCCESS);

  get_output_shape_ctx.FreeValue(2);
  get_host_summary_ctx.FreeValue(2);
}
}  // namespace gert

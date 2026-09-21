/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_fuse_host_inputs.h"
#include "common/checker.h"
#include "graph/ge_error_codes.h"
#include "register/kernel_registry.h"
#include "exe_graph/runtime/tensor_data.h"
#include "exe_graph/runtime/storage_shape.h"
#include "common/table_driven.h"
#include "core/debug/kernel_tracing.h"
#include "kernel/kernel_log.h"
#include "kernel/memory/mem_block.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "exe_graph/runtime/gert_mem_allocator.h"
#include "kernel/common_kernel_impl/memory_copy.h"
#include "graph/utils/type_utils.h"
#include "engine/aicpu/kernel/aicpu_args_handler.h"
#include "graph/utils/math_util.h"
#include "common/plugin/ge_make_unique_util.h"
#include "core/executor/multi_thread_topological/executor/schedule/producer/producers/kernel_tags/critical_section_config.h"
#include "core/utils/tensor_utils.h"
#include "exe_graph/runtime/gert_tensor_data.h"

namespace gert {
namespace kernel {
namespace {
constexpr size_t kMaxTotalHostLen = 128U;

ge::graphStatus CopyTensorToDevice(KernelContext *context, const GertTensorData &src_tensor, const size_t src_len,
                                   const size_t dst_len, GertTensorData &dst_tensor) {
  auto stream = context->GetInputValue<rtStream_t>(static_cast<size_t>(AicpuFuseHostInputs::kStream));
  auto gert_allocator = context->MutableInputPointer<GertAllocator>(
      static_cast<size_t>(AicpuFuseHostInputs::kAllocator));
  GE_ASSERT_NOTNULL(gert_allocator);

  auto mem_block = reinterpret_cast<memory::MultiStreamMemBlock *>(gert_allocator->Malloc(dst_len));
  KERNEL_CHECK_NOTNULL(mem_block);
  KERNEL_CHECK(mem_block->GetAddr() != nullptr, "malloc failed, tensor size=%zu", dst_len);
  KERNEL_TRACE_ALLOC_MEM(gert_allocator->GetStreamId(), mem_block, mem_block->GetAddr(), mem_block->GetSize());
  dst_tensor = TensorUtils::ToGertTensorData(mem_block, gert_allocator->GetPlacement(), gert_allocator->GetStreamId());

  if (src_len > 0U) {
    GE_ASSERT_RT_OK(rtMemcpyAsync(mem_block->GetAddr(), dst_len, src_tensor.GetAddr(), src_len,
                                  RT_MEMCPY_HOST_TO_DEVICE_EX, stream));
  }
  dst_tensor.SetPlacement(kOnDeviceHbm);
  return ge::GRAPH_SUCCESS;
}
}  // namespace
ge::graphStatus CreateOutputForAicpuFuseHost(const ge::FastNode *node, KernelContext *context) {
  (void) node;
  for (size_t i = 0U; i < context->GetOutputNum(); ++i) {
    auto chain = context->GetOutput(i);
    auto tensor_data = ge::MakeUnique<GertTensorData>(0, kOnHost, -1, nullptr);
    GE_ASSERT_NOTNULL(chain);
    GE_ASSERT_NOTNULL(tensor_data);
    chain->SetWithDefaultDeleter(tensor_data.release());
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus AicpuFuseHost(KernelContext *context) {
  const size_t handle_idx = static_cast<size_t>(AicpuFuseHostInputs::kArgsHandler);
  auto args_handle = context->MutableInputPointer<AicpuArgsHandler>(handle_idx);
  auto input_indexes = context->GetInputValue<int32_t *>(static_cast<size_t>(AicpuFuseHostInputs::kInputsIndex));
  GE_ASSERT_NOTNULL(args_handle);
  GE_ASSERT_NOTNULL(input_indexes);

  const size_t output_num = context->GetOutputNum();
  args_handle->ResetHostInputInfo();
  for (size_t i = 0U; i < output_num; ++i) {
    const auto addr_index = static_cast<size_t>(AicpuFuseHostInputs::kAddrAndLengthStart) +  i * kSizeOfCopyToDevice;
    const auto tensor_data = context->GetInputPointer<GertTensorData>(addr_index);
    const auto out_tensor_data = context->GetOutputPointer<GertTensorData>(i);
    GE_ASSERT_NOTNULL(tensor_data);
    GE_ASSERT_NOTNULL(out_tensor_data);

    if (TensorPlacementUtils::IsOnDevice(tensor_data->GetPlacement())) {
      out_tensor_data->ShareFrom(*tensor_data);
    } else if (TensorPlacementUtils::IsOnHost(tensor_data->GetPlacement())) {
      const auto tensor_size = context->GetInputValue<size_t>(addr_index + 1U); // tensor size offset
      const auto storage_shape = context->GetInputPointer<StorageShape>(addr_index + 2U); // shape offset
      const auto data_type = context->GetInputValue<ge::DataType>(addr_index + 3U); // dtype offset
      GE_ASSERT_NOTNULL(storage_shape);

      const auto &src_shape = storage_shape->GetStorageShape();
      const auto host_tensor_size = ge::GetSizeInBytes(src_shape.GetShapeSize(), data_type);
      GE_ASSERT_TRUE(host_tensor_size >= 0);
      const size_t align_size = ge::RoundUp(static_cast<uint64_t>(host_tensor_size),
                                            args_handle->GetInputAddrAlignBytes());
      if (args_handle->GetHostInputSize() + align_size <= kMaxTotalHostLen) {
        out_tensor_data->SetPlacement(kOnHost);
        GE_ASSERT_SUCCESS(args_handle->AddHostInput(*(input_indexes + i), tensor_data->GetAddr(), host_tensor_size,
                                                    align_size));
      } else {
        GELOGD("Total host memory input size lengther then range is %zu, no need optimize", kMaxTotalHostLen);
        GE_ASSERT_SUCCESS(CopyTensorToDevice(context, *tensor_data, static_cast<size_t>(host_tensor_size), tensor_size,
                                             *out_tensor_data));
      }
    }
  }

  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(AicpuFuseHost)
    .RunFunc(AicpuFuseHost)
    .OutputsCreator(CreateOutputForAicpuFuseHost)
    .ConcurrentCriticalSectionKey(kKernelUseMemory);
}  // namespace kernel
}  // namespace gert

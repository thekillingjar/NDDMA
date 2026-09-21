/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/checker.h"
#include "graph/load/model_manager/reusable_stream_allocator.h"
#include "register/kernel_registry.h"

namespace gert {
namespace kernel {
ge::graphStatus ReusableStreamAllocator(KernelContext *context) {
  (void)context;
  return ge::SUCCESS;
}
ge::graphStatus BuildReusableStreamAllocatorOutputs(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  auto reusable_stream_allocator_chain = context->GetOutput(0);
  const auto reusable_stream_allocator = ge::ReusableStreamAllocator::Create();
  GE_ASSERT_NOTNULL(reusable_stream_allocator);
  GELOGD("Create reusable_stream_allocator which expects to reuse streams among all davinci models");
  GE_ASSERT_NOTNULL(reusable_stream_allocator_chain);
  reusable_stream_allocator_chain->SetWithDefaultDeleter(reusable_stream_allocator);

  return ge::SUCCESS;
}
REGISTER_KERNEL(ReusableStreamAllocator)
    .RunFunc(ReusableStreamAllocator)
    .OutputsCreator(BuildReusableStreamAllocatorOutputs);
}  // namespace kernel
}  // namespace gert

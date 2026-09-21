/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bg_reusable_stream_allocator.h"

#include "exe_graph/lowering/frame_selector.h"

namespace gert {
namespace bg {
namespace {
const std::string kReusableStreamAllocator = "ReusableStreamAllocator";
}  // namespace
ValueHolderPtr ReusableStreamAllocator(LoweringGlobalData &global_data) {
  auto builder = []() -> std::vector<bg::ValueHolderPtr> {
    return bg::FrameSelector::OnInitRoot([]() -> std::vector<bg::ValueHolderPtr> {
      return bg::ValueHolder::CreateDataOutput(kReusableStreamAllocator.c_str(), {}, 1U);
    });
  };
  const auto &allocator = global_data.GetOrCreateUniqueValueHolder(kReusableStreamAllocator, builder);
  GE_ASSERT_TRUE(!allocator.empty());
  return allocator[0];
}
}  // namespace bg
}  // namespace gert

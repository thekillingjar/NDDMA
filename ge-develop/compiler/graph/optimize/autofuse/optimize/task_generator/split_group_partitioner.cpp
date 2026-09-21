/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "optimize/task_generator/split_group_partitioner.h"

#include <queue>

#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "graph/utils/graph_utils.h"

#include "ascir_ops.h"
#include "platform/platform_factory.h"

namespace optimize {
namespace {
constexpr uint32_t kMaxOutputNum = 48U;
constexpr int32_t kAlignment = 32;
constexpr int32_t kMinSizeForSmallTail = kAlignment * 2;
}  // namespace

SplitGroupPartitioner::SplitGroupPartitioner(ge::AscNodePtr split_node, size_t split_dim)
    : split_node_(std::move(split_node)), split_dim_(split_dim) {}

Status SplitGroupPartitioner::Initialize() {
  constexpr uint32_t kLargeOutputNum = 512;
  // 防止group过小
  const auto &output_attr = split_node_->outputs[0].attr;
  dtype_size_ = ge::GetSizeByDataType(output_attr.dtype);
  GE_ASSERT_TRUE(dtype_size_ > 0, "unsupported dtype: %d, size = %ld", static_cast<int32_t>(output_attr.dtype),
                 dtype_size_);
  auto platform = PlatformFactory::GetInstance().GetPlatform();
  GE_ASSERT_NOTNULL(platform);
  group_type_to_limit_[kGroupTypeDefault] = kMaxBlockSize / dtype_size_;
  group_type_to_limit_[kGroupTypeAligned] = kMaxBlockSize / dtype_size_;
  group_type_to_limit_[kGroupTypeSmallTail] = kMaxBlockSizeForSmallTail;
  const uint32_t num_outputs = split_node_->outputs().size();
  const uint32_t kMinGroupNum = 2U;
  max_output_num_per_group_ = num_outputs / kMinGroupNum;
  if (num_outputs % kMinGroupNum != 0) {
    max_output_num_per_group_ += 1;
  }
  const uint32_t max_output_num = num_outputs >= kLargeOutputNum ? kMaxOutputNum : 16U;
  max_output_num_per_group_ = std::min(max_output_num_per_group_, max_output_num);
  GELOGI("input_num = %u, max_output_num_per_group_ = %u", num_outputs, max_output_num_per_group_);
  return ge::SUCCESS;
}

Status SplitGroupPartitioner::PartitionGroups([[maybe_unused]]const std::vector<SplitGroup> &groups) {
  GE_ASSERT_SUCCESS(Initialize());
  return ge::SUCCESS;
}

}  // namespace optimize
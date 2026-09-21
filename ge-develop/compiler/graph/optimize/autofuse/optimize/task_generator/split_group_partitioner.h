/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCGEN_DEV_OPTIMIZE_TASK_GENERATOR_SPLIT_GROUP_PARTITIONER_H_
#define ASCGEN_DEV_OPTIMIZE_TASK_GENERATOR_SPLIT_GROUP_PARTITIONER_H_

#include "ascgen_log.h"
#include "ascir_ops.h"

namespace optimize {
class SplitGroupPartitioner {
 public:
  struct SplitGroup {
    size_t start;
    size_t end;
    int32_t group_type;
    int64_t size;
  };

  SplitGroupPartitioner(ge::AscNodePtr split_node, size_t split_dim);

  Status PartitionGroups(const std::vector<SplitGroup> &groups);

 private:
  Status Initialize();
  static constexpr int32_t kGroupTypeDefault = 0;
  static constexpr int32_t kGroupTypeAligned = 1;
  static constexpr int32_t kGroupTypeSmallTail = 2;
  static constexpr int32_t kGroupTypeNone = 3;
  static constexpr int64_t kGroupEltSizeThreshold = 1024 * 4;  // 单个足够大, 不要重复搬运
  static constexpr int64_t kMaxBlockSize = 8192;
  static constexpr int64_t kMaxBlockSizeForSmallTail = 96;
  static constexpr int32_t kSmallTailLimit = 4;

  ge::AscNodePtr split_node_;
  size_t split_dim_;
  uint32_t max_output_num_per_group_ = 0U;
  std::vector<SplitGroup> groups_;
  int32_t group_type_ = -1;
  int64_t index_start_ = -1;
  size_t last_aligned_index_ = 0U;
  int64_t last_aligned_size_ = -1;
  int64_t cur_size_ = 0;
  int64_t dtype_size_ = 0;
  int64_t stride_ = 1;
  bool can_use_small_tail_ = false;
  std::map<std::string, std::set<size_t>> root_to_in_;
  std::set<size_t> multi_ref_input_indices_;
  std::set<std::string> consumed_;
  std::map<int32_t, int64_t> group_type_to_limit_;
};
}  // namespace optimize

#endif  // ASCGEN_DEV_OPTIMIZE_TASK_GENERATOR_SPLIT_GROUP_PARTITIONER_H_

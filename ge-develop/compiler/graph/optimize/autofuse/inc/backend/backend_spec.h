/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_BACKEND_SPEC_H
#define AUTOFUSE_BACKEND_SPEC_H
#include <cstdint>
#include <memory>

namespace optimize {

struct GatherSpec {
  bool enable_non_tail_gather;
  bool enable_reduce_gather_fusion;
  bool enable_gather_concat_fusion;
  bool enable_gather_broadcast_fusion;
  bool enable_gather_elementwise_forward_fusion;
};

struct PgoSpec {
  bool need_ffts;
};

struct SliceSplitSpec {
  bool split_lowered_to_split; // false: A2/A3环境下将Split Lowering成多个StridedSlice; true: A5环境将Split Lowering成多个Split
  bool slice_fuse_with_end_dim_1; // false: A2/A3环境Slice尾轴为1不融合; true: A5环境Slice尾轴为1可以融
  bool enable_split_flatten; // false: A2/A3环境暂不使能flatten; true: A5环境使能flatten
};

enum class TransposeMode : uint32_t {
  TRANSPOSE_MODE_NORMAL = 0,
  TRANSPOSE_MODE_UNNORMAL = 1,
};

struct BackendSpec {
  static std::unique_ptr<BackendSpec> GetInstance();
  uint32_t concat_max_input_num;
  int32_t concat_alg; // 0: by transpose
  GatherSpec gather_spec;
  SliceSplitSpec slice_split_spec;
  uint32_t max_load_num;
  uint32_t max_input_nums_after_fuse = 8U; // 限制融合后的单个节点输入个数最大值：A2A3=8，A5=14
  uint32_t transpose_mode; // 0: normal模式; 1: 非normal模式 A5
  uint32_t set_local_memory_size = 8 * 1024;
  uint32_t max_group_num_per_compile_unit = 5;
  PgoSpec pgo_spec;
  bool enable_matmul_lowering_to_matmul; // 限制A2A3 不能lowering出matmul，不限制matmul lowering成别的ascir类型
};
}  // namespace optimize

#endif  // AUTOFUSE_BACKEND_SPEC_H
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_GEMM_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_GEMM_H_

#include <cstdint>
#include <vector>
#include <map>
#include <string>
#include "cache_tiling.h"
#include "framework/common/debug/ge_log.h"
#include "graph/ge_error_codes.h"

namespace gert {

const int32_t kBlockSize = 16;

// TODO: BatchmatmulParas调整顺序
struct OpRunInfoParas {
  optiling::BatchmatmulParas params;
  int32_t batch_single_core = 1;
  int32_t m_single_core = 1;
  int32_t n_single_core = 1;
  int32_t batch_dim = 1;
  int32_t n_dim = 1;
  int32_t m_dim = 1;
  int32_t k_dim = 1;
  int32_t m_al1 = 1;
  int32_t n_bl1 = 1;
  int32_t cub_n1 = 1;
  int32_t m_l0 = 1;
  int32_t n_l0 = 1;
  int32_t k_l0 = 1;
  int32_t n_ub_l0_time = 1;
  int32_t kal0_factor = 1;
  int32_t kbl0_factor = 1;
  int32_t kal1_factor = 1;
  int32_t kbl1_factor = 1;
  int32_t kal1_16 = 1;
  int32_t kbl1_16 = 1;
  int32_t k_al1 = kal1_16 * kBlockSize;
  int32_t k_bl1 = kbl1_16 * kBlockSize;
  int32_t kl1_times = 1;
  int32_t m_aub = 1;
  int32_t n_bub = 1;
  int32_t k_aub = kBlockSize;
  int32_t k_bub = kBlockSize;
  int32_t multi_n_ub_l1 = 1;
  int32_t multi_m_ub_l1 = 1;
  int32_t multi_k_aub_l1 = 1;
  int32_t multi_k_bub_l1 = 1;
  int32_t a_align_value = 1;
  int32_t b_align_value = 1;
  int32_t aub_align_bound = 0;
  int32_t bub_align_bound = 0;
};

enum DynamicMode {
  DYNAMIC_MKN,
  DYNAMIC_MKNB
};

struct GemmCompileInfo {
  bool trans_a;
  bool trans_b;
  bool repo_seed_flag;
  bool repo_costmodel_flag;
  uint32_t workspace_num = 0;
  uint32_t ub_size = 0;
  optiling::BatchmatmulParas params;
  DynamicMode dynamic_mode;
  std::vector<std::vector<int64_t>> repo_seeds;
  std::vector<std::vector<int64_t>> repo_range;
  std::vector<std::vector<int64_t>> cost_range;
  std::vector<uint64_t> repo_tiling_ids;
  std::vector<uint64_t> cost_tiling_ids;
  std::map<uint64_t, uint32_t> block_dim;
};

class TilingContext;
class KernelContext;
class Shape;
ge::graphStatus TilingForGemm(TilingContext *context);
ge::graphStatus TilingPrepareForGemm(KernelContext *context);
void InferComplementedOutput(bool shape_x1_reshape_flag, bool shape_x2_reshape_flag, Shape& shape_out);

}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_GEMM_H_

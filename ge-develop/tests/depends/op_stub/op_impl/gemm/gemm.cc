/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gemm.h"

#include <nlohmann/json.hpp>
#include <sstream>

#include "exe_graph/runtime/continuous_vector.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_context.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace {
const int64_t kIdxMLow = 0;
const int64_t kIdxMHigh = 1;
const int64_t kIdxKLow = 2;
const int64_t kIdxKHigh = 3;
const int64_t kIdxNLow = 4;
const int64_t kIdxNHigh = 5;
const int64_t kIdxBLow = 6;
const int64_t kIdxBHigh = 7;
const int64_t kIdxM = 0;
const int64_t kIdxK = 1;
const int64_t kIdxN = 2;
const int64_t kIdxBatch = 3;
const int64_t kBlockIn = 16;
const int64_t kBlockReduce = 16;
const int64_t kBlockReduceS8 = 32;
const int64_t kBlockOut = 16;
const int64_t kMinBatchDimNum = 3;
const int64_t kNumTwo = 2;
const int64_t kMinDimNum = 3;
const int64_t KMULTI = 4;
const uint64_t kInvalidTilingId = UINT64_MAX;
const uint64_t kAlignedFlag = 2;
const uint64_t kMinCacheTilingId = 1000000ULL;
const uint64_t kMaxCacheTilingId = 9999999999ULL;

#define CHECK(cond, log_func, expr)                                                                                    \
  do {                                                                                                                 \
    if (cond) {                                                                                                        \
      log_func;                                                                                                        \
      expr;                                                                                                            \
    }                                                                                                                  \
  } while (0)

#define CUBE_INNER_ERR_REPORT(op_name, err_msg, ...)                                                                   \
  do {                                                                                                                 \
    GELOGE(ge::GRAPH_FAILED, err_msg, ##__VA_ARGS__);                                                                  \
  } while (0)

#define OP_LOGD(op_name, err_msg, ...)                                                                                 \
  do {                                                                                                                 \
    GELOGD(err_msg, ##__VA_ARGS__);                                                                                    \
  } while (0)

using nlohmann::json;
using optiling::BatchmatmulParas;
using std::map;
using std::string;
using std::vector;
}  // namespace

namespace gert {

void InferComplementedOutput(bool shape_x1_reshape_flag, bool shape_x2_reshape_flag, Shape &shape_out) {
  size_t dim_num = shape_out.GetDimNum();
  if (dim_num >= 2) {
    if (shape_x1_reshape_flag && !shape_x2_reshape_flag) {
      shape_out.SetDim(dim_num - 2, shape_out.GetDim(dim_num - 1));
      shape_out.SetDimNum(dim_num - 1);
    }

    if (!shape_x1_reshape_flag && shape_x2_reshape_flag) {
      shape_out.SetDimNum(dim_num - 1);
    }
  }
}

bool GetGEMMBatch(const char *op_type, const Shape &shape_a, const Shape &shape_b, BatchmatmulParas &params) {
  // TODO: int32_t -> size_t
  int32_t num_dima = shape_a.GetDimNum();
  int32_t num_dimb = shape_b.GetDimNum();
  if (num_dima < kMinDimNum && num_dimb < kMinDimNum) {
    params.batch = 1;
    params.batch_32 = 1;
    return true;
  }

  params.b_have_batch = num_dimb > kNumTwo;

  const auto &shape_short = num_dima < num_dimb ? shape_a : shape_b;
  const auto &shape_long = num_dima < num_dimb ? shape_b : shape_a;
  auto shape_broadcast(shape_long);  // TODO: temp shape
  int32_t num_dim = shape_long.GetDimNum();
  int32_t offset = num_dim - shape_short.GetDimNum();

  for (int32_t i = num_dim - kMinBatchDimNum; i >= offset; --i) {
    int64_t short_value = shape_short.GetDim(i - offset);
    int64_t long_value = shape_long.GetDim(i);
    CHECK((short_value != long_value && short_value != 1 && long_value != 1),
          CUBE_INNER_ERR_REPORT(op_type, "Tensor a and b do not meet the broadcast rule"), return false);

    shape_broadcast.SetDim(i, std::max(short_value, long_value));
  }

  int64_t batch_value = 1;
  for (int32_t i = 0; i < num_dim - kNumTwo; ++i) {
    batch_value *= shape_broadcast.GetDim(i);
  }
  params.batch = batch_value;
  CHECK((batch_value > INT_MAX),
        CUBE_INNER_ERR_REPORT(op_type, "The batch of a and b tensors' shape must not larger than INT_MAX"),
        return false);
  params.batch_32 = batch_value;
  return true;
}

bool CalcGEMMMknb(const char *op_type, ge::DataType dtype, const Shape &ori_shape_a, const Shape &ori_shape_b,
                  GemmCompileInfo &compile_value) {
  int32_t block_reduce = kBlockReduce, block_in = kBlockIn, block_out = kBlockOut;
  bool is_int8_type = dtype == ge::DT_INT8 || dtype == ge::DT_UINT8;
  if (is_int8_type) {
    block_reduce = kBlockReduceS8;
  }

  int32_t num_dima = ori_shape_a.GetDimNum();
  int32_t num_dimb = ori_shape_b.GetDimNum();
  int32_t idx_m_of_a = num_dima - kNumTwo;
  int32_t idx_k_of_a = num_dima - 1;
  int32_t idx_k_of_b = num_dimb - kNumTwo;
  int32_t idx_n_of_b = num_dimb - 1;
  // TODO: 应该用context传过来的属性
  if (compile_value.trans_a) {
    idx_m_of_a = num_dima - 1;
    idx_k_of_a = num_dima - kNumTwo;
  }
  if (compile_value.trans_b) {
    idx_k_of_b = num_dimb - 1;
    idx_n_of_b = num_dimb - kNumTwo;
  }

  CHECK((ori_shape_a.GetDim(idx_k_of_a) != ori_shape_b.GetDim(idx_k_of_b)),
        CUBE_INNER_ERR_REPORT(op_type, "The k-axis of a and b tensors must be the same"), return false);

  compile_value.params.m = ceil(static_cast<double>(ori_shape_a.GetDim(idx_m_of_a)) / block_in);
  compile_value.params.k = ceil(static_cast<double>(ori_shape_a.GetDim(idx_k_of_a)) / block_reduce);
  compile_value.params.n = ceil(static_cast<double>(ori_shape_b.GetDim(idx_n_of_b)) / block_out);
  bool unvalid_dim =
      compile_value.params.m > INT_MAX || compile_value.params.k > INT_MAX || compile_value.params.n > INT_MAX;
  CHECK(unvalid_dim, CUBE_INNER_ERR_REPORT(op_type, "The m,k,n of a and b tensors' shape must not larger than INT_MAX"),
        return false);
  compile_value.params.m_32 = static_cast<int32_t>(compile_value.params.m);
  compile_value.params.k_32 = static_cast<int32_t>(compile_value.params.k);
  compile_value.params.n_32 = static_cast<int32_t>(compile_value.params.n);

  if (compile_value.params.format_a_nd) {
    compile_value.params.ori_shape_M = ori_shape_a.GetDim(idx_m_of_a);
    compile_value.params.ori_shape_K = ori_shape_a.GetDim(idx_k_of_a);
  }
  if (compile_value.params.format_b_nd) {
    compile_value.params.ori_shape_N = ori_shape_b.GetDim(idx_n_of_b);
  }
  bool unvalid_ori_shape = compile_value.params.ori_shape_M > INT_MAX || compile_value.params.ori_shape_K > INT_MAX ||
      compile_value.params.ori_shape_N > INT_MAX;
  CHECK(unvalid_ori_shape,
        CUBE_INNER_ERR_REPORT(op_type, "The m,k,n of a and b tensors' ori_shape must not larger than INT_MAX"),
        return false);
  bool is_nd_input =
      !compile_value.params.binary_mode_flag && compile_value.params.format_a_nd && compile_value.params.format_b_nd;
  if (is_nd_input) {
    // Aligned schedule pattern selection is only enabled in ND input format
    bool aligned_m = compile_value.params.ori_shape_M % block_in == 0;
    bool aligned_k = compile_value.params.ori_shape_K % block_reduce == 0;
    bool aligned_n = compile_value.params.ori_shape_N % block_out == 0;
    bool aligned_mkn = aligned_m && aligned_k && aligned_n;
    if (aligned_mkn) {
      compile_value.params.used_aligned_pattern = true;
    }
  }
  return GetGEMMBatch(op_type, ori_shape_a, ori_shape_b, compile_value.params);
}

uint64_t CheckTilingInRepo(const char *op_type, const GemmCompileInfo &compile_value, bool isBatchMatmulMode) {
  const auto &repo_seeds = compile_value.repo_seeds;
  const auto &repo_range = compile_value.repo_range;
  const auto &tiling_ids = compile_value.repo_tiling_ids;
  const auto &params = compile_value.params;

  int64_t min_distance = LLONG_MAX;
  int64_t m = params.m;
  int64_t k = params.k;
  int64_t n = params.n;
  int64_t batch = params.batch;

  // TODO:
  // std::cout << "m " << m << " n " << n << " k " << k << " batch " << batch << std::endl;

  size_t idx = 0;
  size_t tiling_id_idx = kInvalidTilingId;
  while (idx < repo_seeds.size() && idx < repo_range.size()) {
    auto &seed = repo_seeds[idx];
    auto &range = repo_range[idx];

    auto in_range = range[kIdxMLow] <= params.m && params.m <= range[kIdxMHigh] && range[kIdxKLow] <= params.k &&
        params.k <= range[kIdxKHigh] && range[kIdxNLow] <= params.n && params.n <= range[kIdxNHigh];

    if (isBatchMatmulMode) {
      in_range = in_range && range[kIdxBLow] <= params.batch && params.batch <= range[kIdxBHigh];
    }

    if (in_range) {
      int64_t dist = (m - seed[kIdxM]) * (m - seed[kIdxM]) + (k - seed[kIdxK]) * (k - seed[kIdxK]) +
          (n - seed[kIdxN]) * (n - seed[kIdxN]);

      if (isBatchMatmulMode) {
        dist += (batch - seed[kIdxBatch]) * (batch - seed[kIdxBatch]);
      }
      if (dist < min_distance) {
        min_distance = dist;
        tiling_id_idx = idx;
        // TODO:
        // std::cout << "dist " << dist << " tiling_id_idx " << tiling_id_idx << " tiling_id " << tiling_ids[tiling_id_idx] << std::endl;
      }
    }
    ++idx;
  }

  // TODO: log
  return tiling_id_idx == kInvalidTilingId ? kInvalidTilingId : tiling_ids[tiling_id_idx];
}

uint64_t CheckTilingInCostModel(const char *op_type, const GemmCompileInfo &compile_value, bool isBatchMatmulMode) {
  const auto &cost_range = compile_value.cost_range;
  const auto &tiling_ids = compile_value.cost_tiling_ids;
  const auto &params = compile_value.params;

  uint64_t tiling_id = kInvalidTilingId;
  for (size_t idx = 0; idx < cost_range.size(); ++idx) {
    const auto &range = cost_range[idx];
    auto in_range = range[kIdxMLow] <= params.m && params.m <= range[kIdxMHigh] && range[kIdxKLow] <= params.k &&
        params.k <= range[kIdxKHigh] && range[kIdxNLow] <= params.n && params.n <= range[kIdxNHigh];
    if (isBatchMatmulMode) {
      in_range = in_range && range[kIdxBLow] <= params.batch && params.batch <= range[kIdxBHigh];
    }
    if (in_range) {
      tiling_id = tiling_ids[idx];
      OP_LOGD(op_type, "match tiling_id(%lu) in costmodel", tiling_id);
      break;
    }
  }
  return tiling_id;
}

void FillRunInfoParas(const optiling::Tiling &tiling, OpRunInfoParas &runinfoparas) {
  runinfoparas.batch_dim = tiling.batch_dim;
  runinfoparas.n_dim = tiling.n_dim;
  runinfoparas.m_dim = tiling.m_dim;
  runinfoparas.k_dim = tiling.k_dim;
  runinfoparas.m_l0 = tiling.m_l0;
  runinfoparas.k_l0 = tiling.k_l0;
  runinfoparas.n_l0 = tiling.n_l0;
  if (!tiling.al1_full_load) {
    runinfoparas.k_al1 = tiling.kal1_16 * kBlockSize;
    runinfoparas.m_al1 = tiling.m_al1;
  } else {
    runinfoparas.k_al1 = runinfoparas.params.k_32 / runinfoparas.k_dim * kBlockSize;
    runinfoparas.m_al1 = ceil(static_cast<double>(runinfoparas.params.m_32 / runinfoparas.m_dim) / runinfoparas.m_l0);
  }
  if (!tiling.bl1_full_load) {
    runinfoparas.k_bl1 = tiling.kbl1_16 * kBlockSize;
    runinfoparas.n_bl1 = tiling.n_bl1;
  } else {
    runinfoparas.k_bl1 = runinfoparas.params.k_32 / runinfoparas.k_dim * kBlockSize;
    runinfoparas.n_bl1 = ceil(static_cast<double>(runinfoparas.params.n_32 / runinfoparas.n_dim) / runinfoparas.n_l0);
  }
  runinfoparas.cub_n1 = tiling.n_cub;
  if (runinfoparas.params.nd_flag) {
    runinfoparas.k_aub = tiling.k_aub;
    runinfoparas.m_aub = tiling.m_aub;
    runinfoparas.multi_m_ub_l1 = runinfoparas.m_al1 * runinfoparas.m_l0 / tiling.m_aub;
    runinfoparas.multi_k_aub_l1 = runinfoparas.k_al1 / (tiling.k_aub * kBlockSize);
    runinfoparas.a_align_value = tiling.a_align_value;
    runinfoparas.aub_align_bound = tiling.aub_align_bound;
    runinfoparas.k_bub = tiling.k_bub;
    runinfoparas.n_bub = tiling.n_bub;
    runinfoparas.multi_n_ub_l1 = runinfoparas.n_bl1 * runinfoparas.n_l0 / tiling.n_bub;
    runinfoparas.multi_k_bub_l1 = runinfoparas.k_bl1 / (tiling.k_bub * kBlockSize);
    runinfoparas.b_align_value = tiling.b_align_value;
    runinfoparas.bub_align_bound = tiling.bub_align_bound;
  }
  runinfoparas.kal1_16 = runinfoparas.k_al1 / kBlockSize;
  runinfoparas.kbl1_16 = runinfoparas.k_bl1 / kBlockSize;
  runinfoparas.kal0_factor = runinfoparas.kal1_16 / runinfoparas.k_l0;
  runinfoparas.kbl0_factor = runinfoparas.kbl1_16 / runinfoparas.k_l0;
  runinfoparas.kal1_factor = runinfoparas.params.k_32 / runinfoparas.k_dim / runinfoparas.kal1_16;
  runinfoparas.kbl1_factor = runinfoparas.params.k_32 / runinfoparas.k_dim / runinfoparas.kbl1_16;
  runinfoparas.kl1_times = (runinfoparas.kal1_16 > runinfoparas.kbl1_16)
      ? (runinfoparas.kal1_16 / runinfoparas.kbl1_16)
      : (runinfoparas.kbl1_16 / runinfoparas.kal1_16);
  runinfoparas.n_ub_l0_time = runinfoparas.n_l0 / runinfoparas.cub_n1;
  runinfoparas.batch_single_core = runinfoparas.params.batch_32 / runinfoparas.batch_dim;
  runinfoparas.m_single_core =
      std::max(runinfoparas.params.m_32 / (runinfoparas.m_dim * runinfoparas.m_al1 * runinfoparas.m_l0), int32_t(1));
  runinfoparas.n_single_core =
      std::max(runinfoparas.params.n_32 /
                   (runinfoparas.n_dim * runinfoparas.n_bl1 * runinfoparas.n_ub_l0_time * runinfoparas.cub_n1),
               int32_t(1));
}

void SetRunInfoForCacheTiling(const OpRunInfoParas &runinfoparas, TilingData *tiling_data) {
  if (runinfoparas.params.format_a_nd) {
    tiling_data->Append(runinfoparas.params.m_32);
    tiling_data->Append(runinfoparas.params.k_32);
  }
  if (runinfoparas.params.format_b_nd) {
    tiling_data->Append(runinfoparas.params.n_32);
  }
  tiling_data->Append(runinfoparas.batch_single_core);
  tiling_data->Append(runinfoparas.m_single_core);
  tiling_data->Append(runinfoparas.n_single_core);
  tiling_data->Append(runinfoparas.batch_dim);
  tiling_data->Append(runinfoparas.n_dim);
  tiling_data->Append(runinfoparas.m_dim);
  tiling_data->Append(runinfoparas.k_dim);
  tiling_data->Append(runinfoparas.m_al1);
  tiling_data->Append(runinfoparas.n_bl1);
  tiling_data->Append(runinfoparas.cub_n1);
  tiling_data->Append(runinfoparas.m_l0);
  tiling_data->Append(runinfoparas.k_l0);
  tiling_data->Append(runinfoparas.n_ub_l0_time);
  tiling_data->Append(runinfoparas.kal0_factor);
  tiling_data->Append(runinfoparas.kbl0_factor);
  tiling_data->Append(runinfoparas.kal1_factor);
  tiling_data->Append(runinfoparas.kbl1_factor);
  tiling_data->Append(runinfoparas.kal1_16);
  tiling_data->Append(runinfoparas.kbl1_16);
  tiling_data->Append(runinfoparas.kl1_times);
  if (runinfoparas.params.nd_flag) {
    tiling_data->Append(runinfoparas.m_aub);
    tiling_data->Append(runinfoparas.n_bub);
    tiling_data->Append(runinfoparas.k_aub);
    tiling_data->Append(runinfoparas.k_bub);
    tiling_data->Append(runinfoparas.multi_n_ub_l1);
    tiling_data->Append(runinfoparas.multi_m_ub_l1);
    tiling_data->Append(runinfoparas.multi_k_aub_l1);
    tiling_data->Append(runinfoparas.multi_k_bub_l1);
    tiling_data->Append(runinfoparas.a_align_value);
    tiling_data->Append(runinfoparas.b_align_value);
    tiling_data->Append(runinfoparas.aub_align_bound);
    tiling_data->Append(runinfoparas.bub_align_bound);
  }
}

void SetRunInfo(const bool &isBatchMatmulMode, uint64_t tiling_id, const map<uint64_t, uint32_t> &block_dim_info,
                const OpRunInfoParas &runinfoparas, TilingContext *context) {
  bool is_cache_tiling = tiling_id >= kMinCacheTilingId && tiling_id <= kMaxCacheTilingId;
  if (is_cache_tiling) {
    int32_t block_dim = runinfoparas.batch_dim * runinfoparas.n_dim * runinfoparas.m_dim * runinfoparas.k_dim;
    context->SetBlockDim(static_cast<uint32_t>(block_dim));
  } else {
    auto block_dim_value = block_dim_info.find(tiling_id);
    if (block_dim_value != block_dim_info.end()) {
      context->SetBlockDim(block_dim_value->second);
    }
  }

  // Used Aligned Pattern if the input shape is aligned. Only enabled in ND input format.
  if (runinfoparas.params.used_aligned_pattern && !is_cache_tiling) {
    std::array<int64_t, 20> tiling_id_seq;  // 20: uint64_t max length
    size_t length = 0;
    while (tiling_id > 0) {
      tiling_id_seq[length++] = tiling_id % 10;
      tiling_id /= 10;
    }

    tiling_id_seq[--length] = kAlignedFlag;
    while (length != 0) {
      tiling_id = tiling_id * 10 + tiling_id_seq[length--];
    }
    tiling_id = tiling_id * 10 + tiling_id_seq[0];
  }
  context->SetTilingKey(tiling_id);

  TilingData *tiling_data = context->GetRawTilingData();
  if (runinfoparas.params.format_a_nd) {
    tiling_data->Append(static_cast<int32_t>(runinfoparas.params.ori_shape_M));
    tiling_data->Append(static_cast<int32_t>(runinfoparas.params.ori_shape_K));
  } else {
    tiling_data->Append(runinfoparas.params.m_32);
    tiling_data->Append(runinfoparas.params.k_32);
  }
  if (runinfoparas.params.format_b_nd) {
    tiling_data->Append(static_cast<int32_t>(runinfoparas.params.ori_shape_N));
  } else {
    tiling_data->Append(runinfoparas.params.n_32);
  }
  if (isBatchMatmulMode) {
    tiling_data->Append(runinfoparas.params.batch_32);
  }
  if (is_cache_tiling) {
    SetRunInfoForCacheTiling(runinfoparas, tiling_data);
  }
}

bool UpdateGemmCompileValue(const char *op_type, TilingContext *context, GemmCompileInfo &compile_value) {
  // TODO: 静态的逻辑，BatchmatmulParas里的数据成员拆分后把该函数提到for循环外的最外层调用
  const auto &ori_shape_a = context->GetInputShape(0)->GetOriginShape();
  const auto &ori_shape_b = context->GetInputShape(1)->GetOriginShape();
  ge::DataType dtype = context->GetInputDesc(0)->GetDataType();

  // TODO: private attr
  // int64_t input_size = 0;
  // int64_t hidden_size = 0;
  // bool input_size_flag = ge::AttrUtils::GetInt(op_desc, "input_size", input_size);
  // bool hidden_size_flag = ge::AttrUtils::GetInt(op_desc, "hidden_size", hidden_size);
  // if (input_size_flag && hidden_size_flag) {
  //   int64_t hidden_size_align = (hidden_size + kBlockSize - 1) / kBlockSize * kBlockSize;
  //   ori_shape_a.SetDim(1, hidden_size_align * KMULTI);
  //   int64_t align_dim = (input_size + kBlockSize - 1) / kBlockSize * kBlockSize + hidden_size_align;
  //   ori_shape_b.SetDim(0, align_dim);
  //   ori_shape_b.SetDim(1, hidden_size_align * KMULTI);
  // }
  CHECK((!CalcGEMMMknb(op_type, dtype, ori_shape_a, ori_shape_b, compile_value)),
        CUBE_INNER_ERR_REPORT(op_type, "Failed to calculate m, k, n, batch"), return false);
  return true;
}

uint64_t GEMMTilingSelect(const char *op_type, TilingContext *context, GemmCompileInfo &compile_value) {
  uint64_t tiling_id = kInvalidTilingId;
  const auto &dynamic_mode = compile_value.dynamic_mode;
  bool isBatchMatmulMode = (dynamic_mode == DYNAMIC_MKNB);
  CHECK((dynamic_mode != DYNAMIC_MKN && !isBatchMatmulMode),
        CUBE_INNER_ERR_REPORT(op_type, "Only support dynamic_mode: dynamic_mkn, dynamic_mknb"), return tiling_id);

  // cost 107ns
  CHECK((!UpdateGemmCompileValue(op_type, context, compile_value)),
        CUBE_INNER_ERR_REPORT(op_type, "Failed to update compile value"), return tiling_id);

  if (compile_value.repo_seed_flag) {
    tiling_id = CheckTilingInRepo(op_type, compile_value, isBatchMatmulMode);
  }

  // TODO:
  // std::cout << "after check repo, tiling id: " << tiling_id << std::endl;

  OpRunInfoParas runinfoparas;
  runinfoparas.params = compile_value.params;  // cost 22ns
  if (tiling_id == kInvalidTilingId) {
    if (compile_value.repo_costmodel_flag) {
      tiling_id = CheckTilingInCostModel(op_type, compile_value, isBatchMatmulMode);
    }
    if (tiling_id == kInvalidTilingId) {
      optiling::Tiling tiling;
      GenTiling(op_type, runinfoparas.params, tiling, tiling_id);
      if (tiling_id != kInvalidTilingId) {
        OP_LOGD(op_type, "match tiling_id(%lu) in cache tiling mode", tiling_id);
      } else {
        CUBE_INNER_ERR_REPORT(op_type, "Failed to calculate tiling from cache tiling mode");
        return tiling_id;
      }
      FillRunInfoParas(tiling, runinfoparas);
    }
  }
  if (tiling_id != kInvalidTilingId) {
    SetRunInfo(isBatchMatmulMode, tiling_id, compile_value.block_dim, runinfoparas, context);
  }
  return tiling_id;
}

ge::graphStatus TilingForGemm(TilingContext *context) {
  auto shape_x1 = context->GetInputShape(0);
  auto shape_x2 = context->GetInputShape(1);
  auto src_td = context->GetInputDesc(0);
  if (shape_x1 == nullptr || shape_x2 == nullptr || src_td == nullptr) {
    return ge::GRAPH_FAILED;
  }

  auto compile_info = reinterpret_cast<vector<GemmCompileInfo> *>(const_cast<void *>(context->GetCompileInfo()));
  if (compile_info == nullptr) {
    return ge::GRAPH_FAILED;
  }

  // TODO: debug input
  // OP_LOGD(op_type, "%s", DebugOpParasGEMM(op_paras).c_str());
  uint64_t tiling_id = kInvalidTilingId;
  for (size_t i = 0; i < compile_info->size(); i++) {
    tiling_id = GEMMTilingSelect(context->GetNodeType(), context, (*compile_info)[i]);
    if (tiling_id != kInvalidTilingId) {
      break;
    }
  }

  if (tiling_id == kInvalidTilingId) {
    CUBE_INNER_ERR_REPORT(op_type,
                          "This shape is not covered by any tiling, "
                          "please modify range and recompile");
    return ge::GRAPH_FAILED;
  }

  return ge::GRAPH_SUCCESS;
}

bool GetGemmCompileValue(const json &compile_info, GemmCompileInfo &compile_value) {
  const auto &dynamic_mode = compile_info["dynamic_mode"];
  if (dynamic_mode == "dynamic_mkn") {
    compile_value.dynamic_mode = DYNAMIC_MKN;
  } else if (dynamic_mode == "dynamic_mknb") {
    compile_value.dynamic_mode = DYNAMIC_MKNB;
  } else {
    // TODO: report error
    return false;
  }

  const string &format_a = compile_info["format_a"];
  const string &format_b = compile_info["format_b"];
  compile_value.params.format_a_nd = format_a == "ND";
  compile_value.params.format_b_nd = format_b == "ND";

  const auto &repo_attr = compile_info["attrs"];
  compile_value.trans_a = repo_attr["transpose_a"];
  compile_value.trans_b = repo_attr["transpose_b"];

  const auto &block_dim = compile_info["block_dim"].get<map<string, uint32_t>>();
  for (auto it = block_dim.begin(); it != block_dim.end(); ++it) {
    compile_value.block_dim.insert({stoull(it->first), it->second});
  }

  if (compile_info.contains("binary_mode_flag")) {
    compile_value.params.binary_mode_flag = compile_info["binary_mode_flag"];
    const auto core_num = block_dim.find("CORE_NUM");
    if (core_num != block_dim.end()) {
      compile_value.params.core_num = core_num->second;
    }
    auto &binary_attrs = compile_info["binary_attrs"];
    compile_value.params.bias_flag = binary_attrs["bias_flag"];
    compile_value.params.nd_flag = binary_attrs["nd_flag"];
    compile_value.params.split_k_flag = binary_attrs["split_k_flag"];
    compile_value.params.l2_size = binary_attrs["l2_size"];
    compile_value.params.trans_a_flag = compile_value.trans_a;
    compile_value.params.trans_b_flag = compile_value.trans_b;
    compile_value.params.ubdb_flag = false;
    compile_value.params.at_l1_flag = true;
    compile_value.params.fused_double_operand_num = compile_value.params.nd_flag ? 1 : 0;
    compile_value.params.aub_double_num = compile_value.params.nd_flag ? 1 : 0;
    compile_value.params.bub_double_num = compile_value.params.nd_flag ? 1 : 0;
    compile_value.params.cub_reused_flag = false;
  }

  if (compile_info.contains("repo_seeds") && compile_info.contains("repo_range")) {
    compile_value.repo_seed_flag = true;

    const auto &repo_seeds = compile_info["repo_seeds"].get<map<string, vector<int64_t>>>();
    const auto &repo_range = compile_info["repo_range"].get<map<string, vector<int64_t>>>();
    auto repo_seeds_iter = repo_seeds.begin();
    auto repo_range_iter = repo_range.begin();
    compile_value.repo_seeds.reserve(repo_seeds.size());
    compile_value.repo_range.reserve(repo_range.size());
    compile_value.repo_tiling_ids.reserve(repo_range.size());
    while (repo_seeds_iter != repo_seeds.end() && repo_range_iter != repo_range.end()) {
      if (repo_seeds_iter->first != repo_range_iter->first) {
        // TODO: report error
        return false;
      }

      compile_value.repo_seeds.push_back(repo_seeds_iter->second);
      compile_value.repo_range.push_back(repo_range_iter->second);
      compile_value.repo_tiling_ids.push_back(stoi(repo_seeds_iter->first));
      ++repo_seeds_iter;
      ++repo_range_iter;
    }
  }

  if (compile_info.contains("cost_range")) {
    compile_value.repo_costmodel_flag = true;
    const auto &cost_range = compile_info["cost_range"].get<map<string, vector<int64_t>>>();
    compile_value.repo_range.reserve(cost_range.size());
    compile_value.repo_tiling_ids.reserve(cost_range.size());
    for (auto it = cost_range.begin(); it != cost_range.end(); ++it) {
      compile_value.cost_range.push_back(it->second);
      compile_value.cost_tiling_ids.push_back(stoi(it->first));
    }
  }

  return true;
}

ge::graphStatus TilingPrepareForGemm(KernelContext *context) {
  auto compile_info = context->GetOutputPointer<vector<GemmCompileInfo>>(0);
  auto json_str = context->GetInputStrPointer(0);
  if (compile_info == nullptr || json_str == nullptr) {
    return ge::GRAPH_FAILED;
  }

  // TODO: op type or op name
  //  const char *op_type = "BatchMatMulV2";
  try {
    OP_LOGD(context->GetNode, "compile info: %s", json_str);
    std::unique_ptr<nlohmann::json> parsed_object_cinfo(new (std::nothrow)
                                                            nlohmann::json(nlohmann::json::parse(json_str)));
    if (parsed_object_cinfo == nullptr) {
      return ge::GRAPH_FAILED;
    }

    if (parsed_object_cinfo->type() == json::value_t::object) {
      compile_info->resize(1);
      CHECK(!(GetGemmCompileValue(*parsed_object_cinfo, compile_info->at(0))),
            CUBE_INNER_ERR_REPORT(op_type, "Parse compile value fail"), return ge::GRAPH_FAILED);
    } else {
      compile_info->resize(parsed_object_cinfo->size());
      for (size_t i = 0; i < parsed_object_cinfo->size(); i++) {
        CHECK(!(GetGemmCompileValue((*parsed_object_cinfo)[i], compile_info->at(i))),
              CUBE_INNER_ERR_REPORT(op_type, "Parse compile value fail"), return ge::GRAPH_FAILED);
      }
    }

  } catch (...) {
    CUBE_INNER_ERR_REPORT(op_type, "get unknown exception, please check compile info json.");
    return ge::GRAPH_FAILED;
  }

  return ge::GRAPH_SUCCESS;
}

}  // namespace gert

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cache_tiling.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_CACHE_TILING_H
#define OPS_BUILT_IN_OP_TILING_CACHE_TILING_H

#include <algorithm>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <climits>
#include <map>
#include <cmath>
#include <numeric>
#include <ratio>
#include <unistd.h>
#include <vector>

namespace optiling {

struct BatchmatmulParas {
  bool used_aligned_pattern = false;
  bool binary_mode_flag = false;
  bool b_have_batch = false;
  bool bias_flag = false;
  bool nd_flag = false;
  bool trans_a_flag = false;
  bool trans_b_flag = false;
  bool ubdb_flag = false;
  bool at_l1_flag = true;
  bool cub_reused_flag = false;
  bool split_k_flag = false;
  bool format_a_nd = false;
  bool format_b_nd = false;
  int32_t m_32 = 1;
  int32_t k_32 = 1;
  int32_t n_32 = 1;
  int32_t batch_32 = 1;
  int32_t l2_size = (1024 * 1024);
  int32_t core_num = 32;
  float fused_double_operand_num = 0;
  float aub_double_num = 0;
  float bub_double_num = 0;
  int64_t m = 1;
  int64_t k = 1;
  int64_t n = 1;
  int64_t batch = 1;
  int64_t ori_shape_M = 1;
  int64_t ori_shape_K = 1;
  int64_t ori_shape_N = 1;
};

struct CoreStatus {
  int32_t batch = 1;
  int32_t m = 1;
  int32_t k = 1;
  int32_t n = 1;
  int32_t batch_dim = 1;
  int32_t m_dim = 1;
  int32_t n_dim = 1;
  int32_t k_dim = 1;
};

struct BlockDimCalculator {
  int32_t batch = 1;
  int32_t m = 1;
  int32_t k = 1;
  int32_t n = 1;
  int32_t k_num = 1;
  int32_t k_bytes = 1;
  int32_t n_dim_factor = 1;
  int32_t batch_dim_factor = 1;
  int32_t m_dim_factor = 1;
  int32_t k_dim_factor = 1;
  int32_t min_load_size = 1;
  int32_t core_use = 1;
  int32_t tmp_core_use = 1;
  int32_t i_idx = 0;
  int32_t j_idx = 0;
  int32_t k_idx = 0;
  int32_t n_idx = 0;
  int32_t batch_dim_cnt = 0;
  int32_t m_dim_cnt = 0;
  int32_t n_dim_cnt = 0;
  int32_t k_dim_cnt = 0;
  int32_t ori_amat_size = 0;
  int32_t ori_bmat_size = 0;
  int32_t amat_size = 0;
  int32_t bmat_size = 0;
  int32_t tmp_amat_size = 0;
  int32_t tmp_bmat_size = 0;
  int32_t tmp_load_size = 0;
  int32_t total_load_size = 0;
  int32_t* batch_dim_array;
  int32_t* m_dim_array;
  int32_t* n_dim_array;
  int32_t* k_dim_array;
  int32_t tmp_value = 0;
  int32_t final_value = 0;
  bool final_blocking_flag = false;
};

struct L0Status {
  int32_t m_l0 = 1;
  int32_t n_l0 = 1;
  int32_t k_l0 = 1;
  int32_t db_l0a = 1;
  int32_t db_l0b = 1;
  int32_t db_l0c = 1;
  int32_t db_cub = 1;
  int32_t final_ml0 = 0;
  int32_t final_kl0 = 0;
  int32_t final_nl0 = 0;
  int32_t final_load_size = INT_MAX;
  float final_l0c_use = 0;
  int32_t final_mul = 0;
  int32_t final_mte1Loop = INT_MAX;
  int32_t max_mk = 1;
  int32_t max_nk = 1;
  int32_t max_mn = 1;
  int32_t max_axis_idx = 0;
  int32_t max_axis_num = 0;
  int32_t max_axis_pnt = 1;
  void SetInitLoadStatus()
  {
    final_ml0 = 0;
    final_kl0 = 0;
    final_nl0 = 0;
    final_load_size = INT_MAX;
    final_l0c_use = 0;
    final_mul = 0;
    final_mte1Loop = INT_MAX;
  }
};

struct L0Factors {
  int32_t final_ml0 = 0;
  int32_t final_kl0 = 0;
  int32_t final_nl0 = 0;
  int32_t final_load_size = INT_MAX;
  float final_l0c_use = 0;
  int32_t final_mul = 0;
  int32_t final_mte1Loop = INT_MAX;
};

struct MKNParasCombo {
  int32_t parasCombo[9];
};

struct L1Status {
  int32_t kal1_16 = 1;
  int32_t kbl1_16 = 1;
  int32_t m_al1 = 1;
  int32_t n_bl1 = 1;
  int32_t db_al1 = 1;
  int32_t db_bl1 = 1;
  int32_t al1_size = 0;
  int32_t bl1_size = 0;
  int32_t al1_times = 1;
  int32_t bl1_times = 1;
  int32_t all_times = 1;
  int32_t load_size = 0;
  int32_t max_m_al1 = 1;
  int32_t max_n_bl1 = 1;
  int32_t max_k_al1 = 1;
  int32_t max_k_bl1 = 1;
  bool both_full_load = false;
  bool al1_full_load = false;
  bool bl1_full_load = false;
  void SetStatus(const int32_t *tmp_l1_factors)
  {
    this->kal1_16 = tmp_l1_factors[0];
    this->kbl1_16 = tmp_l1_factors[1];
    this->m_al1 = tmp_l1_factors[2];
    this->n_bl1 = tmp_l1_factors[3];
    this->db_al1 = tmp_l1_factors[4];
    this->db_bl1 = tmp_l1_factors[5];
  }
};

struct UbStatus {
  int32_t k_aub = 1;
  int32_t m_aub = 1;
  int32_t db_aub = 1;
  int32_t k_bub = 1;
  int32_t n_bub = 1;
  int32_t db_bub = 1;
  int32_t n_cub = 1;
  int32_t db_cub = 1;
  int32_t max_dma_size = 0;
  int32_t min_dma_size = 0;
  int32_t min_load_size = 0;
  int32_t aub_size = 0;
  int32_t bub_size = 0;
  int32_t cub_size = 0;
  int32_t aub_multi_flag = 0;
  int32_t bub_multi_flag = 0;
  int32_t a_align_value = 1;
  int32_t b_align_value = 1;
  int32_t aub_bank_size = 0;
  int32_t bub_bank_size = 0;
  int32_t aub_align_bound = 0;
  int32_t bub_align_bound = 0;
};

class Tiling {
public:
  std::string tiling_id;
  int32_t n_cub = 1;
  int32_t db_cub = 1;
  int32_t m_l0 = 1;
  int32_t k_l0 = 1;
  int32_t n_l0 = 1;
  int32_t batch_dim = 1;
  int32_t n_dim = 1;
  int32_t m_dim = 1;
  int32_t k_dim = 1;
  int32_t kal1_16 = 1;
  int32_t kbl1_16 = 1;
  int32_t m_al1 = 1;
  int32_t n_bl1 = 1;
  int32_t db_al1 = 1;
  int32_t db_bl1 = 1;
  int32_t k_aub = 1;
  int32_t m_aub = 1;
  int32_t db_aub = 1;
  int32_t k_bub = 1;
  int32_t n_bub = 1;
  int32_t db_bub = 1;
  int32_t k_org_dim = 1;
  int32_t db_l0c = 1;
  int32_t aub_multi_flag = 0;
  int32_t bub_multi_flag = 0;
  int32_t a_align_value = 1;
  int32_t b_align_value = 1;
  int32_t aub_align_bound = 0;
  int32_t bub_align_bound = 0;
  int32_t min_kl1_cmp_kl0 = 0;
  int32_t al1_attach_flag = 0;
  int32_t bl1_attach_flag = 0;
  int32_t abkl1_attach_flag = 0;
  bool al1_full_load = false;
  bool bl1_full_load = false;
  Tiling() = default;
  void SetParams(const CoreStatus& coreStatus, const L0Status& l0Status, const L1Status& l1Status,
                 const UbStatus& ubStatus, const BatchmatmulParas& params);
  void SetAttachFlag();
  void GetTilingId(const BatchmatmulParas& params);
  ~Tiling() = default;
};

void GetFactors(int32_t* cnt, int32_t* factorList, const int32_t& num,
                const int32_t& maxNum);
void GetTwoFactors(int32_t* res, const int32_t& base, const int32_t& dim,
                   const int32_t& maxNum);
void GetNearestFactor(const int32_t& base, int32_t& factor);
void BL1FullLoadBlock(const CoreStatus& coreStatus, BlockDimCalculator& blockDimCalculator, int32_t& n0);
void AL1FullLoadBlock(const CoreStatus& coreStatus, BlockDimCalculator& blockDimCalculator, int32_t& m0);
int32_t GetBlockDim(const std::string& op_type, const BatchmatmulParas& params, CoreStatus& coreStatus);
int32_t GetLoadSize(const CoreStatus& coreStatus, const L0Status& l0Status);
MKNParasCombo GetParasCombo(const int32_t& index, const int32_t& blockValue);
void GetFinalMkn(L0Status& l0Status, const CoreStatus& coreStatus);
void GetL0StatusFromParasCombo(L0Status& l0Status, int32_t* parasCombo);
void SetResFactors(L0Factors* resFactors, const L0Status& l0Status);
void GetL0FactorsCand(L0Factors *resFactors, const CoreStatus &coreStatus, L0Status &l0Status,
                      int32_t *parasCombo);
void GetL0Factors(const std::string& op_type, const CoreStatus& coreStatus, const int32_t& blockValue,
                  L0Status& l0Status);
int32_t GetL1Size(const L1Status& l1Status, const L0Status& l0Status);
void CheckSpecialTemplate(const std::string& op_type, const CoreStatus& coreStatus, const L0Status& l0Status,
                          L1Status& l1Status);
void L1StatusBothFullLoad(const CoreStatus& coreStatus, const L0Status& l0Status, L1Status& l1Status,
                          int32_t res[][7]);
void L1StatusAl1FullLoad(const CoreStatus& coreStatus, const L0Status& l0Status, L1Status& l1Status,
                         int32_t res[][7]);
void L1StatusBl1FullLoad(const CoreStatus& coreStatus, const L0Status& l0Status, L1Status& l1Status,
                         int32_t res[][7]);
void NeitherFullLoadDb(const CoreStatus& coreStatus, const L0Status& l0Status, L1Status& l1Status,
                       const int32_t& kbl1Db);
void NeitherFullLoadMN(const CoreStatus& coreStatus, const L0Status& l0Status, L1Status& l1Status,
                       const BatchmatmulParas& params);
void NeitherFullLoadK(const CoreStatus& coreStatus, const L0Status& l0Status, L1Status& l1Status,
                      const BatchmatmulParas& params);
void L1StatusNeitherFullLoad(const CoreStatus& coreStatus, const BatchmatmulParas& params,
                             const L0Status& l0Status, L1Status& l1Status, int32_t res[][7]);
void GetL1Factors(const std::string& op_type, const BatchmatmulParas& params, const CoreStatus& coreStatus,
                  const L0Status& l0Status, L1Status& l1Status);
void GetUbFactors(const std::string& op_type, const L0Status& l0Status, UbStatus& ubStatus);
void GenTiling(const std::string& op_type, const BatchmatmulParas& params, Tiling& tiling, uint64_t& tilingId);
}; // namespace optiling

#endif

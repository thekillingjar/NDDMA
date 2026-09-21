/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATT_TILING_DATA_H_
#define ATT_TILING_DATA_H_
#include <stdint.h>
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
namespace optiling {
#define Status uint64_t
#define SUCCESS 0U
#define FAILED 1U
struct TilingData {
  // input of tiling_data1
  uint32_t CORENUM;
void set_CORENUM(const uint32_t value)
{
CORENUM = value;
}
uint32_t get_CORENUM() const
{
  return CORENUM;
}
  uint32_t k_size;
void set_k_size(const uint32_t value)
{
k_size = value;
}
uint32_t get_k_size() const
{
  return k_size;
}
  uint32_t m_size;
void set_m_size(const uint32_t value)
{
m_size = value;
}
uint32_t get_m_size() const
{
  return m_size;
}
  uint32_t n_size;
void set_n_size(const uint32_t value)
{
n_size = value;
}
uint32_t get_n_size() const
{
  return n_size;
}
  // output of tiling_data1
  uint32_t basek_size;
void set_basek_size(const uint32_t value)
{
basek_size = value;
}
uint32_t get_basek_size() const
{
  return basek_size;
}
  uint32_t basem_size;
void set_basem_size(const uint32_t value)
{
basem_size = value;
}
uint32_t get_basem_size() const
{
  return basem_size;
}
  uint32_t basen_size;
void set_basen_size(const uint32_t value)
{
basen_size = value;
}
uint32_t get_basen_size() const
{
  return basen_size;
}
  uint32_t stepka_size;
void set_stepka_size(const uint32_t value)
{
stepka_size = value;
}
uint32_t get_stepka_size() const
{
  return stepka_size;
}
  uint32_t stepkb_size;
void set_stepkb_size(const uint32_t value)
{
stepkb_size = value;
}
uint32_t get_stepkb_size() const
{
  return stepkb_size;
}
  uint32_t tilem_size;
void set_tilem_size(const uint32_t value)
{
tilem_size = value;
}
uint32_t get_tilem_size() const
{
  return tilem_size;
}
  uint32_t tilen_size;
void set_tilen_size(const uint32_t value)
{
tilen_size = value;
}
uint32_t get_tilen_size() const
{
  return tilen_size;
}
  uint32_t L1;
void set_L1(const uint32_t value)
{
L1 = value;
}
uint32_t get_L1() const
{
  return L1;
}
  uint32_t L2;
void set_L2(const uint32_t value)
{
L2 = value;
}
uint32_t get_L2() const
{
  return L2;
}
  uint32_t L0A;
void set_L0A(const uint32_t value)
{
L0A = value;
}
uint32_t get_L0A() const
{
  return L0A;
}
  uint32_t L0B;
void set_L0B(const uint32_t value)
{
L0B = value;
}
uint32_t get_L0B() const
{
  return L0B;
}
  uint32_t L0C;
void set_L0C(const uint32_t value)
{
L0C = value;
}
uint32_t get_L0C() const
{
  return L0C;
}
  uint32_t Q1;
void set_Q1(const uint32_t value)
{
Q1 = value;
}
uint32_t get_Q1() const
{
  return Q1;
}
  uint32_t MATMUL_OUTPUT1;
void set_MATMUL_OUTPUT1(const uint32_t value)
{
MATMUL_OUTPUT1 = value;
}
uint32_t get_MATMUL_OUTPUT1() const
{
  return MATMUL_OUTPUT1;
}
  // input of tiling_data0
  // output of tiling_data0
  uint32_t stepm_size;
void set_stepm_size(const uint32_t value)
{
stepm_size = value;
}
uint32_t get_stepm_size() const
{
  return stepm_size;
}
  uint32_t stepn_size;
void set_stepn_size(const uint32_t value)
{
stepn_size = value;
}
uint32_t get_stepn_size() const
{
  return stepn_size;
}
};
Status GetTiling(TilingData &tiling_data, uint32_t tilingCaseId);
} // namespace att
#endif


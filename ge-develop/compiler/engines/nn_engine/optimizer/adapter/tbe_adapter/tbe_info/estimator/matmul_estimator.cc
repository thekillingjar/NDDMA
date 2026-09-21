/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "adapter/tbe_adapter/tbe_info/estimator/matmul_estimator.h"
#include "common/math_util.h"
#include "graph/utils/attr_utils.h"

namespace fe {
const uint32_t kInput0Index = 0;
const uint32_t kInput1Index = 1;
const size_t kMinMatMulDimCnt = 4;

namespace {
bool IsCubeMatMul(const ge::GeTensorDescPtr &input0, const ge::GeTensorDescPtr &input1) {
  const ge::GeShape &input0_shape = input0->GetShape();
  const ge::GeShape &input1_shape = input1->GetShape();
  if (input0_shape.GetDimNum() < kMinMatMulDimCnt ||
      input1_shape.GetDimNum() < kMinMatMulDimCnt) {
    return false;
  }

  if (input0->GetFormat() != ge::FORMAT_FRACTAL_NZ ||
      input1->GetFormat() != ge::FORMAT_FRACTAL_NZ) {
    return false;
  }

  if (input0->GetDataType() != ge::DT_FLOAT16) {
    return false;
  }

  return true;
}
}

Status MatMulEstimator::EstimateTime(PlatFormInfos &platform_info, const ge::OpDesc &op_desc,
                                     uint64_t &exec_time) {
  auto input0 = op_desc.MutableInputDesc(kInput0Index);
  auto input1 = op_desc.MutableInputDesc(kInput1Index);
  FE_CHECK_NOTNULL(input0);
  FE_CHECK_NOTNULL(input1);
  if (!IsCubeMatMul(input0, input1)) {
    return FAILED;
  }
  const ge::GeShape &input0_shape = input0->GetShape();
  size_t input0_dim_count = input0_shape.GetDimNum();
  const ge::GeShape &input1_shape = input1->GetShape();

  bool transpose_x1 = false;
  /* In op BatchMatMul & BatchMatMulV2, attribute transpose_x1 and transpose_x2 are described as
   * adj_x1 and adj_x2. */
  ge::AttrUtils::GetBool(op_desc, "transpose_x1", transpose_x1);
  ge::AttrUtils::GetBool(op_desc, "adj_x1", transpose_x1);

  bool transpose_x2 = false;
  ge::AttrUtils::GetBool(op_desc, "transpose_x2", transpose_x2);
  ge::AttrUtils::GetBool(op_desc, "adj_x2", transpose_x2);
  /* Original shape:
   * input0(batch, m, k), Nz shape (batch, k1, m1, m0, k0)
   * input1(batch, k, n), Nz shape (batch, n1, k1, k0, n0)
   * If input0 need transpose, it's shape will become:
   * (batch, k, m), Nz shape (batch, m1, k1, m0, k0)
   * If input1 need transpose, it's shape will become:
   * (batch, n, k), Nz shape (batch, k1, n1, n0, k0)
   * cycle = cube_k * cube_m * cube_n * batch
   * cube_k = k1, cube_m = m1, cube_n = n1. */
  FE_CHECK((input0_dim_count < kMinMatMulDimCnt), FE_LOGE("Input 0 dim count invalid."), return FAILED);
  size_t batch_dim_cnt = input0_dim_count - kMinMatMulDimCnt;
  size_t cube_k_index = transpose_x1 ? batch_dim_cnt + 1 : batch_dim_cnt;
  size_t cube_m_index = transpose_x1 ? batch_dim_cnt : batch_dim_cnt + 1;
  uint64_t cube_k = input0_shape.GetDim(cube_k_index);
  uint64_t cube_m = input0_shape.GetDim(cube_m_index);

  size_t cube_n_index = transpose_x2 ? batch_dim_cnt + 1 : batch_dim_cnt;
  uint64_t cube_n = input1_shape.GetDim(cube_n_index);
  uint64_t batch = 1;
  for (size_t i = 0; i < batch_dim_cnt; i++) {
    int64_t value0 = input0_shape.GetDim(i);
    int64_t value1 = input1_shape.GetDim(i);
    batch *= static_cast<uint64_t>(std::max(value0, value1));
  }
  uint64_t cycle = cube_m;
  FE_MUL_OVERFLOW(cycle, cube_k, cycle);
  FE_MUL_OVERFLOW(cycle, cube_n, cycle);
  FE_MUL_OVERFLOW(cycle, batch, cycle);
  exec_time = BasicEstimator::CalcTimeByCycle(platform_info, cycle);
  FE_LOGD("Matmul batch[%lu], cube_m[%lu], cube_k[%lu], cube_n[%lu], execution time and cycle are %lu and %lu.",
          batch, cube_m, cube_k, cube_n, exec_time, cycle);
  return SUCCESS;
}
}

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "adapter/tbe_adapter/tbe_info/estimator/vector_estimator.h"
#include "common/math_util.h"

namespace fe {
constexpr char const *kVecCalcSize = "vec_calc_size";
const uint32_t kVecCoeff = 3;

namespace {
uint64_t GetLargestVectorShapeSize(const ge::OpDesc &op_desc) {
  int64_t largest_shape_size = 0;
  size_t output_size = op_desc.GetOutputsSize();
  for (size_t i = 0; i < output_size; i++) {
    auto output_tensor = op_desc.MutableOutputDesc(i);
    int64_t curr_shape_size = output_tensor->GetShape().GetShapeSize();
    largest_shape_size = std::max(largest_shape_size, curr_shape_size);
  }

  return static_cast<uint64_t>(largest_shape_size);
}
}

Status VectorEstimator::GetVecCycle(PlatFormInfos &platform_info, const ge::OpDesc &op_desc,
                                    uint64_t &cycle) {
  uint64_t vector_calc_size = BasicEstimator::GetUintParam(platform_info, kAiCoreSpecLbl, kVecCalcSize);
  if (vector_calc_size == 0) {
    REPORT_FE_ERROR("[SubGraphOpt][PreCompileOp][VectorEstimator]vector_calc_size %lu is 0!", vector_calc_size);
    return FAILED;
  }
  uint64_t shape_size = GetLargestVectorShapeSize(op_desc);
  FE_MUL_OVERFLOW(shape_size, kVecCoeff, shape_size)
  cycle = shape_size / vector_calc_size;
  FE_LOGD("Shape size and vec cycle is %lu and %lu.", shape_size, cycle);
  return SUCCESS;
}

Status VectorEstimator::EstimateTime(PlatFormInfos &platform_info, const ge::OpDesc &op_desc,
                                     uint64_t &exec_time) {
  uint64_t cycle = 0UL;
  Status ret = GetVecCycle(platform_info, op_desc, cycle);
  if (ret != SUCCESS) {
    return ret;
  }
  exec_time = BasicEstimator::CalcTimeByCycle(platform_info, cycle);
  return SUCCESS;
}
}

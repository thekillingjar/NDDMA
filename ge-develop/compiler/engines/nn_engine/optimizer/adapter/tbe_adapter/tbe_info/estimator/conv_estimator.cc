/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "adapter/tbe_adapter/tbe_info/estimator/conv_estimator.h"
#include "adapter/tbe_adapter/tbe_info/estimator/vector_estimator.h"
#include "common/fe_op_info_common.h"
#include "common/format/axis_util.h"

namespace fe {
const uint32_t kFmIndex = 0;
const uint32_t kWeightIndex = 1;
constexpr char const *kCubeMSizeKey = "cube_m_size";
constexpr char const *kCubeNSizeKey = "cube_n_size";
const uint32_t kFzDimK1 = 0;
const uint32_t kFzDimN1 = 1;
const uint32_t kFzDimN0 = 2;
const uint32_t kFzDimK0 = 3;

Status ConvEstimator::GetConvCycle(PlatFormInfos &platform_info,
                                   const std::vector<int64_t> &weight_shape,
                                   const std::vector<int64_t> &fm_shape,
                                   uint64_t &cycle) {
  uint64_t cube_m_size = BasicEstimator::GetUintParam(platform_info, kAiCoreSpecLbl, kCubeMSizeKey);
  uint64_t cube_n_size = BasicEstimator::GetUintParam(platform_info, kAiCoreSpecLbl, kCubeNSizeKey);

  FE_LOGD("cube_m_size is %lu and cube_n_size is %lu.", cube_m_size, cube_n_size);
  if (cube_m_size == 0 || cube_n_size == 0) {
    return FAILED;
  }

  if (fm_shape.size() != DIM_DEFAULT_SIZE) {
    FE_LOGW("Fm size %zu is invalid, must be 4.", fm_shape.size());
    return FAILED;
  }

  if (weight_shape.size() != DIM_DEFAULT_SIZE) {
    FE_LOGW("Weight size %zu is invalid, must be 4.", fm_shape.size());
    return FAILED;
  }

  auto cube_n = static_cast<uint64_t>(weight_shape[kFzDimN1]);

  auto matrix_m = static_cast<uint64_t>(fm_shape[NCHW_DIM_H] * fm_shape[NCHW_DIM_W]);
  auto cube_m = DivisionCeiling(matrix_m, cube_m_size);

  auto batch = static_cast<uint64_t>(fm_shape[NCHW_DIM_N]);
  auto cube_k = static_cast<uint64_t>(weight_shape[kFzDimK1]);
  cycle = cube_m * cube_k * cube_n * batch;
  FE_LOGD("Conv batch[%lu], cube_m[%lu], cube_k[%lu], cube_n[%lu], cycle is %lu.",
          batch, cube_m, cube_k, cube_n, cycle);
  return SUCCESS;
}

Status ConvEstimator::EstimateTime(PlatFormInfos &platform_info, const ge::OpDesc &op_desc,
                                   uint64_t &exec_time) {
  auto weight = op_desc.MutableInputDesc(kWeightIndex);
  if (weight == nullptr) { FE_LOGD("weight is nullptr."); return FAILED; }
  auto weight_format = static_cast<ge::Format>(ge::GetPrimaryFormat(weight->GetFormat()));
  if (weight_format != ge::FORMAT_FRACTAL_Z) {
    FE_LOGD("Do not support weight format %d of node %s.", weight_format, op_desc.GetName().c_str());
    return FAILED;
  }

  auto fm = op_desc.MutableInputDesc(kFmIndex);
  if (fm == nullptr) { FE_LOGD("fm is nullptr."); return FAILED; }
  ge::Format fm_format = fm->GetFormat();
  const ge::GeShape &fm_shape = fm->GetShape();

  ge::GeShape new_shape;
  if (fm_format == ge::FORMAT_FRACTAL_NZ) {
    if (fm_shape.GetDimNum() != DIM_DEFAULT_SIZE) {
      FE_LOGW("FRACTAL_NZ's size %zu must be 4.", fm_shape.GetDimNum());
      return FAILED;
    }
    std::vector<int64_t> new_shape_dims;
    new_shape_dims.emplace_back(fm_shape.GetDim(kFzDimN1));
    new_shape_dims.emplace_back(fm_shape.GetDim(kFzDimK1) * fm_shape.GetDim(kFzDimK0));
    new_shape_dims.emplace_back(1);
    new_shape_dims.emplace_back(1);
    new_shape = ge::GeShape(new_shape_dims);
  } else if (fm_format == ge::FORMAT_NC1HWC0) {
    new_shape.AppendDim(fm_shape.GetDim(NC1HWC0_DIM_N));
    new_shape.AppendDim(fm_shape.GetDim(NC1HWC0_DIM_C1) * fm_shape.GetDim(NC1HWC0_DIM_C0));
    new_shape.AppendDim(fm_shape.GetDim(NC1HWC0_DIM_H));
    new_shape.AppendDim(fm_shape.GetDim(NC1HWC0_DIM_W));
  } else {
    ge::DataType fm_dtype = fm->GetDataType();
    ShapeAndFormat shape_and_format_info = {fm_shape, new_shape, fm_format, ge::FORMAT_NCHW, fm_dtype};
    Status ret = GetShapeAccordingToFormat(shape_and_format_info);
    if (ret != SUCCESS) {
      return ret;
    }
  }

  uint64_t conv_cycle = 0;
  auto &weight_shape = weight->GetShape();
  if (GetConvCycle(platform_info, weight_shape.GetDims(), new_shape.GetDims(), conv_cycle) != SUCCESS ||
      conv_cycle == 0) {
    return FAILED;
  }

  uint64_t vec_cycle = 0;
  if (VectorEstimator::GetVecCycle(platform_info, op_desc, vec_cycle) != SUCCESS ||
      vec_cycle == 0) {
    return FAILED;
  }

  FE_LOGD("Conv and Vector cycles are %lu and %lu.", conv_cycle, vec_cycle);
  uint64_t cycle = std::max(conv_cycle, vec_cycle);
  exec_time = BasicEstimator::CalcTimeByCycle(platform_info, cycle);
  return SUCCESS;
}
}

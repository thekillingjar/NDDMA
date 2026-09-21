/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/format/axis_util.h"
#include "common/math_util.h"

namespace fe {
const std::map<ge::Format, GetAxisValueInfoByFormatPtr> AxisUtil::get_axis_value_func_map = {
    {ge::FORMAT_NCHW, std::make_shared<GetAxisValueInfoByFormat>(GetAxisValueByNCHW)},
    {ge::FORMAT_NHWC, std::make_shared<GetAxisValueInfoByFormat>(GetAxisValueByNHWC)},
    {ge::FORMAT_NC1HWC0, std::make_shared<GetAxisValueInfoByFormat>(GetAxisValueByNC1HWC0)},
    {ge::FORMAT_FRACTAL_Z, std::make_shared<GetAxisValueInfoByFormat>(GetAxisValueByFz)},
    {ge::FORMAT_HWCN, std::make_shared<GetAxisValueInfoByFormat>(GetAxisValueByHWCN)},
    {ge::FORMAT_CHWN, std::make_shared<GetAxisValueInfoByFormat>(GetAxisValueByCHWN)},
    {ge::FORMAT_ND, std::make_shared<GetAxisValueInfoByFormat>(GetAxisValueByND)},
    {ge::FORMAT_NDHWC, std::make_shared<GetAxisValueInfoByFormat>(GetAxisValueByNDHWC)},
    {ge::FORMAT_NCDHW, std::make_shared<GetAxisValueInfoByFormat>(GetAxisValueByNCDHW)},
    /* The Last N of NHWCN is considered as Cout, which is the C o NDHWC */
    {ge::FORMAT_DHWCN, std::make_shared<GetAxisValueInfoByFormat>(GetAxisValueByDHWCN)},
    {ge::FORMAT_DHWNC, std::make_shared<GetAxisValueInfoByFormat>(GetAxisValueByDHWNC)}};

uint64_t DivisionCeiling(uint64_t dividend, uint64_t divisor) {
  if (divisor == 0) {
    return 0;
  } else {
    return (dividend + divisor - 1) / divisor;
  }
}

int64_t DivisionCeiling(int64_t dividend, int64_t divisor) {
  if (divisor == 0) {
    return 0;
  } else if (dividend == UNKNOWN_SHAPE_VALUE || divisor == UNKNOWN_SHAPE_VALUE) {
    return UNKNOWN_SHAPE_VALUE;
  } else {
    int64_t tmp_divisor = divisor - 1;
    if (CheckInt64AddOverflow(dividend, tmp_divisor) == SUCCESS) {
      dividend = dividend + tmp_divisor;
    }
    return dividend / divisor;
  }
}

Status AxisUtil::GetAxisValueByOriginFormat(const ge::Format& format, const vector<int64_t>& dim_vec,
                                            const int64_t& c0, vector<int64_t>& axis_value,
                                            vector<int64_t>& nd_value) {
  auto iter_get_axis_func = get_axis_value_func_map.find(format);
  if (iter_get_axis_func == get_axis_value_func_map.end()) {
    FE_LOGW("Can not get axis value of old format %u!", format);
    return FAILED;
  }
  GetAxisValueInfoByFormatPtr get_axis_func = nullptr;
  FE_MAKE_SHARED(get_axis_func = iter_get_axis_func->second, return FAILED);
  FE_CHECK_NOTNULL(get_axis_func);
  return (*get_axis_func)(dim_vec, c0, axis_value, nd_value);
}

bool AxisUtil::HasAxisValueFunc(const ge::Format& format) {
  auto iter_get_axis_func = get_axis_value_func_map.find(format);
  if (iter_get_axis_func == get_axis_value_func_map.end()) {
    FE_LOGW("Can not get axis value of format %u!", format);
    return false;
  }
  return true;
}

Status AxisUtil::CheckParams(const vector<int64_t>& original_dim_vec, const int64_t &c0, vector<int64_t>& nd_value,
                             const size_t& dim_default_size = DIM_DEFAULT_SIZE) {
  nd_value = original_dim_vec;

  size_t dim_size = original_dim_vec.size();
  if (dim_size < dim_default_size) {
    /* Before this funcion, we should call function PadDimensionTo4. */
    FE_LOGW("Dimension size %zu is invalid.", dim_size);
    return FAILED;
  }
  if (c0 == 0) {
    FE_LOGE("c0 is zero!");
    return FAILED;
  }

  return SUCCESS;
}

Status AxisUtil::GetAxisValueByND(const vector<int64_t>& original_dim_vec, const int64_t &c0,
                                  vector<int64_t>& axis_value_b, vector<int64_t>& nd_value) {
  if (axis_value_b.empty()) {
    FE_LOGW("AxisValue is empty!");
    return SUCCESS;
  }
  if (original_dim_vec.empty()) {
    FE_LOGW("Original dim vector is empty!");
    return SUCCESS;
  }
  nd_value = original_dim_vec;
  /* To differentiate the input datatype of int8 and others */
  axis_value_b[AXIS_C0] = c0;
  if (original_dim_vec.size() == NCHW_DIMENSION_NUM) {
    axis_value_b[AXIS_N] = original_dim_vec[NCHW_DIM_N];
    axis_value_b[AXIS_C] = original_dim_vec[NCHW_DIM_C];
    axis_value_b[AXIS_H] = original_dim_vec[NCHW_DIM_H];
    axis_value_b[AXIS_W] = original_dim_vec[NCHW_DIM_W];
    axis_value_b[AXIS_C1] = DivisionCeiling(original_dim_vec[NCHW_DIM_C], c0);
    axis_value_b[AXIS_Co] = c0;
  }
  return SUCCESS;
}

Status AxisUtil::GetAxisValueByNCHW(const vector<int64_t>& original_dim_vec, const int64_t &c0,
                                    vector<int64_t>& axis_value_a, vector<int64_t>& nd_value) {
  if (axis_value_a.empty()) {
    FE_LOGW("AxisValue is empty!");
    return SUCCESS;
  }
  if (original_dim_vec.empty()) {
    FE_LOGW("Original dim vector is empty!");
    return SUCCESS;
  }
  /* C0 Must be set for case ND or 2D-NCHW to NZ */
  axis_value_a[AXIS_C0] = c0;
  if (CheckParams(original_dim_vec, c0, nd_value) != SUCCESS) {
    FE_LOGW("Parameter is invalid!");
    return FAILED;
  }

  axis_value_a[AXIS_N] = original_dim_vec[NCHW_DIM_N];
  axis_value_a[AXIS_C] = original_dim_vec[NCHW_DIM_C];
  axis_value_a[AXIS_H] = original_dim_vec[NCHW_DIM_H];
  axis_value_a[AXIS_W] = original_dim_vec[NCHW_DIM_W];
  axis_value_a[AXIS_C1] = DivisionCeiling(axis_value_a[AXIS_C], c0);
  axis_value_a[AXIS_Co] = c0;
  return SUCCESS;
}

Status AxisUtil::GetOriginAxisAttribute(const ge::OpDesc& op_desc, const ge::GeShape shape,
                                        vector<int64_t>& axis_index_vec) {
  if (!ge::AttrUtils::GetListInt(op_desc, AXES_ATTR_NAME, axis_index_vec)) {
    FE_LOGW("Can not get reduce op [%s] axis or its value is empty!", op_desc.GetName().c_str());
    return FAILED;
  }
  if (axis_index_vec.empty()) {
    bool noop_with_empty_axes = true;
    ge::AttrUtils::GetBool(op_desc, "noop_with_empty_axes", noop_with_empty_axes);
    if (noop_with_empty_axes) {
      FE_LOGW("axis or its value is empty and noop_with_empty_axes is not false, %s", op_desc.GetName().c_str());
      return FAILED;
    } else {
      for (size_t i = 0; i < shape.GetDimNum(); i++) {
        axis_index_vec.emplace_back(i);
      }
    }
  }
  size_t size = axis_index_vec.size();
  for (size_t i = 0; i != size; ++i) {
    if (axis_index_vec[i] < 0) {
        axis_index_vec[i] = axis_index_vec[i] + shape.GetDimNum();
    }
    FE_LOGD("Reduce op [%s] axis value is %ld.", op_desc.GetName().c_str(), axis_index_vec[i]);
  }
  return SUCCESS;
}

Status AxisUtil::GetAxisValueByNHWC(const vector<int64_t>& original_dim_vec, const int64_t &c0,
                                    vector<int64_t>& axis_value_c, vector<int64_t>& nd_value) {
  if (axis_value_c.empty()) {
    FE_LOGW("AxisValue is empty!");
    return SUCCESS;
  }
  if (original_dim_vec.empty()) {
    FE_LOGW("Original dim vector is empty!");
    return SUCCESS;
  }
  /* C0 Must be set for case ND or 2D-NHWC to NZ */
  axis_value_c[AXIS_C0] = c0;
  if (CheckParams(original_dim_vec, c0, nd_value) != SUCCESS) {
    FE_LOGW("Parameter is invalid!");
    return FAILED;
  }

  axis_value_c[AXIS_N] = original_dim_vec[NHWC_DIM_N];
  axis_value_c[AXIS_C] = original_dim_vec[NHWC_DIM_C];
  axis_value_c[AXIS_H] = original_dim_vec[NHWC_DIM_H];
  axis_value_c[AXIS_W] = original_dim_vec[NHWC_DIM_W];
  axis_value_c[AXIS_C1] = DivisionCeiling(axis_value_c[AXIS_C], c0);
  axis_value_c[AXIS_Co] = c0;
  return SUCCESS;
}

Status AxisUtil::GetAxisValueByNC1HWC0(const vector<int64_t>& original_dim_vec_a, const int64_t &c0,
                                       vector<int64_t>& axis_value_j, vector<int64_t>& nd_value) {
  if (axis_value_j.empty()) {
    FE_LOGW("AxisValue is empty!");
    return SUCCESS;
  }
  if (original_dim_vec_a.empty()) {
    FE_LOGW("Original dim vector is empty!");
    return SUCCESS;
  }
  if (CheckParams(original_dim_vec_a, c0, nd_value) != SUCCESS) {
    FE_LOGW("Parameter is invalid!");
    return FAILED;
  }

  auto dim_size = original_dim_vec_a.size();
  if (dim_size == DIM_DEFAULT_SIZE + 1) {
    axis_value_j[AXIS_C1] = original_dim_vec_a[NC1HWC0_DIM_C1];
    axis_value_j[AXIS_C0] = original_dim_vec_a[NC1HWC0_DIM_C0];
    axis_value_j[AXIS_C] = axis_value_j[AXIS_C1] * axis_value_j[AXIS_C0];
  } else {
    axis_value_j[AXIS_C1] = DivisionCeiling(original_dim_vec_a[NCHW_DIM_C], c0);
    axis_value_j[AXIS_C0] = c0;
    axis_value_j[AXIS_C] = original_dim_vec_a[NCHW_DIM_C];
  }

  axis_value_j[AXIS_N] = original_dim_vec_a[NCHW_DIM_N];
  axis_value_j[AXIS_H] = original_dim_vec_a[NCHW_DIM_H];
  axis_value_j[AXIS_W] = original_dim_vec_a[NCHW_DIM_W];
  return SUCCESS;
}

/* !!!!Deprecated!!!! For current stage, we consider fz as nchw.
 * Actually, it is {HWC/16, N, 16,16} */
Status AxisUtil::GetAxisValueByFz(const vector<int64_t>& original_dim_vec_b, const int64_t &c0,
                                  vector<int64_t>& axis_value_f, vector<int64_t>& nd_value) {
  if (axis_value_f.empty()) {
    FE_LOGW("AxisValue is empty!");
    return SUCCESS;
  }
  if (original_dim_vec_b.empty()) {
    FE_LOGW("Original dim vector is empty!");
    return SUCCESS;
  }
  if (CheckParams(original_dim_vec_b, c0, nd_value) != SUCCESS) {
    FE_LOGW("Parameter is invalid!");
    return FAILED;
  }

  axis_value_f[AXIS_N] = original_dim_vec_b[NCHW_DIM_N];
  axis_value_f[AXIS_C] = original_dim_vec_b[NCHW_DIM_C];
  axis_value_f[AXIS_H] = original_dim_vec_b[NCHW_DIM_H];
  axis_value_f[AXIS_W] = original_dim_vec_b[NCHW_DIM_W];
  axis_value_f[AXIS_C1] = DivisionCeiling(original_dim_vec_b[NCHW_DIM_C], c0);
  axis_value_f[AXIS_C0] = c0;
  return SUCCESS;
}

Status AxisUtil::GetAxisValueByHWCN(const vector<int64_t>& original_dim_vec, const int64_t &c0,
                                    vector<int64_t>& axis_value_d, vector<int64_t>& nd_value) {
  if (axis_value_d.empty()) {
    FE_LOGW("AxisValue is empty!");
    return SUCCESS;
  }
  if (original_dim_vec.empty()) {
    FE_LOGW("Original dim vector is empty!");
    return SUCCESS;
  }
  /* C0 Must be set for case ND or 2D-HWCN to NZ */
  axis_value_d[AXIS_C0] = c0;
  if (CheckParams(original_dim_vec, c0, nd_value) != SUCCESS) {
    FE_LOGW("Parameter is invalid!");
    return FAILED;
  }

  axis_value_d[AXIS_N] = original_dim_vec[HWCN_DIM_N];
  axis_value_d[AXIS_C] = original_dim_vec[HWCN_DIM_C];
  axis_value_d[AXIS_H] = original_dim_vec[HWCN_DIM_H];
  axis_value_d[AXIS_W] = original_dim_vec[HWCN_DIM_W];
  axis_value_d[AXIS_C1] = DivisionCeiling(axis_value_d[AXIS_C], c0);
  axis_value_d[AXIS_Co] = c0;
  return SUCCESS;
}

Status AxisUtil::GetAxisValueByCHWN(const vector<int64_t>& original_dim_vec, const int64_t &c0,
                                    vector<int64_t>& axis_value_e, vector<int64_t>& nd_value) {
  if (axis_value_e.empty()) {
    FE_LOGW("AxisValue is empty!");
    return SUCCESS;
  }
  if (original_dim_vec.empty()) {
    FE_LOGW("Original dim vector is empty!");
    return SUCCESS;
  }
  /* C0 Must be set for case ND or 2D-CHWN to NZ */
  axis_value_e[AXIS_C0] = c0;
  if (CheckParams(original_dim_vec, c0, nd_value) != SUCCESS) {
    FE_LOGW("Parameter is invalid!");
    return FAILED;
  }

  axis_value_e[AXIS_N] = original_dim_vec[CHWN_DIM_N];
  axis_value_e[AXIS_C] = original_dim_vec[CHWN_DIM_C];
  axis_value_e[AXIS_H] = original_dim_vec[CHWN_DIM_H];
  axis_value_e[AXIS_W] = original_dim_vec[CHWN_DIM_W];
  axis_value_e[AXIS_C1] = DivisionCeiling(axis_value_e[AXIS_C], c0);
  axis_value_e[AXIS_Co] = c0;
  return SUCCESS;
}

Status AxisUtil::GetAxisValueByNDHWC(const vector<int64_t>& original_dim_vec_c, const int64_t &c0,
                                     vector<int64_t>& axis_value_f, vector<int64_t>& nd_value) {
  if (axis_value_f.empty()) {
    FE_LOGW("AxisValue is empty!");
    return SUCCESS;
  }
  if (original_dim_vec_c.empty()) {
    FE_LOGW("Original dim vector is empty!");
    return SUCCESS;
  }
  axis_value_f[AXIS_C0] = c0;
  if (CheckParams(original_dim_vec_c, c0, nd_value, DIMENSION_NUM_FIVE) != SUCCESS) {
    FE_LOGW("Parameter is invalid!");
    return FAILED;
  }

  axis_value_f[AXIS_N] = original_dim_vec_c[NDHWC_DIM_N];
  axis_value_f[AXIS_C] = original_dim_vec_c[NDHWC_DIM_C];
  axis_value_f[AXIS_H] = original_dim_vec_c[NDHWC_DIM_H];
  axis_value_f[AXIS_W] = original_dim_vec_c[NDHWC_DIM_W];
  axis_value_f[AXIS_C1] = DivisionCeiling(axis_value_f[AXIS_C], c0);
  axis_value_f[AXIS_C0] = c0;
  axis_value_f[AXIS_Co] = c0;
  axis_value_f[AXIS_D] = original_dim_vec_c[NDHWC_DIM_D];
  return SUCCESS;
}

Status AxisUtil::GetAxisValueByNCDHW(const vector<int64_t>& original_dim_vec_d, const int64_t &c0,
                                     vector<int64_t>& axis_value_g, vector<int64_t>& nd_value) {
  if (axis_value_g.empty()) {
    FE_LOGW("AxisValue is empty!");
    return SUCCESS;
  }
  if (original_dim_vec_d.empty()) {
    FE_LOGW("Original dim vector is empty!");
    return SUCCESS;
  }
  axis_value_g[AXIS_C0] = c0;
  if (CheckParams(original_dim_vec_d, c0, nd_value, DIMENSION_NUM_FIVE) != SUCCESS) {
    FE_LOGW("Parameter is invalid!");
    return FAILED;
  }

  axis_value_g[AXIS_N] = original_dim_vec_d[NCDHW_DIM_N];
  axis_value_g[AXIS_C] = original_dim_vec_d[NCDHW_DIM_C];
  axis_value_g[AXIS_H] = original_dim_vec_d[NCDHW_DIM_H];
  axis_value_g[AXIS_W] = original_dim_vec_d[NCDHW_DIM_W];
  axis_value_g[AXIS_C1] = DivisionCeiling(axis_value_g[AXIS_C], c0);
  axis_value_g[AXIS_C0] = c0;
  axis_value_g[AXIS_Co] = c0;
  axis_value_g[AXIS_D] = original_dim_vec_d[NCDHW_DIM_D];
  return SUCCESS;
}

Status AxisUtil::GetAxisValueByDHWCN(const vector<int64_t>& original_dim_vec_e, const int64_t &c0,
                                     vector<int64_t>& axis_value_h, vector<int64_t>& nd_value) {
  if (axis_value_h.empty()) {
    FE_LOGW("AxisValue is empty!");
    return SUCCESS;
  }
  if (original_dim_vec_e.empty()) {
    FE_LOGW("Original dim vector is empty!");
    return SUCCESS;
  }
  axis_value_h[AXIS_C0] = c0;
  if (CheckParams(original_dim_vec_e, c0, nd_value, DIMENSION_NUM_FIVE) != SUCCESS) {
    FE_LOGW("Parameter is invalid!");
    return FAILED;
  }

  axis_value_h[AXIS_N] = original_dim_vec_e[DHWCN_DIM_N];
  axis_value_h[AXIS_C] = original_dim_vec_e[DHWCN_DIM_C];
  axis_value_h[AXIS_H] = original_dim_vec_e[DHWCN_DIM_H];
  axis_value_h[AXIS_W] = original_dim_vec_e[DHWCN_DIM_W];
  axis_value_h[AXIS_C1] = DivisionCeiling(axis_value_h[AXIS_C], c0);
  axis_value_h[AXIS_C0] = c0;
  axis_value_h[AXIS_Co] = c0;
  axis_value_h[AXIS_D] = original_dim_vec_e[DHWCN_DIM_D];
  return SUCCESS;
}

Status AxisUtil::GetAxisValueByDHWNC(const vector<int64_t>& original_dim_vec_f, const int64_t &c0,
                                     vector<int64_t>& axis_value_i, vector<int64_t>& nd_value) {
  if (axis_value_i.empty()) {
    FE_LOGW("AxisValue is empty!");
    return SUCCESS;
  }
  if (original_dim_vec_f.empty()) {
    FE_LOGW("Original dim vector is empty!");
    return SUCCESS;
  }
  axis_value_i[AXIS_C0] = c0;
  if (CheckParams(original_dim_vec_f, c0, nd_value, DIMENSION_NUM_FIVE) != SUCCESS) {
    FE_LOGW("Parameter is invalid!");
    return FAILED;
  }

  axis_value_i[AXIS_N] = original_dim_vec_f[DHWNC_DIM_N];
  axis_value_i[AXIS_C] = original_dim_vec_f[DHWNC_DIM_C];
  axis_value_i[AXIS_H] = original_dim_vec_f[DHWNC_DIM_H];
  axis_value_i[AXIS_W] = original_dim_vec_f[DHWNC_DIM_W];
  axis_value_i[AXIS_C1] = DivisionCeiling(axis_value_i[AXIS_C], c0);
  axis_value_i[AXIS_C0] = c0;
  axis_value_i[AXIS_Co] = c0;
  axis_value_i[AXIS_D] = original_dim_vec_f[DHWNC_DIM_D];
  return SUCCESS;
}
};  // namespace fe

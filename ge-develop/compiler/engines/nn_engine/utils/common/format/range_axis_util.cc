/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/format/range_axis_util.h"

namespace fe {
const std::map<ge::Format, GetRangeAxisValueInfoByFormatPtr> RangeAxisUtil::get_range_axis_value_func_map = {
    {ge::FORMAT_NCHW, std::make_shared<GetRangeAxisValueInfoByFormat>(GetRangeAxisValueByNCHW)},
    {ge::FORMAT_NHWC, std::make_shared<GetRangeAxisValueInfoByFormat>(GetRangeAxisValueByNHWC)},
    {ge::FORMAT_NC1HWC0, std::make_shared<GetRangeAxisValueInfoByFormat>(GetRangeAxisValueByNC1HWC0)},
    {ge::FORMAT_FRACTAL_Z, std::make_shared<GetRangeAxisValueInfoByFormat>(GetRangeAxisValueByFz)},
    {ge::FORMAT_HWCN, std::make_shared<GetRangeAxisValueInfoByFormat>(GetRangeAxisValueByHWCN)},
    {ge::FORMAT_CHWN, std::make_shared<GetRangeAxisValueInfoByFormat>(GetRangeAxisValueByCHWN)},
    {ge::FORMAT_ND, std::make_shared<GetRangeAxisValueInfoByFormat>(GetRangeAxisValueByND)},
    {ge::FORMAT_NDHWC, std::make_shared<GetRangeAxisValueInfoByFormat>(GetRangeAxisValueByNDHWC)},
    {ge::FORMAT_NCDHW, std::make_shared<GetRangeAxisValueInfoByFormat>(GetRangeAxisValueByNCDHW)},
    /* The Last N of NHWCN is considered as Cout, which is the C o NDHWC */
    {ge::FORMAT_DHWCN, std::make_shared<GetRangeAxisValueInfoByFormat>(GetRangeAxisValueByDHWCN)},
    {ge::FORMAT_DHWNC, std::make_shared<GetRangeAxisValueInfoByFormat>(GetRangeAxisValueByDHWNC)}};

Status RangeAxisUtil::CheckParamValue(const vector<std::pair<int64_t, int64_t>>& original_range_vec,
                                      const vector<int64_t>& original_dim_vec, const uint32_t& c0,
                                      vector<std::pair<int64_t, int64_t>>& range_value,
                                      const size_t& min_size = DIM_DEFAULT_SIZE) {
  if (range_value.size() < AXIS_BOTTOM) {
    FE_LOGW("rangeValue is empty!");
    return FAILED;
  }
  if (original_dim_vec.empty()) {
    FE_LOGW("Original dim vector is empty!");
    return FAILED;
  }
  if (original_dim_vec.size() < min_size) {
    FE_LOGW("Original dim vector size: %zu is less than %zu!", original_dim_vec.size(), min_size);
    return FAILED;
  }
  if (original_dim_vec.size() != original_range_vec.size()) {
    FE_LOGW("Size of shape is different from size of range!");
    return FAILED;
  }
  if (c0 == 0) {
    FE_LOGE("c0 is zero!");
    return FAILED;
  }
  return SUCCESS;
}

Status RangeAxisUtil::GetRangeAxisValueByOriginFormat(const vector<std::pair<int64_t, int64_t>>& original_range_vec,
                                                      const ge::Format& format, const vector<int64_t>& dim_vec,
                                                      const uint32_t& c0,
                                                      vector<std::pair<int64_t, int64_t>>& range_value) {
  auto iter_range_get_axis_func = get_range_axis_value_func_map.find(format);
  if (iter_range_get_axis_func == get_range_axis_value_func_map.end()) {
    FE_LOGW("Can not get range axis value of old format %u!", format);
    return FAILED;
  }
  GetRangeAxisValueInfoByFormatPtr get_range_axis_func = nullptr;
  FE_MAKE_SHARED(get_range_axis_func = iter_range_get_axis_func->second, return FAILED);
  FE_CHECK_NOTNULL(get_range_axis_func);
  return (*get_range_axis_func)(original_range_vec, dim_vec, c0, range_value);
}

bool RangeAxisUtil::HasAxisValueFunc(const ge::Format& format) {
  auto iter_get_axis_func = get_range_axis_value_func_map.find(format);
  if (iter_get_axis_func == get_range_axis_value_func_map.end()) {
    FE_LOGW("Can not get range axis value of format %u!", format);
    return false;
  }
  return true;
}

Status RangeAxisUtil::GetRangeAxisValueByND(const vector<std::pair<int64_t, int64_t>>& original_range_vec,
                                            const vector<int64_t>& original_dim_vec, const uint32_t& c0,
                                            vector<std::pair<int64_t, int64_t>>& range_value) {
  if (range_value.size() < AXIS_BOTTOM) {
    FE_LOGW("rangeValue is empty!");
    return FAILED;
  }
  if (original_dim_vec.empty()) {
    FE_LOGW("Original dim vector is empty!");
    return FAILED;
  }
  /* To differentiate the input datatype of int8 and others */
  std::pair<int64_t, int64_t> c0_range_b(c0, c0);
  range_value[AXIS_C0] = c0_range_b;

  FE_LOGD("Size of original_range_vec is %zu, original_dim_vec is %zu.",
          original_range_vec.size(), original_dim_vec.size());
  /* Check original_range_vec size, to avoid array bound */
  if ((original_dim_vec.size() == NCHW_DIMENSION_NUM) && (original_range_vec.size() == NCHW_DIMENSION_NUM)) {
    range_value[AXIS_N] = original_range_vec[NCHW_DIM_N];
    range_value[AXIS_C] = original_range_vec[NCHW_DIM_C];
    range_value[AXIS_H] = original_range_vec[NCHW_DIM_H];
    range_value[AXIS_W] = original_range_vec[NCHW_DIM_W];
    int64_t c1_first_range_b = DivisionCeiling(original_range_vec[NCHW_DIM_C].first, static_cast<int64_t>(c0));
    int64_t c1_second_range_b = DivisionCeiling(original_range_vec[NCHW_DIM_C].second, static_cast<int64_t>(c0));

    range_value[AXIS_C1] = std::pair<int64_t, int64_t>(c1_first_range_b, c1_second_range_b);
    range_value[AXIS_Co] = c0_range_b;
  }
  return SUCCESS;
}

Status RangeAxisUtil::GetRangeAxisValueByNCHW(const vector<std::pair<int64_t, int64_t>>& original_range_vec,
                                              const vector<int64_t>& original_dim_vec, const uint32_t& c0,
                                              vector<std::pair<int64_t, int64_t>>& range_value) {
  /* C0 Must be set for case ND or 2D-NCHW to NZ */
  std::pair<int64_t, int64_t> c0_range_a(c0, c0);
  range_value[AXIS_C0] = c0_range_a;
  if (CheckParamValue(original_range_vec, original_dim_vec, c0, range_value) != SUCCESS) {
    FE_LOGW("Parameter is invalid!");
    return FAILED;
  }

  range_value[AXIS_N] = original_range_vec[NCHW_DIM_N];
  range_value[AXIS_C] = original_range_vec[NCHW_DIM_C];
  range_value[AXIS_H] = original_range_vec[NCHW_DIM_H];
  range_value[AXIS_W] = original_range_vec[NCHW_DIM_W];
  int64_t c1_first_range_a = DivisionCeiling(original_range_vec[NCHW_DIM_C].first, static_cast<int64_t>(c0));
  int64_t c1_second_range_a = DivisionCeiling(original_range_vec[NCHW_DIM_C].second, static_cast<int64_t>(c0));
  range_value[AXIS_C1] = std::pair<int64_t, int64_t>(c1_first_range_a, c1_second_range_a);
  range_value[AXIS_Co] = c0_range_a;
  return SUCCESS;
}

Status RangeAxisUtil::GetRangeAxisValueByNHWC(const vector<std::pair<int64_t, int64_t>>& original_range_vec,
                                              const vector<int64_t>& original_dim_vec, const uint32_t& c0,
                                              vector<std::pair<int64_t, int64_t>>& range_value) {
  /* C0 Must be set for case ND or 2D-NHWC to NZ */
  std::pair<int64_t, int64_t> c0_range_c(c0, c0);
  range_value[AXIS_C0] = c0_range_c;
  if (CheckParamValue(original_range_vec, original_dim_vec, c0, range_value) != SUCCESS) {
    FE_LOGW("Parameter is invalid!");
    return FAILED;
  }

  range_value[AXIS_N] = original_range_vec[NHWC_DIM_N];
  range_value[AXIS_C] = original_range_vec[NHWC_DIM_C];
  range_value[AXIS_H] = original_range_vec[NHWC_DIM_H];
  range_value[AXIS_W] = original_range_vec[NHWC_DIM_W];
  int64_t c1_first_range_c = DivisionCeiling(original_range_vec[NHWC_DIM_C].first, static_cast<int64_t>(c0));
  int64_t c1_second_range_c = DivisionCeiling(original_range_vec[NHWC_DIM_C].second, static_cast<int64_t>(c0));
  range_value[AXIS_C1] = std::pair<int64_t, int64_t>(c1_first_range_c, c1_second_range_c);
  range_value[AXIS_Co] = c0_range_c;
  return SUCCESS;
}

Status RangeAxisUtil::GetRangeAxisValueByNC1HWC0(const vector<std::pair<int64_t, int64_t>>& original_range_vec,
                                                 const vector<int64_t>& original_dim_vec, const uint32_t& c0,
                                                 vector<std::pair<int64_t, int64_t>>& range_value) {
  if (CheckParamValue(original_range_vec, original_dim_vec, c0, range_value) != SUCCESS) {
    FE_LOGW("Parameter is invalid!");
    return FAILED;
  }
  auto dim_size_a = original_dim_vec.size();
  if (dim_size_a == DIM_DEFAULT_SIZE + 1) {
    range_value[AXIS_C1] = original_range_vec[NC1HWC0_DIM_C1];
    range_value[AXIS_C0] = original_range_vec[NC1HWC0_DIM_C0];
    int64_t range_lower = 0;
    int64_t range_higher = 0;
    FE_MUL_OVERFLOW(range_value[AXIS_C1].first, range_value[AXIS_C0].first, range_lower);
    FE_MUL_OVERFLOW(range_value[AXIS_C1].second, range_value[AXIS_C0].second, range_higher);
    range_value[AXIS_C] = std::pair<int64_t, int64_t>(range_lower, range_higher);
  } else {
    int64_t c1_first_range_d = DivisionCeiling(original_range_vec[NCHW_DIM_C].first, static_cast<int64_t>(c0));
    int64_t c1_second_range_d = DivisionCeiling(original_range_vec[NCHW_DIM_C].second, static_cast<int64_t>(c0));
    range_value[AXIS_C1] = std::pair<int64_t, int64_t>(c1_first_range_d, c1_second_range_d);
    range_value[AXIS_C0] = std::pair<int64_t, int64_t>(c0, c0);
    range_value[AXIS_C] = original_range_vec[NCHW_DIM_C];
  }

  range_value[AXIS_N] = original_range_vec[NCHW_DIM_N];
  range_value[AXIS_H] = original_range_vec[NCHW_DIM_H];
  range_value[AXIS_W] = original_range_vec[NCHW_DIM_W];
  return SUCCESS;
}

/* !!!!Deprecated!!!! For current stage, we consider fz as nchw.
 * Actually, it is {HWC/16, N, 16,16} */
Status RangeAxisUtil::GetRangeAxisValueByFz(const vector<std::pair<int64_t, int64_t>>& original_range_vec,
                                            const vector<int64_t>& original_dim_vec, const uint32_t& c0,
                                            vector<std::pair<int64_t, int64_t>>& range_value) {
  if (CheckParamValue(original_range_vec, original_dim_vec, c0, range_value) != SUCCESS) {
    FE_LOGW("Parameter is invalid!");
    return FAILED;
  }
  range_value[AXIS_N] = original_range_vec[NCHW_DIM_N];
  range_value[AXIS_C] = original_range_vec[NCHW_DIM_C];
  range_value[AXIS_H] = original_range_vec[NCHW_DIM_H];
  range_value[AXIS_W] = original_range_vec[NCHW_DIM_W];
  int64_t c1_first_range_e = DivisionCeiling(original_range_vec[NCHW_DIM_C].first, static_cast<int64_t>(c0));
  int64_t c1_second_range_e = DivisionCeiling(original_range_vec[NCHW_DIM_C].second, static_cast<int64_t>(c0));
  range_value[AXIS_C1] = std::pair<int64_t, int64_t>(c1_first_range_e, c1_second_range_e);
  range_value[AXIS_C0] = std::pair<int64_t, int64_t>(c0, c0);
  return SUCCESS;
}

Status RangeAxisUtil::GetRangeAxisValueByHWCN(const vector<std::pair<int64_t, int64_t>>& original_range_vec,
                                              const vector<int64_t>& original_dim_vec, const uint32_t& c0,
                                              vector<std::pair<int64_t, int64_t>>& range_value) {
  /* C0 Must be set for case ND or 2D-HWCN to NZ */
  std::pair<int64_t, int64_t> c0_range_f(c0, c0);
  range_value[AXIS_C0] = c0_range_f;
  if (CheckParamValue(original_range_vec, original_dim_vec, c0, range_value) != SUCCESS) {
    FE_LOGW("Parameter is invalid!");
    return FAILED;
  }

  range_value[AXIS_N] = original_range_vec[HWCN_DIM_N];
  range_value[AXIS_C] = original_range_vec[HWCN_DIM_C];
  range_value[AXIS_H] = original_range_vec[HWCN_DIM_H];
  range_value[AXIS_W] = original_range_vec[HWCN_DIM_W];
  int64_t c1_first_range_f = DivisionCeiling(original_range_vec[HWCN_DIM_C].first, static_cast<int64_t>(c0));
  int64_t c1_second_range_f = DivisionCeiling(original_range_vec[HWCN_DIM_C].second, static_cast<int64_t>(c0));
  range_value[AXIS_C1] = std::pair<int64_t, int64_t>(c1_first_range_f, c1_second_range_f);
  range_value[AXIS_Co] = c0_range_f;
  return SUCCESS;
}

Status RangeAxisUtil::GetRangeAxisValueByCHWN(const vector<std::pair<int64_t, int64_t>>& original_range_vec,
                                              const vector<int64_t>& original_dim_vec, const uint32_t& c0,
                                              vector<std::pair<int64_t, int64_t>>& range_value) {
  /* C0 Must be set for case ND or 2D-CHWN to NZ */
  std::pair<int64_t, int64_t> c0_range_g(c0, c0);
  range_value[AXIS_C0] = c0_range_g;
  if (CheckParamValue(original_range_vec, original_dim_vec, c0, range_value) != SUCCESS) {
    FE_LOGW("Parameter is invalid!");
    return FAILED;
  }

  range_value[AXIS_N] = original_range_vec[CHWN_DIM_N];
  range_value[AXIS_C] = original_range_vec[CHWN_DIM_C];
  range_value[AXIS_H] = original_range_vec[CHWN_DIM_H];
  range_value[AXIS_W] = original_range_vec[CHWN_DIM_W];
  int64_t c1_first_range_g = DivisionCeiling(original_range_vec[CHWN_DIM_C].first, static_cast<int64_t>(c0));
  int64_t c1_second_range_g = DivisionCeiling(original_range_vec[CHWN_DIM_C].second, static_cast<int64_t>(c0));
  range_value[AXIS_C1] = std::pair<int64_t, int64_t>(c1_first_range_g, c1_second_range_g);
  range_value[AXIS_Co] = c0_range_g;
  return SUCCESS;
}

Status RangeAxisUtil::GetRangeAxisValueByNDHWC(const vector<std::pair<int64_t, int64_t>>& original_range_vec,
                                               const vector<int64_t>& original_dim_vec, const uint32_t& c0,
                                               vector<std::pair<int64_t, int64_t>>& range_value) {
  std::pair<int64_t, int64_t> c0_range_h(c0, c0);
  range_value[AXIS_C0] = c0_range_h;
  if (CheckParamValue(original_range_vec, original_dim_vec, c0, range_value, DIMENSION_NUM_FIVE) != SUCCESS) {
    FE_LOGW("Parameter is invalid!");
    return FAILED;
  }

  range_value[AXIS_N] = original_range_vec[NDHWC_DIM_N];
  range_value[AXIS_C] = original_range_vec[NDHWC_DIM_C];
  range_value[AXIS_H] = original_range_vec[NDHWC_DIM_H];
  range_value[AXIS_W] = original_range_vec[NDHWC_DIM_W];
  int64_t c1_first_range_h = DivisionCeiling(original_range_vec[NDHWC_DIM_C].first, static_cast<int64_t>(c0));
  int64_t c1_second_range_h = DivisionCeiling(original_range_vec[NDHWC_DIM_C].second, static_cast<int64_t>(c0));
  range_value[AXIS_C1] = std::pair<int64_t, int64_t>(c1_first_range_h, c1_second_range_h);
  range_value[AXIS_Co] = c0_range_h;
  range_value[AXIS_D] = original_range_vec[NDHWC_DIM_D];
  return SUCCESS;
}

Status RangeAxisUtil::GetRangeAxisValueByNCDHW(const vector<std::pair<int64_t, int64_t>>& original_range_vec,
                                               const vector<int64_t>& original_dim_vec, const uint32_t& c0,
                                               vector<std::pair<int64_t, int64_t>>& range_value) {
  std::pair<int64_t, int64_t> c0_range_i(c0, c0);
  range_value[AXIS_C0] = c0_range_i;
  if (CheckParamValue(original_range_vec, original_dim_vec, c0, range_value, DIMENSION_NUM_FIVE) != SUCCESS) {
    FE_LOGW("Parameter is invalid!");
    return FAILED;
  }

  range_value[AXIS_N] = original_range_vec[NCDHW_DIM_N];
  range_value[AXIS_C] = original_range_vec[NCDHW_DIM_C];
  range_value[AXIS_H] = original_range_vec[NCDHW_DIM_H];
  range_value[AXIS_W] = original_range_vec[NCDHW_DIM_W];
  int64_t c1_first_range_i = DivisionCeiling(original_range_vec[NCDHW_DIM_C].first, static_cast<int64_t>(c0));
  int64_t c1_second_range_i = DivisionCeiling(original_range_vec[NCDHW_DIM_C].second, static_cast<int64_t>(c0));
  range_value[AXIS_C1] = std::pair<int64_t, int64_t>(c1_first_range_i, c1_second_range_i);
  range_value[AXIS_Co] = c0_range_i;
  range_value[AXIS_D] = original_range_vec[NCDHW_DIM_D];
  return SUCCESS;
}

Status RangeAxisUtil::GetRangeAxisValueByDHWCN(const vector<std::pair<int64_t, int64_t>>& original_range_vec,
                                               const vector<int64_t>& original_dim_vec, const uint32_t& c0,
                                               vector<std::pair<int64_t, int64_t>>& range_value) {
  std::pair<int64_t, int64_t> c0_range_j(c0, c0);
  range_value[AXIS_C0] = c0_range_j;
  if (CheckParamValue(original_range_vec, original_dim_vec, c0, range_value, DIMENSION_NUM_FIVE) != SUCCESS) {
    FE_LOGW("Parameter is invalid!");
    return FAILED;
  }

  range_value[AXIS_N] = original_range_vec[DHWCN_DIM_N];
  range_value[AXIS_C] = original_range_vec[DHWCN_DIM_C];
  range_value[AXIS_H] = original_range_vec[DHWCN_DIM_H];
  range_value[AXIS_W] = original_range_vec[DHWCN_DIM_W];
  int64_t c1_first_range_j = DivisionCeiling(original_range_vec[DHWCN_DIM_C].first, static_cast<int64_t>(c0));
  int64_t c1_second_range_j = DivisionCeiling(original_range_vec[DHWCN_DIM_C].second, static_cast<int64_t>(c0));
  range_value[AXIS_C1] = std::pair<int64_t, int64_t>(c1_first_range_j, c1_second_range_j);
  range_value[AXIS_Co] = c0_range_j;
  range_value[AXIS_D] = original_range_vec[DHWCN_DIM_D];
  return SUCCESS;
}

Status RangeAxisUtil::GetRangeAxisValueByDHWNC(const vector<std::pair<int64_t, int64_t>>& original_range_vec,
                                               const vector<int64_t>& original_dim_vec, const uint32_t& c0,
                                               vector<std::pair<int64_t, int64_t>>& range_value) {
  std::pair<int64_t, int64_t> c0_range_k(c0, c0);
  range_value[AXIS_C0] = c0_range_k;
  if (CheckParamValue(original_range_vec, original_dim_vec, c0, range_value, DIMENSION_NUM_FIVE) != SUCCESS) {
    FE_LOGW("Parameter is invalid!");
    return FAILED;
  }

  range_value[AXIS_N] = original_range_vec[DHWNC_DIM_N];
  range_value[AXIS_C] = original_range_vec[DHWNC_DIM_C];
  range_value[AXIS_H] = original_range_vec[DHWNC_DIM_H];
  range_value[AXIS_W] = original_range_vec[DHWNC_DIM_W];
  int64_t c1_first_range_k = DivisionCeiling(original_range_vec[DHWNC_DIM_C].first, static_cast<int64_t>(c0));
  int64_t c1_second_range_k = DivisionCeiling(original_range_vec[DHWNC_DIM_C].second, static_cast<int64_t>(c0));
  range_value[AXIS_C1] = std::pair<int64_t, int64_t>(c1_first_range_k, c1_second_range_k);
  range_value[AXIS_Co] = c0_range_k;
  range_value[AXIS_D] = original_range_vec[DHWNC_DIM_D];
  return SUCCESS;
}
};  // namespace fe

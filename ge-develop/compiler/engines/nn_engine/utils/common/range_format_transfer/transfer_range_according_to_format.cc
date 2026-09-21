/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "transfer_range_according_to_format.h"
#include "common/format/range_axis_util.h"
#include "common/unknown_shape_util.h"
#include "common/op_info_common.h"
#include "common/fe_type_utils.h"
#include "common/aicore_util_constants.h"
#include "common/platform_utils.h"

namespace fe {
const std::map<ge::Format, GetNewRangeByAxisValueAndFormatPtr> RangeTransferAccordingToFormat::get_new_range_func_map =
    {{ge::FORMAT_NCHW, std::make_shared<GetNewRangeByAxisValueAndFormat>(GetNCHWRangeByAxisValue)},
     {ge::FORMAT_NHWC, std::make_shared<GetNewRangeByAxisValueAndFormat>(GetNHWCRangeByAxisValue)},
     {ge::FORMAT_NC1HWC0, std::make_shared<GetNewRangeByAxisValueAndFormat>(GetNC1HWC0RangeByAxisValue)},
     {ge::FORMAT_C1HWC0, std::make_shared<GetNewRangeByAxisValueAndFormat>(GetC1HWC0RangeByAxisValue)},
     {ge::FORMAT_NC1HWC0_C04, std::make_shared<GetNewRangeByAxisValueAndFormat>(GetNC1HWC0RangeByAxisValue)},
     {ge::FORMAT_FRACTAL_Z, std::make_shared<GetNewRangeByAxisValueAndFormat>(GetFzRangeByAxisValue)},
     {ge::FORMAT_FRACTAL_Z_WINO, std::make_shared<GetNewRangeByAxisValueAndFormat>(GetFzWinoRangeByAxisValue)},
     {ge::FORMAT_FRACTAL_Z_C04, std::make_shared<GetNewRangeByAxisValueAndFormat>(GetFzC04RangeByAxisValue)},
     {ge::FORMAT_HWCN, std::make_shared<GetNewRangeByAxisValueAndFormat>(GetHWCNRangeByAxisValue)},
     {ge::FORMAT_C1HWNCoC0, std::make_shared<GetNewRangeByAxisValueAndFormat>(GetC1HWNCoC0RangeByAxisValue)},
     {ge::FORMAT_CHWN, std::make_shared<GetNewRangeByAxisValueAndFormat>(GetCHWNRangeByAxisValue)},
     {ge::FORMAT_FRACTAL_NZ, std::make_shared<GetNewRangeByAxisValueAndFormat>(GetNzRangeByAxisValue)},
     {ge::FORMAT_FRACTAL_Z_3D, std::make_shared<GetNewRangeByAxisValueAndFormat>(GetFz3DRangeByAxisValue)},
     {ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE,
      std::make_shared<GetNewRangeByAxisValueAndFormat>(GetFz3DTransposeRangeByAxisValue)},
     {ge::FORMAT_NDC1HWC0, std::make_shared<GetNewRangeByAxisValueAndFormat>(GetNDC1HWC0RangeByAxisValue)},
     {ge::FORMAT_FRACTAL_ZN_RNN, std::make_shared<GetNewRangeByAxisValueAndFormat>(GetFznRNNRangeByAxisValue)},
     {ge::FORMAT_ND_RNN_BIAS, std::make_shared<GetNewRangeByAxisValueAndFormat>(GetNDRNNRangeByAxisValue)}};

Status RangeTransferAccordingToFormat::GetNCHWRangeByAxisValue(
    vector<std::pair<int64_t, int64_t>> &new_range, const int64_t &impl_type,
    const vector<std::pair<int64_t, int64_t>> &range_value, const vector<std::pair<int64_t, int64_t>> &nd_range_value) {
  (void)impl_type;
  (void)nd_range_value;
  new_range.clear();
  if (range_value.empty()) {
    FE_LOGW("AxisValue is empty.");
    return SUCCESS;
  }
  /* axis_value is initialized as a size 6 vector. */
  new_range.emplace_back(range_value[AXIS_N]);
  new_range.emplace_back(range_value[AXIS_C]);
  new_range.emplace_back(range_value[AXIS_H]);
  new_range.emplace_back(range_value[AXIS_W]);
  return SUCCESS;
}

Status RangeTransferAccordingToFormat::GetNHWCRangeByAxisValue(
    vector<std::pair<int64_t, int64_t>> &new_range, const int64_t &impl_type,
    const vector<std::pair<int64_t, int64_t>> &range_value, const vector<std::pair<int64_t, int64_t>> &nd_range_value) {
  (void)nd_range_value;
  (void)impl_type;
  new_range.clear();
  if (range_value.empty()) {
    FE_LOGW("AxisValue is empty.");
    return SUCCESS;
  }
  /* axis_value is initialized as a size 6 vector. */
  new_range.emplace_back(range_value[AXIS_N]);
  new_range.emplace_back(range_value[AXIS_H]);
  new_range.emplace_back(range_value[AXIS_W]);
  new_range.emplace_back(range_value[AXIS_C]);
  return SUCCESS;
}

Status RangeTransferAccordingToFormat::GetFzWinoRangeByAxisValue(
    vector<std::pair<int64_t, int64_t>> &new_range, const int64_t &impl_type,
    const vector<std::pair<int64_t, int64_t>> &range_value, const vector<std::pair<int64_t, int64_t>> &nd_range_value) {
  (void)impl_type;
  (void)nd_range_value;
  new_range.clear();
  if (range_value.empty()) {
    FE_LOGW("AxisValue is empty.");
    return SUCCESS;
  }

  // C1, N1, 2, HW, 8, C0
  new_range.emplace_back(range_value[AXIS_C1]);
  int64_t n1_lower = DivisionCeiling(range_value[AXIS_N].first, SHAPE_NUMBER_16);
  int64_t n1_higher = DivisionCeiling(range_value[AXIS_N].second, SHAPE_NUMBER_16);
  new_range.emplace_back(std::pair<int64_t, int64_t>(n1_lower, n1_higher));
  new_range.emplace_back(std::pair<int64_t, int64_t>(SHAPE_NUMBER_2, SHAPE_NUMBER_2));
  FE_INT64_MULCHECK(range_value[AXIS_H].first, range_value[AXIS_W].first);
  FE_INT64_MULCHECK(range_value[AXIS_H].second, range_value[AXIS_W].second);
  new_range.emplace_back(std::pair<int64_t, int64_t>(range_value[AXIS_H].first * range_value[AXIS_W].first,
                                                     range_value[AXIS_H].second * range_value[AXIS_W].second));
  new_range.emplace_back(std::pair<int64_t, int64_t>(SHAPE_NUMBER_8, SHAPE_NUMBER_8));
  new_range.emplace_back(range_value[AXIS_C0]);
  return SUCCESS;
}

Status RangeTransferAccordingToFormat::GetNC1HWC0RangeByAxisValue(
    vector<std::pair<int64_t, int64_t>> &new_range, const int64_t &impl_type,
    const vector<std::pair<int64_t, int64_t>> &range_value, const vector<std::pair<int64_t, int64_t>> &nd_range_value) {
  (void)nd_range_value;
  new_range.clear();
  if (range_value.empty()) {
    FE_LOGW("AxisValue is empty.");
    return SUCCESS;
  }
  /* axis_value is initialized as a size 6 vector. */
  if (impl_type == EN_IMPL_HW_TBE || impl_type == EN_IMPL_CUSTOM_TBE ||
      impl_type == EN_IMPL_NON_PERSISTENT_CUSTOM_TBE) {
    new_range.emplace_back(range_value[AXIS_N]);
    new_range.emplace_back(range_value[AXIS_C1]);
    new_range.emplace_back(range_value[AXIS_H]);
    new_range.emplace_back(range_value[AXIS_W]);
    new_range.emplace_back(range_value[AXIS_C0]);
  } else {
    new_range.emplace_back(range_value[AXIS_N]);
    new_range.emplace_back(range_value[AXIS_C]);
    new_range.emplace_back(range_value[AXIS_H]);
    new_range.emplace_back(range_value[AXIS_W]);
  }
  return SUCCESS;
}

Status RangeTransferAccordingToFormat::GetC1HWC0RangeByAxisValue(
    vector<std::pair<int64_t, int64_t>> &new_range, const int64_t &impl_type,
    const vector<std::pair<int64_t, int64_t>> &range_value, const vector<std::pair<int64_t, int64_t>> &nd_range_value) {
  (void)nd_range_value;
  (void)impl_type;
  new_range.clear();
  if (range_value.empty()) {
    FE_LOGW("AxisValue is empty.");
    return SUCCESS;
  }
  new_range.emplace_back(range_value[AXIS_C1]);
  new_range.emplace_back(range_value[AXIS_H]);
  new_range.emplace_back(range_value[AXIS_W]);
  new_range.emplace_back(range_value[AXIS_C0]);
  return SUCCESS;
}

Status RangeTransferAccordingToFormat::GetFzRangeByAxisValue(
    vector<std::pair<int64_t, int64_t>> &new_range, const int64_t &impl_type,
    const vector<std::pair<int64_t, int64_t>> &range_value, const vector<std::pair<int64_t, int64_t>> &nd_range_value) {
  (void)impl_type;
  new_range.clear();
  if (range_value.empty()) {
    FE_LOGW("AxisValue is empty.");
    return SUCCESS;
  }
  if (nd_range_value.empty()) {
    FE_LOGW("ndRangeValue is empty.");
    return SUCCESS;
  }
  /* axis_value is initialized as a size 6 vector. */
  if (nd_range_value.size() == SIZE_OF_CN) {
    auto size_of_original_vec = nd_range_value.size();
    new_range = nd_range_value;
    if (new_range.size() < MINIMUM_NZ_SHAPE_DIM_NUM) {
      FE_LOGW("ndValue's dim num is less than 2!");
      return SUCCESS;
    }
    /* size_of_original_vec - 1 mean the last value of original vec
     * size_of_original_vec - 2 mean the second last value of original vec */
    new_range[size_of_original_vec - MINUS_VALUE_ONE] = std::pair<int64_t, int64_t>(
        DivisionCeiling(nd_range_value[size_of_original_vec - MINUS_VALUE_TWO].first,
                        SHAPE_NUMBER_16),
        DivisionCeiling(nd_range_value[size_of_original_vec - MINUS_VALUE_TWO].second,
                        SHAPE_NUMBER_16));

    new_range[size_of_original_vec - MINUS_VALUE_TWO] = std::pair<int64_t, int64_t>(
        DivisionCeiling(nd_range_value[size_of_original_vec - MINUS_VALUE_ONE].first, range_value[AXIS_C0].first),
        DivisionCeiling(nd_range_value[size_of_original_vec - MINUS_VALUE_ONE].second, range_value[AXIS_C0].second));

    new_range.emplace_back(std::pair<int64_t, int64_t>(SHAPE_NUMBER_16, SHAPE_NUMBER_16));
    new_range.emplace_back(std::pair<int64_t, int64_t>(range_value[AXIS_C0].first, range_value[AXIS_C0].second));
  } else {
    FE_INT64_MULCHECK(range_value[AXIS_C1].first, range_value[AXIS_H].first);
    int64_t hwc1_first_range = range_value[AXIS_C1].first * range_value[AXIS_H].first;
    FE_INT64_MULCHECK(hwc1_first_range, range_value[AXIS_W].first);
    hwc1_first_range *= range_value[AXIS_W].first;

    FE_INT64_MULCHECK(range_value[AXIS_C1].second, range_value[AXIS_H].second);
    int64_t hwc1_second_range = range_value[AXIS_C1].second * range_value[AXIS_H].second;
    FE_INT64_MULCHECK(hwc1_second_range, range_value[AXIS_W].second);
    hwc1_second_range *= range_value[AXIS_W].second;

    if (range_value[AXIS_C1].second == UNKNOWN_SHAPE_VALUE || range_value[AXIS_H].second == UNKNOWN_SHAPE_VALUE ||
        range_value[AXIS_W].second == UNKNOWN_SHAPE_VALUE) {
      hwc1_first_range = 1;
      hwc1_second_range = UNKNOWN_SHAPE_VALUE;
    }
    new_range.emplace_back(std::pair<int64_t, int64_t>(hwc1_first_range, hwc1_second_range));
    new_range.emplace_back(std::pair<int64_t, int64_t>(DivisionCeiling(range_value[AXIS_N].first, NI),
                                                       DivisionCeiling(range_value[AXIS_N].second, NI)));
    new_range.emplace_back(std::pair<int64_t, int64_t>(NI, NI));
    new_range.emplace_back(range_value[AXIS_C0]);
  }
  return SUCCESS;
}

Status RangeTransferAccordingToFormat::GetFzC04RangeByAxisValue(
    vector<std::pair<int64_t, int64_t>> &new_range, const int64_t &impl_type,
    const vector<std::pair<int64_t, int64_t>> &range_value, const vector<std::pair<int64_t, int64_t>> &nd_range_value) {
  (void)nd_range_value;
  new_range.clear();
  if (range_value.empty()) {
    FE_LOGW("AxisValue is empty.");
    return SUCCESS;
  }
  /* axis_value is initialized as a size 6 vector. */
  if (impl_type == EN_IMPL_HW_TBE || impl_type == EN_IMPL_CUSTOM_TBE) {
    FE_INT64_MULCHECK(SHAPE_DIM_VALUE_C04, range_value[AXIS_H].first);
    int64_t x_first_range = SHAPE_DIM_VALUE_C04 * range_value[AXIS_H].first;
    FE_INT64_MULCHECK(x_first_range, range_value[AXIS_W].first);
    x_first_range *= range_value[AXIS_W].first;

    FE_INT64_MULCHECK(SHAPE_DIM_VALUE_C04, range_value[AXIS_H].second);
    int64_t x_second_range = SHAPE_DIM_VALUE_C04 * range_value[AXIS_H].second;
    FE_INT64_MULCHECK(x_second_range, range_value[AXIS_W].second);
    x_second_range *= range_value[AXIS_W].second;
    if (range_value[AXIS_H].second == UNKNOWN_SHAPE_VALUE || range_value[AXIS_W].second == UNKNOWN_SHAPE_VALUE) {
      x_first_range = 1;
      x_second_range = UNKNOWN_SHAPE_VALUE;
    }
    std::pair<int64_t, int64_t> x(x_first_range, x_second_range);
    new_range.emplace_back(std::pair<int64_t, int64_t>(DivisionCeiling(x.first, range_value[AXIS_C0].first),
                                                       DivisionCeiling(x.second, range_value[AXIS_C0].second)));

    new_range.emplace_back(std::pair<int64_t, int64_t>(DivisionCeiling(range_value[AXIS_N].first, NI),
                                                       DivisionCeiling(range_value[AXIS_N].second, NI)));
    new_range.emplace_back(std::pair<int64_t, int64_t>(NI, NI));
    new_range.emplace_back(range_value[AXIS_C0]);
  } else {
    new_range.emplace_back(range_value[AXIS_N]);
    new_range.emplace_back(range_value[AXIS_C]);
    new_range.emplace_back(range_value[AXIS_H]);
    new_range.emplace_back(range_value[AXIS_W]);
  }
  return SUCCESS;
}

Status RangeTransferAccordingToFormat::GetHWCNRangeByAxisValue(
    vector<std::pair<int64_t, int64_t>> &new_range, const int64_t &impl_type,
    const vector<std::pair<int64_t, int64_t>> &range_value, const vector<std::pair<int64_t, int64_t>> &nd_range_value) {
  (void)nd_range_value;
  (void)impl_type;
  if (range_value.empty()) {
    FE_LOGW("AxisValue is empty.");
    return SUCCESS;
  }
  /* axis_value is initialized as a size 6 vector. */
  new_range.emplace_back(range_value[AXIS_H]);
  new_range.emplace_back(range_value[AXIS_W]);
  new_range.emplace_back(range_value[AXIS_C]);
  new_range.emplace_back(range_value[AXIS_N]);
  return SUCCESS;
}

Status RangeTransferAccordingToFormat::GetCHWNRangeByAxisValue(
    vector<std::pair<int64_t, int64_t>> &new_range, const int64_t &impl_type,
    const vector<std::pair<int64_t, int64_t>> &range_value, const vector<std::pair<int64_t, int64_t>> &nd_range_value) {
  (void)nd_range_value;
  (void)impl_type;
  new_range.clear();
  if (range_value.empty()) {
    FE_LOGW("AxisValue is empty.");
    return SUCCESS;
  }
  /* axis_value is initialized as a size 6 vector. */
  new_range.emplace_back(range_value[AXIS_C]);
  new_range.emplace_back(range_value[AXIS_H]);
  new_range.emplace_back(range_value[AXIS_W]);
  new_range.emplace_back(range_value[AXIS_N]);
  return SUCCESS;
}

Status RangeTransferAccordingToFormat::GetC1HWNCoC0RangeByAxisValue(
    vector<std::pair<int64_t, int64_t>> &new_range, const int64_t &impl_type,
    const vector<std::pair<int64_t, int64_t>> &range_value, const vector<std::pair<int64_t, int64_t>> &nd_range_value) {
  (void)impl_type;
  (void)nd_range_value;
  new_range.clear();
  if (range_value.empty()) {
    FE_LOGW("AxisValue is empty.");
    return SUCCESS;
  }
  new_range.emplace_back(range_value[AXIS_C1]);
  new_range.emplace_back(range_value[AXIS_H]);
  new_range.emplace_back(range_value[AXIS_W]);
  new_range.emplace_back(range_value[AXIS_N]);
  new_range.emplace_back(range_value[AXIS_Co]);
  new_range.emplace_back(range_value[AXIS_C0]);
  return SUCCESS;
}

Status RangeTransferAccordingToFormat::GetNzRangeByAxisValue(
    vector<std::pair<int64_t, int64_t>> &new_range, const int64_t &impl_type,
    const vector<std::pair<int64_t, int64_t>> &range_value, const vector<std::pair<int64_t, int64_t>> &nd_range_value) {
  (void)impl_type;
  if (nd_range_value.empty()) {
    FE_LOGW("ND dim value format is missing!");
    return SUCCESS;
  }
  if (range_value.empty() || range_value.size() <= AXIS_C0) {
    FE_LOGW("AxisValue is empty or its size %zu is less than or equal to AXIS_C0[%u].", range_value.size(), AXIS_C0);
    return SUCCESS;
  }
  size_t size_of_original_vec = nd_range_value.size();
  vector<std::pair<int64_t, int64_t>> nd_range_value_temp = nd_range_value;
  if (size_of_original_vec < MINIMUM_NZ_SHAPE_DIM_NUM) {
    FE_LOGW("Format ND dim num is less than 2!");
    // Add one dimension at the end
    nd_range_value_temp.emplace_back(std::make_pair(1, 1));
    size_of_original_vec++;
  }

  new_range = nd_range_value_temp;
  /* size_of_original_vec - 1 mean the last value of original vec
   * size_of_original_vec - 2 mean the second last value of original vec */
  new_range[size_of_original_vec - MINUS_VALUE_ONE] = std::pair<int64_t, int64_t>(
      DivisionCeiling(nd_range_value_temp[size_of_original_vec - MINUS_VALUE_TWO].first,
                      SHAPE_NUMBER_16),
      DivisionCeiling(nd_range_value_temp[size_of_original_vec - MINUS_VALUE_TWO].second,
                      SHAPE_NUMBER_16));

  new_range[size_of_original_vec - MINUS_VALUE_TWO] = std::pair<int64_t, int64_t>(
      DivisionCeiling(nd_range_value_temp[size_of_original_vec - MINUS_VALUE_ONE].first, range_value[AXIS_C0].first),
      DivisionCeiling(nd_range_value_temp[size_of_original_vec - MINUS_VALUE_ONE].second, range_value[AXIS_C0].second));

  new_range.emplace_back(std::pair<int64_t, int64_t>(SHAPE_NUMBER_16, SHAPE_NUMBER_16));
  new_range.emplace_back(std::pair<int64_t, int64_t>(range_value[AXIS_C0].first, range_value[AXIS_C0].second));
  return SUCCESS;
}

Status RangeTransferAccordingToFormat::GetNDC1HWC0RangeByAxisValue(
    vector<std::pair<int64_t, int64_t>> &new_range, const int64_t &impl_type,
    const vector<std::pair<int64_t, int64_t>> &range_value, const vector<std::pair<int64_t, int64_t>> &nd_range_value) {
  (void)impl_type;
  (void)nd_range_value;
  new_range.clear();
  if (range_value.empty()) {
    FE_LOGW("AxisValue is empty.");
    return SUCCESS;
  }
  /* axis_value is initialized as a size 6 vector. */
  new_range.emplace_back(range_value[AXIS_N]);
  new_range.emplace_back(range_value[AXIS_D]);
  new_range.emplace_back(range_value[AXIS_C1]);
  new_range.emplace_back(range_value[AXIS_H]);
  new_range.emplace_back(range_value[AXIS_W]);
  new_range.emplace_back(range_value[AXIS_C0]);
  return SUCCESS;
}

Status RangeTransferAccordingToFormat::GetFz3DRangeByAxisValue(
    vector<std::pair<int64_t, int64_t>> &new_range, const int64_t &impl_type,
    const vector<std::pair<int64_t, int64_t>> &range_value, const vector<std::pair<int64_t, int64_t>> &nd_range_value) {
  (void)impl_type;
  (void)nd_range_value;
  new_range.clear();
  if (range_value.empty()) {
    FE_LOGW("AxisValue is empty.");
    return SUCCESS;
  }
  /* range_value is initialized as a size 6 vector. */
  FE_INT64_MULCHECK(range_value[AXIS_C1].first, range_value[AXIS_D].first);

  int64_t dhwc1_first_range = range_value[AXIS_C1].first * range_value[AXIS_D].first;
  FE_INT64_MULCHECK(dhwc1_first_range, range_value[AXIS_H].first);
  dhwc1_first_range *= range_value[AXIS_H].first;
  FE_INT64_MULCHECK(dhwc1_first_range, range_value[AXIS_W].first);
  dhwc1_first_range *= range_value[AXIS_W].first;

  FE_INT64_MULCHECK(range_value[AXIS_C1].second, range_value[AXIS_D].second);
  int64_t dhwc1_second_range = range_value[AXIS_C1].second * range_value[AXIS_D].second;
  FE_INT64_MULCHECK(dhwc1_second_range, range_value[AXIS_H].second);
  dhwc1_second_range *= range_value[AXIS_H].second;
  FE_INT64_MULCHECK(dhwc1_second_range, range_value[AXIS_W].second);
  dhwc1_second_range *= range_value[AXIS_W].second;

  if (range_value[AXIS_D].second == UNKNOWN_SHAPE_VALUE || range_value[AXIS_H].second == UNKNOWN_SHAPE_VALUE ||
      range_value[AXIS_W].second == UNKNOWN_SHAPE_VALUE || range_value[AXIS_C1].second == UNKNOWN_SHAPE_VALUE) {
    dhwc1_second_range = UNKNOWN_SHAPE_VALUE;
  }
  new_range.emplace_back(std::pair<int64_t, int64_t>(dhwc1_first_range, dhwc1_second_range));
  new_range.emplace_back(std::pair<int64_t, int64_t>(DivisionCeiling(range_value[AXIS_N].first, NI),
                                                     DivisionCeiling(range_value[AXIS_N].second, NI)));
  new_range.emplace_back(std::pair<int64_t, int64_t>(NI, NI));
  new_range.emplace_back(std::pair<int64_t, int64_t>(range_value[AXIS_C0].first, range_value[AXIS_C0].second));
  return SUCCESS;
}

Status RangeTransferAccordingToFormat::GetFz3DTransposeRangeByAxisValue(
    vector<std::pair<int64_t, int64_t>> &new_range, const int64_t &impl_type,
    const vector<std::pair<int64_t, int64_t>> &range_value, const vector<std::pair<int64_t, int64_t>> &nd_range_value) {
  (void)impl_type;
  (void)nd_range_value;
  new_range.clear();
  if (range_value.empty()) {
    FE_LOGW("AxisValue is empty.");
    return SUCCESS;
  }
  /* range_value is initialized as a size 6 vector. */
  int64_t n1_first = DivisionCeiling(range_value[AXIS_N].first, NI);
  int64_t n1_secend = DivisionCeiling(range_value[AXIS_N].second, NI);

  FE_INT64_MULCHECK(n1_first, range_value[AXIS_H].first);
  int64_t dhwn1_first = n1_first * range_value[AXIS_H].first;
  FE_INT64_MULCHECK(dhwn1_first, range_value[AXIS_W].first);
  dhwn1_first *= range_value[AXIS_W].first;
  FE_INT64_MULCHECK(dhwn1_first, range_value[AXIS_D].first);
  dhwn1_first *= range_value[AXIS_D].first;

  FE_INT64_MULCHECK(n1_secend, range_value[AXIS_H].second);
  int64_t dhwn1_second = n1_secend * range_value[AXIS_H].second;
  FE_INT64_MULCHECK(dhwn1_second, range_value[AXIS_W].second);
  dhwn1_second *= range_value[AXIS_W].second;
  FE_INT64_MULCHECK(dhwn1_second, range_value[AXIS_D].second);
  dhwn1_second *= range_value[AXIS_D].second;
  if (range_value[AXIS_D].second == UNKNOWN_SHAPE_VALUE || range_value[AXIS_H].second == UNKNOWN_SHAPE_VALUE ||
      range_value[AXIS_W].second == UNKNOWN_SHAPE_VALUE) {
    dhwn1_second = UNKNOWN_SHAPE_VALUE;
  }
  new_range.emplace_back(std::pair<int64_t, int64_t>(dhwn1_first, dhwn1_second));
  new_range.emplace_back(std::pair<int64_t, int64_t>(range_value[AXIS_C1].first, range_value[AXIS_C1].second));
  new_range.emplace_back(std::pair<int64_t, int64_t>(NI, NI));
  new_range.emplace_back(std::pair<int64_t, int64_t>(range_value[AXIS_C0].first, range_value[AXIS_C0].second));

  return SUCCESS;
}

Status RangeTransferAccordingToFormat::GetFznRNNRangeByAxisValue(
    vector<std::pair<int64_t, int64_t>> &new_range, const int64_t &impl_type,
    const vector<std::pair<int64_t, int64_t>> &range_value, const vector<std::pair<int64_t, int64_t>> &nd_range_value) {
  (void)impl_type;
  new_range = nd_range_value;
  if (nd_range_value.empty()) {
    FE_LOGW("ND dim value format is missing!");
    return SUCCESS;
  }
  if (range_value.empty() || range_value.size() <= AXIS_C0) {
    FE_LOGW("AxisValue is empty or its size %zu is less than or equal to AXIS_C0[%u]", range_value.size(), AXIS_C0);
    return SUCCESS;
  }
  size_t size_of_original_vec = new_range.size();
  if (size_of_original_vec < MINIMUM_ND_TO_RNN_SHAPE_NUM) {
    FE_LOGW("Format ND dim num is less than 2!");
    return SUCCESS;
  }
  /* check nd shape value */
  std::pair<int64_t, int64_t> k_value_range = nd_range_value[size_of_original_vec - MINUS_VALUE_TWO];
  std::pair<int64_t, int64_t> n_value_range = nd_range_value[size_of_original_vec - MINUS_VALUE_ONE];
  FE_INT64_ZEROCHECK(range_value[AXIS_HIDEEN_SIZE].first);
  FE_INT64_ZEROCHECK(range_value[AXIS_HIDEEN_SIZE].second);
  std::pair<int64_t, int64_t> n_num_range =
      std::pair<int64_t, int64_t>(n_value_range.first / range_value[AXIS_HIDEEN_SIZE].first,
                                  n_value_range.second / range_value[AXIS_HIDEEN_SIZE].second);

  /* size_of_original_vec - 1 mean the last value of original vec
   * size_of_original_vec - 2 mean the second last value of original vec */
  int64_t min_first_dim_value =
      min(DivisionCeiling(k_value_range.first, SHAPE_NUMBER_16),
          DivisionCeiling(range_value[AXIS_INPUT_SIZE].first, SHAPE_NUMBER_16) +
              DivisionCeiling(range_value[AXIS_HIDEEN_SIZE].first, SHAPE_NUMBER_16));
  int64_t max_first_dim_value =
      max(DivisionCeiling(k_value_range.second, SHAPE_NUMBER_16),
          (DivisionCeiling(range_value[AXIS_INPUT_SIZE].first, SHAPE_NUMBER_16) +
           DivisionCeiling(range_value[AXIS_HIDEEN_SIZE].first, SHAPE_NUMBER_16)));

  new_range[size_of_original_vec - MINUS_VALUE_TWO] =
      std::pair<int64_t, int64_t>(min_first_dim_value, max_first_dim_value);
  FE_INT64_MULCHECK(n_num_range.first,
                    DivisionCeiling(range_value[AXIS_HIDEEN_SIZE].first, range_value[AXIS_C0].first));
  FE_INT64_MULCHECK(n_num_range.second,
                    DivisionCeiling(range_value[AXIS_HIDEEN_SIZE].second, range_value[AXIS_C0].second));
  new_range[size_of_original_vec - MINUS_VALUE_ONE] = std::pair<int64_t, int64_t>(
      n_num_range.first * DivisionCeiling(range_value[AXIS_HIDEEN_SIZE].first, range_value[AXIS_C0].first),
      n_num_range.second * DivisionCeiling(range_value[AXIS_HIDEEN_SIZE].second, range_value[AXIS_C0].second));
  new_range.emplace_back(std::pair<int64_t, int64_t>(SHAPE_NUMBER_16, SHAPE_NUMBER_16));
  new_range.emplace_back(std::pair<int64_t, int64_t>(range_value[AXIS_C0].first, range_value[AXIS_C0].second));
  return SUCCESS;
}

Status RangeTransferAccordingToFormat::GetNDRNNRangeByAxisValue(
    vector<std::pair<int64_t, int64_t>> &new_range, const int64_t &impl_type,
    const vector<std::pair<int64_t, int64_t>> &range_value, const vector<std::pair<int64_t, int64_t>> &nd_range_value) {
  (void)impl_type;
  new_range = nd_range_value;
  if (nd_range_value.empty()) {
    FE_LOGW("ND dim value format is missing!");
    return SUCCESS;
  }
  if (range_value.empty() || range_value.size() <= AXIS_C0) {
    FE_LOGW("AxisValue is empty or its size %zu is less than or equal to AXIS_C0[%u]", range_value.size(), AXIS_C0);
    return SUCCESS;
  }
  size_t size_of_original_vec = new_range.size();

  /* check nd shape value */
  std::pair<int64_t, int64_t> n_num_range;
  std::pair<int64_t, int64_t> n_value_range = new_range[size_of_original_vec - MINUS_VALUE_ONE];
  FE_INT64_ZEROCHECK(range_value[AXIS_INPUT_SIZE].first);
  FE_INT64_ZEROCHECK(range_value[AXIS_HIDEEN_SIZE].second);
  n_num_range = std::make_pair(n_value_range.first / range_value[AXIS_HIDEEN_SIZE].first,
                               n_value_range.second / range_value[AXIS_HIDEEN_SIZE].second);
  FE_INT64_MULCHECK(n_num_range.first,
                    DivisionCeiling(range_value[AXIS_HIDEEN_SIZE].first, range_value[AXIS_C0].first));
  FE_INT64_MULCHECK(n_num_range.second,
                    DivisionCeiling(range_value[AXIS_HIDEEN_SIZE].second, range_value[AXIS_C0].second));
  new_range[size_of_original_vec - MINUS_VALUE_ONE] = std::pair<int64_t, int64_t>(
      n_num_range.first * DivisionCeiling(range_value[AXIS_HIDEEN_SIZE].first, range_value[AXIS_C0].first) *
      range_value[AXIS_C0].first,
      n_num_range.second * DivisionCeiling(range_value[AXIS_HIDEEN_SIZE].second, range_value[AXIS_C0].second) *
      range_value[AXIS_C0].second);
  return SUCCESS;
}

void RangeTransferAccordingToFormat::SetRNNRangeAttr(const RangeAndFormat &range_and_format_info,
                                                     std::vector<std::pair<int64_t, int64_t>> &range_value) {
  if (range_and_format_info.new_format != ge::FORMAT_FRACTAL_ZN_RNN &&
      range_and_format_info.new_format != ge::FORMAT_ND_RNN_BIAS) {
    return;
  }
  if (range_value.size() < AXIS_BOTTOM) {
    return;
  }
  range_value[AXIS_INPUT_SIZE] = std::pair<int64_t, int64_t>(range_and_format_info.extra_attr.input_size,
                                                             range_and_format_info.extra_attr.input_size);
  range_value[AXIS_HIDEEN_SIZE] = std::pair<int64_t, int64_t>(range_and_format_info.extra_attr.hidden_size,
                                                              range_and_format_info.extra_attr.hidden_size);
}

Status RangeTransferAccordingToFormat::GetRangeAccordingToFormat(RangeAndFormat &range_and_format_info) {
  /* The default new range is old range */
  range_and_format_info.new_range = range_and_format_info.old_range;
  if (range_and_format_info.old_format >= ge::FORMAT_END || range_and_format_info.new_format >= ge::FORMAT_END) {
    REPORT_FE_ERROR("[GraphOptJdgInst][RangTrans][GetRange] Invalid format: %s (old) or %s (new)!",
                    ge::TypeUtils::FormatToSerialString(range_and_format_info.old_format).c_str(),
                    ge::TypeUtils::FormatToSerialString(range_and_format_info.new_format).c_str());
    return FAILED;
  }

  if (range_and_format_info.current_data_type == ge::DT_UNDEFINED ||
      range_and_format_info.current_data_type >= ge::DT_MAX) {
    REPORT_FE_ERROR("[GraphOptJdgInst][RangTrans][GetRange] Invalid currentDataType: %s!",
                    ge::TypeUtils::DataTypeToSerialString(range_and_format_info.current_data_type).c_str());
    return FAILED;
  }
  if (!RangeAxisUtil::HasAxisValueFunc(range_and_format_info.old_format)) {
    return SUCCESS;
  }

  auto iter_get_new_range_func = get_new_range_func_map.find(range_and_format_info.new_format);
  if (iter_get_new_range_func == get_new_range_func_map.end()) {
    FE_LOGW("Cannot get the new shape of the new format %u!", range_and_format_info.new_format);
    return SUCCESS;
  }
  FE_LOGD("Original format %u, new format %u.", range_and_format_info.old_format, range_and_format_info.new_format);

  std::vector<std::pair<int64_t, int64_t>> range_value;
  for (uint32_t i = 0; i < AXIS_BOTTOM; i++) {
    range_value.emplace_back(std::pair<int64_t, int64_t>(1, 1));
  }

  int64_t c0 = GetC0ValByDataType(range_and_format_info.current_data_type);
  if (ge::HasC0Format(range_and_format_info.new_format)) {
    c0 = ge::GetC0Value(range_and_format_info.new_format);
  }
  // The value of C0 should be 4 while format is 5HD-4 or FRAZ-4
  if (range_and_format_info.new_format == ge::FORMAT_NC1HWC0_C04) {
    c0 = SHAPE_DIM_VALUE_C04;
  }
  FE_LOGD("range_and_format_info.old_range = %s.", ShapeRangeToStr(range_and_format_info.old_range).c_str());
  Status status =
      RangeAxisUtil::GetRangeAxisValueByOriginFormat(range_and_format_info.old_range, range_and_format_info.old_format,
                                                     range_and_format_info.old_shape.GetDims(), c0, range_value);
  if (status != SUCCESS && range_and_format_info.new_format != ge::FORMAT_FRACTAL_NZ) {
    return SUCCESS;
  }

  SetRNNRangeAttr(range_and_format_info, range_value);
  std::vector<std::pair<int64_t, int64_t>> nd_range_value =
      GetAlignShapeRange(range_and_format_info.old_range, range_and_format_info.old_shape);
  GetNewRangeByAxisValueAndFormatPtr get_new_range_func = nullptr;
  FE_MAKE_SHARED(get_new_range_func = iter_get_new_range_func->second, return FAILED);
  FE_CHECK_NOTNULL(get_new_range_func);
  int64_t main_type = GetMainImplType<int64_t>(range_and_format_info.op_impl_type);
  (*get_new_range_func)(range_and_format_info.new_range, main_type, range_value, nd_range_value);
  return SUCCESS;
}
};  // namespace fe

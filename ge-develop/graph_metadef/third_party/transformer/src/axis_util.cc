/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "axis_util.h"
#include "axis_constants.h"
#include "framework/common/debug/ge_log.h"
#include "expand_dimension.h"
#include "graph/utils/type_utils.h"

namespace transformer {
const size_t DIM_SIZE_TWO = 2;
const size_t DIM_SIZE_FOUR = 4;
const size_t DIM_SIZE_FIVE = 5;
const size_t DIM_SIZE_SIX = 6;

const size_t EXT_INDEX_INPUT_SIZE = 0;
const size_t EXT_INDEX_HIDDEN_SIZE = 1;
const size_t EXT_INDEX_STATE_SIZE = 2;
const size_t EXT_INDEX_M0_VAL = 3;

const int32_t AXIS_NCHW_DIM_N = 0;
const int32_t AXIS_NCHW_DIM_C = 1;
const int32_t AXIS_NCHW_DIM_H = 2;
const int32_t AXIS_NCHW_DIM_W = 3;

const int32_t AXIS_NHWC_DIM_N = 0;
const int32_t AXIS_NHWC_DIM_H = 1;
const int32_t AXIS_NHWC_DIM_W = 2;
const int32_t AXIS_NHWC_DIM_C = 3;

const int32_t AXIS_HWCN_DIM_H = 0;
const int32_t AXIS_HWCN_DIM_W = 1;
const int32_t AXIS_HWCN_DIM_C = 2;
const int32_t AXIS_HWCN_DIM_N = 3;

const int32_t AXIS_CHWN_DIM_C = 0;
const int32_t AXIS_CHWN_DIM_H = 1;
const int32_t AXIS_CHWN_DIM_W = 2;
const int32_t AXIS_CHWN_DIM_N = 3;

const int32_t NDHWC_DIM_N = 0;
const int32_t NDHWC_DIM_D = 1;
const int32_t NDHWC_DIM_H = 2;
const int32_t NDHWC_DIM_W = 3;
const int32_t NDHWC_DIM_C = 4;

const int32_t NCDHW_DIM_N = 0;
const int32_t NCDHW_DIM_C = 1;
const int32_t NCDHW_DIM_D = 2;
const int32_t NCDHW_DIM_H = 3;
const int32_t NCDHW_DIM_W = 4;

const int32_t DHWCN_DIM_D = 0;
const int32_t DHWCN_DIM_H = 1;
const int32_t DHWCN_DIM_W = 2;
const int32_t DHWCN_DIM_C = 3;
const int32_t DHWCN_DIM_N = 4;

const int32_t DHWNC_DIM_D = 0;
const int32_t DHWNC_DIM_H = 1;
const int32_t DHWNC_DIM_W = 2;
const int32_t DHWNC_DIM_N = 3;
const int32_t DHWNC_DIM_C = 4;

const int32_t AXIS_NC1HWC0_DIM_N = 0;
const int32_t AXIS_NC1HWC0_DIM_C1 = 1;
const int32_t AXIS_NC1HWC0_DIM_H = 2;
const int32_t AXIS_NC1HWC0_DIM_W = 3;
const int32_t AXIS_NC1HWC0_DIM_C0 = 4;

const int32_t AXIS_NDC1HWC0_DIM_N = 0;
const int32_t AXIS_NDC1HWC0_DIM_D = 1;
const int32_t AXIS_NDC1HWC0_DIM_C1 = 2;
const int32_t AXIS_NDC1HWC0_DIM_H = 3;
const int32_t AXIS_NDC1HWC0_DIM_W = 4;
const int32_t AXIS_NDC1HWC0_DIM_C0 = 5;

const int32_t AXIS_C1HWNCoC0_DIM_C1 = 0;
const int32_t AXIS_C1HWNCoC0_DIM_H = 1;
const int32_t AXIS_C1HWNCoC0_DIM_W = 2;
const int32_t AXIS_C1HWNCoC0_DIM_N = 3;
const int32_t AXIS_C1HWNCoC0_DIM_Co = 4;

const int32_t AXIS_FZ_DIM_C1HW = 0;
const int32_t AXIS_FZ_DIM_N1 = 1;
const int32_t AXIS_FZ_DIM_N0 = 2;
const int32_t AXIS_FZ_DIM_C0 = 3;

const std::map<ge::Format, std::map<std::string, int32_t>> kFormatAxisIndexMap = {
    {ge::Format::FORMAT_NCHW, {{"N", AXIS_NCHW_DIM_N}, {"C", AXIS_NCHW_DIM_C},
                               {"H", AXIS_NCHW_DIM_H}, {"W", AXIS_NCHW_DIM_W}}},
    {ge::Format::FORMAT_HWCN, {{"N", AXIS_HWCN_DIM_N}, {"C", AXIS_HWCN_DIM_C},
                               {"H", AXIS_HWCN_DIM_H}, {"W", AXIS_HWCN_DIM_W}}},
    {ge::Format::FORMAT_NHWC, {{"N", AXIS_NHWC_DIM_N}, {"C", AXIS_NHWC_DIM_C},
                               {"H", AXIS_NHWC_DIM_H}, {"W", AXIS_NHWC_DIM_W}}},
    {ge::Format::FORMAT_CHWN, {{"N", AXIS_CHWN_DIM_N}, {"C", AXIS_CHWN_DIM_C},
                               {"H", AXIS_CHWN_DIM_H}, {"W", AXIS_CHWN_DIM_W}}},
    {ge::Format::FORMAT_NDHWC, {{"N", NDHWC_DIM_N}, {"C", NDHWC_DIM_C},
                                {"H", NDHWC_DIM_H}, {"W", NDHWC_DIM_W}, {"D", NDHWC_DIM_D}}},
    {ge::Format::FORMAT_NCDHW, {{"N", NCDHW_DIM_N}, {"C", NCDHW_DIM_C},
                                {"H", NCDHW_DIM_H}, {"W", NCDHW_DIM_W}, {"D", NCDHW_DIM_D}}},
    {ge::Format::FORMAT_DHWCN, {{"N", DHWCN_DIM_N}, {"C", DHWCN_DIM_C},
                                {"H", DHWCN_DIM_H}, {"W", DHWCN_DIM_W}, {"D", DHWCN_DIM_D}}},
    {ge::Format::FORMAT_DHWNC, {{"N", DHWNC_DIM_N}, {"C", DHWNC_DIM_C},
                                {"H", DHWNC_DIM_H}, {"W", DHWNC_DIM_W}, {"D", DHWNC_DIM_D}}},
    {ge::Format::FORMAT_NC1HWC0, {{"N", AXIS_NC1HWC0_DIM_N}, {"C1", AXIS_NC1HWC0_DIM_C1},
                                  {"H", AXIS_NC1HWC0_DIM_H}, {"W", AXIS_NC1HWC0_DIM_W}, {"C0", AXIS_NC1HWC0_DIM_C0}}},
    {ge::Format::FORMAT_NDC1HWC0, {{"N", AXIS_NDC1HWC0_DIM_N}, {"D", AXIS_NDC1HWC0_DIM_D},
                                   {"C1", AXIS_NDC1HWC0_DIM_C1}, {"H", AXIS_NDC1HWC0_DIM_H},
                                   {"W", AXIS_NDC1HWC0_DIM_W}, {"C0", AXIS_NDC1HWC0_DIM_C0}}},
    {ge::Format::FORMAT_FRACTAL_Z, {{"C1HW", AXIS_FZ_DIM_C1HW}, {"N1", AXIS_FZ_DIM_N1},
                                    {"N0", AXIS_FZ_DIM_N0}, {"C0", AXIS_FZ_DIM_C0}}}};

const std::map<ge::Format, std::vector<std::string>> kFormatAxisVec = {
    {ge::Format::FORMAT_NCHW, {"N", "C", "H", "W"}},
    {ge::Format::FORMAT_HWCN, {"H", "W", "C", "N"}},
    {ge::Format::FORMAT_NHWC, {"N", "H", "W", "C"}},
    {ge::Format::FORMAT_CHWN, {"C", "H", "W", "N"}},
    {ge::Format::FORMAT_NDHWC, {"N", "D", "H", "W", "C"}},
    {ge::Format::FORMAT_NCDHW, {"N", "C", "D", "H", "W"}},
    {ge::Format::FORMAT_DHWCN, {"D", "H", "W", "C", "N"}},
    {ge::Format::FORMAT_DHWNC, {"D", "H", "W", "N", "C"}},
    {ge::Format::FORMAT_NC1HWC0, {"N", "C1", "H", "W", "C0"}},
    {ge::Format::FORMAT_NDC1HWC0, {"N", "D", "C1", "H", "W", "C0"}},
    {ge::Format::FORMAT_FRACTAL_Z, {"C1HW", "N1", "N0", "C0"}}};

const std::map<ge::Format, std::map<std::string, std::vector<std::string>>> kFormatSplitOrConcatAxisMap {
    {ge::Format::FORMAT_NC1HWC0, {{"C", {"C1", "C0"}}, {"C1", {"C"}}, {"C0", {"C"}}}},
    {ge::Format::FORMAT_NDC1HWC0, {{"C", {"C1", "C0"}}, {"C1", {"C"}}, {"C0", {"C"}}}},
    {ge::Format::FORMAT_FRACTAL_Z, {{"N", {"N1", "N0"}}, {"C", {"C1HW", "C0"}}, {"H", {"C1HW"}}, {"W", {"C1HW"}},
                                    {"C1HW", {"C", "H", "W"}}, {"N1", {"N"}}, {"N0", {"N"}}, {"C0", {"C"}}}}};

bool AxisUtil::GetAxisValueByOriginFormat(const ge::Format &format, const gert::Shape &shape, AxisValue &axis_value) {
  CHECK(shape.IsScalar(), GELOGI("Original dim vector is empty!"), return true);
  switch (format) {
    case ge::FORMAT_NCHW:
      return GetAxisValueByNCHW(shape, axis_value);
    case ge::FORMAT_NHWC:
      return GetAxisValueByNHWC(shape, axis_value);
    case ge::FORMAT_HWCN:
      return GetAxisValueByHWCN(shape, axis_value);
    case ge::FORMAT_ND:
      return GetAxisValueByND(shape, axis_value);
    case ge::FORMAT_NCDHW:
      return GetAxisValueByNCDHW(shape, axis_value);
    case ge::FORMAT_NDHWC:
      return GetAxisValueByNDHWC(shape, axis_value);
    case ge::FORMAT_DHWCN:
      return GetAxisValueByDHWCN(shape, axis_value);
    case ge::FORMAT_DHWNC:
      return GetAxisValueByDHWNC(shape, axis_value);
    case ge::FORMAT_NC1HWC0:
      return GetAxisValueByNC1HWC0(shape, axis_value);
    case ge::FORMAT_C1HWNCoC0:
      return GetAxisValueByC1HWNCoC0(shape, axis_value);
    default:
      GELOGI("Could not retrieve axis value for old format %d.", format);
      return false;
  }
}

bool AxisUtil::GetAxisValueByND(const gert::Shape &shape, AxisValue &axis_value) {
  /* To differentiate the input datatype of int8 and others */
  if (shape.GetDimNum() == DIM_SIZE_FOUR) {
    axis_value[AXIS_N] = shape.GetDim(AXIS_NCHW_DIM_N);
    axis_value[AXIS_C] = shape.GetDim(AXIS_NCHW_DIM_C);
    axis_value[AXIS_H] = shape.GetDim(AXIS_NCHW_DIM_H);
    axis_value[AXIS_W] = shape.GetDim(AXIS_NCHW_DIM_W);
    axis_value[AXIS_C1] = DivisionCeiling(axis_value[AXIS_C], axis_value[AXIS_C0]);
    axis_value[AXIS_Co] = axis_value[AXIS_C0];
  }
  return true;
}

bool AxisUtil::GetAxisValueByNCHW(const gert::Shape &shape, AxisValue &axis_value) {
  CHECK(shape.GetDimNum() < DIM_SIZE_FOUR, GELOGI("Dim size is less than 4."), return false);
  /* C0 Must be set for case ND or 2D-NCHW to NZ */
  axis_value[AXIS_N] = shape.GetDim(AXIS_NCHW_DIM_N);
  axis_value[AXIS_C] = shape.GetDim(AXIS_NCHW_DIM_C);
  axis_value[AXIS_H] = shape.GetDim(AXIS_NCHW_DIM_H);
  axis_value[AXIS_W] = shape.GetDim(AXIS_NCHW_DIM_W);
  axis_value[AXIS_C1] = DivisionCeiling(axis_value[AXIS_C], axis_value[AXIS_C0]);
  axis_value[AXIS_Co] = axis_value[AXIS_C0];
  return true;
}

bool AxisUtil::GetAxisValueByNHWC(const gert::Shape &shape, AxisValue &axis_value) {
  CHECK(shape.GetDimNum() < DIM_SIZE_FOUR, GELOGI("Dim size is less than 4."), return false);
  /* C0 Must be set for case ND or 2D-NHWC to NZ */
  axis_value[AXIS_N] = shape.GetDim(AXIS_NHWC_DIM_N);
  axis_value[AXIS_C] = shape.GetDim(AXIS_NHWC_DIM_C);
  axis_value[AXIS_H] = shape.GetDim(AXIS_NHWC_DIM_H);
  axis_value[AXIS_W] = shape.GetDim(AXIS_NHWC_DIM_W);
  axis_value[AXIS_C1] = DivisionCeiling(axis_value[AXIS_C], axis_value[AXIS_C0]);
  axis_value[AXIS_Co] = axis_value[AXIS_C0];
  return true;
}

bool AxisUtil::GetAxisValueByNC1HWC0(const gert::Shape &shape, AxisValue &axis_value) {
  CHECK(shape.GetDimNum() < DIM_SIZE_FOUR, GELOGI("Dim size is less than 4."), return false);
  if (shape.GetDimNum() == DIM_SIZE_FIVE) {
    axis_value[AXIS_C0] = shape.GetDim(AXIS_NC1HWC0_DIM_C0);
    axis_value[AXIS_C1] = shape.GetDim(AXIS_NC1HWC0_DIM_C1);
    axis_value[AXIS_C] = axis_value[AXIS_C1] * axis_value[AXIS_C0];
  } else {
    axis_value[AXIS_C] = shape.GetDim(AXIS_NCHW_DIM_C);
    axis_value[AXIS_C1] = DivisionCeiling(axis_value[AXIS_C], axis_value[AXIS_C0]);
  }

  axis_value[AXIS_N] = shape.GetDim(AXIS_NC1HWC0_DIM_N);
  axis_value[AXIS_H] = shape.GetDim(AXIS_NC1HWC0_DIM_H);
  axis_value[AXIS_W] = shape.GetDim(AXIS_NC1HWC0_DIM_W);
  return true;
}

bool AxisUtil::GetAxisValueByHWCN(const gert::Shape &shape, AxisValue &axis_value) {
  CHECK(shape.GetDimNum() < DIM_SIZE_FOUR, GELOGI("Dim size is less than 4."), return false);
  /* C0 Must be set for case ND or 2D-NHWC to NZ */
  axis_value[AXIS_N] = shape.GetDim(AXIS_HWCN_DIM_N);
  axis_value[AXIS_C] = shape.GetDim(AXIS_HWCN_DIM_C);
  axis_value[AXIS_H] = shape.GetDim(AXIS_HWCN_DIM_H);
  axis_value[AXIS_W] = shape.GetDim(AXIS_HWCN_DIM_W);
  axis_value[AXIS_C1] = DivisionCeiling(axis_value[AXIS_C], axis_value[AXIS_C0]);
  axis_value[AXIS_Co] = axis_value[AXIS_C0];
  return true;
}

bool AxisUtil::GetAxisValueByC1HWNCoC0(const gert::Shape &shape, AxisValue &axis_value) {
  CHECK(shape.GetDimNum() < DIM_SIZE_SIX, GELOGI("Dim size is less than 6."), return false);
  /* C0 Must be set for case ND or 2D-NHWC to NZ */
  axis_value[AXIS_N] = shape.GetDim(AXIS_C1HWNCoC0_DIM_N);
  axis_value[AXIS_C] = shape.GetDim(AXIS_C1HWNCoC0_DIM_C1) * axis_value[AXIS_C0];
  axis_value[AXIS_H] = shape.GetDim(AXIS_C1HWNCoC0_DIM_H);
  axis_value[AXIS_W] = shape.GetDim(AXIS_C1HWNCoC0_DIM_W);
  axis_value[AXIS_C1] = shape.GetDim(AXIS_C1HWNCoC0_DIM_C1);
  axis_value[AXIS_Co] = shape.GetDim(AXIS_C1HWNCoC0_DIM_Co);
  return true;
}

bool AxisUtil::GetAxisValueByNDHWC(const gert::Shape &shape, AxisValue &axis_value) {
  CHECK(shape.GetDimNum() < DIM_SIZE_FIVE, GELOGI("Dim size is less than 5."), return false);

  axis_value[AXIS_N] = shape.GetDim(NDHWC_DIM_N);
  axis_value[AXIS_C] = shape.GetDim(NDHWC_DIM_C);
  axis_value[AXIS_H] = shape.GetDim(NDHWC_DIM_H);
  axis_value[AXIS_W] = shape.GetDim(NDHWC_DIM_W);
  axis_value[AXIS_D] = shape.GetDim(NDHWC_DIM_D);
  axis_value[AXIS_C1] = DivisionCeiling(axis_value[AXIS_C], axis_value[AXIS_C0]);
  axis_value[AXIS_Co] = axis_value[AXIS_C0];
  return true;
}

bool AxisUtil::GetAxisValueByNCDHW(const gert::Shape &shape, AxisValue &axis_value) {
  CHECK(shape.GetDimNum() < DIM_SIZE_FIVE, GELOGI("Dim size is less than 5."), return false);

  axis_value[AXIS_N] = shape.GetDim(NCDHW_DIM_N);
  axis_value[AXIS_C] = shape.GetDim(NCDHW_DIM_C);
  axis_value[AXIS_H] = shape.GetDim(NCDHW_DIM_H);
  axis_value[AXIS_W] = shape.GetDim(NCDHW_DIM_W);
  axis_value[AXIS_D] = shape.GetDim(NCDHW_DIM_D);
  axis_value[AXIS_C1] = DivisionCeiling(axis_value[AXIS_C], axis_value[AXIS_C0]);
  axis_value[AXIS_Co] = axis_value[AXIS_C0];
  return true;
}

bool AxisUtil::GetAxisValueByDHWCN(const gert::Shape &shape, AxisValue &axis_value) {
  CHECK(shape.GetDimNum() < DIM_SIZE_FIVE, GELOGI("Dim size is less than 5."), return false);

  axis_value[AXIS_N] = shape.GetDim(DHWCN_DIM_N);
  axis_value[AXIS_C] = shape.GetDim(DHWCN_DIM_C);
  axis_value[AXIS_H] = shape.GetDim(DHWCN_DIM_H);
  axis_value[AXIS_W] = shape.GetDim(DHWCN_DIM_W);
  axis_value[AXIS_D] = shape.GetDim(DHWCN_DIM_D);
  axis_value[AXIS_C1] = DivisionCeiling(axis_value[AXIS_C], axis_value[AXIS_C0]);
  axis_value[AXIS_Co] = axis_value[AXIS_C0];
  return true;
}

bool AxisUtil::GetAxisValueByDHWNC(const gert::Shape &shape, AxisValue &axis_value) {
  CHECK(shape.GetDimNum() < DIM_SIZE_FIVE, GELOGI("Dim size is less than 5."), return false);
  axis_value[AXIS_N] = shape.GetDim(DHWNC_DIM_N);
  axis_value[AXIS_C] = shape.GetDim(DHWNC_DIM_C);
  axis_value[AXIS_H] = shape.GetDim(DHWNC_DIM_H);
  axis_value[AXIS_W] = shape.GetDim(DHWNC_DIM_W);
  axis_value[AXIS_D] = shape.GetDim(DHWNC_DIM_D);
  axis_value[AXIS_C1] = DivisionCeiling(axis_value[AXIS_C], axis_value[AXIS_C0]);
  axis_value[AXIS_Co] = axis_value[AXIS_C0];

  return true;
}

int32_t AxisUtil::GetAxisIndexByFormat(const ge::Format &format, const std::string &axis) {
  auto iter = kFormatAxisIndexMap.find(static_cast<ge::Format>(GetPrimaryFormat(format)));
  if (iter == kFormatAxisIndexMap.end()) {
    GELOGW("Does not support this format: %s.", ge::TypeUtils::FormatToSerialString(format).c_str());
    return -1;
  }
  auto iter2 = iter->second.find(axis);
  if (iter2 == iter->second.end()) {
    GELOGW("Format %s does not have this axis %s.", ge::TypeUtils::FormatToSerialString(format).c_str(),
           axis.c_str());
    return -1;
  }
  return iter2->second;
}

int32_t AxisUtil::GetAxisIndexByFormat(const ge::Format &format, const std::string &axis,
                                       const std::map<std::string, int32_t> &valid_axis_map) {
  int32_t axis_index = GetAxisIndexByFormat(format, axis);
  if (axis_index == -1) {
    return -1;
  }
  auto iter = valid_axis_map.find(axis);
  if (iter == valid_axis_map.end()) {
    GELOGW("The axis %s is invalid.", axis.c_str());
    return -1;
  }
  return axis_index - iter->second;
}

std::vector<std::string> AxisUtil::GetAxisVecByFormat(const ge::Format &format) {
  auto iter = kFormatAxisVec.find(static_cast<ge::Format>(GetPrimaryFormat(format)));
  if (iter == kFormatAxisVec.end()) {
    GELOGW("Does not support this format: %s", ge::TypeUtils::FormatToSerialString(format).c_str());
    return {};
  }
  return iter->second;
}

std::vector<std::string> AxisUtil::GetReshapeTypeAxisVec(const ge::Format &format, const int64_t &reshape_type_mask) {
  std::vector<std::string> format_axis_vec = GetAxisVecByFormat(static_cast<ge::Format>(GetPrimaryFormat(format)));
  if (format_axis_vec.empty()) {
    GELOGW("Does not support this format: %s", ge::TypeUtils::FormatToSerialString(format).c_str());
    return {};
  }
  std::vector<std::string> axis_vec;
  for (size_t i = 0; i < format_axis_vec.size(); ++i) {
    int64_t bit_value = (reshape_type_mask >> i) & 1;
    if (bit_value == 0) {
      axis_vec.emplace_back(format_axis_vec.at(i));
    }
  }
  return axis_vec;
}

std::map<std::string, int32_t> AxisUtil::GetReshapeTypeAxisMap(const ge::Format &format,
                                                               const int64_t &reshape_type_mask) {
  std::vector<std::string> format_axis_vec = GetAxisVecByFormat(static_cast<ge::Format>(GetPrimaryFormat(format)));
  if (format_axis_vec.empty()) {
    GELOGW("Does not support this format: %s", ge::TypeUtils::FormatToSerialString(format).c_str());
    return {};
  }
  std::map<std::string, int32_t> axis_map;
  int32_t expand_dims_cnt = 0;
  for (size_t i = 0; i < format_axis_vec.size(); ++i) {
    int64_t bit_value = (reshape_type_mask >> i) & 1;
    if (bit_value == 0) {
      axis_map[format_axis_vec.at(i)] = expand_dims_cnt;
    } else {
      ++expand_dims_cnt;
    }
  }
  return axis_map;
}

std::vector<std::string> AxisUtil::GetSplitOrConcatAxisByFormat(const ge::Format &format, const std::string &axis) {
  auto iter = kFormatSplitOrConcatAxisMap.find(static_cast<ge::Format>(GetPrimaryFormat(format)));
  if (iter == kFormatSplitOrConcatAxisMap.end()) {
    GELOGW("Does not support this format: %s.", ge::TypeUtils::FormatToSerialString(format).c_str());
    return {};
  }
  auto iter2 = iter->second.find(axis);
  if (iter2 == iter->second.end()) {
    GELOGW("There is no need to split or concatenate this axis: %s.", axis.c_str());
    return {};
  }
  return iter2->second;
}
} // namespace transformer

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
#include <algorithm>

namespace transformer {
bool RangeTransferAccordingToFormat::GetRangeAccordingToFormat(const ge::OpDescPtr &op_desc,
                                                               RangeAndFormat &range_and_format_info) {
  /* The default new range is old range */
  std::vector<int64_t> range_upper_old;
  std::vector<int64_t> range_low_old;
  for (auto &i : range_and_format_info.old_range) {
    range_low_old.emplace_back(i.first);
    range_upper_old.emplace_back(i.second);
  }

  ge::GeShape shape_low(range_low_old);
  ge::GeShape shape_upper(range_upper_old);
  transformer::ShapeAndFormat shape_and_format_info_low {shape_low, range_and_format_info.old_format,
      range_and_format_info.new_format, range_and_format_info.current_data_type};
  transformer::ShapeAndFormat shape_and_format_info_upper {shape_upper, range_and_format_info.old_format,
      range_and_format_info.new_format, range_and_format_info.current_data_type};
  ShapeTransferAccordingToFormat shape_transfer;
  bool res = (shape_transfer.GetShapeAccordingToFormat(op_desc, shape_and_format_info_low) &&
      shape_transfer.GetShapeAccordingToFormat(op_desc, shape_and_format_info_upper));
  if (!res || (shape_low.GetDimNum() != shape_upper.GetDimNum())) {
    return false;
  }
  range_and_format_info.new_range.clear();
  for (size_t i = 0; i < shape_low.GetDimNum(); ++i) {
    range_and_format_info.new_range.emplace_back(shape_low.GetDim(i), shape_upper.GetDim(i));
  }
  return res;
}

bool RangeTransferAccordingToFormat::GetRangeAccordingToFormat(const ExtAxisOpValue &op_value,
                                                               RangeAndFormat &range_and_format_info) {
  /* The default new range is old range */
  std::vector<int64_t> range_upper_old;
  std::vector<int64_t> range_low_old;
  for (auto &i : range_and_format_info.old_range) {
    range_low_old.emplace_back(i.first);
    range_upper_old.emplace_back(i.second);
  }

  ge::GeShape shape_low(range_low_old);
  ge::GeShape shape_upper(range_upper_old);
  transformer::ShapeAndFormat shape_and_format_info_low {shape_low, range_and_format_info.old_format,
      range_and_format_info.new_format, range_and_format_info.current_data_type};
  transformer::ShapeAndFormat shape_and_format_info_upper {shape_upper, range_and_format_info.old_format,
      range_and_format_info.new_format, range_and_format_info.current_data_type};
  ShapeTransferAccordingToFormat shape_transfer;
  bool res = (shape_transfer.GetShapeAccordingToFormat(op_value, shape_and_format_info_low) &&
              shape_transfer.GetShapeAccordingToFormat(op_value, shape_and_format_info_upper));
  if (!res || (shape_low.GetDimNum() != shape_upper.GetDimNum())) {
    return false;
  }
  range_and_format_info.new_range.clear();
  for (size_t i = 0; i < shape_low.GetDimNum(); ++i) {
    range_and_format_info.new_range.emplace_back(shape_low.GetDim(i), shape_upper.GetDim(i));
  }
  return res;
}

bool RangeTransferAccordingToFormat::GetRangeAccordingToFormat(RangeAndFormat &range_and_format_info) {
  /* The default new range is old range */
  std::vector<int64_t> range_upper_old;
  std::vector<int64_t> range_low_old;
  for (auto &i : range_and_format_info.old_range) {
    range_low_old.emplace_back(i.first);
    range_upper_old.emplace_back(i.second);
  }

  ge::GeShape shape_low(range_low_old);
  ge::GeShape shape_upper(range_upper_old);
  transformer::ShapeAndFormat shape_and_format_info_low {shape_low, range_and_format_info.old_format,
      range_and_format_info.new_format, range_and_format_info.current_data_type};
  transformer::ShapeAndFormat shape_and_format_info_upper {shape_upper, range_and_format_info.old_format,
      range_and_format_info.new_format, range_and_format_info.current_data_type};
  ShapeTransferAccordingToFormat shape_transfer;
  bool res = (shape_transfer.GetShapeAccordingToFormat(shape_and_format_info_low) &&
      shape_transfer.GetShapeAccordingToFormat(shape_and_format_info_upper));
  if (!res || (shape_low.GetDimNum() != shape_upper.GetDimNum())) {
    return false;
  }
  range_and_format_info.new_range.clear();
  for (size_t i = 0; i < shape_low.GetDimNum(); ++i) {
    range_and_format_info.new_range.emplace_back(shape_low.GetDim(i), shape_upper.GetDim(i));
  }
  return res;
}
};  // namespace fe

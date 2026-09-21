/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMON_UTILS_TRANSFORMER_INC_TRANSFER_RANGE_ACCORDING_TO_FORMAT_H_
#define COMMON_UTILS_TRANSFORMER_INC_TRANSFER_RANGE_ACCORDING_TO_FORMAT_H_

#include <vector>
#include "transfer_shape_according_to_format.h"

namespace transformer {
struct RangeAndFormatInfo {
  ge::GeShape old_shape;
  std::vector<std::pair<int64_t, int64_t>> old_range;
  std::vector<std::pair<int64_t, int64_t>> &new_range;
  ge::Format old_format;
  ge::Format new_format;
  ge::DataType current_data_type;
  CalcShapeExtraAttr extra_attr;
  RangeAndFormatInfo(ge::GeShape old_shape, std::vector<std::pair<int64_t, int64_t>> old_range,
                     std::vector<std::pair<int64_t, int64_t>> &new_range, ge::Format old_format,
                     ge::Format new_format, ge::DataType current_data_type) :
          old_shape(old_shape), old_range(old_range), new_range(new_range), old_format(old_format),
          new_format(new_format), current_data_type(current_data_type), extra_attr(CalcShapeExtraAttr()) {}
  RangeAndFormatInfo(ge::GeShape old_shape, std::vector<std::pair<int64_t, int64_t>> old_range,
                     std::vector<std::pair<int64_t, int64_t>> &new_range, ge::Format old_format,
                     ge::Format new_format, ge::DataType current_data_type, CalcShapeExtraAttr extra_attr) :
          old_shape(old_shape), old_range(old_range), new_range(new_range), old_format(old_format),
          new_format(new_format), current_data_type(current_data_type), extra_attr(extra_attr) {}
};

using RangeAndFormat = struct RangeAndFormatInfo;

class RangeTransferAccordingToFormat {
 public:
  RangeTransferAccordingToFormat() = default;

  ~RangeTransferAccordingToFormat() = default;

  RangeTransferAccordingToFormat(const RangeTransferAccordingToFormat &) = delete;

  RangeTransferAccordingToFormat &operator=(const RangeTransferAccordingToFormat &) = delete;

  static bool GetRangeAccordingToFormat(RangeAndFormat &range_and_format_info);

  // deprecated ATTRIBUTED_DEPRECATED(static bool GetRangeAccordingToFormat(const ExtAxisOpValue &, RangeAndFormat &))
  static bool GetRangeAccordingToFormat(const ge::OpDescPtr &op_desc, RangeAndFormat &range_and_format_info);

  static bool GetRangeAccordingToFormat(const ExtAxisOpValue &op_value, RangeAndFormat &range_and_format_info);
};
}  // namespace fe

#endif  // FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_RANGE_FORMAT_TRANSFER_TRANSFER_RANGE_ACCORDING_TO_FORMAT_H_

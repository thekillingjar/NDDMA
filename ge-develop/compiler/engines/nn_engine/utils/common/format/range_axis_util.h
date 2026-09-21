/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_UTILS_COMMON_FORMAT_RANGE_AXIS_UTIL_H_
#define FUSION_ENGINE_UTILS_COMMON_FORMAT_RANGE_AXIS_UTIL_H_

#include <functional>
#include <vector>
#include "common/fe_inner_error_codes.h"
#include "common/fe_utils.h"
#include "common/format/axis_util.h"
#include "common/math_util.h"
#include "common/util/op_info_util.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"

namespace fe {
/* Range axis value is arranged as {N,C,H,W,C1,C0,...} */
/* The first parameter is old shape's dimension,
 * second is c0 and third is axis value. */
using GetRangeAxisValueInfoByFormat =
    std::function<Status(const vector<std::pair<int64_t, int64_t>>&, const vector<int64_t>&, const uint32_t&,
                         vector<std::pair<int64_t, int64_t>>&)>;

using GetRangeAxisValueInfoByFormatPtr = std::shared_ptr<GetRangeAxisValueInfoByFormat>;

class RangeAxisUtil {
 public:
  static Status GetRangeAxisValueByOriginFormat(const vector<std::pair<int64_t, int64_t>>& original_range_vec,
                                                const ge::Format& format, const vector<int64_t>& dim_vec,
                                                const uint32_t& c0, vector<std::pair<int64_t, int64_t>>& range_value);

  static bool HasAxisValueFunc(const ge::Format& format);

 private:
  static Status GetRangeAxisValueByNCHW(const vector<std::pair<int64_t, int64_t>>& original_range_vec,
                                        const vector<int64_t>& original_dim_vec, const uint32_t& c0,
                                        vector<std::pair<int64_t, int64_t>>& range_value);

  static Status GetRangeAxisValueByNHWC(const vector<std::pair<int64_t, int64_t>>& original_range_vec,
                                        const vector<int64_t>& original_dim_vec, const uint32_t& c0,
                                        vector<std::pair<int64_t, int64_t>>& range_value);

  static Status GetRangeAxisValueByNC1HWC0(const vector<std::pair<int64_t, int64_t>>& original_range_vec,
                                           const vector<int64_t>& original_dim_vec, const uint32_t& c0,
                                           vector<std::pair<int64_t, int64_t>>& range_value);

  static Status GetRangeAxisValueByFz(const vector<std::pair<int64_t, int64_t>>& original_range_vec,
                                      const vector<int64_t>& original_dim_vec, const uint32_t& c0,
                                      vector<std::pair<int64_t, int64_t>>& range_value);

  static Status GetRangeAxisValueByHWCN(const vector<std::pair<int64_t, int64_t>>& original_range_vec,
                                        const vector<int64_t>& original_dim_vec, const uint32_t& c0,
                                        vector<std::pair<int64_t, int64_t>>& range_value);

  static Status GetRangeAxisValueByCHWN(const vector<std::pair<int64_t, int64_t>>& original_range_vec,
                                        const vector<int64_t>& original_dim_vec, const uint32_t& c0,
                                        vector<std::pair<int64_t, int64_t>>& range_value);

  static Status GetRangeAxisValueByND(const vector<std::pair<int64_t, int64_t>>& original_range_vec,
                                      const vector<int64_t>& original_dim_vec, const uint32_t& c0,
                                      vector<std::pair<int64_t, int64_t>>& range_value);

  static Status GetRangeAxisValueByNDHWC(const vector<std::pair<int64_t, int64_t>>& original_range_vec,
                                         const vector<int64_t>& original_dim_vec, const uint32_t& c0,
                                         vector<std::pair<int64_t, int64_t>>& range_value);

  static Status GetRangeAxisValueByNCDHW(const vector<std::pair<int64_t, int64_t>>& original_range_vec,
                                         const vector<int64_t>& original_dim_vec, const uint32_t& c0,
                                         vector<std::pair<int64_t, int64_t>>& range_value);

  static Status GetRangeAxisValueByDHWCN(const vector<std::pair<int64_t, int64_t>>& original_range_vec,
                                         const vector<int64_t>& original_dim_vec, const uint32_t& c0,
                                         vector<std::pair<int64_t, int64_t>>& range_value);

  static Status GetRangeAxisValueByDHWNC(const vector<std::pair<int64_t, int64_t>>& original_range_vec,
                                         const vector<int64_t>& original_dim_vec, const uint32_t& c0,
                                         vector<std::pair<int64_t, int64_t>>& range_value);

  /* map of GetAxisValueInfoByFormat, get axis value by different original
   * formats. */
  static const std::map<ge::Format, GetRangeAxisValueInfoByFormatPtr> get_range_axis_value_func_map;

  static Status CheckParamValue(const vector<std::pair<int64_t, int64_t>>& original_range_vec,
                                const vector<int64_t>& original_dim_vec, const uint32_t& c0,
                                vector<std::pair<int64_t, int64_t>>& range_value, const size_t& min_size);
};
}  // namespace fe

#endif  // FUSION_ENGINE_UTILS_COMMON_FORMAT_RANGE_AXIS_UTIL_H_

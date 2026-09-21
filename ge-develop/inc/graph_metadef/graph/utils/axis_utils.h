/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef METADEF_CXX_INC_GRAPH_UTILS_AXIS_UTILS_H_
#define METADEF_CXX_INC_GRAPH_UTILS_AXIS_UTILS_H_
#include <tuple>
#include "graph/symbolizer/symbolic.h"
#include "ascendc_ir/ascendc_ir_core/ascendc_ir.h"

namespace ge {
class AxisUtils {
 public:
  static View ReduceView(const View &src_view, int64_t reduce_axis);
  static View ReorderView(const View &src_view, const std::vector<int64_t> &my_api_sched_axes);

  static View SplitView(const View &src_view, const ge::Expression &split_size, const int64_t outter_id,
                        const int64_t inner_id, const int64_t original_id);
  static View MergeView(const View &src_view, const int64_t merged_axis_id,
                        const std::vector<int64_t> &original);
  static std::pair<bool, View> UpdateViewIfCrossLoop(const TransInfoRoadOfGraph &trans_info_road_of_graph,
                                                     const std::vector<int64_t> &input_api_sched_axes,
                                                     const std::vector<int64_t> &my_api_sched_axes,
                                                     View &&tensor_view_to_update);
  static std::vector<int64_t> GetDefaultVectorizedAxis(const std::vector<int64_t> &tensor_axis, int64_t loop_axis);
};
}  // namespace ge
#endif // METADEF_CXX_INC_GRAPH_UTILS_AXIS_UTILS_H_

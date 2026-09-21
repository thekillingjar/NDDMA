/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __INC_AUTOSCHEDULE_AXIS_GROUP_H__
#define __INC_AUTOSCHEDULE_AXIS_GROUP_H__

#include <vector>
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir_def.h"

namespace optimize::autoschedule {
struct AxisGroup {
  std::vector<ge::AxisId> x_group;  // 当为了和elemwise的ygroup分开切分时需要放入x_group,比如transpose前和transpose后
  std::vector<ge::AxisId> y_group;  // elemwise轴的分组
  std::vector<ge::AxisId> r_group;  // Reduce轴所在分组
  std::vector<ge::AxisId> n_group;  // 不可切分轴，或者只能作向量化轴
  std::vector<size_t> axes_order;
  bool operator==(const AxisGroup &other) const {
    return x_group == other.x_group && y_group == other.y_group && r_group == other.r_group &&
           n_group == other.n_group && axes_order == other.axes_order;
  }

  bool IsEmpty() const;

  std::string ToString() const;
};

extern "C" {
using optimize::autoschedule::AxisGroup;

int32_t GenAscGraphAxisGroup(const ge::AscGraph &graph, AxisGroup &axes_group);

bool CanMergeAxisGroup(const AxisGroup &lhs, const AxisGroup &rhs, AxisGroup &merged_group, const bool is_ge_call = false);
}
}  // namespace optimize::autoschedule

#endif

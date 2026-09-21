/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "scalar_to_1d_tensor.h"
#include "ascir_ops_utils.h"
#include "ascgraph_info_complete.h"
#include "graph_utils.h"
#include "node_utils.h"
#include "schedule_utils.h"

namespace optimize {
Status ScalarTo1DTensorPass::RunPass(ge::AscGraph &graph) {
  bool need_trans{false};
  for (const auto &node : graph.GetAllNodes()) {
    if (!ScheduleUtils::IsBuffer(node) && node->attr.sched.axis.empty()) {
      need_trans = true;
      break;
    }
  }
  if (!need_trans) {
    return ge::SUCCESS;
  }

  auto const_axis = graph.CreateAxis("axis_1d", ge::ops::One);
  for (const auto &node : graph.GetAllNodes()) {
    if (!ScheduleUtils::IsBuffer(node) && node->attr.sched.axis.empty()) {
      node->attr.sched.axis = {const_axis.id};
      for (auto output_attr : node->outputs()) {
        if (output_attr->attr.axis.empty()) {
          output_attr->attr.axis = {const_axis.id};
          output_attr->attr.repeats = {ge::ops::One};
          output_attr->attr.strides = {ge::ops::One};
        }
      }
    }
  }
  return ge::SUCCESS;
}
}  // namespace optimize

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "utils/autofuse_attrs.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "common/scope_tracing_recorder.h"
#include "backend/backend_spec.h"

namespace optimize {
std::unique_ptr<BackendSpec> BackendSpec::GetInstance() {
  auto instance = std::make_unique<BackendSpec>();
  instance->concat_max_input_num = 0;
  instance->concat_alg = 0;
  return instance;
}
}  // namespace optimize
namespace optimize::autoschedule {
  std::string optimize::autoschedule::AxisGroup::ToString() const {
    std::ostringstream oss;
    oss << "x_group: ";
    for (const auto &value : x_group) {
      oss << value << " ";
    }
    oss << "; y_group: ";
    for (const auto &value : y_group) {
      oss << value << " ";
    }
    oss << "; r_group: ";
      for (const auto &value : r_group) {
        oss << value << " ";
    }
    oss << "; n_group: ";
    for (const auto &value : n_group) {
      oss << value << " ";
    }
    return oss.str();
  }
  bool optimize::autoschedule::AxisGroup::IsEmpty() const {
    return x_group.empty() && y_group.empty() && r_group.empty();
  }
}  // namespace optimize::autoschedule
namespace ge {
  ScopeTracingRecorder::ScopeTracingRecorder([[maybe_unused]] const TracingModule stage,
                                             [[maybe_unused]] std::string msg) {}
  ScopeTracingRecorder::ScopeTracingRecorder([[maybe_unused]] const TracingModule stage,
                                             [[maybe_unused]] const std::vector<std::string> &msgs) {}
  ScopeTracingRecorder::~ScopeTracingRecorder() {}
}

extern "C" {
int32_t GenAscGraphAxisGroup(const ge::AscGraph &graph, optimize::autoschedule::AxisGroup &axes_group) {
  axes_group.x_group.push_back(0);
  axes_group.y_group.push_back(0);
  axes_group.r_group.push_back(0);
  axes_group.n_group.push_back(0);
  for (const auto &node : ge::AscGraphUtils::GetComputeGraph(graph)->GetAllNodes()) {
    auto attr = ge::GetOrCreateAutoFuseAttrs(node->GetOpDesc());
    GE_ASSERT_NOTNULL(attr);
    ge::GetInterAttrs(attr).axis_group = axes_group;
  }
  return 0;
}

bool CanMergeAxisGroup(const optimize::autoschedule::AxisGroup &lhs, const optimize::autoschedule::AxisGroup &rhs,
                       optimize::autoschedule::AxisGroup &merged_group) {
  merged_group = lhs;
  return true;
}

// 需要把atrace功能从自动融合目录中剥离
void TracingRecordDuration(const ge::TracingModule stage, const std::vector<std::string> &msgs,
                           const uint64_t start, const uint64_t duration) {}
void ReportTracingRecordDuration(const ge::TracingModule stage) {};
}


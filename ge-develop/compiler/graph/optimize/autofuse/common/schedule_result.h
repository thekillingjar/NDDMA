/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCGEN_DEV_BASE_COMMON_SCHEDULE_RESULT_H_
#define ASCGEN_DEV_BASE_COMMON_SCHEDULE_RESULT_H_

#include "ascendc_ir/ascendc_ir_core/ascendc_ir.h"

namespace ascir {
struct ScheduleGroup {
  std::vector<ge::AscGraph> impl_graphs;
  std::map<std::string, std::string> graph_name_to_score_funcs;
  bool double_buffer{false};
};

enum class CubeTemplateType : int32_t {
  kDefault = -1,   // no cube
  kFixpip,         // fixpip模板
  kCommon,         // 兜底模板
  kUBFuse,         // ub复用模板
  kL2Fuse,         // L2复用模板
};

struct ScheduledResult {
  std::vector<ScheduleGroup> schedule_groups;
  // dst -> src, <dst_groupid, <src_groupid, <dst_var_name, src_var>>>;
  std::map<size_t, std::map<size_t, std::map<std::string, ge::Expression>>> var_relations;
  ge::AscendString score_func;
  bool is_reduce_mem_reuse{false};
  bool enable_group_parallel{false};
  CubeTemplateType cube_type{CubeTemplateType::kDefault};
};

struct FusedScheduledResult {
  ge::AscendString fused_graph_name;
  std::vector<ge::AscNodePtr> input_nodes;
  std::vector<ge::AscNodePtr> output_nodes;
  std::vector<ge::AscNodePtr> workspace_nodes;
  std::vector<ge::Expression> origin_vars;
  std::vector<std::vector<ScheduledResult>> node_idx_to_scheduled_results;
};
}  // namespace ascir

#endif  // ASCGEN_DEV_BASE_COMMON_SCHEDULE_RESULT_H_

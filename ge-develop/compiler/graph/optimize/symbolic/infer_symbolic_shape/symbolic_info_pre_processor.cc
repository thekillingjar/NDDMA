/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "symbolic_info_pre_processor.h"

#include "graph/attribute_group/attr_group_symbolic_desc.h"
#include "graph/attribute_group/attr_group_shape_env.h"
#include "graph/utils/graph_utils.h"
#include "graph/optimize/symbolic/codegen/guard_codegen.h"
#include "graph/passes/standard_optimize/constant_folding/dimension_compute_pass.h"
#include "graph/passes/control_flow_and_stream/switch_dead_branch_elimination.h"
#include "graph/passes/control_flow_and_stream/switch_logic_remove_pass.h"
#include "graph/passes/control_flow_and_stream/cond_remove_pass.h"
#include "graph/passes/symbolic/symbolic_cond_remove_pass.h"
#include "graph/unfold/graph_unfolder.h"
#include "common/compile_profiling/ge_trace_wrapper.h"
#include "api/aclgrph/option_utils.h"
#include <graph/optimize/symbolic/shape_env_guarder.h>

namespace ge {
Status SymbolicInfoPreProcessor::Run(const ComputeGraphPtr &graph, const std::vector<GeTensor> &graph_inputs) {
  /*
   * 1. DimensionComputePass: 将size，shape等算子的输出转化为常量
   * 2. CondRemovePass: 消除if算子
   * 3. SwitchDeadBranchElimination, SwitchLogicRemovePass: 消除case算子
   * 5. SymbolicCondRemovePass: 如果if/case的条件输入是data, 通过加guard来消除if/case算子
   * 6. GraphUnfold: 将partitioncall子图在根图上展开
   */

  if (EnableSliceSchedule()) {
    GELOGI("SliceSchedule is enable, add SymbolicCondRemovePass");
    auto shape_env = graph->GetAttrsGroup<ShapeEnvAttr>();
    GE_ASSERT_NOTNULL(shape_env);
    ShapeEnvGuarder guarder(shape_env);

    DimensionComputePass dimension_compute_pass;
    CondRemovePass condition_remove_pass;
    SwitchDeadBranchElimination switch_dead_branch_elimination;
    SwitchLogicRemovePass switch_logic_remove_pass;
    SymbolicCondRemovePass symbolic_cond_remove_pass(graph_inputs);

    NamesToPass names_to_passes;
    names_to_passes.emplace_back("DimensionComputePass", &dimension_compute_pass);
    names_to_passes.emplace_back("CondRemovePass", &condition_remove_pass);
    names_to_passes.emplace_back("SwitchDeadBranchElimination", &switch_dead_branch_elimination);
    names_to_passes.emplace_back("SwitchLogicRemovePass", &switch_logic_remove_pass);
    names_to_passes.emplace_back("SymbolicCondRemovePass", &symbolic_cond_remove_pass);

    GE_TRACE_START(names_to_passes);
    const auto ret = GEPass(graph).Run(names_to_passes);
    GE_COMPILE_TRACE_TIMESTAMP_END(
        names_to_passes, "SymbolicShapeInference::PreProcess::" + graph->GetName());
    GE_ASSERT_SUCCESS(ret, "[Run][GEPasses] SymbolicShapeInference PreProcess failed, ret:%d.", ret);
  }

  GE_ASSERT_SUCCESS(gert::GraphUnfolder::UnfoldAllPartitioncallInPlace(graph));
  GE_DUMP(graph, "Autofuser_AfterSymbolicInferPreProcess");
  return SUCCESS;
}
} // namespace ge
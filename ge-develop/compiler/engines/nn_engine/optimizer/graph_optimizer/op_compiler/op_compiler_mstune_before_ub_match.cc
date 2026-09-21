/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/op_compiler/op_compiler_mstune_before_ub_match.h"

namespace fe {
OpCompilerMstuneBeforeUbMatch::OpCompilerMstuneBeforeUbMatch(const std::string& compiler_name,
                                                             const std::string& engine_name,
                                                             const LxFusionOptimizerPtr &lx_fusion_optimizer)
    : OpCompiler(compiler_name, engine_name, false, lx_fusion_optimizer) {}

OpCompilerMstuneBeforeUbMatch::~OpCompilerMstuneBeforeUbMatch() {}

Status OpCompilerMstuneBeforeUbMatch::RunCompileProcess(ge::ComputeGraph& graph) {
  /* Some nodes needs to be re-pre-compiled after Ub fusion matching.
   * Because there format or data type. */
  bool need_re_precompile_graph = false;
  (void)ge::AttrUtils::GetBool(graph, NEED_RE_PRECOMPILE, need_re_precompile_graph);
  if (need_re_precompile_graph) {
    Status ret = PreCompileOp(graph);
    if (ret != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][OpCmplMstune] PreCompileOp failed after buffer fusion.");
      return ret;
    }
  }
  std::vector<ge::NodePtr> buff_fus_compile_failed_nodes;
  CompileInfoParam compile_info(buff_fus_compile_failed_nodes);
  vector<ge::NodePtr> nodes_be_compiled;
  vector<ge::NodePtr> buff_fus_rollback_nodes;

  Status ret = GetFusionScope(graph, buff_fus_rollback_nodes, compile_info.fusion_nodes_map, nodes_be_compiled);
  if (ret != SUCCESS) {
    return ret;
  }

  /* compile and roll-back(fusion op if fused compiling failed.) all tbe op first to make sure
   * the sub-graph return to tuning tools can be successfully compiled. */
  ret = CompileOpOnly(compile_info);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][OpCmplMstune] Failed to compile graph %s in step before ub matching.",
                    graph.GetName().c_str());
    return ret;
  }

  FE_LOGI("Stop to do optimize fused graph in step before ub matching of tuning process.");
  return SUCCESS;
}
}

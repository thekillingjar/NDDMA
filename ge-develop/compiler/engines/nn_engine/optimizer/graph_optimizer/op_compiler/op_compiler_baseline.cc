/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/op_compiler/op_compiler_baseline.h"
#include "common/graph/fe_graph_utils.h"
#include <unordered_set>
#include "common/configuration.h"

namespace fe {
OpCompilerBaseline::OpCompilerBaseline(const std::string &compiler_name, const std::string &engine_name,
                                       const LxFusionOptimizerPtr &lx_fusion_optimizer)
    : OpCompiler(compiler_name, engine_name, lx_fusion_optimizer) {}

OpCompilerBaseline::~OpCompilerBaseline() {}

Status OpCompilerBaseline::GetFusionScope(ge::ComputeGraph &graph, ScopeNodeIdMap &fusion_nodes_map,
                                          std::vector<ge::NodePtr> &nodes_be_compiled,
                                          std::vector<ge::NodePtr> &all_nodes_after_lx_fusion) {
  std::string graph_name = graph.GetName();
  auto all_nodes = graph.GetDirectNode();
  /* Find the minimum scope id in this graph. */
  for (auto &node : all_nodes) {
    (void)ge::AttrUtils::SetStr(node->GetOpDesc(), "_graph_name", graph_name);
  }

  // if auto tune is on, all nodes should be re-compiled
  for (auto &node : all_nodes) {
    nodes_be_compiled.emplace_back(node);
    all_nodes_after_lx_fusion.emplace_back(node);
  }

  if (GetScopeNodeMap(graph, nodes_be_compiled, fusion_nodes_map) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][GetFusScope] GetScopeNodeMap failed, graph [%s].", graph.GetName().c_str());
    return FAILED;
  }

  return SUCCESS;
}

Status OpCompilerBaseline::PreCompileProcess(ge::ComputeGraph& graph, bool &sgt_flag) {
  /* Some nodes needs to be re-pre-compiled after Ub fusion matching.
   * Because their format or data type maybe changed. */
  bool need_re_precompile_graph = false;
  (void)ge::AttrUtils::GetBool(graph, NEED_RE_PRECOMPILE, need_re_precompile_graph);
  FE_LOGI("Graph %s re-precompile flag is %u.", graph.GetName().c_str(), need_re_precompile_graph);
  if (need_re_precompile_graph) {
    Status res = PreCompileOp(graph);
    if (res != SUCCESS) {
      return res;
    }
  }

  FE_LOGD("PreCompileProcess thread op on graph %s. FFTS SGT flag is %u.", graph.GetName().c_str(), sgt_flag);
  Status ret = PreCompileThreadOp(graph, sgt_flag);
  if (ret != SUCCESS) {
    return ret;
  }
  FE_LOGI("Finish PreCompileProcess graph %s. Ffts sgt flag is %u.", graph.GetName().c_str(), sgt_flag);
  return SUCCESS;
}

Status OpCompilerBaseline::RunCompileProcess(ge::ComputeGraph &graph) {
  FE_LOGD("Running compile process graph %s for baseline.", graph.GetName().c_str());
  bool sgt_flag = false;
  Status ret = PreCompileProcess(graph, sgt_flag);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][Baseline] PreCompile failed after buffer fusion for graph %s.",
                    graph.GetName().c_str());
    return ret;
  }
  std::vector<ge::NodePtr> buff_fus_compile_failed_nodes;
  CompileInfoParam compile_info(buff_fus_compile_failed_nodes);
  compile_info.compile_strategy = CompileStrategy::COMPILE_STRATEGY_NO_TUNE;
  vector<ge::NodePtr> nodes_be_compiled;
  vector<ge::NodePtr> buff_fus_rollback_nodes;

  ret = OpCompiler::GetFusionScope(graph, buff_fus_rollback_nodes, compile_info.fusion_nodes_map, nodes_be_compiled);
  if (ret != SUCCESS) {
    return SUCCESS;
  }

  FE_LOGD("Compile op to ensure the entire graph is successfully compiled.");
  FE_LOGD("Graph %s: nodes_be_compiled total size is [%zu]", graph.GetName().c_str(), nodes_be_compiled.size());
  ret = CompileOpOnly(compile_info);
  if (ret != SUCCESS) {
    return ret;
  }

  LxFusionOptimizeResult buffer_ret = LxFusionOptimizeResult::NO_FUSION_STRATEGY;

  FE_CHECK_NOTNULL(lx_fusion_optimizer_ptr_);
  bool need_re_compile = false;
  ret = lx_fusion_optimizer_ptr_->LxFusionOptimize(graph, buffer_ret, need_re_compile);
  /* Call Lxfusion to optimize the graph. */
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][Normal] BufferFusionProcess failed for graph %s result %u.",
                    graph.GetName().c_str(), ret);
    return FAILED;
  }
  if (GenerateFormatTuneResult(graph, buffer_ret, need_re_compile) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][FormatTune] Failed to update tuneFormat by cannkb result for graph %s.",
                    graph.GetName().c_str());
    return FAILED;
  }
  FE_LOGD("Finish optimizing the graph by lxfusion, graph %s.", graph.GetName().c_str());

  // pre compile op which l1fusion or l2fusion changed
  compile_info.compile_strategy = CompileStrategy::COMPILE_STRATEGY_OP_SPEC;
  if (buffer_ret != LxFusionOptimizeResult::NO_FUSION_STRATEGY) {
    FE_LOGI("Lx-fusion changed the graph, and we need to re-precompile the graph.");
    ret = ReCompileOpAfterLxFusion(graph, compile_info, buffer_ret);
    if (ret != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][RunCmplProc] Failed to re-compile op after lx fusion for graph [%s]",
                      graph.GetName().c_str());
      return FAILED;
    }
  } else {
    ret = CompileOpOnly(compile_info);
    if (ret != SUCCESS) {
      return ret;
    }
    /* using the first compiling result only because no new nodes are created. */
    FE_LOGI("Lx-Fusion does not alter the graph; we simply use the old compile result.");
    ret = ParseJsonAndCompressOp(graph, compile_info.compile_ret_map, nodes_be_compiled);
    if (ret != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][Baseline] Failed to parse json or compress op for graph[%s].",
                      graph.GetName().c_str());
      return ret;
    }
  }
  ret = PostCompileOp(graph, buff_fus_compile_failed_nodes);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][Baseline] Failed to post compile for graph[%s].",
                    graph.GetName().c_str());
    return ret;
  }
  FE_LOGI("Successfully compiled op in Baseline mode for graph %s.", graph.GetName().c_str());
  return SUCCESS;
}
}  // namespace fe

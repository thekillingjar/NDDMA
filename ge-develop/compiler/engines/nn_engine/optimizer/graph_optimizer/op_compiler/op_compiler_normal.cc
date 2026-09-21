/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/op_compiler/op_compiler_normal.h"
#include <unordered_set>
#include "common/fe_utils.h"

using namespace std;
namespace fe {
OpCompilerNormal::OpCompilerNormal(const std::string& compiler_name, const std::string& engine_name,
                                   const LxFusionOptimizerPtr &lx_fusion_optimizer)
    :OpCompiler(compiler_name, engine_name, lx_fusion_optimizer) {}

OpCompilerNormal::~OpCompilerNormal() {}

bool OpCompilerNormal::HasCompileStrategy(const vector<ge::NodePtr> &nodes_be_compiled) const {
  if (!nodes_be_compiled.empty()) {
    for (const ge::NodePtr &node_ptr : nodes_be_compiled) {
      std::string op_compile_strategy;
      if (ge::AttrUtils::GetStr(node_ptr->GetOpDesc(), ge::ATTR_NAME_OP_COMPILE_STRATEGY, op_compile_strategy) &&
          !op_compile_strategy.empty()) {
        FE_LOGI("Node[%s, %s] has compile strategy[%s] and this graph needs to be recompile.",
                node_ptr->GetName().c_str(), node_ptr->GetType().c_str(), op_compile_strategy.c_str());
        return true;
      }
    }
  }
  return false;
}

Status OpCompilerNormal::PreCompileProcess(ge::ComputeGraph& graph, bool &sgt_flag) {
  /* Some nodes needs to be re-pre-compiled after Ub fusion matching.
   * Because there format or data type. */
  Status ret;
  bool need_re_precompile_graph = false;
  (void)ge::AttrUtils::GetBool(graph, NEED_RE_PRECOMPILE, need_re_precompile_graph);
  FE_LOGI("Graph %s re-precompile flag is %u.", graph.GetName().c_str(), need_re_precompile_graph);
  if (need_re_precompile_graph) {
    ret = PreCompileOp(graph);
    if (ret != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][Normal] PreCompileOp failed after buffer fusion for graph [%s].",
                      graph.GetName().c_str());
      return ret;
    }
  }

  ret = PreCompileThreadOp(graph, sgt_flag);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][Normal] PreCompileThreadOp failed after buffer fusion for graph [%s].",
                    graph.GetName().c_str());
    return ret;
  }
  FE_LOGI("Finish PreCompileProcess graph %s. FFTS SGT flag is %u.", graph.GetName().c_str(), sgt_flag);
  return SUCCESS;
}

Status OpCompilerNormal::ReCompileWithNoFusionStrategy(const ge::ComputeGraph& graph,
                                                       const vector<ge::NodePtr> &nodes_be_compiled,
                                                       CompileInfoParam &compile_info) {
  // if nodes has compile strategy, try to compile
  Status ret;
  if (HasCompileStrategy(nodes_be_compiled)) {
    ret = CompileOpOnly(compile_info);
    if (ret != SUCCESS) {
      return ret;
    }
  }
  /* using the first compiling result only because no new nodes are created. */
  FE_LOGI("Lx-Fusion does not change the graph. We just use old compile result.");
  ret = ParseJsonAndCompressOp(graph, compile_info.compile_ret_map, nodes_be_compiled);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][Normal] Failed to parse json or compress op for graph[%s].",
                    graph.GetName().c_str());
    return ret;
  }
  return SUCCESS;
}

Status OpCompilerNormal::RunCompileProcess(ge::ComputeGraph& graph) {
  FE_LOGD("Run compile process graph %s for normal.", graph.GetName().c_str());
  bool sgt_flag = false;
  Status ret = PreCompileProcess(graph, sgt_flag);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][Compile][Normal] PreCompile failed after buffer fusion for graph %s.",
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

  /* Compile first to get a completely same graph as tuinng step which will lead us find
   * a optimization strategy. */
  FE_LOGD("Compile the op to ensure the entire graph compiles successfully.");
  FE_LOGD("Nodes_be_compiled total size is [%zu].", nodes_be_compiled.size());
  ret = CompileOpOnly(compile_info);
  if (ret != SUCCESS) {
    return ret;
  }

  LxFusionOptimizeResult buffer_ret = LxFusionOptimizeResult::NO_FUSION_STRATEGY;
  bool need_lx_opt = true;
  (void)ge::AttrUtils::GetBool(graph, kNeedSetSliceInfo, need_lx_opt);
  if (!sgt_flag && need_lx_opt) {
    FE_CHECK_NOTNULL(lx_fusion_optimizer_ptr_);
    /* Call Lxfusion to optimize the graph. */
    FE_TIMECOST_START(bufferFusionFunc);
    bool need_re_compile = false;
    ret = lx_fusion_optimizer_ptr_->LxFusionOptimize(graph, buffer_ret, need_re_compile);
    if (ret != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][Normal] BufferFusionProcess failed for graph %s result %u.",
                      graph.GetName().c_str(), ret);
      return FAILED;
    }
    
    if (GenerateFormatTuneResult(graph, buffer_ret, need_re_compile) != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][FormatTune] Failed to update tuneFormat by cannkb result for graph %s",
                      graph.GetName().c_str());
      return FAILED;
    }
    FE_TIMECOST_END(bufferFusionFunc, "bufferFusionFunc.lxfusion");
  }
  FE_LOGD("Finished optimizing the graph [%s] with lxfusion. LxFusion optimization result: %d.",
          graph.GetName().c_str(), static_cast<int32_t>(buffer_ret));
  // pre compile op which l1fusion or l2fusion changed
  compile_info.compile_strategy = CompileStrategy::COMPILE_STRATEGY_OP_SPEC;
  if (buffer_ret != LxFusionOptimizeResult::NO_FUSION_STRATEGY) {
    FE_LOGI("Lx-fusion change the graph and we need re-precompile graph.");
    ret = ReCompileOpAfterLxFusion(graph, compile_info, buffer_ret);
    if (ret != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][RunCmplProc] Failed to re-compile op after lx fusion for graph [%s]",
                      graph.GetName().c_str());
      return FAILED;
    }
  } else {
    ret = ReCompileWithNoFusionStrategy(graph, nodes_be_compiled, compile_info);
    if (ret != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][Compile][RunCmplProc] Failed to re-compile op with no lx fusion for graph [%s]",
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
  FE_LOGI("Successfully compiled op in normal mode: graph[%s].", graph.GetName().c_str());
  return SUCCESS;
}
}

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_PNE_DATA_FLOW_GRAPH_PROCESS_POINT_LOADER_H
#define AIR_COMPILER_PNE_DATA_FLOW_GRAPH_PROCESS_POINT_LOADER_H

#include "framework/common/debug/log.h"
#include "dflow/compiler/data_flow_graph/compile_config_json.h"
#include "dflow/compiler/data_flow_graph/data_flow_graph.h"
#include "dflow/compiler/data_flow_graph/function_compile.h"
#include "proto/dflow.pb.h"
#include "dflow/compiler/model/flow_model_cache.h"

namespace ge {
class ProcessPointLoader {
 public:
  static Status LoadProcessPoint(const dataflow::ProcessPoint &process_point, DataFlowGraph &data_flow_graph,
                                 const NodePtr &node);
  static Status RemoveGraphFromParent(const ComputeGraphPtr &root_graph, const ComputeGraphPtr &sub_graph);
 private:
  static Status LoadFunctionProcessPoint(const dataflow::ProcessPoint &process_point, DataFlowGraph &data_flow_graph,
                                         const NodePtr &node);
  static Status LoadUserFunctionProcessPoint(const dataflow::ProcessPoint &process_point,
                                             DataFlowGraph &data_flow_graph, const NodePtr &node);
  static Status CompileUserFunctionProcessPoint(const CompileConfigJson::FunctionPpConfig &func_pp_cfg,
                                                const std::string &pp_name,const ComputeGraphPtr &compute_graph,
                                                const NodePtr &node, DataFlowGraph &data_flow_graph);
  static Status SetUserFunctionProcessPointAttrs(const CompileConfigJson::FunctionPpConfig &func_pp_cfg,
                                                 const ComputeGraphPtr &compute_graph, const NodePtr &node,
                                                 const OpDescPtr &flow_func_desc);
  static Status LoadBuiltInFunctionProcessPoint(const dataflow::ProcessPoint &process_point,
                                                DataFlowGraph &data_flow_graph, const NodePtr &node);
  static Status LoadGraphProcessPoint(const dataflow::ProcessPoint &process_point, DataFlowGraph &data_flow_graph,
                                      const NodePtr &node);
  static Status SetFlowAttrToGraph(const NodePtr &node, const ComputeGraphPtr &graph);
  static Status AddSubGraphsToProcessPointGraph(const ComputeGraphPtr &root_graph, const ComputeGraphPtr &pp_graph);
  static Status CreateFlowFuncOpDescFromProcessPoint(const dataflow::ProcessPoint &process_point,
                                                     const CompileConfigJson::FunctionPpConfig &func_pp_cfg,
                                                     OpDescPtr &op_desc);
  static Status SetAttrBinPathForFlowFunc(const CompileConfigJson::FunctionPpConfig &func_pp_cfg,
                                          OpDescPtr &flow_func_desc);
  static Status SetAttrFuncsForFlowFunc(const CompileConfigJson::FunctionPpConfig &func_pp_cfg,
                                        OpDescPtr &flow_func_desc);
  static Status SetCpuNumForFlowFunc(const CompileConfigJson::FunctionPpConfig &func_pp_cfg,
                                     OpDescPtr &flow_func_desc);
  static Status PreProcessSubgraphAttrs(const ComputeGraphPtr &subgraph);
  static Status SetAttrFuncsForFlowFunc(const dataflow::ProcessPoint &process_point, OpDescPtr &flow_func_desc);
  static Status SetCustomizedAttrsForFlowFunc(const dataflow::ProcessPoint &process_point, OpDescPtr &flow_func_desc);
  static Status AddInputsForFlowFunc(NodePtr &flow_func_node, ComputeGraphPtr &compute_graph);
  static Status AddOutputsForFlowFunc(NodePtr &flow_func_node, ComputeGraphPtr &compute_graph);
  static Status LoadInvokedProcessPoint(const dataflow::ProcessPoint &process_point,
                                        DataFlowGraph &data_flow_graph, OpDescPtr &flow_func_desc,
                                        const NodePtr &node, bool is_built_in_flow_func);
  static Status UpdateGraphInputsDesc(const CompileConfigJson::GraphPpConfig &graph_pp_cfg,
                                      ComputeGraphPtr &compute_graph);
  static void AddPrefixForGraphNodeName(ComputeGraphPtr &graph, const std::string &prefix_name);
  static Status CompileFunctionProcessPoint(const std::string &pp_name,
                                            const CompileConfigJson::FunctionPpConfig &func_pp_cfg,
                                            const NodePtr &node);
  static Status SetCompileResultToNode(const std::string &pp_name, const FunctionCompile::CompileResult &result,
                                       const NodePtr &node);
  static Status CheckFunctionPpConfigIsValid(const CompileConfigJson::FunctionPpConfig &func_pp_cfg,
                                             const std::string &pp_name);
  static Status SetNMappingAttrIfHasStreamInput(const CompileConfigJson::FunctionPpConfig &func_pp_cfg,
                                                const ComputeGraphPtr &graph);
  static Status CreateFunctionProcessPointComputeGraph(const DataFlowGraph &data_flow_graph,
                                                       const std::string &graph_name, uint32_t input_num,
                                                       uint32_t output_num, ComputeGraphPtr &compute_graph);
  static Status AddFlowFuncToComputeGraph(const OpDescPtr &flow_func_desc, ComputeGraphPtr &compute_graph);

  static Status AddGraphInfoForCache(const DataFlowGraph &data_flow_graph, const ComputeGraphPtr &graph,
                                     const std::map<std::string, std::string> &build_options);

  static Status SetEschedPriority(const ComputeGraphPtr &graph, const OpDescPtr &op_desc,
                                  const std::string &priority_name);
  static Status SetEschedPriorities(const ComputeGraphPtr &graph, const OpDescPtr &op_desc);

  static Status FixGraphOutputNode(const ComputeGraphPtr &graph);

  static void LoadCompileResultFromCache(const CacheCompileResult &cache_compile_result,
                                         FunctionCompile::CompileResult &result);

  static Status TransferInputShapeToRange(const ComputeGraphPtr &graph,
                                          std::map<std::string, std::string> &build_options);

  static Status CheckBufConfigIsValid(const std::vector<CompileConfigJson::BufCfg> &user_buf_cfg);

  static Status SetDataFlowScope(const OpDescPtr &flow_func_desc, const std::string &data_flow_scope);

  static Status SetDataFlowInvokedScopes(const OpDescPtr &flow_func_desc,
                                         const std::vector<std::string> &invoked_scopes);
};
}  // namespace ge
#endif

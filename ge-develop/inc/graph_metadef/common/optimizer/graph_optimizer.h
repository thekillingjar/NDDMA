/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_COMMON_OPTIMIZER_GRAPH_OPTIMIZER_H_
#define INC_COMMON_OPTIMIZER_GRAPH_OPTIMIZER_H_

#include <map>
#include <string>
#include "graph_optimizer_types.h"
#include "optimize_utility.h"
#include "common/ge_common/ge_inner_error_codes.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "graph/compute_graph.h"
#include "graph/op_kernel_bin.h"

/*lint -e148*/
namespace ge {
class GraphOptimizer {
 public:
  virtual ~GraphOptimizer() {}

  // initialize graphOptimizer
  virtual Status Initialize(const std::map<std::string, std::string> &options,
                            OptimizeUtility *const optimize_utility) = 0;

  // close graphOptimizer
  virtual Status Finalize() = 0;

  virtual Status FinalizeSessionInfo(ComputeGraph& graph) {
    (void)graph;
    return SUCCESS;
  }

  // init process for optimize graph every time because options may different in different build process
  // 当前引擎获取编译option是在OptimizeGraphPrepare接口中获取，该接口默认会过滤vector engine。
  // 当前出现问题场景是子图优化阶段因为算子融合直接选择了vector engine的场景，出现了vector engine获取不到编译option导致问题。
  // 当前决策新增OptimizeGraphInit接口，该接口不会过滤引擎，全部调用.这样获取到build option操作就从OptimizeGraphPrepare剥离。
  virtual Status OptimizeGraphInit(ComputeGraph& graph) {
    (void)graph;
    return SUCCESS;
  }

  // optimize original graph for FE quant optimize
  virtual Status OptimizeGraphPrepare(ComputeGraph& graph) {
    (void)graph;
    return SUCCESS;
  }

  // optimize graph after normalization, include multi dims and pre/post process
  virtual Status OptimizeAfterGraphNormalization(const ComputeGraphPtr& graph) {
    (void)graph;
    return SUCCESS;
  }

  // optimize graph before build for RTS
  virtual Status OptimizeGraphBeforeBuild(ComputeGraph& graph) {
    (void)graph;
    return SUCCESS;
  }

  // optimize original graph, using in graph preparation stage
  virtual Status OptimizeOriginalGraph(ComputeGraph &graph) = 0;

  // optimize original graph, using for conversion operator insert in graph preparation stage
  virtual Status OptimizeOriginalGraphJudgeInsert(ComputeGraph &graph) {
    (void)graph;
    return SUCCESS;
  }

  // optimize fused graph
  virtual Status OptimizeFusedGraph(ComputeGraph &graph) = 0;

  // optimize whole graph, using after graph merged stage
  virtual Status OptimizeWholeGraph(ComputeGraph &graph) = 0;

  // get attribute of graph optimizer
  virtual Status GetAttributes(GraphOptimizerAttribute &attrs) const = 0;

  // optimize streamed Graph
  virtual Status OptimizeStreamGraph(ComputeGraph &graph, const RunContext &context) {
    (void)graph;
    (void)context;
    return SUCCESS;
  }

  // optimize streamed whole Graph
  virtual Status OptimizeStreamedWholeGraph(ComputeGraph &graph) {
    (void)graph;
    return SUCCESS;
  }

  // op compile
  virtual Status OptimizeFusedGraphAfterGraphSlice(ComputeGraph &graph) {
    (void)graph;
    return SUCCESS;
  }

  // optimize whole graph, using after stage1
  virtual Status OptimizeAfterStage1(ComputeGraph &graph) {
    (void)graph;
    return SUCCESS;
  }

  // recover compile result of precompiled op
  using KernelLookup = std::function<OpKernelBinPtr(const std::string &kernel_name)>;
  virtual Status OptimizeSubgraphOfPrecompiledOp(ComputeGraph &graph, const KernelLookup &lookup) {
    static_cast<void>(graph);
    static_cast<void>(lookup);
    return SUCCESS;
  }

  // 为避免子图优化中多线程操作导致的数据读写冲突，提供子图优化前后的单线程接口，由引擎实现以实现改图功能
  virtual Status OptimizeSubgraphPreProc(ComputeGraph &graph) {
    (void)graph;
    return SUCCESS;
  }
  virtual Status OptimizeSubgraphPostProc(ComputeGraph &graph) {
    (void)graph;
    return SUCCESS;
  }
  // 格式选择接口，从OptimizeOriginalGraphJudgeInsert中独立出来格式选择能力，OptimizeOriginalGraphJudgeInsert接口实现精度选择能力
  virtual Status OptimizeOriginalGraphJudgeFormatInsert(ComputeGraph &graph) {
    (void)graph;
    return SUCCESS;
  }
};
}  // namespace ge
/*lint +e148*/
#endif  // INC_COMMON_OPTIMIZER_GRAPH_OPTIMIZER_H_

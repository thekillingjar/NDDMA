/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_COMPILER_OP_COMPILER_BASELINE_H_
#define FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_COMPILER_OP_COMPILER_BASELINE_H_

#include "graph_optimizer/op_compiler/op_compiler.h"

namespace fe {
class OpCompilerBaseline : public OpCompiler {
 public:
  using ConstComputeGraph = const ge::ComputeGraph;
  template <class T>
  using Vistor = RangeVistor<T, std::shared_ptr<ConstComputeGraph>>;

  OpCompilerBaseline(const std::string& compiler_name, const std::string &engine_name,
                     const LxFusionOptimizerPtr &lx_fusion_optimizer);

  ~OpCompilerBaseline() override;

  /**
   * @ingroup fe
   * @brief prohibit copy and assign construct
   */
  OpCompilerBaseline(const OpCompilerBaseline &) = delete;

  OpCompilerBaseline &operator=(const OpCompilerBaseline &) = delete;

  Status PreCompileProcess(ge::ComputeGraph& graph, bool &sgt_flag);

  Status RunCompileProcess(ge::ComputeGraph& graph) override;

  using OpCompiler::GetFusionScope;
  Status GetFusionScope(ge::ComputeGraph &graph, ScopeNodeIdMap &fusion_nodes_map,
                        std::vector<ge::NodePtr> &nodes_be_compiled,
                        std::vector<ge::NodePtr> &all_nodes_after_lx_fusion) override;
};
}

#endif //FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_COMPILER_OP_COMPILER_BASELINE_H_

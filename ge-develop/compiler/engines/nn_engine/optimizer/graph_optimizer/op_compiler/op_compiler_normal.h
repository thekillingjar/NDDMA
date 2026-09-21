/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_COMPILER_OP_COMPILER_NORMAL_MODE_H_
#define FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_COMPILER_OP_COMPILER_NORMAL_MODE_H_

#include "graph_optimizer/op_compiler/op_compiler.h"
#include "common/configuration.h"

namespace fe {
class OpCompilerNormal : public OpCompiler {
 public:
  using ConstComputeGraph = const ge::ComputeGraph;
  template <class T>
  using Vistor = RangeVistor<T, std::shared_ptr<ConstComputeGraph>>;

  OpCompilerNormal(const std::string& compiler_name, const std::string &engine_name,
                   const LxFusionOptimizerPtr &lx_fusion_optimizer);

  ~OpCompilerNormal() override;

  /**
   * @ingroup fe
   * @brief prohibit copy and assign construct
   */
  OpCompilerNormal(const OpCompilerNormal &) = delete;

  OpCompilerNormal &operator=(const OpCompilerNormal &) = delete;

  /* There are two scenarios which are different from baseline:
   * 1. Some of the nodes in a fusion scope are set attribute "need_re_precompile":
   *     In this case, we need to find all nodes with that scope id compile them together.
   *     That means if one node of a fusion scope need to be re-pre-compiled, the whole scope
   *     needs to be RE-COMPILED.
   *
   * 2. some fusion nodes with negative scope id are duplicated by lx-fusion:
   *     In this case, we need to give them different new scope id. Because they
   *     are expect to be compiled alone.
   *     And the new (negative) scope id should be different from the other negative ones*/
  Status RunCompileProcess(ge::ComputeGraph& graph) override;

 private:
  bool HasCompileStrategy(const vector<ge::NodePtr> &nodes_be_compiled) const;
  Status PreCompileProcess(ge::ComputeGraph& graph, bool &sgt_flag);
  Status ReCompileWithNoFusionStrategy(const ge::ComputeGraph& graph, const vector<ge::NodePtr> &nodes_be_compiled,
                                       CompileInfoParam &compile_info);
};
}

#endif //FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_COMPILER_OP_COMPILER_NORMAL_MODE_H_

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_COMPILER_OP_COMPILER_OPTUNE_H_
#define FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_COMPILER_OP_COMPILER_OPTUNE_H_

#include "graph_optimizer/op_compiler/op_compiler.h"

namespace fe {
class OpCompilerOpTune : public OpCompiler {
 public:
  OpCompilerOpTune(const std::string& compiler_name, const std::string &engine_name,
                   const LxFusionOptimizerPtr &lx_fusion_optimizer,
                   const FEOpsKernelInfoStorePtr &ops_kernel_info_store_ptr);

  ~OpCompilerOpTune() override;

  /**
   * @ingroup fe
   * @brief prohibit copy and assign construct
   */
  OpCompilerOpTune(const OpCompilerOpTune &) = delete;

  OpCompilerOpTune &operator=(const OpCompilerOpTune &) = delete;

  /* buff_fus_compile_failed_nodes is not necessary for op-tune. */
  Status RunCompileProcess(ge::ComputeGraph& graph) override;

 private:
  Status UpdateNodeCompileParams(const ge::NodePtr &node, const OpKernelInfoPtr &op_kernel_info_ptr) const override;

  FEOpsKernelInfoStorePtr ops_kernel_info_store_ptr_;
};
}
#endif //FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_COMPILER_OP_COMPILER_OPTUNE_H_

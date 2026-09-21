/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_LOWERING_PASS_OFFLINE_OPTIMIZER_H_
#define AIR_CXX_RUNTIME_V2_LOWERING_PASS_OFFLINE_OPTIMIZER_H_
#include "runtime/gert_api.h"
#include "base_optimizer.h"
#include "model_out_tensor_zero_copy.h"
#include "common/plugin/ge_make_unique_util.h"
#include "constant_expression_motion.h"
#include "copy_flow_launch_fuse.h"
#include "engine/aicpu/pass/aicpu_host_inputs_fuse_pass.h"
#include "remove_launch_free_edge.h"
#include "trust_out_tensor.h"
#include "graph/fast_graph/execute_graph.h"
#include "utils/utils.h"
#include "exe_graph/lowering/lowering_global_data.h"
namespace gert {
namespace bg {
class OfflineOptimizer {
 public:
  explicit OfflineOptimizer(const LoweringOption &option, const LoweringGlobalData &global_data)
      : optimize_option_(option) {
    AddPass("ZeroCopy", ge::MakeUnique<ModelOutTensorZeroCopy>());
    AddPass("CEM", ge::MakeUnique<ConstantExpressionMotion>(&global_data));
    AddPass("CopyFlowLaunch", ge::MakeUnique<CopyFlowLaunchFuse>());
    AddPass("TrustOutTensor", ge::MakeUnique<TrustOutTensor>());
    // todo 这里不应该调用引擎的头文件，整改掉
    AddPass("AicpuFuseHostInputs", ge::MakeUnique<AicpuHostInputsFusePass>());
    if (IsEnableRmLaunchFreeEdge()) {
      AddOncePass("RemoveLaunchFreeEdge", ge::MakeUnique<RemoveLaunchFreeEdge>());
    }
  }

  ge::graphStatus Run(ge::ExecuteGraph *graph) {
    bool changed;
    return optimizer_.Run(graph, changed);
  }

 private:
  void AddPass(const char *name, std::unique_ptr<Pass> pass) {
    if (pass == nullptr) {
      GELOGW("The pass %s initialized failed, skip it", name);
      return;
    }
    pass->SetLoweringOption(optimize_option_);
    optimizer_.AddPass(name, std::move(pass));
  }

  void AddOncePass(const char *name, std::unique_ptr<Pass> pass) {
    if (pass == nullptr) {
      GELOGW("The pass %s initialized failed, skip it", name);
      return;
    }
    pass->SetLoweringOption(optimize_option_);
    optimizer_.AddOncePass(name, std::move(pass));
  }

 private:
  LoweringOption optimize_option_;
  BaseOptimizer optimizer_;
};
}  // namespace bg
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_LOWERING_PASS_OFFLINE_OPTIMIZER_H_

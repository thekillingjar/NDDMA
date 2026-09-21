/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_FUSION_GRAPH_OPTIMIZER_LX_FUSION_OPTIMIZER_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_FUSION_GRAPH_OPTIMIZER_LX_FUSION_OPTIMIZER_H_

#include <string>
#include <memory>
#include "common/aicore_util_types.h"
#include "common/plugin_manager.h"
#include "common/util/op_info_util.h"
#include "fusion_config_manager/fusion_priority_manager.h"
#include "common/opskernel/ops_kernel_info_store.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"
#include "common/graph_tuner_errorcode.h"
#include "graph/compute_graph.h"

namespace fe {
using L2FusionPreOptimizerFunc = std::function<tune::Status(ge::ComputeGraph &, AOEOption,
    const std::vector<std::string> &, std::vector<PassChangeInfo> &)>;
using LxFusionOptimizerFunc = std::function<tune::Status(ge::ComputeGraph &, AOEOption)>;
using LxFusionRecoveryFunc = std::function<tune::Status(ge::ComputeGraph &, const std::vector<ge::NodePtr> &,
    std::vector<ge::NodePtr> *, std::vector<ge::NodePtr> *)>;
using LxFusionFinalizeFunc = std::function<tune::Status(ge::ComputeGraph &)>;

class LxFusionOptimizer {
 public:
  LxFusionOptimizer(const FusionPriorityMgrPtr &fusion_priority_mgr,
                    const OpsKernelInfoStorePtr &ops_kernel_info_store);
  ~LxFusionOptimizer();
  Status Initialize();
  Status Finalize();

  Status LxFusionOptimize(ge::ComputeGraph& graph, LxFusionOptimizeResult& buffer_ret, bool &need_re_compile);
  Status LxFusionRecovery(ge::ComputeGraph& graph, const std::vector<ge::NodePtr>& buff_fus_compile_failed_nodes,
                          std::vector<ge::NodePtr> &buff_fus_rollback_nodes,
                          std::vector<ge::NodePtr> &buff_fus_to_del_nodes);
  Status LxFusionFinalize(ge::ComputeGraph &graph);

 private:
  Status InitFunctions();
  void ClosePlugin();
  Status L2FusionOptimize(ge::ComputeGraph& graph, const OptimizeConfig &optimize_config);
  void UpdateFusionTimesAndSliceInfo(const ge::ComputeGraph& graph,
                                     const std::vector<PassChangeInfo> &pass_info_vec) const;
  Status OptimizeConcatCAxis(ge::ComputeGraph& graph, bool &need_re_compile) const;
  // Status GenerateFormatTuneResult(ge::ComputeGraph& graph, bool &need_re_precompile) const;
  static void GetLxfusionOptimizeConfig(OptimizeConfig &optimize_config);
  Status DoLxFusionOptimize(ge::ComputeGraph& graph, LxFusionOptimizeResult &buffer_ret);
  static LxFusionOptimizeResult GenerateLxFusionOptimizeResult(const Status &l1_buffer_ret,
                                                               const Status &l2_buffer_ret);
private:
  bool init_flag_;
  // priority mngr
  FusionPriorityMgrPtr fusion_priority_mgr_ptr_;
  OpsKernelInfoStorePtr ops_kernel_info_store_ptr_;
  PluginManagerPtr lx_fusion_plugin_manager_;
  LxFusionOptimizerFunc l1_fusion_optimizer_func_{nullptr};
  L2FusionPreOptimizerFunc l2_fusion_pre_optimizer_func_{nullptr};
  LxFusionOptimizerFunc l2_fusion_optimizer_func_{nullptr};
  LxFusionRecoveryFunc l1_recovery_func_{nullptr};
  LxFusionRecoveryFunc l2_recovery_func_{nullptr};
  LxFusionFinalizeFunc lx_fusion_finalize_func_{nullptr};
};
using LxFusionOptimizerPtr = std::shared_ptr<LxFusionOptimizer>;
}
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_FUSION_GRAPH_OPTIMIZER_LX_FUSION_OPTIMIZER_H_

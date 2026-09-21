/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_OPSKERNEL_MANAGER_OPS_KERNEL_MANAGER_H_
#define GE_OPSKERNEL_MANAGER_OPS_KERNEL_MANAGER_H_

#include <map>
#include <set>
#include <memory>
#include <string>
#include <vector>
#include <mutex>

#include "framework/common/debug/log.h"
#include "graph_metadef/common/plugin/plugin_manager.h"
#include "common/plugin/op_tiling_manager.h"
#include "framework/common/ge_inner_error_codes.h"
#include "common/opskernel/ops_kernel_info_store.h"
#include "common/optimizer/graph_optimizer.h"
#include "graph/optimize/graph_optimize.h"
#include "graph/manager/util/graph_optimize_utility.h"
#include "framework/common/ge_inner_error_codes.h"
#include "ge/ge_api_types.h"
#include "runtime/base.h"


namespace ge {
using OpsKernelInfoStorePtr = std::shared_ptr<OpsKernelInfoStore>;

class GE_FUNC_VISIBILITY OpsKernelManager {
 public:
  friend class GELib;
  static OpsKernelManager &GetInstance();
  // get opsKernelInfo by type
  std::vector<OpInfo> GetOpsKernelInfo(const std::string &op_type);

  // get all opsKernelInfo
  const std::map<std::string, std::vector<OpInfo>> &GetAllOpsKernelInfo() const;

  // get opsKernelInfoStore by name
  OpsKernelInfoStorePtr GetOpsKernelInfoStore(const std::string &name) const;

  // get all opsKernelInfoStore
  const std::map<std::string, OpsKernelInfoStorePtr> &GetAllOpsKernelInfoStores() const;

  // get all graph_optimizer
  const std::map<std::string, GraphOptimizerPtr> &GetAllGraphOptimizerObjs() const;

  // get all graph_optimizer by priority
  const std::vector<std::pair<std::string, GraphOptimizerPtr>> &GetAllGraphOptimizerObjsByPriority() const {
    return atomic_first_optimizers_by_priority_;
  }

  const std::map<std::string, std::set<std::string>> &GetCompositeEngines() const {
    return composite_engines_;
  }

  const std::map<std::string, std::string> &GetCompositeEngineKernelLibNames() const {
    return composite_engine_kernel_lib_names_;
  }

  // get subgraphOptimizer by engine name
  void GetGraphOptimizerByEngine(const std::string &engine_name, std::vector<GraphOptimizerPtr> &graph_optimizer);

  bool GetEnableFftsFlag() const {
    return enable_ffts_flag_;
  }

 private:
  OpsKernelManager();
  ~OpsKernelManager();

  // opsKernelManager initialize, load all opsKernelInfoStore and graph_optimizer
  Status Initialize(const std::map<std::string, std::string> &init_options);

  // opsKernelManager finalize, unload all opsKernelInfoStore and graph_optimizer
  Status Finalize();

  Status InitOpKernelInfoStores(const std::map<std::string, std::string> &options);

  Status CheckPluginPtr() const;

  void GetExternalEnginePath(std::string &extern_engine_path, const std::map<std::string, std::string>& options) const;

  void InitOpsKernelInfo();

  Status InitGraphOptimizers(const std::map<std::string, std::string> &options);

  void ClassifyGraphOptimizers();

  void InitGraphOptimizerPriority();

  // Finalize other ops kernel resource
  Status FinalizeOpsKernel();

  PluginManager plugin_manager_;
  OpTilingManager op_tiling_manager_;
  GraphOptimizeUtility graph_optimize_utility_;
  // opsKernelInfoStore
  std::map<std::string, OpsKernelInfoStorePtr> ops_kernel_store_{};
  // graph_optimizer
  std::map<std::string, GraphOptimizerPtr> graph_optimizers_{};
  // composite_graph_optimizer
  std::map<std::string, GraphOptimizerPtr> composite_graph_optimizers_{};
  // atomic_graph_optimizer
  std::map<std::string, GraphOptimizerPtr> atomic_graph_optimizers_{};
  // ordered atomic_graph_optimizer
  std::vector<std::pair<std::string, GraphOptimizerPtr>> atomic_graph_optimizers_by_priority_{};
  // atomic_first graph_optimizer
  std::vector<std::pair<std::string, GraphOptimizerPtr>> atomic_first_optimizers_by_priority_{};
  // {composite_engine, {containing atomic_engine_names}}
  std::map<std::string, std::set<std::string>> composite_engines_{};
  // {composite_engine, composite_engine_kernel_lib_name}
  std::map<std::string, std::string> composite_engine_kernel_lib_names_{};
  // opsKernelInfo
  std::map<std::string, std::vector<OpInfo>> ops_kernel_info_{};

  std::map<std::string, std::string> initialize_{};

  std::vector<OpInfo> empty_op_info_{};

  bool init_flag_;

  bool enable_ffts_flag_{false};
};
}  // namespace ge
#endif  // GE_OPSKERNEL_MANAGER_OPS_KERNEL_MANAGER_H_

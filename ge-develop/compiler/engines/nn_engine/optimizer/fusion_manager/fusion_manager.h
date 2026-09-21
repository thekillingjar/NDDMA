/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_FUSION_MANAGER_FUSION_MANAGER_H_
#define FUSION_ENGINE_OPTIMIZER_FUSION_MANAGER_FUSION_MANAGER_H_

#include <map>
#include <memory>
#include <string>
#include "graph_optimizer/fe_graph_optimizer.h"
#include "graph_optimizer/dsa_optimizer/dsa_graph_optimizer.h"
#include "ops_kernel_store/fe_ops_kernel_info_store.h"
#include "platform/platform_info.h"

namespace fe {
using GraphOptimizerPtr = std::shared_ptr<ge::GraphOptimizer>;
using FEGraphOptimizerPtr = std::shared_ptr<fe::FEGraphOptimizer>;
using DSAGraphOptimizerPtr = std::shared_ptr<fe::DSAGraphOptimizer>;

class FusionManager {
 public:
  static FusionManager &Instance(const std::string &engine_name);

  /*
   * to initialize the subparts of fusion manager
   * param[in] the options of init
   * param[in] engine_name
   * param[in] soc_version soc version from ge
   * return Status(SUCCESS/FAILED)
   */
  Status Initialize(const std::string &engine_name, const map<string, string> &options);

  /*
   * to release the source of fusion manager
   * return Status(SUCCESS/FAILED)
   */
  Status Finalize();

  /*
   * to get the information of OpsKernel InfoStores
   * param[out] the map of OpsKernel InfoStores
   */
  void GetOpsKernelInfoStores(map<string, OpsKernelInfoStorePtr> &op_kern_infos, const std::string &engine_name);

  /*
   * to get the information of Graph Optimizer
   * param[out] the map of Graph Optimizer
   */
  void GetGraphOptimizerObjs(map<string, GraphOptimizerPtr> &graph_optimizers, const std::string &engine_name);

  Status InitBinKernelInfoStore(const std::string &engine_name) const;

  void UpdateOpsKernelInfo(const std::string &engine_name, const FEOpsKernelInfoStorePtr ops_kernel_info_store);

 private:
  FusionManager();
  ~FusionManager();
  FEOpsKernelInfoStorePtr ops_kernel_info_store_;
  FEGraphOptimizerPtr graph_opt_;
  DSAGraphOptimizerPtr dsa_graph_opt_;
  bool inited_;
};
}  // namespace fe

#endif  // FUSION_ENGINE_OPTIMIZER_FUSION_MANAGER_FUSION_MANAGER_H_

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_COMMON_PASS_MANAGER_H_
#define FUSION_ENGINE_OPTIMIZER_COMMON_PASS_MANAGER_H_

#include <vector>
#include "common/opskernel/ops_kernel_info_store.h"
#include "graph/graph.h"
#include "register/graph_optimizer/graph_fusion/graph_pass.h"

namespace fe {
using OpsKernelInfoStorePtr = std::shared_ptr<ge::OpsKernelInfoStore>;
/**
 * @ingroup fe
 * @brief to manager passes
 * @author
 */
/** @brief define pass manager, which provides two interface:
* 1. add passes; 2. run passes */
class PassManager {
 public:
  explicit PassManager();

  /**
   * get passes added through AddPass() in the list
   * @author
   */
  const vector<GraphPass *> &GraphPasses();

  /**
   * add graph level pass through AddPass()
   * @param [in] pass to be added, which will be destructed with pass manager
   * @author
   */
  Status AddPass(const std::string &pass_name, const std::string &engine_name, GraphPass *pass,
                 const std::string fusion_type);

  /**
   * execute all the added passes , to optimize the graph
   * @param [inout] graph, the graph waiting for optimization
   * @return SUCCESS, successfully optimized the graph
   * @return NOT_CHANGED, the graph did not change even though traversed all
   * passes
   * @return FAILED, fail to modify graph
   * @author
   */
  Status Run(ge::ComputeGraph &graph);

  /**
   * execute the appointed passes , to optimize the graph
   * @param [inout] graph, the graph waiting for optimization
   * @return SUCCESS, successfully optimized the graph
   * @return NOT_CHANGED, the graph did not change even though traversed all
   * passes
   * @return FAILED, fail to modify graph
   * @author
   */
  static Status Run(ge::ComputeGraph &graph, vector<GraphPass *> &passes);

  /**
   * execute all the added passes , to optimize the graph
   * @param [inout] graph, the graph waiting for optimization
   * @return SUCCESS, successfully optimized the graph
   * @return NOT_CHANGED, the graph did not change even though traversed all
   * passes
   * @return FAILED, fail to modify graph
   * @author
   */
  Status Run(ge::ComputeGraph &graph, OpsKernelInfoStorePtr ops_kernel_info_store_ptr);

  /**
   * execute the appointed passes , to optimize the graph
   * @param [inout] graph, the graph waiting for optimization
   * @return SUCCESS, successfully optimized the graph
   * @return NOT_CHANGED, the graph did not change even though traversed all
   * passes
   * @return FAILED, fail to modify graph
   * @author
   */
  static Status Run(ge::ComputeGraph &graph, vector<GraphPass *> &passes,
                    OpsKernelInfoStorePtr ops_kernel_info_store_ptr);

  ~PassManager();

 private:
  vector<GraphPass *> graph_passes_;
};
}  // namespace fe
#endif  // FUSION_ENGINE_OPTIMIZER_COMMON_PASS_MANAGER_H_

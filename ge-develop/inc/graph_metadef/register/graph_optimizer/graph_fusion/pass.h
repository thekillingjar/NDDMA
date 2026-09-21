/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/** @defgroup FUSION_PASS_GROUP Fusion Pass Interface */

#ifndef INC_REGISTER_GRAPH_OPTIMIZER_PASS_H_
#define INC_REGISTER_GRAPH_OPTIMIZER_PASS_H_

#include "graph/compute_graph.h"
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"

namespace fe {

/** fusion pass
 * @ingroup GRAPH_PASS_GROUP
 * network level pass
 */
template <typename T>
class Pass {
 public:
  virtual ~Pass() {}

  /** execute pass
   *
   * @param [in] graph, the graph waiting for pass level optimization
   * @return SUCCESS, successfully optimized the graph by the pass
   * @return NOT_CHANGED, the graph did not change
   * @return FAILED, fail to modify graph
   */
  virtual Status Run(ge::ComputeGraph &graph) = 0;

  void SetName(const std::string &name) { name_ = name; }

  std::string GetName() { return name_; }

 private:
  std::string name_;
};

}  // namespace fe

#endif  // INC_REGISTER_GRAPH_OPTIMIZER_PASS_H_

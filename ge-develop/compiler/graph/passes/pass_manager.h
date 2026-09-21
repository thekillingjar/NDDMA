/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_INC_PASS_MANAGER_H_
#define GE_INC_PASS_MANAGER_H_

#include <vector>
#include "graph/passes/graph_pass.h"

namespace ge {
/// @ingroup domi_omg
/// @brief pass manager
class PassManager {
 public:
  /// get graph passes
  const std::vector<std::pair<std::string, GraphPass *>> &GraphPasses() const;

  /// Add graph pass
  /// @param [in] pass  Pass to be added, it will be destroyed when pass manager destroys.
  Status AddPass(const std::string &pass_name, GraphPass *pass);

  /// Optimize graph with added pass
  /// @param [inout] graph graph to be optimized
  /// @return SUCCESS optimize successfully
  /// @return NOT_CHANGED not optimized
  /// @return others optimize failed
  Status Run(const ge::ComputeGraphPtr &graph);

  /// Optimize graph with specified pass
  /// @param [inout] graph graph to be optimized
  /// @param [in] passes passes to be used
  /// @return SUCCESS optimize successfully
  /// @return NOT_CHANGED not optimized
  /// @return others optimized failed
  static Status Run(const ge::ComputeGraphPtr &graph,
                    std::vector<std::pair<std::string, GraphPass *>> &names_to_passes);

  ~PassManager();

 private:
  std::vector<std::pair<std::string, GraphPass *>> names_to_graph_passes_;
};
}  // namespace ge
#endif  // GE_INC_PASS_MANAGER_H_

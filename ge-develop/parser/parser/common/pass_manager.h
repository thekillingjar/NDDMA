/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PARSER_COMMON_PASS_MANAGER_H_
#define PARSER_COMMON_PASS_MANAGER_H_

#include <vector>

#include "common/graph_pass.h"

namespace ge {
namespace parser {
///
/// @ingroup domi_omg
/// @brief pass manager
/// @author
///
class PassManager {
public:

  /// get graph passes
  /// @author
  const std::vector<std::pair<std::string, GraphPass *>> &GraphPasses() const;

  /// Add graph pass
  /// @param [in] pass  Pass to be added, it will be destroyed when pass manager destroys.
  /// @author
  Status AddPass(const string &pass_name, GraphPass *const pass);

  /// Optimize graph with added pass
  /// @param [inout] graph graph to be optimized
  /// @return SUCCESS optimize successfully
  /// @return NOT_CHANGED not optimized
  /// @return others optimize failed
  /// @author
  Status Run(const ge::ComputeGraphPtr &graph);

  /// Optimize graph with specified pass
  /// @param [inout] graph graph to be optimized
  /// @param [in] passes passes to be used
  /// @return SUCCESS optimize successfully
  /// @return NOT_CHANGED not optimized
  /// @return others optimized failed
  /// @author
  static Status Run(const ge::ComputeGraphPtr &graph,
                    std::vector<std::pair<std::string, GraphPass *>> &names_to_passes);

  ~PassManager();

private:
  std::vector<std::pair<std::string, GraphPass *>> names_to_graph_passes_;
};
}  // namespace parser
}  // namespace ge
#endif  // PARSER_COMMON_PASS_MANAGER_H_

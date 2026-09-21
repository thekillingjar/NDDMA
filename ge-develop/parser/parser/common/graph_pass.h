/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PARSER_COMMON_GRAPH_PASS_H_
#define PARSER_COMMON_GRAPH_PASS_H_

#include "framework/common/debug/ge_log.h"
#include "graph/compute_graph.h"
#include "common/pass.h"
#include "base/err_msg.h"

namespace ge {
/// @ingroup domi_omg
/// @brief graph pass
/// @author
class GraphPass : public Pass<ge::ComputeGraph> {
 public:
  /// run graph pass
  /// @param [in] graph graph to be optimized
  /// @return SUCCESS optimize successfully
  /// @return NOT_CHANGED not optimized
  /// @return others optimized failed
  /// @author
  virtual Status Run(ge::ComputeGraphPtr graph) = 0;
  virtual Status ClearStatus() { return SUCCESS; };
};
}  // namespace ge

#endif  // PARSER_COMMON_GRAPH_PASS_H_

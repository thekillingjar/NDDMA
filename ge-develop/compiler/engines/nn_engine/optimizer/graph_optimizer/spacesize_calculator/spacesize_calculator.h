/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_SPACESIZE_CALCULATOR_SPACESIZE_CALCULATOR_H_
#define FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_SPACESIZE_CALCULATOR_SPACESIZE_CALCULATOR_H_

#include "graph/compute_graph.h"
#include "graph/op_desc.h"
#include "graph_optimizer/graph_optimize_register_error_codes.h"

namespace fe {

/** @brief provide the capability of calculating
* workspace and input/output size */
class SpaceSizeCalculator {
 public:
  SpaceSizeCalculator();

  ~SpaceSizeCalculator();

  /**
   * @ingroup fe
   * @brief prohibit copy and assign construct
   */
  SpaceSizeCalculator(const SpaceSizeCalculator &) = delete;

  SpaceSizeCalculator &operator=(const SpaceSizeCalculator &) = delete;

  /**
  * Calculate params of OP in the graph
  * @param graph ComputeGraph
  *      false : calculate the param of OP whose impl type is not tbe
  */
  Status CalculateRunningParams(const ge::ComputeGraph &graph) const;

  Status CalculateAICoreRunningParams(const ge::ComputeGraph &graph) const;
};
}  // namespace fe

#endif  // FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_SPACESIZE_CALCULATOR_SPACESIZE_CALCULATOR_H_

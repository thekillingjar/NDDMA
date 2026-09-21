/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_NODE_OPTIMIZER_SPLIT_N_OPTIMIZER_H_
#define FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_NODE_OPTIMIZER_SPLIT_N_OPTIMIZER_H_

#include <memory>
#include <string>
#include "common/fe_inner_error_codes.h"
#include "common/fe_log.h"
#include "common/fe_utils.h"
#include "common/graph/fe_graph_utils.h"
#include "common/op_info_common.h"
#include "common/optimizer/graph_optimizer.h"
#include "common/optimizer/graph_optimizer_types.h"
#include "graph/compute_graph.h"

namespace fe {
/** @brief provide two interface: 1. optimize original graph 2. optimize fused
* sub graph */
class SplitNOptimizer {
 public:
  /**
   *  @ingroup fe
   *  @brief set fusion_virtual_op info for op
   *  @param [in|out] graph compute graph
   *  @return SUCCESS or FAILED
   */
  Status SetFusionVirtualOp(const ge::ComputeGraph &graph) const;
  bool InputCheck(ge::NodePtr split_node) const;
  bool OutputCheck(ge::NodePtr split_node) const;

 private:
  bool NeedSkip(const ge::NodePtr &node, const ge::OpDescPtr &op_desc) const;
  void GetRealSplitDimFromOriginalFormatToFormat(const ge::OpDescPtr &op_desc, int64_t &split_dim) const;
  static bool InvalidNodeType(const string& node_type);
  static bool InvalidNodeAttr(const ge::OpDescPtr& node_desc);
};
}  // namespace fe
#endif  // FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_NODE_OPTIMIZER_SPLIT_N_OPTIMIZER_H_

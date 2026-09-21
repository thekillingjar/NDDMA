/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_DFLOW_DATA_FLOW_ATTR_UTILS_H
#define AIR_DFLOW_DATA_FLOW_ATTR_UTILS_H
#include "graph/compute_graph.h"
#include "ge_common/ge_api_types.h"

namespace ge {
class DataFlowAttrUtils {
 public:
  static Status SupplementFlowAttr(const ComputeGraphPtr &root_graph);

 private:
  static Status SupplementNodeFlowAttr(const NodePtr &node);
  static graphStatus SupplementMismatchEdge(const DataAnchorPtr &peer_out_anchor, const NodePtr &node,
                                            GeTensorDescPtr &in_tensor, const int32_t def_fifo_depth,
                                            const std::string &def_enqueue_policy);
};
}  // namespace ge
#endif  // AIR_DFLOW_DATA_FLOW_ATTR_UTILS_H

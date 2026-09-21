/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_UB_FUSION_L1_FUSION_GRAPH_MERGE_H_
#define FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_UB_FUSION_L1_FUSION_GRAPH_MERGE_H_

#include "graph_optimizer/ub_fusion/fusion_graph_merge/fusion_graph_merge.h"

namespace fe {
class L1FusionGraphMerge : public FusionGraphMerge {
 public:
  L1FusionGraphMerge(const std::string &scope_attr, const GraphCommPtr &graph_comm_ptr);
  ~L1FusionGraphMerge() override;
};
}

#endif  // FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_UB_FUSION_L1_FUSION_GRAPH_MERGE_H_

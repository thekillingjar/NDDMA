/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/ub_fusion/fusion_graph_merge/l1_fusion_graph_merge.h"

namespace fe {
L1FusionGraphMerge::L1FusionGraphMerge(const std::string &scope_attr, const GraphCommPtr &graph_comm_ptr)
    : FusionGraphMerge(scope_attr, graph_comm_ptr) {}

L1FusionGraphMerge::~L1FusionGraphMerge() {}
}

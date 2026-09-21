/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PARSER_COMMON_AUTO_MAPPING_SUBGRAPH_IO_INDEX_FUNC_H_
#define PARSER_COMMON_AUTO_MAPPING_SUBGRAPH_IO_INDEX_FUNC_H_

#include <functional>
#include "graph/graph.h"
#include "register/register_error_codes.h"
#include "graph/node.h"

namespace ge {
domi::Status AutoMappingSubgraphIndexByDataNodeAndOutputNodesInfo(
    const ge::Graph &graph,
    const std::function<domi::Status(int data_index, int &parent_input_index)> &input,
    const std::function<domi::Status(int netoutput_index, int &parent_output_index)> &output);
// only data node may set default NHWC/NCHW format by parser when call NpuOnnxGraphOp
// this function should be called before AutoMappingSubgraphIndexByDataNodeAndOutputNodesInfo
domi::Status AutoMappingSubgraphDataFormat(const NodePtr &parent_node, const ge::Graph &graph);
} // namespace ge
#endif  // PARSER_COMMON_AUTO_MAPPING_SUBGRAPH_IO_INDEX_FUNC_H_

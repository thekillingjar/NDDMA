/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_COMMON_OME_CONTEXT_H_
#define GE_GRAPH_COMMON_OME_CONTEXT_H_

#include "framework/omg/omg_inner_types.h"

namespace ge {
struct OmeContext {
  std::string dynamic_node_type;
  std::vector<NodePtr> data_nodes;
  std::vector<NodePtr> getnext_nosink_nodes;
  std::vector<std::vector<int32_t>> dynamic_shape_dims;
  std::vector<std::pair<std::string, std::vector<int64_t>>> user_input_dims;
  bool is_subgraph_multi_batch = false;
};
}  // namespace ge
#endif  // GE_GRAPH_COMMON_OME_CONTEXT_H_

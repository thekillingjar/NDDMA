/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMPILER_PNE_DATA_FLOW_GRAPH_INNER_PP_LOADER_H_
#define GE_COMPILER_PNE_DATA_FLOW_GRAPH_INNER_PP_LOADER_H_

#include "dflow/compiler/data_flow_graph/data_flow_graph.h"

namespace ge {
class InnerPpLoader {
 public:
  static Status LoadProcessPoint(const dataflow::ProcessPoint &process_point, DataFlowGraph &data_flow_graph,
                                 const NodePtr &node);

 private:
  static Status LoadModelPp(const dataflow::ProcessPoint &process_point, DataFlowGraph &data_flow_graph,
                            const NodePtr &node);
  static Status LoadModel(const DataFlowGraph &data_flow_graph, const std::string &model_path,
                          FlowModelPtr &flow_model);
};
}  // namespace ge
#endif  // GE_COMPILER_PNE_DATA_FLOW_GRAPH_INNER_PP_LOADER_H_

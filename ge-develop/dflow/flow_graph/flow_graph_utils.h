/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLOW_GRAPH_FLOW_GRAPH_UTILS_H_
#define FLOW_GRAPH_FLOW_GRAPH_UTILS_H_

#include "flow_graph/data_flow.h"

namespace ge {
namespace dflow {
class FlowGraphUtils {
public:
  static const std::map<const ge::string, const GraphPp> &GetInvokedClosures(const FunctionPp *func_pp);
  static const std::map<const ge::string, const FlowGraphPp> &GetInvokedFlowClosures(const FunctionPp *func_pp);
};
}  // namespace dflow
}  // namespace ge
#endif // FLOW_GRAPH_FLOW_GRAPH_UTILS_H_

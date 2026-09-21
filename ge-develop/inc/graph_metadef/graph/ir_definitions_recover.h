/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GRAPH_IR_DEFINITIONS_RECOVER_H_
#define GRAPH_IR_DEFINITIONS_RECOVER_H_

#include <string>
#include "graph/compute_graph.h"

namespace ge {
ge::graphStatus RecoverIrDefinitions(const ge::ComputeGraphPtr &graph, const vector<std::string> &attr_names = {});
ge::graphStatus RecoverOpDescIrDefinition(const ge::OpDescPtr &desc, const std::string &op_type = "");
bool CheckIrSpec(const ge::OpDescPtr &desc);
} // namespace ge
#endif  // GRAPH_IR_DEFINITIONS_RECOVER_H_

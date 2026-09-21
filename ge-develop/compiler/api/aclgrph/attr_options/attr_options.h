/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATTR_OPTIONS_H_
#define ATTR_OPTIONS_H_

#include <string>
#include "graph/compute_graph.h"
#include "graph/ge_error_codes.h"

namespace ge {
bool IsOriginalOpFind(const OpDescPtr &op_desc, const std::string &op_name);
bool IsOpTypeEqual(const ge::NodePtr &node, const std::string &op_type);
bool IsContainOpType(const std::string &cfg_line, std::string &op_type);
graphStatus KeepDtypeFunc(const ComputeGraphPtr &graph, const std::string &cfg_path);
graphStatus WeightCompressFunc(const ComputeGraphPtr &graph, const std::string &cfg_path);
}  // namespace
#endif // ATTR_OPTIONS_H_

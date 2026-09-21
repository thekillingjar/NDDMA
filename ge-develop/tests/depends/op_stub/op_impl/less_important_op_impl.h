/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_LESS_IMPORTANT_OP_IMPL_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_LESS_IMPORTANT_OP_IMPL_H
#include "graph/node.h"

namespace gert {
void MockLessImportantNodeConverter(const ge::NodePtr &node, bool override = false);
void MockLessImportantNodeKernel(const ge::NodePtr &node);
void MockLessImportantNodeConverter(const ge::ComputeGraphPtr &graph, const std::set<std::string> &exclude_ops = {},
                                    const std::set<std::string> &exclude_nodes = {}, bool override = false);
void MockLessImportantNodeKernel(const ge::ComputeGraphPtr &graph, const std::set<std::string> &exclude_ops = {},
                                 const std::set<std::string> &exclude_nodes = {});
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_LESS_IMPORTANT_OP_IMPL_H

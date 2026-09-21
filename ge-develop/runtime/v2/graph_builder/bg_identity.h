/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_IDENTITY_H_
#define AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_IDENTITY_H_
#include "exe_graph/lowering/dev_mem_value_holder.h"
#include "graph/node.h"
#include "kernel/tensor_attr.h"
namespace gert {
namespace bg {
ValueHolderPtr IdentityShape(const ValueHolderPtr &shape);
std::vector<ValueHolderPtr> IdentityShape(const std::vector<ValueHolderPtr> &shapes);

ValueHolderPtr IdentityAddr(const ValueHolderPtr &addr);
std::vector<ValueHolderPtr> IdentityAddr(const std::vector<ValueHolderPtr> &addrs);

std::vector<ValueHolderPtr> IdentityShapeAndAddr(const std::vector<ValueHolderPtr> &shapes,
                                                 const std::vector<ValueHolderPtr> &addrs);

std::vector<DevMemValueHolderPtr> IdentityAddr(const std::vector<DevMemValueHolderPtr> &addrs, const int64_t stream_id);

std::vector<DevMemValueHolderPtr> IdentityShapeAndAddr(const std::vector<ValueHolderPtr> &shapes,
                                                       const std::vector<DevMemValueHolderPtr> &addrs,
                                                       const int64_t stream_id);
}  // namespace bg
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_IDENTITY_H_

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_IR_ATTRS_H_
#define AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_IR_ATTRS_H_
#include "graph/node.h"
#include "value_holder.h"
namespace gert {
namespace bg {
bool GetAllIrAttrs(const ge::NodePtr &node, std::vector<std::vector<uint8_t>> &runtime_attrs);
std::unique_ptr<uint8_t[]> CreateAttrBuffer(const ge::NodePtr &node, size_t &size);
std::unique_ptr<uint8_t[]> CreateAttrBuffer(const ge::NodePtr &node,
                                            const std::vector<ge::AnyValue> &runtime_attrs_list,
                                            size_t &size);
std::unique_ptr<uint8_t[]> CreateAttrBufferWithoutIr(const ge::NodePtr &node,
                                                     const std::vector<ge::AnyValue> &runtime_attrs_list,
                                                     size_t &size);
}
}
#endif  // AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_IR_ATTRS_H_

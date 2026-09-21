/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_KERNEL_CONTEXT_EXTEND_H_
#define AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_KERNEL_CONTEXT_EXTEND_H_
#include "graph/node.h"
#include "buffer_pool.h"
#include "register/op_impl_registry.h"
namespace gert {
namespace bg {
std::unique_ptr<uint8_t[]> CreateComputeNodeInfo(const ge::NodePtr &node, BufferPool &buffer_pool);
std::unique_ptr<uint8_t[]> CreateComputeNodeInfo(const ge::NodePtr &node, BufferPool &buffer_pool, size_t &total_size);
std::unique_ptr<uint8_t[]> CreateComputeNodeInfo(const ge::NodePtr &node, BufferPool &buffer_pool,
    const gert::OpImplRegisterV2::PrivateAttrList &private_attrs, size_t &total_size);
std::unique_ptr<uint8_t[]> CreateComputeNodeInfoWithoutIrAttr(const ge::NodePtr &node, BufferPool &buffer_pool,
    const gert::OpImplRegisterV2::PrivateAttrList &private_attrs, size_t &total_size);
}
}
#endif  // AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_KERNEL_CONTEXT_EXTEND_H_

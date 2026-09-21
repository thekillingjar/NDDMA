/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_TENSOR_H_
#define AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_TENSOR_H_
#include "exe_graph/lowering/value_holder.h"
#include "graph/node.h"
#include "kernel/tensor_attr.h"
namespace gert {
namespace bg {
ValueHolderPtr CreateBuildTensorAttr(const ge::NodePtr &node, int32_t index, TensorPlacement placement);
ValueHolderPtr BuildTensor(const ge::NodePtr &node, int32_t index, TensorPlacement placement,
                           const ValueHolderPtr &storage_shape, const ValueHolderPtr &addr);

// RefTensor只保存地址，不管理生命周期
ValueHolderPtr BuildRefTensor(const ge::NodePtr &node, int32_t index, TensorPlacement placement,
                                   const ValueHolderPtr &storage_shape, const ValueHolderPtr &addr);

ValueHolderPtr BuildTensorWithoutGuarder(const ge::NodePtr &node, int32_t index, TensorPlacement placement,
                                         const ValueHolderPtr &storage_shape, const ValueHolderPtr &addr);

ValueHolderPtr CopyD2H(const ge::NodePtr &node, int32_t out_index, const ValueHolderPtr &storage_shape,
                       const ValueHolderPtr &addr);
}  // namespace bg
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_TENSOR_H_

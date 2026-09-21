/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_INFER_SHAPE_H_
#define AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_INFER_SHAPE_H_
#include "exe_graph/lowering/value_holder.h"
#include "graph/node.h"
#include "graph/debug/ge_attr_define.h"
#include "exe_graph/lowering/lowering_global_data.h"

namespace gert {
namespace bg {
// 为了减少执行图中const节点构造，构图新提供了一个infershape接口，新增一个global_data作为入参，接口的作用用于：
// 减少执行图时const节点的构造，对于infershape而言，一个optype对应一个infershape.
// 例如未使用新接口的场景下，一个node在lowering infershape时会产生一个Const(op_type) + FindInferShapeFunc，
// 使用新接口后，对于相同optype的infershape节点，不必产生多个infershape函数，只需产生一个infershape函数即可,存储与global_data中
std::vector<ValueHolderPtr> InferStorageShape(const ge::NodePtr &node, const std::vector<ValueHolderPtr> &input_shapes,
                                              LoweringGlobalData &global_data);
std::vector<ValueHolderPtr> InferUbGraphShape(const ge::ComputeGraphPtr &compute_graph,
                                              const std::vector<ValueHolderPtr> &input_shapes,
                                              LoweringGlobalData &global_data);
bool IsUnkownShape(const ge::OpDescPtr &op_desc);
bool IsOutputUnkownShape(const ge::OpDescPtr &op_desc);

inline int64_t GetParentNodeInputIndex(const ge::NodePtr &node) {
  int64_t index = -1;
  ge::AttrUtils::GetInt(node->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, index);
  return index;
}
}
}
#endif // AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_BG_INFER_SHAPE_H_

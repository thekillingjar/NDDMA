/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef H53F1A984_1D06_4458_9595_0A6DC60EA9CE
#define H53F1A984_1D06_4458_9595_0A6DC60EA9CE

#include "easy_graph/graph/node.h"
#include "ge_graph_dsl/ge.h"
#include "ge_graph_dsl/op_desc/op_desc_ptr_box.h"
#include "ge_graph_dsl/op_desc/op_desc_cfg_box.h"
#include "graph/op_desc.h"

GE_NS_BEGIN

inline const ::EG_NS::NodeId OpDescNodeBuild(const ::EG_NS::NodeId &id) {
  return id;
}

template<typename... GRAPHS, SUBGRAPH_CONCEPT(GRAPHS, ::EG_NS::Graph)>
inline ::EG_NS::Node OpDescNodeBuild(const ::EG_NS::NodeId &id, const GRAPHS &... graphs) {
  return ::EG_NS::Node(id, graphs...);
}

template<typename... GRAPHS, SUBGRAPH_CONCEPT(GRAPHS, ::EG_NS::Graph)>
inline ::EG_NS::Node OpDescNodeBuild(const OpDescPtr &op, const GRAPHS &... graphs) {
  return ::EG_NS::Node(op->GetName(), BOX_OF(::GE_NS::OpDescPtrBox, op), graphs...);
}

template<typename... GRAPHS, SUBGRAPH_CONCEPT(GRAPHS, ::EG_NS::Graph)>
inline ::EG_NS::Node OpDescNodeBuild(const ::EG_NS::NodeId &id, const OpType &opType, const GRAPHS &... graphs) {
  return ::EG_NS::Node(id, BOX_OF(OpDescCfgBox, opType), graphs...);
}

template<typename... GRAPHS, SUBGRAPH_CONCEPT(GRAPHS, ::EG_NS::Graph)>
inline ::EG_NS::Node OpDescNodeBuild(const ::EG_NS::NodeId &id, const OpDescCfgBox &opBox, const GRAPHS &... graphs) {
  return ::EG_NS::Node(id, BOX_OF(OpDescCfgBox, opBox), graphs...);
}

GE_NS_END

#endif /* H53F1A984_1D06_4458_9595_0A6DC60EA9CE */

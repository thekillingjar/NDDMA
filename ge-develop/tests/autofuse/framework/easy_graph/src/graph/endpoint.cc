/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "easy_graph/graph/endpoint.h"

EG_NS_BEGIN

Endpoint::Endpoint(const NodeId &nodeId, const PortId &portId) : node_id_(nodeId), port_id_(portId) {}

__DEF_EQUALS(Endpoint) {
  return (node_id_ == rhs.node_id_) && (port_id_ == rhs.port_id_);
}

__DEF_COMP(Endpoint) {
  if (node_id_ < rhs.node_id_)
    return true;
  if ((node_id_ == rhs.node_id_) && (port_id_ < rhs.port_id_))
    return true;
  return false;
}

NodeId Endpoint::getNodeId() const {
  return node_id_;
}

PortId Endpoint::getPortId() const {
  return port_id_;
}

EG_NS_END

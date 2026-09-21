/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef H8DB48A37_3257_4E15_8869_09E58221ADE8
#define H8DB48A37_3257_4E15_8869_09E58221ADE8

#include "easy_graph/graph/node_id.h"
#include "easy_graph/graph/port_id.h"
#include "easy_graph/infra/operator.h"

EG_NS_BEGIN

struct Endpoint {
  Endpoint(const NodeId &, const PortId &);

  __DECL_COMP(Endpoint);

  NodeId getNodeId() const;
  PortId getPortId() const;

 private:
  NodeId node_id_;
  PortId port_id_;
};

EG_NS_END

#endif

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef H35695B82_E9E5_419D_A6B4_C13FB0842C9F
#define H35695B82_E9E5_419D_A6B4_C13FB0842C9F

#include <string>
#include "easy_graph/graph/edge_type.h"
#include "easy_graph/graph/port_id.h"

EG_NS_BEGIN

struct Link {
  explicit Link(EdgeType type) : type_(type) {
    Reset(type);
  }

  Link(EdgeType type, const std::string &label, PortId srcPortId, PortId dstPortId)
      : type_(type), label_(label), src_port_id_(srcPortId), dst_port_id_(dstPortId) {}

  void Reset(EdgeType type) {
    this->type_ = type;
    this->label_ = "";
    this->src_port_id_ = UNDEFINED_PORT_ID;
    this->dst_port_id_ = UNDEFINED_PORT_ID;
  }

  EdgeType type_;
  std::string label_;
  PortId src_port_id_;
  PortId dst_port_id_;
};

EG_NS_END

#endif

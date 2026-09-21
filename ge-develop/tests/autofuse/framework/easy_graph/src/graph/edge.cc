/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "easy_graph/graph/edge.h"

EG_NS_BEGIN

Edge::Edge(const EdgeType type, const std::string &label, const Endpoint &src, const Endpoint &dst)
    : type_(type), label_(label), src_(src), dst_(dst) {}

__DEF_EQUALS(Edge) {
  return (type_ == rhs.type_) && (src_ == rhs.src_) && (dst_ == rhs.dst_);
}

__DEF_COMP(Edge) {
  if (src_ < rhs.src_)
    return true;
  if ((src_ == rhs.src_) && (dst_ < rhs.dst_))
    return true;
  if ((src_ == rhs.src_) && (dst_ < rhs.dst_) && (type_ < rhs.type_))
    return true;
  return false;
}

EdgeType Edge::GetType() const {
  return type_;
}

std::string Edge::GetLabel() const {
  return label_;
}

Endpoint Edge::GetSrc() const {
  return src_;
}
Endpoint Edge::GetDst() const {
  return dst_;
}

EG_NS_END

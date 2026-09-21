/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/lowering/value_holder_utils.h"

namespace gert {
namespace bg {
bool ValueHolderUtils::IsNodeValid(const ValueHolderPtr &holder) {
  if (holder == nullptr) {
    return false;
  }
  return (holder->fast_node_ != nullptr);
}

bool ValueHolderUtils::IsNodeEqual(const ValueHolderPtr &src, const ValueHolderPtr &dst) {
  if (src == dst) {
    return true;
  }
  if ((src == nullptr) || (dst == nullptr)) {
    return false;
  }
  return (src->fast_node_ == dst->fast_node_);
}

std::string ValueHolderUtils::GetNodeName(const ValueHolderPtr &holder) {
  if (holder == nullptr) {
    return "";
  }
  return holder->op_desc_->GetName();
}
const char *ValueHolderUtils::GetNodeNameBarePtr(const ValueHolderPtr &holder) {
  if (holder == nullptr) {
    return "";
  }
  return holder->op_desc_->GetNamePtr();
}
std::string ValueHolderUtils::GetNodeType(const ValueHolderPtr &holder) {
  if (holder == nullptr) {
    return "";
  }
  return holder->op_desc_->GetType();
}
const char *ValueHolderUtils::GetNodeTypeBarePtr(const ValueHolderPtr &holder) {
  if (holder == nullptr) {
    return "";
  }
  return holder->op_desc_->GetTypePtr();
}

ge::OpDescPtr ValueHolderUtils::GetNodeOpDesc(const ValueHolderPtr &holder) {
  if (holder == nullptr) {
    return nullptr;
  }
  return holder->op_desc_;
}
ge::OpDesc *ValueHolderUtils::GetNodeOpDescBarePtr(const ValueHolderPtr &holder) {
  if (holder == nullptr) {
    return nullptr;
  }
  return holder->op_desc_.get();
}

bool ValueHolderUtils::IsDirectlyControlled(const bg::ValueHolderPtr &src, const bg::ValueHolderPtr &dst) {
  if (src == nullptr || dst == nullptr) {
    return false;
  }
  return dst->fast_node_->IsDirectlyControlledByNode(src->fast_node_);
}
} // bg
} // gert

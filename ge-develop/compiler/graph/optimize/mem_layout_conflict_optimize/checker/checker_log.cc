/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "checker_log.h"
#include <sstream>
#include "common/checker.h"
#include "graph/optimize/mem_layout_conflict_optimize/checker/checker.h"
#include "graph/optimize/mem_layout_conflict_optimize/mem_layout_conflict_util.h"

namespace ge {
namespace {
constexpr size_t kMaxLogLen = 200U;
}
const std::string &CheckerLog::ToStr(const AnchorAttribute &type) {
  const std::map<AnchorAttribute, std::string>::const_iterator iter = Checker::kAnchorTypeStr.find(type);
  if (iter != Checker::kAnchorTypeStr.cend()) {
    return iter->second;
  }
  static std::string dumy_str{"undefined"};
  return dumy_str;
}

std::string CheckerLog::ToStr(const SmallVector<AnchorAttribute, ANCHOR_ATTR_RESERVED> &types) {
  std::stringstream ss;
  ss << "[";
  for (auto iter = types.cbegin(); iter != types.cend(); ++iter) {
    ss << ToStr(*iter);
    if (iter + 1U != types.cend()) {
      ss << ", ";
    }
  }
  ss << "]";
  return ss.str();
}

std::string CheckerLog::ToStr(const NodeIndexIO &node_index_io) {
  auto name = node_index_io.node_->GetName().substr(0, kMaxLogLen);
  if (name.size() == kMaxLogLen) {
    name += "...TopoId_" + std::to_string(node_index_io.node_->GetOpDesc()->GetId());
  }
  return name + ((node_index_io.io_type_ == kOut) ? "_out_" : "_in_") + std::to_string(node_index_io.index_);
}

std::string CheckerLog::ToStr(const OutDataAnchorPtr &out_anchor) {
  GE_ASSERT_NOTNULL(out_anchor);
  const auto node = out_anchor->GetOwnerNodeBarePtr();
  GE_ASSERT_NOTNULL(node);
  auto name = node->GetName().substr(0, kMaxLogLen);
  if (name.size() == kMaxLogLen) {
    name += "...TopoId_" + std::to_string(node->GetOpDesc()->GetId());
  }
  return name + "_out_" + std::to_string(out_anchor->GetIdx());
}

std::string CheckerLog::ToStr(const InDataAnchorPtr &in_anchor) {
  GE_ASSERT_NOTNULL(in_anchor);
  const auto node = in_anchor->GetOwnerNodeBarePtr();
  GE_ASSERT_NOTNULL(node);
  auto name = node->GetName().substr(0, kMaxLogLen);
  if (name.size() == kMaxLogLen) {
    name += "...TopoId_" + std::to_string(node->GetOpDesc()->GetId());
  }
  return name + "_in_" + std::to_string(in_anchor->GetIdx());
}
}  // namespace ge

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_MEM_CHECKER_LOG_H_
#define GE_GRAPH_PASSES_MEM_CHECKER_LOG_H_

#include <sstream>
#include "graph/small_vector.h"
#include "common/ge_common/debug/log.h"
#include "checker.h"

namespace ge {
struct CheckerLog {
  static const std::string &ToStr(const AnchorAttribute &type);
  static std::string ToStr(const SmallVector<AnchorAttribute, ATTR_BIT_MAX_LEN> &types);
  static std::string ToStr(const NodeIndexIO &node_index_io);
  static std::string ToStr(const OutDataAnchorPtr &out_anchor);
  static std::string ToStr(const InDataAnchorPtr &in_anchor);
};

#define GE_MEM_LAYOUT_CONFLICT_LOGI(context, node_index_io)                                                 \
  do {                                                                                                      \
    GELOGI("[MemConflict][Conflict] type: [%s, %s], anchor: [%s, %s], will insert %s %s",                   \
           CheckerLog::ToStr((context).type_a[(context).type_a_index]).c_str(),                             \
           CheckerLog::ToStr((context).type_b[(context).type_b_index]).c_str(),                             \
           CheckerLog::ToStr((context).node_a).c_str(), CheckerLog::ToStr((context).node_b).c_str(),        \
           (node_index_io).io_type_ == kIn ? "before" : "after", CheckerLog::ToStr(node_index_io).c_str()); \
  } while (false)
}  // namespace ge
#endif  // GE_GRAPH_PASSES_MEM_CHECKER_LOG_H_

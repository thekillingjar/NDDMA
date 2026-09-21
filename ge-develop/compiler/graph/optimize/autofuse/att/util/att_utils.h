/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef UTIL_ATT_UTILS_H_
#define UTIL_ATT_UTILS_H_

#include "ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "base/model_info.h"
#include <set>
#include "gen_model_info/parser/tuning_space.h"
namespace att {
class AttUtils {
 public:
  static bool IsLoadNode(ge::AscNode *node);
  static bool IsStoreNode(ge::AscNode *node);
  static bool IsLoadStoreNode(ge::AscNode *node);
  static bool IsTileSplitAxis(const AttAxisPtr &axis);
  static bool GetLastTileSplitAxisName(ge::AscNode &node, const ge::AscGraph &graph, std::string &axis_name);

  // 收集Reduce轴的原始轴名称（供 ascend_graph_parser 和 arg_list_reorder 共用）
  static void CollectReduceAxisNames(const NodeInfo &node_info,
                                     std::set<std::string> &reduce_axis_orig_names);

  // 收集Broadcast轴的原始轴名称（供 ascend_graph_parser 和 arg_list_reorder 共用）
  static void CollectBroadcastAxisNames(const NodeInfo &node_info,
                                        std::set<std::string> &broadcast_axis_orig_names);
};
}
#endif // UTIL_ATT_UTILS_H_

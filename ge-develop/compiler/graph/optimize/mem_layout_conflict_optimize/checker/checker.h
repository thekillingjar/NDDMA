/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_MEM_CHECKER_H_
#define GE_GRAPH_PASSES_MEM_CHECKER_H_
#include <map>
#include <vector>
#include "framework/common/debug/ge_log.h"
#include "common/plugin/ge_make_unique_util.h"
#include "graph/utils/graph_utils.h"
#include "graph/small_vector.h"
namespace ge {

enum AnchorAttribute {
  ANCHOR_ATTR_USER_MEMORY_INPUT = 0,
  ANCHOR_ATTR_USER_MEMORY_OUTPUT,
  ANCHOR_ATTR_IMMUTABLE_ADDRESS_OUTPUT,
  ANCHOR_ATTR_UNSUPPORTED_ADDRESS_REFRESH_OPERATOR_OUTPUT,
  ANCHOR_ATTR_UNSUPPORTED_ADDRESS_REFRESH_OPERATOR_INPUT,
  ANCHOR_ATTR_CONTINUOUS_INPUT,
  ANCHOR_ATTR_CONTINUOUS_OUTPUT,
  ANCHOR_ATTR_NOPADDING_CONTINUOUS_INPUT,
  ANCHOR_ATTR_NOPADDING_CONTINUOUS_OUTPUT,
  ANCHOR_ATTR_RTS_SPECIAL_TYPE_INPUT,
  ANCHOR_ATTR_RTS_SPECIAL_TYPE_OUTPUT,
  ANCHOR_ATTR_REFERENCE_OUTPUT,
  ANCHOR_ATTR_NORMAL_INPUT,
  ANCHOR_ATTR_NORMAL_OUTPUT,
  ANCHOR_ATTR_RESERVED
};

#define ATTR_BIT_MAX_LEN ANCHOR_ATTR_RESERVED

using TypeVector = SmallVector<AnchorAttribute, ATTR_BIT_MAX_LEN>;

using AnchorTypes = std::unordered_map<AnchorPtr, TypeVector>;

using NodeIndexIOVector = std::vector<NodeIndexIO>;

using AnchorSet = std::set<AnchorPtr>;

struct GraphInfo {
  bool is_root_graph_static;
  bool is_feature_map_refreshable;
  bool is_support_user_input_nopadding_continuous_output;
  /*
   * 开启动态图和静态图复用为true，纯静态图虚拟地址不变化，但是物理地址会变化。
   * 静态图中有2种rts算子STREAMSWITCH/LABELSWITCHBYINDEX会使用物理地址，因此不支持地址刷新。
   * 所以如果开启了动静态内存复用，需要将feature map分为2段，这3种算子的内存在feature map单独一段，其虚拟地址和物理地址不会变化。
   */
  bool is_physical_memory_refreshable;
};

struct CheckFuncContext {
  const ge::NodeIndexIO &node_a;
  const ge::NodeIndexIO &node_b;
  size_t type_a_index;
  size_t type_b_index;
  const SmallVector<AnchorAttribute, ATTR_BIT_MAX_LEN> &type_a;
  const SmallVector<AnchorAttribute, ATTR_BIT_MAX_LEN> &type_b;
  const NodeIndexIOVector &all_nodes;
  const GraphInfo &graph_info;
  AnchorSet &result;
};

struct CheckerFunc {
  CheckerFunc(const AnchorAttribute &type_a, const AnchorAttribute &type_b,
              std::function<Status(CheckFuncContext &)> function, bool as_symblol = false)
      : type_a_(type_a), type_b_(type_b), func_(function), call_as_symbol_(as_symblol) {}

  std::function<Status(CheckFuncContext &)> GetFunc() const { return func_; }
  void SetCallAsSymbol() { call_as_symbol_ = true; }
  bool IsCallAsSymbol() const { return call_as_symbol_; }

  AnchorAttribute type_a_;
  AnchorAttribute type_b_;
  std::function<Status(CheckFuncContext &)> func_;
  bool call_as_symbol_; // 默认是将symbol中anchor两两组合调用checker函数，as_symbol为true表示按照symbol调用该checker函数，一个symbol调用一次
};

std::set<std::pair<AnchorAttribute, AnchorAttribute>> RegAllAbsoluteConflict();

std::set<std::pair<AnchorAttribute, AnchorAttribute>> RegAllAbsoluteNoConflict();

static inline void RegAbsoluteSet(AnchorAttribute type_a, AnchorAttribute type_b,
                                  std::set<std::pair<AnchorAttribute, AnchorAttribute>> &set) {
  set.emplace(type_a, type_b);
  set.emplace(type_b, type_a);
}

inline bool IsInAbsoluteSet(const AnchorAttribute &type_a,
                                   const AnchorAttribute &type_b,
                                   const std::set<std::pair<AnchorAttribute, AnchorAttribute>> &set) {
  return set.find({type_a, type_b}) != set.cend();
}

class Checker {
 public:
  Checker();

  ~Checker() = default;

  void SaveTypes(const SmallVector<AnchorAttribute, ANCHOR_ATTR_RESERVED> &types, const AnchorPtr anchor);

  void SaveRefMap(const OutDataAnchorPtr &out_anchor, const NodePtr &src_node);

  const NodePtr &GetRefSrcAnchor(const OutDataAnchorPtr &out_anchor) const;

  void ClearTypes();

  const SmallVector<AnchorAttribute, ATTR_BIT_MAX_LEN> &GetTypes(const ge::NodeIndexIO &node) const;

  Status CheckConditionConflict(const AnchorAttribute &type_a,
                                const AnchorAttribute &type_b,
                                CheckFuncContext &context) const;

  Status CheckConflict(CheckFuncContext &context, std::vector<vector_bit_t> &visit_flag) const;

  Status CheckConditionConflictAsSymbol(const AnchorAttribute &type_a,
                                        const AnchorAttribute &type_b,
                                        CheckFuncContext &context,
                                        std::vector<vector_bit_t> &visit_flag) const;

  bool CheckNoConflict(const AnchorAttribute &type_a, const AnchorAttribute &type_b) const;

  Status CheckAbsoluteConflict(CheckFuncContext &context) const;

  bool IsSkip(CheckFuncContext &context) const;

  static const std::map<AnchorAttribute, std::string> kAnchorTypeStr;
 private:
  std::vector<vector<CheckerFunc *>> checkers_;
  std::set<std::pair<AnchorAttribute, AnchorAttribute>> absolute_conflict_set_;
  std::set<std::pair<AnchorAttribute, AnchorAttribute>> absolute_no_conflict_set_;
  AnchorTypes anchor_type_vectors_;
  std::map<OutDataAnchorPtr, NodePtr> ref_map_;
};

}  // namespace ge

#endif  // GE_GRAPH_PASSES_MEM_CHECKER_H_

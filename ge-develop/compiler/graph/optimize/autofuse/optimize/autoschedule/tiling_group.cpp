/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tiling_group.h"
#include "ascir_ops_utils.h"
#include "ascgraph_info_complete.h"
#include "ascir_utils.h"
#include "schedule_utils.h"
#include "graph/symbolizer/symbolic_utils.h"

namespace optimize::autoschedule {
namespace {
constexpr int64_t kDefaultGroup = -1;
void PrintGroup(std::stringstream &ss, const std::string &name, const std::vector<ge::AxisId> &group) {
  ss << name << "[";
  for (auto axis_id : group) {
    ss << axis_id << ",";
  }
  ss << "],";
}
}  // namespace

std::string AxisGroup::ToString() const {
  std::stringstream ss;
  PrintGroup(ss, "XGroup:", x_group);
  PrintGroup(ss, "YGroup:", y_group);
  PrintGroup(ss, "RGroup:", r_group);
  PrintGroup(ss, "NGroup:", n_group);

  ss << "Order:[";
  for (auto &axis : axes_order) {
    ss << axis << ",";
  }
  ss << "]";
  return ss.str();
}

bool AxisGroup::IsEmpty() const {
  return x_group.empty() && y_group.empty() && r_group.empty() && n_group.empty();
}

extern "C" {
int32_t GenAscGraphAxisGroup(const ge::AscGraph &graph, AxisGroup &axes_group) {
  GELOGD("Enter [GenAscGraphAxisGroup], graph name: %s, axis size: %zu", graph.GetName().c_str(),
         graph.GetAllAxis().size());
  for (auto &axis : graph.GetAllAxis()) {
    GE_CHECK_NOTNULL(axis);
    GELOGD("[GenAscGraphAxisGroup] AscGraph axis.name=%s, axis.id=%ld, axis.size=%s", axis->name.c_str(), axis->id,
           axis->size.Str().get());
  }
  GE_CHK_STATUS_RET(AscGraphInfoComplete::CompleteApiInfo(graph));

  GE_CHK_STATUS_RET(TilingGroup::GenTilingGroup(graph, axes_group), "Gen axis map failed, graph[%s] may be invalid.",
                    graph.GetName().c_str());
  GELOGD("Finish [GenAscGraphAxisGroup], graph name: %s", graph.GetName().c_str());
  return ge::SUCCESS;
}

bool CanMergeAxisGroup(const AxisGroup &lhs, const AxisGroup &rhs, AxisGroup &merged_group, bool is_ge_call) {
  merged_group = lhs;
  AxisGroup new_rhs = rhs;
  auto ret = TilingGroup::MergeAxesGroup(merged_group, new_rhs, true, is_ge_call);
  GELOGI("Merged axis group: %s to target: %s, res:[%d].", rhs.ToString().c_str(), lhs.ToString().c_str(), ret);
  return ret;
}
}  // extern "C"

static bool CheckYAndYR(const std::vector<ascir::AxisId> &cur_y_group, const std::vector<ascir::AxisId> &new_y_group,
                        const std::vector<ascir::AxisId> &new_r_group, const std::vector<size_t> &new_axes_order,
                        const bool is_canfuse_call) {
  if (cur_y_group.size() != (new_y_group.size() + new_r_group.size())) {
    return false;
  }

  if (is_canfuse_call) {
    std::set<int64_t> l_g(cur_y_group.begin(), cur_y_group.end());
    std::set<int64_t> r_g(new_y_group.begin(), new_y_group.end());
    r_g.insert(new_r_group.begin(), new_r_group.end());
    return l_g == r_g;
  }

  for (size_t i = 0UL; i < new_y_group.size(); i++) {
    if (cur_y_group[new_axes_order[i]] != new_y_group[i]) {
      return false;
    }
  }
  size_t y_size = new_y_group.size();
  for (size_t i = 0UL; i < new_r_group.size(); i++) {
    if (cur_y_group[new_axes_order[y_size + i]] != new_r_group[i]) {
      return false;
    }
  }
  return true;
}

static bool CheckYAndR(const std::vector<ascir::AxisId> &cur_y_group, const std::vector<ascir::AxisId> &new_r_group,
                       const bool is_canfuse_call) {
  if (is_canfuse_call) {
    std::set<int64_t> l_g(cur_y_group.begin(), cur_y_group.end());
    std::set<int64_t> r_g(new_r_group.begin(), new_r_group.end());
    return l_g == r_g;
  }
  return cur_y_group == new_r_group;
}

static bool MergeYAndY(AxisGroup &lhs_group, AxisGroup &rhs_group, const bool is_canfuse_call, const bool is_ge_call) {
  (void)is_ge_call;
  if (lhs_group.y_group.size() != rhs_group.y_group.size()) {
    return false;
  }
  if (is_canfuse_call) {
    std::set<int64_t> lhs(lhs_group.y_group.begin(), lhs_group.y_group.end());
    std::set<int64_t> rhs(rhs_group.y_group.begin(), rhs_group.y_group.end());
    return lhs == rhs;
  }

  for (size_t i = 0UL; i < lhs_group.y_group.size(); i++) {
    if (lhs_group.y_group[i] != rhs_group.y_group[i]) {
      return false;
    }
  }
  return true;
}

static bool MergeYAndR(AxisGroup &lhs_group, AxisGroup &rhs_group, const bool is_canfuse_call, const bool is_ge_call) {
  (void)is_ge_call;
  if (!CheckYAndR(lhs_group.y_group, rhs_group.r_group, is_canfuse_call)) {
    return false;
  }
  lhs_group = rhs_group;
  return true;
}

static bool MergeRAndY(AxisGroup &lhs_group, AxisGroup &rhs_group, const bool is_canfuse_call, const bool is_ge_call) {
  (void)is_ge_call;
  return CheckYAndR(rhs_group.y_group, lhs_group.r_group, is_canfuse_call);
}

static bool MergeYAndXY(AxisGroup &lhs_group, AxisGroup &rhs_group, const bool is_canfuse_call, const bool is_ge_call) {
  (void)is_canfuse_call;
  (void)is_ge_call;
  std::set<int64_t> lhs(lhs_group.y_group.begin(), lhs_group.y_group.end());
  std::set<int64_t> rhs(rhs_group.y_group.begin(), rhs_group.y_group.end());
  rhs.insert(rhs_group.x_group.begin(), rhs_group.x_group.end());
  if (lhs != rhs) {
    return false;
  }
  lhs_group = rhs_group;
  return true;
}

static bool MergeXYAndY(AxisGroup &lhs_group, AxisGroup &rhs_group, const bool is_canfuse_call, const bool is_ge_call) {
  (void)is_canfuse_call;
  (void)is_ge_call;
  std::set<int64_t> lhs(lhs_group.y_group.begin(), lhs_group.y_group.end());
  lhs.insert(lhs_group.x_group.begin(), lhs_group.x_group.end());
  std::set<int64_t> rhs(rhs_group.y_group.begin(), rhs_group.y_group.end());
  if (lhs != rhs) {
    return false;
  }
  return true;
}

static bool MergeYAndYR(AxisGroup &lhs_group, AxisGroup &rhs_group, const bool is_canfuse_call, const bool is_ge_call) {
  (void)is_ge_call;
  if (!CheckYAndYR(lhs_group.y_group, rhs_group.y_group, rhs_group.r_group, rhs_group.axes_order, is_canfuse_call)) {
    return false;
  }
  lhs_group = rhs_group;
  return true;
}

static bool MergeYRAndY(AxisGroup &lhs_group, AxisGroup &rhs_group, const bool is_canfuse_call, const bool is_ge_call) {
  (void)is_ge_call;
  return CheckYAndYR(rhs_group.y_group, lhs_group.y_group, lhs_group.r_group, lhs_group.axes_order, is_canfuse_call);
}

static bool MergeYRAndYR(AxisGroup &lhs_group, AxisGroup &rhs_group, const bool is_canfuse_call, const bool is_ge_call) {
  (void)is_ge_call;
  if (is_ge_call && is_canfuse_call) {
    return false;
  }
  // y0 == y1, r0 == r1，可以做融合
  if (is_canfuse_call) {
    std::set<int64_t> l_y(lhs_group.y_group.begin(), lhs_group.y_group.end());
    std::set<int64_t> r_y(rhs_group.y_group.begin(), rhs_group.y_group.end());
    std::set<int64_t> l_r(lhs_group.r_group.begin(), lhs_group.r_group.end());
    std::set<int64_t> r_r(rhs_group.r_group.begin(), rhs_group.r_group.end());
    return (l_y == r_y) && (l_r == r_r);
  }
  return (lhs_group == rhs_group);
}

GroupType TilingGroup::GetGroupType(const AxisGroup &axes_group) {
  GroupType type = GroupType::GROUP_INVALID;
  if (!axes_group.x_group.empty()) {
    type = static_cast<GroupType>(type | GroupType::GROUP_X);
  }
  if (!axes_group.y_group.empty()) {
    type = static_cast<GroupType>(type | GroupType::GROUP_Y);
  }
  if (!axes_group.r_group.empty()) {
    type = static_cast<GroupType>(type | GroupType::GROUP_R);
  }
  return type;
}

static void RemoveNGroupAxisInXYRGroup(const std::set<ge::AxisId> &n_groups, AxisGroup &single_node_axes_group) {
  if (n_groups.empty()) {
    return;
  }
  single_node_axes_group.n_group.assign(n_groups.begin(), n_groups.end());
  std::vector<size_t> new_axes_order;
  size_t idx = 0UL;
  for (auto iter = single_node_axes_group.x_group.begin(); iter != single_node_axes_group.x_group.end();) {
    if (n_groups.count(*iter) > 0UL) {
      iter = single_node_axes_group.x_group.erase(iter);
    } else {
      ++iter;
      new_axes_order.push_back(single_node_axes_group.axes_order[idx]);
    }
    ++idx;
  }

  for (auto iter = single_node_axes_group.y_group.begin(); iter != single_node_axes_group.y_group.end();) {
    if (n_groups.count(*iter) > 0UL) {
      iter = single_node_axes_group.y_group.erase(iter);
    } else {
      ++iter;
      new_axes_order.push_back(single_node_axes_group.axes_order[idx]);
    }
    ++idx;
  }

  for (auto iter = single_node_axes_group.r_group.begin(); iter != single_node_axes_group.r_group.end();) {
    if (n_groups.count(*iter) > 0UL) {
      iter = single_node_axes_group.r_group.erase(iter);
    } else {
      ++iter;
      new_axes_order.push_back(single_node_axes_group.axes_order[idx]);
    }
    ++idx;
  }

  single_node_axes_group.axes_order = new_axes_order;
}

static void MergeNGroup(AxisGroup &lhs_group, AxisGroup &rhs_group) {
  std::set<ge::AxisId> n_groupset(lhs_group.n_group.begin(), lhs_group.n_group.end());
  n_groupset.insert(rhs_group.n_group.begin(), rhs_group.n_group.end());
  RemoveNGroupAxisInXYRGroup(n_groupset, lhs_group);
  RemoveNGroupAxisInXYRGroup(n_groupset, rhs_group);
}

// 在此之前不应该做任何的合轴，否则很难推断出轴关系
// 1. (1, y0, 1) merge (1, y1, 1) ==> (1, max(y0, y1), 1), 如果节点2是broadcast节点，要求节点1的tiling
// group与节点2的输入节点的tiling group一致，否则应该满足y0 == y1
// 2. (1, y0, 1) merge (1, y1, r1) ==> (1, y1, r1), 要求y0 == y1 U r1
// 3. (1, y0, 1) merge (x1, y1, 1) ==> (x1, y1, 1)
// 4. (1, y0, 1) merge (x1, y1, r1) ==> (x1, y1, r1)
// 5. (1, y0, r0) merge (1, y1, 1) ==> (1, y0, r0), 要求y1 == y0 U r0
// 6. (1, y0, r0) merge (1, y1, r1) ==> (1, y0, r0), 要求y1 == y0, r1 == r0
// 7. (1, y0, r0) merge (x1, y1, 1) ==> (x2, y2, r0)
// 8. (1, y0, r0) merge (x1, y1, r1) ==> (x1, y1, r1)
// 9. (x0, y0, 1) merge (1, y1, 1) ==> (x0, y0, 1)
// 10. (x0, y0, 1) merge (1, y1, r1) ==> (x2, y2, r1)
// 11. (x0, y0, 1) merge (x1, y1, 1) ==> (x0, y0, 1)
// 12. (x0, y0, 1) merge (x1, y1, r1) ==> (x1, y1, r1)
// 13. (x0, y0, r0) merge (1, y1, 1) ==> (x0, y0, r0)
// 14. (x0, y0, r0) merge (1, y1, r1) ==> (x0, y0, r0)
// 15. (x0, y0, r0) merge (x1, y1, 1) ==> (x0, y0, r0)
// 16. (x0, y0, r0) merge (x1, y1, r1) ==> (x0, y0, r0)
bool TilingGroup::MergeAxesGroup(AxisGroup &target, AxisGroup &src, const bool is_canfuse_call, const bool is_ge_call) {
  // 先单独处理Ngroup 再做其他group的合并, 否则当前多套轴下无法做merge
  MergeNGroup(target, src);
  if (target.x_group.empty() && target.y_group.empty() && target.r_group.empty()) {
    target = src;
    return true;
  }

  static std::map<std::pair<GroupType, GroupType>, AxisGroupMergeFunc> type_to_merge_func = {
      {{GroupType::GROUP_Y, GroupType::GROUP_Y}, MergeYAndY},
      {{GroupType::GROUP_Y, GroupType::GROUP_YR}, MergeYAndYR},
      {{GroupType::GROUP_YR, GroupType::GROUP_Y}, MergeYRAndY},
      {{GroupType::GROUP_YR, GroupType::GROUP_YR}, MergeYRAndYR},
      {{GroupType::GROUP_Y, GroupType::GROUP_XY}, MergeYAndXY},
      {{GroupType::GROUP_XY, GroupType::GROUP_Y}, MergeXYAndY},
      {{GroupType::GROUP_Y, GroupType::GROUP_R}, MergeYAndR},
      {{GroupType::GROUP_R, GroupType::GROUP_Y}, MergeRAndY},
  };

  auto merge_type = std::make_pair(GetGroupType(target), GetGroupType(src));
  auto iter = type_to_merge_func.find(merge_type);
  if (iter == type_to_merge_func.end()) {
    return false;
  }
  return iter->second(target, src, is_canfuse_call, is_ge_call);
}

Status TilingGroup::GenTilingGroup(const ascir::ImplGraph &impl_graph, AxisGroup &tiling_group, bool is_reduce_fullload) {
  std::vector<std::pair<std::string, AxisGroup>> node_name_to_tiling_group;
  std::set<ge::AxisId> n_groupset;
  for (const auto &node : impl_graph.GetAllNodes()) {
    if (ScheduleUtils::IsBuffer(node)) {
      continue;
    }
    AxisGroup single_node_axes_group;
    GE_CHK_STATUS_RET(GenAxisGroupForSingleNode(*node, single_node_axes_group, is_reduce_fullload));
    n_groupset.insert(single_node_axes_group.n_group.begin(), single_node_axes_group.n_group.end());
    node_name_to_tiling_group.emplace_back(node->GetName(), single_node_axes_group);
    GELOGD("GenTilingGroup node: %s, group: %s.", node->GetName().c_str(), single_node_axes_group.ToString().c_str());
  }
  tiling_group.n_group.assign(n_groupset.begin(), n_groupset.end());

  // merge xyr groups.
  for (auto &iter : node_name_to_tiling_group) {
    GE_ASSERT_TRUE(MergeAxesGroup(tiling_group, iter.second),
                   "Merged axis group: %s to target: %s failed, the graph [%s] cannot be fused.",
                   iter.second.ToString().c_str(), tiling_group.ToString().c_str(), impl_graph.GetName().c_str());
  }

  return ge::SUCCESS;
}

Status TilingGroup::GenElewiseTilingGroup(ge::AscNode &node, AxisGroup &axes_group) {
  GE_CHK_STATUS_RET(ScheduleUtils::GetLoopAxis(node, axes_group.y_group));
  axes_group.axes_order.reserve(axes_group.y_group.size());
  for (size_t i = 0UL; i < axes_group.y_group.size(); ++i) {
    axes_group.axes_order.push_back(i);
  }
  return ge::SUCCESS;
}

std::vector<ge::AxisId> CalcReduceAxes(const std::vector<ge::Expression>& src_strides,
                                       const std::vector<ge::Expression>& dst_strides,
                                       const std::vector<ascir::AxisId>& axes) {
  GE_ASSERT_TRUE((src_strides.size() == dst_strides.size()),
                 "The output dim cnt [%zu] of reduce mismatch with input dim cnt [%zu].", dst_strides.size(),
                 src_strides.size());
  GE_ASSERT_TRUE((src_strides.size() == axes.size()),
                 "The input dim cnt [%zu] of reduce mismatch with input dim cnt [%zu].", src_strides.size(),
                 axes.size());
  std::vector<ascir::AxisId> reduce_axes;
  for (size_t i = 0; i < src_strides.size(); ++i) {
    if (ge::SymbolicUtils::StaticCheckEq(src_strides[i], dst_strides[i]) != ge::TriBool::kTrue &&
        ge::SymbolicUtils::StaticCheckEq(dst_strides[i], ge::sym::kSymbolZero) == ge::TriBool::kTrue) {
      reduce_axes.push_back(axes[i]);
    }
  }
  return reduce_axes;
}

Status TilingGroup::GenReduceTilingGroup(ge::AscNode &node, AxisGroup &axes_group) {
  std::vector<ascir::AxisId> axes;
  GE_CHK_STATUS_RET(ScheduleUtils::GetLoopAxis(node, axes), "Get loop axis failed.");
  axes_group.axes_order.resize(axes.size());
  std::vector<ascir::SizeExpr> src_strides;
  GE_CHK_STATUS_RET(ScheduleUtils::GetReduceInputStrides(node, src_strides), "Get loop strides failed.");
  axes_group.r_group = CalcReduceAxes(src_strides, node.outputs[0].attr.strides, axes);
  int64_t y_order_index = 0;
  int64_t r_order_index = axes.size() - axes_group.r_group.size();
  for (size_t i = 0; i < axes.size(); ++i) {
    if (std::find(axes_group.r_group.begin(), axes_group.r_group.end(), axes[i]) == axes_group.r_group.end()) {
      axes_group.y_group.push_back(axes[i]);
      axes_group.axes_order[y_order_index++] = i;
    } else {
      axes_group.axes_order[r_order_index++] = i;
    }
  }
  return ge::SUCCESS;
}

Status TilingGroup::GenReduceTilingGroupFullLoad(ge::AscNode &node, AxisGroup &axes_group) {
  std::vector<ascir::AxisId> axes;
  GE_CHK_STATUS_RET(ScheduleUtils::GetLoopAxis(node, axes), "Get loop axis failed.");
  axes_group.axes_order.resize(axes.size());
  std::vector<ascir::SizeExpr> src_strides;
  GE_CHK_STATUS_RET(ScheduleUtils::GetReduceInputStrides(node, src_strides), "Get loop strides failed.");
  axes_group.n_group = CalcReduceAxes(src_strides, node.outputs[0].attr.strides, axes);
  int64_t y_order_index = 0;
  int64_t r_order_index = axes.size() - axes_group.n_group.size();
  for (size_t i = 0; i < axes.size(); ++i) {
    if (std::find(axes_group.n_group.begin(), axes_group.n_group.end(), axes[i]) == axes_group.n_group.end()) {
      axes_group.y_group.push_back(axes[i]);
      axes_group.axes_order[y_order_index++] = i;
    } else {
      axes_group.axes_order[r_order_index++] = i;
    }
  }
  return ge::SUCCESS;
}

void PlaceRemainingAxis(const int64_t index, std::set<int64_t> &remaining_axis, AxisGroup &axes_group,
                        vector<ascir::AxisId> &output_axis) {
  for (int64_t j = index; j >= 0; --j) {
    if (remaining_axis.find(output_axis[j]) == remaining_axis.end()) {
      continue;
    }
    axes_group.y_group.emplace(axes_group.y_group.begin(), output_axis[j]);
  }
}

Status TilingGroup::GenTransposeTilingGroup(ge::AscNode &node, AxisGroup &axes_group) {
  std::vector<ascir::AxisId> input_axis;
  GE_CHK_STATUS_RET(ScheduleUtils::GetInputForTranspose(node, input_axis), "Get Transpose loop axis failed.");
  std::vector<ascir::AxisId> &output_axis = node.outputs[0].attr.axis;
  GE_ASSERT_TRUE((input_axis.size() == output_axis.size()),
                 "The output dim cnt [%zu] of Transpose mismatch with input dim cnt [%zu].", output_axis.size(),
                 input_axis.size());
  std::set<int64_t> remaining_axis(input_axis.begin(), input_axis.end());
  GELOGD("GenTransposeTilingGroup input_axis %s, output_axis %s", ge::ViewMemberToString(input_axis).c_str(),
         ge::ViewMemberToString(output_axis).c_str());

  // 1. 从尾轴向前取轴，如果轴相同，分别放入x_group和y_group,直到轴不同为止
  int64_t i = static_cast<int64_t>(input_axis.size()) - 1;
  for (; i >= 0; --i) {
    // 如果input和output在该位置上的轴一致 => ngroup加入此轴, 当前att不能保证Xgroup和YGroup切同一根轴的结果一样
    if (input_axis[i] == output_axis[i]) {
      axes_group.n_group.emplace(axes_group.n_group.begin(), input_axis[i]);
      remaining_axis.erase(input_axis[i]);
    } else {
      break;
    }
  }

  // 2. 从第一个不同的轴开始处理
  for (; i >= 0; --i) {
    // 看输入的轴是否在y_group中，如果不在，则放入x_group
    if (std::find(axes_group.y_group.begin(), axes_group.y_group.end(), input_axis[i]) == axes_group.y_group.end()) {
      axes_group.x_group.emplace(axes_group.x_group.begin(), input_axis[i]);
      remaining_axis.erase(input_axis[i]);
    }
    // 看输出的轴是否在x_group中，如果不在则放入y_group中
    if (std::find(axes_group.x_group.begin(), axes_group.x_group.end(), output_axis[i]) == axes_group.x_group.end()) {
      axes_group.y_group.emplace(axes_group.y_group.begin(), output_axis[i]);
      remaining_axis.erase(output_axis[i]);
    }
    // 如果都在，则停止，将剩余的轴放入y_group
    if (std::find(axes_group.y_group.begin(), axes_group.y_group.end(), input_axis[i]) != axes_group.y_group.end() &&
        std::find(axes_group.x_group.begin(), axes_group.x_group.end(), output_axis[i]) != axes_group.x_group.end()) {
      // 将剩余的轴放入y_group
      PlaceRemainingAxis(i, remaining_axis, axes_group, output_axis);
      break;
    }
  }

  for (int64_t axis : axes_group.x_group) {
    axes_group.axes_order.emplace_back(axis);
  }

  for (int64_t axis : axes_group.y_group) {
    axes_group.axes_order.emplace_back(axis);
  }

  GELOGD("GenTransposeTilingGroup TilingGroup %s", axes_group.ToString().c_str());
  return ge::SUCCESS;
}

Status TilingGroup::GenConcatTilingGroup(ge::AscNode &node, AxisGroup &axes_group) {
  std::vector<ascir::AxisId> axes;
  GE_CHK_STATUS_RET(ScheduleUtils::GetLoopAxis(node, axes), "Get loop axis failed.");
  const std::vector<ascir::SizeExpr> &input_repeats = node.inputs[0].attr.repeats;    // 前端保证
  const std::vector<ascir::SizeExpr> &output_repeats = node.outputs[0].attr.repeats;  // 前端保证
  GE_ASSERT_TRUE((input_repeats.size() == output_repeats.size()),
                 "The output dim cnt [%zu] of concat mismatch with input dim cnt [%zu].", output_repeats.size(),
                 input_repeats.size());

  size_t concat_dim{0UL};
  for (size_t i = 0UL; i < input_repeats.size(); ++i) {
    if (ge::SymbolicUtils::StaticCheckEq(input_repeats[i], output_repeats[i]) != ge::TriBool::kTrue) {
      GELOGD("Concat node [%s], input_repeats[%zu]=%s, output_repeats[%zu]=%s", node.GetNamePtr(), i,
             input_repeats[i].Str().get(), i, output_repeats[i].Str().get());
      concat_dim = i;
      break;
    }
  }
  GELOGD("Concat node [%s], concat_dim is %zu", node.GetNamePtr(), concat_dim);
  axes_group.axes_order.reserve(axes.size());
  for (size_t i = 0UL; i < axes.size(); ++i) {
    if (i < concat_dim) {
      axes_group.y_group.push_back(axes[i]);
      axes_group.axes_order.push_back(i);
    } else {
      axes_group.n_group.push_back(axes[i]);
    }
  }
  // 由于concat的输入和输出大小不同，codegen需要输入输出在concat_dim上用不同符号来表达
  // 因此需要同时将输入和输出的concat_dim的axis_id加到中ngroup中
  for (auto &input_view : node.inputs()) {
    if (input_view->attr.axis.size() > concat_dim) {
      axes_group.n_group.push_back(input_view->attr.axis[concat_dim]);
    }
  }
  return ge::SUCCESS;
}

Status TilingGroup::GenSplitTilingGroup(ge::AscNode &node, AxisGroup &axes_group) {
  std::vector<ascir::AxisId> axes;
  GE_CHK_STATUS_RET(ScheduleUtils::GetLoopAxis(node, axes), "Get loop axis failed.");
  const std::vector<ascir::SizeExpr> &input_repeats = node.inputs[0].attr.repeats;    // 前端保证
  const std::vector<ascir::SizeExpr> &output_repeats = node.outputs[0].attr.repeats;  // 前端保证
  GE_ASSERT_TRUE((input_repeats.size() == output_repeats.size()),
                 "The output dim cnt [%zu] of split mismatch with input dim cnt [%zu].", output_repeats.size(),
                 input_repeats.size());

  ascir::SizeExpr pre_size = ge::ops::One;
  size_t split_dim{0UL};
  for (size_t i = 0UL; i < input_repeats.size(); ++i) {
    if (ge::SymbolicUtils::StaticCheckEq(input_repeats[i], output_repeats[i]) != ge::TriBool::kTrue) {
      GELOGD("Split node [%s], input_repeats[%zu]=%s, output_repeats[%zu]=%s, pre_size=%s.", node.GetNamePtr(), i,
             input_repeats[i].Str().get(), i, output_repeats[i].Str().get(), pre_size.Str().get());
      split_dim = i;
      break;
    }
    pre_size = pre_size * input_repeats[i];
  }
  GELOGD("Split node [%s], split_dim is %zu", node.GetNamePtr(), split_dim);
  // 首轴Split在schedule前会被替换成store，因此canfuse会按照elewise生成AxisGroup
  if (split_dim == 0UL || (ge::SymbolicUtils::StaticCheckEq(pre_size, ge::ops::One) == ge::TriBool::kTrue)) {
    GE_CHK_STATUS_RET(ScheduleUtils::GetLoopAxis(node, axes_group.y_group));
    axes_group.axes_order.reserve(axes_group.y_group.size());
    for (size_t i = 0UL; i < axes_group.y_group.size(); ++i) {
      axes_group.axes_order.push_back(i);
    }
    return ge::SUCCESS;
  }  

  axes_group.axes_order.reserve(axes.size());
  for (size_t i = 0UL; i < axes.size(); ++i) {
    if (i < split_dim) {
      axes_group.y_group.push_back(axes[i]);
      axes_group.axes_order.push_back(i);
    } else {
      axes_group.n_group.push_back(axes[i]);
    }
  }
  // 由于split的输入和输出大小不同，codegen需要输入输出在split_dim上用不同符号来表达
  // 因此需要同时将输入和输出的split_dim的axis_id加到中ngroup中
  for (auto &input_view : node.outputs()) {
    if (input_view->attr.axis.size() > split_dim) {
      axes_group.n_group.push_back(input_view->attr.axis[split_dim]);
    }
  }
  return ge::SUCCESS;
}

void TilingGroup::NormGroup(AxisGroup &group) {
  if (group.x_group.empty()) {
    group.x_group.push_back(kDefaultGroup);
  }
  if (group.r_group.empty()) {
    group.r_group.push_back(kDefaultGroup);
  }
}

// tiling group的生成应该考虑api的能力，特别是当api有明确的轴不可切时，应当放入Ngroup
Status TilingGroup::GenAxisGroupForSingleNode(ge::AscNode &node, AxisGroup &axes_group, bool is_reduce_ar_fullLoad) {
  static std::map<ascir::ComputeType, AxisGroupGenFunc> compute_type_to_group_gen_func = {
      {ge::ComputeType::kComputeElewise, TilingGroup::GenElewiseTilingGroup},
      {ge::ComputeType::kComputeBroadcast, TilingGroup::GenElewiseTilingGroup},
      {ge::ComputeType::kComputeGather, TilingGroup::GenElewiseTilingGroup},
      {ge::ComputeType::kComputeLoad, TilingGroup::GenElewiseTilingGroup},
      {ge::ComputeType::kComputeStore, TilingGroup::GenElewiseTilingGroup},
      {ge::ComputeType::kComputeReduce, TilingGroup::GenReduceTilingGroup},
      {ge::ComputeType::kComputeConcat, TilingGroup::GenConcatTilingGroup},
      {ge::ComputeType::kComputeTranspose, TilingGroup::GenTransposeTilingGroup},
      {ge::ComputeType::kComputeSplit, TilingGroup::GenSplitTilingGroup},
      {ge::ComputeType::kComputeCube, TilingGroup::GenElewiseTilingGroup},
  };

  if (node.attr.api.compute_type == ge::ComputeType::kComputeReduce && is_reduce_ar_fullLoad) {
    GE_CHK_STATUS_RET(GenReduceTilingGroupFullLoad(node, axes_group), "Gen tiling case failed, compute type [%u].",
                      static_cast<uint32_t>(node.attr.api.compute_type));
    return ge::SUCCESS;
  }

  auto iter = compute_type_to_group_gen_func.find(node.attr.api.compute_type);
  GE_ASSERT_TRUE(iter != compute_type_to_group_gen_func.end(), "Unsupported compute type [%u].",
                 node.attr.api.compute_type);
  GE_CHK_STATUS_RET(iter->second(node, axes_group), "Gen tiling case failed, compute type [%u].",
                    static_cast<uint32_t>(node.attr.api.compute_type));
  return ge::SUCCESS;
}

}  // namespace optimize::autoschedule

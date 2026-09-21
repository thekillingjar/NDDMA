/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
 */
#include "att_utils.h"
#include "att_const_values.h"
#include <numeric>
namespace att {
namespace {
bool IsOps(const ge::AscNode *node, const std::string &node_type) {
  return node->GetType() == node_type;
}

// 辅助函数：收集轴的原始名称
void CollectOrigAxisNames(const SubAxis *dim, std::set<std::string> &orig_names) {
  for (auto *orig_axis : dim->orig_axis) {
    orig_names.insert(orig_axis->name);
  }
}

// 辅助函数：检查单个维度是否为Reduce轴
bool IsReduceAxis(const SubAxis *dim, const Expr &repeat, const Expr &stride,
                         const std::vector<TensorPtr> &output_tensors) {
  for (const auto &output_tensor : output_tensors) {
    for (size_t j = 0; j < output_tensor->dim_info.size(); j++) {
      if (output_tensor->dim_info[j]->name != dim->name) {
        continue;
      }
      auto output_repeat = output_tensor->repeat[j];
      auto output_stride = output_tensor->stride[j];

      bool is_reduce = (repeat != ge::sym::kSymbolOne && output_repeat == ge::sym::kSymbolOne) ||
                       (stride != ge::sym::kSymbolZero && output_stride == ge::sym::kSymbolZero);
      if (is_reduce) {
        return true;
      }
    }
  }
  return false;
}
}
bool AttUtils::IsLoadNode(ge::AscNode *node) {
  GE_ASSERT_NOTNULL(node);
  if (IsOps(node, kData) || IsOps(node, kWorkspace)) {
    return false;
  }
  const auto input_size = node->inputs().size();
  std::vector<size_t> indices(input_size);
  std::iota(indices.begin(), indices.end(), 0U);
  bool is_any_input_gm = std::any_of(indices.begin(), indices.end(), [&node](size_t id) {
    const auto &input = node->inputs[id];
    return (input.attr.mem.hardware == ge::MemHardware::kMemHardwareGM);
  });
  return is_any_input_gm;
}

bool AttUtils::IsStoreNode(ge::AscNode *node) {
  GE_ASSERT_NOTNULL(node);
  if (IsOps(node, kData) || IsOps(node, kWorkspace)) {
    return false;
  }
  const auto output_size = node->outputs().size();
  std::vector<size_t> indices(output_size);
  std::iota(indices.begin(), indices.end(), 0U);
  bool is_any_output_gm = std::any_of(indices.begin(), indices.end(), [&node](size_t id) {
    const auto &output = node->outputs[id];
    return (output.attr.mem.hardware == ge::MemHardware::kMemHardwareGM);
  });
  return is_any_output_gm;
}

bool AttUtils::IsLoadStoreNode(ge::AscNode *node) {
  return IsLoadNode(node) || IsStoreNode(node);
}

bool AttUtils::IsTileSplitAxis(const AttAxisPtr &axis) {
  return (axis->axis_pos == AxisPosition::INNER) && (!axis->bind_multicore);
}

bool AttUtils::GetLastTileSplitAxisName(ge::AscNode &node, const ge::AscGraph &graph, std::string &axis_name) {
  if (node.outputs().empty()) {
    return false;
  }
  const auto &node_attr = node.outputs[0].attr;
  if (node_attr.axis.empty()) {
    return false;
  }
  const auto &last_axis_id = node_attr.axis.back();
  for (const auto &axis : graph.GetAllAxis()) {
    if (axis->id == last_axis_id) {
      axis_name = axis->name;
      return true;
    }
  }
  return false;
}

// 收集Reduce轴的原始轴名称
// 参考 arg_list_reorder.cpp 的 CheckAxisProperty 逻辑
void AttUtils::CollectReduceAxisNames(const NodeInfo &node_info,
                                      std::set<std::string> &reduce_axis_orig_names) {
  std::set<std::string> reduce_axis_names;
  GELOGD("[DFX] CollectReduceAxisNames: node_name=%s, node_type=%s",
         node_info.name.c_str(), node_info.node_type.c_str());

  for (const auto &tensor : node_info.inputs) {
    // 校验各个成员的数量是否一致
    size_t dim_size = tensor->dim_info.size();
    if (tensor->repeat.size() != dim_size || tensor->stride.size() != dim_size) {
      GELOGW("[DFX] CollectReduceAxisNames: tensor=%s has inconsistent sizes", tensor->ToString().c_str());
      continue;
    }

    for (size_t i = 0; i < dim_size; i++) {
      auto dim = tensor->dim_info[i];
      auto repeat = tensor->repeat[i];
      auto stride = tensor->stride[i];

      if (IsReduceAxis(dim, repeat, stride, node_info.outputs)) {
        // 收集该轴的所有原始轴名称
        CollectOrigAxisNames(dim, reduce_axis_orig_names);
        reduce_axis_names.insert(dim->name);
      }
    }
  }
  GELOGD("[DFX] Collected reduce_axis_name:%s, reduce_axis_orig_names: %s",
         std::accumulate(
             reduce_axis_names.begin(), reduce_axis_names.end(), std::string(),
             [](const std::string &acc, const std::string &name) { return acc.empty() ? name : acc + "," + name; })
             .c_str(),
         std::accumulate(
             reduce_axis_orig_names.begin(), reduce_axis_orig_names.end(), std::string(),
             [](const std::string &acc, const std::string &name) { return acc.empty() ? name : acc + "," + name; })
             .c_str());
}

// 辅助函数：检查单个维度是否为Broadcast轴
static bool IsBroadcastAxis(const SubAxis *dim, const Expr &repeat, const Expr &stride,
                            const std::vector<TensorPtr> &output_tensors) {
  for (const auto &output_tensor : output_tensors) {
    for (size_t j = 0; j < output_tensor->dim_info.size(); j++) {
      if (output_tensor->dim_info[j]->name != dim->name) {
        continue;
      }
      auto output_repeat = output_tensor->repeat[j];
      auto output_stride = output_tensor->stride[j];

      bool is_broadcast = (output_repeat != ge::sym::kSymbolOne && repeat == ge::sym::kSymbolOne) ||
                          (output_stride != ge::sym::kSymbolZero && stride == ge::sym::kSymbolZero);
      if (is_broadcast) {
        return true;
      }
    }
  }
  return false;
}

// 收集Broadcast轴的原始轴名称
// 参考 arg_list_reorder.cpp 的 CheckAxisProperty 逻辑
void AttUtils::CollectBroadcastAxisNames(const NodeInfo &node_info,
                                         std::set<std::string> &broadcast_axis_orig_names) {
  std::set<std::string> broadcast_axis_names;
  GELOGD("[DFX] CollectBroadcastAxisNames: node_name=%s, node_type=%s",
         node_info.name.c_str(), node_info.node_type.c_str());

  for (const auto &tensor : node_info.inputs) {
    // 校验各个成员的数量是否一致
    size_t dim_size = tensor->dim_info.size();
    if (tensor->repeat.size() != dim_size ||
        tensor->stride.size() != dim_size ||
        tensor->gm_stride.size() != dim_size) {
      GELOGW("[DFX] CollectBroadcastAxisNames: tensor=%s has inconsistent sizes", tensor->ToString().c_str());
      continue;
    }

    for (size_t i = 0; i < dim_size; i++) {
      auto dim = tensor->dim_info[i];
      auto repeat = tensor->repeat[i];
      auto stride = tensor->stride[i];
      // 待补齐，Nddma场景下还需要判断gm_stride的变化以推测是否为Broadcast轴
      auto gm_stride = tensor->gm_stride[i];
      if (IsBroadcastAxis(dim, repeat, stride, node_info.outputs)) {
        // 收集该轴的所有原始轴名称
        CollectOrigAxisNames(dim, broadcast_axis_orig_names);
        broadcast_axis_names.insert(dim->name);
      }
    }
  }
  GELOGD("[DFX] Collected broadcast_axis_name:%s, broadcast_axis_orig_names: %s",
         std::accumulate(
             broadcast_axis_names.begin(), broadcast_axis_names.end(), std::string(),
             [](const std::string &acc, const std::string &name) { return acc.empty() ? name : acc + "," + name; })
             .c_str(),
         std::accumulate(
             broadcast_axis_orig_names.begin(), broadcast_axis_orig_names.end(), std::string(),
             [](const std::string &acc, const std::string &name) { return acc.empty() ? name : acc + "," + name; })
             .c_str());
}
}  // namespace att
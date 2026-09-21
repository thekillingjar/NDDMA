/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "asc_backend_torch_fusion_decider.h"
#include "common/checker.h"
#include "fusion_decider_registry.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "utils/autofuse_attrs.h"
#include "can_fuse/strategy/fusion_strategy_registry.h"
#include "utils/autofuse_utils.h"
#include "graph/utils/type_utils.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "utils/auto_fuse_config.h"
#include "post_process/post_process_util.h"
#include "post_process/scheduler_adapter/torch_adaption_fallback_load.h"
#include "post_process/scheduler_adapter/adaption_complete_node_attrs.h"

namespace ge {
namespace {
bool IsProduct(const ge::Expression &target, const std::deque<ge::Expression> &vec, uint32_t &size) {
  ge::Expression product = kSymbolOne;
  for (size_t i = 0U; i < vec.size(); ++i) {
    product = product * vec[i];
    if (BackendUtils::IsEq(target, product)) {
      size = ++i;
      return true;
    }
  }
  return false;
}

/*
根据两个graph的轴大小找到拆封轴和映射关系
图1：std::vector<int> graph1 = {6, 4, 5}; // s0 * s1, s2, s3
图2：std::vector<int> graph2 = {2, 3, 20}; // s0, s1, s2 * s3
输出：base: 2 3 4 5
输出：node1_map:
 6: 0 1
 4: 2
 5: 3
输出：node2_map:
 2: 0
 3: 1
 20: 2 3
*/
bool FindSplitAxisMap(const std::vector<ge::Expression> &node1_size, const std::vector<int64_t> &node1_axis,
                      const std::vector<ge::Expression> &node2_size, const std::vector<int64_t> &node2_axis,
                      AxisMapInfo &map_info) {
  std::deque<ge::Expression> node1_deque(node1_size.begin(), node1_size.end());
  std::deque<ge::Expression> node2_deque(node2_size.begin(), node2_size.end());
  uint32_t node1_index = 0U;
  uint32_t node2_index = 0U;
  uint32_t base_index = 0U;

  auto process_deque = [&map_info, &base_index](
                           std::deque<ge::Expression> &source_deque, std::deque<ge::Expression> &target_deque,
                           const std::vector<int64_t> &source_axis, const std::vector<int64_t> &target_axis,
                           const std::vector<AxisPtr> &source_node_axis, const std::vector<AxisPtr> &target_node_axis,
                           std::unordered_map<int64_t, std::vector<int64_t>> &source_map, uint32_t &source_index,
                           std::unordered_map<int64_t, std::vector<int64_t>> &target_map,
                           uint32_t &target_index) -> bool {
    while (!source_deque.empty()) {
      uint32_t size = 0U;
      if (target_deque.empty()) {
        map_info.base.push_back(source_deque.front());
        map_info.base_axis.push_back(source_node_axis[source_index]);
        source_map[source_axis[source_index]].push_back(base_index++);
        source_index++;
        source_deque.pop_front();
        continue;
      }
      if (IsProduct(source_deque.front(), target_deque, size)) {
        for (uint32_t j = 0U; j < size; ++j) {
          map_info.base.push_back(target_deque[j]);
          map_info.base_axis.push_back(target_node_axis[target_index]);
          target_map[target_axis[target_index]].push_back(base_index);
          source_map[source_axis[source_index]].push_back(base_index);
          base_index++;
          target_index++;
        }
        source_index++;
        target_deque.erase(target_deque.begin(), target_deque.begin() + size);
        source_deque.pop_front();
        return true;
      } else {
        return false;
      }
    }
    return true;
  };

  while (!node1_deque.empty() || !node2_deque.empty()) {
    bool found = false;
    found = process_deque(node1_deque, node2_deque, node1_axis, node2_axis, map_info.node1_axis, map_info.node2_axis,
                          map_info.node1_map, node1_index, map_info.node2_map, node2_index);
    if (found && !node1_deque.empty()) {
      continue;
    }

    found = process_deque(node2_deque, node1_deque, node2_axis, node1_axis, map_info.node2_axis, map_info.node1_axis,
                          map_info.node2_map, node2_index, map_info.node1_map, node1_index);
    if (!found) {
      return false;
    }
  }
  return true;
}

Status UpdateSchedAxis(const std::unordered_map<int64_t, std::vector<int64_t>> &node_map, std::vector<int64_t> &axis) {
  std::vector<int64_t> temp;
  for (const auto &key : axis) {
    auto iter = node_map.find(key);
    if (iter != node_map.end()) {
      temp.insert(temp.end(), iter->second.begin(), iter->second.end());
    } else {
      return FAILED;
    }
  }
  axis = temp;
  return SUCCESS;
}

Status UpdateTensorRepeats(const std::unordered_map<int64_t, std::vector<int64_t>> &node_map,
                           const std::vector<int64_t> &axis, const std::vector<ge::Expression> &base,
                           std::vector<ge::Expression> &repeats) {
  std::vector<ge::Expression> temp;
  auto origin_index = 0U;
  for (const auto &key : axis) {
    auto iter = node_map.find(key);
    if (iter != node_map.end()) {
      // 如果只有一个映射关系，说明不是拆轴，这个时候使用元素轴信息，reduce场景下base信息size是1，不是真正的轴大小
      if (iter->second.size() == 1U) {
        temp.push_back(repeats[origin_index]);
        origin_index++;
        continue;
      }
      for (const auto &index : iter->second) {
        GE_ASSERT_TRUE(static_cast<size_t>(index) < base.size());
        temp.push_back(base[static_cast<size_t>(index)]);
      }
    } else {
      return FAILED;
    }
    origin_index++;
  }
  repeats = temp;
  return SUCCESS;
}

// 根据 node_map 和 axis 的映射关系，将 strides 向量中的每个步长表达式拆分为多个子表达式，并更新 strides
Status UpdateTensorStrides(const std::unordered_map<int64_t, std::vector<int64_t>> &node_map,
                           const std::vector<int64_t> &axis, const std::vector<ge::Expression> &base,
                           std::vector<ge::Expression> &strides) {
  auto stride_index = 0U;
  std::vector<ge::Expression> temp;
  for (const auto &key : axis) {
    auto iter = node_map.find(key);
    if (iter != node_map.end()) {
      if (iter->second.size() == 1U) {
        temp.push_back(strides[stride_index]);
        stride_index++;
        continue;
      }
      GE_ASSERT_TRUE(iter->second.size() > 0U);
      std::vector<ge::Expression> split_stride;
      ge::Expression cur_stride = strides[stride_index];
      // 根据stride计算规则，需要反序计算
      for (auto index = 0U; index < iter->second.size(); index++) {
        auto sub_index = iter->second.size() - 1U - index;
        GE_ASSERT_TRUE(static_cast<size_t>(iter->second[sub_index]) < base.size());
        if (BackendUtils::IsEqZero(base[static_cast<size_t>(iter->second[sub_index])])) {
          split_stride.push_back(kSymbolZero);
        } else {
          split_stride.push_back(cur_stride);
          cur_stride = cur_stride * base[static_cast<size_t>(iter->second[sub_index])];
        }
      }
      std::reverse(split_stride.begin(), split_stride.end());
      temp.insert(temp.end(), split_stride.begin(), split_stride.end());
      stride_index++;
    } else {
      return FAILED;
    }
  }
  strides = temp;
  return SUCCESS;
}

Status UpdateGraphAxis(const std::unordered_map<int64_t, std::vector<int64_t>> &node_map,
                       const std::vector<ge::Expression> &base, const std::vector<AxisPtr> &base_axis,
                       std::vector<AxisPtr> &axis) {
  std::vector<AxisPtr> temp;
  for (const auto &key : axis) {
    auto iter = node_map.find(key->id);
    if (iter != node_map.end()) {
      if (iter->second.size() == 1U) {
        auto graph_axis = ComGraphMakeShared<Axis>();
        GE_ASSERT_NOTNULL(graph_axis);
        *graph_axis = *key;
        graph_axis->id = iter->second[0];
        temp.push_back(graph_axis);
        continue;
      }
      for (const auto index : iter->second) {
        GE_ASSERT_TRUE(static_cast<size_t>(index) < base.size());
        auto graph_axis = ComGraphMakeShared<Axis>();
        GE_ASSERT_NOTNULL(graph_axis);
        // 轴的信息还是需要继承的，比如轴名
        *graph_axis = *base_axis[index];
        graph_axis->id = index;
        graph_axis->size = base[static_cast<size_t>(index)];
        temp.push_back(graph_axis);
      }
    } else {
      return FAILED;
    }
  }
  axis = temp;
  return SUCCESS;
}

Status GetGraphAxis(const NodePtr &node, const std::vector<int64_t> &axis, std::vector<AxisPtr> &node_axis) {
  ComputeGraphPtr graph;
  GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(node, graph));
  auto graph_attr = graph->GetAttrsGroup<AscGraphAttr>();
  GE_ASSERT_NOTNULL(graph_attr);

  for (const auto axis_id : axis) {
    for (auto &axis_info : graph_attr->axis) {
      if (axis_id == axis_info->id) {
        node_axis.push_back(axis_info);
      }
    }
  }
  GE_ASSERT_TRUE(node_axis.size() == axis.size());
  return SUCCESS;
}

// 根据轴映射关系，尝试拆轴
Status SplitSubGraphAxisInfo(const ComputeGraphPtr &graph,
                             const std::unordered_map<int64_t, std::vector<int64_t>> &node_map,
                             const std::vector<ge::Expression> &base, const std::vector<AxisPtr> &base_axis) {
  for (auto &asc_node : graph->GetAllNodes()) {
    GE_ASSERT_NOTNULL(asc_node);
    if (asc_node->GetType() == kScalarType) {
      continue;
    }
    auto asc_node_op_desc = asc_node->GetOpDesc();
    GE_ASSERT_NOTNULL(asc_node_op_desc);
    AscNodeAttr *asc_node_attr = asc_node_op_desc->GetAttrsGroup<AscNodeAttr>();
    GE_ASSERT_NOTNULL(asc_node_attr);
    GE_ASSERT_SUCCESS(UpdateSchedAxis(node_map, asc_node_attr->sched.axis));

    for (auto &output_desc : asc_node_op_desc->GetAllOutputsDescPtr()) {
      GE_ASSERT_NOTNULL(output_desc);
      auto output_desc_tensor_attr = output_desc->GetAttrsGroup<AscTensorAttr>();
      GE_ASSERT_NOTNULL(output_desc_tensor_attr);
      GE_ASSERT_TRUE(output_desc_tensor_attr->axis.size() == output_desc_tensor_attr->repeats.size());
      GE_ASSERT_TRUE(output_desc_tensor_attr->axis.size() == output_desc_tensor_attr->strides.size());
      GE_ASSERT_SUCCESS(
          UpdateTensorRepeats(node_map, output_desc_tensor_attr->axis, base, output_desc_tensor_attr->repeats));
      GE_ASSERT_SUCCESS(
          UpdateTensorStrides(node_map, output_desc_tensor_attr->axis, base, output_desc_tensor_attr->strides));
      GE_ASSERT_SUCCESS(UpdateSchedAxis(node_map, output_desc_tensor_attr->axis));
    }
  }
  auto graph_attr = graph->GetAttrsGroup<AscGraphAttr>();
  GE_ASSERT_NOTNULL(graph_attr);
  GE_ASSERT_SUCCESS(UpdateGraphAxis(node_map, base, base_axis, graph_attr->axis));
  for (auto &axis_info : graph_attr->axis) {
    GELOGD("after split axis info: axis name(%s), axis id(%ld), axis size(%s), graph name(%s).",
           axis_info->name.c_str(), axis_info->id, std::string(axis_info->size.Str().get()).c_str(),
           graph->GetName().c_str());
  }
  return SUCCESS;
}

Status GetPreNodeAttrs(const NodePtr &node, const int32_t index, std::vector<int64_t> &axis,
                       std::vector<ge::Expression> &repeats) {
  NodePtr peer_node;
  InDataAnchorPtr in_anchor;
  GE_ASSERT_SUCCESS(BackendUtils::GetPreNodeAndAnchor(node, index, peer_node, in_anchor));

  if (peer_node->GetType() == kAscBackendType) {
    ComputeGraphPtr graph;
    GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(peer_node, graph));
    GE_ASSERT_SUCCESS(BackendUtils::UpdateSubgraphOutputAttr(graph, peer_node));
    GE_ASSERT_SUCCESS(BackendUtils::GetPreAscNodeAttrs(peer_node, in_anchor, axis, repeats));
  } else {
    // nothing, 共同输入是普通ge node，没有axis和repeats属性
  }
  return SUCCESS;
}

Status GetCurAscNodeAttrs(const NodePtr &parent_node, const int32_t index, std::vector<int64_t> &axis,
                          std::vector<ge::Expression> &repeats) {
  ComputeGraphPtr graph;
  GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(parent_node, graph));
  const auto subgraph_input_nodes = graph->GetInputNodes();
  GE_ASSERT_TRUE(subgraph_input_nodes.size() > static_cast<size_t>(index));
  auto data_node = subgraph_input_nodes.at(index);
  GE_ASSERT_NOTNULL(data_node);
  GELOGD("get cur node %s(%s) index=%d.", data_node->GetNamePtr(), data_node->GetType().c_str(), index);
  const auto load_node = BackendUtils::GetDataNextNode(data_node);
  GE_ASSERT_NOTNULL(load_node);
  auto node_op_desc = load_node->GetOpDesc();
  GE_ASSERT_NOTNULL(node_op_desc);
  auto node_output_desc = node_op_desc->MutableOutputDesc(0);
  GE_ASSERT_NOTNULL(node_output_desc);
  auto node_tensor_attr = node_output_desc->GetAttrsGroup<AscTensorAttr>();
  GE_ASSERT_NOTNULL(node_tensor_attr);
  repeats = node_tensor_attr->repeats;
  axis = node_tensor_attr->axis;
  GE_ASSERT_TRUE(axis.size() == repeats.size());
  return SUCCESS;
}

Status FlashSubgraphAxis(const NodePtr &node1, const NodePtr &node2, const NodeFuseInfo &fuse_info,
                         AxisMapInfo &map_info) {
  ComputeGraphPtr compute_graph1;
  ComputeGraphPtr compute_graph2;
  GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(node1, compute_graph1));
  GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(node2, compute_graph2));
  GE_ASSERT_SUCCESS(SplitSubGraphAxisInfo(compute_graph1, map_info.node1_map, map_info.base, map_info.base_axis));
  GE_ASSERT_SUCCESS(SplitSubGraphAxisInfo(compute_graph2, map_info.node2_map, map_info.base, map_info.base_axis));
  GE_ASSERT_SUCCESS(BackendUtils::TuningSubgraphBeforeMerge(node1, node2, compute_graph1, compute_graph2, fuse_info));
  return SUCCESS;
}

Status GroupMergeCheck(const NodePtr &node1, const NodePtr &node2, const AscGraph &graph1, const AscGraph &graph2) {
  // 判断group merge是否成功
  optimize::autoschedule::AxisGroup axes_group1;
  if (GenAscGraphAxisGroup(graph1, axes_group1) != SUCCESS) {
    GELOGD("node %s(%s) convert group axis failed, can fuse false", node1->GetNamePtr(), node1->GetType().c_str());
    return FAILED;
  }
  optimize::autoschedule::AxisGroup axes_group2;
  if (GenAscGraphAxisGroup(graph2, axes_group2) != SUCCESS) {
    GELOGD("node %s(%s) convert group axis failed, can fuse false", node2->GetNamePtr(), node2->GetType().c_str());
    return FAILED;
  }
  optimize::autoschedule::AxisGroup merged_axes_group;
  if (!BackendUtils::IsCanMergeAxisGroup(axes_group1, axes_group2, merged_axes_group)) {
    GELOGD("node %s(%s) and node %s(%s) axis group merge failed, can fuse false", node1->GetNamePtr(),
           node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status TryMergeSubgraph(const NodePtr &node1, const NodePtr &node2, const NodeFuseInfo &fuse_info,
                        AxisMapInfo &map_info) {
  // 在canfuse阶段，需要临时拷贝一份ascgraph用来做graph merge，因为不确定是否真的能融合
  auto asc_graph1 = BackendUtils::GetNodeFusedAscGraph(node1);
  GE_ASSERT_NOTNULL(asc_graph1);
  AscGraph temp1_graph(asc_graph1->GetName().c_str());
  temp1_graph.CopyFrom(*asc_graph1);
  auto asc_graph2 = BackendUtils::GetNodeFusedAscGraph(node2);
  GE_ASSERT_NOTNULL(asc_graph2);
  AscGraph temp2_graph(asc_graph2->GetName().c_str());
  temp2_graph.CopyFrom(*asc_graph2);

  const auto compute_graph1 = AscGraphUtils::GetComputeGraph(temp1_graph);
  const auto compute_graph2 = AscGraphUtils::GetComputeGraph(temp2_graph);
  GE_ASSERT_SUCCESS(SplitSubGraphAxisInfo(compute_graph1, map_info.node1_map, map_info.base, map_info.base_axis));
  GE_ASSERT_SUCCESS(SplitSubGraphAxisInfo(compute_graph2, map_info.node2_map, map_info.base, map_info.base_axis));

  auto get_graph_axis = [](const std::vector<AxisPtr> &axis_info_list) -> std::vector<int64_t> {
    std::vector<int64_t> axes;
    for (auto &axis_info : axis_info_list) {
      axes.push_back(axis_info->id);
    }
    return axes;
  };
  auto graph_attr1 = compute_graph1->GetAttrsGroup<AscGraphAttr>();
  GE_ASSERT_NOTNULL(graph_attr1);
  auto graph_attr2 = compute_graph2->GetAttrsGroup<AscGraphAttr>();
  GE_ASSERT_NOTNULL(graph_attr2);
  std::vector<int64_t> axis1 = get_graph_axis(graph_attr1->axis);
  std::vector<int64_t> axis2 = get_graph_axis(graph_attr2->axis);

  GELOGD("get graph axis info, node %s(%s) axis(%s), node %s(%s) axis(%s).", node1->GetNamePtr(),
         node1->GetType().c_str(), AutofuseUtils::VectorToStr(axis1).c_str(), node2->GetNamePtr(),
         node2->GetType().c_str(), AutofuseUtils::VectorToStr(axis2).c_str());

  if (axis1 != axis2) {
    // 判断轴是否是顺序子集关系，子集关系认为也是可以循环合并的，后期schedue adapter补轴实现
    if (!BackendUtils::CheckAxisSubsetRelation(axis1, axis2)) {
      GELOGI("sched axis diffrent and not subset relation, try tuning subgraph transpose axis.");
      // 如果不是循序子集尝试做transpose轴变换
      if (BackendUtils::TuningSubgraphBeforeMerge(node1, node2, compute_graph1, compute_graph2, fuse_info) != SUCCESS) {
        GELOGI("tuning subgraph transpose axis failed, can fuse false.");
        return FAILED;
      }
    }
  }

  // 判断group merge是否成功
  if (GroupMergeCheck(node1, node2, temp1_graph, temp2_graph) != SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}
}  // namespace

Status AscBackendTorchFusionDecider::UnifySubgraphAxis(const NodePtr &node1, const NodePtr &node2,
                                                       const NodeFuseInfo &fuse_info, AxisMapInfo &map_info) const {
  std::vector<int64_t> node1_axis;
  std::vector<ge::Expression> node1_size;
  std::vector<int64_t> node2_axis;
  std::vector<ge::Expression> node2_size;

  // 映射只需要选择一个链接关系，优先垂直
  if (!fuse_info.GetNode1ToNode2LinkMap().empty()) {
    const auto &subgraph_link = fuse_info.GetNode1ToNode2LinkMap().front();
    GE_ASSERT_SUCCESS(GetPreNodeAttrs(node2, subgraph_link.second, node1_axis, node1_size));
    GELOGD("get node %s(%s) pre store info: index=%d, repeats=%s, axis=%s.", node2->GetNamePtr(),
            node2->GetType().c_str(), subgraph_link.second, AutofuseUtils::VectorToStr(node1_size).c_str(),
            AutofuseUtils::VectorToStr(node1_axis).c_str());
    GE_ASSERT_SUCCESS(GetGraphAxis(node1, node1_axis, map_info.node1_axis));
    GE_ASSERT_SUCCESS(GetCurAscNodeAttrs(node2, subgraph_link.second, node2_axis, node2_size));
    GELOGD("get node %s(%s) cur load info: index=%d, repeats=%s, axis=%s.", node2->GetNamePtr(),
            node2->GetType().c_str(), subgraph_link.second, AutofuseUtils::VectorToStr(node2_size).c_str(),
            AutofuseUtils::VectorToStr(node2_axis).c_str());
    GE_ASSERT_SUCCESS(GetGraphAxis(node2, node2_axis, map_info.node2_axis));
  } else {
    if (!fuse_info.GetSameInputMap().empty()) {
      const auto &same_input = fuse_info.GetSameInputMap().front();
      GE_ASSERT_SUCCESS(GetCurAscNodeAttrs(node1, same_input.first, node1_axis, node1_size));
      GELOGD("get node %s(%s) cur load info: index=%d, repeats=%s, axis=%s.", node1->GetNamePtr(),
             node1->GetType().c_str(), same_input.first, AutofuseUtils::VectorToStr(node1_size).c_str(),
             AutofuseUtils::VectorToStr(node1_axis).c_str());
      GE_ASSERT_SUCCESS(GetGraphAxis(node1, node1_axis, map_info.node1_axis));
      GE_ASSERT_SUCCESS(GetCurAscNodeAttrs(node2, same_input.second, node2_axis, node2_size));
      GELOGD("get node %s(%s) cur load info: index=%d, repeats=%s, axis=%s.", node2->GetNamePtr(),
             node2->GetType().c_str(), same_input.second, AutofuseUtils::VectorToStr(node2_size).c_str(),
             AutofuseUtils::VectorToStr(node2_axis).c_str());
      GE_ASSERT_SUCCESS(GetGraphAxis(node2, node2_axis, map_info.node2_axis));
    }
  }

  if (!FindSplitAxisMap(node1_size, node1_axis, node2_size, node2_axis, map_info)) {
    GELOGD("node %s(%s) and node %s(%s) can't align axis info, can fuse false.", node1->GetNamePtr(),
           node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str());
    return FAILED;
  }
  GE_ASSERT_TRUE(map_info.base_axis.size() == map_info.base.size());
  GELOGD("node %s(%s) and node %s(%s) unify succ, axis map:%s.", node1->GetNamePtr(), node1->GetType().c_str(),
         node2->GetNamePtr(), node2->GetType().c_str(), map_info.ToString().c_str());
  return SUCCESS;
}

bool AscBackendTorchFusionDecider::CheckFusionStrategy(const NodePtr &node1, const NodePtr &node2) const {
  ComputeGraphPtr graph1;
  ComputeGraphPtr graph2;
  thread_local static std::set<std::string> dumped_nodes;  // 存放已经dump过的node，如果node已经dump过了，就不再重复dump这个node的图
  GELOGI("AscBackendGraphFuse:can fuse check, node: %s(%s) and node: %s(%s).", node1->GetNamePtr(),
         node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str());

  if (BackendUtils::IsCanFuseBlackList(node1) || BackendUtils::IsCanFuseBlackList(node2)) {
    return false;
  }
  if (!BackendUtils::IsBackendFuseNode(node1) || !BackendUtils::IsBackendFuseNode(node2)) {
    return false;
  }

  const auto fuse_type = BackendUtils::GetAllFuseType(node1, node2);
  for (const auto fusion_strategy : FusionStrategyRegistry::Instance().Get(fuse_type)) {
    if (fusion_strategy != nullptr) {
      if (!fusion_strategy->CanFuse(node1, node2)) {
        return false;
      }
    }
  }

  if (dumped_nodes.find(node1->GetName()) == dumped_nodes.end()) {
    GELOGD("dump node:%s(%s) asc subgraph info:", node1->GetNamePtr(), node1->GetType().c_str());
    BackendUtils::DumpAscGraph(node1);
    dumped_nodes.insert(node1->GetName());
  }

  if (dumped_nodes.find(node2->GetName()) == dumped_nodes.end()) {
    GELOGD("dump node:%s(%s) asc subgraph info:", node2->GetNamePtr(), node2->GetType().c_str());
    BackendUtils::DumpAscGraph(node2);
    dumped_nodes.insert(node2->GetName());
  }

  if (!BackendUtils::CanMergeLoopByStrategy(node1, node2)) {
    GELOGI("AscBackendGraphFuse: check AscBackend graph fuse, node: %s(%s) and node %s(%s) can't loop merge.",
           node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str());
    return false;
  }

  GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(node1, graph1));
  GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(node2, graph2));
  GE_ASSERT_SUCCESS(BackendUtils::UpdateSubgraphOutputAttr(graph1, node1));
  GE_ASSERT_SUCCESS(BackendUtils::UpdateSubgraphOutputAttr(graph2, node2));
  return true;
}

bool AscBackendTorchFusionDecider::CanFuse(const NodePtr &node1, const NodePtr &node2) const {
  // 检查基础融合策略
  if (!CheckFusionStrategy(node1, node2)) {
    return false;
  }
  // 尝试做轴映射，映射成功需要刷新ascgraph，如果无法映射融合失败
  NodeFuseInfo node_fuse_info(false);
  GE_ASSERT_SUCCESS(node_fuse_info.UpdateNodeFuseInfo(node1, node2));
  AxisMapInfo map_info;
  if (UnifySubgraphAxis(node1, node2, node_fuse_info, map_info) != SUCCESS) {
    GELOGD("node %s(%s) and node %s(%s) can't unify subgraph axis, can fuse false.", node1->GetNamePtr(),
           node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str());
    return false;
  }

  if (TryMergeSubgraph(node1, node2, node_fuse_info, map_info) != SUCCESS) {
    GELOGD("node %s(%s) and node %s(%s) can't merge axis, can fuse false.", node1->GetNamePtr(),
           node1->GetType().c_str(), node2->GetNamePtr(), node2->GetType().c_str());
    return false;
  }

  GELOGI("AscBackendGraphFuse:node: %s(%s) and node %s(%s) can fuse.", node1->GetNamePtr(), node1->GetType().c_str(),
         node2->GetNamePtr(), node2->GetType().c_str());
  return true;
}

NodePtr AscBackendTorchFusionDecider::Fuse(const NodePtr &node1, const NodePtr &node2, const CounterPtr &counter) {
  ComputeGraphPtr graph1;
  ComputeGraphPtr graph2;
  NodeFuseInfo node_fuse_info;
  GE_ASSERT_SUCCESS(node_fuse_info.UpdateNodeFuseInfo(node1, node2));
  const auto &orgin_graph = node1->GetOwnerComputeGraph();
  GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(node1, graph1));
  GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(node2, graph2));
  GELOGI("AscBackendGraphFuse: before fuse AscBackend graph: %s, fuse node: %s(%s) and %s(%s).",
         orgin_graph->GetName().c_str(), node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(),
         node2->GetType().c_str());
  GE_ASSERT_SUCCESS(BackendUtils::UpdateSubgraphOutputAttr(graph1, node1));
  GE_ASSERT_SUCCESS(BackendUtils::UpdateSubgraphOutputAttr(graph2, node2));
  AxisMapInfo map_info;
  GE_ASSERT_SUCCESS(UnifySubgraphAxis(node1, node2, node_fuse_info, map_info));
  GE_ASSERT_SUCCESS(FlashSubgraphAxis(node1, node2, node_fuse_info, map_info));
  GELOGD("dump node:%s(%s) asc subgraph split info:", node1->GetNamePtr(), node1->GetType().c_str());
  BackendUtils::DumpAscGraph(node1);
  GELOGD("dump node:%s(%s) asc subgraph split info:", node2->GetNamePtr(), node2->GetType().c_str());
  BackendUtils::DumpAscGraph(node2);
  const auto merged_graph = MergeAscGraphByLoop(graph1, graph2, node1, node2, node_fuse_info);
  GE_ASSERT_NOTNULL(merged_graph);
  auto new_node = FuseNode(node1, node2, merged_graph, node_fuse_info, counter);
  GE_ASSERT_NOTNULL(new_node);
  GELOGI("AscBackendGraphFuse: merged asc graph: %s, parent node: %s(%s).", merged_graph->GetName().c_str(),
         new_node->GetNamePtr(), new_node->GetType().c_str());
  GELOGD("dump node:%s(%s) asc subgraph merged info:", new_node->GetNamePtr(), new_node->GetType().c_str());
  BackendUtils::DumpAscGraph(new_node);

  GELOGI("AscBackendGraphFuse: after fuse AscBackend graph: %s, fuse node: %s(%s) and %s(%s).",
         orgin_graph->GetName().c_str(), node1->GetNamePtr(), node1->GetType().c_str(), node2->GetNamePtr(),
         node2->GetType().c_str());
  GE_ASSERT_SUCCESS(BackendUtils::UpdateSubgraphOutputAttr(merged_graph, new_node));
  GE_ASSERT_SUCCESS(UpdateSubgraphAxisAttr(new_node, node1, node2));

  // merge后做反推和补轴
  GE_ASSERT_SUCCESS(BackendUtils::ProcessAscgraphAfterMerge(new_node));
  return new_node;
}

REGISTER_FUSION_DECIDER(AscBackendTorchFusionDecider, AutoFuseFwkType::kTorch);
}  // namespace ge

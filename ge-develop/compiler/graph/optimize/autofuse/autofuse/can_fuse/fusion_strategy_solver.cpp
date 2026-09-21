/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fusion_strategy_solver.h"
#include "utils/auto_fuse_config.h"

#include "common/util/mem_utils.h"
#include "common/checker.h"
#include "common/ge_common/ge_types.h"
#include "attribute_group/attr_group_symbolic_desc.h"
#include "can_fuse/backend/fusion_decider_registry.h"
#include "common/scope_tracing_recorder.h"
#include "graph/utils/graph_utils.h"
#include "utils/autofuse_attrs.h"
#include "graph/ascendc_ir/utils/asc_graph_utils.h"
#include "can_fuse/backend/backend_utils.h"
#include "utils/autofuse_utils.h"
#include "graph/symbolizer/symbolic_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "post_process/post_process_util.h"
#include "utils/not_fuse_reason_code.h"

namespace {
const ge::Expression kSymbolZero = ge::Symbol(0);
const int64_t kInvalidId = -1;

ge::Expression GetMemorySize(const ge::DataType data_type, const ge::Expression &symbol_shape_size) {
  uint32_t data_type_length = 1U;
  (void) ge::TypeUtils::GetDataTypeLength(data_type, data_type_length);
  data_type_length = (data_type_length > static_cast<uint32_t>(ge::kDataTypeSizeBitOffset))
      ? (data_type_length - static_cast<uint32_t>(ge::kDataTypeSizeBitOffset)) : data_type_length;
  return symbol_shape_size * ge::Symbol(data_type_length);
}

ge::Status GetReadsByNode(const ge::NodePtr &node, std::set<ge::MemoryBuffer> &reads) {
  for (const auto in_anchor : node->GetAllInDataAnchorsPtr()) {
    GE_ASSERT_NOTNULL(in_anchor);
    const auto &peer_out_anchor = in_anchor->GetPeerOutAnchor();
    if (peer_out_anchor == nullptr) {
      GELOGW("Node:%s in_anchor:%u peer_out_anchor is nullptr.", node->GetNamePtr(), in_anchor->GetIdx());
      continue;
    }
    const auto peer_node = peer_out_anchor->GetOwnerNodeBarePtr();
    GE_ASSERT_NOTNULL(peer_node);
    const auto peer_node_desc = peer_node->GetOpDesc();
    GE_ASSERT_NOTNULL(peer_node_desc);
    const auto output_tensor_desc = peer_node_desc->MutableOutputDesc(peer_out_anchor->GetIdx());
    GE_ASSERT_NOTNULL(output_tensor_desc);
    const auto attr_group = output_tensor_desc->GetAttrsGroup<ge::SymbolicDescAttr>();
    if(attr_group == nullptr) {
      continue;
    }
    reads.insert({peer_out_anchor.get(), GetMemorySize(output_tensor_desc->GetDataType(),
        attr_group->symbolic_tensor.GetOriginSymbolShape().GetSymbolShapeSize())});
  }
  return ge::SUCCESS;
}

ge::Status GetWritesByNode(const ge::NodePtr &node, std::set<ge::MemoryBuffer> &writes) {
  for (const auto out_anchor : node->GetAllOutDataAnchorsPtr()) {
    GE_ASSERT_NOTNULL(out_anchor);
    const auto node_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(node_desc);
    auto output_tensor_desc = node_desc->MutableOutputDesc(out_anchor->GetIdx());
    GE_ASSERT_NOTNULL(output_tensor_desc);
    const auto attr_group = output_tensor_desc->GetAttrsGroup<ge::SymbolicDescAttr>();
    if (attr_group == nullptr) {
      continue;
    }
    writes.insert({out_anchor, GetMemorySize(output_tensor_desc->GetDataType(),
        attr_group->symbolic_tensor.GetOriginSymbolShape().GetSymbolShapeSize())});
  }
  return ge::SUCCESS;
}

ge::FusionStatistics GetFusionStatisticsByGraph(const ge::ComputeGraphPtr &graph) {
  ge::FusionStatistics fusion_statistics;
  if (!IsLogEnable(GE_MODULE_NAME, DLOG_INFO)) {
    return fusion_statistics;
  }
  fusion_statistics.rw_memory_size = kSymbolZero;
  fusion_statistics.nodes_size = graph->GetAllNodesSize();
  fusion_statistics.min_scale = std::numeric_limits<size_t>::max();
  fusion_statistics.max_scale = 1U;
  fusion_statistics.average_scale = 1.0;
  size_t asc_node_count = 0U;
  size_t asc_node_size = 0U;
  for (const auto &node: graph->GetAllNodes()) {
    std::set<ge::MemoryBuffer> read_writes;
    (void) GetReadsByNode(node, read_writes);
    (void) GetWritesByNode(node, read_writes);
    for (const auto &memory_buffer : read_writes) {
      fusion_statistics.rw_memory_size = fusion_statistics.rw_memory_size + memory_buffer.size;
    }
    if ((node->GetType() == ge::kAscBackendType) || (node->GetType() == ge::kFusedAscBackendType)) {
      const auto attr = node->GetOpDesc()->GetAttrsGroup<ge::AutoFuseAttrs>();
      size_t node_size = 1U;
      if ((attr != nullptr) && (attr->GetAscGraph() != nullptr)) {
        node_size = ge::AscGraphUtils::GetComputeGraph(*(attr->GetAscGraph()))->GetAllNodesSize();
      }
      fusion_statistics.min_scale = std::min(node_size, fusion_statistics.min_scale);
      fusion_statistics.max_scale = std::max(node_size, fusion_statistics.max_scale);
      asc_node_count++;
      asc_node_size += node_size;
    }
  }
  if (asc_node_count > 0U) {
    fusion_statistics.average_scale = static_cast<float_t>(asc_node_size) / static_cast<float_t>(asc_node_count);
  }
  if (fusion_statistics.min_scale == std::numeric_limits<size_t>::max()) {
    fusion_statistics.min_scale = 1U;
  }
  return fusion_statistics;
}

void PrintFusionStatistics(const ge::FusionStatistics &before_fusion, const ge::FusionStatistics &after_fusion) {
  GELOGI("before: nodes size:%zu, rw memory size:%s", before_fusion.nodes_size,
         before_fusion.rw_memory_size.Str().get());
  GELOGI("after : nodes size:%zu, rw memory size:%s", after_fusion.nodes_size,
         after_fusion.rw_memory_size.Str().get());
  GELOGI("reduce: nodes size:%zu, rw memory size:%s", before_fusion.nodes_size - after_fusion.nodes_size,
         (before_fusion.rw_memory_size - after_fusion.rw_memory_size).Str().get());
  GELOGI("AscBackend scale:min %zu, max:%zu, average:%.1f", after_fusion.min_scale, after_fusion.max_scale,
         after_fusion.average_scale);
}

// 检查融合后是否存在输入或输出是自己的，这个topo会认为成环
ge::Status CheckCycle(const ge::NodePtr &node) {
  for (const auto in_node : node->GetInNodesPtr()) {
    if (in_node == node.get()) {
      GELOGE(ge::FAILED, "Has cycle node:%s input:%s is self.", node->GetNamePtr(), in_node->GetNamePtr());
      return ge::FAILED;
    }
  }
  for (const auto out_node : node->GetOutNodesPtr()) {
    if (out_node == node.get()) {
      GELOGE(ge::FAILED, "Has cycle node:%s output:%s is self.", node->GetNamePtr(), out_node->GetNamePtr());
      return ge::FAILED;
    }
  }
  return ge::SUCCESS;
}
}

namespace ge {
using namespace autofuse;
// 内存大小符号比较结果不稳定，这里用name保证顺序稳定
static bool CompareBuffer(const MemoryBuffer *const left, const MemoryBuffer *const right) {
  GE_ASSERT_NOTNULL(left);
  GE_ASSERT_NOTNULL(right);
  // node相同用index比较
  if (left->node == right->node) {
    return left->index < right->index;
  }
  // node不同用名字比较
  return left->hash_name < right->hash_name;
}

// 融合异常流程dump正在融合的dump图和前序融合缓存的dump图
Status DumpCacheGraphForExceptionMerging() {
  if (!IsLogEnable(GE_MODULE_NAME, DLOG_DEBUG)) {
    return SUCCESS;
  }

  GE_ASSERT_SUCCESS(BackendUtils::DumpCurrentGraphAndSubgraphs());
  return SUCCESS;
}

bool MemoryBuffer::operator<(const MemoryBuffer &other) const {
  if (buffer == other.buffer) {
    return false;
  }
  return CompareBuffer(this, &other);
}

FusingNode::FusingNode(const NodePtr &node)
    : min_order_(0),
      max_order_(0),
      id_(0),
      node_(node),
      fusion_nodes_size_(0U) {
}

void FusingNode::UpdateOrder() {
  // 重新拓扑排序后FusingNode的id和对应Node的id保持一致
  id_ = node_->GetOpDesc()->GetId();
  min_order_ = GetId();
  max_order_ = min_order_;
  GELOGD("%s id:%ld.", GetNamePtr(), GetId());
}

Status FusingNode::Init() {
  GE_ASSERT_SUCCESS(GetReadsByNode(node_, read_writes_));
  GE_ASSERT_SUCCESS(GetWritesByNode(node_, read_writes_));
  return SUCCESS;
}

void FusingNode::MergeNode(const FusingNodePtr &node) {
  if ((min_order_ == kInvalidId) || (node->GetMinOrder() < min_order_)) {
    min_order_ = node->GetMinOrder();
  }

  if (node->GetMaxOrder() > max_order_) {
    max_order_ = node->GetMaxOrder();
  }
  if (!node->GetNodes().empty()) {
    nodes_.insert(nodes_.end(), node->GetNodes().begin(), node->GetNodes().end());
  }
  nodes_.emplace_back(node);

  // 合并ancestors
  const auto &ancestors = node->GetAncestors();
  if (!ancestors.empty()) {
    ancestors_.insert(ancestors.begin(), ancestors.end());
  }

  fusion_nodes_size_ += node->GetFusionNodesSize();
}

bool FusingNode::IsAncestor(const FusingNodePtr &node) const {
  // 判断node是否是自己的祖先节点
  return ancestors_.find(node->GetOrgNode()) != ancestors_.end();
}

Status FusingNode::Fuse(const FusingNodePtr &node1, const FusingNodePtr &node2, const FusingNodes &fusing_nodes) {
  MergeNode(node1);
  MergeNode(node2);

  // 更新融合节点自己的读写内存信息
  GE_ASSERT_SUCCESS(UpdateReadsAndWrites());
  GELOGD("%s input size:%u output size:%u.", node_->GetNamePtr(), node_->GetAllInDataAnchorsSize(),
         node_->GetAllOutDataAnchorsSize());

  // 融合后，融合节点的输出节点对应读写内存信息发生变化，需要更新
  for (const auto out_anchor : node_->GetAllOutDataAnchorsPtr()) {
    GE_ASSERT_NOTNULL(out_anchor);
    for (const auto &peer_in_anchor : out_anchor->GetPeerAnchors()) {
      GE_ASSERT_NOTNULL(peer_in_anchor);
      const auto peer_node = peer_in_anchor->GetOwnerNodeBarePtr();
      GE_ASSERT_NOTNULL(peer_node);
      const auto it = fusing_nodes.find(peer_node);
      if (it != fusing_nodes.end()) {
        // 把自己和自己的祖先加入输出节点的祖先里
        auto &ancestors = it->second->MutableAncestors();
        ancestors.insert(node_);
        ancestors.insert(ancestors_.begin(), ancestors_.end());
        GELOGD("Add %s to %s's ancestors.", GetNamePtr(), it->second->GetNamePtr());
        it->second->UpdateReadsAndWrites();
      } else {
        GELOGE(FAILED, "node:%s find FusingNode by:%s failed", node_->GetNamePtr(), peer_node->GetNamePtr());
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

Status FusingNode::UpdateReadsAndWrites() {
  GELOGD("Updating reads and writes for node: %s.", GetNamePtr());
  read_writes_.clear();
  GE_ASSERT_SUCCESS(GetReadsByNode(node_, read_writes_));
  GE_ASSERT_SUCCESS(GetWritesByNode(node_, read_writes_));
  return SUCCESS;
}

void FusingNode::PrintOriginalNodes() {
  if (nodes_.empty()) {
    return;
  }
  GELOGI("%s {", GetNamePtr());
  for (const auto &node : nodes_) {
    if (node->GetNodes().size() <= 1U) {
      GELOGI("%s", node->GetOrgNode()->GetNamePtr());
    }
  }
  GELOGI("}");
}

Expression ScoreFusion::ScoreFusionMemory(const FusingNode &node1, const FusingNode &node2) {
  const auto &node1_memory_buffers = node1.GetReadWrites();
  Expression reduced_memory = kSymbolZero;
  // 读写相同的内存，融合后就是可减少的搬运内存
  for (const auto &memory_buffer : node2.GetReadWrites()) {
    const auto it = node1_memory_buffers.find(memory_buffer);
    if (it != node1_memory_buffers.end()) {
      // 减少首次的+运算，提升性能
      reduced_memory = (BackendUtils::IsEqZero(reduced_memory)) ? memory_buffer.size : (reduced_memory + memory_buffer.size);
    }
  }
  return reduced_memory;
}

bool ScoreFusion::Compare(const NodePair &pair1, const NodePair &pair2) {
  // 根据内存大小和邻近性综合评分，内存大小优先
  const auto &memory_score1 = pair1.memory_score;
  const auto &memory_score2 = pair2.memory_score;
  const int64_t proximity_score1 = pair1.proximity_score;
  const int64_t proximity_score2 = pair2.proximity_score;

  // 内存大小相同，再看邻近性
  if (memory_score1 == memory_score2) {
    if (proximity_score1 == proximity_score2) {
      return pair1.first->GetMinOrder() < pair2.first->GetMinOrder();
    }
    return proximity_score1 < proximity_score2;
  }

  // Expression对于常量可以获取具体值进行比较
  int64_t const_value1 = 0;
  int64_t const_value2 = 0;
  if (memory_score1.GetConstValue<int64_t>(const_value1) && memory_score2.GetConstValue<int64_t>(const_value2)) {
    return const_value1 > const_value2;
  }
  return SymbolicUtils::StaticCheckGt(memory_score1, memory_score2) == ge::TriBool::kTrue;
}

NodePair::NodePair(const FusingNodePtr &f, const FusingNodePtr &s)
    : first(f), second(s) {
  memory_score = std::move(ScoreFusion::ScoreFusionMemory(*first, *second));
  proximity_score = std::max(abs(first->GetMinOrder() - second->GetMaxOrder()),
                             abs(second->GetMinOrder() - first->GetMaxOrder()));
}

bool NodePair::operator<(const NodePair &other) const {
  return ScoreFusion::Compare(*this, other);
}

Status FusionStrategySolver::Fuse(const ComputeGraphPtr &graph) const {
  const auto &config = AutoFuseConfig::Config().GetFusionStrategySolver();
  GE_ASSERT_NOTNULL(graph);
  const auto statistics_before_fusion = GetFusionStatisticsByGraph(graph);

  // 原图融合
  TRACING_DURATION_START(FusedAscBackendGraphLoopMergeCanFuse);
  GE_ASSERT_SUCCESS(FuseGraph(graph, config.max_fuse_rounds, ""));

  // 融合后的fusedAscBc graph再次尝试loop融合
  for (const auto &fused_ascbc_node : graph->GetAllNodesPtr()) {
    GE_ASSERT_NOTNULL(fused_ascbc_node);
    const auto op_desc = fused_ascbc_node->GetOpDescBarePtr();
    GE_ASSERT_NOTNULL(op_desc);
    const auto attr = op_desc->GetAttrsGroup<AutoFuseAttrs>();
    if ((attr == nullptr) || (attr->GetFuseComputeGraph() == nullptr)) {
      continue;
    }
    GE_ASSERT_SUCCESS(
        BackendUtils::UpdateSubgraphOutputAttr(attr->GetFuseComputeGraph(), fused_ascbc_node->shared_from_this()));
    GE_ASSERT_SUCCESS(BackendUtils::UpdateFusedAscBackendNetOutput(fused_ascbc_node->shared_from_this()));
    const auto fused_graph_attr = attr->GetFuseComputeGraph()->GetAttrsGroup<AutoFuseAttrs>();
    if ((fused_graph_attr == nullptr) || GetInterAttrs(fused_graph_attr).possible_fusion_nodes.empty()) {
      continue;
    }

    GE_ASSERT_SUCCESS(FuseGraph(attr->GetFuseComputeGraph(), 1U, "_" + fused_ascbc_node->GetName()));
  }
  TRACING_COMPILE_DURATION_END(FusedAscBackendGraphLoopMergeCanFuse, "FusedAscBackendGraphLoopMerge", "CanFuse");

  const auto statistics_after_fusion = GetFusionStatisticsByGraph(graph);
  PrintFusionStatistics(statistics_before_fusion, statistics_after_fusion);
  if (!asc_adapt::IsTorchType()) {
    GE_ASSERT_SUCCESS(PrintAndSetOriginInputAndOutput(graph));
  }
  return SUCCESS;
}

Status FusionStrategySolver::FuseGraph(const ComputeGraphPtr &graph, const uint32_t max_fuse_rounds,
                                       const std::string &suffix) const {
  TRACING_PERF_SCOPE(TracingModule::kModelCompile, "OriginGraphFuse", "CanFuse");
  GE_ASSERT_NOTNULL(GetBackEnd(graph));
  std::vector<FusingNodePtr> nodes;
  GE_ASSERT_SUCCESS(GetNodes(graph, nodes));
  // 创建连接矩阵前必须要先topo排序，否则可能导致融合后成环检测失效，因此要放在GetNodes后(GetNodes里有topo排序)
  const CycleDetectorSharedPtr cycle_detector = GraphUtils::CreateSharedCycleDetector(graph);
  GE_ASSERT_NOTNULL(cycle_detector);
  GE_ASSERT_SUCCESS(ComputeAncestors(nodes));
  AutofuseUtils::DumpGEGraph(graph, kCanFuseDir, "BeforeCanFuse" + suffix);
  AutofuseUtils::DumpGraphToOnnx(*graph, kCanFuseDir, "BeforeCanFuse" + suffix);

  // 第一轮融合已经判断A，B不能融合，直接保存结果，第二轮不需要重复判断
  std::set<std::pair<FusingNode*, FusingNode*>> can_not_fuse_nodes;
  for (uint32_t i = 0U; i < max_fuse_rounds; ++i) {
    const size_t old_nodes_size = nodes.size();
    GELOGI("===== attempting fusion, round %u/%u, current nodes num %zu =====", i + 1U, max_fuse_rounds,
           old_nodes_size);

    GE_ASSERT_SUCCESS(FuseNodesOnce(i, graph, cycle_detector, nodes, can_not_fuse_nodes));
    const size_t new_nodes_size = nodes.size();
    GELOGI("completed fusion round %u/%u, fused %zu nodes into %zu nodes.", i + 1U, max_fuse_rounds, old_nodes_size,
           new_nodes_size);

    if ((new_nodes_size == old_nodes_size) || (new_nodes_size == 1U)) {
      GELOGI("===== fusion complete (%u iterations) =====", i + 1U);
      break;
    }

    if (i < max_fuse_rounds - 1U) {
      AutofuseUtils::DumpGEGraph(graph, kCanFuseDir, "RoundCanFuse" + suffix + "_" + std::to_string(i + 1U));
      AutofuseUtils::DumpGraphToOnnx(*graph, kCanFuseDir, "RoundCanFuse" + suffix + "_" + std::to_string(i + 1U));
    }
  }

  AutofuseUtils::DumpGEGraph(graph, kCanFuseDir, "AfterCanFuse" + suffix);
  AutofuseUtils::DumpGraphToOnnx(*graph, kCanFuseDir, "AfterCanFuse" + suffix);

  if (IsLogEnable(GE_MODULE_NAME, DLOG_INFO)) {
    for (const auto &node : nodes) {
      node->PrintOriginalNodes();
    }
  }
  return SUCCESS;
}

Status FusionStrategySolver::UpdateNodesAndTopoId(const ComputeGraphPtr &graph, const FusingNodes &fusing_nodes,
                                                  std::vector<FusingNodePtr> &nodes) const {
  if (fusing_nodes.size() < nodes.size()) {
    GE_ASSERT_SUCCESS(graph->TopologicalSorting());
    nodes.resize(fusing_nodes.size());
    for (const auto &fusing_node : fusing_nodes) {
      GE_ASSERT_TRUE((static_cast<size_t>(fusing_node.first->GetOpDesc()->GetId()) < nodes.size()));
      // 用topo id进行更新，减少一次排序操作，提高性能
      nodes[static_cast<size_t>(fusing_node.first->GetOpDesc()->GetId())] = fusing_node.second;
      for (const auto &node : fusing_node.second->GetNodes()) {
        // 已经被融合掉的原图上的节点，id设置为无效值
        if (node != fusing_node.second) {
          node->SetId(kInvalidId);
        }
      }
    }
  }
  return SUCCESS;
}

Status FusionStrategySolver::FuseNodesOnce(const uint32_t round, const ComputeGraphPtr &graph,
    const CycleDetectorSharedPtr &cycle_detector, std::vector<FusingNodePtr> &nodes, std::set<std::pair<FusingNode*,
    FusingNode*>> &can_not_fuse_nodes) const {
  FusingNodes fusing_nodes;
  std::vector<FusingNodePtr> node_to_fused_node;
  const size_t nodes_size = nodes.size();
  node_to_fused_node.resize(nodes_size);
  for (const auto &node : nodes) {
    fusing_nodes[node->GetOrgNode().get()] = node;
    node->UpdateOrder();
    GE_ASSERT_TRUE(static_cast<size_t>(node->GetId()) < nodes_size);
    node_to_fused_node[node->GetId()] = node;
  }

  for (const auto &node_pair: GetPossibleFusions(round, graph, nodes, fusing_nodes, can_not_fuse_nodes)) {
    auto primal_node1 = node_pair.first;
    auto primal_node2 = node_pair.second;
    // 获取到两个pair {node1, node2} {node1, node3},
    // node1,node2先融合成new_node，处理{node1, node3}时会判断new_node和node3是否能融合
    auto fused_node1 = node_to_fused_node[primal_node1->GetId()];
    auto fused_node2 = node_to_fused_node[primal_node2->GetId()];
    // 保证node1是node2的前序
    if (fused_node1->IsAncestor(fused_node2)) {
      std::swap(fused_node1, fused_node2);
      std::swap(primal_node1, primal_node2);
    }
    GELOGI("try to fuse node1:%s(fusing_node:%s), node2:%s(fusing_node:%s).", primal_node1->GetNamePtr(),
           fused_node1->GetNamePtr(), primal_node2->GetNamePtr(), fused_node2->GetNamePtr());

    // node1, node2如果未被融合不需要重复判断CanFuse，在GetPossibleFusions已经判断过了
    if (((fused_node1 != primal_node1) || (fused_node2 != primal_node2))
        && (!CanFuse(graph, fused_node1, fused_node2, can_not_fuse_nodes))) {
      continue;
    }

    if (WillFusionCreateCycle(cycle_detector, fused_node1, fused_node2)) {
      GELOGI(
          "node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s][it will create cycle after fuse]",
          fused_node1->GetNamePtr(), fused_node1->GetOrgNode()->GetType().c_str(), fused_node2->GetNamePtr(),
          fused_node2->GetOrgNode()->GetType().c_str(), ge::NotFuseReasonCode(ge::NotFuseReason::kWillCreateCycle));
      continue;
    }

    const auto new_node = FuseNode(graph, fused_node1, fused_node2, fusing_nodes);
    GE_ASSERT_NOTNULL(new_node);

    for (const auto &node : new_node->GetNodes()) {
      // 只有本轮融合处理涉及的节点才要更新，上轮被融合的节点id已经被设置为-1
      if (node->GetId() != kInvalidId) {
        GE_ASSERT_TRUE((static_cast<size_t>(node->GetId()) < nodes_size));
        node_to_fused_node[node->GetId()] = new_node;
      }
    }

    cycle_detector->ExpandAndUpdate({fused_node1->GetOrgNode(), fused_node2->GetOrgNode()}, new_node->GetNamePtr());
  }

  // 产生融合后才需要更新节点和拓扑排序，提高性能
  GE_ASSERT_SUCCESS(UpdateNodesAndTopoId(graph, fusing_nodes, nodes));
  return SUCCESS;
}

NodePairList FusionStrategySolver::GetPossibleFusions(const uint32_t round, const ComputeGraphPtr &graph,
    std::vector<FusingNodePtr> &nodes, const FusingNodes &fusing_nodes,
    std::set<std::pair<FusingNode*, FusingNode*>> &can_not_fuse_nodes) const {
  NodePairList possible_fusions;
  // 主要用于减少CanFuse判断，node1，node2有多个输入时会产生多个相同的node1,node2节点对
  std::set<std::pair<FusingNode*, FusingNode*>> repeat_check;

  // 如果是配置了可能融合的node信息，从属性中获取直接返回，并且清空，保证只做一次fuse
  if (graph->GetAttrsGroup<AutoFuseAttrs>() != nullptr) {
    for (const auto &node_pair : GetInterAttrs(graph->GetAttrsGroup<AutoFuseAttrs>()).possible_fusion_nodes) {
      const auto it1 = fusing_nodes.find(node_pair.first.get());
      const auto it2 = fusing_nodes.find(node_pair.second.get());
      GE_ASSERT_TRUE((it1 != fusing_nodes.end()) && (it2 != fusing_nodes.end()));
      if (CanFuse(graph, it1->second, it2->second, can_not_fuse_nodes)) {
        possible_fusions.emplace_back(it1->second, it2->second);
      }
    }
    GetInterAttrs(graph->GetAttrsGroup<AutoFuseAttrs>()).possible_fusion_nodes.clear();
    GE_ASSERT_SUCCESS(FusionStrategySolver::SortAndLogPossibleFusions(possible_fusions));
    return possible_fusions;
  }

  auto check_all_pairs =
      [&graph, &possible_fusions, &repeat_check, &can_not_fuse_nodes, this](const std::vector<FusingNodePtr> &nodes) {
    for (size_t index1 = 0U; index1 < nodes.size(); ++index1) {
      for (size_t index2 = index1 + 1U; index2 < nodes.size(); ++index2) {
        const auto &node1 = nodes[index1];
        const auto &node2 = nodes[index2];
        if (repeat_check.find({node1.get(), node2.get()}) != repeat_check.end()) {
          continue;
        }
        repeat_check.insert({node1.get(), node2.get()});
        if (CanFuse(graph, node1, node2, can_not_fuse_nodes)) {
          possible_fusions.emplace_back(node1, node2);
        }
      }
    }
  };

  // 先按照读写相同内存把节点进行分组
  std::map<const MemoryBuffer*, std::vector < FusingNodePtr>, decltype(&CompareBuffer)> buffer_grouping(&CompareBuffer);
  for (const auto &node : nodes) {
    for (const auto &buf : node->GetReadWrites()) {
      buffer_grouping[&buf].emplace_back(node);
    }
  }

  // 再把每个分组中的节点两两组对
  for (const auto &node_group : buffer_grouping) {
    check_all_pairs(node_group.second);
  }

  // 先优先级排序，再按照融合打分把这个节点对进行排序，融合打分主要参考融合后节省的内存搬运大小和节点的邻近性
  return GetPossibleFusionsWithPrioritySort(round, graph, possible_fusions);
}

NodePairList FusionStrategySolver::GetPossibleFusionsWithPrioritySort(const uint32_t round,
    const ComputeGraphPtr &graph, const NodePairList &possible_fusions) const {
  (void) round;
  NodePairList sorted_possible_fusions;

  // 根据backend返回的优先级进行分组，返回最高优先级对应的节点对，Priority值越小优先级越高.
  if (possible_fusions.empty()) {
    return possible_fusions;
  }
  std::map<FusionPriority, NodePairList> possible_fusions_group_by_priority;
  for (const auto &node_pair : possible_fusions) {
    const auto &node1 = node_pair.first;
    const auto &node2 = node_pair.second;
    FusionPriority fusion_pair_priority = GetBackEnd(graph)->GetFusionPairPriority(node1->GetOrgNode(), node2->GetOrgNode());
    possible_fusions_group_by_priority[fusion_pair_priority].emplace_back(std::move(node_pair));
  }

  GE_ASSERT_TRUE(!possible_fusions_group_by_priority.empty());
  for (auto &it : possible_fusions_group_by_priority) {
    GE_ASSERT_TRUE(!it.second.empty());
    GELOGI("Priority[%d] node pairs[%zu].", it.first, it.second.size());
    GE_ASSERT_SUCCESS(FusionStrategySolver::SortAndLogPossibleFusions(it.second));
    sorted_possible_fusions.insert(sorted_possible_fusions.end(), std::make_move_iterator(it.second.begin()),
                                   std::make_move_iterator(it.second.end()));
  }

  return sorted_possible_fusions;
}

bool FusionStrategySolver::CanFuse(const ComputeGraphPtr &graph, const FusingNodePtr &node1,
    const FusingNodePtr &node2, std::set<std::pair<FusingNode*, FusingNode*>> &can_not_fuse_nodes) const {
  // A,B,C融合成A_B_C后，A,B,C获取到的融合节点都是A_B_C，判断{B,C}融合对时可能会出现node1=node2的场景
  if (node1 == node2) {
    return false;
  }

  // 利用保存的后端判断的结果，提高性能
  if (can_not_fuse_nodes.find({node1.get(), node2.get()}) != can_not_fuse_nodes.end()) {
    return false;
  }

  // 融合后节省的内存读写大小为0不做融合
  if (BackendUtils::IsEqZero(ScoreFusion::ScoreFusionMemory(*node1, *node2))) {
    GELOGI(
        "node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s][node1 and node2 have no shared data]",
        node1->GetNamePtr(), node1->GetOrgNode()->GetType().c_str(), node2->GetNamePtr(),
        node2->GetOrgNode()->GetType().c_str(), ge::NotFuseReasonCode(ge::NotFuseReason::kNoSharedData));
    can_not_fuse_nodes.insert({node1.get(), node2.get()});
    return false;
  }

  // node1是node2的祖先节点，判断纵向融合，否则判断横向融合
  if (node2->IsAncestor(node1)) {
    if(!GetBackEnd(graph)->CanFuseVertical(node1->GetOrgNode(), node2->GetOrgNode())) {
      GELOGI("cannot fuse %s with %s: cannot fuse vertical by backend.", node1->GetNamePtr(), node2->GetNamePtr());
      can_not_fuse_nodes.insert({node1.get(), node2.get()});
      return false;
    }
  } else {
    // 横向融合导致内存峰值增加的不做融合
    if (CanFusionIncreasePeakMemory(node1, node2) || CheckWriteMemoryAfterFusion(node1, node2)) {
      GELOGI(
          "node1 %s(%s) and node2 %s(%s) can not fuse, the reason is [%s][it will increase peak memory after fuse]",
          node1->GetNamePtr(), node1->GetOrgNode()->GetType().c_str(), node2->GetNamePtr(),
          node2->GetOrgNode()->GetType().c_str(), ge::NotFuseReasonCode(ge::NotFuseReason::kIncreasePeakMemory));
      return false;
    }
    if (!GetBackEnd(graph)->CanFuseHorizontal(node1->GetOrgNode(), node2->GetOrgNode())) {
      GELOGI("cannot fuse %s with %s: cannot fuse horizontal by backend.", node1->GetNamePtr(), node2->GetNamePtr());
      can_not_fuse_nodes.insert({node1.get(), node2.get()});
      return false;
    }
  }
  return true;
}

bool FusionStrategySolver::WillFusionCreateCycle(const CycleDetectorSharedPtr &cycle_detector,
                                                 const FusingNodePtr &node1,
                                                 const FusingNodePtr &node2) const {
  return cycle_detector->HasDetectedCycle({{node1->GetOrgNode(), node2->GetOrgNode()}});
}

FusingNodePtr FusionStrategySolver::FuseNode(const ComputeGraphPtr &graph, const FusingNodePtr &node1,
    const FusingNodePtr &node2, FusingNodes &fusing_nodes) const {
  const auto fused_node = GetBackEnd(graph)->Fuse(node1->GetOrgNode(), node2->GetOrgNode(), counter_);
  if (fused_node != nullptr) {
    GE_ASSERT_SUCCESS(CheckCycle(fused_node));
    const auto snode = CreateFusingNode(fused_node);
    GE_ASSERT_NOTNULL(snode);
    GE_ASSERT_SUCCESS(snode->Fuse(node1, node2, fusing_nodes));
    auto attr = BackendUtils::GetNodeAutoFuseAttr(fused_node);
    GE_ASSERT_NOTNULL(attr);
    attr->SetFusionNodesSize(snode->GetFusionNodesSize());
    fusing_nodes.erase(node1->GetOrgNode().get());
    fusing_nodes.erase(node2->GetOrgNode().get());
    fusing_nodes[snode->GetOrgNode().get()] = snode;
    GELOGI("fuse %s with %s to %s success.", node1->GetNamePtr(), node2->GetNamePtr(), snode->GetNamePtr());
    return snode;
  } else {
    GELOGE(FAILED, "fuse %s with %s failed, start to dump cache graphs.", node1->GetNamePtr(), node2->GetNamePtr());
    // dump正在融合的dump图和前序融合缓存的dump图
    GE_ASSERT_SUCCESS(DumpCacheGraphForExceptionMerging());
  }
  return nullptr;
}

bool FusionStrategySolver::CanFusionIncreasePeakMemory(const FusingNodePtr &node1, const FusingNodePtr &node2) const {
  const auto &config = AutoFuseConfig::Config().GetFusionStrategySolver();
  // 融合后如果可能导致内存峰值增加就不融合，当前简单用节点间距离判断，距离越大导致内存峰值增加的可能性越大
  const auto proximity_score = std::max(abs(node1->GetMinOrder() - node2->GetMaxOrder()),
      abs(node2->GetMinOrder() - node1->GetMaxOrder()));
  const bool can_fusion_increase = proximity_score > config.max_proximity;
  if (can_fusion_increase) {
    GELOGI("proximity:%ld > max proximity:%ld", proximity_score, config.max_proximity);
  }
  return can_fusion_increase;
}

const std::unique_ptr<FusionDecider>& FusionStrategySolver::GetBackEnd(const ComputeGraphPtr &graph) const {
  if (graph != nullptr) {
    const auto attr = graph->GetAttrsGroup<AutoFuseAttrs>();
    if ((attr != nullptr) && (GetInterAttrs(attr).decider != nullptr)) {
      return GetInterAttrs(attr).decider;
    }
  }
  return FusionDeciderRegistry::Instance().Get(AutoFuseConfig::Config().GetFusionStrategySolver().fwk_type);
}

Status FusionStrategySolver::GetNodes(const ComputeGraphPtr &graph, std::vector<FusingNodePtr> &nodes) const {
  // 排序后获取Id
  GE_ASSERT_SUCCESS(graph->TopologicalSorting());
  nodes.reserve(graph->GetAllNodesSize());
  for (const auto &node: graph->GetAllNodes()) {
    const auto &snode = CreateFusingNode(node);
    GE_ASSERT_NOTNULL(snode);
    const auto attr = node->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
    size_t node_size = 1U;
    if ((attr != nullptr) && (attr->GetAscGraph() != nullptr)) {
      node_size = AscGraphUtils::GetComputeGraph(*(attr->GetAscGraph()))->GetAllNodesSize();
    }
    snode->AddFusionNodeSize(node_size);
    if (attr != nullptr) {
      attr->SetFusionNodesSize(snode->GetFusionNodesSize());
    }
    nodes.emplace_back(snode);
  }
  return SUCCESS;
}

FusingNodePtr FusionStrategySolver::CreateFusingNode(const NodePtr &node) const {
  GE_ASSERT_NOTNULL(node);
  GE_ASSERT_NOTNULL(node->GetOpDesc());
  const auto snode = MakeShared<FusingNode>(node);
  GE_ASSERT_NOTNULL(snode);
  // 非可融合节点不需要处理，提高性能
  if (ge::AutofuseUtils::IsAutoFuseNode(node->GetOpDesc())) {
    GE_ASSERT_SUCCESS(snode->Init());
  }
  return snode;
}

Status FusionStrategySolver::ComputeAncestors(std::vector<FusingNodePtr> &snodes) const {
  for (const auto &snode : snodes) {
    auto &ancestors = snode->MutableAncestors();
    for (const auto &in_node : snode->GetOrgNode()->GetInDataNodes()) {
      ancestors.insert(in_node);
    }
  }
  return SUCCESS;
}

Status FusionStrategySolver::PrintAndSetOriginInputAndOutput(const ComputeGraphPtr &graph) const {
  for (const auto& node : graph->GetAllNodes()) {
    if ((node->GetType() != kAscBackendType) && (node->GetType() != kFusedAscBackendType)) {
      continue;
    }
    const auto origin_output_infos = GetInterAttrs(GetOrCreateAutoFuseAttrs(node->GetOpDesc())).origin_output_names_;
    uint32_t index = 0U;
    const auto output_desc = node->GetOpDesc()->GetAllOutputsDescPtr();
    GE_ASSERT_TRUE(output_desc.size() == origin_output_infos.size(), "node %s(%s) output desc size:%zu not equal to "
                   "origin output info size:%zu", node->GetNamePtr(), node->GetType().c_str(), output_desc.size(),
                   origin_output_infos.size());
    for (const auto& origin_info : origin_output_infos) {
      GE_ASSERT_TRUE(static_cast<size_t>(index) < origin_output_infos.size(), "node %s(%s) output index:%zu >= origin "
                     "output info size:%zu", node->GetNamePtr(), node->GetType().c_str(), static_cast<size_t>(index),
                     origin_output_infos.size());
      GELOGD("ascbc_dfx_log(can_fuse), %s, output_idx: %u, origin_ge_node: %s, output_idx: %d", node->GetNamePtr(),
             index, origin_info.first.c_str(), origin_info.second);
      ge::AttrUtils::SetStr(output_desc.at(index), ge::ATTR_NAME_DATA_DUMP_ORIGIN_NAME, origin_info.first);
      ge::AttrUtils::SetInt(output_desc.at(index), ge::ATTR_NAME_DATA_DUMP_ORIGIN_OUTPUT_INDEX, origin_info.second);
      ++index;
    }
  }

  return SUCCESS;
}

bool FusionStrategySolver::CheckWriteMemoryAfterFusion(const FusingNodePtr &node1, const FusingNodePtr &node2) const {
  std::set<ge::MemoryBuffer> node1_write_memory;
  GE_ASSERT_SUCCESS(GetWritesByNode(node1->GetOrgNode(), node1_write_memory));
  std::set<ge::MemoryBuffer> node2_write_memory;
  GE_ASSERT_SUCCESS(GetWritesByNode(node2->GetOrgNode(), node2_write_memory));
  std::set<ge::MemoryBuffer> output_memory_set;
  std::set_union(node1_write_memory.begin(), node1_write_memory.end(),
                 node2_write_memory.begin(), node2_write_memory.end(),
                 std::inserter(output_memory_set, output_memory_set.begin()));
  Expression output_memory = kSymbolZero;
  for (const auto &memory_buffer : output_memory_set) {
    output_memory = output_memory + memory_buffer.size;
  }
  const auto &config = AutoFuseConfig::Config().GetFusionStrategySolver();
  ge::Expression max_output_memory_size = ge::Symbol(config.max_output_memory_size_after_fusion);
  int64_t const_value1 = 0;
  int64_t const_value2 = 0;
  if (output_memory.GetConstValue<int64_t>(const_value1) && max_output_memory_size.GetConstValue<int64_t>(const_value2)) {
    GELOGD("after node1 %s(%s) and node2 %s(%s) fusion, output memory size(%" PRId64 ") VS threshold(%" PRId64 ")",
           node1->GetNamePtr(), node1->GetOrgNode()->GetType().c_str(), node2->GetNamePtr(),
           node2->GetOrgNode()->GetType().c_str(), const_value1, const_value2);
    return const_value1 > const_value2;
  }
  return SymbolicUtils::StaticCheckGt(output_memory, max_output_memory_size) == ge::TriBool::kTrue;
}
}
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/build/memory/checker/dependency_analyzer.h"
#include <stack>
#include <limits>
#include "graph/build/memory/block_mem_assigner.h"
#include "graph/debug/ge_attr_define.h"
#include "common/checker.h"
#include "graph/utils/op_type_utils.h"
#include "graph/optimize/mem_layout_conflict_optimize/mem_layout_conflict_util.h"

namespace ge {
namespace {
constexpr size_t kPrintMaxNameLen = 350U;
std::string kOutNodeMeaning =
    "the output node refers to all nodes that use this memory as input, not only directly connected output "
    "nodes, but also consider RefNode, continuous memory";
constexpr int64_t kInvalidTopoId = std::numeric_limits<int64_t>::max();
constexpr size_t kMaxLogCharNum = 1200U;

int64_t GetTopoId(const Node *node) {
  return node->GetOpDescBarePtr()->GetId();
}

std::string NodeNameOut(const ge::Node *n, const uint32_t out_index) {
  std::stringstream ss;
  ss << n->GetName().substr(0, kPrintMaxNameLen) << "_out_" << out_index << "(" << n->GetType()
     << ", stream:" << n->GetOpDescBarePtr()->GetStreamId() << ", topo:" << GetTopoId(n) << ")";
  return ss.str();
}

std::string NodeName(const ge::Node *n) {
  std::stringstream ss;
  ss << n->GetName().substr(0, kPrintMaxNameLen) << "(" << n->GetType()
     << ", stream:" << n->GetOpDescBarePtr()->GetStreamId() << ", topo:" << GetTopoId(n) << ")";
  return ss.str();
}

std::string TopoIdTypeOutIndex(const ge::Node *n, const uint32_t out_index) {
  std::stringstream ss;
  int64_t offset = -1;
  const auto outputs_offset = n->GetOpDescBarePtr()->GetOutputOffset();
  if (out_index < outputs_offset.size()) {
    offset = outputs_offset[out_index];
  }
  const auto output_tensor_desc = n->GetOpDescBarePtr()->MutableOutputDesc(static_cast<uint32_t>(out_index));
  int64_t mem_size = -1;
  if (output_tensor_desc != nullptr) {
    MemReuseUtils::GetTensorSize(*output_tensor_desc, mem_size, false);
  }
  const auto stream = n->GetOpDescBarePtr()->GetStreamId();
  ss << GetTopoId(n) << "(" << n->GetType() << ")[out:" << out_index << ", s:" << stream << ", offset:" << offset
     << ", size:" << mem_size << "]";
  return ss.str();
}

std::string TopoIdTypeInIndex(const ge::Node *n, const uint32_t in_index) {
  std::stringstream ss;
  const auto stream = n->GetOpDescBarePtr()->GetStreamId();
  ss << GetTopoId(n) << "(" << n->GetType() << ")[in:" << in_index << ", s:" << stream << "]";
  return ss.str();
}

std::string TopoIdTypeIndex(const NodeIndexIO &node_index_io) {
  if (node_index_io.io_type_ == kOut) {
    return TopoIdTypeOutIndex(node_index_io.node_ptr_, node_index_io.index_);
  } else {
    return TopoIdTypeInIndex(node_index_io.node_ptr_, node_index_io.index_);
  }
}
std::string NodesIdStr(const std::vector<Node *> &nodes) {
  std::stringstream ss;
  for (const auto node : nodes) {
    ss << GetTopoId(node) << ", ";
  }
  return ss.str();
}

std::vector<std::string> IdsStr(const std::vector<int64_t> &ids) {
  std::vector<std::string> ret;
  if (ids.empty()) {
    return ret;
  }
  std::stringstream ss;
  auto start = ids[0U];
  auto end = start;
  for (size_t i = 1U; i < ids.size(); ++i) {
    if (ids[i] == end + 1) {
      end = ids[i];
    } else {
      if (start == end) {
        ss << start;
      } else {
        ss << start << "-" << end;
      }
      ss << ", ";
      end = ids[i];
      start = ids[i];
      if (ss.str().length() > kMaxLogCharNum) {
        ret.emplace_back(ss.str());
        ss.str("");
        ss.clear();
      }
    }
  }

  if (start == end) {
    ss << start;
  } else {
    ss << start << "-" << end;
  }
  if (!ss.str().empty()) {
    ret.emplace_back(ss.str());
  }
  return ret;
}

bool IsNeedMergeContinuousInput(const Node *node) {
  if (MemLayoutConflictUtil::IsNoPaddingContinuousInput(node)) {
    return true;
  }
  if (MemLayoutConflictUtil::IsContinuousInput(node)) {
    for (const auto in_anchor : node->GetAllInDataAnchorsPtr()) {
      if (in_anchor->GetPeerOutAnchor() != nullptr) {
        const auto in_node = in_anchor->GetPeerOutAnchor()->GetOwnerNodeBarePtr();
        GE_ASSERT_NOTNULL(in_node);
        GE_ASSERT_NOTNULL(in_node->GetOpDescBarePtr());
        std::vector<int64_t> offsets_for_fusion = {};
        if (AttrUtils::GetListInt(in_node->GetOpDescBarePtr(), ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION,
                                  offsets_for_fusion)) {
          GELOGI("continuous_input node %s(%s) input node %s has %s attr, need merge symbol", NodeName(node).c_str(),
                 node->GetTypePtr(), NodeName(in_node).c_str(), ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION.c_str());
          return true;
        }
      }
    }
  }
  return false;
}
}  // namespace

WrapperInfo::WrapperInfo(const ComputeGraphPtr &sub_graph, const size_t bitmap_size)
    : subgraph_all_nodes_bitmap(bitmap_size) {
  const auto &direct_nodes_visitor = sub_graph->GetDirectNode();
  std::for_each(direct_nodes_visitor.begin(), direct_nodes_visitor.end(), [this](const NodePtr &node) {
    if (node->GetOpDesc() != nullptr) {
      direct_nodes.emplace_back(GetTopoId(node.get()));
    }
  });
  const auto &all_nodes_visitor = sub_graph->GetAllNodes();
  std::for_each(all_nodes_visitor.begin(), all_nodes_visitor.end(), [this](const NodePtr &node) {
    if (node->GetOpDesc() != nullptr) {
      subgraph_all_nodes_bitmap.SetBit(GetTopoId(node.get()));
    }
  });
}

bool DependencyAnalyzer::CanAReuseB(const Node *const a, uint32_t a_output_index, const Node *const b,
                                    uint32_t b_output_index) const {
  if (!init_flag_) {
    GELOGE(ge::FAILED, "node memory dependency analyzer is not inited.");
    return false;
  }
  if (!CheckParam(a, a_output_index, b, b_output_index)) {
    GELOGE(FAILED, "invalid param.");
    return false;
  }
  return CanAReuseBInner(a, a_output_index, b, b_output_index);
}

bool DependencyAnalyzer::CanAReuseBInner(const Node *const a, uint32_t a_output_index, const Node *const b,
                                         uint32_t b_output_index) const {
  const auto &b_node_index_ios = GetSymbolNodeIndexIOList(b, b_output_index);
  if (b_node_index_ios.empty()) {
    GELOGW("b can not find symbol. b: %s, b_out_index: %u", b->GetNamePtr(), b_output_index);
    return true;
  }
  bool suspended = true;
  for (const auto &b_out_node : b_node_index_ios) {
    if ((b_out_node.io_type_ != kIn) || (b_out_node.node_ptr_ == nullptr) ||
        (b_out_node.node_ptr_->GetOpDescBarePtr() == nullptr)) {
      continue;
    }
    const auto out_node_id = GetTopoId(b_out_node.node_ptr_);

    if (IsOutNodeSkip(b_out_node.node_ptr_, b)) {
      continue;
    }
    suspended = false;
    if (!CanReachAllSameBlockAnchor(a, a_output_index, out_node_id)) {
      return false;
    }
  }

  if (suspended && (!CanReachAllSameBlockAnchor(a, a_output_index, GetTopoId(b)))) {
    return false;
  }
  return true;
}

bool DependencyAnalyzer::CanReachAllSameBlockAnchor(const Node *const a, const uint32_t a_output_index,
                                                    const uint32_t out_node_id) const {
  const auto a_out_anchor = a->GetOutDataAnchor(a_output_index);
  const auto &id_iter = out_data_anchor_2_id_map_.find(a_out_anchor);
  if (id_iter == out_data_anchor_2_id_map_.end()) {
    GELOGW("can not find id for out anchor, node: %s, out_index: %u", a->GetNamePtr(), a_output_index);
    return true;
  }
  if (same_block_anchor_ids_[id_iter->second] != nullptr) {
    for (const auto same_out_anchor_id : *same_block_anchor_ids_[id_iter->second]) {
      const auto out_anchor = id_2_out_data_anchor_table_[same_out_anchor_id];
      if ((out_anchor == nullptr) || (out_anchor->GetOwnerNodeBarePtr() == nullptr) ||
          (out_anchor->GetOwnerNodeBarePtr()->GetOpDescBarePtr() == nullptr)) {
        return true;
      }
      const auto dst_id = GetTopoId(out_anchor->GetOwnerNodeBarePtr());
      if (IsOutNodeSkip(out_anchor->GetOwnerNodeBarePtr(), a)) {
        continue;
      }
      if (!reach_nodes_bit_map_table_[out_node_id].GetBit(dst_id)) {
        return false;
      }
    }
  } else {
    if (!reach_nodes_bit_map_table_[out_node_id].GetBit(GetTopoId(a))) {
      return false;
    }
  }
  return true;
}

std::string DependencyAnalyzer::WhyACannotReuseB(const ge::Node *a, uint32_t a_output_index, const ge::Node *b,
                                                 uint32_t b_output_index) {
  if (!CheckParam(a, a_output_index, b, b_output_index)) {
    return "invalid param";
  }
  if (CanAReuseB(a, a_output_index, b, b_output_index)) {
    return NodeNameOut(a, a_output_index) + " can reuse " + NodeNameOut(b, b_output_index);
  }
  ErrorLogReachNodes(b);
  if (!reach_nodes_bit_map_table_[GetTopoId(b)].GetBit(GetTopoId(a))) {
    return NodeName(b) + " can not reach " + NodeName(a);
  }
  std::stringstream ss;
  auto b_output_nodes = GetAllUseSameInBlockNodes(b, b_output_index);
  ErrorLogReachNodes(b_output_nodes);
  if (b_output_nodes.empty()) {
    ss << NodeNameOut(b, b_output_index) << " has no output node, ";
    // 如果b的这个输出anchor没有连接输出节点，那么如果b能到达a及与a同符号的其他节点，那么a就能复用b
    b_output_nodes.push_back(b);
  }

  for (const auto out_node : b_output_nodes) {
    if (IsOutNodeSkip(out_node, b)) {
      continue;
    }
    if (!reach_nodes_bit_map_table_[GetTopoId(out_node)].GetBit(GetTopoId(a))) {
      if (out_node != a) {
        ss << GetTopoId(b) << "\'s output node " << NodeName(out_node) << " can not reach " << NodeName(a) << ", "
           << kOutNodeMeaning;
      } else {
        ss << GetTopoId(b) << "\'s output node " << NodeName(out_node) << " can not reuse its input, "
           << kOutNodeMeaning;
      }
      ErrorLogOutSymbol(b, b_output_index);
      ErrorLogOutSymbol(a, a_output_index);
      return ss.str();
    }
  }
  WhyACannotReuseBInner({a, a_output_index, b, b_output_index}, b_output_nodes, ss);
  return ss.str();
}

void DependencyAnalyzer::WhyACannotReuseBInner(const MemCheckParam &param,
                                               const std::list<const Node *> &b_output_nodes, std::stringstream &ss) {
  const auto same_out_anchors = GetAllUseSameOutBlockAnchors(param.a, param.a_output_index);
  ErrorLogDependNodes(same_out_anchors);
  for (const auto out_node : b_output_nodes) {
    if (IsOutNodeSkip(out_node, param.b)) {
      continue;
    }
    for (const auto &dst_anchor : same_out_anchors) {
      if (IsOutNodeSkip(dst_anchor->GetOwnerNodeBarePtr(), param.a)) {
        continue;
      }
      if (!reach_nodes_bit_map_table_[GetTopoId(out_node)].GetBit(GetTopoId(dst_anchor->GetOwnerNodeBarePtr()))) {
        if (out_node != param.b) {
          ss << GetTopoId(param.b) << "\'s output ";
        }
        ss << "node " << NodeName(out_node) << " can not reach "
           << NodeNameOut(dst_anchor->GetOwnerNodeBarePtr(), dst_anchor->GetIdx());
        if (dst_anchor->GetOwnerNodeBarePtr() != param.a) {
          ss << ", which use same memory block with " << NodeNameOut(param.a, param.a_output_index);
        }
        if (out_node != param.b) {
          ss << ", " << kOutNodeMeaning;
        }
        ErrorLogOutSymbol(param.b, param.b_output_index);
        ErrorLogOutSymbol(param.a, param.a_output_index);
        return;
      }
    }
  }
}

void DependencyAnalyzer::Debug() {
  debug_mode = true;
  DeInitWrapperInfo();
  (void)InitTables();
  InitReachNodesMap();
}

DependencyAnalyzer::DependencyAnalyzer(ComputeGraphPtr &graph, AnchorToSymbol &anchor_to_symbol,
                                       SymbolToAnchors &symbol_to_anchors)
    : compute_graph_(graph), anchor_to_symbol_(anchor_to_symbol), symbol_to_anchors_(symbol_to_anchors) {}

Status DependencyAnalyzer::Init() {
  const auto start_time = GetCurrentTimestamp();
  GELOGI("init start.");
  root_graph_ = compute_graph_;
  while (root_graph_->GetParentGraph() != root_graph_) {
    root_graph_ = root_graph_->GetParentGraph();
    if (root_graph_ == nullptr) {
      root_graph_ = compute_graph_;
      break;
    }
  }
  if (InitTables() != SUCCESS) {
    GELOGW("init table failed. skip memory reuse check");
    return FAILED;
  }
  ExtendSymbolMapByContinuousMemory();
  InitSameOutBlockNodesMap();
  InitReachNodesMap();
  GELOGI("init finish.");
  const auto end_time = GetCurrentTimestamp();
  GELOGI("cost time: %llu microseconds, graph: %s has %zu nodes and %zu out data anchors",
         end_time - start_time, compute_graph_->GetName().c_str(), id_max_, id_2_out_data_anchor_table_.size());
  return SUCCESS;
}

Status DependencyAnalyzer::InitTables() {
  id_2_node_table_.clear();
  graph_all_nodes_.clear();
  const auto &all_nodes = compute_graph_->GetAllNodesPtr();  // node will not be nullptr
  id_max_ = all_nodes.size();
  id_2_node_table_.resize(id_max_, nullptr);

  for (const auto node : all_nodes) {
    graph_all_nodes_.emplace_back(node);
    if (node->GetOpDescBarePtr() == nullptr) {
      GELOGW("op_desc is nullptr, node: %s", node->GetNamePtr());
      return FAILED;
    }
    const auto id = static_cast<size_t>(GetTopoId(node));
    if (id >= id_max_) {
      GELOGW("node topo id: %" PRId64 " >= id_max_: %zu", id, id_max_);
      return FAILED;
    }
    if (id_2_node_table_[id] != nullptr) {
      GELOGW("topo id duplicated, id: %" PRId64 ", origin node: %s, current node: %s", id,
             id_2_node_table_[id]->GetNamePtr(), node->GetNamePtr());
      return FAILED;
    }
    id_2_node_table_[GetTopoId(node)] = node;
  }

  InitOutDataAnchorId(graph_all_nodes_);
  InitPreAndNextNodeOnSameStreamTable();
  InitNodeTypeTable();
  InitWrapperInfo();
  init_flag_ = true;
  GELOGI("init tables finish.");
  return SUCCESS;
}

void DependencyAnalyzer::InitOutDataAnchorId(const std::vector<const Node *> &all_nodes) {
  GELOGI("init out data anchor id begin.");
  id_2_out_data_anchor_table_.clear();
  out_data_anchor_2_id_map_.clear();

  size_t num = 0U;
  for (const auto &node : all_nodes) {
    num += node->GetAllOutDataAnchorsSize();
  }
  id_2_out_data_anchor_table_.resize(num);
  out_data_anchor_2_id_map_.reserve(num);
  same_block_anchor_ids_.resize(num, nullptr);

  size_t id = 0U;
  for (const auto &node : all_nodes) {
    // out_data_anchor will not be nullptr
    for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
      id_2_out_data_anchor_table_[id] = out_data_anchor;
      out_data_anchor_2_id_map_[out_data_anchor] = id;
      ++id;
    }
  }
  GELOGI("init out data anchor id finish.");
}

// a b
// \ /
//  c
// c需要连续输入内存, anchor a_0_out和anchor b_0_out共用一个block，将a_0_out和b_0_out符号表合并
void DependencyAnalyzer::ExtendSymbolMapByContinuousMemory() {
  for (const auto &cur_node : graph_all_nodes_) {
    if (MemLayoutConflictUtil::IsNoPaddingContinuousOutput(cur_node)) {
      OutContinuousMergeSymbol(cur_node);
    }
    if (IsNeedMergeContinuousInput(cur_node)) {
      InContinuousMergeSymbol(cur_node);
    }
  }
  GELOGI("extend symbols finish.");
}

const std::list<NodeIndexIO> &DependencyAnalyzer::GetSymbolNodeIndexIOList(const Node *const node,
                                                                           const uint32_t output_index) const {
  static std::list<NodeIndexIO> dummy_ret;
  const auto cur_anchor = NodeIndexIO(node, output_index, kOut);
  const auto &symbol_iter = anchor_to_symbol_.find(cur_anchor.ToString());
  if (symbol_iter != anchor_to_symbol_.end()) {
    const auto &anchor_iter = symbol_to_anchors_.find(symbol_iter->second);
    if (anchor_iter != symbol_to_anchors_.end()) {
      return anchor_iter->second;
    }
  }
  GELOGW("can not find symbol for node %s, out_index: %u", node->GetNamePtr(), output_index);
  return dummy_ret;
}

// 获取所有以node的第out_index输出作为输入的节点
std::list<const Node *> DependencyAnalyzer::GetAllUseSameInBlockNodes(const Node *node, uint32_t out_index) {
  return GetAllUseSameBlockNodesByType(node, out_index, kIn);
}

std::list<OutDataAnchorPtr> DependencyAnalyzer::GetAllUseSameOutBlockAnchors(const Node *node, uint32_t out_index) {
  const auto cur_anchor = NodeIndexIO(node, out_index, kOut);
  const auto &symbol = anchor_to_symbol_[cur_anchor.ToString()];
  std::list<OutDataAnchorPtr> use_same_block_anchors;
  std::for_each(symbol_to_anchors_[symbol].cbegin(), symbol_to_anchors_[symbol].cend(),
                [&use_same_block_anchors](const NodeIndexIO &anchor) {
                  if ((anchor.io_type_ == kOut) && (anchor.node_ != nullptr)) {
                    use_same_block_anchors.emplace_back(anchor.node_->GetOutDataAnchor(anchor.index_));
                  }
                });
  return use_same_block_anchors;
}

std::list<const Node *> DependencyAnalyzer::GetAllUseSameBlockNodesByType(const Node *node, uint32_t out_index,
                                                                          IOType type) {
  const auto cur_anchor = NodeIndexIO(node, out_index, kOut);
  const auto &symbol = anchor_to_symbol_[cur_anchor.ToString()];
  std::list<const Node *> use_same_block_nodes;
  std::for_each(symbol_to_anchors_[symbol].begin(), symbol_to_anchors_[symbol].end(),
                [&use_same_block_nodes, type](const NodeIndexIO &anchor) {
                  if ((anchor.io_type_ == type) && (anchor.node_ != nullptr)) {
                    use_same_block_nodes.emplace_back(anchor.node_.get());
                  }
                });
  return use_same_block_nodes;
}

void DependencyAnalyzer::InContinuousMergeSymbol(const Node *cur_node) {
  std::vector<OutDataAnchorPtr> out_anchors;
  for (const auto &in_anchor : cur_node->GetAllInDataAnchors()) {
    if ((in_anchor != nullptr) && (in_anchor->GetPeerOutAnchor() != nullptr)) {
      out_anchors.emplace_back(in_anchor->GetPeerOutAnchor());
    }
  }
  if (!out_anchors.empty()) {
    MergeOutAnchorSymbol(out_anchors);
  }
}

void DependencyAnalyzer::OutContinuousMergeSymbol(const Node *cur_node) {
  std::vector<OutDataAnchorPtr> out_anchors;
  for (const auto &out_anchor : cur_node->GetAllOutDataAnchors()) {
    if (out_anchor != nullptr) {
      out_anchors.emplace_back(out_anchor);
    }
  }
  if (!out_anchors.empty()) {
    MergeOutAnchorSymbol(out_anchors);
  }
}

void DependencyAnalyzer::MergeOutAnchorSymbol(const std::vector<OutDataAnchorPtr> &out_anchors) {
  std::vector<NodeIndexIO> anchors_vec;
  std::unordered_set<std::string> symbols_set;
  for (const auto &out_anchor : out_anchors) {
    const auto out_node_index_io = NodeIndexIO(out_anchor->GetOwnerNode(), out_anchor->GetIdx(), kOut);
    if (anchor_to_symbol_.find(out_node_index_io.ToString()) == anchor_to_symbol_.end()) {
      GELOGW("%s is not in anchor_to_symbol", out_node_index_io.ToString().c_str());
      continue;
    }
    const auto &symbol = anchor_to_symbol_[out_node_index_io.ToString()];
    if (!symbols_set.insert(symbol).second) {
      continue;
    }
    for (const auto &node_index_io : symbol_to_anchors_[symbol]) {
      anchors_vec.emplace_back(node_index_io);
    }
  }
  MergeSymbols(anchors_vec, symbols_set);
}

void DependencyAnalyzer::MergeSymbols(const std::vector<NodeIndexIO> &anchors_vec,
                                      const std::unordered_set<std::string> &symbols_set) {
  if (symbols_set.size() <= 1U) {
    return;
  }
  const std::string &the_one_symbol = *symbols_set.begin();
  symbol_to_anchors_[the_one_symbol].clear();
  for (const auto &anchor : anchors_vec) {
    anchor_to_symbol_[anchor.ToString()] = the_one_symbol;
    symbol_to_anchors_[the_one_symbol].emplace_back(anchor);
  }
  auto iter = symbols_set.begin();
  ++iter;
  for (; iter != symbols_set.end(); ++iter) {
    symbol_to_anchors_.erase(*iter);
  }
}

void DependencyAnalyzer::InitSameOutBlockNodesMap() {
  std::set<OutDataAnchorPtr> visit_set;
  for (const auto node : graph_all_nodes_) {
    for (uint32_t i = 0U; i < node->GetAllOutDataAnchorsSize(); ++i) {
      if (!visit_set.insert(node->GetOutDataAnchor(i)).second) {
        continue;
      }
      const auto anchors_with_same_block = GetAllUseSameOutBlockAnchors(node, i);
      // 只需要保留大于1个的，大部分都是1个
      if (anchors_with_same_block.size() <= 1U) {
        continue;
      }
      std::list<size_t> ids;
      for (const auto &anchor : anchors_with_same_block) {
        visit_set.insert(anchor);
        const auto iter = out_data_anchor_2_id_map_.find(anchor);
        if (iter != out_data_anchor_2_id_map_.end()) {
          ids.emplace_back(iter->second);
        }
      }
      same_block_anchor_id_holder_.emplace_back(std::move(ids));
      for (const auto id : same_block_anchor_id_holder_.back()) {
        same_block_anchor_ids_[id] = &same_block_anchor_id_holder_.back();
      }
    }
  }
  GELOGI("init same output block anchor map finish, size: %zu.", same_block_anchor_id_holder_.size());
}

void DependencyAnalyzer::InitPreAndNextNodeOnSameStreamTable() {
  pre_node_on_same_stream_table_.clear();
  next_node_on_same_stream_table_.clear();

  pre_node_on_same_stream_table_.resize(id_max_, kInvalidTopoId);
  next_node_on_same_stream_table_.resize(id_max_, kInvalidTopoId);

  std::map<int64_t, int64_t> last_node_map;
  std::map<int64_t, std::vector<int64_t>> stream_2_ids;
  for (const auto node : id_2_node_table_) {
    const auto id = GetTopoId(node);
    const auto stream_id = node->GetOpDescBarePtr()->GetStreamId();
    if (debug_mode) {
      stream_2_ids[stream_id].push_back(id);
    }
    if (stream_id == kInvalidStreamId) {
      continue;
    }
    auto iter = last_node_map.find(stream_id);
    if (iter == last_node_map.end()) {
      last_node_map[stream_id] = id;
      continue;
    }
    pre_node_on_same_stream_table_[id] = iter->second;
    next_node_on_same_stream_table_[iter->second] = id;
    iter->second = id;
  }
  if (debug_mode) {
    for (const auto &ids : stream_2_ids) {
      const auto ids_strs = IdsStr(ids.second);
      for (const auto &str : ids_strs) {
        GELOGI("stream: %" PRId64 ", node topo id: [%s]", ids.first, str.c_str());
      }
    }
  }
  GELOGI("init pre/next node on same stream table finish, stream size: %u.", last_node_map.size());
}

void DependencyAnalyzer::InitNodeTypeTable() {
  is_wrapper_.clear();
  wrapper_ids_.clear();
  is_wrapper_.resize(id_max_, false);
  for (const auto node : graph_all_nodes_) {
    const auto id = GetTopoId(node);
    if (!node->GetOpDescBarePtr()->GetSubgraphInstanceNames().empty()) {
      is_wrapper_[id] = true;
      wrapper_ids_.emplace_back(id);
    }
  }
}

void DependencyAnalyzer::DeInitWrapperInfo() {
  wrapper_info_holder_.clear();
  parent_2_childs_map_.clear();
  parents_on_root_graph_.clear();
}

void DependencyAnalyzer::InitWrapperInfo() {
  for (const auto parent_node : graph_all_nodes_) {
    if ((parent_node->GetOpDescBarePtr() == nullptr) || (parent_node->GetOwnerComputeGraphBarePtr() == nullptr) ||
        parent_node->GetOpDescBarePtr()->GetSubgraphInstanceNames().empty()) {
      continue;
    }
    if (debug_mode) {
      GELOGI("parent_node: %s, topo_id: %" PRId64 "", parent_node->GetNamePtr(), GetTopoId(parent_node));
    }
    for (const auto &sub_graph_name : parent_node->GetOpDescBarePtr()->GetSubgraphInstanceNames()) {
      const auto &sub_graph = root_graph_->GetSubgraph(sub_graph_name);
      if (sub_graph != nullptr) {
        wrapper_info_holder_[parent_node].emplace_back(WrapperInfo(sub_graph, id_max_));
      }
    }
    const Node *grand_parent_node = parent_node->GetOwnerComputeGraphBarePtr()->GetParentNodeBarePtr();
    if ((grand_parent_node != nullptr) && (parent_node->GetOwnerComputeGraphBarePtr() != compute_graph_.get())) {
      parent_2_childs_map_[grand_parent_node].emplace_back(parent_node);
    } else {
      parents_on_root_graph_.insert(parent_node);
    }
  }
  if (debug_mode) {
    DebugWrapperInfo();
  }
}

// A通过控制边连接B，也算是A能到达B
void DependencyAnalyzer::InitReachNodesMap() {
  reach_nodes_bit_map_table_.clear();
  reach_nodes_bit_map_table_.resize(id_max_, LargeBitmap(id_max_));
  std::stack<const Node *> node_stack;
  vector_bit_t visited_flag(id_max_, false);
  vector_bit_t searched_flag(id_max_, false);
  for (const auto node : graph_all_nodes_) {
    if (node->GetType() == NETOUTPUT) {
      node_stack.push(node);
      if (debug_mode) {
        GELOGI("Netoutput nodes: %" PRId64 ", stream_id: %" PRId64 "", GetTopoId(node),
               node->GetOpDescBarePtr()->GetStreamId());
      }
    }
  }
  if (node_stack.empty() && !graph_all_nodes_.empty()) {
    GELOGW("graph %s has no NETOUTPUT node, use %s as last node", compute_graph_->GetName().c_str(),
           graph_all_nodes_.back()->GetNamePtr());
    node_stack.push(graph_all_nodes_.back());
    if (debug_mode) {
      GELOGI("Netoutput nodes: %" PRId64 ", stream_id: %" PRId64 "", GetTopoId(graph_all_nodes_.back()),
             graph_all_nodes_.back()->GetOpDescBarePtr()->GetStreamId());
    }
  }

  while (!node_stack.empty()) {
    const auto cur_node = node_stack.top();
    node_stack.pop();
    const auto cur_node_topo_id = GetTopoId(cur_node);
    if (!visited_flag[cur_node_topo_id]) {
      visited_flag[cur_node_topo_id] = true;
      InitReachNodesMap(cur_node, searched_flag);
      searched_flag[cur_node_topo_id] = true;
      PushPreNode(cur_node, node_stack);
    }
    if (node_stack.empty()) {
      PushNotVisited(visited_flag, node_stack);
    }
  }
  ExtendReachNodesMapCrossSubGraph();
  GELOGI("init reach nodes map finish.");
}

void DependencyAnalyzer::InitReachNodesMap(const Node *root_node, const vector_bit_t &searched_flag) {
  std::stack<const Node *> nodes_stack;
  vector_bit_t visited_flag(id_max_, false);
  PushNextNode(root_node, nodes_stack);
  const auto root_node_topo_id = GetTopoId(root_node);

  while (!nodes_stack.empty()) {
    const auto cur_node = nodes_stack.top();
    const auto cur_node_topo_id = GetTopoId(cur_node);
    nodes_stack.pop();
    if (visited_flag[cur_node_topo_id]) {
      continue;
    }
    visited_flag[cur_node_topo_id] = true;
    reach_nodes_bit_map_table_[root_node_topo_id].SetBit(cur_node_topo_id);
    if (searched_flag[cur_node_topo_id]) {
      reach_nodes_bit_map_table_[root_node_topo_id].Or(reach_nodes_bit_map_table_[cur_node_topo_id]);
    } else {
      PushNextNode(cur_node, nodes_stack);
    }
  }
}

/*
 * 1 子图内所有节点都能到达父节点（除去Netoutput和与父节点同符号的节点），但反过来不行。
 * 2 所有到达父节点的，都能到达该父节点对应子图内所有节点。
 * 3 设置reuable map时，获取输出节点应该刨除父节点。
 *   刨除后如果输出节点为空，表示该节点的输出没人用，能到达的节点都可以复用该节点。
 */
void DependencyAnalyzer::ExtendReachNodesMapCrossSubGraph() {
  WrapperNotReachSubGraphNodes();
  std::stack<const Node *> nodes_stack;

  // 子图外节点reach map补全，解决子图外节点无法到达子图内的问题
  for (const auto &node : parents_on_root_graph_) {
    nodes_stack.push(node);
  }
  while (!nodes_stack.empty()) {
    const auto parent_node = nodes_stack.top();
    const auto parent_id = GetTopoId(parent_node);
    nodes_stack.pop();
    for (const auto &child : parent_2_childs_map_[parent_node]) {
      nodes_stack.push(child);
    }
    const auto owner_graph = parent_node->GetOwnerComputeGraphBarePtr();
    for (const auto &node : owner_graph->GetDirectNode()) {
      if (!reach_nodes_bit_map_table_[GetTopoId(node.get())].GetBit(parent_id)) {
        continue;
      }
      for (const auto &wrapper_info : wrapper_info_holder_[parent_node]) {
        reach_nodes_bit_map_table_[GetTopoId(node.get())].Or(wrapper_info.subgraph_all_nodes_bitmap);
        if (debug_mode) {
          GELOGI("%" PRId64 " reach %" PRId64 "(%s), extend to reach[%" PRId64 "-%" PRId64 "]", GetTopoId(node.get()),
              parent_id, parent_node->GetTypePtr(), wrapper_info.direct_nodes.front(),
              wrapper_info.direct_nodes.back());
        }
      }
    }
  }

  // 子图内节点reach map补全，解决子图内节点无法到达子图外的问题, 必须从外到内
  for (const auto &node : parents_on_root_graph_) {
    nodes_stack.push(node);
  }
  while (!nodes_stack.empty()) {
    const auto parent_node = nodes_stack.top();
    nodes_stack.pop();
    for (const auto &child : parent_2_childs_map_[parent_node]) {
      nodes_stack.push(child);
    }
    const auto same_out_nodes_with_wrapper_set = GetAllNodeIdsSameSymbolWithWrapper(parent_node);
    const auto parent_id = GetTopoId(parent_node);
    const auto &parent_reach_map = reach_nodes_bit_map_table_[parent_id];
    for (const auto &wrapper_info : wrapper_info_holder_[parent_node]) {
      for (const auto direct_node_id : wrapper_info.direct_nodes) {
        reach_nodes_bit_map_table_[direct_node_id].Or(parent_reach_map);
        // 子图内所有节点都能到达父节点（除去Netoutput和与父节点同符号的节点）
        if (same_out_nodes_with_wrapper_set.find(direct_node_id) == same_out_nodes_with_wrapper_set.cend()) {
          reach_nodes_bit_map_table_[direct_node_id].SetBit(parent_id);
        }
      }
    }
  }
}

/*
 * 父节点拓扑序比子图节点的小，如果分配在同一个stream上，那么会存在父节点能够达到子图内节点的情况。
 * 由于父节点和子图内与netoutput相连的节点同一块内存，所以只允许子图内节点到达父节点。
 */
void DependencyAnalyzer::WrapperNotReachSubGraphNodes() {
  for (const auto &wrapper_pair : wrapper_info_holder_) {
    const auto &parent_node = wrapper_pair.first;
    const auto parent_id = GetTopoId(parent_node);
    for (const auto &subgraph_name : parent_node->GetOpDescBarePtr()->GetSubgraphInstanceNames()) {
      const auto &sub_graph = root_graph_->GetSubgraph(subgraph_name);
      if (sub_graph != nullptr) {
        for (const auto &node : sub_graph->GetAllNodes()) {
          reach_nodes_bit_map_table_[parent_id].ClearBit(GetTopoId(node.get()));
        }
      }
    }
  }
}

std::unordered_set<int64_t> DependencyAnalyzer::GetAllNodeIdsSameSymbolWithWrapper(const Node *node) {
  std::unordered_set<int64_t> ret;
  for (uint32_t index = 0U; index < node->GetAllOutDataAnchorsSize(); ++index) {
    const auto cur_anchor = NodeIndexIO(node, index, kOut);
    const auto &symbol = anchor_to_symbol_[cur_anchor.ToString()];
    std::for_each(symbol_to_anchors_[symbol].begin(), symbol_to_anchors_[symbol].end(),
                  [&ret](const NodeIndexIO &anchor) {
                    if ((anchor.node_ != nullptr) && (anchor.node_->GetOpDesc() != nullptr)) {
                      ret.insert(GetTopoId(anchor.node_.get()));
                    }
                  });
  }
  return ret;
}

void DependencyAnalyzer::PushPreNode(const Node *node, std::stack<const Node *> &node_stack) const {
  for (auto &pre_node : node->GetInNodesPtr()) {
    node_stack.push(pre_node);
  }
  if (debug_mode) {
    GELOGI("%" PRId64 " input nodes: [%s]", GetTopoId(node), NodesIdStr(node->GetInNodesPtr()).c_str());
  }
  const auto cur_node_topo_id = GetTopoId(node);
  auto pre_node_on_same_stream = GetPreNodeOnSameStream(cur_node_topo_id);
  if (pre_node_on_same_stream != nullptr) {
    node_stack.push(pre_node_on_same_stream);
    if (debug_mode) {
      GELOGI("%" PRId64 " prev node %" PRId64 " on stream %" PRId64 " ", GetTopoId(node),
             GetTopoId(pre_node_on_same_stream), node->GetOpDescBarePtr()->GetStreamId());
    }
  }
}

void DependencyAnalyzer::PushNextNode(const Node *node, std::stack<const Node *> &nodes_stack) const {
  for (const auto &next_node : node->GetOutNodesPtr()) {
    nodes_stack.push(next_node);
  }
  if (debug_mode) {
    GELOGI("%" PRId64 " out nodes: [%s]", GetTopoId(node), NodesIdStr(node->GetOutNodesPtr()).c_str());
  }
  auto next_node_on_same_stream = GetNextNodeOnSameStream(GetTopoId(node));
  if (next_node_on_same_stream != nullptr) {
    nodes_stack.push(next_node_on_same_stream);
    if (debug_mode) {
      GELOGI("%" PRId64 " next node %" PRId64 " on stream %" PRId64 "", GetTopoId(node),
             GetTopoId(next_node_on_same_stream), node->GetOpDescBarePtr()->GetStreamId());
    }
  }
}

void DependencyAnalyzer::PushNotVisited(const vector_bit_t &visited_flag, std::stack<const Node *> &nodes_stack) const {
  if (id_max_ > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
    return;
  }
  for (auto i = static_cast<int64_t>(id_max_) - 1; i >= 0; --i) {
    if ((!visited_flag[i]) && (id_2_node_table_[i] != nullptr)) {
      nodes_stack.push(id_2_node_table_[i]);
      if (debug_mode) {
        GELOGI("push not visited node: %" PRId64 "", i);
      }
      break;
    }
  }
}

const Node *DependencyAnalyzer::GetPreNodeOnSameStream(const int64_t topo_id) const {
  auto pre_id = pre_node_on_same_stream_table_[topo_id];
  if (static_cast<size_t>(pre_id) >= id_2_node_table_.size()) {
    return nullptr;
  }
  return id_2_node_table_[pre_id];
}

const Node *DependencyAnalyzer::GetNextNodeOnSameStream(const int64_t topo_id) const {
  auto next_id = next_node_on_same_stream_table_[topo_id];
  if (static_cast<size_t>(next_id) >= id_2_node_table_.size()) {
    return nullptr;
  }
  return id_2_node_table_[next_id];
}

void DependencyAnalyzer::ErrorLogDependNodes(const ge::Node *node) const {
  const auto node_id = GetTopoId(node);
  bool continue_id_flag = false;
  size_t start = std::numeric_limits<size_t>::max();
  std::stringstream ss;
  if (reach_nodes_bit_map_table_[0U].GetBit(node_id)) {
    continue_id_flag = true;
    start = 0U;
  }
  for (size_t i = 1U; i < id_max_; ++i) {
    if (reach_nodes_bit_map_table_[i].GetBit(node_id)) {
      if (!continue_id_flag) {
        start = i;
        continue_id_flag = true;
      }
    } else if (continue_id_flag) {
      if ((i - 1U) == start) {
        ss << start << ", ";
      } else {
        ss << start << "-" << (i - 1U) << ", ";
      }
      start = std::numeric_limits<size_t>::max();
      continue_id_flag = false;
    }
  }
  if (continue_id_flag) {
    if (start == id_max_) {
      ss << start;
    } else {
      ss << start << "-" << id_max_;
    }
  }
  GELOGE(ge::FAILED, "%zu depends nodes[%s].", node_id, ss.str().c_str());
}

void DependencyAnalyzer::ErrorLogDependNodes(const std::list<OutDataAnchorPtr> &out_anchors) const {
  std::set<OutDataAnchor *> visited_set;
  for (auto out_anchor : out_anchors) {
    if (visited_set.insert(out_anchor.get()).second) {
      ErrorLogDependNodes(out_anchor->GetOwnerNodeBarePtr());
    }
  }
}

void DependencyAnalyzer::ErrorLogReachNodes(const ge::Node *node) const {
  const auto &bit_map = reach_nodes_bit_map_table_[GetTopoId(node)];
  bool continue_id = false;
  size_t start = std::numeric_limits<size_t>::max();
  std::stringstream ss;
  for (size_t i = 1U; i < id_max_; ++i) {
    if (bit_map.GetBit(i)) {
      if (!continue_id) {
        start = i;
        continue_id = true;
      }
    } else if (continue_id) {
      if (start == (i - 1U)) {
        ss << start << ", ";
      } else {
        ss << start << "-" << (i - 1U) << ", ";
      }
      start = std::numeric_limits<size_t>::max();
      continue_id = false;
    }
  }
  if (continue_id) {
    if (start == id_max_) {
      ss << start;
    } else {
      ss << start << "-" << id_max_;
    }
  }
  GELOGE(ge::FAILED, "%zu can reach [%s].", GetTopoId(node), ss.str().c_str());
}

void DependencyAnalyzer::ErrorLogReachNodes(const std::list<const Node *> &nodes) const {
  std::set<const Node *> visited_set;
  for (const auto node : nodes) {
    if (visited_set.insert(node).second) {
      ErrorLogReachNodes(node);
    }
  }
}

void DependencyAnalyzer::ErrorLogOutSymbol(const Node *node, uint32_t out_index) const {
  NodeIndexIO node_index_io(node, out_index, kOut);
  ErrorLogSymbol(node_index_io);
}

void DependencyAnalyzer::ErrorLogSymbol(const NodeIndexIO &node_index_io) const {
  const auto symbol_iter = anchor_to_symbol_.find(node_index_io.ToString());
  if (symbol_iter == anchor_to_symbol_.end()) {
    return;
  }
  const auto anchors_iter = symbol_to_anchors_.find(symbol_iter->second);
  if (anchors_iter == symbol_to_anchors_.end()) {
    return;
  }
  GELOGE(FAILED, "print anchors in same symbol with %s, start-----------------------------",
         TopoIdTypeIndex(node_index_io).c_str());
  for (const auto &anchor : anchors_iter->second) {
    if (anchor.node_ptr_ == nullptr) {
      continue;
    }
    if (anchor.io_type_ == kIn) {
      const auto in_anchor = anchor.node_ptr_->GetInDataAnchor(anchor.index_);
      if (in_anchor == nullptr) {
        continue;
      }
      const auto peer_out_anchor = in_anchor->GetPeerOutAnchor().get();
      if (peer_out_anchor != nullptr) {
        GELOGE(FAILED, "%s --> %s",
               TopoIdTypeOutIndex(peer_out_anchor->GetOwnerNodeBarePtr(), peer_out_anchor->GetIdx()).c_str(),
               TopoIdTypeIndex(anchor).c_str());
      } else {
        GELOGE(FAILED, "suspended in data anchor: %s", TopoIdTypeIndex(anchor).c_str());
      }
    } else {
      const auto out_anchor = anchor.node_ptr_->GetOutDataAnchor(anchor.index_);
      if ((out_anchor != nullptr) && (out_anchor->GetPeerInDataAnchorsPtr().empty())) {
        GELOGE(FAILED, "suspended out data anchor: %s", TopoIdTypeIndex(anchor).c_str());
      }
    }
  }
  GELOGE(FAILED, "print anchors in same symbol with %s, end-------------------------------",
         TopoIdTypeIndex(node_index_io).c_str());
}

bool DependencyAnalyzer::CheckParam(const ge::Node *a, uint32_t a_output_index, const ge::Node *b,
                                    uint32_t b_output_index) const {
  if ((a == nullptr) || (b == nullptr) || (a->GetOutDataAnchor(a_output_index) == nullptr) ||
      (b->GetOutDataAnchor(b_output_index) == nullptr) || (a->GetOpDescBarePtr() == nullptr) ||
      (b->GetOpDescBarePtr() == nullptr)) {
    GELOGE(FAILED, "invalid param.");
    return false;
  }
  if ((static_cast<size_t>(GetTopoId(a)) >= id_max_) || (static_cast<size_t>(GetTopoId(b)) >= id_max_)) {
    GELOGE(FAILED, "invalid param.");
    return false;
  }
  return true;
}

bool DependencyAnalyzer::IsOutNodeSkip(const Node *const node, const Node *const origin_node) const {
  const auto node_id = GetTopoId(node);
  // 父节点无法到达子图内节点，这里去掉父节点，否则会出现子图内节点无法复用子图输入的问题
  if ((static_cast<size_t>(node_id) < id_max_) && is_wrapper_[node_id]) {
    return true;
  }
  // 子图内data节点不设置stream_id，如果输出恰好空悬，可能无法到达任何节点。
  if (OpTypeUtils::IsDataNode(node->GetType())) {
    return true;
  }
  // 不分配流，建立的到达关系受限，会产生误判
  if (node->GetOpDescBarePtr()->GetStreamId() == ge::kInvalidStreamId) {
    return true;
  }
  std::string batch_label;
  (void)ge::AttrUtils::GetStr(node->GetOpDescBarePtr(), ATTR_NAME_BATCH_LABEL, batch_label);
  std::string origin_node_batch_label;
  (void)ge::AttrUtils::GetStr(origin_node->GetOpDescBarePtr(), ATTR_NAME_BATCH_LABEL, origin_node_batch_label);
  if (batch_label != origin_node_batch_label) {
    return true;
  }
  return false;
}

void DependencyAnalyzer::DebugWrapperInfo() const {
  std::stringstream ss;
  for (const auto root_parent : parents_on_root_graph_) {
    ss << GetTopoId(root_parent) << "(" << root_parent->GetType() << ")"
       << ", ";
  }
  GELOGI("parent nodes on root graph: [%s]", ss.str().c_str());
  ss.str("");
  ss.clear();
  for (const auto &parent_to_childs : parent_2_childs_map_) {
    for (const auto child : parent_to_childs.second) {
      ss << GetTopoId(child) << "(" << child->GetType() << ")"
         << ", ";
    }
    GELOGI("parent node: %" PRId64 "(type: %s), children: [%s]", GetTopoId(parent_to_childs.first),
           parent_to_childs.first->GetTypePtr(), ss.str().c_str());
    ss.str("");
    ss.clear();
  }
  ss.str("");
  ss.clear();
  for (const auto &parent_2_childs : wrapper_info_holder_) {
    size_t sub_graph_num = 1U;
    for (const auto &child : parent_2_childs.second) {
      if (!child.direct_nodes.empty()) {
        GELOGI("parent node: %" PRId64 "(type: %s), subgraph:%zu, children: [%" PRId64 "-%" PRId64 "]",
               GetTopoId(parent_2_childs.first), parent_2_childs.first->GetTypePtr(), sub_graph_num,
               child.direct_nodes.front(), child.direct_nodes.back());
      }
      ++sub_graph_num;
    }
  }
}
}  // namespace ge

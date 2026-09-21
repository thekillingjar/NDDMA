/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include "mmpa/mmpa_api.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/graph_utils.h"
#include "register/graph_optimizer/fusion_common/fusion_statistic_recorder.h"
#include "register/graph_optimizer/fusion_common/graph_pass_util.h"
#include "register/graph_optimizer/graph_fusion/pattern_fusion_base_pass_impl.h"
#include "register/graph_optimizer/fusion_common/fusion_config_info.h"

namespace fe {
namespace {
void PrintAllNodes(const std::vector<ge::NodePtr> &scope_nodes) {
  for (const auto &node : scope_nodes) {
    if (node == nullptr) {
      GELOGD("type: null, name: null");
    } else {
      GELOGD("type: %s, name: %s", node->GetType().c_str(), node->GetName().c_str());
    }
  }
}

void StoreOriginNodes(const Mapping &mapping,
                      GraphPassUtil::OriginOpAttrsVec &origin_op_attrs,
                      std::vector<ge::NodePtr> &original_nodes) {
  for (const auto &item : mapping) {
    if (item.second.empty()) {
      continue;
    }
    for (const auto &node : item.second) {
      (void)original_nodes.emplace_back(node);
      std::vector<std::string> origin_op_attrs_vec;
      origin_op_attrs_vec.push_back(node->GetName());
      origin_op_attrs_vec.push_back(node->GetType());
      (void)origin_op_attrs.emplace_back(origin_op_attrs_vec);
    }
  }
}
}
static const std::string STREAM_LABEL = "_stream_label";
static const std::string ATTR_OP_COMPILE_STRATEGY = "_op_compile_strategy";
static const std::string ATTR_KEEP_DTYPE = "_keep_dtype";
PatternFusionBasePass::PatternFusionBasePass() {
  pattern_fusion_base_pass_impl_ptr_ = std::make_shared<PatternFusionBasePassImpl>();
}

PatternFusionBasePass::~PatternFusionBasePass() {}

Status PatternFusionBasePass::Run(ge::ComputeGraph &graph, OpsKernelInfoStorePtr ops_kernel_info_store_ptr) {
  // save the opskernelstoreptr which will be uesd while checking op support
  pattern_fusion_base_pass_impl_ptr_->SetOpsKernelInfoStore(ops_kernel_info_store_ptr);

  Status ret = Run(graph);
  if (ret != SUCCESS) {
    // do not update cache when not success
    int64_t run_count_attr;
    if (ge::AttrUtils::GetInt(graph, "run_count", run_count_attr)) {
      (void)ge::AttrUtils::SetInt(graph, "run_count", ++run_count_attr);
    }
  }
  return ret;
}
/**
 * @ingroup fe
 * @brief execute pass
 */
Status PatternFusionBasePass::Run(ge::ComputeGraph &graph) {
  bool is_patterns_ok = true;
  // build Pattern
  std::vector<FusionPattern *> patterns;
  pattern_fusion_base_pass_impl_ptr_->GetPatterns(patterns);
  if (patterns.empty()) {
    patterns = DefinePatterns();
    for (FusionPattern *pattern : patterns) {
      if (pattern != nullptr) {
        const bool ok = pattern->Build();
        if (!ok) {
          GELOGW("[RunFusionPass][Check] pattern %s build failed", pattern->GetName().c_str());
        }
        pattern->Dump();
        is_patterns_ok = is_patterns_ok && ok;
      }
    }

    pattern_fusion_base_pass_impl_ptr_->SetPatterns(patterns);
  }

  if (!is_patterns_ok) {
    return FAILED;
  }
  NodeMapInfoPtr node_map_info = nullptr;
  if (GraphPassUtil::GetOpTypeMapToGraph(node_map_info, graph) == SUCCESS) {
    if (node_map_info->run_count == std::numeric_limits<int64_t>::max()) {
      GELOGE(ge::FAILED, "Run count is overflow.");
      return FAILED;
    }
    node_map_info->run_count++;
  }

  // do matching and fusion for each pattern
  bool final_changed = false;
  for (const FusionPattern * const pattern : patterns) {
    if (pattern != nullptr) {
      bool changed = false;
      const Status ret = RunOnePattern(graph, *pattern, changed);
      if (ret != SUCCESS) {
        GELOGW("[RunFusionPass][Check] Running pattern %s failed; the graph was not altered by it.", pattern->GetName().c_str());
        return ret;
      }

      final_changed = final_changed || changed;
    }
  }
  return final_changed ? SUCCESS : NOT_CHANGED;
}

static bool CheckStreamLabel(std::vector<ge::NodePtr> &fused_nodes) {
  std::string stream_label = "";
  for (auto &n : fused_nodes) {
    std::string stream_label_tmp = "";
    if (!ge::AttrUtils::GetStr(n->GetOpDesc(), STREAM_LABEL, stream_label_tmp)) {
      stream_label_tmp = "null";
    }
    if (stream_label == "") {
      stream_label = stream_label_tmp;
    } else if ((stream_label != "") && (stream_label != stream_label_tmp)) {
      return false;
    }
  }
  return true;
}

static bool SetStreamLabelToFusedNodes(std::vector<ge::NodePtr> &fused_nodes,
                                       const std::vector<ge::NodePtr> &original_nodes) {
  if (original_nodes.empty() || original_nodes[0] == nullptr) {
    return true;
  }

  std::string stream_label = "";
  if (ge::AttrUtils::GetStr(original_nodes[0]->GetOpDesc(), STREAM_LABEL, stream_label)) {
    for (ge::NodePtr &node : fused_nodes) {
      if (!ge::AttrUtils::SetStr(node->GetOpDesc(), STREAM_LABEL, stream_label)) {
        GELOGW("[Set][Attr] node %s set attr _stream_label failed", node->GetName().c_str());
        return false;
      }
    }
  }
  return true;
}

void PatternFusionBasePass::DumpMapping(const FusionPattern &pattern, const Mapping &mapping) const {
  std::ostringstream oss;
  oss << std::endl << "Mapping of pattern ";
  oss << pattern.GetName() << ":" << std::endl;
  oss << " Mapping: "  << std::endl;
  for (const auto &item : mapping) {
    const std::shared_ptr<OpDesc> op_desc = item.first;
    const ge::NodePtr node = item.second[0U];
    if ((op_desc != nullptr) && (node != nullptr)) {
      oss << "    " << op_desc->id << " -> " << node->GetName() << std::endl;
    }
  }
  GELOGE(FAILED, "%s", oss.str().c_str());
}

/**
 * @ingroup fe
 * @brief do matching and fusion in graph based on the pattern
 */
Status PatternFusionBasePass::RunOnePattern(ge::ComputeGraph &graph, const FusionPattern &pattern, bool &changed) {
  changed = false;
  Mappings mappings;
  int32_t effect_times = 0;
  const uint32_t graph_id = graph.GetGraphID();
  FusionInfo fusion_info(graph.GetSessionID(), to_string(graph_id), GetName(), static_cast<int32_t>(mappings.size()),
                         effect_times);
  origin_op_anchors_map_.clear();
  // match all patterns in graph, and save them to mappings
  if (!MatchAll(graph, pattern, mappings)) {
    GELOGD("GraphFusionPass[%s]: pattern=%s, matched_times=%zu, effected_times=%d.", GetName().c_str(),
           pattern.GetName().c_str(), mappings.size(), effect_times);
    return SUCCESS;
  }

  GELOGD("This graph has been matched with pattern[%s]. The mappings are as follows.", pattern.GetName().c_str());

  // print the results of matching
  pattern_fusion_base_pass_impl_ptr_->DumpMappings(pattern, mappings);
  NodeMapInfoPtr node_map_info = nullptr;
  // get nodes by type from node
  (void)GraphPassUtil::GetOpTypeMapToGraph(node_map_info, graph);
  // do fusion for each mapping
  for (Mapping &mapping : mappings) {
    GraphPassUtil::OriginOpAttrsVec origin_op_attrs;
    std::vector<ge::NodePtr> original_nodes;
    StoreOriginNodes(mapping, origin_op_attrs, original_nodes);
    bool backward = false;
    GraphPassUtil::GetBackWardAttr(original_nodes, backward, BackWardInheritMode::kFusedNode);
    std::vector<ge::NodePtr> fus_nodes;
    const Status status = Fusion(graph, mapping, fus_nodes);

    const bool isGraphCycle = FusionConfigInfo::Instance().IsEnableNetworkAnalysis() && CheckGraphCycle(graph);
    if (isGraphCycle) {
        GELOGE(FAILED, "Failed to do topological sorting after graph fusion, graph is cyclic, graph name:%s",
               graph.GetName().c_str());
        GELOGE(FAILED, "This graph is cyclic. The mapping and new nodes are as follows.");
        DumpMapping(pattern, mapping);

        std::ostringstream oss;
        for (const auto &node_ : fus_nodes) {
          oss << "name:" << node_->GetName() << ", type:" << node_->GetType() << std::endl;
        }
        GELOGE(FAILED, "%s", oss.str().c_str());
        ge::GraphUtils::DumpGEGraphToOnnx(graph, "graph_cyclic_after " + pattern.GetName());
        return GRAPH_FUSION_CYCLE;
    }

    if (!SetStreamLabelToFusedNodes(fus_nodes, original_nodes)) {
      return FAILED;
    }

    if ((status != SUCCESS) && (status != NOT_CHANGED)) {
      GELOGE(status, "[Fuse][Graph]Fail with pattern[%s].", pattern.GetName().c_str());
      return status;
    }

    if (status == SUCCESS) {
      effect_times++;
      SetDataDumpAttr(original_nodes, fus_nodes);
      for (ge::NodePtr &node : fus_nodes) {
        const ge::OpDescPtr fusion_op = node->GetOpDesc();
        GraphPassUtil::RecordOriginalOpAttrs(original_nodes, fusion_op, GetName(), origin_op_attrs);
        (void)GraphPassUtil::StoreAndUpdataOriginFusionPassName(fusion_op, original_nodes, GetName());
        (void)GraphPassUtil::AddNodeFromOpTypeMap(node_map_info, node);
      }
      const BackWardInheritMode inherit_mode = backward ? BackWardInheritMode::kInheritTrue :
          BackWardInheritMode::kDoNotInherit;
      GraphPassUtil::InheritAttrFromOriNodes(original_nodes, fus_nodes, inherit_mode);
    }
    changed = (changed || (status == SUCCESS));
  }

  // get match times and effect times
  FusionStatisticRecorder &fusion_statistic_inst = FusionStatisticRecorder::Instance();
  fusion_info.SetMatchTimes(static_cast<int32_t>(mappings.size()));
  fusion_info.SetEffectTimes(effect_times);
  fusion_statistic_inst.UpdateGraphFusionMatchTimes(fusion_info);
  fusion_statistic_inst.UpdateGraphFusionEffectTimes(fusion_info);
  GELOGI("GraphId[%d], GraphFusionPass[%s]: pattern=%s, matched_times=%zu, effected_times=%d.", graph_id,
         GetName().c_str(), pattern.GetName().c_str(), mappings.size(), effect_times);
  return SUCCESS;
}

std::vector<FusionPattern *> PatternFusionBasePass::DefineInnerPatterns() {
  std::vector<FusionPattern *> ret;
  return ret;
}

void PatternFusionBasePass::SetDataDumpAttr(const std::vector<ge::NodePtr> &fused_nodes,
                                            const std::vector<ge::NodePtr> &fusion_nodes) {
  // if pass do not specify fused nodes, all matched nodes will be handled as the fused nodes
  const std::vector<ge::NodePtr> &actual_fused_nodes = pattern_fusion_base_pass_impl_ptr_->GetActualFusedNodes();
  if (actual_fused_nodes.empty()) {
    SetOriginalOpDumpAttr(fused_nodes, fusion_nodes);
  } else {
    SetOriginalOpDumpAttr(actual_fused_nodes, fusion_nodes);
  }

  if (fusion_nodes.size() > 1) {
    const bool is_multi_op = true;
    for (const ge::NodePtr &node : fusion_nodes) {
      (void)ge::AttrUtils::SetBool(node->GetOpDesc(), ge::ATTR_NAME_DATA_DUMP_IS_MULTIOP, is_multi_op);
    }
  }

  SetOriginalOutputDumpAttr(fused_nodes, fusion_nodes);
}

void PatternFusionBasePass::SetOriginalOutputDumpAttr(const std::vector<ge::NodePtr> &fused_nodes,
                                                      const std::vector<ge::NodePtr> &fusion_nodes) {
  for (const ge::NodePtr &ori_node : fused_nodes) {
    const auto iter = origin_op_anchors_map_.find(ori_node);
    if (iter != origin_op_anchors_map_.end()) {
      for (const auto &anchor_iter : iter->second) {
        const auto next_node_in_anchor = anchor_iter.first;
        const auto fusion_node_out_data_anchor = next_node_in_anchor->GetPeerOutAnchor();
        if (fusion_node_out_data_anchor == nullptr) {
          GELOGW("[Set][Attr] peer_out_anchor is null");
          return;
        }

        // owner_node of anchor should not be null
        auto fusion_node = fusion_node_out_data_anchor->GetOwnerNode();
        if (fusion_node == nullptr) {
          GELOGW("[Set][Attr] fusion_node is null.");
          return;
        }
        if (pattern_fusion_base_pass_impl_ptr_->IsNodesExist(fusion_node, fusion_nodes)) {
          const auto origin_node_out_anchor = anchor_iter.second;
          if (origin_node_out_anchor == nullptr) {
            GELOGW("[Set][Attr] The ori_out_anchor of node %s is null.", ori_node->GetName().c_str());
            return;
          }

          // owner_node of anchor should not be null
          auto origin_node = origin_node_out_anchor->GetOwnerNode();
          if (origin_node == nullptr) {
            GELOGW("[Set][Attr] origin_node is null");
            return;
          }
          const uint32_t origin_index = static_cast<uint32_t>(origin_node_out_anchor->GetIdx());
          const uint32_t fusion_index = static_cast<uint32_t>(fusion_node_out_data_anchor->GetIdx());
          GraphPassUtil::SetOutputDescAttr(origin_index, fusion_index, origin_node, fusion_node);
        }
      }
    }
  }
}

void PatternFusionBasePass::SetOriginalOpDumpAttr(const std::vector<ge::NodePtr> &fused_nodes,
                                                  const std::vector<ge::NodePtr> &fusion_nodes) {
  for (const ge::NodePtr &node : fusion_nodes) {
    GraphPassUtil::RecordOriginalNames(fused_nodes, node);
  }
}

void PatternFusionBasePass::SetActualFusedNodes(const std::vector<ge::NodePtr> &fused_nodes) {
  pattern_fusion_base_pass_impl_ptr_->SetActualFusedNodes(fused_nodes);
}

bool PatternFusionBasePass::CheckOpSupported(const ge::OpDescPtr &op_desc_ptr) const {
  return pattern_fusion_base_pass_impl_ptr_->CheckOpSupported(op_desc_ptr);
}

bool PatternFusionBasePass::CheckOpSupported(const ge::NodePtr &node) const {
  return pattern_fusion_base_pass_impl_ptr_->CheckOpSupported(node);
}

bool PatternFusionBasePass::CheckAccuracySupported(const ge::NodePtr &node) const {
  return pattern_fusion_base_pass_impl_ptr_->CheckAccuracySupported(node);
}

bool PatternFusionBasePass::CheckEachPeerOut(const ge::NodePtr &node,
                                             const std::unordered_set<ge::NodePtr> &scope_nodes_set,
                                             const std::vector<ge::NodePtr> &scope_nodes) const {
  for (const auto &peer_out : node->GetOutAllNodes()) {
    if (scope_nodes_set.count(peer_out) > 0) {
      continue;
    }
    for (const auto &node_temp :scope_nodes) {
      if ((node_temp == nullptr) || (node_temp == node)) {
        continue;
      }
      GELOGD("Check %s and %s.", peer_out->GetName().c_str(), node_temp->GetName().c_str());

      if (connectivity_->IsConnected(peer_out, node_temp)) {
        GELOGD("There is a path between %s and %s after fusion:",
               peer_out->GetName().c_str(),
               node_temp->GetName().c_str());
        PrintAllNodes(scope_nodes);
        return true;
      }
    }
  }
  return false;
}

bool PatternFusionBasePass::DetectOneScope(const std::vector<ge::NodePtr> &scope_nodes) const {
  /* Create a set for accelerating the searching. */
  const std::unordered_set<ge::NodePtr> scope_nodes_set(scope_nodes.begin(), scope_nodes.end());

  for (const auto &node: scope_nodes) {
    if (node == nullptr) {
      continue;
    }
    if (CheckEachPeerOut(node, scope_nodes_set, scope_nodes)) {
      return true;
    }
  }
  return false;
}

void PatternFusionBasePass::GetConnectionMatrix(std::unique_ptr<ConnectionMatrix> &connection_matrix) {
  connection_matrix = std::move(connectivity_);
}

void PatternFusionBasePass::SetConnectionMatrix(std::unique_ptr<ConnectionMatrix> &connection_matrix) {
  connectivity_ = std::move(connection_matrix);
}

bool PatternFusionBasePass::CycleDetection(const ge::ComputeGraph &graph,
                                           const std::vector<std::vector<ge::NodePtr>> &fusion_nodes) {
  if (connectivity_ == nullptr) {
    try {
      connectivity_ = std::unique_ptr<fe::ConnectionMatrix>(new(std::nothrow) fe::ConnectionMatrix(graph));
    } catch (...) {
      GELOGW("Make shared failed");
      return false;
    }
    connectivity_->Generate(graph);
  }

  for (const auto &scope_nodes : fusion_nodes) {
    if (DetectOneScope(scope_nodes)) {
      return true;
    }
  }
  return false;
}

const std::vector<FusionPattern *> &PatternFusionBasePass::GetPatterns() {
  const auto &patterns = pattern_fusion_base_pass_impl_ptr_->GetPatterns();
  if (!patterns.empty()) {
    return patterns;
  }

  const auto new_defined_patterns = DefinePatterns();
  for (FusionPattern *pattern : new_defined_patterns) {
    if (pattern != nullptr) {
      const bool build_result = pattern->Build();
      if (!build_result) {
        GELOGW("[GetPatterns][Check] Pattern %s build failed", pattern->GetName().c_str());
        return patterns;
      }
      pattern->Dump();
    }
  }

  pattern_fusion_base_pass_impl_ptr_->SetPatterns(new_defined_patterns);
  return pattern_fusion_base_pass_impl_ptr_->GetPatterns();
}

const std::vector<FusionPattern *> &PatternFusionBasePass::GetInnerPatterns() {
  const auto &inner_patterns = pattern_fusion_base_pass_impl_ptr_->GetInnerPatterns();
  if (!inner_patterns.empty()) {
    return inner_patterns;
  }

  const auto new_defined_inner_patterns = DefineInnerPatterns();
  for (FusionPattern *inner_pattern : new_defined_inner_patterns) {
    if (inner_pattern != nullptr) {
      const bool build_result = inner_pattern->Build();
      if (!build_result) {
        GELOGW("[GetPatterns][Check] Pattern %s build failed", inner_pattern->GetName().c_str());
        return inner_patterns;
      }
      inner_pattern->Dump();
    }
  }

  pattern_fusion_base_pass_impl_ptr_->SetInnerPatterns(new_defined_inner_patterns);
  return pattern_fusion_base_pass_impl_ptr_->GetInnerPatterns();
}

bool PatternFusionBasePass::MatchFromOutput(const ge::NodePtr &output_node,
                                            const std::shared_ptr<OpDesc> &output_op_desc, Mapping &mapping) {
  return pattern_fusion_base_pass_impl_ptr_->MatchFromOutput(output_node, output_op_desc, mapping);
}

bool PatternFusionBasePass::CycleDetection(const ge::ComputeGraph &graph,
                                           const std::vector<ge::NodePtr> &fusion_nodes) {
  if (connectivity_ == nullptr) {
    try {
      connectivity_ = std::unique_ptr<fe::ConnectionMatrix>(new(std::nothrow) fe::ConnectionMatrix(graph));
    } catch (...) {
      GELOGW("Make shared failed");
      return false;
    }
    connectivity_->Generate(graph);
  }

  return DetectOneScope(fusion_nodes);
}

bool PatternFusionBasePass::CheckGraphCycle(ge::ComputeGraph &graph) const {
  const Status ret = graph.TopologicalSorting();
  if (ret != ge::GRAPH_SUCCESS) {
    return true;
  }
  return false;
}

/**
 * @ingroup fe
 * @brief match all nodes in graph according to pattern
 * match nodes in graph according to pattern, the algorithm is shown as following:
 * 1. get output node from pattern
 * 2. Search for candidate nodes in Graph (network Graph generated after parsing) according to Op Type and
 * (optional), and add the candidate node to the list of candidates
 * 3. For each Node in the candidate list, check whether the type and the number
 * of precursors are consistent with the description of corresponding Op in pattern.
 * If they are consistent, add the precursor Node to the
 * candidate list, and add "PatternOp-GraphNode" to the mapping; otherwise, return an empty mapping
 * 4. repeat step 3 until all the Ops in pattern are matched
 * 5. if all the Ops in pattern are matched successfully, return the mapping of PatternOp and GraphNode
 */
bool PatternFusionBasePass::MatchAll(const ge::ComputeGraph &graph, const FusionPattern &pattern,
    Mappings &mappings) {
  std::vector<ge::NodePtr> matched_output_nodes;

  // find all the output nodes of pattern in the graph based on Op type
  std::shared_ptr<FusionPattern::OpDesc> output_op_desc = pattern.GetOutput();
  if (output_op_desc == nullptr) {
    return false;
  }

  if (!pattern_fusion_base_pass_impl_ptr_->GetMatchOutputNodes(graph, pattern, matched_output_nodes)) {
    return false;
  }

  // begin matching from every output node
  for (ge::NodePtr &output_node : matched_output_nodes) {
    Mapping mapping;
    if (pattern_fusion_base_pass_impl_ptr_->MatchFromOutput(output_node, output_op_desc, mapping)) {
      // node attr _stream_label must be equal
      auto fusion_nodes = GetNodesFromMapping(mapping);
      if (!CheckStreamLabel(fusion_nodes)) {
        return false;
      }
      std::string reason_not_support;
      const bool ret = graph.IsSupportFuse(fusion_nodes, reason_not_support);
      if (!ret) {
        GELOGD("IsSupportFuse did not succeed, reason is [%s].", reason_not_support.c_str());
        continue;
      }
      mappings.push_back(mapping);

      // Record output nodes anchor vs succeed node anchor map
      RecordOutputAnchorMap(output_node);
    }
  }
  // if matching is successful, return true; otherwise false
  return !mappings.empty();
}

/*
 * @brief: get all fusion nodes matched
 * @param [in] mapping: fusion node group
 * @return std::vector<ge::NodePtr>: all fusion nodes list
 */
std::vector<ge::NodePtr> PatternFusionBasePass::GetNodesFromMapping(const Mapping &mapping) const {
  std::vector<ge::NodePtr> nodes;
  for (auto &item : mapping) {
    for (const auto &node : item.second) {
      nodes.push_back(node);
    }
  }
  return nodes;
}

/**
 * @ingroup fe
 * @brief get an op from mapping according to ID
 */
ge::NodePtr PatternFusionBasePass::GetNodeFromMapping(const std::string &id, const Mapping &mapping) const {
  for (auto &item : mapping) {
    const std::shared_ptr<OpDesc> op_desc = item.first;
    if ((op_desc != nullptr) && (op_desc->id == id)) {
      if (item.second.empty()) {
        return nullptr;
      } else {
        return item.second[0];
      }
    }
  }
  return nullptr;
}

void PatternFusionBasePass::StoreOriginOpNames(const Mapping &mapping,
                                               std::vector<std::string> &origin_op_names) const {
  for (const auto &item : mapping) {
    if (item.second.empty()) {
      continue;
    }
    for (const auto &node : item.second) {
      origin_op_names.push_back(node->GetOpDesc()->GetName());
    }
  }
}

void PatternFusionBasePass::RecordOutputAnchorMap(ge::NodePtr output_node) {
  for (const auto &output_anchor : output_node->GetAllOutDataAnchors()) {
    if (output_anchor == nullptr) {
      continue;
    }

    for (const auto &peer_in_anchor : output_anchor->GetPeerInDataAnchors()) {
      if (peer_in_anchor == nullptr) {
        continue;
      }

      // Record anchor map
      const auto iter = origin_op_anchors_map_.find(output_node);
      if (iter == origin_op_anchors_map_.end()) {
        std::map<ge::InDataAnchorPtr, ge::OutDataAnchorPtr> anchorMap;
        anchorMap[peer_in_anchor] = output_anchor;
        (void)origin_op_anchors_map_.emplace(make_pair(output_node, anchorMap));
      } else {
        (void)iter->second.emplace(make_pair(peer_in_anchor, output_anchor));
      }
    }
  }
}

void PatternFusionBasePass::ClearOutputAnchorMap() { origin_op_anchors_map_.clear(); }
}  // namespace fe

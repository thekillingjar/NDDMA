/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/ub_fusion/base_buffer_fusion_pass_runner.h"
#include "common/scope_allocator.h"
#include "common/op_info_common.h"
#include "common/platform_utils.h"
#include "common/calc_slice_utils.h"
#include "common/util/trace_manager/trace_manager.h"

namespace fe {
namespace {
constexpr char const *kAttrAutoFusionPattern = "_auto_fusion_pattern";
constexpr char const *kAiCore = "AiCore";
const std::unordered_set<std::string> kNoneOpTypeVec = {"StridedRead", "StridedWrite"};
bool CompareByNodeId(const ge::NodePtr &left_node, const ge::NodePtr &right_node) {
  if (left_node == nullptr || right_node == nullptr) {
    return false;
  }
  return left_node->GetOpDesc()->GetId() < right_node->GetOpDesc()->GetId();
}
}

BaseBufferFusionPassRunner::BaseBufferFusionPassRunner(const std::string &pass_name,
    BufferFusionPassBase *(*create_fn)(), const FusionCycleDetectorPtr &fusion_cycle_detector)
    : pass_name_(pass_name), fusion_cycle_detector_(fusion_cycle_detector), is_fusion_check_(false) {
  buffer_fusion_pass_base_ptr_ = std::unique_ptr<BufferFusionPassBase>(create_fn());
  buffer_fusion_pass_base_ptr_->SetName(pass_name);
}

BaseBufferFusionPassRunner::BaseBufferFusionPassRunner(const std::string &pass_name,
    BufferFusionPassBase *(*create_fn)(), const FusionCycleDetectorPtr &fusion_cycle_detector,
    const OpStoreAdapterBasePtr &op_store_adapter_ptr) :
                                pass_name_(pass_name), fusion_cycle_detector_(fusion_cycle_detector),
                                is_fusion_check_(true), op_store_adapter_ptr_(op_store_adapter_ptr) {
  buffer_fusion_pass_base_ptr_ = std::unique_ptr<BufferFusionPassBase>(create_fn());
  buffer_fusion_pass_base_ptr_->SetName(pass_name);
}

BaseBufferFusionPassRunner::BaseBufferFusionPassRunner(const std::string &pass_name,
    BufferFusionPassBase *(*create_fn)(), const FusionCycleDetectorPtr &fusion_cycle_detector,
    const bool is_fusion_check, const OpStoreAdapterBasePtr &op_store_adapter_ptr) :
                                pass_name_(pass_name), fusion_cycle_detector_(fusion_cycle_detector),
                                is_fusion_check_(is_fusion_check), op_store_adapter_ptr_(op_store_adapter_ptr) {
  buffer_fusion_pass_base_ptr_ = std::unique_ptr<BufferFusionPassBase>(create_fn());
  buffer_fusion_pass_base_ptr_->SetName(pass_name);
}

BaseBufferFusionPassRunner::~BaseBufferFusionPassRunner() {}

Status BaseBufferFusionPassRunner::Run(const ge::ComputeGraph &graph) {
  // 1. get fusion pattern
  std::vector<BufferFusionPattern *> patterns;
  GetFusionPatterns(patterns);
  if (patterns.empty()) {
    FE_LOGD("The defined buffer fusion pass pattern [%s] is empty.", GetPassName().c_str());
    return SUCCESS;
  }
  if (is_fusion_check_) {
    BackUpCycleDetector(); // back up cycle data before match pattern if do fusion check
  }
  // 2. get fusion nodes map by all patterns
  // match result, scope id - nodes
  std::map<int64_t, std::vector<ge::NodePtr>> match_nodes_map;
  for (BufferFusionPattern *pattern : patterns) {
    if (pattern == nullptr) {
      continue;
    }
    if (pattern->GetErrorCnt() > 0) {
      FE_LOGW("Params of pattern [%s] is invalid, count is [%ld].",
              pattern->GetName().c_str(), pattern->GetErrorCnt());
      continue;
    }
    ge::TraceOwnerGuard guard(FE_MODULE_NAME, pattern->GetName(), graph.GetName());
    // 3. get matched nodes by each pattern
    if (MatchEachPattern(graph, *pattern, match_nodes_map) != SUCCESS) {
      FE_LOGW("Failed to match nodes with pattern [%s] in fusion pass [%s].",
              pattern->GetName().c_str(), GetPassName().c_str());
      continue;
    }
  }

  // 4. release fusion patterns
  ReleaseFusionPatterns(patterns);

  // 5. fusion check
  if (is_fusion_check_) {
    // compile check
    bool has_rollback_fusion = false;
    if (FusionCheck(match_nodes_map, has_rollback_fusion) != SUCCESS) {
      return FAILED;
    }
    // if existed rollback fusion, restore the data before match pattern and update it with the fusion nodes
    if (has_rollback_fusion) {
      RollbackCycleDetector(graph, match_nodes_map);
    }
  }

  // 6. calc slice info
  bool is_set_slice_info = true;
  (void)ge::AttrUtils::GetBool(graph, kNeedSetSliceInfo, is_set_slice_info);
  if (is_set_slice_info) {
    for (std::pair<const int64_t, std::vector<ge::NodePtr>> &fusion_nodes_pair : match_nodes_map) {
      CalcSliceInfo(fusion_nodes_pair.second);
    }
  }
  // 7. set pass name attr
  for (const std::pair<const int64_t, std::vector<ge::NodePtr>> &fusion_nodes_pair : match_nodes_map) {
    SetFusionNodesPassNameAttr(fusion_nodes_pair.second);
  }
  return SUCCESS;
}

void BaseBufferFusionPassRunner::GetFusionPatterns(std::vector<BufferFusionPattern *> &patterns) const {
  if (buffer_fusion_pass_base_ptr_ == nullptr) {
    return;
  }
  patterns = buffer_fusion_pass_base_ptr_->DefinePatterns();
}

void BaseBufferFusionPassRunner::ReleaseFusionPatterns(std::vector<BufferFusionPattern *> &patterns) {
  for (BufferFusionPattern *pattern : patterns) {
    if (pattern != nullptr) {
      delete (pattern);
      pattern = nullptr;
    }
  }
}

bool BaseBufferFusionPassRunner::IsNodeFusible(const ge::NodePtr &node) const {
  if (node == nullptr) {
    return false;
  }
  if (ScopeAllocator::Instance().HasScopeAttr(node->GetOpDesc())) {
    return false;
  }
  if (!IsTbeOp(node)) {
    return false;
  }
  if (IsThread1Node(node)) {
    return false;
  }
  if (IsDumpableOp(node->GetOpDesc())) {
    FE_LOGD("Op[%s, %s] has been marked as dumpable.", node->GetNamePtr(), node->GetTypePtr());
    return false;
  }
  return true;
}

bool BaseBufferFusionPassRunner::IsNodePatternMatched(const ge::NodePtr &node,
                                                      const std::vector<std::string> &patterns) {
  if (node == nullptr) {
    return false;
  }
  std::string op_pattern;
  (void)ge::AttrUtils::GetStr(node->GetOpDesc(), kOpPattern, op_pattern);
  if (op_pattern.empty()) {
    return false;
  }
  return std::find(patterns.begin(), patterns.end(), op_pattern) != patterns.end();
}

const BufferFusionOpDesc* BaseBufferFusionPassRunner::GetMatchedFusionDesc(const ge::NodePtr &node,
                                                                           const BufferFusionPattern &pattern,
                                                                           const BufferFusionMapping &mapping) {
  for (const BufferFusionOpDesc *op_desc : pattern.GetOpDescs()) {
    if (op_desc == nullptr) {
      continue;
    }
    if (!IsNodePatternMatched(node, op_desc->types)) {
      continue;
    }
    // if repeate_max is -1, we treat it as unlimite
    if (op_desc->repeate_max > 0) {
      auto iter = mapping.find(op_desc);
      if (iter != mapping.end() && iter->second.size() >= static_cast<size_t>(op_desc->repeate_max)) {
        continue;
      }
    }
    return op_desc;
  }
  return nullptr;
}

void BaseBufferFusionPassRunner::AddNodeToMapping(const ge::NodePtr &node, const BufferFusionOpDesc *op_desc,
                                                  BufferFusionMapping &mapping) {
  auto iter = mapping.find(op_desc);
  if (iter != mapping.end()) {
    iter->second.push_back(node);
  } else {
    std::vector<ge::NodePtr> node_vec = {node};
    mapping.emplace(op_desc, node_vec);
  }
}

UBFusionType BaseBufferFusionPassRunner::GetUBFusionType(const BufferFusionMapping &mapping) const {
  CubeVecStateNew cv_state = PlatformUtils::Instance().GetCubeVecState();
  ISAArchVersion isa_arch_version = PlatformUtils::Instance().GetIsaArchVersion();
  std::vector<ge::NodePtr> fusion_nodes = buffer_fusion_pass_base_ptr_->GetMatchedNodes(mapping);
  bool cube_vector_both_exist = IsCubeVecAllExist(fusion_nodes);
  /*
   * In 310B scene, If CV arch is split and cube vector op all exists, no need fuse
   */
  if (isa_arch_version == ISAArchVersion::EN_ISA_ARCH_V300) {
    if (cv_state == CubeVecStateNew::CUBE_VEC_SPLIT && cube_vector_both_exist) {
      return UBFusionType::FUSION_TYPE_NONE;
    }
    return UBFusionType::FUSION_TYPE_UB;
  }
  if (cv_state == CubeVecStateNew::CUBE_VEC_FUSE || cv_state == CubeVecStateNew::CUBE_VEC_UNKNOWN) {
    return UBFusionType::FUSION_TYPE_UB;
  }
  if (cv_state == CubeVecStateNew::CUBE_VEC_SPLIT && PlatformUtils::Instance().IsHardwareSupportCoreSync()) {
    if (cube_vector_both_exist) {
      return UBFusionType::FUSION_TYPE_MIXL2;
    }
  }
  return UBFusionType::FUSION_TYPE_UB;
}

bool BaseBufferFusionPassRunner::IsCubeVecAllExist(const vector<ge::NodePtr> &fusion_nodes) {
  bool find_cube_op = false;
  bool find_vector_op = false;
  for (const auto &node : fusion_nodes) {
    if (node == nullptr || node->GetType() == TRANSDATA) {
      continue;
    }
    if (kNoneOpTypeVec.count(node->GetType()) > 0U) {
      return false;
    }
    if (find_cube_op && find_vector_op) {
      break;
    }
    std::string core_type;
    (void)ge::AttrUtils::GetStr(node->GetOpDesc(), ATTR_NAME_CUBE_VECTOR_CORE_TYPE, core_type);
    if (core_type == kAiCore) {
      find_cube_op = true;
    } else {
      find_vector_op = true;
    }
  }

  return find_cube_op && find_vector_op;
}

Status BaseBufferFusionPassRunner::GetFusionNodesByMapping(const ge::NodePtr &first_node,
                                                           const BufferFusionMapping &mapping,
                                                           std::vector<ge::NodePtr> &fusion_nodes) const {
  UBFusionType fusion_type = GetUBFusionType(mapping);
  Status status = SUCCESS;
  if (fusion_type == UBFusionType::FUSION_TYPE_UB) {
    status = buffer_fusion_pass_base_ptr_->GetFusionNodes(mapping, fusion_nodes);
  } else if (fusion_type == UBFusionType::FUSION_TYPE_MIXL2) {
    status = buffer_fusion_pass_base_ptr_->GetMixl2FusionNodes(mapping, fusion_nodes);
    bool is_ffts_plus = false;
    (void)ge::AttrUtils::GetBool(first_node->GetOpDesc(), kTypeFFTSPlus, is_ffts_plus);
    if (!is_ffts_plus && kMixL2PassName.count(pass_name_) > 0U) {
      fusion_nodes.clear();
      status = buffer_fusion_pass_base_ptr_->GetFusionNodes(mapping, fusion_nodes);
    }
  }
  return status;
}

void BaseBufferFusionPassRunner::CalcSliceInfo(std::vector<ge::NodePtr> &fusion_nodes) const {
  CalcSliceUtils::CalcSliceInfo(buffer_fusion_pass_base_ptr_, fusion_nodes);
}

int64_t BaseBufferFusionPassRunner::SetFusionNodesScopeId(const std::vector<ge::NodePtr> &fusion_nodes) {
  int64_t scope_id = ScopeAllocator::Instance().AllocateScopeId();
  for (const ge::NodePtr &node : fusion_nodes) {
    if (node == nullptr) {
      continue;
    }
    (void)ScopeAllocator::SetScopeAttr(node->GetOpDesc(), scope_id);
  }
  FE_LOGD("Set scope id [%ld] attribute for fusion nodes [%s].", scope_id, GetFusionNodesDescStr(fusion_nodes).c_str());
  return scope_id;
}

void BaseBufferFusionPassRunner::SetSingleOpUbPassNameAttr(const ge::NodePtr &node) const {
  if (node == nullptr) {
    return;
  }
  std::vector<string> single_op_pass_name;
  (void)ge::AttrUtils::GetListStr(node->GetOpDesc(), kSingleOpUbPassNameAttr, single_op_pass_name);
  single_op_pass_name.emplace_back(pass_name_);
  (void)ge::AttrUtils::SetListStr(node->GetOpDesc(), kSingleOpUbPassNameAttr, single_op_pass_name);
  FE_LOGD("Add single op pass name[%s] for node[%s, %s].", pass_name_.c_str(), node->GetNamePtr(), node->GetTypePtr());
}

void BaseBufferFusionPassRunner::SetFusionNodesPassNameAttr(const std::vector<ge::NodePtr> &fusion_nodes) const {
  for (const ge::NodePtr &node : fusion_nodes) {
    if (node == nullptr) {
      continue;
    }
    (void)ge::AttrUtils::SetStr(node->GetOpDesc(), kPassNameUbAttr, pass_name_);
    FE_LOGD("Node[%s]: set pass_name[%s] successfully.", node->GetName().c_str(), pass_name_.c_str());
  }
}

bool BaseBufferFusionPassRunner::IsNodeDataFlowConnected(const ge::NodePtr &node_a, const ge::NodePtr &node_b) const {
  if (fusion_cycle_detector_ == nullptr) {
    return false;
  }
  return fusion_cycle_detector_->IsDataFlowConnected(node_a, node_b);
}

Status BaseBufferFusionPassRunner::UpdateCycleDetector(const ge::ComputeGraph &graph,
                                                       const vector<ge::NodePtr> &fusion_nodes) {
  FE_CHECK_NOTNULL(fusion_cycle_detector_);
  return fusion_cycle_detector_->UpdateConnectionMatrix(graph, fusion_nodes);
}

void BaseBufferFusionPassRunner::BackUpCycleDetector() {
  if (fusion_cycle_detector_ == nullptr) {
    return;
  }
  fusion_cycle_detector_->BackupConnectionMatrix();
}

void BaseBufferFusionPassRunner::RestoreCycleDetector() {
  if (fusion_cycle_detector_ == nullptr) {
    return;
  }
  fusion_cycle_detector_->RestoreConnectionMatrix();
}

void BaseBufferFusionPassRunner::GetFusionNodesMap(const ge::ComputeGraph &graph,
                                                   std::map<int64_t, std::vector<ge::NodePtr>> &fusion_nodes_map) {
  for (const ge::NodePtr &node : graph.GetDirectNode()) {
    if (node == nullptr) {
      continue;
    }
    int64_t scope_id = 0;
    if (!ScopeAllocator::Instance().GetScopeAttr(node->GetOpDesc(), scope_id)) {
      continue;
    }
    auto iter = fusion_nodes_map.find(scope_id);
    if (iter != fusion_nodes_map.end()) {
      iter->second.push_back(node);
    } else {
      std::vector<ge::NodePtr> nodes_vec = { node };
      fusion_nodes_map.emplace(scope_id, nodes_vec);
    }
  }
}

void BaseBufferFusionPassRunner::TopoSortForFusionNodes(std::vector<ge::NodePtr> &fusion_nodes) {
  if (fusion_nodes.empty()) {
    return;
  }
  std::sort(fusion_nodes.begin(), fusion_nodes.end(), CompareByNodeId);
}

Status BaseBufferFusionPassRunner::FusionCheck(std::map<int64_t, std::vector<ge::NodePtr>> &match_nodes_map,
                                               bool &has_rollback_fusion) const {
  if (op_store_adapter_ptr_ == nullptr) {
    FE_LOGE("Op compiler is null.");
    return FAILED;
  }
  std::vector<ge::NodePtr> compile_failed_nodes;
  CompileInfoParam compile_info(compile_failed_nodes);
  for (const std::pair<const int64_t, std::vector<ge::NodePtr>> &nodes_pair : match_nodes_map) {
    std::vector<ge::Node *> nodes;
    for (const ge::NodePtr &node : nodes_pair.second) {
      (void)ge::AttrUtils::SetBool(node->GetOpDesc(), kAttrAutoFusionPattern, true);
      nodes.push_back(node.get());
    }
    compile_info.fusion_nodes_map.emplace(nodes_pair.first, nodes);
  }
  compile_info.is_fusion_check = true;
  if (compile_info.fusion_nodes_map.empty()) {
    FE_LOGI("No node in graph need to compile.");
  } else {
    Status status = op_store_adapter_ptr_->CompileOp(compile_info);
    if (status != SUCCESS) {
      REPORT_FE_ERROR("Failed to check if nodes can be fused during the buffer fusion pass matching.");
      return status;
    }
  }
  // atfer compile clear attr, clear auto fusion pattern attr if node does not have scope id attr
  for (auto iter = match_nodes_map.begin(); iter != match_nodes_map.end();) {
    bool fusion_check = true;
    for (const ge::NodePtr &node : iter->second) {
      if (!ScopeAllocator::Instance().HasScopeAttr(node->GetOpDesc())) {
        fusion_check = false;
        node->GetOpDesc()->DelAttr(kAttrAutoFusionPattern);
      }
    }
    if (!fusion_check) {
      FE_LOGD("Fusion nodes [%s] with scope ID [%ld] cannot be fused after the fusion check.",
              GetFusionNodesDescStr(iter->second).c_str(), iter->first);
      iter = match_nodes_map.erase(iter);
      has_rollback_fusion = true;
      continue;
    }
    iter++;
  }
  return SUCCESS;
}

void BaseBufferFusionPassRunner::RollbackCycleDetector(const ge::ComputeGraph &graph,
    const std::map<int64_t, std::vector<ge::NodePtr>> &fusion_nodes_map) {
  RestoreCycleDetector();
  for (const std::pair<const int64_t, std::vector<ge::NodePtr>> &fusion_nodes : fusion_nodes_map) {
    (void)UpdateCycleDetector(graph, fusion_nodes.second);
  }
}

Status BaseBufferFusionPassRunner::CheckMatchedNodesCanFusion(const BufferFusionNodeDescMap &fusion_nodes,
                                                              const ge::NodePtr &next_node) {
  return buffer_fusion_pass_base_ptr_->CheckNodeCanFusion(fusion_nodes, next_node);
}

Status BaseBufferFusionPassRunner::GetFusionNodes(const BufferFusionMapping &mapping,
                                                  std::vector<ge::NodePtr> &fusion_nodes) const {
  FE_CHECK_NOTNULL(buffer_fusion_pass_base_ptr_);
  return buffer_fusion_pass_base_ptr_->GetFusionNodes(mapping, fusion_nodes);
}

const std::string& BaseBufferFusionPassRunner::GetPassName() const {
  return pass_name_;
}

FusionCycleDetectorPtr BaseBufferFusionPassRunner::GetFusionCycleDetectorPtr() const {
  return fusion_cycle_detector_;
}
}

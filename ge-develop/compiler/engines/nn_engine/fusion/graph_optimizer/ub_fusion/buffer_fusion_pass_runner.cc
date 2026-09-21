/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/ub_fusion/buffer_fusion_pass_runner.h"
#include <queue>
#include "common/fe_log.h"
#include "common/configuration.h"
#include "common/fusion_statistic/fusion_statistic_writer.h"
#include "common/op_info_common.h"
#include "platform/platform_info.h"
#include "common/platform_utils.h"
#include "common/scope_allocator.h"
#include "register/graph_optimizer/fusion_common/graph_pass_util.h"
#include "common/util/trace_manager/trace_manager.h"
#include "common/sgt_slice_type.h"
#include "register/graph_optimizer/fusion_common/unknown_shape_utils.h"
namespace fe {
namespace {
const char *STREAM_LABEL = "_stream_label";
const int32_t kInvalidShapeRuleTypeIndex = -1;
constexpr size_t kMaxRecursiveTimes = 10000;
// some op must be fused, do not belong to vector
const std::unordered_set<std::string> kNoneOpTypeSet = {"StridedRead", "StridedWrite"};
size_t GetMappingSize(const BufferFusionMapping &mapping) {
  size_t mapping_size = 0;
  for (const auto &ele : mapping) {
    mapping_size += ele.second.size();
  }
  return mapping_size;
}

size_t GetDistinctNodeSize(const BufferFusionMapping &mapping) {
  std::set<ge::NodePtr> distinct_nodes;
  for (const auto &ele : mapping) {
    for (const auto &node : ele.second) {
      distinct_nodes.emplace(node);
    }
  }
  return distinct_nodes.size();
}

void PrintMapping(const BufferFusionMapping &mapping, string stage = "") {
  if (CheckLogLevel(FE, DLOG_DEBUG) == 1) {
    FE_LOGD("Print mapping at stage [%s]:", stage.c_str());

    for (auto &ele : mapping) {
      for (auto &node : ele.second) {
        FE_LOGD("[%s] : [%s]", ele.first->desc_name.c_str(),
                node->GetName().c_str());
      }
    }
  }
}

string AssmbleDescNames(const vector<BufferFusionOpDesc *> &curr_queue_descs) {
  string node_name;
  for (auto &desc : curr_queue_descs) {
    node_name += desc->desc_name;
    node_name += ", ";
  }
  return node_name;
}

Status SaveMatchResult(bool not_any_and_output_op, BufferFusionOpDesc *out_desc,
                       const ge::NodePtr &out_node, std::map<BufferFusionOpDesc *, bool> &usage_flags,
                       MatchInfo& m_info) {
  auto out_node_name = out_node->GetName();
  if (not_any_and_output_op) {
    m_info.m_st.queue_nodes.push_back(out_node);
    m_info.m_st.queue_descs.push_back(out_desc);
  }

  /* If the desc is TBE_PATTERN_OUTPUT_NODE or TBE_PATTERN_OP_TYPE_ANY,
   * the op_desc should also be inserted into matched output nodes.
   * Because the TBE_PATTERN_OUTPUT_NODE and TBE_PATTERN_OP_TYPE_ANY will
   * always be matched and if previous path is valid. When the father node
   * is multi-output, for example:
   *              convolution
   *                   |
   *                   |
   *              element-wise (normal matched op desc)
   *                  /\
   *                 /  \
   *                /    \
   *      output-node   quant(which maybe optional)
   *
   * First, we match output-node and than in function GetCurrMatchStatus
   * the output-node will be considered as matched. Then
   * we recover the matching queue with op_desc element-wise.
   *
   * Then if the output-node is not recorded in the matched_output_nodes,
   * we will still match the output-node again.
   *
   * Finally, the matching and recovering will be done infinitely.
   *
   * So, here we should put every matched node into matched_output_nodes.
   * */
  auto it = m_info.matched_output_nodes.find(out_desc->desc_name);
  if (it != m_info.matched_output_nodes.end()) {
    (it->second)[out_desc->repeate_curr].push_back(out_node_name);
  } else {
    std::map<int32_t, std::vector<std::string>> temp;
    temp.insert(std::pair<int32_t, std::vector<std::string>>(out_desc->repeate_curr, {out_node_name}));
    m_info.matched_output_nodes.insert(
        std::pair<std::string, std::map<int32_t, std::vector<std::string>>>(out_desc->desc_name, temp));
  }

  // add fusioned node to mapping
  m_info.m_st.m_rs.mapping[out_desc].push_back(out_node);
  // repeat desc need to plus while has been matched
  if (CheckInt64AddOverflow(out_desc->repeate_curr, 1) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][UbFusion][MtcFollowNd] repeateCurr++ overflow. (out_desc:%s)",
                    out_desc->desc_name.c_str());
    return FAILED;
  }
  out_desc->repeate_curr++;
  usage_flags[out_desc] = true;
  return SUCCESS;
}

bool GetOutputs(BufferFusionOpDesc *op_desc, std::vector<BufferFusionOpDesc *> &outputs,
                bool ignore_repeat = false) {
  if (op_desc == nullptr) {
    FE_LOGW("[GetOutputs][Check] op_desc is null.");
    return false;
  }

  // add curr desc can be reused while repeat_curr < repeate_max
  if ((!ignore_repeat) && (op_desc->repeate_curr < op_desc->repeate_max)) {
    outputs.push_back(op_desc);
  }

  // check candidate desc
  for (auto desc : op_desc->outputs) {
    if (desc == nullptr) {
      continue;
    }

    if (desc->repeate_curr >= desc->repeate_max) {
      continue;
    }
    // add out desc
    outputs.push_back(desc);

    // add sub out_descs while repeate_min == 0
    if (desc->repeate_min == 0) {
      std::vector<BufferFusionOpDesc *> sub_output;
      if (GetOutputs(desc, sub_output, true)) {
        for (const auto &sub_desc : sub_output) {
          outputs.push_back(sub_desc);
        }
      }
    }
  }

  return true;
}
} // namespace

BufferFusionPassRunner::BufferFusionPassRunner(const string &pass_name, BufferFusionPassBase *(*create_fn)(),
  const FusionCycleDetectorPtr &cycle_detector, const OpStoreAdapterBasePtr &op_store_adapter_ptr,
  const bool is_fusion_check, const BufferFusionOptimizerPtr &buffer_fusion_optimizer)
  : BaseBufferFusionPassRunner(pass_name, create_fn, cycle_detector, is_fusion_check, op_store_adapter_ptr),
    buffer_fusion_optimizer_(buffer_fusion_optimizer) {}

BufferFusionPassRunner::BufferFusionPassRunner(const string &pass_name, BufferFusionPassBase *(*create_fn)(),
  const FusionCycleDetectorPtr &cycle_detector, const BufferFusionOptimizerPtr &buffer_fusion_optimizer)
  : BaseBufferFusionPassRunner(pass_name, create_fn, cycle_detector),
    buffer_fusion_optimizer_(buffer_fusion_optimizer) {}

BufferFusionPassRunner::~BufferFusionPassRunner() {}

bool BufferFusionPassRunner::IsOpTypeAny(const std::vector<string> &types) const {
  return find(types.begin(), types.end(), TBE_PATTERN_OP_TYPE_ANY) != types.end();
}

bool BufferFusionPassRunner::IsOutputNode(const std::vector<string> &types) const {
  return find(types.begin(), types.end(), TBE_PATTERN_OUTPUT_NODE) != types.end();
}

int32_t BufferFusionPassRunner::GetPatternTypeIndex(const vector<string> &types, const string &op_type) const {
  auto iter = find(types.begin(), types.end(), op_type);
  if (iter == types.end()) {
    return kInvalidShapeRuleTypeIndex;
  }
  return distance(types.begin(), iter);
}

/*
 * @brief: check whether graph node is matched with pattern desc
 * @param [in] node: graph node
 * @param [in] op_desc: candidated pattern desc
 * @return bool: check result
 */
bool BufferFusionPassRunner::IsOpTypeExist(const ge::NodePtr node, const BufferFusionOpDesc *op_desc,
                                           int32_t &pattern_type_index) const {
  string op_type;
  FE_CHECK(node == nullptr || op_desc == nullptr, FE_LOGW("Unexpected null pointer."), return false);
  string name = node->GetName();
  const std::vector<string> types = op_desc->types;
  bool res = GetOpAttrType(node, op_type, op_desc->not_pattern);
  FE_LOGD("op type(pattern) of %s is %s.", name.c_str(), op_type.c_str());
  if (!res) {
      pattern_type_index = GetPatternTypeIndex(types, TBE_PATTERN_OUTPUT_NODE);
      if (pattern_type_index != kInvalidShapeRuleTypeIndex) {
      FE_LOGD("Node [%s] is an output node.", node->GetName().c_str());
      return true;
    } else {
      FE_LOGD("Node [%s] is not an output node.", node->GetName().c_str());
      return false;
    }
  }

  pattern_type_index = GetPatternTypeIndex(types, op_type);
  if (pattern_type_index != kInvalidShapeRuleTypeIndex) {
    FE_LOGD("Node:%s, Type:%s, Match Op Type, type index:%d", name.c_str(), op_type.c_str(), pattern_type_index);
    return true;
  }
  pattern_type_index = GetPatternTypeIndex(types, TBE_PATTERN_OP_TYPE_ANY);
  if (pattern_type_index != kInvalidShapeRuleTypeIndex) {
    FE_LOGD("Node: %s, Type: %s, Match Operation Pattern: ANY, Type Index: %d", name.c_str(), op_type.c_str(), pattern_type_index);
    return true;
  }
  pattern_type_index = GetPatternTypeIndex(types, TBE_PATTERN_OUTPUT_NODE);
  if (pattern_type_index != kInvalidShapeRuleTypeIndex) {
    FE_LOGD("Node: %s, Type: %s, Match Operation Pattern: OUTNODE, Type Index: %d",
            name.c_str(), op_type.c_str(), pattern_type_index);
    return true;
  }
  return false;
}

/*
 * @brief: check whether node output size is same with candidate desc output
 * size
 * @param [in] node: graph node
 * @param [in] op_desc: candidated pattern desc
 * @return bool: check result
 */
bool BufferFusionPassRunner::SkipDiffSizeDesc(ge::NodePtr node, const BufferFusionOpDesc *op_desc,
                                              const string &pattern_name) const {
  (void) pattern_name;
  FE_CHECK(node == nullptr, REPORT_FE_ERROR("[SubGraphOpt][UbFusion][SkipDiffSizeDesc] Node is null."), return false);
  FE_CHECK(op_desc == nullptr,
           REPORT_FE_ERROR("[SubGraphOpt][UbFusion][SkipDiffSizeDesc] opDesc is null."), return false);

  // single output node match single desc, and binary node match binary desc
  if (node->GetOutDataNodes().size() == 1 && op_desc->out_branch_type == TBE_OUTPUT_BRANCH_MULTI) {
    FE_LOGD("Node[%s]: The size of out_data_nodes is 1, but the out_brand_type is TBE_OUTPUT_BRANCH_MULTI; skipping.",
            node->GetName().c_str());
    return true;
  }

  if (node->GetOutDataNodes().size() > 1 && op_desc->out_branch_type == TBE_OUTPUT_BRANCH_SINGLE) {
    if (op_desc->ignore_output_num) {
      return false;
    }

    FE_LOGD("Skip node[%s]: the size of out data nodes is > 1, but desc %s's out type is TBE_OUTPUT_BRANCH_SINGLE.",
            node->GetName().c_str(), op_desc->desc_name.c_str());
    return true;
  }

  return false;
}

bool BufferFusionPassRunner::CheckPatternOpTypeMatch(ge::NodePtr node, const BufferFusionOpDesc *op_desc,
                                                     const int32_t shape_rule_type_index) const {
  if (node == nullptr || op_desc == nullptr) {
    return false;
  }
  bool is_dyn_op = UnknownShapeUtils::IsUnknownShapeOp(*(node->GetOpDesc()));
  if (is_dyn_op && Configuration::Instance(AI_CORE_NAME).GetJitCompileCfg() == JitCompileCfg::CFG_AUTO) {
    return false;
  }
  bool is_match = false;

  // types in BufferFusionOpDesc support (n:1 or n:n), should choose the corrent one
  ShapeTypeRule shape_type_rule = op_desc->shape_type_rules[0];
  if (op_desc->shape_type_rules.size() > 1 &&
      shape_rule_type_index < static_cast<int32_t>(op_desc->shape_type_rules.size())) {
    shape_type_rule = op_desc->shape_type_rules[shape_rule_type_index];
  }
  switch (shape_type_rule) {
    case IGNORE_SHAPE_TYPE:
      is_match = true;
      break;
    case ONLY_SUPPORT_STATIC:
      is_match = !is_dyn_op;
      break;
    case ONLY_SUPPORT_DYNAMIC:
      is_match = is_dyn_op;
      break;
    default:
      break;
  }

  FE_LOGD("Node[%s, %s] shape is %s and pattern rule is %s.", node->GetName().c_str(),
          node->GetType().c_str(), (is_dyn_op ? "Dynamic" : "Static"),
          (kShapeTypeRuleToStr.find(shape_type_rule) != kShapeTypeRuleToStr.end() ?
            kShapeTypeRuleToStr.find(shape_type_rule)->second.c_str() : "Unknown"));
  return is_match;
}

/*
 * @brief: get current loop fusiton match status
 * @param [in] is_parallel: graph node is multi branch or single branch
 * @param [in] op_descs: candidated pattern desc
 * @param [in] usage: record whether desc has beed matched
 * @return bool: all current loop descs have beed matched or not
 */
bool BufferFusionPassRunner::GetCurrMatchStatus(const BufferFusionOpDesc *op_desc, size_t out_node_size,
                                                const std::vector<BufferFusionOpDesc *> &op_descs,
                                                std::map<BufferFusionOpDesc *, bool> &usage) const {
  bool is_parallel = !op_desc->ignore_output_num && out_node_size > 1;
  bool match_status = false;

  // check match status
  if (is_parallel) {
    match_status = true;
    for (const auto &desc : op_descs) {
      if (usage.find(desc) != usage.end()) {
        if (!usage[desc]) {
          match_status = false;
          break;
        }
      }
    }
  } else {
    match_status = false;
    for (const auto &desc : op_descs) {
      if (usage.find(desc) != usage.end()) {
        if (usage[desc]) {
          match_status = true;
          break;
        }
      }
    }
  }

  return match_status;
}

/*
 * @brief: get pattern fusiton match status
 * @param [in] pattern: fusion pattern info
 * @return bool: the pattern has beed matched or not
 */
bool BufferFusionPassRunner::GetPatternMatchStatus(BufferFusionPattern &pattern) const {
  std::map<int64_t, bool> group_status;
  // find same group desc match status
  for (auto desc : pattern.GetOpDescs()) {
    if (desc->types[0] == TBE_PATTERN_INPUT_NODE) {
      continue;
    }
    if (desc->group_id == TBE_PATTERN_GROUPID_INVALID) {
      continue;
    }
    if (group_status.find(desc->group_id) == group_status.end()) {
      group_status[desc->group_id] = false;
    }
    if (desc->repeate_curr >= desc->repeate_min) {
      group_status[desc->group_id] = true;
    }
  }
  // find all pattern descs matched status
  bool status = true;
  for (auto desc : pattern.GetOpDescs()) {
    if (desc->types[0] == TBE_PATTERN_INPUT_NODE) {
      continue;
    }
    if (desc->group_id != TBE_PATTERN_GROUPID_INVALID) {
      if (!group_status[desc->group_id]) {
        FE_LOGD("group[%ld] not match", desc->group_id);
        status = false;
        break;
      }
    } else if (desc->repeate_curr < desc->repeate_min) {
      FE_LOGD("pattern %s did not match; description name=[%s], current match count=[%ld], minimum match count=[%ld]",
          pattern.GetName().c_str(), desc->desc_name.c_str(), desc->repeate_curr, desc->repeate_min);
      status = false;
      break;
    }
  }

  return status;
}

/*
 * @brief: get fusiton pattern head desc matched
 * @param [in] node: graph node
 * @param [in] head_descs: candidated head desc list
 * @return BufferFusionOpDesc*: head desc ptr
 */
BufferFusionOpDesc *BufferFusionPassRunner::GetMatchedHeadDesc(ge::NodePtr node, const string &pattern_name,
                                                               std::vector<BufferFusionOpDesc *> head_descs) const {
  for (auto desc : head_descs) {
    int32_t pattern_type_index = kInvalidShapeRuleTypeIndex;
    if (!IsOpTypeExist(node, desc, pattern_type_index)) {
      continue;
    }
    if (SkipDiffSizeDesc(node, desc, pattern_name) || !CheckPatternOpTypeMatch(node, desc, pattern_type_index)) {
      continue;
    }
    FE_LOGD("Node [%s], desc[%s] from graph has matched to head desc from fusion pattern %s.",
            node->GetName().c_str(), desc->desc_name.c_str(), pattern_name.c_str());
    return desc;
  }
  return nullptr;
}

/*
 * @brief: get current loop desc matched
 * @param [in] node: graph node
 * @param [in] head_descs: valid head desc
 * @param [in] usage: record whether desc has beed matched
 * @return BufferFusionOpDesc*: matched desc ptr
 */
BufferFusionOpDesc *BufferFusionPassRunner::GetMatchedNormalDesc(
    const ge::NodePtr &node, const std::vector<BufferFusionOpDesc *> &descs,
    std::map<BufferFusionOpDesc *, bool> &usage, MatchInfo &m_info) {
  /* If node cannot match any concrete pattern, we use the
   * lower priority patterns such as "OutputData" or "OpTypeAny" */
  BufferFusionOpDesc *lower_prior_desc = nullptr;
  std::string node_name = node->GetName();

  for (auto out_desc : descs) {
    FE_LOGD("Check whether description %s and node %s are matched. Usage: %d.",
            out_desc->desc_name.c_str(), node->GetName().c_str(), usage[out_desc]);
    int32_t pattern_type_index = kInvalidShapeRuleTypeIndex;
    if (!IsOpTypeExist(node, out_desc, pattern_type_index)) {
      continue;
    }
    if (SkipNodeForNormalDesc(out_desc, node, m_info, pattern_type_index)) {
      continue;
    }

    if (!usage[out_desc]) {
      if (IsOpTypeAny(out_desc->types) || IsOutputNode(out_desc->types)) {
        lower_prior_desc = out_desc;
        continue;
      }
      FE_LOGD("Match node [name:%s, description:%s].", node_name.c_str(), out_desc->desc_name.c_str());
      return out_desc;
    }
  }

  if (lower_prior_desc != nullptr) {
    FE_LOGD("Match node [name:%s, description:%s].", node_name.c_str(), lower_prior_desc->desc_name.c_str());
  }
  return lower_prior_desc;
}

bool BufferFusionPassRunner::TempCycleDetection(const ge::NodePtr &out_node, bool &is_stacked,
                                                MatchInfo &m_info, MatchInfoPair &m_info_pair,
                                                MatchInfoStack &m_info_stack) {
  ge::NodePtr node_lead_to_cycle;
  if (CheckLoopForward(m_info.m_st.m_rs.mapping, out_node, node_lead_to_cycle)) {
    FE_LOGD("Temp cycle has been detected when matching node %s in pattern %s. Cycle from [%s], flag %d.",
            out_node->GetName().c_str(), m_info.pattern->GetName().c_str(), node_lead_to_cycle->GetName().c_str(),
            is_stacked);
    if (m_info.black_list.count(out_node) == 0) {
      /* Do not need to back track to out_node again,
       * because its father node is already in the back-tracking
       * list. */
      for (const auto &node: m_info.m_st.m_rs.nodes_lead_to_cycle) {
        if (node == node_lead_to_cycle) {
          FE_LOGD("%s has already been checked and added into the stack.", node->GetName().c_str());
          return false;
        }
      }

      PrintMapping(m_info_pair.second.m_st.m_rs.mapping, "Temp Cycle Detected");
      if (is_stacked) {
        /* If is_stacked we only expand the first element.
         * The first element are the list of black list nodes. */
        auto &stacked_m_info = m_info_stack.back();
        stacked_m_info.first.emplace_back(out_node);
        FE_LOGD("Update black list with node %s.", out_node->GetName().c_str());
      } else {
        /* Using std::move to avoid redundant copy of m_info. */
        m_info_pair.first.emplace_back(out_node);
        m_info_stack.emplace_back(std::move(m_info_pair));
        m_info.m_st.m_rs.cycle_pos.emplace_back(m_info_stack.size() - 1);
        FE_LOGD("Push this mapping into the stack. Cycle-related node: %s. Cycle position: %zu.",
                out_node->GetName().c_str(), (m_info_stack.size() - 1));
        is_stacked = true;
      }
      m_info.m_st.m_rs.nodes_lead_to_cycle.emplace_back(node_lead_to_cycle);
      return false;
    } else {
      FE_LOGD("Cannot match %s because it is in the blacklist.", out_node->GetName().c_str());
      return true;
    }
  }
  return false;
}

bool BufferFusionPassRunner::IsSgtInfoConsistant(const ge::NodePtr &consumer, const ge::NodePtr &producer) const {
  ffts::ThreadSliceMapPtr produce_slice_info_ptr = nullptr;
  ffts::ThreadSliceMapPtr consumer_slice_info_ptr = nullptr;
  produce_slice_info_ptr = producer->GetOpDesc()->TryGetExtAttr(ffts::kAttrSgtStructInfo, produce_slice_info_ptr);
  consumer_slice_info_ptr = consumer->GetOpDesc()->TryGetExtAttr(ffts::kAttrSgtStructInfo, consumer_slice_info_ptr);
  if (produce_slice_info_ptr == nullptr && consumer_slice_info_ptr == nullptr) {
    return true;
  }
  if (produce_slice_info_ptr != nullptr && consumer_slice_info_ptr != nullptr) {
    if (produce_slice_info_ptr->slice_instance_num == consumer_slice_info_ptr->slice_instance_num) {
      return true;
    }
  }
  return false;
}

void BufferFusionPassRunner::MatchFollowingNodes(ge::NodePtr node, std::vector<BufferFusionOpDesc *> &out_descs,
    std::map<BufferFusionOpDesc *, bool> &usage_flags, MatchInfo& m_info, MatchInfoStack &m_info_stack) {
  auto out_nodes = node->GetOutDataNodes();
  FE_LOGD("Match successors for node %s, successor size %zu.", node->GetName().c_str(), out_nodes.size());
  for (auto desc : out_descs) {
    usage_flags[desc] = false;
  }

  /* Store the matched info before matching following nodes of node. */
  m_info.UpdateRepTimes();
  MatchInfo m_info_back = m_info;
  MatchInfoPair m_info_pair = std::make_pair(vector<ge::NodePtr>{}, std::move(m_info_back));

  bool is_stacked = false;
  for (auto &out_node : out_nodes) {
    if (IsThread1Node(node) || IsDumpableOp(out_node->GetOpDesc())) {
      continue;
    }
    if (CheckMatchedNodesCanFusion(m_info.m_st.m_rs.matched_nodes, out_node) != SUCCESS) {
      FE_LOGD("Node [%s] cannot be fused.", out_node->GetName().c_str());
      continue;
    }
    if (!m_info.m_st.m_rs.matched_nodes.empty()) {
      if (!IsSgtInfoConsistant(m_info.m_st.m_rs.matched_nodes.begin()->first, out_node)) {
        continue;
      }
    }
    BufferFusionOpDesc *out_desc = GetMatchedNormalDesc(out_node, out_descs, usage_flags, m_info);
    if (out_desc != nullptr) {
      bool not_any_op = !IsOpTypeAny(out_desc->types);
      bool not_out_op = !IsOutputNode(out_desc->types);
      if (NeedIgnoreOp(out_node, out_desc->not_pattern) && not_any_op && not_out_op) {
        FE_LOGD("outDesc node [%s] is ignored, out_desc:%s", out_node->GetName().c_str(), out_desc->desc_name.c_str());
        continue;
      }
      /* Op with pattern TBE_PATTERN_OUTPUT_NODE will not be fused,
       * so we do not check cycles related to those output op. */
      if (not_out_op) {
        if (TempCycleDetection(out_node, is_stacked, m_info, m_info_pair, m_info_stack)) {
          continue;
        }
        m_info.m_st.m_rs.matched_nodes[out_node] = out_desc;
      }

      bool not_any_and_output_op = not_any_op && not_out_op;
      if (SaveMatchResult(not_any_and_output_op, out_desc, out_node,
                          usage_flags, m_info) != SUCCESS) {
        return;
      }

      if (m_info.m_st.queue_descs.front()->out_branch_type != TBE_OUTPUT_BRANCH_MULTI) {
        break;
      }
    } else {
      FE_LOGD("Output node [%s] ARE NOT MATCHED to any desc from fusion pattern.", out_node->GetName().c_str());
    }
  }
}

void BufferFusionPassRunner::GetExistingFusionScopes(ge::ComputeGraph &graph,
                                                     std::map<int64_t, vector<ge::NodePtr>> &fusion_scopes) const {
  for (auto &node : graph.GetDirectNode()) {
    if (ScopeAllocator::HasScopeAttr(node->GetOpDesc())) {
      int64_t scope_id = 0;
      if (!ScopeAllocator::GetScopeAttr(node->GetOpDesc(), scope_id)) {
        continue;
      }
      fusion_scopes[scope_id].push_back(node);
    }
  }
}

bool BufferFusionPassRunner::IsOptionalOutput(const BufferFusionOpDesc *desc) const {
  FE_CHECK(desc == nullptr, FE_LOGW("Unexpected null pointer."), return false);
  if (desc->out_branch_type > static_cast<int>(desc->outputs.size())) {
    FE_LOGW("%s outputs size is less than out_branch_type required, consider it as optional output.",
        desc->desc_name.c_str());
    return true;
  } else if (desc->out_branch_type == TBE_OUTPUT_BRANCH_SINGLE && desc->outputs.size() > 1) {
    for (auto out_desc : desc->outputs) {
      if (!IsOpTypeAny(out_desc->types) && !IsOutputNode(out_desc->types) && out_desc->repeate_min > 0) {
        continue;
      } else if (!IsOptionalOutput(out_desc)) {
        continue;
      }
      return true;
    }
    return false;
  } else {
    for (auto out_desc : desc->outputs) {
      if (!IsOpTypeAny(out_desc->types) && !IsOutputNode(out_desc->types) && out_desc->repeate_min > 0) {
        return false;
      } else if (!IsOptionalOutput(out_desc)) {
        return false;
      }
    }
    return true;
  }
}

bool BufferFusionPassRunner::CheckLoopForward(const BufferFusionMapping &mapping,
                                              const ge::NodePtr &target_node,
                                              ge::NodePtr &node_lead_to_cycle) {
  std::vector<ge::NodePtr> all_fuse_nodes;
  for (const auto &it : mapping) {
    if (IsOpTypeAny(it.first->types) || IsOutputNode(it.first->types)) {
      continue;
    }
    for (const auto &node : it.second) {
      all_fuse_nodes.push_back(node);
    }
  }

  for (auto it = mapping.begin(); it != mapping.end(); it++) {
    for (const auto& node : it->second) {
      for (const auto& n : node->GetOutAllNodes()) {
        if (n == target_node) {
          continue;
        }
        if (find(all_fuse_nodes.begin(), all_fuse_nodes.end(), n) != all_fuse_nodes.end()) {
          continue;
        }
        if (GetFusionCycleDetectorPtr()->IsConnected(n, target_node)) {
          node_lead_to_cycle = n;
          FE_LOGD("target node %s is a sub node of %s, a loop will be generated.",
                  target_node->GetName().c_str(), n->GetName().c_str());
          return true;
        }
      }
    }
  }
  return false;
}

bool BufferFusionPassRunner::CompareMappings(const MatchResult &curr_m_rs,
                                             MatchResult &longest_m_rs,
                                             size_t &longest_num) const {
  size_t curr_num = GetMappingSize(curr_m_rs.mapping);
  if (curr_num > longest_num) {
    longest_m_rs = curr_m_rs;
    longest_num = curr_num;
    FE_LOGD("Set current mapping as the longest mapping(size %zu). Mapping element size is %zu.",
            longest_num, longest_m_rs.mapping.size());
    PrintMapping(curr_m_rs.mapping, "Found Longer Mapping");
    return true;
  } else if (curr_num == longest_num) {
    // When mapping size is the same, we compare the nodes size.
    // The mapping contains more distinct nodes will be considered
    // as larger one.
    size_t curr_distinct_num = GetDistinctNodeSize(curr_m_rs.mapping);
    size_t longest_distinct_num = GetDistinctNodeSize(longest_m_rs.mapping);

    FE_LOGD("Current and longest distinct nodes number are %zu and %zu.",
            curr_distinct_num, longest_distinct_num);
    if (curr_distinct_num > longest_distinct_num) {
      longest_m_rs = curr_m_rs;
      longest_num = curr_num;
      FE_LOGD("Set current mapping as the longest mapping(size %zu). Mapping element size is %zu.",
              longest_num, longest_m_rs.mapping.size());
      PrintMapping(curr_m_rs.mapping, "Found Longer Node Size Mapping");
      return true;
    }
  }

  FE_LOGD("Current mapping (size %zu) is <= longest mapping(size %zu). Mapping element size is %zu.",
          curr_num, longest_num, longest_m_rs.mapping.size());
  PrintMapping(curr_m_rs.mapping, "Found Shorter Mapping");
  return false;
}

void BufferFusionPassRunner::RecoverMappingAndQueue(
    MatchInfo &m_info, bool match_error,
    size_t &longest_num) const {
  if (match_error) {
    m_info.m_st.queue_descs.clear();
    m_info.m_st.queue_nodes.clear();
  }

  if (m_info.m_st.queue_descs.empty() && m_info.m_st.queue_nodes.empty()) {
    FE_LOGD("Queue descs is empty, compare matching result.");
    bool longest_updated = false;
    if (GetPatternMatchStatus(*m_info.pattern) && CheckAttrMatch(m_info.m_st.m_rs.mapping)) {
      longest_updated = CompareMappings(m_info.m_st.m_rs, m_info.longest_m_rs, longest_num);
    }
    for (auto desc : m_info.pattern->GetOpDescs()) {
      if (m_info.m_st.m_rs.mapping.find(desc) != m_info.m_st.m_rs.mapping.end()) {
        desc->repeate_curr = 0;
      }
    }
    if (!m_info.saved_m_st.empty()) {
      /* Keep matching from last save queue descs. */
      m_info.m_st = m_info.saved_m_st.back();
      m_info.saved_m_st.pop_back();
      FE_LOGD("Recovered to saved descriptions: {%s}. Updated %d entries.",
              AssmbleDescNames(m_info.m_st.queue_descs).c_str(), longest_updated);
    } else {
      /* Matching ends. */
      m_info.m_st.m_rs = m_info.longest_m_rs;
    }
    for (auto desc : m_info.pattern->GetOpDescs()) {
      if (m_info.m_st.m_rs.mapping.find(desc) != m_info.m_st.m_rs.mapping.end()) {
        desc->repeate_curr = m_info.m_st.m_rs.mapping.find(desc)->second.size();
      }
    }
  }
}

bool BufferFusionPassRunner::SkipNodeForNormalDesc(BufferFusionOpDesc *out_desc, ge::NodePtr node,
                                                   MatchInfo &m_info, const int32_t pattern_type_index) const {
  FE_CHECK(out_desc == nullptr || node == nullptr, FE_LOGW("Unexpected null pointer."), return false);
  string node_name = node->GetName();

  auto it = m_info.matched_output_nodes.find(out_desc->desc_name);
  if (it != m_info.matched_output_nodes.end()) {
    if (find((it->second)[out_desc->repeate_curr].begin(), (it->second)[out_desc->repeate_curr].end(), node_name) !=
        (it->second)[out_desc->repeate_curr].end()) {
      FE_LOGD("skip matched node %s for opdesc %s.", node_name.c_str(), out_desc->desc_name.c_str());
      return true;
    }
  }
  // check the same size branch firstly, if not, check the diff size branch
  bool output_match = out_desc->ignore_output_num || !SkipDiffSizeDesc(node, out_desc, m_info.pattern->GetName());
  if (!output_match) {
    if (!IsOpTypeAny(out_desc->types) && !IsOutputNode(out_desc->types) && !IsOptionalOutput(out_desc)) {
      FE_LOGD("[node %s, desc %s]'s output count does not match.", node->GetName().c_str(),
              out_desc->desc_name.c_str());
      return true;
    }
  }
  bool input_match = out_desc == m_info.head_desc || out_desc->ignore_input_num ||
                      node->GetInDataNodes().size() == out_desc->inputs.size() || IsOutputNode(out_desc->types);
  if (!input_match) {
    FE_LOGD("node size does not match desc, node name=[%s], input count=[%zu], desc input size=[%zu].",
        node_name.c_str(), node->GetInDataNodes().size(), out_desc->inputs.size());
    return true;
  }

  if (!CheckPatternOpTypeMatch(node, out_desc, pattern_type_index)) {
    return true;
  }

  return false;
}

bool BufferFusionPassRunner::SkipNodeBeforeMatch(const ge::NodePtr &node, size_t curr_node_num, size_t curr_desc_num,
                                                 const BufferFusionOpDesc *op_desc, bool get_output_result) const {
  if (!curr_node_num) {
    FE_LOGD("Current node %s has no output nodes. Skipping it.", node->GetName().c_str());
    return true;
  }
  if (op_desc->output_max_limit != -1 && curr_node_num > static_cast<size_t>(op_desc->output_max_limit)) {
    FE_LOGD("output nodes[%zu] of current node %s exceeds 10. skipping...", curr_node_num,
            node->GetName().c_str());
    return true;
  }
  if (!get_output_result) {
    FE_LOGD("Get output desc for %s unsuccessful. skip it.", op_desc->desc_name.c_str());
    return true;
  }
  if (!op_desc->ignore_output_num && curr_node_num > 1 &&
      (curr_node_num != curr_desc_num || op_desc->out_branch_type != TBE_OUTPUT_BRANCH_MULTI)) {
    FE_LOGI("Not match info: out relation [%ld], outnode size [%zu], outdesc size [%zu]. Node %s, desc %s.",
        op_desc->out_branch_type, curr_node_num, curr_desc_num,
        node->GetName().c_str(), op_desc->desc_name.c_str());
    return true;
  }
  return false;
}

void BufferFusionPassRunner::SaveQueueBeforeMatch(std::vector<BufferFusionOpDesc *> &out_descs, ge::NodePtr node,
                                                  const BufferFusionOpDesc *op_desc, MatchInfo &m_info) const {
  BufferFusionOpDesc *first_desc = nullptr;
  if (!out_descs.empty()) {
    first_desc = out_descs.front();
  }

  auto curr_nodes = node->GetOutDataNodes();
  if (first_desc && !first_desc->multi_output_skip_status.empty() &&
      first_desc->repeate_max > first_desc->repeate_curr &&
      first_desc->multi_output_skip_status[first_desc->repeate_curr] == SkipStatus::AVAILABLE &&
      curr_nodes.size() == 1 && curr_nodes.at(0)->GetOutDataNodes().size() > 1) {
    first_desc->multi_output_skip_status[first_desc->repeate_curr] = SkipStatus::SKIPPED;
    FE_LOGD("Trying to skip node %s from repeated opdesc %s first.", curr_nodes.at(0)->GetName().c_str(),
            first_desc->desc_name.c_str());
    out_descs.erase(out_descs.cbegin(), out_descs.cbegin() + 1);
    m_info.saved_m_st.emplace_back(m_info.m_st);

    PrintMapping(m_info.m_st.m_rs.mapping, "SaveQueueBeforeMatch");
    FE_LOGD("Saved queue for skipped case");
  }

  /* When current op_desc's output type is TBE_OUTPUT_BRANCH_MULTI
   * or the op_desc does not care about the output,
   * we need to back track the current node.
   * For example:
   *              A
   *             / \
   *            B   C
   *                 \
   *                  D
   * When matching the pattern above,
   * After matching head A, first we search the left successor B
   * and then we back track from A to C because there may be a longer
   * mapping. So before matching B, we do the following operations
   * to save A into the saved_queue_descs.
   * Then if B is matched, we mark B as visited and recover the queue
   * with A.
   */
  if (!op_desc->ignore_output_num && op_desc->out_branch_type == TBE_OUTPUT_BRANCH_MULTI) {
    m_info.saved_m_st.emplace_back(m_info.m_st);
    m_info.saved_count++;

    FE_LOGD("saved queue for multiple output branches");
  }

  if (op_desc->ignore_output_num && curr_nodes.size() > 1) {
    m_info.saved_m_st.emplace_back(m_info.m_st);
    m_info.saved_count++;

    FE_LOGD("save queue for optional output.");
  }
}

void BufferFusionPassRunner::BreathFirstMatch(MatchInfo &m_info,
                                              MatchInfoStack &m_info_stack) {
  size_t longest_num = GetMappingSize(m_info.longest_m_rs.mapping);

  while (!m_info.m_st.queue_descs.empty() && !m_info.m_st.queue_nodes.empty()) {
    ge::NodePtr node = m_info.m_st.queue_nodes.front();
    BufferFusionOpDesc *op_desc = m_info.m_st.queue_descs.front();
    auto out_nodes = node->GetOutDataNodes();
    std::vector<BufferFusionOpDesc *> out_descs;
    bool res = GetOutputs(op_desc, out_descs);

    FE_LOGD("Out descs for %s are: {%s}.", op_desc->desc_name.c_str(),
            AssmbleDescNames(out_descs).c_str());

    if (SkipNodeBeforeMatch(node, out_nodes.size(), out_descs.size(), op_desc, res)) {
      RecoverMappingAndQueue(m_info, true, longest_num);
      continue;
    }
    if (out_descs.empty() && m_info.m_st.queue_descs.size() > 1 && m_info.m_st.queue_nodes.size() > 1) {
      m_info.m_st.queue_nodes.erase(m_info.m_st.queue_nodes.cbegin());
      m_info.m_st.queue_descs.erase(m_info.m_st.queue_descs.cbegin());
      continue;
    }

    m_info.saved_count = 0;
    SaveQueueBeforeMatch(out_descs, node, op_desc, m_info);
    std::map<BufferFusionOpDesc *, bool> usage_flags;
    // match head node's following nodes
    MatchFollowingNodes(node, out_descs, usage_flags, m_info, m_info_stack);
    // check whether match is ok
    bool match_status =
        GetCurrMatchStatus(op_desc, out_nodes.size(), out_descs, usage_flags);
    FE_LOGD("matched status for output of [desc %s, node %s] is %d.", op_desc->desc_name.c_str(),
            node->GetName().c_str(), match_status);
    if (match_status) {
      m_info.m_st.queue_nodes.erase(m_info.m_st.queue_nodes.cbegin());
      m_info.m_st.queue_descs.erase(m_info.m_st.queue_descs.cbegin());
      RecoverMappingAndQueue(m_info, false, longest_num);
    } else {
      /* If none of the output nodes is matched, we just pop
       * the last saved queue because the queue is meaningless. */
      for (uint32_t i = 0; i < m_info.saved_count; i++) {
        auto save_op_desc = m_info.saved_m_st.back().queue_descs;
        m_info.saved_m_st.pop_back();
        FE_LOGD("remove last queue {%s} for unsuccess match.",
                AssmbleDescNames(save_op_desc).c_str());
      }

      RecoverMappingAndQueue(m_info, true, longest_num);
    }
  }
  FE_LOGD("After breadth-first matching, the longest mapping is:");
  PrintMapping(m_info.longest_m_rs.mapping, "After One Time Matching");
}

bool BufferFusionPassRunner::FinalCycleDetection(const MatchResult &longest_match_result) const {
  std::vector<ge::NodePtr> fusion_nodes;
  Status ret = GetFusionNodes(longest_match_result.mapping, fusion_nodes);
  if (ret != SUCCESS || fusion_nodes.empty()) {
    return true;
  }

  ge::ComputeGraphPtr graph;
  graph = fusion_nodes[0]->GetOwnerComputeGraph();
  if (graph == nullptr) {
    FE_LOGW("[BufferFusion][Match]Owner graph of node %s is nullptr.",
            fusion_nodes[0]->GetName().c_str());
    return false;
  }

  return GetFusionCycleDetectorPtr()->CycleDetection(*graph, fusion_nodes);
}

void BufferFusionPassRunner::CheckMatchingResult(BufferFusionMapping &longest_mapping,
                                                 MatchInfo &m_info, bool &is_matched) const {
  // check pattern status
  bool pattern_status = GetPatternMatchStatus(*m_info.pattern);
  if (!pattern_status) {
    FE_LOGD("Skip head node %s because some op_desc are not matched.", m_info.head_node->GetName().c_str());
    return;
  }
  if (!CheckAttrMatch(m_info.longest_m_rs.mapping)) {
    return;
  }

  longest_mapping = m_info.longest_m_rs.mapping;
  is_matched = true;
}


Status BufferFusionPassRunner::MatchFusionPattern(MatchInfo &m_info, BufferFusionMapping &longest_mapping) {
  /* m_info_stack is for real cycle detection. Because currently
   * we match the pattern from top to bottom, if one node is
   * marked as a key node() then all successors will not be pushed
   * into this stack. So the stack only contains at most one
   * element. */
  MatchInfoStack m_info_stack;
  size_t recursive_times = 0;
  bool is_matched = false;
  do {
    /* The first time matching we just use the matched info from parameter. */
    recursive_times++;
    if (recursive_times != 1) {
      uint32_t last_bt_pos = m_info.longest_m_rs.cycle_pos.back();
      auto m_info_pair = m_info_stack.at(last_bt_pos);
      auto cycle_related_nodes = m_info_pair.first;
      m_info = m_info_pair.second;
      /* Each time there is only one key_node, because in a connected
       * fusion graph, if one node is marked as cycle-related node, all
       * its successors will be ignore. And out matching strategy make sure
       * that we match from head node and each node is fusion graph is connected
       * to head node. */
      if (cycle_related_nodes.empty()) {
        continue;
      }

      FE_LOGD("Trying to recover to the cycle-related node %s.", cycle_related_nodes[0]->GetName().c_str());
      for (auto &ele : cycle_related_nodes) {
        m_info.black_list.emplace(ele);
      }

      if (m_info.RestoreRepTimes() != SUCCESS) {
        continue;
      }
    }

    BreathFirstMatch(m_info, m_info_stack);
    FE_LOGD("After breadth-first match, m_info_stack is %zu.", m_info_stack.size());
    if (!m_info.longest_m_rs.cycle_pos.empty() && FinalCycleDetection(m_info.longest_m_rs)) {
      /* When cycle is found, we need to back track to the
       * state that some one of fusion nodes is not matched.
       * Because in function MatchFollowingNodes, we have marked
       * nodes which may lead to cycle, we just revert the matching
       * result to */
      FE_LOGD("A cycle has been detected; need to trace back to the last node. Temporary cycle vector: %s.",
              StringUtils::IntegerVecToString(m_info.longest_m_rs.cycle_pos).c_str());
      continue;
    } else {
      FE_CHECK_NOTNULL(m_info.pattern);
      /* No cycle is found and we got the longgest match. */
      FE_LOGD("No final cycle was found, and we have obtained the longer match for pattern %s. Temp cycle vector: %s.",
              m_info.pattern->GetName().c_str(),
              StringUtils::IntegerVecToString(m_info.longest_m_rs.cycle_pos).c_str());

      CheckMatchingResult(longest_mapping, m_info, is_matched);
      break;
    }
  } while (!m_info.longest_m_rs.cycle_pos.empty() && recursive_times < kMaxRecursiveTimes);

  if (!is_matched) {
    FE_CHECK_NOTNULL(m_info.head_node);
    FE_LOGD("Cannot matched pattern %s with head %s.",
            m_info.pattern->GetName().c_str(), m_info.head_node->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status BufferFusionPassRunner::MatchFromHead(const ge::NodePtr &head_node, BufferFusionPattern &pattern,
                                             BufferFusionMapping &mapping) {
  // get matched head desc
  BufferFusionOpDesc *head_desc = GetMatchedHeadDesc(head_node, pattern.GetName(), pattern.GetHead());
  MatchInfo m_info(head_node, head_desc, mapping, &pattern);

  if (head_desc != nullptr) {
    m_info.m_st.m_rs.mapping[head_desc].push_back(head_node);
    m_info.m_st.m_rs.matched_nodes[head_node] = head_desc;
    head_desc->repeate_curr++;
    m_info.m_st.queue_nodes.push_back(head_node);
    m_info.m_st.queue_descs.push_back(head_desc);
    m_info.longest_m_rs = m_info.m_st.m_rs;
  } else {
    FE_LOGD("Node [%s] from graph has not been matched to any head desc from fusion pattern %s.",
            head_node->GetName().c_str(), pattern.GetName().c_str());
    return FAILED;
  }

  // match fusion pattern from head node
  return MatchFusionPattern(m_info, mapping);
}

/*
 * @brief: match one pattern, and do fusion for the matched node
 * @param [in] graph: graph node
 * @param [in] pattern: fusion pattern info
 * @param [in] mappings: fusion group node set
 * @return bool: match current pattern ok or not
 */
Status BufferFusionPassRunner::MatchEachPattern(const ge::ComputeGraph &graph, BufferFusionPattern &pattern,
                                                std::map<int64_t, std::vector<ge::NodePtr>> &match_nodes_map) {
  if (buffer_fusion_optimizer_ == nullptr) {
    FE_LOGW("Buffer fusion optimizer is null.");
    return FAILED;
  }
  std::vector<ge::NodePtr> head_nodes;
  buffer_fusion_optimizer_->GetHeadNodesByFusionPattern(pattern, head_nodes);
  size_t matched_times = 0;
  string pattern_name = pattern.GetName();
  // 1. compare 1st pattern op and graph op(include compare op type and TBE type
  for (const ge::NodePtr &node_g : head_nodes) {
    if (!IsNodeFusible(node_g)) {
      continue;
    }
    // initial all descs repeat curr cnt
    InitRepeatCurr(pattern.GetOpDescs());

    BufferFusionMapping mapping;
    if (MatchFromHead(node_g, pattern, mapping) != SUCCESS) {
      continue;
    }

    vector<ge::NodePtr> fusion_nodes;
    Status status = GetFusionNodesByMapping(node_g, mapping, fusion_nodes);
    if (status != SUCCESS && status != NOT_CHANGED) {
      REPORT_FE_ERROR("[SubGraphOpt][UB][RunOnePtn] Pass[%s], Pattern[%s], Failed to obtain fusion nodes due to %u.",
                      GetPassName().c_str(), pattern_name.c_str(), status);
      return FAILED;
    }
    if (fusion_nodes.empty()) {
      FE_LOGD("Fusion nodes are empty, no need to perform buffer fusion.");
      continue;
    }

    if (fusion_nodes.at(0) == nullptr) {
      continue;
    }

    if (fusion_nodes.size() == 1) {
      FE_LOGD("Fusion nodes only contain one node, no need to do buffer fusion.");
      SetSingleOpUbPassNameAttr(fusion_nodes.at(0));
      continue;
    }

    std::string reason_not_support;
    if (!graph.IsSupportFuse(fusion_nodes, reason_not_support)) {
      FE_LOGD("IsSupportFuse unsuccessful, reason: [%s].", reason_not_support.c_str());
      continue;
    }

    TopoSortForFusionNodes(fusion_nodes);
    int64_t scope_id = SetFusionNodesScopeId(fusion_nodes);
    match_nodes_map.emplace(scope_id, fusion_nodes);
    (void)UpdateCycleDetector(graph, fusion_nodes);
    matched_times++;
  }

  FE_LOGD("Match times of buffer fusion pass[%s] and pattern[%s] is [%zu].",
          GetPassName().c_str(), pattern_name.c_str(), matched_times);
  return SUCCESS;
}

bool BufferFusionPassRunner::CheckAttrMatch(BufferFusionMapping &mapping) const {
  // node attr _stream_label must be equal
  auto fusion_nodes = BufferFusionPassBase::GetMatchedNodes(mapping);
  string stream_label;
  for (const auto& n : fusion_nodes) {
    string stream_label_tmp;
    if (!ge::AttrUtils::GetStr(n->GetOpDesc(), STREAM_LABEL, stream_label_tmp)) {
      stream_label_tmp = "null";
    } else {
      FE_LOGD("Fusion nodes have _stream_label %s.", stream_label_tmp.c_str());
    }

    if (stream_label.empty()) {
      stream_label = stream_label_tmp;
    } else if (!stream_label.empty() && stream_label != stream_label_tmp) {
      FE_LOGD("_stream_label is not equal, the pattern matching unsuccessful.");
      return false;
    }
  }
  return true;
}

/*
 * @brief: init all pattern desc repeate_curr to 0
 * @param [in] pattern: fusion pattern desc
 * @return void */
void BufferFusionPassRunner::InitRepeatCurr(const std::vector<BufferFusionOpDesc *> &ops) const {
  for (auto desc : ops) {
    desc->repeate_curr = 0;
    if (!desc->multi_output_skip_status.empty() &&
        desc->multi_output_skip_status[desc->repeate_min] != SkipStatus::DISABLED) {
      for (int64_t i = desc->repeate_min; i < desc->repeate_max; i++) {
        desc->multi_output_skip_status[i] = SkipStatus::AVAILABLE;
      }
    }
  }
}
}  // namespace fe

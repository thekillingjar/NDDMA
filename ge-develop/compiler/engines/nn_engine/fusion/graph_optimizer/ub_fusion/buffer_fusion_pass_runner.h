/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_FUSION_GRAPH_OPTIMIZER_UB_FUSION_BUFFER_FUSION_PASS_RUNNER_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_FUSION_GRAPH_OPTIMIZER_UB_FUSION_BUFFER_FUSION_PASS_RUNNER_H_

#include "graph_optimizer/ub_fusion/base_buffer_fusion_pass_runner.h"
#include "graph_optimizer/ub_fusion/buffer_fusion_optimizer.h"

namespace fe {
struct MatchResult {
  BufferFusionMapping mapping;

  /* Matched node is a temporary record, it's another format of
   * mapping's second element. Using unordered_map to accelerate matching.
   * !!!!ATTENTION: matched_nodes will not include TBE_PATTERN_OP_TYPE_ANY
   * and TBE_PATTERN_OUTPUT_NODE. Those op descs can be matched multiple times.
   */
  std::unordered_map<ge::NodePtr, BufferFusionOpDesc*> matched_nodes;

  /*                                ------> temp ------>
   *                               /                    \
   * conv2d(depthwise)   -->     add  -->  relu6  -->   mul1   -->    mul2
   *   |                      x--/                      /           y-/
   *   |-----------------------------------------------/
   *
   * Here add->temp->mul1 are control edges. If
   * we match the pattern (con2d + add + relu6 + mul1 + mul2) above,
   * we will get a cycle between fusion node and temp.
   * So in cycle detection we need to recognize this kind of cycles.
   * But one problem is the cycle is introduced by node temp and mul1,
   * and when check every direct or indirect successor of mul1, we
   * will get a cycle again. So many cycles will generate so many
   * back-tracking paths. To avoid some duplicated and redundant paths,
   * if one node is marked as cycle related node, all of its direct or
   * indirect successors will not be back-tracked.
   */
  std::vector<ge::NodePtr> nodes_lead_to_cycle;

  /* We need to record how to back track when real cycle happens.
   * Each matching state's (back-tracking-match-info) is different and
   * we do not want to copy the whole (back-tracking-match-info) every
   * time we recover the matching state. So we use cycle_pos_in_stack
   * to reveal this matching state's position in the whole vector. */
  std::vector<uint32_t> cycle_pos;
  MatchResult() {}

  MatchResult(BufferFusionMapping &mapping_param,
              std::unordered_map<ge::NodePtr, BufferFusionOpDesc*> &matched_nodes_param,
              std::vector<ge::NodePtr> &cycle_related_nodes_param,
              std::vector<uint32_t> &cycle_pos_param) {
    mapping = mapping_param;
    matched_nodes = matched_nodes_param;
    nodes_lead_to_cycle = cycle_related_nodes_param;
    cycle_pos = cycle_pos_param;
  }

  MatchResult& operator=(const MatchResult &m_rs_param) {
    if (this == &m_rs_param) {
      return *this;
    }
    mapping = m_rs_param.mapping;
    matched_nodes = m_rs_param.matched_nodes;
    nodes_lead_to_cycle = m_rs_param.nodes_lead_to_cycle;
    cycle_pos = m_rs_param.cycle_pos;
    return *this;
  }

  MatchResult(const MatchResult &m_rs_param) {
    if (this == &m_rs_param) {
      return;
    }
    *this = m_rs_param;
  }

  MatchResult& operator=(MatchResult &&m_rs_param) noexcept {
    if (this == &m_rs_param) {
      return *this;
    }
    mapping = std::move(m_rs_param.mapping);
    matched_nodes = std::move(m_rs_param.matched_nodes);
    nodes_lead_to_cycle = std::move(m_rs_param.nodes_lead_to_cycle);
    cycle_pos = std::move(m_rs_param.cycle_pos);
    return *this;
  }

  MatchResult(MatchResult &&m_rs_param) noexcept {
    mapping = std::move(m_rs_param.mapping);
    matched_nodes = std::move(m_rs_param.matched_nodes);
    nodes_lead_to_cycle = std::move(m_rs_param.nodes_lead_to_cycle);
    cycle_pos = std::move(m_rs_param.cycle_pos);
  }
};

struct MatchState {
  std::vector<BufferFusionOpDesc *> queue_descs;
  std::vector<ge::NodePtr> queue_nodes;

  MatchResult m_rs;
  MatchState() {}

  MatchState(std::vector<BufferFusionOpDesc *> &queue_descs_param,
             std::vector<ge::NodePtr> &queue_nodes_param,
             MatchResult &m_result_param) {
    queue_descs = queue_descs_param;
    queue_nodes = queue_nodes_param;
    m_rs = m_result_param;
  }

  MatchState& operator=(const MatchState &ms_param) {
    if (this == &ms_param) {
      return *this;
    }
    queue_descs = ms_param.queue_descs;
    queue_nodes = ms_param.queue_nodes;
    m_rs = ms_param.m_rs;
    return *this;
  }

  MatchState(const MatchState &ms_param) {
    if (this == &ms_param) {
      return;
    }
    *this = ms_param;
  }

  MatchState& operator=(MatchState &&ms_param) noexcept {
    if (this == &ms_param) {
      return *this;
    }
    queue_descs = std::move(ms_param.queue_descs);
    queue_nodes = std::move(ms_param.queue_nodes);
    m_rs = std::move(ms_param.m_rs);
    return *this;
  }

  MatchState(MatchState &&ms_param) noexcept {
    queue_descs = std::move(ms_param.queue_descs);
    queue_nodes = std::move(ms_param.queue_nodes);
    m_rs = std::move(ms_param.m_rs);
  }
};

struct MatchInfo {
  std::vector<MatchState> saved_m_st;
  uint32_t saved_count;

  MatchState m_st;
  // longest matching result
  MatchResult longest_m_rs;

  /* matched_output_nodes's key is op-desc name.
   * Its value is a map of repetitive times and node name.
   * matched_output_nodes is a global record for marking
   * whether a node has been matched with an op-desc or not.
   * If matched before, it will be skipped.
   * For example:
   *             ---> C1 --> E
   *            /
   * A ------->B----> C2 --> F
   * We want to match A->B->C(only allow matching one C)->F
   * So after matching B, we have two possible choices C1 and C2.
   * We first choose C1 and failed to matching F with E, than we
   * go back to B and matching C2 because C1 is matched.
   *
   * The repetitive times is used for the following case:
   *             ------------
   *            /            \
   * A ------->B----> C2 --> C1 --> E
   *                          \ --> F
   * We want to match A->B->C(allow matching 5 C at max) ->F
   * First we got a pass of A -> B -> C1 -> F.
   * If we do not have repetitive times, C1 will not be matched
   * again and we cannot get the longer pass of A -> B ->C2 ->C1 -> F,
   * So we mark C1 is matched as C's first occurrence. And
   * it can be matched as the second occurrence C.
   *
   * But out matching algorithm can not handle all combinations.
   * For example, we want to match A -> B -> C1
   *                                     \-> C2
   *            ----> C1
   *           /----> C2
   * A -----> B-----> C3
   *           \----> C4
   * Currently, we first match C1 and C2 and then C3 and C4.
   * We cannot match C1 and C3, C2 and C3.....
   * That's a strategy problem instead of a bug.
   */
  std::map<std::string, std::map<int32_t, std::vector<std::string>>> matched_output_nodes;

  /* If a node is in black list, it cannot be fused when it will lead to
   * a cycle. At first, black list is empty. Then in function CheckLoopForward
   * we will detect cycles(which may not be correct), if a cycle is
   * detected after fusing node X, we will keep matching with X and
   * push current matching information into stack will key X.
   * When a cycle is finally detected(which is definitely correct,
   * we need to back track to the state of X and push X into black_list.
   * X will not be matched next time if CheckLoopForward(X) returns true. */
  std::unordered_set<ge::NodePtr> black_list;
  BufferFusionOpDesc *head_desc;
  ge::NodePtr head_node;

  BufferFusionPattern *pattern;
  /* The only thing that will change in pattern is the
   * current repetitive times of BufferFusionOpDesc.
   * Each BufferFusionOpDesc is stored as pointer, so
   * the current repetitive times will change during the matching.
   * In back tracking, we need to recover those repetitive times.
   * So we use a vector to store them. The sequence of the
   * following vector is the same of attribute ops_ in pattern. */
  std::vector<int64_t> repetitive_times;

  MatchInfo(const ge::NodePtr &head_node_param,
            BufferFusionOpDesc *head_desc_param,
            const BufferFusionMapping &mapping_param,
            BufferFusionPattern *pattern_param)
              :saved_count(0),
               head_desc(head_desc_param),
               head_node(head_node_param),
               pattern(pattern_param) {
  for (const auto desc : pattern->GetOpDescs()) {
    if (desc != nullptr) {
      repetitive_times.emplace_back(desc->repeate_curr);
    }
  }
  m_st.m_rs.mapping = mapping_param;
  longest_m_rs.mapping = mapping_param;
}

MatchInfo& operator=(const MatchInfo &m_info_param) {
  if (this == &m_info_param) {
    return *this;
  }
  saved_m_st = m_info_param.saved_m_st;

  saved_count = m_info_param.saved_count;

  m_st = m_info_param.m_st;
  longest_m_rs = m_info_param.longest_m_rs;

  matched_output_nodes = m_info_param.matched_output_nodes;
  black_list = m_info_param.black_list;
  head_desc = m_info_param.head_desc;
  head_node = m_info_param.head_node;

  pattern = m_info_param.pattern;
  repetitive_times = m_info_param.repetitive_times;
  return *this;
}

MatchInfo(const MatchInfo &m_info_param) {
  if (this == &m_info_param) {
    return;
  }
  *this = m_info_param;
}

MatchInfo(MatchInfo &m_info_param) {
  if (this == &m_info_param) {
    return;
  }
  *this = m_info_param;
}

MatchInfo(MatchInfo &&m_info_param)  noexcept {
  saved_m_st = std::move(m_info_param.saved_m_st);
  saved_count =  m_info_param.saved_count;

  m_st = std::move(m_info_param.m_st);
  longest_m_rs = std::move(m_info_param.longest_m_rs);

  matched_output_nodes =  std::move(m_info_param.matched_output_nodes);
  black_list =  std::move(m_info_param.black_list);
  head_desc =  m_info_param.head_desc;
  m_info_param.head_desc = nullptr;
  head_node = std::move(m_info_param.head_node);

  pattern =  m_info_param.pattern;
  m_info_param.pattern = nullptr;
  repetitive_times =  std::move(m_info_param.repetitive_times);
}

void UpdateRepTimes() {
  repetitive_times.clear();
  for (const auto desc : pattern->GetOpDescs()) {
    if (desc != nullptr) {
      repetitive_times.emplace_back(desc->repeate_curr);
    }
  }
}

Status RestoreRepTimes() {
  const auto op_descs = pattern->GetOpDescs();
  if (repetitive_times.size() != op_descs.size()) {
    REPORT_FE_ERROR("[SubGraphOpt][UB][RestoreRepTimes] rep times %zu and op descs size %zu is not same.",
                    repetitive_times.size(), op_descs.size());
    return FAILED;
  }

  for (size_t i = 0; i < op_descs.size(); i++) {
    auto &desc = op_descs[i];
    if (desc != nullptr) {
      desc->repeate_curr = repetitive_times[i];
    }
  }
  return SUCCESS;
}
};
using MatchInfoPair = std::pair<vector<ge::NodePtr>, MatchInfo>;
using MatchInfoStack = std::vector<MatchInfoPair>;

class BufferFusionPassRunner : public BaseBufferFusionPassRunner {
 public:
  BufferFusionPassRunner(const string &pass_name, BufferFusionPassBase *(*create_fn)(),
                         const FusionCycleDetectorPtr &cycle_detector,
                         const OpStoreAdapterBasePtr &op_store_adapter_ptr, const bool is_fusion_check,
                         const BufferFusionOptimizerPtr &buffer_fusion_optimizer);

  BufferFusionPassRunner(const string &pass_name, BufferFusionPassBase *(*create_fn)(),
                         const FusionCycleDetectorPtr &cycle_detector,
                         const BufferFusionOptimizerPtr &buffer_fusion_optimizer);

  virtual ~BufferFusionPassRunner();

 protected:
  /*
   * @brief: match one pattern, and do fusion for the matched node
   * @param [in] graph: graph node
   * @param [in] pattern: fusion pattern info
   * @return bool: match current pattern ok or not
   */
  Status MatchEachPattern(const ge::ComputeGraph &graph, BufferFusionPattern &pattern,
                          std::map<int64_t, std::vector<ge::NodePtr>> &match_nodes_map) override;

 private:
  /*
   * @brief: check whether node output size is same with candidate desc output
   * size
   * @param [in] node: graph node
   * @param [in] op_desc: candidated pattern desc
   * @return bool: check result
   */
  bool SkipDiffSizeDesc(ge::NodePtr node, const BufferFusionOpDesc *op_desc, const string &pattern_name) const;

  bool CheckPatternOpTypeMatch(ge::NodePtr node, const BufferFusionOpDesc *op_desc,
                               const int32_t shape_rule_type_index) const;

  /*
   * @brief: get current loop fusiton match status
   * @param [in] Is_parallel: graph node is multi branch or single branch
   * @param [in] opdescs: candidated pattern desc
   * @param [in] usage: record whether desc has beed matched
   * @return bool: all current loop descs have beed matched or not
   */
  bool GetCurrMatchStatus(const BufferFusionOpDesc *op_desc, size_t out_node_size,
                          const std::vector<BufferFusionOpDesc *> &op_descs,
                          std::map<BufferFusionOpDesc *, bool> &usage) const;

  /*
   * @brief: get pattern fusiton match status
   * @param [in] pattern: fusion pattern info
   * @return bool: the pattern has beed matched or not
   */
  bool GetPatternMatchStatus(BufferFusionPattern &pattern) const;

  /*
   * @brief: get fusiton pattern head desc matched
   * @param [in] node: graph node
   * @param [in] head_descs: candidated head desc list
   * @return BufferFusionOpDesc*: head desc ptr
   */
  BufferFusionOpDesc *GetMatchedHeadDesc(ge::NodePtr node, const string &pattern_name,
                                         std::vector<BufferFusionOpDesc *> head_descs) const;

  /*
   * @brief: get current loop desc matched
   * @param [in] node: graph node
   * @param [in] head_descs: valid head desc
   * @param [in] usage: record whether desc has beed matched
   * @return BufferFusionOpDesc*: matched desc ptr
   */
  BufferFusionOpDesc *GetMatchedNormalDesc(
      const ge::NodePtr &node, const std::vector<BufferFusionOpDesc *> &descs,
      std::map<BufferFusionOpDesc *, bool> &usage,
      MatchInfo &m_info);

  /* Temp means: Detection is executed during the breath first matching process.
   *
   * There will be false positives in temp detection and we will keep matching
   * when a cycle is detected in temp detection. At the same time, we
   * mark the node that leads to the cycle(which may not be a cycle finally)
   * and save the matched information at that moment.
   *
   * For example:
   * X1--->X2--->A--->B--->C
   *             \--------/
   *
   * X1, X2, A, B, C will be fused together, but after matching A, when we want
   * to match C, because the program does not know whether B is in the fusion
   * scope or not, it will consider that A and C is an entity and there is a edge
   * from (X1_X2_A_C) to B and there is also a edge from B to (X1_X2_A_C). So a
   * false positive cycle is detected. For handling this case, we will ignore the
   * cycle generated by C and keep matching B. Simultaneously we save the state of
   * (X1, X2, A).
   *
   * If a cycle is detected finally, for example B can not be matched(the cycle
   * will be like this
   *      -------->
   *     /         \
   * fused_node<----B
   * we will back track to the state of (X1, X2, A) and mark C as a black-list
   * node which will not be ignored when temp cycle is detected. That means,
   * we will not match C and consider X1, X2, A is the longgest match. */
  bool TempCycleDetection(const ge::NodePtr &out_node, bool &is_stacked, MatchInfo &m_info,
                          MatchInfoPair &m_info_pair,
                          MatchInfoStack &m_info_stack);

  /* Final means: Detection is executed when all possible nodes are matched.
   * Final detection will always be correct. */
  bool FinalCycleDetection(const MatchResult &longest_match_result) const;

  /* Do a breath first matching once. If cycle exists after complete matching,
   * we will back track to last available match information and do breath first
   * matching without specific "cyclic" node which is marked at the previous
   * matching. */
  void BreathFirstMatch(MatchInfo &m_info,
                        MatchInfoStack &m_info_stack);

  void CheckMatchingResult(BufferFusionMapping &longest_mapping,
                           MatchInfo &m_info, bool &is_matched) const;

  Status MatchFusionPattern(MatchInfo &m_info, BufferFusionMapping &longest_mapping);

  bool IsSgtInfoConsistant(const ge::NodePtr &consumer, const ge::NodePtr &producer) const;

  void MatchFollowingNodes(ge::NodePtr node, std::vector<BufferFusionOpDesc *> &out_descs,
                           std::map<BufferFusionOpDesc *, bool> &usage_flags, MatchInfo& m_info,
                           MatchInfoStack &m_info_stack);

  void RecoverMappingAndQueue(MatchInfo &m_info, bool match_error,
                              size_t &longest_num) const;

  bool CompareMappings(const MatchResult &curr_m_rs, MatchResult &longest_m_rs,
                       size_t &longest_num) const;

  bool CheckLoopForward(const BufferFusionMapping &mapping, const ge::NodePtr &target_node,
                        ge::NodePtr &node_lead_to_cycle);

  bool IsOptionalOutput(const BufferFusionOpDesc *desc) const;

  bool SkipNodeForNormalDesc(BufferFusionOpDesc *out_desc, ge::NodePtr node,
                             MatchInfo& m_info, const int32_t pattern_type_index) const;

  bool SkipNodeBeforeMatch(const ge::NodePtr &node, size_t curr_node_num, size_t curr_desc_num,
                           const BufferFusionOpDesc *op_desc, bool get_output_result) const;

  void SaveQueueBeforeMatch(std::vector<BufferFusionOpDesc *> &out_descs, ge::NodePtr node,
                            const BufferFusionOpDesc *op_desc, MatchInfo &m_info) const;

  void GetExistingFusionScopes(ge::ComputeGraph &graph, std::map<int64_t, vector<ge::NodePtr>> &fusion_scopes) const;

  /*
   * @brief: check whether graph node is matched with pattern desc
   * @param [in] node: graph node
   * @param [in] op_desc: candidated pattern desc
   * @return bool: check result
   */
  bool IsOpTypeExist(const ge::NodePtr node, const BufferFusionOpDesc *op_desc, int32_t &pattern_type_index) const;

  bool IsOpTypeAny(const vector<string> &types) const;

  bool IsOutputNode(const vector<string> &types) const;

  int32_t GetPatternTypeIndex(const vector<string> &types, const string &op_type) const;

  bool CheckAttrMatch(BufferFusionMapping &mapping) const;

  Status MatchFromHead(const ge::NodePtr &head_node, BufferFusionPattern &pattern,
                       BufferFusionMapping &mapping);

  void InitRepeatCurr(const std::vector<BufferFusionOpDesc *> &ops) const;

 private:
  BufferFusionOptimizerPtr buffer_fusion_optimizer_;
};
}  // namespace fe
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_FUSION_GRAPH_OPTIMIZER_UB_FUSION_BUFFER_FUSION_PASS_RUNNER_H_

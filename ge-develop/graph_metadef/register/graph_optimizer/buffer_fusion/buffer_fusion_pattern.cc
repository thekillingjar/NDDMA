/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/graph_optimizer/buffer_fusion/buffer_fusion_pattern.h"
#include <string>
#include <vector>
#include "framework/common/debug/ge_log.h"
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"

namespace fe {
using std::map;
using std::string;
using std::vector;

const int64_t TBE_FUSION_OP_NUM_MAX = 5L;
const int64_t TBE_PATTERN_NUM_MAX = 5L;
const int64_t TBE_PATTERN_NUM_NONE = 0L;
const int64_t TBE_PATTERN_NUM_DEFAULT = 1L;
const int64_t TBE_OUTPUT_BRANCH_DEFAULT = 0L;
const int64_t TBE_OUTPUT_BRANCH_SINGLE = 1L;
const int64_t TBE_OUTPUT_BRANCH_MULTI = 2L;
const int64_t TBE_PATTERN_GROUPID_INVALID = -1L;
const int32_t TBE_OUTPUT_MAX_NUM_LIMIT = 10;

const std::map<ShapeTypeRule, const std::string> kShapeTypeRuleToStr {
        {IGNORE_SHAPE_TYPE, "IGNORE_SHAPE_TYPE"},
        {ONLY_SUPPORT_STATIC, "ONLY_SUPPORT_STATIC"},
        {ONLY_SUPPORT_DYNAMIC, "ONLY_SUPPORT_DYNAMIC"}
};

inline bool IsAddOverflow(const int64_t &a, const int64_t &b) {
  return ((b > 0) && (a > (static_cast<int64_t>(INT64_MAX) - b))) || \
      ((b < 0) && (a < (static_cast<int64_t>(INT64_MIN) - b)));
}

BufferFusionPattern::BufferFusionPattern(string name, int64_t op_max_count)
    : name_(name), op_max_count_(op_max_count), error_count_(0),
      graph_mod_type_(0) {}

BufferFusionPattern::~BufferFusionPattern() {
  for (auto op : ops_) {
    if (op == nullptr) {
      continue;
    }
    delete (op);
  }
}

bool BufferFusionPattern::IsOpDescValid(const std::string &desc_name, int64_t repeat_min, int64_t repeat_max) const {
  if (desc_name.empty()) {
    GELOGW("[IsOpDescValid][Check] The desc_name cannot be empty.");
    return false;
  }

  if (repeat_min > repeat_max) {
    GELOGW("[IsOpDescValid][Check] Check desc %s failed as repeat_min > repeat_max; repeat_min=%ld, repeat_max=%ld",
           desc_name.c_str(), repeat_min, repeat_max);
    return false;
  }

  if (GetOpDesc(desc_name) != nullptr) {
    GELOGW("[IsOpDescValid][Check] Desc_name repeated. (desc_name:%s)", desc_name.c_str());
    return false;
  }
  return true;
}

bool BufferFusionPattern::IsShapeRulesSizeValid(const size_t &types_size, const size_t &rules_size) const {
  if (rules_size == 1 || types_size == rules_size) {
    return true;
  }
  GELOGW("[IsShapeRulesSizeValid][Check] rule size invalid, rules_size:%zu, types_size:%zu", rules_size, types_size);
  return false;
}

/*
 * @brief:  add op desc info
 * @param [in] desc_name: node desc name
 * @param [in] types: node desc type
 * @param [in] repeate_min: the min count for fusion match,
 *                         patter match failed if real count lower than the
 * value
 * @param [in] repeate_max: the max count for fusion match,
 *                         the op will be ignored if current match count equal
 * with the value
 * @return BufferFusionPattern: pattern object
 */
BufferFusionPattern &BufferFusionPattern::AddOpDesc(const std::string &desc_name, const std::vector<std::string> &types,
                                                    const int64_t repeat_min, const int64_t repeat_max,
                                                    const int64_t group_id, const ShapeTypeRule shape_type_rule,
                                                    const bool not_pattern, const bool is_allow_series) {
  std::vector<ShapeTypeRule> shape_type_rules = {shape_type_rule};
  return AddOpDescTypeRules(desc_name, types, repeat_min, repeat_max, group_id, shape_type_rules,
                            not_pattern, is_allow_series);
}

BufferFusionPattern &BufferFusionPattern::AddOpDesc(const std::string &desc_name, const std::vector<std::string> &types,
                                                    const int64_t repeat_min, const int64_t repeat_max,
                                                    bool is_allow_series) {
  return AddOpDescTypeRules(desc_name, types, repeat_min, repeat_max, TBE_PATTERN_GROUPID_INVALID,
                            {ONLY_SUPPORT_STATIC}, false, is_allow_series);
}

BufferFusionPattern &BufferFusionPattern::AddOpDescTypeRules(const std::string &desc_name,
                                                             const std::vector<std::string> &types,
                                                             const int64_t repeat_min, const int64_t repeat_max,
                                                             const int64_t group_id,
                                                             const std::vector<ShapeTypeRule> &shape_type_rules,
                                                             const bool not_pattern, const bool is_allow_series) {
  if (!IsOpDescValid(desc_name, repeat_min, repeat_max)) {
    IncreaseErrorCount();
    return *this;
  }
  if (!IsShapeRulesSizeValid(types.size(), shape_type_rules.size())) {
    IncreaseErrorCount();
    return *this;
  }

  BufferFusionOpDesc *op = new (std::nothrow) BufferFusionOpDesc();
  if (op == nullptr) {
    GELOGW("[AddOpDesc][Check] Failed to create a new object.");
    IncreaseErrorCount();
    return *this;
  }

  op->desc_name = desc_name;
  op->types = types;
  op->repeate_min = repeat_min;
  op->repeate_max = repeat_max;
  op->repeate_curr = 0;
  op->group_id = group_id;
  op->shape_type_rules = shape_type_rules;
  op->match_status = false;
  op->out_branch_type = TBE_OUTPUT_BRANCH_DEFAULT;
  op->output_max_limit = TBE_OUTPUT_MAX_NUM_LIMIT;
  op->ignore_input_num = false;
  op->ignore_output_num = false;
  op->not_pattern = not_pattern;
  op->is_allow_series = is_allow_series;
  if (repeat_max > repeat_min) {
    for (int64_t i = repeat_min; i < repeat_max; i++) {
      (void)op->multi_output_skip_status.insert(std::pair<int64_t, SkipStatus>(i, SkipStatus::DISABLED));
    }
  }
  ops_.push_back(op);
  op_map_[desc_name] = op;

  op->outputs.clear();
  return *this;
}

/*
 * @brief:  set output desc info
 * @param [in] desc_name: node desc name
 * @param [in] output_ids: output desc
 * @param [in] relation:   output desc relation (1: serial, 2:parallel)
 * @return BufferFusionPattern: pattern object
 */
BufferFusionPattern &BufferFusionPattern::SetOutputs(const string &desc_name, const std::vector<string> &output_ids,
                                                     int64_t relation, bool ignore_input_num, bool ignore_output_num,
                                                     int32_t output_max_limit) {
  if (desc_name.empty()) {
    GELOGW("[SetOutputs][Check] Desc_name must not be empty.");
    IncreaseErrorCount();
    return *this;
  }

  BufferFusionOpDesc *op_desc = GetOpDesc(desc_name);
  if (op_desc == nullptr) {
    GELOGW("[SetOutputs][Check] Desc_name %s does not exist", desc_name.c_str());
    IncreaseErrorCount();
    return *this;
  }
  op_desc->output_max_limit = output_max_limit;
  op_desc->ignore_input_num = ignore_input_num;
  op_desc->ignore_output_num = ignore_output_num;
  if (op_desc->out_branch_type == TBE_OUTPUT_BRANCH_DEFAULT) {
    op_desc->out_branch_type = relation;
  }

  UpdateSkipStatus(op_desc);

  // support one multi output for one op_type
  for (const string &output_id : output_ids) {
    BufferFusionOpDesc *output_op_desc = GetOpDesc(output_id);
    if (output_op_desc == nullptr) {
      GELOGW("[SetOutputs][Check] Desc_name does not exist. (desc_name:%s)", desc_name.c_str());
      if (IsAddOverflow(error_count_, 1) != SUCCESS) {
        GELOGW("[SetOutputs][Check] errorCount_++ overflow. (desc_name:%s)", desc_name.c_str());
        return *this;
      }
      IncreaseErrorCount();
      return *this;
    }
    if (op_desc == output_op_desc) {
      continue;
    }

    op_desc->outputs.push_back(output_op_desc);
    output_op_desc->inputs.push_back(op_desc);

    if (op_desc->out_branch_type != relation) {
      GELOGW("[SetOutputs][Check] Setting outputs relation failed. Current value: %ld, New value: %ld.", op_desc->out_branch_type,
             relation);
      return *this;
    }
  }
  return *this;
}

/*
 * @brief:  get output desc info
 * @param [in]  op_desc: current desc
 * @param [out] outputs: candidate output desc set
 * @return bool: get output desc ok or not
 */
bool BufferFusionPattern::GetOutputs(BufferFusionOpDesc *op_desc, std::vector<BufferFusionOpDesc *> &outputs,
                                     bool ignore_repeat) {
  if (op_desc == nullptr) {
    GELOGW("[GetOutputs][Check] op_desc is null.");
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

void BufferFusionPattern::IncreaseErrorCount() {
  if (error_count_ < std::numeric_limits<int64_t>::max()) {
    error_count_++;
    return;
  }
  GELOGW("[IncreaseErrorCount][Check] error_count_ has overflowed.");
}
/*
 * @brief: set fusion pattern head
 * @param [in] head_ids: node list
 * @return bool: set head desc ok or not
 */
BufferFusionPattern &BufferFusionPattern::SetHead(const std::vector<string> &head_ids) {
  if (head_ids.empty()) {
    GELOGW("[SetHead][Check] The input head_ids is empty.");
    IncreaseErrorCount();
    return *this;
  }
  for (const string &head_id : head_ids) {
    BufferFusionOpDesc *head_op_desc = GetOpDesc(head_id);
    if (head_op_desc == nullptr) {
      GELOGW("[SetHead][Check] descName does not exist. (desc_name:%s)", head_id.c_str());
      if (IsAddOverflow(error_count_, 1) != SUCCESS) {
        GELOGW("[SetHead][Check] errorCount_++ overflow. (desc_name:%s)", head_id.c_str());
        return *this;
      }
      IncreaseErrorCount();
      return *this;
    }
    // Head desc repeat number can not exceed 1
    // if must be exceed 1, it can be realized by several descs
    if (head_op_desc->repeate_max > 1) {
      GELOGW("[SetHead][Check] Head description named %s repeats more than once, current max repeat count is %ld", head_id.c_str(),
             head_op_desc->repeate_max);
      if (IsAddOverflow(error_count_, 1) != SUCCESS) {
        GELOGW("[SetHead][Check] errorCount_++ overflow. (desc_name:%s)", head_id.c_str());
        return *this;
      }
      IncreaseErrorCount();
      return *this;
    }
    head_.push_back(head_op_desc);
  }

  // check head desc repeat min total value, it can not excceed 1
  int64_t desc_total_min = 0;
  for (const auto &desc : head_) {
    if (IsAddOverflow(desc_total_min, desc->repeate_min) != SUCCESS) {
      GELOGW("[SetHead][Check] desc_total_min[%ld] + repeate_min[%ld] overflow", desc_total_min, desc->repeate_min);
      return *this;
    }
    desc_total_min += desc->repeate_min;
  }

  if (desc_total_min > 1) {
    GELOGW("[SetHead][Check] Head desc repeat min total value cannot exceed 1, current value is %ld",
           desc_total_min);
    IncreaseErrorCount();
    return *this;
  }
  return *this;
}

void BufferFusionPattern::UpdateSkipStatus(const BufferFusionOpDesc *op_desc) const {
  if (op_desc->out_branch_type == TBE_OUTPUT_BRANCH_MULTI) {
    for (auto &input_desc : op_desc->inputs) {
      if (input_desc->types.size() != op_desc->types.size()) {
        continue;
      }
      bool is_same_type = true;
      for (size_t i = 0; i < input_desc->types.size(); i++) {
        if (input_desc->types[i] != op_desc->types[i]) {
          is_same_type = false;
          break;
        }
      }
      if (is_same_type && (input_desc->ignore_output_num)) {
        for (int64_t i = input_desc->repeate_min; i < input_desc->repeate_max; i++) {
          input_desc->multi_output_skip_status[i] = SkipStatus::AVAILABLE;
        }
      }
    }
  }
}

BufferFusionPattern &BufferFusionPattern::SetRelation(const std::string &src_desc_name,
                                                      const std::string &dst_desc_name,
                                                      const PatternRelation pattern_relation) {
  if (src_desc_name.empty() || dst_desc_name.empty()) {
    GELOGW("[SetRelation][Check] Source description name or destination description name is empty.");
    IncreaseErrorCount();
    return *this;
  }
  BufferFusionOpDesc *src_op_desc = GetOpDesc(src_desc_name);
  if (src_op_desc == nullptr) {
    GELOGW("[SetRelation][Check] Op desc of [%s] is null.", src_desc_name.c_str());
    IncreaseErrorCount();
    return *this;
  }
  BufferFusionOpDesc *dst_op_desc = GetOpDesc(dst_desc_name);
  if (dst_op_desc == nullptr) {
    GELOGW("[SetRelation][Check] Op desc of [%s] is null.", dst_desc_name.c_str());
    IncreaseErrorCount();
    return *this;
  }
  src_op_desc->relations.push_back(std::make_pair(dst_op_desc, pattern_relation));
  if (pattern_relation == PatternRelation::RELATIVE_POSITION_CONSISTENT) {
    dst_op_desc->relations.push_back(std::make_pair(src_op_desc, pattern_relation));
  }
  return *this;
}

/*
 * @brief: get description ptr by name
 * @param [in] desc_name: fusion pattern desc name
 * @return BufferFusionOpDesc*: description ptr
 */
BufferFusionOpDesc *BufferFusionPattern::GetOpDesc(const string &desc_name) const {
  const auto it = op_map_.find(desc_name);
  if (it != op_map_.end()) {
    return it->second;
  }
  return nullptr;
}

const std::vector<BufferFusionOpDesc *>& BufferFusionPattern::GetHead() const { return head_; }

const std::string& BufferFusionPattern::GetName() const { return name_; }

int64_t BufferFusionPattern::GetOpMaxCount() const { return op_max_count_; }

int64_t BufferFusionPattern::GetErrorCnt() const { return error_count_; }

void BufferFusionPattern::SetGraphModType(int64_t graph_mod_type) {
  graph_mod_type_ = graph_mod_type;
}

int64_t BufferFusionPattern::GetGraphModType() const { return graph_mod_type_; }

const std::vector<BufferFusionOpDesc *>& BufferFusionPattern::GetOpDescs() const { return ops_; }
}  // namespace fe

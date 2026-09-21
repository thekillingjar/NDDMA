/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_STRATEGY_OP_FORMAT_DTYPE_STRATEGY_MANAGER_H_
#define FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_STRATEGY_OP_FORMAT_DTYPE_STRATEGY_MANAGER_H_

#include "common/fe_op_info_common.h"
#include "common/fe_context_utils.h"
#include "graph_optimizer/op_judge/format_and_dtype/strategy/dtype_strategy/op_dtype_selection_strategy_default_mode.h"
#include "graph_optimizer/op_judge/format_and_dtype/strategy/dtype_strategy/op_dtype_selection_strategy_allow_fp32_to_fp16.h"
#include "graph_optimizer/op_judge/format_and_dtype/strategy/dtype_strategy/op_dtype_selection_strategy_allow_fp32_to_bf16.h"
#include "graph_optimizer/op_judge/format_and_dtype/strategy/dtype_strategy/op_dtype_selection_strategy_allow_fp32_to_lowprecision.h"
#include "graph_optimizer/op_judge/format_and_dtype/strategy/dtype_strategy/op_dtype_selection_strategy_allow_mix_precision_bf16.h"
#include "graph_optimizer/op_judge/format_and_dtype/strategy/dtype_strategy/op_dtype_selection_strategy_allow_mix_precision_fp16.h"
#include "graph_optimizer/op_judge/format_and_dtype/strategy/dtype_strategy/op_dtype_selection_strategy_base.h"
#include "graph_optimizer/op_judge/format_and_dtype/strategy/dtype_strategy/op_dtype_selection_strategy_force_fp16.h"
#include "graph_optimizer/op_judge/format_and_dtype/strategy/dtype_strategy/op_dtype_selection_strategy_force_lowerprecision.h"
#include "graph_optimizer/op_judge/format_and_dtype/strategy/dtype_strategy/op_dtype_selection_strategy_force_fp32.h"
#include "graph_optimizer/op_judge/format_and_dtype/strategy/dtype_strategy/op_dtype_selection_strategy_cube_hif8.h"
#include "graph_optimizer/op_judge/format_and_dtype/strategy/dtype_strategy/op_dtype_selection_strategy_mixed_hif8.h"

#include "graph_optimizer/op_judge/format_and_dtype/strategy/format_strategy/op_format_selection_strategy_default_mode.h"
#include "graph_optimizer/op_judge/format_and_dtype/strategy/format_strategy/op_format_selection_strategy_follow_predecessor.h"

namespace fe {
using OpDtypeSeletionStrategyBasePtr = std::shared_ptr<OpDtypeSeletionStrategyBase>;
using OpFormatSelectionStrategyDefaultModePtr = std::shared_ptr<OpFormatSelectionStrategyDefaultMode>;
using OpFormatSelectionStrategyFollowPredecessorPtr = std::shared_ptr<OpFormatSelectionStrategyFollowPredecessor>;

class OpFormatDtypeStrategyManager {
 public:
  OpFormatDtypeStrategyManager(const std::string& engine_name, FormatDtypeQuerierPtr format_dtype_querier_ptr);
  ~OpFormatDtypeStrategyManager();

  Status Initialize();
  /**
   * get the matched index vector for the input desc
   * @param prio_index_map: this map stores the tensor index which are sorted by
   * priority. Usually the cosnt and variable tensors are in low priority
   * @param op_desc_ptr: current node's opdesc pointer
   * @param tensor_index_name_map: this map stores the relation between
   * tensor index and it's kernel name
   * @param op_kernel_info_ptr: it's the op kernel information
   * @param matched_index_vec: matched index vector of the op kernel
   * Get all possible dtype index according to the precision mode
   * @return SUCCESS or FAIL
   */
  Status GetAllPossibleDtypeIndexByPrecisionMode(const std::map<uint32_t, int> &prio_index_map,
                                                 const IndexNameMap &tensor_index_name_map, const ge::NodePtr &node_ptr,
                                                 const OpKernelInfoPtr &op_kernel_info_ptr, const bool &is_input,
                                                 vector<uint32_t> &matched_index_vec);

  Status MatchByDifferentMode(FormatDtypeSelectionBasicInfo &basic_info);

  /**
   * Initially generate matched index by ops kernel.
   * @param is_matched_index_vec_inited a flag tells whether the Matched index
   * is initialized or not
   * @param input_format_vec input_format which this op supports in ops kernel
   * combined with dynamic selection
   * @param vector<uint32_t>& matched_index_vec,
   * @return SUCCESS or FAIL
   */
  Status GenerateInitialMatchedIndexVec(bool &is_matched_index_vec_inited, vector<uint32_t> &matched_index_vec,
                                        const std::vector<ge::Format> &input_format_vec);

  /**
   * get the matched index vector for the input desc
   * @param prio_index_map: this map stores the tensor index which are sorted by
   * priority. Usually the cosnt and variable tensors are in low priority
   * @param op_desc_ptr: current node's opdesc pointer
   * @param tensor_index_name_map: this map stores the relation between
   * tensor index and it's kernel name
   * @param op_kernel_info_ptr: it's the op kernel information
   * @param matched_index_vec: matched index vector of the op kernel
   * Get all possible dtype index according to the precision mode
   * @return SUCCESS or FAIL
   */
  Status GetAllPossibleFormatIndexByDefaultMode(const std::map<uint32_t, int> &prio_index_map,
                                                const IndexNameMap &tensor_index_name_map, const ge::NodePtr &node_ptr,
                                                const OpKernelInfoPtr &op_kernel_info_ptr, const bool &is_input,
                                                vector<uint32_t> &matched_index_vec);
  void SetPrecisionMode(const fe::PrecisionMode &precision_mode);

  fe::PrecisionMode GetPrecisionMode() const;

 private:
  static bool IsInputHeavyOpData(const ge::NodePtr &node, const int32_t input_index);

 private:
  std::string engine_name_;
  fe::PrecisionMode precision_mode_;
  FormatDtypeQuerierPtr format_dtype_querier_ptr_;
  std::vector<OpDtypeSeletionStrategyBasePtr> dtype_selection_strategies_;
  OpFormatSelectionStrategyDefaultModePtr format_selection_default_strategy_;
  OpFormatSelectionStrategyFollowPredecessorPtr format_selection_prev_strategy_;
};
}  // namespace fe

#endif  // FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_STRATEGY_OP_FORMAT_DTYPE_STRATEGY_MANAGER_H_

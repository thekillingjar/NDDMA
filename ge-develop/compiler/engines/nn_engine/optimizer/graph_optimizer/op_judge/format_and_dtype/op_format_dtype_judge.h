/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_OP_FORMAT_DTYPE_JUDGE_H_
#define FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_OP_FORMAT_DTYPE_JUDGE_H_

#include "format_selector/manager/format_dtype_querier.h"
#include "graph_optimizer/op_judge/format_and_dtype/strategy/op_format_dtype_strategy_manager.h"
#include "graph_optimizer/op_judge/format_and_dtype/update_desc/op_format_dtype_update_desc.h"
#include "graph_optimizer/op_judge/format_and_dtype/update_desc/subgraph/sub_data_format_dtype_update.h"
#include "graph_optimizer/op_judge/format_and_dtype/update_desc/subgraph/sub_netoutput_format_dtype_update.h"
#include "graph_optimizer/op_judge/op_judge_base.h"

namespace fe {
using OpFormatDtypeStrategyManagerPtr = std::shared_ptr<OpFormatDtypeStrategyManager>;
using OpFormatDtypeUpdateDescPtr = std::shared_ptr<OpFormatDtypeUpdateDesc>;
using FormatDtypeQuerierPtr = std::shared_ptr<FormatDtypeQuerier>;
using SubDataFormatDtypeUpdatePtr = std::shared_ptr<SubDataFormatDtypeUpdate>;
using SubNetOutputFormatDtypeUpdatePtr = std::shared_ptr<SubNetOutputFormatDtypeUpdate>;

struct OpJudgeParam {
  ge::NodePtr node_ptr;
  OpKernelInfoPtr op_kernel_info_ptr;
  IndexNameMap input_index_map;
  IndexNameMap output_index_map;
  std::map<uint32_t, int> prio_index_map;
  OpJudgeParam(ge::NodePtr node_param, OpKernelInfoPtr op_kernel_info_ptr_param)
      : node_ptr(std::move(node_param)),
        op_kernel_info_ptr(std::move(op_kernel_info_ptr_param)) {}
};

class OpFormatDtypeJudge : public OpJudgeBase {
 public:
  OpFormatDtypeJudge(const std::string& engine_name, RefRelationsPtr reflection_builder_ptr);

  virtual ~OpFormatDtypeJudge() override;

  Status Judge(ge::ComputeGraph &graph) override;
  Status SetFormat(ge::ComputeGraph &graph) override;

  Status Initialize();

  void SetPrecisionMode(const PrecisionMode &precision_mode);

 private:
  /**
   * judge the format and data type for the node
   * @param node_ptr current node
   * @return SUCCESS or FAIL
   */
  Status JudgeByNode(ge::NodePtr node_ptr);
  /**
   * set the highest prior imply type of op, update the format and data type of
   * the input or output desc of the current_node
   * @param node_ptr current node
   * @param imply_type_str imply type
   * @return SUCCESS or FAIL
   */
  Status SetDtypeByPrecisionMode(ge::NodePtr node_ptr, const std::string &imply_type_str,
                                 const OpImplType &op_imply_type);
  Status SetFormatByJudgeResult(ge::NodePtr node_ptr, const std::string &imply_type_str);

  /**
   * Get all possible input data type index by specific precision mode
   * @param node_ptr: current node
   * @param op_kernel_info_ptr: op kernel information
   * @param input_map: the index name map of tensors
   * @param prio_index_map: this is the sorted tensor index, we will loop among
   * tensors according to the sequence of this map
   * @param matched_index_vec: the vector which stores matched dtype index
   * @return SUCCESS or FAIL
   */
  Status GetInputAndOutputDtypeIndex(const ge::NodePtr &node_ptr, const OpKernelInfoPtr &op_kernel_info_ptr,
                                     const IndexNameMap &input_map, const IndexNameMap &output_map,
                                     const std::map<uint32_t, int> &prio_index_map,
                                     vector<uint32_t> &matched_index_vec);

  /**
   * Get all possible input data type index by specific precision mode
   * @param node_ptr: current node
   * @param op_kernel_info_ptr: op kernel information
   * @param output_map: the index name map of tensors
   * @param prio_index_map: this is the sorted tensor index, we will loop among
   * tensors according to the sequence of this map
   * @param matched_index_vec: the vector which stores matched dtype index
   * @return SUCCESS or FAIL
   */
  Status GetInputAndOutputFormatIndex(const OpJudgeParam &judge_param,
                                      vector<uint32_t> &matched_index_vec);
  /**
   * @brief set node attr that whether support fp16 in and fp32 out at the same time
   *
   * @param node_ptr: current node
   * @param op_kernel_info_ptr: op kernel information
   * @param input_index_map: the index name map of input tensors
   * @param output_index_map: the index name map of output tensors
   * @return SUCCESS or FAIL
   */
  Status SetSupport16In32outAttr(ge::NodePtr node_ptr, const OpKernelInfoPtr &op_kernel_info_ptr,
                                 const IndexNameMap &input_index_map, const IndexNameMap &output_index_map);
  /**
   * @brief check the node whether support fp16 in and fp32 out at the same time
   *
   * @param node_ptr: current node
   * @param op_kernel_info_ptr: op kernel information
   * @param input_index_map: the index name map of input tensors
   * @param output_index_map: the index name map of output tensors
   * @return true
   * @return false
   */
  bool IsNodeSupport16In32out(ge::NodePtr node_ptr, const OpKernelInfoPtr &op_kernel_info_ptr,
                              const IndexNameMap &input_index_map, const IndexNameMap &output_index_map);
  /**
   * sort the input desc: non const/constant/variable inputs first
   * @param node_ptr current node
   * @param op_desc_ptr current op desc
   * @param key is priority, value is the input index
   * @return SUCCESS or FAIL
   */
  Status SortInputBySequence(ge::NodePtr node_ptr, ge::OpDescPtr op_desc_ptr, std::map<uint32_t, int> &prio_index_map);

  /**
   * Get all possible input data type index by specific precision mode
   * @param node_ptr: current node
   * @param op_kernel_info_ptr: op kernel information
   * @param input_map: the index name map of tensors
   * @param prio_index_map: this is the sorted tensor index, we will loop among
   * tensors according to the sequence of this map
   * @param matched_index_vec: the vector which stores matched dtype index
   * @return SUCCESS or FAIL
   */
  Status GetInputDtypeIndex(const ge::NodePtr &node_ptr, const OpKernelInfoPtr &op_kernel_info_ptr,
                            const IndexNameMap &input_map, const std::map<uint32_t, int> &prio_index_map,
                            vector<uint32_t> &matched_index_vec);

  /**
   * Get all possible input data type index by specific precision mode
   * @param node_ptr: current node
   * @param op_kernel_info_ptr: op kernel information
   * @param input_map: the index name map of tensors
   * @param prio_index_map: this is the sorted tensor index, we will loop among
   * tensors according to the sequence of this map
   * @param matched_index_vec: the vector which stores matched dtype index
   * @return SUCCESS or FAIL
   */
  Status GetInputFormatIndex(const ge::NodePtr &node_ptr, const OpKernelInfoPtr &op_kernel_info_ptr,
                             const IndexNameMap &input_map, const std::map<uint32_t, int> &prio_index_map,
                             vector<uint32_t> &matched_index_vec);

  /**
   * Get all possible input data type index by specific precision mode
   * @param node_ptr: current node
   * @param op_kernel_info_ptr: op kernel information
   * @param input_map: the index name map of tensors
   * @param matched_index_vec: the vector which stores matched dtype index
   * @return SUCCESS or FAIL
   */
  Status GetOutputDtypeIndex(const ge::NodePtr &node_ptr, const OpKernelInfoPtr &op_kernel_info_ptr,
                             const IndexNameMap &output_map, vector<uint32_t> &matched_index_vec);

  /**
   * Get all possible input data type index by specific precision mode
   * @param node_ptr: current node
   * @param op_kernel_info_ptr: op kernel information
   * @param output_map: the index name map of tensors
   * @param matched_index_vec: the vector which stores matched dtype index
   * @return SUCCESS or FAIL
   */
  Status GetOutputFormatIndex(const ge::NodePtr &node_ptr, const OpKernelInfoPtr &op_kernel_info_ptr,
                              const IndexNameMap &output_map, vector<uint32_t> &matched_index_vec);

  /**
   * whether to judge the format. The Data, Const, Variable,
   * NetOutput and non fe imply type don't need to judge the format.
   * @param op_desc_ptr currrent op desc
   * @param imply_type the imply type of the op_desc_ptr
   * @return result
   */
  bool IsNoNeedJudge(const ge::OpDescPtr &op_desc_ptr, int64_t &imply_type) const;

  Status InitMatchedIndexByCustomDtype(const ge::NodePtr &node,
                                       const OpKernelInfoPtr &op_kernel_info_ptr,
                                       const IndexNameMap &input_index_map,
                                       const IndexNameMap &output_index_map,
                                       vector<uint32_t> &matched_index_vec) const;

  void FilterDtypeIndexByCustom(const ge::NodePtr &node,
                                const OpKernelInfoPtr &op_kernel_info_ptr,
                                const vector<ge::DataType> &cust_dtypes,
                                const IndexNameMap &index_map,
                                const bool &is_input,
                                vector<bool> &filter_index_vec) const;

  uint32_t GetMatchedIndex(const vector<uint32_t> &matched_index_vec,
                           const vector<uint32_t> &cust_filter_matched_index_vec,
                           const string &op_name, const string &op_type) const;

  void RecordC04FormatForSingleTensor(const ge::NodePtr &node_ptr,
                                      const OpKernelInfoPtr &op_kernel_info_ptr,
                                      const InputOrOutputInfoPtr &tensor_info,
                                      set<uint32_t> &matched_index_set,
                                      unordered_set<uint32_t> &c04_format_index_set) const;

  void RecordC04FormatIndex(const vector<ge::Format> &input_or_output_format,
                            const set<uint32_t> &matched_index_set,
                            unordered_set<uint32_t> &c04_format_index_set) const;

  void FilterBySmallChannel(const ge::NodePtr &node_ptr, const OpKernelInfoPtr &op_kernel_info_ptr,
                            const IndexNameMap &input_map, const IndexNameMap &output_map,
                            vector<uint32_t> &matched_index_vec) const;

  void RecordNDNZFormatIndex(const InputOrOutputInfoPtr &tensor_info, const std::vector<uint32_t> &matched_index_vec,
                             const std::vector<ge::Format> &kernel_format_vec, vector<uint32_t> &filter_index) const;

  void RecordFormatFilterIndex(const std::vector<uint32_t> &matched_index_vec,
                               const std::vector<ge::Format> &kernel_format_vec,
                               const std::unordered_set<ge::Format> &filter_format_type, bool is_filter_match,
                               std::vector<uint32_t> &filter_index) const;

  Status FilterFormatIndexByFormatMode(const ge::NodePtr &node_ptr, const OpKernelInfoPtr &op_kernel_info_ptr,
                                       const IndexNameMap &input_map, const FormatModeType &cfg_type,
                                       vector<uint32_t> &matched_index_vec) const;

  void FilterByFormatMode(const ge::NodePtr &node_ptr, const OpKernelInfoPtr &op_kernel_info_ptr,
                          const IndexNameMap &input_map, vector<uint32_t> &matched_index_vec) const;

  void FilterDiffFormatIndex(const ge::NodePtr &node_ptr, const OpKernelInfoPtr &op_kernel_info_ptr,
                             const InputOrOutputInfoPtr &tensor_info, const ge::Format &ori_format,
                             std::vector<uint32_t> &matched_index_vec) const;

  void FilterKernelFormatByOriginFormat(const ge::NodePtr &node_ptr, const OpKernelInfoPtr &op_kernel_info_ptr,
                                        const IndexNameMap &input_map, const IndexNameMap &output_map,
                                        vector<uint32_t> &matched_index_vec) const;

  void DoPromoteMatch(const OpJudgeParam &op_judge_param, bool must_promote_flag, vector<uint32_t> &matched_index_vec);

  Status PromoteTypeMatch(const OpJudgeParam &op_judge_param,
      const std::vector<std::vector<int>> &promote_inputs_indexes, vector<uint32_t> &matched_index_vec) const;

  Status MatchForPromoteInput(const OpJudgeParam &op_judge_param, const std::vector<int> &promote_index,
      const ge::DataType &target_dtype, vector<uint32_t> &matched_index_vec) const;

  static void FindPromoteDtype(const std::vector<ge::DataType> &input_dtype_vec, const ge::DataType &target_dtype,
      std::vector<uint32_t> &matched_index_vec);

  Status JudgeByPrecisionMode(const OpJudgeParam &op_judge_param, bool must_promote_flag,
      vector<uint32_t> &matched_index_vec);

  RefRelationsPtr reflection_builder_ptr_;
  OpFormatDtypeStrategyManagerPtr op_format_dtype_strategy_manager_ptr_;
  OpFormatDtypeUpdateDescPtr op_format_dtype_update_desc_ptr_;
  FormatDtypeQuerierPtr format_dtype_querier_ptr_;
  SubDataFormatDtypeUpdatePtr sub_data_format_dtype_update_ptr_;
  SubNetOutputFormatDtypeUpdatePtr sub_net_output_format_dtype_update_ptr_;
};
}  // namespace fe

#endif  // FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_OP_FORMAT_DTYPE_JUDGE_H_

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_HEAVY_FORMAT_PROPAGATION_HEAVY_FORMAT_SELECTOR_H_
#define FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_HEAVY_FORMAT_PROPAGATION_HEAVY_FORMAT_SELECTOR_H_
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "common/fe_inner_error_codes.h"
#include "common/fe_log.h"
#include "common/fe_utils.h"
#include "common/math_util.h"
#include "common/fe_op_info_common.h"
#include "common/util/op_info_util.h"
#include "format_selector/manager/format_dtype_querier.h"
#include "graph/compute_graph.h"
#include "graph_optimizer/op_judge/format_and_dtype/strategy/matcher/dtype/op_dtype_precise_matcher.h"

namespace fe {
using FormatDtypeQuerierPtr = std::shared_ptr<FormatDtypeQuerier>;

enum class FormatScore {
  DISTRIBUTED_HEAVY_FORMAT_SCORE = 2000,
  OTHER_HEAVY_FORMAT_SCORE = 200,
  ORIGINAL_FORMAT_SCORE = 100,
  OTHER_FORMAT_SCORE = 1,
  SUB_FORMAT_MISMATCH_SCORE = 0
};

constexpr int32_t INVALID_KERNEL_INDEX = -1;

enum class FormatSelectionType {
  /* FORMAT_DEPENDS_ON_OP_KERNEL_INFO means this op can not use arbitrary
   * format. So we need to loop up for the ops kernel first to get all supported
   * formats. */
  FORMAT_DEPENDS_ON_OP_KERNEL_INFO = 0,

  /* FORMAT_AGNOSTIC_FOR_ALL_INPUTS_AND_OUTPUTS means this op can use arbitrary
   * format. And we need to set the format of all inputs and outputs as the
   * same. */
  FORMAT_AGNOSTIC_FOR_ALL_INPUTS_AND_OUTPUTS = 1,

  /* FORMAT_AGNOSTIC_FOR_PAIRED_INPUT_AND_OUTPUT means this op can use arbitrary
   * format. And we need to set the format for each paired input and output as
   * the same. For example, the input0 and output0 should be the same format and
   * the input1 and output1 should be the same. Paired is definedas:
   * input:x and output:x and the size of inputs should be the same as the
   * size of outputs. */
  FORMAT_AGNOSTIC_FOR_PAIRED_INPUT_AND_OUTPUT = 2,

  /* FORMAT_PENETRATION means this op may not support heavy format but if we
   * distribute the heavy format through this op and the next op of this op
   * will be set as heavy format and the this op and the trans-nodes between
   * this op and the next op will be merged of fused. It will decrease the
   * number of transdata.
   * For example:
   * original graph: A(Fz) -> TransData(Fz->HWCN) -> B(HWCN),
   *
   * after penetrating TransData:
   * A(Fz) -> TransData(Fz->HWCN) -> B(Fz)
   *
   * after trans-nodes insertion:
   * A(Fz) -> TransData(Fz->HWCN) -> TransData(HWCN->Fz) -> B(Fz)
   *
   * after merging trans-nodes:
   * A(Fz) -> B(Fz), so the op TransData is defined as the FORMAT_PENETRATION.
   *
   * Another case:
   * original graph:
   *                const(NCHW)                                const(NCHW)
   *                /      \       ->(penetrate const)          /      \
   *            A(Fz)     B(NCHW)                           A(Fz)     B(Fz)
   * after trans-nodes insertion:
   *               const(NCHW)
   *              /            \
   *  TransData(NCHW->Fz)      TransData(NCHW->Fz)
   *        |                       |
   *      A(Fz)                   B(Fz)
   *
   * after trans-nodes and const fusion:
   *            const(Fz)
   *             /      \
   *          A(Fz)     B(Fz)
   */
  FORMAT_PENETRATION = 3,

  /* FORMAT_AGNOSTIC_FUNCTION_OP means this op is function op. */
  FORMAT_AGNOSTIC_FUNCTION_OP = 4,

  FORMAT_AGNOSTIC_BOTTOM = 5
};

struct PropagationInfo {
  int64_t group;            /* The value of attribute _fe_group of last node */
  std::string reshape_type; /* reshape type for reduce op */
  ge::Format heavy_format;
  int32_t sub_format;
  int32_t c0_format;
  explicit PropagationInfo(int64_t group_value = 0)
      : group(group_value), reshape_type(""), heavy_format(ge::FORMAT_RESERVED), sub_format(0), c0_format(0) {}
};

struct NodeInfo;
using NodeInfoPtr = std::shared_ptr<NodeInfo>;
struct NodeInfo {
  PropagationInfo propagation_info;
  ge::NodePtr current_node;
  OpKernelInfoPtr current_node_op_kernel_ptr;
  FormatSelectionType format_selection;
  int32_t anchor_index_of_curr_node;
  NodeInfoPtr last_node_info; /* record the last_node_info Ptr */
  ge::NodePtr last_node;
  bool is_sub_graph_data_or_nt_opt;
  bool is_input_of_curr_node;
  vector<IndexNameMap> tensor_map; /* the first map is for input and the second is for output */
  NodeInfo(ge::NodePtr node, NodeInfoPtr last_node_info_param, ge::Format heavy_format_param, int32_t sub_format_param,
           int32_t c0_format_param,
           PropagationInfo &propagation_info_param, int32_t anchor_index_of_curr_node_param = 0,
           bool is_input_of_curr_node_param = true, bool is_sub_graph_data_or_nt_opt_param = false)
      : current_node(std::move(node)),
        current_node_op_kernel_ptr(nullptr),
        format_selection(FormatSelectionType::FORMAT_AGNOSTIC_BOTTOM),
        anchor_index_of_curr_node(anchor_index_of_curr_node_param),
        last_node_info(last_node_info_param),
        is_sub_graph_data_or_nt_opt(is_sub_graph_data_or_nt_opt_param),
        is_input_of_curr_node(is_input_of_curr_node_param) {
    propagation_info.group = propagation_info_param.group;
    propagation_info.reshape_type = propagation_info_param.reshape_type;
    propagation_info.heavy_format = heavy_format_param;
    propagation_info.sub_format = sub_format_param;
    propagation_info.c0_format = c0_format_param;

    if (last_node_info_param == nullptr) {
      last_node = nullptr;
    } else {
      last_node = last_node_info_param->current_node;
    }
  }
};

struct FormatAndDataTypeInfo {
  ge::Format heavy_format = ge::FORMAT_RESERVED;
  int32_t sub_format = 0;
  int32_t c0_format = 0;
  ge::DataType data_type = ge::DT_FLOAT;
  ge::OpDescPtr op_desc_ptr;
  ge::NodePtr node_ptr;
  int32_t anchor_index; /* Index of input or output anchor */
  int32_t format_index; /* The column of the selected format */
  IndexNameMap &tensor_map;
  InputOrOutputInfoPtr op_kernel_tensor_info;
  bool is_input;
  bool is_forward;
  bool is_sub_graph_data_or_nt_opt;
  FormatSelectionType format_selection_type;
  bool only_to_paired_input_or_output;
  PropagationInfo propagation_info;
  FormatAndDataTypeInfo(const ge::NodePtr &node_ptr_param, ge::OpDescPtr &op_desc_ptr_param, int32_t anchor_index_param,
                        int32_t format_index_param, bool is_input_param, bool is_forward_param,
                        FormatSelectionType format_selection_type_param, PropagationInfo &propagation_info_param,
                        IndexNameMap &tensor_map_param)
      : op_desc_ptr(op_desc_ptr_param),
        node_ptr(node_ptr_param),
        anchor_index(anchor_index_param),
        format_index(format_index_param),
        tensor_map(tensor_map_param),
        is_input(is_input_param),
        is_forward(is_forward_param),
        is_sub_graph_data_or_nt_opt(false),
        format_selection_type(format_selection_type_param),
        only_to_paired_input_or_output((format_selection_type ==
        FormatSelectionType::FORMAT_AGNOSTIC_FOR_PAIRED_INPUT_AND_OUTPUT)) {
    propagation_info.reshape_type = propagation_info_param.reshape_type;
    propagation_info.group = propagation_info_param.group;
    propagation_info.heavy_format = propagation_info_param.heavy_format;
    propagation_info.sub_format = propagation_info_param.sub_format;
    propagation_info.c0_format = propagation_info_param.c0_format;
  }
};

using TensorInfoPtr = std::shared_ptr<FormatAndDataTypeInfo>;

/** @brief First select qualified format index by the current data type of each
* input and out put. */
class HeavyFormatSelector {
 public:
  using PreciseDtypeMatcherPtr = std::shared_ptr<OpDtypePreciseMatcher>;
  explicit HeavyFormatSelector(FormatDtypeQuerierPtr format_dtype_querier_ptr);

  ~HeavyFormatSelector();

  Status Initalize();
  /* Sort the format by loop around all inputs and outputs in ops kernel.
   * Sort format in ops-kernel by the following priority:
   * The highest priority: as same as the input heavy format
   * The second priority: other heavy format
   * The third priority: original format
   * Other: other format
   * And we consider that the input anchor which peer out anchor's owner node
   * is const or variable has more weight, because const or variable can be merged
   * with trans-nodes.
   * The format score is define as below:
   * Same as the heavy format which is distributed right now : 2000;
   * Other heavy format : 200
   * Same as the tensor's original format(ND included) : 100
   * Other format: 1 */
  Status GetMostSuitableFormatIndex(const fe::OpKernelInfoPtr& op_kernel_info_ptr, const ge::NodePtr& current_node,
                                    const HeavyFormatInfo& heavy_format_info,
                                    const std::vector<IndexNameMap>& tensor_map, int32_t& most_suitable_index);

  static bool IsHeavyFormatConsistentWithOriFormat(const ge::GeTensorDescPtr &current_tensor,
      const ge::Format &heavy_format, const ge::DataType &dst_dtype, const ge::OpDescPtr &op_desc_ptr);
 private:
  /* For specifc input or output (which the heavy format is distributed from),
   * we need to ensure its format is exactly the same as the heavy format.
   * So we first select the format index which qualify this requirement. */
  Status SelectQualifiedFormat(const OpKernelInfoPtr& op_kernel_info_ptr, const ge::NodePtr& current_node,
                               const HeavyFormatInfo& heavy_format_info, const std::vector<IndexNameMap>& tensor_map);

  /* Get the initial heavy format index by the heavy format and the
   * corresponding input's kernel info. If and only if the format of
   * the input's or output's kernel is exactly as same as the heavy format and
   * the data type of the input's or output's kernel is exactly as same as the
   * current tensor's data type, we will push it to the matched_index_
   * */
  Status SearchHeavyFormatInKernel(const OpKernelInfoPtr& op_kernel_info_ptr, const ge::NodePtr& current_node,
                                   const HeavyFormatInfo& heavy_format_info);

  /* When doing distribution, we should ensure the data type which selected by
   * opjudge will not change because the precision is much more important
   * than the format. So the data types of the heavy format in the ops kernel
   * should be exactly the same as current ones */
  Status MatchDtypeForAllInputAndOutput(const OpKernelInfoPtr& op_kernel_info_ptr, const ge::NodePtr& current_node);

  Status CalcFormatScore(const ge::OpDesc::Vistor<ge::GeTensorDescPtr>& all_tensors,
                         const fe::OpKernelInfoPtr& op_kernel_info_ptr, const ge::NodePtr &node,
                         uint32_t kernel_format_index, const HeavyFormatInfo& heavy_format_info,
                         InputOrOutputIndex in_or_out, uint64_t& score);

  Status Match(const OpKernelInfoPtr& op_kernel_info_ptr,
               const ge::NodePtr &node, const ge::OpDesc::Vistor<ge::GeTensorDescPtr>& all_tensors,
               InputOrOutputIndex in_or_out);

  static bool IsDtypeSensitiveOpForHeavyFormatConsistentWithOriFormat(const ge::OpDescPtr &op_desc_ptr);

  FormatDtypeQuerierPtr format_dtype_querier_ptr_;

  PreciseDtypeMatcherPtr precise_dtype_matcher_ptr_;

  std::vector<std::vector<InputOrOutputInfoPtr>> input_and_output_kernel_;

  std::vector<uint32_t> matched_index_;
};

Status GetInputAndOutputIndexNameMap(const OpKernelInfoPtr& op_kernel_info_ptr, const ge::NodePtr& current_node,
                                     std::vector<IndexNameMap>& tensor_map);
}  // namespace fe

#endif  // FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_HEAVY_FORMAT_PROPAGATION_HEAVY_FORMAT_SELECTOR_H_

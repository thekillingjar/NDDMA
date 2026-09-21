/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_DSA_OPTIMIZER_FE_GRAPH_OPTIMIZER_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_DSA_OPTIMIZER_FE_GRAPH_OPTIMIZER_H_

#include <map>
#include <memory>
#include <string>
#include "adapter/adapter_itf/op_store_adapter.h"
#include "common/fe_inner_error_codes.h"
#include "common/fe_utils.h"
#include "common/graph/fe_graph_utils.h"
#include "common/optimizer/graph_optimizer.h"
#include "common/optimizer/graph_optimizer_types.h"
#include "format_selector/manager/format_dtype_setter.h"
#include "graph/compute_graph.h"
#include "graph_optimizer/op_judge/format_and_dtype/op_format_dtype_judge.h"
#include "graph_optimizer/op_judge/imply_type/op_impl_type_judge.h"
#include "graph_optimizer/spacesize_calculator/spacesize_calculator.h"
#include "ops_kernel_store/fe_ops_kernel_info_store.h"
#include "common/fe_error_code.h"
#include "base/err_msg.h"

namespace fe {
using OpImplyTypeJudgePtr = std::shared_ptr<OpImplTypeJudge>;
using FormatDtypeSetterPtr = std::shared_ptr<FormatDtypeSetter>;
using OpFormatDtypeJudgePtr = std::shared_ptr<OpFormatDtypeJudge>;
using ReflectionBuilderPtr = std::shared_ptr<ge::RefRelations>;
using SpaceSizeCalculatorPtr = std::shared_ptr<SpaceSizeCalculator>;

class DSAGraphOptimizer : public ge::GraphOptimizer {
 public:
  explicit DSAGraphOptimizer(FEOpsKernelInfoStorePtr fe_ops_kernel_info_store_ptr,
                             std::string engine_name = fe::AI_CORE_NAME);
  ~DSAGraphOptimizer() override;

  /**
   * @ingroup fe
   * @brief prohibit copy and assign construct
   */
  DSAGraphOptimizer(const DSAGraphOptimizer&) = delete;
  DSAGraphOptimizer& operator=(const DSAGraphOptimizer&) = delete;

  /*
   *  @ingroup fe
   *  @brief   initialize graph optimizer
   *  @param   [in] options
   *  @return  SUCCESS or FAILED
   */
  Status Initialize(const std::map<string, string>& options, ge::OptimizeUtility *const optimize_utility) override;

  /*
   *  @ingroup fe
   *  @brief   close graph optimizer
   *  @return  SUCCESS or FAILED
   */
  Status Finalize() override;

  /*
   *  @ingroup fe
   *  @brief   optimize original graph
   *  @param   [in|out] graph  compute graph
   *  @return  SUCCESS or FAILED
   */
  Status OptimizeOriginalGraph(ge::ComputeGraph& graph) override;

  /*
   *  @ingroup fe
   *  @brief   optimize fused graph
   *  @param   [in|out] graph   compute graph
   *  @return  SUCCESS or FAILED
   */
  Status OptimizeFusedGraph(ge::ComputeGraph& graph) override;

  /*
   *  @ingroup fe
   *  @brief   get attribute of graph optimizer
   *  @param   [in|out] attrs
   *  @return  SUCCESS or FAILED
   */
  Status GetAttributes(ge::GraphOptimizerAttribute& attrs) const override;

  Status OptimizeWholeGraph(ge::ComputeGraph& graph) override;

 private:
  void SetInputSplitInfo(AxisSplitMap& axis_split_map, const int8_t& input_index, const int8_t& input_axis) const;
  void SetOutputSplitInfo(AxisSplitMap& axis_split_map, const int8_t& output_index, const int8_t& output_axis) const;
  void SetConstInputsValue(const ge::NodePtr &node) const;
  Status SetDSAOpSliceInfo(const ge::ComputeGraph &graph) const;
  Status SetDSAOpWorkspaces(const ge::ComputeGraph &graph) const;
  Status OptimizeOriginalGraphOpJudgeAndFormatDtypeSetter(ge::ComputeGraph& graph) const;

  // op info mgr
  FEOpsKernelInfoStorePtr ops_kernel_info_store_ptr_;

  // op judge pointer
  OpImplyTypeJudgePtr op_impl_type_judge_ptr_;

  // format_dtype_setter
  FormatDtypeSetterPtr format_dtype_setter_ptr_;

  // Space Size calculator pointer
  SpaceSizeCalculatorPtr space_size_calculator_ptr_;

  // attribute of graph optimizer
  ge::GraphOptimizerAttribute graph_optimizer_attr_;

  // init flag
  bool init_flag_;
};
}  // namespace fe
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_DSA_OPTIMIZER_FE_GRAPH_OPTIMIZER_H_

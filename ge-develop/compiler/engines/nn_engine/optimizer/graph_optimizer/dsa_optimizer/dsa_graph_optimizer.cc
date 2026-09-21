/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/dsa_optimizer/dsa_graph_optimizer.h"
#include <memory>
#include <vector>
#include "common/string_util.h"
#include "graph/ge_context.h"
#include "common/fe_utils.h"
#include "common/aicore_util_constants.h"
#include "common/aicore_util_attr_define.h"
#include "common/aicore_util_types.h"
#include "common/platform_utils.h"

namespace fe {
DSAGraphOptimizer::DSAGraphOptimizer(FEOpsKernelInfoStorePtr fe_ops_kernel_info_store_ptr,
                                     std::string engine_name)
        : ops_kernel_info_store_ptr_(fe_ops_kernel_info_store_ptr),
          op_impl_type_judge_ptr_(nullptr),
          format_dtype_setter_ptr_(nullptr),
          space_size_calculator_ptr_(nullptr),
          graph_optimizer_attr_({engine_name, ge::ENGINE}),
          init_flag_(false) {}

DSAGraphOptimizer::~DSAGraphOptimizer() {}

Status DSAGraphOptimizer::Initialize(const std::map<string, string>& options,
                                     ge::OptimizeUtility *const optimize_utility) {
  (void)options;
  (void)optimize_utility;
  // if graph optimizer has been initialized, return success
  if (init_flag_) {
    FE_LOGW("DSAGraphOptimizer has been initialized.");
    return SUCCESS;
  }

  init_flag_ = true;
  FE_LOGD("Begin initializing DSAGraphOptimizer in engine [%s]", graph_optimizer_attr_.engineName.c_str());
  // initialize op compiler
  FE_CHECK(ops_kernel_info_store_ptr_ == nullptr, FE_LOGE("[GraphOpt][Init] opsKernelInfoStorePtr_ is null."),
           return FAILED);

  FE_MAKE_SHARED(op_impl_type_judge_ptr_ = std::make_shared<OpImplTypeJudge>(graph_optimizer_attr_.engineName,
                                                                             ops_kernel_info_store_ptr_),
                  return GRAPH_OPTIMIZER_MAKE_SHARED_FAILED);

  FE_MAKE_SHARED(format_dtype_setter_ptr_ = std::make_shared<FormatDtypeSetter>(graph_optimizer_attr_.engineName),
                 return GRAPH_OPTIMIZER_MAKE_SHARED_FAILED);
  FE_MAKE_SHARED(space_size_calculator_ptr_ = std::make_shared<SpaceSizeCalculator>(),
                 return GRAPH_OPTIMIZER_MAKE_SHARED_FAILED);

  FE_LOGI("Initialize successfully.");

  return SUCCESS;
}

Status DSAGraphOptimizer::Finalize() {
  if (!init_flag_) {
    FE_LOGW("DSAGraphOptimizer finalize is not allowed, initialize first is necessary.");
    return SUCCESS;
  }

  init_flag_ = false;
  FE_LOGD("Finalized successfully.");

  return SUCCESS;
}

Status DSAGraphOptimizer::OptimizeOriginalGraph(ge::ComputeGraph& graph) {
  if (!init_flag_) {
    REPORT_FE_ERROR("[GraphOpt][init] DSAGraphOptimizer has not been initialized.");
    return FAILED;
  }
  OpFormatDtypeJudgePtr op_format_dtype_judge_ptr = nullptr;
  ReflectionBuilderPtr reflection_builder_ptr = nullptr;

  FE_MAKE_SHARED(reflection_builder_ptr = std::make_shared<ge::RefRelations>(),
                 return GRAPH_OPTIMIZER_MAKE_SHARED_FAILED);
  ops_kernel_info_store_ptr_->SetCheckSupportedStaticFlag(false);
  FE_LOGD("Beginning optimization of the original graph to judge insert graph[%s], in engine[%s]", graph.GetName().c_str(),
          graph_optimizer_attr_.engineName.c_str());

  FE_MAKE_SHARED(op_format_dtype_judge_ptr = std::make_shared<OpFormatDtypeJudge>(
          graph_optimizer_attr_.engineName, reflection_builder_ptr),
                 return GRAPH_OPTIMIZER_MAKE_SHARED_FAILED);
  if (op_format_dtype_judge_ptr->Initialize() != SUCCESS) {
    REPORT_FE_ERROR("[GraphOptJdgInst][Init] Failed to initialize op_format_dtype_judge_ptr for graph[%s].",
                    graph.GetName().c_str());
    return FAILED;
  }

  Status ret = OptimizeOriginalGraphOpJudgeAndFormatDtypeSetter(graph);
  if (ret != SUCCESS) {
    FE_LOGE("[GraphOptJdgInst][JudgeAndFormat] Failed to judge operation implementation or set supported format and data type for graph [%s]!",
            graph.GetName().c_str());
    return ret;
  }
  ret = op_format_dtype_judge_ptr->Judge(graph);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOptJdgInst][Judge] Failed to judge the format and data type of the graph [%s].",
                    graph.GetName().c_str());
    FeGraphUtils::DumpGraphAndOnnx(graph, "OptimizeOriginalGraph_DSAFeOpJudgeAfter_Failed");
    FeGraphUtils::DumpSubGraphAndOnnx(graph, "OptimizeOriginalGraph_DSAFeOpJudgeAfter_Failed_Subgraph");
    return ret;
  }
  FeGraphUtils::DumpGraphAndOnnx(graph, "OptimizeOriginalGraph_DSAFeOpJudgeAfter");
  FeGraphUtils::DumpSubGraphAndOnnx(graph, "OptimizeOriginalGraph_DSAFeOpJudgeAfter_Subgraph");
  FE_LOGI("Optimized original graph [%s]: Successfully determined the format and data type of input and output descriptors for the operation.",
          graph.GetName().c_str());

  ops_kernel_info_store_ptr_->SetCheckSupportedStaticFlag(true);
  return SUCCESS;
}

Status DSAGraphOptimizer::OptimizeOriginalGraphOpJudgeAndFormatDtypeSetter(ge::ComputeGraph& graph) const {
  Status ret;
  // set the highest prior imply type for op
  ret = op_impl_type_judge_ptr_->Judge(graph);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOptJdgInst][Judge] Failed to judge the op implementation, graph[%s].", graph.GetName().c_str());
    return ret;
  }
  FE_LOGI("Optimizing original graph[%s] judge op implementation successfully.", graph.GetName().c_str());

  ret = format_dtype_setter_ptr_->SetSupportFormatDtype(graph);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOptJdgInst][SetFormat] Failed to set the supported format and dtype information for graph [%s].",
                    graph.GetName().c_str());
    return ret;
  }
  FE_LOGI("Optimizing original graph[%s] set the support format and dtype information successfully.",
          graph.GetName().c_str());
  return SUCCESS;
}

void DSAGraphOptimizer::SetInputSplitInfo(AxisSplitMap& axis_split_map, const int8_t& input_index,
                                          const int8_t& input_axis) const {
  InputSplitInfo input_split_info;
  if (!input_split_info.Initialize()) {
    REPORT_FE_ERROR("[GraphOptJdgInst][SetOpInfo][SetInSplitInfo] Failed to initialize input_split_info.");
    return;
  }
  input_split_info.SetIndex(input_index);
  std::vector<int64_t> input_vec_first_tensor = {input_axis};
  input_split_info.SetAxis(input_vec_first_tensor);
  axis_split_map.AddInputSplitInfo(input_split_info);
}

void DSAGraphOptimizer::SetOutputSplitInfo(AxisSplitMap& axis_split_map, const int8_t& output_index,
                                           const int8_t& output_axis) const {
  OutputSplitInfo output_split_info;
  if (!output_split_info.Initialize()) {
    REPORT_FE_ERROR("[GraphOptJdgInst][SetOpInfo][SetOutSplitInfo] Failed to initialize output_split_info");
    return;
  }
  output_split_info.SetIndex(output_index);
  std::vector<int64_t> output_vec = {output_axis};
  output_split_info.SetAxis(output_vec);
  axis_split_map.AddOutputSplitInfo(output_split_info);
}

Status DSAGraphOptimizer::SetDSAOpSliceInfo(const ge::ComputeGraph &graph) const {
  auto nodes = graph.GetDirectNode();
  for (auto &node : nodes) {
    ge::OpDescPtr op_desc = node->GetOpDesc();
    FE_CHECK_NOTNULL(op_desc);
    int64_t imply_type = -1;
    (void)ge::AttrUtils::GetInt(op_desc, FE_IMPLY_TYPE, imply_type);
    OpImplType op_impl_type = GetMainImplType<OpImplType>(imply_type);
    if (op_impl_type == EN_IMPL_HW_DSA) {
      OpCalcInfo op_calc_info;
      if (!op_calc_info.Initialize()) {
        REPORT_FE_ERROR("[DSAOptimizer][SetSliceInfo] op_calc_info initialize failed.");
        return FAILED;
      }
      std::vector<AxisSplitMap> axis_split_maps;
      AxisSplitMap axis_split_map;
      if (!axis_split_map.Initialize()) {
        REPORT_FE_ERROR("[DSAOptimizer][SetSliceInfo] axis_split_map initialize failed.");
        return FAILED;
      }
      size_t input_size = node->GetInDataNodes().size();
      size_t output_size = node->GetOutDataNodes().size();
      for (size_t i = 0; i < input_size; i++) {
        SetInputSplitInfo(axis_split_map, i, 0);
      }
      for (size_t i = 0; i < output_size; i++) {
        SetOutputSplitInfo(axis_split_map, i, 0);
      }
      axis_split_maps.push_back(axis_split_map);
      op_calc_info.SetAxisSplitMaps(axis_split_maps);
      std::string op_calc_info_str;
      SetOpSliceInfoToJson(op_calc_info, op_calc_info_str);
      (void)ge::AttrUtils::SetStr(op_desc, OP_SLICE_INFO, op_calc_info_str);
    }
  }
  return SUCCESS;
}

Status DSAGraphOptimizer::OptimizeWholeGraph(ge::ComputeGraph& graph) {
  FE_LOGD("Begin optimizing the entire graph in engine [%s].", graph_optimizer_attr_.engineName.c_str());
  auto nodes = graph.GetAllNodes();
  for (const auto &node : nodes) {
    ge::OpDescPtr op_desc = node->GetOpDesc();
    FE_CHECK_NOTNULL(op_desc);
    int64_t imply_type = -1;
    (void)ge::AttrUtils::GetInt(op_desc, FE_IMPLY_TYPE, imply_type);
    OpImplType op_impl_type = GetMainImplType<OpImplType>(imply_type);
    if (op_impl_type == EN_IMPL_HW_DSA) {
      SetConstInputsValue(node);
    }
  }
  return SUCCESS;
}

Status DSAGraphOptimizer::SetDSAOpWorkspaces(const ge::ComputeGraph &graph) const {
  auto nodes = graph.GetAllNodes();
  for (auto &node : nodes) {
    ge::OpDescPtr op_desc = node->GetOpDesc();
    FE_CHECK_NOTNULL(op_desc);
    int64_t imply_type = -1;
    (void)ge::AttrUtils::GetInt(op_desc, FE_IMPLY_TYPE, imply_type);
    OpImplType op_impl_type = GetMainImplType<OpImplType>(imply_type);
    if (op_impl_type == EN_IMPL_HW_DSA) {
      int64_t count_workspace_size = PlatformUtils::Instance().GetDsaWorkspaceSize();
      std::vector<int64_t> tvm_workspace_sizes = {count_workspace_size};
      std::vector<int32_t> memory_no_reuse = {1};
      if (!kDSAStatelessOps.count(node->GetType())) {
        tvm_workspace_sizes.push_back(count_workspace_size);
        memory_no_reuse.push_back(1);
      }
      op_desc->SetWorkspaceBytes(tvm_workspace_sizes);
      (void)ge::AttrUtils::SetListInt(op_desc, ge::ATTR_NAME_WORKSPACE_MEMORY_NO_REUSE_SCOPE, memory_no_reuse);
      FE_LOGD("Set workspace bytes to dsa node[name:%s, type:%s] successfully.",
              node->GetName().c_str(), node->GetType().c_str());
    }
  }
  return SUCCESS;
}

Status DSAGraphOptimizer::OptimizeFusedGraph(ge::ComputeGraph& graph) {
  FE_LOGD("Begin optimizing fused graph in engine [%s].", graph_optimizer_attr_.engineName.c_str());

  if (!init_flag_) {
    FE_LOGW("OptimizeFusedGraph is not permitted; initialization must be performed first.");
    return FAILED;
  }
  FE_TIMECOST_START(OptimizeFusedGraph);
  // calculate the input and output size of op.
  FE_CHECK(space_size_calculator_ptr_ == nullptr,
           REPORT_FE_ERROR("[GraphOpt][PostProcess][CalcuRunPara] The spaceSizeCalculatorPtr is null."),
           return FAILED);
  Status ret = space_size_calculator_ptr_->CalculateRunningParams(graph);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][PostProcess][CalcuRunPara] Failed to calculate running parameters for graph [%s]",
                    graph.GetName().c_str());
    return ret;
  }

  if (SetDSAOpWorkspaces(graph) != SUCCESS) {
    return FAILED;
  }
  FE_TIMECOST_END(OptimizeFusedGraph, "DSAGraphOptimizer::OptimizeFusedGraph");
  return SUCCESS;
}

Status DSAGraphOptimizer::GetAttributes(ge::GraphOptimizerAttribute& attrs) const {
  attrs = graph_optimizer_attr_;
  return SUCCESS;
}

void DSAGraphOptimizer::SetConstInputsValue(const ge::NodePtr &node) const {
  auto op = ge::OpDescUtils::CreateOperatorFromNode(node);
  std::vector<std::string> value_list;
  auto input_desc_size = node->GetOpDesc()->GetAllInputsSize();
  for (size_t input_idx = 0; input_idx < input_desc_size; input_idx++) {
    std::string value;
    auto const_tensor = ge::OpDescUtils::GetInputConstData(op, input_idx);
    if (const_tensor == nullptr) {
      FE_LOGW("[DSAOptimizer][SetConstInputsValue][node_name %s] get const data failed.",
              node->GetName().c_str());
      value_list.emplace_back(value);
      continue;
    }
    value.assign(reinterpret_cast<const char *>(const_tensor->GetData().GetData()), const_tensor->GetData().GetSize());
    value_list.emplace_back(value);
  }
  FE_LOGD("[DSAOptimizer][SetConstInputsValue][node_name %s] get const data successfully.", node->GetName().c_str());
  (void)ge::AttrUtils::SetListStr(node->GetOpDesc(), kOpConstValueList, value_list);
}
}  // namespace fe

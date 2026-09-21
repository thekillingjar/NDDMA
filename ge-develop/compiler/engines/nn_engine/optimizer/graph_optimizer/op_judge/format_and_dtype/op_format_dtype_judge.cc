/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/op_judge/format_and_dtype/op_format_dtype_judge.h"
#include <utility>
#include "common/fe_inner_attr_define.h"
#include "common/fe_utils.h"
#include "common/graph/fe_graph_utils.h"
#include "common/fe_op_info_common.h"
#include "common/util/op_info_util.h"
#include "common/configuration.h"
#include "common/math_util.h"
#include "param_calculate/tensorsize_calculator.h"
#include "ffts_type.h"
#include "ops_store/ops_kernel_manager.h"
#include "ops_kernel_store/sub_ops_store.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_attr_value.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/op_type_utils.h"
#include "common/fe_inner_attr_define.h"
#include "framework/common/ge_types.h"
#include "register/graph_optimizer/fusion_common/unknown_shape_utils.h"
#include "graph/ge_local_context.h"
#include "external/op_common/data_type_utils.h"
#include "adapter/common/op_store_adapter_manager.h"

namespace fe {
static const std::string kStageSetDtypeFmtByPrecisMode = "[GraphOptJdgInst][UpdFmtAndDtype][SetDtypeFmtByPrecisMode]";
static vector<vector<int>> stub_list = {{0, 1}};
static vector<vector<int>> stub_axpyv2_list = {{0, 1, 2}};
static std::unordered_map<std::string, vector<vector<int>>> kPromoteTypeStubMap = {
  std::make_pair("Mul", stub_list), std::make_pair("Pow", stub_list), std::make_pair("AxpyV2", stub_axpyv2_list)
};

OpFormatDtypeJudge::OpFormatDtypeJudge(const std::string& engine_name, RefRelationsPtr reflection_builder_ptr)
    : OpJudgeBase(engine_name),
      reflection_builder_ptr_(reflection_builder_ptr),
      op_format_dtype_strategy_manager_ptr_(nullptr),
      op_format_dtype_update_desc_ptr_(nullptr),
      format_dtype_querier_ptr_(nullptr),
      sub_data_format_dtype_update_ptr_(nullptr),
      sub_net_output_format_dtype_update_ptr_(nullptr) {}

OpFormatDtypeJudge::~OpFormatDtypeJudge() {}

void OpFormatDtypeJudge::SetPrecisionMode(const PrecisionMode &precision_mode) {
  op_format_dtype_strategy_manager_ptr_->SetPrecisionMode(precision_mode);
}

Status OpFormatDtypeJudge::Initialize() {
  FE_MAKE_SHARED(format_dtype_querier_ptr_ = std::make_shared<FormatDtypeQuerier>(engine_name_),
                 return FAILED);
  FE_CHECK_NOTNULL(format_dtype_querier_ptr_);

  FE_MAKE_SHARED(op_format_dtype_strategy_manager_ptr_ = std::make_shared<OpFormatDtypeStrategyManager>(engine_name_,
    format_dtype_querier_ptr_), return FAILED);
  FE_CHECK_NOTNULL(op_format_dtype_strategy_manager_ptr_);
  Status ret = op_format_dtype_strategy_manager_ptr_->Initialize();
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOptJdgInst][UpdFmtAndDtype][Init] Failed to initialize OpFormatDtypeStrategyManager!");
    return ret;
  }

  FE_MAKE_SHARED(op_format_dtype_update_desc_ptr_ =
                 std::make_shared<OpFormatDtypeUpdateDesc>(format_dtype_querier_ptr_), return FAILED);
  FE_CHECK_NOTNULL(op_format_dtype_update_desc_ptr_);
  ret = op_format_dtype_update_desc_ptr_->Initialize();
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOptJdgInst][UpdFmtAndDtype][Init] Failed to initialize OpFormatDtypeUpdateDesc!");
    return ret;
  }

  FE_MAKE_SHARED(sub_data_format_dtype_update_ptr_ =
                 std::make_shared<SubDataFormatDtypeUpdate>(reflection_builder_ptr_), return FAILED);
  FE_CHECK_NOTNULL(sub_data_format_dtype_update_ptr_);

  FE_MAKE_SHARED(sub_net_output_format_dtype_update_ptr_ =
                 std::make_shared<SubNetOutputFormatDtypeUpdate>(reflection_builder_ptr_), return FAILED);
  FE_CHECK_NOTNULL(sub_net_output_format_dtype_update_ptr_);

  return SUCCESS;
}

Status OpFormatDtypeJudge::Judge(ge::ComputeGraph &graph) {
  FE_TIMECOST_START(OpFormatDtypeJudge);
  for (auto &node : graph.GetAllNodes()) {
    Status status = JudgeByNode(node);
    if (status != SUCCESS) {
      return status;
    }
  }
  FE_TIMECOST_END(OpFormatDtypeJudge, "OpDtypeJudge during FEGraphOptimizer::OptimizeOriginalGraph");
  return SUCCESS;
}

Status OpFormatDtypeJudge::SetFormat(ge::ComputeGraph &graph) {
  FE_TIMECOST_START(OpFormatFormatJudge);
  for (auto &node_ptr : graph.GetAllNodes()) {
    FE_CHECK_NOTNULL(node_ptr);
    ge::OpDescPtr op_desc_ptr = node_ptr->GetOpDesc();
    FE_CHECK_NOTNULL(op_desc_ptr);

    if (ge::OpTypeUtils::IsAutofuseNode(op_desc_ptr)) {
      FE_LOGD("Check auto fuse node %s success.", op_desc_ptr->GetNamePtr());
      continue;
    }
    int64_t imply_type = -1;
    if (IsNoNeedJudge(op_desc_ptr, imply_type)) {
      continue;
    }

    FE_LOGD("Node[%s, %s]: update format info.", op_desc_ptr->GetNamePtr(), op_desc_ptr->GetTypePtr());
    OpImplType op_imply_type = static_cast<OpImplType>(imply_type);
    FEOpsStoreInfo op_store_info;
    if (Configuration::Instance(engine_name_).GetOpStoreInfoByImplType(op_imply_type, op_store_info) != SUCCESS) {
      FE_LOGW("Engine[%s] Op[name=%s, type=%s]: Failed to get the op store info by imply_type %ld.", engine_name_.c_str(),
      op_desc_ptr->GetNamePtr(), op_desc_ptr->GetTypePtr(), op_imply_type);
      continue;
    }
    string imply_type_str = op_store_info.fe_ops_store_name;

    Status ret = SetFormatByJudgeResult(node_ptr, imply_type_str);
    if (ret != SUCCESS) {
      REPORT_FE_ERROR("[SetFormat][UpdFmt] Node [%s, %s]: failed to set format.",
                      op_desc_ptr->GetNamePtr(), op_desc_ptr->GetTypePtr());
      return ret;
    }
  }
  FE_TIMECOST_END(OpFormatFormatJudge, "OpFormatJudge during FEGraphOptimizer::OptimizeOriginalGraph");
  return SUCCESS;
}

Status OpFormatDtypeJudge::JudgeByNode(ge::NodePtr node_ptr) {
  FE_CHECK_NOTNULL(node_ptr);
  ge::OpDescPtr op_desc_ptr = node_ptr->GetOpDesc();
  FE_CHECK_NOTNULL(op_desc_ptr);
  auto owner_graph = node_ptr->GetOwnerComputeGraph();
  FE_CHECK_NOTNULL(owner_graph);

  string graph_name = owner_graph->GetName();
  string op_name = op_desc_ptr->GetName();
  string op_type = op_desc_ptr->GetType();

  // 1. subgraph data
  if (FeGraphUtils::IsSubGraphData(op_desc_ptr)) {
    FE_LOGD("Graph[%s] Op [name=%s, type=%s]: This node is the data of the subgraph.", graph_name.c_str(), op_name.c_str(),
            op_type.c_str());
    (void)sub_data_format_dtype_update_ptr_->UpdateTensorDesc(node_ptr);
  }

  // 2. subgraph netoutput
  if (FeGraphUtils::IsSubGraphNetOutput(op_desc_ptr)) {
    FE_LOGD("Graph[%s] Op[name=%s, type=%s]: This node is the netoutput of the subgraph.", graph_name.c_str(),
            op_name.c_str(), op_type.c_str());
    (void)sub_net_output_format_dtype_update_ptr_->UpdateTensorDesc(node_ptr);
  }

  // 3. no need judge
  int64_t imply_type = -1;
  if (IsNoNeedJudge(op_desc_ptr, imply_type)) {
    return SUCCESS;
  }

  // 5. get op kernel info store name
  FE_LOGD("Graph[%s], Node[%s, %s]: judge the format and the dtype.",
          graph_name.c_str(), op_name.c_str(), op_type.c_str());
  OpImplType op_imply_type = static_cast<OpImplType>(imply_type);
  FEOpsStoreInfo op_store_info;
  if (Configuration::Instance(engine_name_).GetOpStoreInfoByImplType(op_imply_type, op_store_info) != SUCCESS) {
    FE_LOGW("Engine[%s] Op[name=%s, type=%s]: Failed to get the op store info by imply_type %ld.", engine_name_.c_str(),
            op_name.c_str(), op_type.c_str(), op_imply_type);
    return SUCCESS;
  }
  string imply_type_str = op_store_info.fe_ops_store_name;

  // 6. set input and output data type and format
  Status ret = SetDtypeByPrecisionMode(node_ptr, imply_type_str, op_imply_type);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOptJdgInst][UpdDtype][JdgByNd] Failed to set dtype for Graph[%s], Node[%s, %s].",
                    graph_name.c_str(), op_name.c_str(), op_type.c_str());
    return ret;
  }

  return SUCCESS;
}

uint32_t OpFormatDtypeJudge::GetMatchedIndex(const vector<uint32_t> &matched_index_vec,
                                             const vector<uint32_t> &cust_filter_matched_index_vec,
                                             const string &op_name,
                                             const string &op_type) const {
  if (matched_index_vec.empty()) {
    // no matching dtype or format for this op, so we pick the first column of
    // dtype and format
    FE_LOGW("Op[name=%s,type=%s]: the dtype and format is different from the op store, set the matched index to be 0.",
        op_name.c_str(), op_type.c_str());
    return cust_filter_matched_index_vec.empty() ? 0 : cust_filter_matched_index_vec[0];
  } else {
    FE_LOGD(
        "Op[name=%s,type=%s]: the size of the matched_index_vec is %zu,"
        "get the first matched index %u.", op_name.c_str(), op_type.c_str(),
        matched_index_vec.size(), matched_index_vec[0]);
    return matched_index_vec[0];
  }
}

bool OpFormatDtypeJudge::IsNodeSupport16In32out(ge::NodePtr node_ptr, const OpKernelInfoPtr &op_kernel_info_ptr,
                                                const IndexNameMap &input_index_map,
                                                const IndexNameMap &output_index_map) {
  ge::OpDescPtr op_desc_ptr = node_ptr->GetOpDesc();

  InputOrOutputInfoPtr input_tensor_kernel_info_ptr = nullptr;
  InputOrOutputInfoPtr output_tensor_kernel_info_ptr = nullptr;
  if ((input_index_map.find(0) == input_index_map.end()) || (output_index_map.find(0) == output_index_map.end())) {
    FE_LOGW("Node[type=%s, name=%s]: Could not find input or output index 0 in index_map.",
            op_desc_ptr->GetType().c_str(), op_desc_ptr->GetName().c_str());
    return false;
  }
  Status ret_input = op_kernel_info_ptr->GetTensorInfoByName(true, input_index_map.at(0),
                                                             input_tensor_kernel_info_ptr);
  if (ret_input != SUCCESS || input_tensor_kernel_info_ptr == nullptr) {
    FE_LOGW("Node[type=%s,name=%s]: the input is not found in the ops store.", op_desc_ptr->GetType().c_str(),
            op_desc_ptr->GetName().c_str());
    return false;
  }

  Status ret_output = op_kernel_info_ptr->GetTensorInfoByName(false, output_index_map.at(0),
                                                              output_tensor_kernel_info_ptr);
  if (ret_output != SUCCESS || output_tensor_kernel_info_ptr == nullptr) {
    FE_LOGW("Node[type=%s,name=%s]: the output is not found in the ops store.", op_desc_ptr->GetType().c_str(),
            op_desc_ptr->GetName().c_str());
    return false;
  }
  std::vector<ge::DataType> input_dtype_vec;
  std::vector<ge::DataType> output_dtype_vec;
  if (format_dtype_querier_ptr_->GetSupportDataTypes(op_kernel_info_ptr, input_tensor_kernel_info_ptr,
                                                     node_ptr, input_dtype_vec) != SUCCESS) {
    return false;
  }

  if (format_dtype_querier_ptr_->GetSupportDataTypes(op_kernel_info_ptr, output_tensor_kernel_info_ptr,
                                                     node_ptr, output_dtype_vec) != SUCCESS) {
    return false;
  }
  FE_CHECK((input_dtype_vec.size() > output_dtype_vec.size()), FE_LOGW("Input type size exceeds output."), return false);
  // check whether support fp16 in and fp32 out
  for (size_t i = 0; i < input_dtype_vec.size(); i++) {
    if ((input_dtype_vec[i] == ge::DT_FLOAT16) && (output_dtype_vec[i] == ge::DT_FLOAT)) {
      (void) ge::AttrUtils::SetBool(op_desc_ptr, kAttrSupported_16In_32Out, true);
      return true;
    }
  }
  return false;
}

Status OpFormatDtypeJudge::SetSupport16In32outAttr(ge::NodePtr node_ptr, const OpKernelInfoPtr &op_kernel_info_ptr,
                                                   const IndexNameMap &input_index_map,
                                                   const IndexNameMap &output_index_map) {
  if (IsNodeSupport16In32out(node_ptr, op_kernel_info_ptr, input_index_map, output_index_map)) {
    return ge::AttrUtils::SetBool(node_ptr->GetOpDesc(), kAttrSupported_16In_32Out, true);
  }
  return ge::AttrUtils::SetBool(node_ptr->GetOpDesc(), kAttrSupported_16In_32Out, false);
}

void OpFormatDtypeJudge::FindPromoteDtype(const std::vector<ge::DataType> &input_dtype_vec,
    const ge::DataType &target_dtype, std::vector<uint32_t> &matched_index_vec) {
  std::vector<uint32_t> promote_vec_idx;
  std::vector<uint32_t> matched_idx;
  for (const uint32_t &idx : matched_index_vec) {
    ge::DataType kernel_dtype = input_dtype_vec[idx];
    if (target_dtype == kernel_dtype) {
      matched_idx.emplace_back(idx);
      continue;
    }
    ge::DataType promoted_dtype = opcommon::PromoteType(target_dtype, kernel_dtype);
    if (promoted_dtype != ge::DT_UNDEFINED) {
      if (promoted_dtype == kernel_dtype) {
        promote_vec_idx.emplace_back(idx);
      }
    }
  }
  matched_idx.insert(matched_idx.end(), promote_vec_idx.begin(), promote_vec_idx.end());
  if (!matched_idx.empty()) {
    matched_index_vec.swap(matched_idx);
    return;
  }
}

Status OpFormatDtypeJudge::MatchForPromoteInput(const OpJudgeParam &op_judge_param,
    const std::vector<int> &promote_index, const ge::DataType &target_dtype,
    vector<uint32_t> &matched_index_vec) const {
  for (const auto &index : promote_index) {
    const auto &input_desc = op_judge_param.node_ptr->GetOpDesc()->MutableInputDesc(index);
    auto tensor_iter = op_judge_param.input_index_map.find(index);
    if (tensor_iter == op_judge_param.input_index_map.end()) {
      FE_LOGW("[GraphOpt][FmtJdg][MatPromote][Op %s,type=%s]: input index %d is not found in the ops store.",
              op_judge_param.node_ptr->GetNamePtr(), op_judge_param.node_ptr->GetTypePtr(), index);
      return OP_JUDGE_MAP_KEY_FIND_FAILED;
    }
    InputOrOutputInfoPtr tensor_kernel_info_ptr = nullptr;
    Status ret = op_judge_param.op_kernel_info_ptr->GetTensorInfoByName(true, tensor_iter->second,
                                                                        tensor_kernel_info_ptr);
    if (ret != SUCCESS) {
      FE_LOGW("[GraphOpt][FmtJdg][MatPromote][Op %s,type=%s]: op_kernel is not found in the ops store.",
              op_judge_param.node_ptr->GetNamePtr(), op_judge_param.node_ptr->GetTypePtr());
      return ret;
    }
    FE_CHECK_NOTNULL(tensor_kernel_info_ptr);
    vector<ge::DataType> input_dtype_vec;
    if (format_dtype_querier_ptr_->GetSupportDataTypes(op_judge_param.op_kernel_info_ptr,
        tensor_kernel_info_ptr, op_judge_param.node_ptr, input_dtype_vec) != SUCCESS) {
      FE_LOGW("[GraphOpt][DtypeJdg][MatPromote] Failed to get the support data_types.");
      return FAILED;
    }
    if (matched_index_vec.empty()) {
      for (uint32_t i = 0; i < input_dtype_vec.size(); ++i) {
        matched_index_vec.emplace_back(i);
      }
    }
    std::stringstream ss0;
    ss0 << "[";
    for (const auto &item : input_dtype_vec) {
      ss0 << item << ", ";
    }
    ss0 << "]";
    ss0 << ", matched_index_vec is: [";
    for (const auto &item : matched_index_vec) {
      ss0 << item << ", ";
    }
    ss0 << "]";
    FE_LOGD("Node[%s, %s] ready to find promote type, input dtype vec is %s.",
            op_judge_param.node_ptr->GetNamePtr(), op_judge_param.node_ptr->GetTypePtr(), ss0.str().c_str());
    FindPromoteDtype(input_dtype_vec, target_dtype, matched_index_vec);
    std::stringstream ss1;
    ss1 << "After promote match, matched_index_vec is: [";
    for (const auto &item : matched_index_vec) {
      ss1 << item << ", ";
    }
    ss1 << "]";
    FE_LOGD("Node[%s, %s] %s.",
    op_judge_param.node_ptr->GetNamePtr(), op_judge_param.node_ptr->GetTypePtr(), ss1.str().c_str());
  }
  return SUCCESS;
}

Status OpFormatDtypeJudge::PromoteTypeMatch(const OpJudgeParam &op_judge_param,
    const std::vector<std::vector<int>> &promote_inputs_indexes, vector<uint32_t> &matched_index_vec) const {
  FE_CHECK_NOTNULL(op_judge_param.node_ptr);
  auto op_desc_ptr = op_judge_param.node_ptr->GetOpDesc();
  const string &op_name = op_desc_ptr->GetName();
  const string &op_type = op_desc_ptr->GetType();
  for (const auto &promote_index : promote_inputs_indexes) {
    FE_LOGD("Node[%s, %s] promote index size is [%zu].",
        op_judge_param.node_ptr->GetNamePtr(), op_judge_param.node_ptr->GetTypePtr(),
        promote_index.size());
    if (promote_index.size() < kMinPromoteSize) {
      continue;
    }

    const auto &input_desc_0 = op_desc_ptr->MutableInputDesc(promote_index[0]);
    const auto &input_desc_1 = op_desc_ptr->MutableInputDesc(promote_index[1]);
    if (input_desc_0 == nullptr || input_desc_1 == nullptr) {
      FE_LOGW("Node[%s, %s] get inputDesc by promote input idx [%u, %u] is nullptr.",
              op_desc_ptr->GetNamePtr(), op_desc_ptr->GetTypePtr(), promote_index[0], promote_index[1]);
      return FAILED;
    }

    auto promote_dtype_0 = input_desc_0->GetDataType();
    auto promote_dtype_1 = input_desc_1->GetDataType();
    ge::DataType target_dtype = opcommon::PromoteType(promote_dtype_0, promote_dtype_1);
    bool promote_flag = false;
    if (target_dtype != promote_dtype_0 || target_dtype != promote_dtype_1) {
      promote_flag = true;
    }

    size_t inputs_size = op_desc_ptr->GetAllInputsSize();
    for (size_t i = 2; i < promote_index.size(); ++i) {
      if (static_cast<size_t>(promote_index[i]) >= inputs_size) {
        FE_LOGW("Node[%s, %s] promote input idx [%u] is invalid, while inputs size is %zu.",
                op_desc_ptr->GetNamePtr(), op_desc_ptr->GetTypePtr(), promote_index[i], inputs_size);
        return FAILED;
      }
      const auto &input_desc = op_desc_ptr->MutableInputDesc(promote_index[i]);
      if (input_desc == nullptr) {
        FE_LOGW("Node[%s, %s] get inputDesc by promote input idx [%u] is nullptr.",
                op_desc_ptr->GetNamePtr(), op_desc_ptr->GetTypePtr(), promote_index[i]);
        return FAILED;
      }
      auto cur_dtype = input_desc->GetDataType();
      if (target_dtype != cur_dtype) {
        promote_flag = true;
        target_dtype = opcommon::PromoteType(cur_dtype, target_dtype);
      }
    }
    FE_LOGD("Node[%s, %s], promote flag is %d, target dtype is [%d].",
            op_judge_param.node_ptr->GetNamePtr(), op_judge_param.node_ptr->GetTypePtr(),
            promote_flag, target_dtype);
    if (!promote_flag) {
      continue;
    }
    Status res = MatchForPromoteInput(op_judge_param, promote_index, target_dtype, matched_index_vec);
    if (res != SUCCESS || matched_index_vec.empty()) {
      FE_LOGW("Node [%s, %s] failed to match promote type for input or matched_index_vec is null.",
              op_desc_ptr->GetNamePtr(), op_desc_ptr->GetTypePtr());
      return FAILED;
    }
  }
  return SUCCESS;
}

void OpFormatDtypeJudge::DoPromoteMatch(const OpJudgeParam &op_judge_param, bool must_promote_flag,
    vector<uint32_t> &matched_index_vec) {
  if (matched_index_vec.empty() || must_promote_flag) {
    std::vector<std::vector<int>> promote_info;
    promote_info = op_judge_param.node_ptr->GetOpDesc()->TryGetExtAttr(kPromoteInfo, promote_info);
    if (promote_info.empty()) {
      FE_LOGW("Node[%s, %s] get promote input list from op info is empty.",
              op_judge_param.node_ptr->GetNamePtr(), op_judge_param.node_ptr->GetTypePtr());
      return;
    }
    Status ret = PromoteTypeMatch(op_judge_param, promote_info, matched_index_vec);
    if (ret != SUCCESS || matched_index_vec.empty()) {
      FE_LOGW("[GraphOptJdgInst][DoPromoteMatch] Node[%s, %s]: failed to promote type match.",
              op_judge_param.node_ptr->GetNamePtr(), op_judge_param.node_ptr->GetTypePtr());
      return;
    }
    FE_LOGD("Node[%s, %s] promote dtype match finished, ready to judge format.", op_judge_param.node_ptr->GetNamePtr(),
            op_judge_param.node_ptr->GetTypePtr());
    ret = GetInputAndOutputFormatIndex(op_judge_param, matched_index_vec);
    if (ret != SUCCESS) {
      FE_LOGW("[GraphOptJdgInst][UpdFmtAndDtype][DoPromoteMatch] Node[%s, %s]: failed to retrieve the index for format and data type.",
              op_judge_param.node_ptr->GetNamePtr(), op_judge_param.node_ptr->GetTypePtr());
      return;
    }
    FE_LOGD("Node[%s, %s] judge format finished, matched index vec size is [%zu].",
            op_judge_param.node_ptr->GetNamePtr(), op_judge_param.node_ptr->GetTypePtr(),
            matched_index_vec.size());
  }
}

Status OpFormatDtypeJudge::JudgeByPrecisionMode(const OpJudgeParam &op_judge_param, bool must_promote_flag,
    vector<uint32_t> &matched_index_vec) {
  if (must_promote_flag) {
    FE_LOGD("Node[%s, %s] must do promote match, no need to judge by precision mode.",
            op_judge_param.node_ptr->GetNamePtr(), op_judge_param.node_ptr->GetTypePtr());
    return SUCCESS;
  }
  // 1. get all supported data type index of all inputs and outputs first
  Status ret = GetInputAndOutputDtypeIndex(op_judge_param.node_ptr, op_judge_param.op_kernel_info_ptr,
                                    op_judge_param.input_index_map, op_judge_param.output_index_map,
                                    op_judge_param.prio_index_map, matched_index_vec);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][FmtJdg][SetDtypeFmtByPrecisMode][Op %s,type=%s]: failed to get format and dtype index.",
                    op_judge_param.node_ptr->GetNamePtr(), op_judge_param.node_ptr->GetTypePtr());
    return ret;
  }
  /* 2. get supported format index by original format then */
  ret = GetInputAndOutputFormatIndex(op_judge_param, matched_index_vec);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("%s Node[%s, %s]: Failed to obtain the index for format and dtype.", kStageSetDtypeFmtByPrecisMode.c_str(),
                    op_judge_param.node_ptr->GetNamePtr(), op_judge_param.node_ptr->GetTypePtr());
    return ret;
  }
  return SUCCESS;
}

Status OpFormatDtypeJudge::SetDtypeByPrecisionMode(ge::NodePtr node_ptr, const std::string &imply_type_str,
                                                   const OpImplType &op_imply_type) {
  FE_CHECK_NOTNULL(node_ptr);
  ge::OpDescPtr op_desc_ptr = node_ptr->GetOpDesc();
  FE_CHECK_NOTNULL(op_desc_ptr);

  // 1. get the op_kernel_info_ptr
  OpKernelInfoPtr op_kernel_info_ptr =
      OpsKernelManager::Instance(engine_name_).GetOpKernelInfoByOpType(imply_type_str, op_desc_ptr->GetType());
  FE_CHECK_NOTNULL(op_kernel_info_ptr);
  OpStoreAdapterPtr op_store_adapter =
      OpStoreAdapterManager::Instance(engine_name_).GetOpStoreAdapter(op_imply_type);
  FE_CHECK_NOTNULL(op_store_adapter);
  const string &op_name = op_desc_ptr->GetName();
  const string &op_type = op_desc_ptr->GetType();

  if (op_store_adapter->IsNeedSkipOpJudge(node_ptr, op_kernel_info_ptr)) {
    FE_LOGI("Node[%s, %s]: need to skip judge format and dtype.", op_name.c_str(), op_type.c_str());
    return SUCCESS;
  }

  OpJudgeParam op_judge_param(node_ptr, op_kernel_info_ptr);
  // 2. get input index name map
  Status ret = GetInputIndexNameMap(*(op_desc_ptr.get()), *op_kernel_info_ptr, op_judge_param.input_index_map);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("%s Node[%s, %s]: failed to get input index name map.", kStageSetDtypeFmtByPrecisMode.c_str(),
                    op_name.c_str(), op_type.c_str());
    return ret;
  }

  // 3. get output index name map
  ret = GetOutputIndexNameMap(*(op_desc_ptr.get()), *op_kernel_info_ptr, op_judge_param.output_index_map);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("%s Node[%s, %s]: failed to obtain output index name map.", kStageSetDtypeFmtByPrecisMode.c_str(),
                    op_name.c_str(), op_type.c_str());
    return ret;
  }

  // 4. get the matched_index_vec according to inputs
  vector<uint32_t> matched_index_vec;
  ret = InitMatchedIndexByCustomDtype(node_ptr, op_kernel_info_ptr, op_judge_param.input_index_map,
      op_judge_param.output_index_map, matched_index_vec);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("%s Node[%s, %s]: failed to filter matched index by customize dtypes.",
                    kStageSetDtypeFmtByPrecisMode.c_str(), op_name.c_str(), op_type.c_str());
    return ret;
  }
  // save a copy of matched_index_vec
  vector<uint32_t> cust_filter_matched_index_vec = matched_index_vec;

  // 5. sort inputs
  SortInputBySequence(node_ptr, op_desc_ptr, op_judge_param.prio_index_map);
  bool must_promote_flag = false;
  (void)ge::AttrUtils::GetBool(op_desc_ptr, kMustPromoteFlag, must_promote_flag);
  // 6. get all supported data type / format index of all inputs and outputs first
  ret = JudgeByPrecisionMode(op_judge_param, must_promote_flag, matched_index_vec);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOpt][FmtJdg][SetDtypeFmtByPrecisMode][Op %s,type=%s]: failed to get format and dtype index.",
                    op_name.c_str(), op_type.c_str());
    return ret;
  }
  // 7. do promote type match
  DoPromoteMatch(op_judge_param, must_promote_flag, matched_index_vec);
  // 8. get the matched index
  uint32_t matched_index = GetMatchedIndex(matched_index_vec, cust_filter_matched_index_vec, op_name, op_type);

  // 9. update the input and output desc of the node
  ret = op_format_dtype_update_desc_ptr_->UpdateTensorDtypeInfo(op_kernel_info_ptr, matched_index,
                                          op_judge_param.input_index_map, true, node_ptr);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOptJdgInst][UpdDtype][SetDtypeByPrecisMode] Failed to update input of %s",
                    node_ptr->GetName().c_str());
    return ret;
  }
  ret = op_format_dtype_update_desc_ptr_->UpdateTensorDtypeInfo(op_kernel_info_ptr, matched_index,
                                          op_judge_param.output_index_map, false, node_ptr);
  (void)ge::AttrUtils::SetInt(node_ptr->GetOpDesc(), "judge_match_idx", matched_index);
  FE_LOGD("Set node [%s, %s] judge_match_idx[%u] after dtype judge %s.",
          op_name.c_str(), op_type.c_str(), matched_index, imply_type_str.c_str());
  return ret;
}

Status OpFormatDtypeJudge::SetFormatByJudgeResult(ge::NodePtr node_ptr, const std::string &imply_type_str) {
  FE_CHECK_NOTNULL(node_ptr);
  ge::OpDescPtr op_desc_ptr = node_ptr->GetOpDesc();
  FE_CHECK_NOTNULL(op_desc_ptr);
  const string &op_name = op_desc_ptr->GetName();
  const string &op_type = op_desc_ptr->GetType();

  uint32_t matched_index = 0;
  if (!ge::AttrUtils::GetInt(op_desc_ptr, "judge_match_idx", matched_index)) {
    FE_LOGW("Get node [%s, %s] judge_match_idx failed, imply %s.",
            op_name.c_str(), op_type.c_str(), imply_type_str.c_str());
    return SUCCESS;
  }

  FE_LOGD("Get node [%s, %s] judge_match_idx[%u] imply %s.",
          op_name.c_str(), op_type.c_str(), matched_index, imply_type_str.c_str());
  OpKernelInfoPtr op_kernel_info_ptr =
      OpsKernelManager::Instance(engine_name_).GetOpKernelInfoByOpType(imply_type_str, op_desc_ptr->GetType());
  FE_CHECK_NOTNULL(op_kernel_info_ptr);

  IndexNameMap input_index_map;
  IndexNameMap output_index_map;
  // 2. get input index name map
  Status ret = GetInputIndexNameMap(*(op_desc_ptr.get()), *op_kernel_info_ptr, input_index_map);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("GetInputIndexNameMap Node[%s, %s]: failed to retrieve input index name map.",
                    op_name.c_str(), op_type.c_str());
    return ret;
  }

  // 3. get output index name map
  ret = GetOutputIndexNameMap(*(op_desc_ptr.get()), *op_kernel_info_ptr, output_index_map);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("GetOutputIndexNameMap Node[%s, %s]: failed to retrieve output index name map.",
                    op_name.c_str(), op_type.c_str());
    return ret;
  }

  ret = op_format_dtype_update_desc_ptr_->UpdateTensorFormatInfo(op_kernel_info_ptr, matched_index,
                                                                input_index_map, true, node_ptr);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[GraphOptJdgInst][UpdFmt][UpdateTensorFormatInfo] Failed to update input of %s",
                    node_ptr->GetName().c_str());
    return ret;
  }
  ret = op_format_dtype_update_desc_ptr_->UpdateTensorFormatInfo(op_kernel_info_ptr, matched_index,
                                                                output_index_map, false, node_ptr);
  return ret;
}

Status OpFormatDtypeJudge::InitMatchedIndexByCustomDtype(const ge::NodePtr &node,
                                                         const OpKernelInfoPtr &op_kernel_info_ptr,
                                                         const IndexNameMap &input_index_map,
                                                         const IndexNameMap &output_index_map,
                                                         vector<uint32_t> &matched_index_vec) const {
  OpCustomizeDtype op_cust_dtypes;
  auto op_desc_ptr = node->GetOpDesc();
  if (!Configuration::Instance(engine_name_).GetCustomizeDtypeByOpName(op_desc_ptr->GetName(), op_cust_dtypes)) {
    if (!Configuration::Instance(engine_name_).GetCustomizeDtypeByOpType(op_desc_ptr->GetType(), op_cust_dtypes)) {
      return SUCCESS;
    }
  }
  vector<bool> filter_index_vec;
  FilterDtypeIndexByCustom(node, op_kernel_info_ptr, op_cust_dtypes.input_dtypes, input_index_map, true,
                           filter_index_vec);
  FilterDtypeIndexByCustom(node, op_kernel_info_ptr, op_cust_dtypes.output_dtypes, output_index_map, false,
                           filter_index_vec);
  for (size_t i = 0; i < filter_index_vec.size(); i++) {
    if (filter_index_vec[i]) {
      matched_index_vec.push_back(i);
    }
  }
  if (matched_index_vec.empty()) {
    FE_LOGE("After filtering dtypes by custom dtypes, the supported dtypes for op [%s, %s] are empty.",
            op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
    return FAILED;
  }
  return SUCCESS;
}

void OpFormatDtypeJudge::FilterDtypeIndexByCustom(const ge::NodePtr &node,
                                                  const OpKernelInfoPtr &op_kernel_info_ptr,
                                                  const vector<ge::DataType> &cust_dtypes,
                                                  const IndexNameMap &index_map,
                                                  const bool &is_input, vector<bool> &filter_index_vec) const {
  for (size_t i = 0; i < cust_dtypes.size(); i++) {
    ge::DataType cust_dtype = cust_dtypes[i];
    if (cust_dtype == ge::DT_UNDEFINED) {
      continue;
    }
    IndexNameMap::const_iterator iter_name = index_map.find(i);
    if (iter_name == index_map.end()) {
      continue;
    }
    InputOrOutputInfoPtr tesor_kernel_info_ptr = nullptr;
    Status ret = op_kernel_info_ptr->GetTensorInfoByName(is_input, iter_name->second, tesor_kernel_info_ptr);
    if (ret != SUCCESS || tesor_kernel_info_ptr == nullptr) {
      continue;
    }
    std::vector<ge::DataType> dtype_vec;
    ret = format_dtype_querier_ptr_->GetSupportDataTypes(op_kernel_info_ptr, tesor_kernel_info_ptr,
                                                         node, dtype_vec);
    if (ret != SUCCESS) {
      continue;
    }
    if (filter_index_vec.empty()) {
      filter_index_vec.assign(dtype_vec.size(), true);
    }
    for (size_t j = 0; j < dtype_vec.size(); j++) {
      if (dtype_vec[j] != cust_dtype && j < filter_index_vec.size()) {
        filter_index_vec[j] = false;
      }
    }
  }
}

Status OpFormatDtypeJudge::SortInputBySequence(ge::NodePtr node_ptr,
                                               ge::OpDescPtr op_desc_ptr,
                                               std::map<uint32_t, int> &prio_index_map) {
  uint32_t default_prio = 0;
  uint32_t prio_gap = 0xff;
  uint32_t final_prio;
  uint32_t input_size = op_desc_ptr->GetAllInputsSize();
  for (uint32_t i = 0; i < input_size; ++i) {
    auto in_anchor = node_ptr->GetInDataAnchor(i);
    if (in_anchor == nullptr || in_anchor->GetPeerOutAnchor() == nullptr ||
        in_anchor->GetPeerOutAnchor()->GetOwnerNode() == nullptr ||
        in_anchor->GetPeerOutAnchor()->GetOwnerNode()->GetOpDesc() == nullptr) {
      FE_UINT32_ADDCHECK(prio_gap, default_prio);
      final_prio = prio_gap + default_prio;
      prio_index_map.emplace(std::make_pair(final_prio, i));
      ++default_prio;
      FE_LOGW("Op[name=%s,type=%s]: the input anchor %u is invalid.", node_ptr->GetName().c_str(),
              node_ptr->GetType().c_str(), i);
    } else {
      auto peer_out_op_type = in_anchor->GetPeerOutAnchor()->GetOwnerNode()->GetOpDesc()->GetType();
      if (peer_out_op_type == CONSTANT || peer_out_op_type == CONSTANTOP || peer_out_op_type == VARIABLE) {
        FE_UINT32_ADDCHECK(prio_gap, default_prio);
        final_prio = default_prio + prio_gap;
      } else {
        final_prio = default_prio;
      }
      prio_index_map.emplace(std::make_pair(final_prio, i));
      ++default_prio;
    }
  }

  if (prio_index_map.empty()) {
    for (uint32_t i = 0; i < input_size; ++i) {
      FE_UINT32_ADDCHECK(default_prio, 1);
      prio_index_map.emplace(default_prio++, i);
    }
  }
  FE_LOGD("Op[name=%s,type=%s]: after sorting, the total input size is %zu.",
          node_ptr->GetName().c_str(), node_ptr->GetType().c_str(), prio_index_map.size());
  return SUCCESS;
}

Status OpFormatDtypeJudge::GetInputAndOutputDtypeIndex(const ge::NodePtr &node_ptr,
                                                       const OpKernelInfoPtr &op_kernel_info_ptr,
                                                       const IndexNameMap &input_map, const IndexNameMap &output_map,
                                                       const std::map<uint32_t, int> &prio_index_map,
                                                       vector<uint32_t> &matched_index_vec) {
  (void)SetSupport16In32outAttr(node_ptr, op_kernel_info_ptr, input_map, output_map);
  Status ret = GetInputDtypeIndex(node_ptr, op_kernel_info_ptr, input_map, prio_index_map, matched_index_vec);
  if (ret != SUCCESS) {
    return ret;
  } else {
    return GetOutputDtypeIndex(node_ptr, op_kernel_info_ptr, output_map, matched_index_vec);
  }
}

void OpFormatDtypeJudge::RecordC04FormatForSingleTensor(const ge::NodePtr &node_ptr,
                                                        const OpKernelInfoPtr &op_kernel_info_ptr,
                                                        const InputOrOutputInfoPtr &tensor_info,
                                                        set<uint32_t> &matched_index_set,
                                                        unordered_set<uint32_t> &c04_format_index_set) const {
  vector<ge::Format> input_or_output_format;
  if (format_dtype_querier_ptr_->GetSupportFormats(op_kernel_info_ptr, tensor_info,
                                                   node_ptr, input_or_output_format) != SUCCESS) {
    return;
  }
  RecordC04FormatIndex(input_or_output_format, matched_index_set, c04_format_index_set);
}

void OpFormatDtypeJudge::RecordC04FormatIndex(const vector<ge::Format> &input_or_output_format,
                                              const set<uint32_t> &matched_index_set,
                                              unordered_set<uint32_t> &c04_format_index_set) const {
  for (size_t j = 0; j < input_or_output_format.size(); j++) {
    if (matched_index_set.count(j) == 0) {
      continue;
    }
    if (c04_format_index_set.count(j) != 0) {
      continue;
    }
    bool is_c04_format = std::find(FE_C04_FORMAT_VECTOR.begin(), FE_C04_FORMAT_VECTOR.end(),
                                   input_or_output_format[j]) != FE_C04_FORMAT_VECTOR.end();
    if (is_c04_format) {
      c04_format_index_set.emplace(j);
    }
  }
}

void OpFormatDtypeJudge::FilterBySmallChannel(const ge::NodePtr &node_ptr, const OpKernelInfoPtr &op_kernel_info_ptr,
                                              const IndexNameMap &input_map, const IndexNameMap &output_map,
                                              vector<uint32_t> &matched_index_vec) const {
  if (matched_index_vec.size() <= 1) {
    return;
  }

  if (!op_kernel_info_ptr->IsHeavyOp()) {
    return;
  }

  if (FE_C04_SUPPORT_CUBE_OP.count(node_ptr->GetType()) == 0) {
    return;
  }

  uint32_t input_map_size = static_cast<uint32_t>(input_map.size());
  uint32_t output_map_size = static_cast<uint32_t>(output_map.size());
  unordered_set<uint32_t> c04_format_index_set;
  set<uint32_t> matched_index_set(matched_index_vec.begin(), matched_index_vec.end());
  for (uint32_t i = 0; i < node_ptr->GetAllInDataAnchorsSize(); ++i) {
    if (i >= input_map_size) {
      continue;
    }
    InputOrOutputInfoPtr tensor_info = nullptr;
    Status ret_input = op_kernel_info_ptr->GetInputInfoByName(input_map.at(i), tensor_info);
    if (ret_input != SUCCESS || tensor_info == nullptr) {
      FE_LOGW("Node[%s,%s]:input %u is not found.", node_ptr->GetType().c_str(), node_ptr->GetName().c_str(), i);
      return;
    }
    RecordC04FormatForSingleTensor(node_ptr, op_kernel_info_ptr, tensor_info, matched_index_set, c04_format_index_set);
  }
  for (uint32_t i = 0; i < node_ptr->GetAllOutDataAnchorsSize(); ++i) {
    if (i >= output_map_size) {
      continue;
    }
    InputOrOutputInfoPtr tensor_info = nullptr;
    Status ret_output = op_kernel_info_ptr->GetOutputInfoByName(output_map.at(i), tensor_info);
    if (ret_output != SUCCESS || tensor_info == nullptr) {
      FE_LOGW("Node[%s,%s]: output %u is not found.", node_ptr->GetType().c_str(), node_ptr->GetName().c_str(), i);
      return;
    }
    RecordC04FormatForSingleTensor(node_ptr, op_kernel_info_ptr, tensor_info, matched_index_set, c04_format_index_set);
  }
  FE_LOGD("Before filtering smc for node %s, indices are %s. C04 index are %s.",
          node_ptr->GetName().c_str(), StringUtils::IntegerVecToString(matched_index_vec).c_str(),
          StringUtils::IntegerVecToString(c04_format_index_set).c_str());
  if (c04_format_index_set.empty() || c04_format_index_set.size() >= matched_index_set.size()) {
    return;
  }

  bool enable_small_channel = Configuration::Instance(AI_CORE_NAME).IsEnableSmallChannel();
  bool is_first_lay = false;
  (void)ge::AttrUtils::GetBool(node_ptr->GetOpDesc(), IS_FIRST_LAYER_CONV, is_first_lay);
  matched_index_vec.clear();
  if (enable_small_channel && (is_first_lay || node_ptr->GetType() == CONV2D)) {
    for (const auto &ele : matched_index_set) {
      if (c04_format_index_set.count(ele) != 0) { // only keep C04 formats
        matched_index_vec.emplace_back(ele);
      }
    }
  } else {
    for (const auto &ele : matched_index_set) {
      if (c04_format_index_set.count(ele) == 0) { // do not keep C04 formats
        matched_index_vec.emplace_back(ele);
      }
    }
  }
  FE_LOGD("After filtering smc for node %s, indices are %s.",
          node_ptr->GetName().c_str(), StringUtils::IntegerVecToString(matched_index_vec).c_str());
}

void OpFormatDtypeJudge::RecordNDNZFormatIndex(const InputOrOutputInfoPtr &tensor_info,
                                               const std::vector<uint32_t> &matched_index_vec,
                                               const std::vector<ge::Format> &kernel_format_vec,
                                               vector<uint32_t> &filter_index) const {
  std::unordered_set<ge::Format> filter_format_type;
  if (tensor_info->GetIndex() == 0) {
    filter_format_type.emplace(ge::FORMAT_ND);
  } else if (tensor_info->GetIndex() == 1) {
    filter_format_type.emplace(ge::FORMAT_FRACTAL_NZ);
  } else {
    FE_LOGD("tensor %s for matmul op no need filter ndnz", tensor_info->GetUniqueName().c_str());
    filter_index = matched_index_vec;
    return;
  }
  RecordFormatFilterIndex(matched_index_vec, kernel_format_vec, filter_format_type, false, filter_index);
  FE_LOGD("After filtering tensor %s format index for ndnz index, it is %s.",
          tensor_info->GetUniqueName().c_str(), StringUtils::IntegerVecToString(filter_index).c_str());
  return;
}

void OpFormatDtypeJudge::RecordFormatFilterIndex(const std::vector<uint32_t> &matched_index_vec,
                                                 const std::vector<ge::Format> &kernel_format_vec,
                                                 const std::unordered_set<ge::Format> &filter_format_type,
                                                 bool is_filter_match, std::vector<uint32_t> &filter_index) const {
  for (auto index : matched_index_vec) {
    if (index >= kernel_format_vec.size()) {
      FE_LOGW("Index[%u] over format vec size[%zu].", index, kernel_format_vec.size());
      continue;
    }
    bool is_matched = (!is_filter_match && (filter_format_type.count(kernel_format_vec[index]) != 0)) ||
                      (is_filter_match && (filter_format_type.count(kernel_format_vec[index]) == 0));
    if (is_matched) {
      filter_index.emplace_back(index);
    }
  }
}

Status OpFormatDtypeJudge::FilterFormatIndexByFormatMode(const ge::NodePtr &node_ptr,
                                                         const OpKernelInfoPtr &op_kernel_info_ptr,
                                                         const IndexNameMap &input_map, const FormatModeType &cfg_type,
                                                         vector<uint32_t> &matched_index_vec) const {
  FE_LOGD("Start filter node [%s:%s] format index with mode %u and matched_index %s.",
          node_ptr->GetName().c_str(), node_ptr->GetType().c_str(), static_cast<uint32_t>(cfg_type),
          StringUtils::IntegerVecToString(matched_index_vec).c_str());
  if (cfg_type == FormatModeType::FORMAT_MODE_NDND) {
    return SUCCESS;
  }
  std::vector<uint32_t> backup_matched_index_vec = matched_index_vec;
  uint32_t input_map_size = static_cast<uint32_t>(input_map.size());
  for (uint32_t i = 0; i < node_ptr->GetAllInDataAnchorsSize(); ++i) {
    if (i >= input_map_size) {
      continue;
    }
    InputOrOutputInfoPtr tensor_info = nullptr;
    Status ret_input = op_kernel_info_ptr->GetInputInfoByName(input_map.at(i), tensor_info);
    if (ret_input != SUCCESS || tensor_info == nullptr) {
      FE_LOGW("Node[%s:%s]: the input %s is not found in the ops store.", node_ptr->GetName().c_str(),
              node_ptr->GetType().c_str(), input_map.at(i).c_str());
      return FAILED;
    }
    vector<ge::Format> input_or_output_format;
    if (format_dtype_querier_ptr_->GetSupportFormats(op_kernel_info_ptr, tensor_info, node_ptr,
                                                     input_or_output_format) != SUCCESS) {
      FE_LOGW("[GraphOptJdgInst][GetSupportFormats] Node [%s:%s] failed to retrieve supported formats, return FAILED.",
              node_ptr->GetName().c_str(), node_ptr->GetType().c_str());
      return FAILED;
    }
    std::vector<uint32_t> filter_index;
    if (cfg_type == FormatModeType::FORMAT_MODE_NDNZ) {
      RecordNDNZFormatIndex(tensor_info, backup_matched_index_vec, input_or_output_format, filter_index);
      if (filter_index.empty()) {
        FE_LOGW("Node[%s:%s] tensor %s can not match NDNZ format.", node_ptr->GetName().c_str(),
                node_ptr->GetType().c_str(), tensor_info->GetUniqueName().c_str());
        return FAILED;
      }
    } else {
      RecordFormatFilterIndex(backup_matched_index_vec, input_or_output_format, FE_ORIGIN_FORMAT_SET, true,
                              filter_index);
    }
    if (!filter_index.empty()) {
      backup_matched_index_vec = filter_index;
    }
    FE_LOGD("After filter node [%s:%s] tensor %s format index is %s.", node_ptr->GetName().c_str(),
            node_ptr->GetType().c_str(), tensor_info->GetUniqueName().c_str(),
            StringUtils::IntegerVecToString(filter_index).c_str());
  }
  matched_index_vec = backup_matched_index_vec;
  FE_LOGD("After filter node [%s:%s] format index is %s.", node_ptr->GetName().c_str(), node_ptr->GetType().c_str(),
          StringUtils::IntegerVecToString(matched_index_vec).c_str());
  return SUCCESS;
}

void OpFormatDtypeJudge::FilterByFormatMode(const ge::NodePtr &node_ptr, const OpKernelInfoPtr &op_kernel_info_ptr,
                                            const IndexNameMap &input_map, vector<uint32_t> &matched_index_vec) const {
  if (matched_index_vec.size() <= 1) {
    return;
  }

  if (!op_kernel_info_ptr->IsHeavyOp()) {
    return;
  }

  bool ffts_switch = false;
  (void)ge::AttrUtils::GetBool(node_ptr->GetOwnerComputeGraph(), ffts::kFftsSwitch, ffts_switch);
  if (ffts_switch) {
    return;
  }

  if (KFeFormatModeFilterOp.count(node_ptr->GetType()) == 0) {
    return;
  }

  FormatModeType cfg_type = Configuration::Instance(engine_name_).GetFormatModeCfg();
  Status ret = FilterFormatIndexByFormatMode(node_ptr, op_kernel_info_ptr, input_map, cfg_type, matched_index_vec);
  if (ret != SUCCESS && cfg_type == FormatModeType::FORMAT_MODE_NDNZ) {
    FE_LOGD("Node [%s:%s] retries to filter format index.", node_ptr->GetName().c_str(), node_ptr->GetType().c_str());
    (void)FilterFormatIndexByFormatMode(node_ptr, op_kernel_info_ptr, input_map, FormatModeType::FORMAT_MODE_NZNZ,
                                        matched_index_vec);
  }
  return;
}

void OpFormatDtypeJudge::FilterDiffFormatIndex(const ge::NodePtr &node_ptr,
                                               const OpKernelInfoPtr &op_kernel_info_ptr,
                                               const InputOrOutputInfoPtr &tensor_info,
                                               const ge::Format &ori_format,
                                               std::vector<uint32_t> &matched_index_vec) const {
  if (ori_format == ge::FORMAT_ND) {
    return;
  }
  vector<ge::Format> input_or_output_format;
  if (format_dtype_querier_ptr_->GetSupportFormats(op_kernel_info_ptr, tensor_info, node_ptr,
                                                   input_or_output_format) != SUCCESS) {
    FE_LOGW("[FormatJudge][FilterDiffFormatIndex] node [%s:%s] Failed to get the support formats, deal next tensor.",
            node_ptr->GetNamePtr(), node_ptr->GetTypePtr());
    return;
  }
  for (auto iter = matched_index_vec.begin(); iter != matched_index_vec.end();) {
    if (*iter < static_cast<uint32_t>(input_or_output_format.size())) {
      ge::Format op_kernel_format = input_or_output_format.at(*iter);
      if (op_kernel_format == ge::FORMAT_ND) {
        iter++;
        continue;
      }
      if ((FE_3D_FORMAT_SET.count(ori_format) != 0 && FE_3D_FORMAT_SET.count(op_kernel_format) != 0) ||
          (FE_3D_FORMAT_SET.count(ori_format) == 0 && FE_3D_FORMAT_SET.count(op_kernel_format) == 0)) {
        iter++;
      } else {
        iter = matched_index_vec.erase(iter);
      }
    } else {
      iter++;
    }
  }
  return;
}

void OpFormatDtypeJudge::FilterKernelFormatByOriginFormat(const ge::NodePtr &node_ptr,
                                                          const OpKernelInfoPtr &op_kernel_info_ptr,
                                                          const IndexNameMap &input_map,
                                                          const IndexNameMap &output_map,
                                                          vector<uint32_t> &matched_index_vec) const {
  if (matched_index_vec.size() <= 1) {
    return;
  }

  uint32_t input_map_size = static_cast<uint32_t>(input_map.size());
  uint32_t output_map_size = static_cast<uint32_t>(output_map.size());
  for (uint32_t i = 0; i < node_ptr->GetAllInDataAnchorsSize(); ++i) {
    if (i >= input_map_size) {
      continue;
    }
    InputOrOutputInfoPtr tensor_info = nullptr;
    Status ret_input = op_kernel_info_ptr->GetInputInfoByName(input_map.at(i), tensor_info);
    if (ret_input != SUCCESS || tensor_info == nullptr) {
      continue;
    }
    auto input_desc = node_ptr->GetOpDesc()->MutableInputDesc(i);
    if (input_desc == nullptr) {
      continue;
    }
    ge::Format ori_format = input_desc->GetOriginFormat();
    FilterDiffFormatIndex(node_ptr, op_kernel_info_ptr, tensor_info, ori_format, matched_index_vec);
  }
  for (uint32_t i = 0; i < node_ptr->GetAllOutDataAnchorsSize(); ++i) {
    if (i >= output_map_size) {
      continue;
    }
    InputOrOutputInfoPtr tensor_info = nullptr;
    Status ret_output = op_kernel_info_ptr->GetOutputInfoByName(output_map.at(i), tensor_info);
    if (ret_output != SUCCESS || tensor_info == nullptr) {
      continue;
    }
    auto output_desc = node_ptr->GetOpDesc()->MutableOutputDesc(i);
    if (output_desc == nullptr) {
      continue;
    }
    ge::Format ori_format = output_desc->GetOriginFormat();
    FilterDiffFormatIndex(node_ptr, op_kernel_info_ptr, tensor_info, ori_format, matched_index_vec);
  }
  return;
}

Status OpFormatDtypeJudge::GetInputAndOutputFormatIndex(const OpJudgeParam &judge_param,
                                                        vector<uint32_t> &matched_index_vec) {
  FilterBySmallChannel(judge_param.node_ptr, judge_param.op_kernel_info_ptr, judge_param.input_index_map,
                       judge_param.output_index_map, matched_index_vec);
  FilterByFormatMode(judge_param.node_ptr, judge_param.op_kernel_info_ptr,
                     judge_param.input_index_map, matched_index_vec);
  FilterKernelFormatByOriginFormat(judge_param.node_ptr, judge_param.op_kernel_info_ptr, judge_param.input_index_map,
                                   judge_param.output_index_map, matched_index_vec);
  Status ret = GetInputFormatIndex(judge_param.node_ptr, judge_param.op_kernel_info_ptr, judge_param.input_index_map,
                                   judge_param.prio_index_map, matched_index_vec);
  if (ret != SUCCESS) {
    return ret;
  } else {
    return GetOutputFormatIndex(judge_param.node_ptr, judge_param.op_kernel_info_ptr, judge_param.output_index_map,
                                matched_index_vec);
  }
}

Status OpFormatDtypeJudge::GetInputDtypeIndex(const ge::NodePtr &node_ptr, const OpKernelInfoPtr &op_kernel_info_ptr,
                                              const IndexNameMap &input_map,
                                              const std::map<uint32_t, int> &prio_index_map,
                                              vector<uint32_t> &matched_index_vec) {
  FE_CHECK_NOTNULL(node_ptr);
  FE_CHECK_NOTNULL(op_kernel_info_ptr);
  ge::OpDescPtr op_desc_ptr = node_ptr->GetOpDesc();
  FE_CHECK_NOTNULL(op_desc_ptr);
  const string &op_name = op_desc_ptr->GetName();
  const string &op_type = op_desc_ptr->GetType();
  uint32_t input_size = op_desc_ptr->GetInputsSize();
  FE_LOGD("Op[name=%s,type=%s]: the input size is %u.", op_name.c_str(), op_type.c_str(), input_size);

  // select data type by different precison mode
  Status ret = op_format_dtype_strategy_manager_ptr_->GetAllPossibleDtypeIndexByPrecisionMode(
      prio_index_map, input_map, node_ptr, op_kernel_info_ptr, true, matched_index_vec);
  FE_LOGD("Op[name=%s,type=%s]: After matching dtype, matched vec is (size: %zu), %s", op_name.c_str(), op_type.c_str(),
          matched_index_vec.size(), StringUtils::IntegerVecToString(matched_index_vec).c_str());
  return ret;
}

Status OpFormatDtypeJudge::GetInputFormatIndex(const ge::NodePtr &node_ptr, const OpKernelInfoPtr &op_kernel_info_ptr,
                                               const IndexNameMap &input_map,
                                               const std::map<uint32_t, int> &prio_index_map,
                                               vector<uint32_t> &matched_index_vec) {
  ge::OpDescPtr op_desc_ptr = node_ptr->GetOpDesc();
  const string &op_name = op_desc_ptr->GetName();
  const string &op_type = op_desc_ptr->GetType();
  uint32_t input_size = op_desc_ptr->GetInputsSize();
  FE_LOGD("Op[name=%s, type=%s]: matches the original format for input; input size is %u.", op_name.c_str(), op_type.c_str(),
          input_size);
  Status ret = op_format_dtype_strategy_manager_ptr_->GetAllPossibleFormatIndexByDefaultMode(
      prio_index_map, input_map, node_ptr, op_kernel_info_ptr, true, matched_index_vec);
  FE_LOGD("Op[name=%s,type=%s]: After matching input format, size of matched vec is %zu.",
      op_name.c_str(), op_type.c_str(), matched_index_vec.size());
  return ret;
}

Status OpFormatDtypeJudge::GetOutputFormatIndex(const ge::NodePtr &node_ptr, const OpKernelInfoPtr &op_kernel_info_ptr,
                                                const IndexNameMap &output_map, vector<uint32_t> &matched_index_vec) {
  ge::OpDescPtr op_desc_ptr = node_ptr->GetOpDesc();
  const string &op_name = op_desc_ptr->GetName();
  const string &op_type = op_desc_ptr->GetType();
  uint32_t output_size = op_desc_ptr->GetOutputsSize();
  FE_LOGD("Op[name=%s,type=%s]: match origin format for output, output size is %u.", op_name.c_str(), op_type.c_str(),
          output_size);
  /* For output the sequence of tensor index is just the same as the original
   * sequence */
  std::map<uint32_t, int> prio_index_map;
  for (size_t i = 0; i < op_desc_ptr->GetOutputsSize(); ++i) {
    prio_index_map.emplace(std::make_pair(i, static_cast<int>(i)));
  }
  Status ret = op_format_dtype_strategy_manager_ptr_->GetAllPossibleFormatIndexByDefaultMode(
      prio_index_map, output_map, node_ptr, op_kernel_info_ptr, false, matched_index_vec);
  FE_LOGD("Op[name=%s,type=%s]: After matching output format, size of matched vec is %zu.",
      op_name.c_str(), op_type.c_str(), matched_index_vec.size());
  return ret;
}

Status OpFormatDtypeJudge::GetOutputDtypeIndex(const ge::NodePtr &node_ptr, const OpKernelInfoPtr &op_kernel_info_ptr,
                                               const IndexNameMap &output_map, vector<uint32_t> &matched_index_vec) {
  ge::OpDescPtr op_desc_ptr = node_ptr->GetOpDesc();
  const string &op_name = op_desc_ptr->GetName();
  const string &op_type = op_desc_ptr->GetType();
  uint32_t output_size = op_desc_ptr->GetOutputsSize();
  FE_LOGD("Op[name=%s,type=%s]: the output size is %u.", op_name.c_str(), op_type.c_str(), output_size);
  /* For output the sequence of tensor index is just the same as the original
   * sequence */
  std::map<uint32_t, int> prio_index_map;
  for (size_t i = 0; i < op_desc_ptr->GetOutputsSize(); ++i) {
    prio_index_map.emplace(std::make_pair(i, static_cast<int>(i)));
  }
  // select data type by different precison mode
  Status ret = op_format_dtype_strategy_manager_ptr_->GetAllPossibleDtypeIndexByPrecisionMode(
      prio_index_map, output_map, node_ptr, op_kernel_info_ptr, false, matched_index_vec);
  FE_LOGD("Op[name=%s,type=%s]: After matching dtype, size of matched vec is %zu.", op_name.c_str(), op_type.c_str(),
          matched_index_vec.size());
  return ret;
}

bool OpFormatDtypeJudge::IsNoNeedJudge(const ge::OpDescPtr &op_desc_ptr,
                                       int64_t &imply_type) const {
  const string &op_type = op_desc_ptr->GetType();
  bool is_special_op = (FeGraphUtils::IsMainGraphData(op_desc_ptr) || FeGraphUtils::IsMainGraphNetOutput(op_desc_ptr) ||
                        op_type == CONSTANT || op_type == CONSTANTOP || op_type == VARIABLE || op_type == TRANSDATA ||
                        op_type == AIPPDATA || CheckVirtualOp(op_desc_ptr));
  bool has_fe_imply_type = ge::AttrUtils::GetInt(op_desc_ptr, FE_IMPLY_TYPE, imply_type);
  return is_special_op || !has_fe_imply_type;
}
}  // namespace fe

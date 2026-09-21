/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/op_compiler/op_format_tune.h"
#include "ops_store/ops_kernel_manager.h"
#include "common/graph/fe_graph_utils.h"
#include "adapter/common/op_store_adapter_manager.h"
#include "graph/compute_graph.h"
#include "graph/ge_local_context.h"
#include "graph/utils/type_utils_inner.h"
#include "register/graph_optimizer/fusion_common/unknown_shape_utils.h"
#include "common/cann_kb_adapter.h"
#include "graph_metadef/common/ge_common/util.h"

namespace {
constexpr const char* kCannKbOpFormatType = "format_type";
constexpr const char* kCannKbKnowledge = "knowledge";
}
namespace fe {
OpCompilerFormatTune::OpCompilerFormatTune(const std::string &engine_name)
    : engine_name_(engine_name) {}

OpCompilerFormatTune::~OpCompilerFormatTune() {}

Status OpCompilerFormatTune::SetTuneFormatReq(ge::ComputeGraph& graph, const FEOpsKernelInfoStorePtr &ops_kernel_info_store_ptr) {
  std::string aoe_type;
  if (FeGraphUtils::GetAoeTypeFromRootGraph(graph, aoe_type) != SUCCESS) {
    return FAILED;
  }
  if (aoe_type.find("op_format") == std::string::npos) {
    return SUCCESS;
  }
  FE_LOGD("Start to get format solutionspace.");
  auto nodes = graph.GetDirectNode();
  for (auto &node : nodes) {
    if (IsRootGraphData(node->GetType()) || node->GetType() == CONSTANT || node->GetType() == CONSTANTOP ||
        node->GetType() == NETOUTPUT || node->GetType() == OP_TYPE_PLACE_HOLDER) {
      continue;
    }
    auto op_desc = node->GetOpDesc();
    OpKernelInfoPtr op_kernel_info_ptr = OpsKernelManager::Instance(AI_CORE_NAME).GetOpKernelInfoByOpDesc(op_desc);
    if (op_kernel_info_ptr == nullptr) {
      continue;
    }
    std::vector<int64_t> input_tuneformat_index_vec;
    std::vector<int64_t> output_tuneformat_index_vec;
    if (!IsNeedTuneFormat(node, op_kernel_info_ptr, input_tuneformat_index_vec, output_tuneformat_index_vec)) {
      continue;
    }
    if (GetFormatSolutionSpace(node, op_kernel_info_ptr, ops_kernel_info_store_ptr, input_tuneformat_index_vec,
                               output_tuneformat_index_vec) != SUCCESS) {
      continue;
    }
    FE_LOGD("node[%s] get format solutionspace successfully", node->GetName().c_str());
    (void)ge::AttrUtils::SetBool(op_desc, IS_NEED_TUNEFORMAT, true);
  }
  return SUCCESS;
}

bool OpCompilerFormatTune::IsFftsPlusThreadReuseOp(const ge::NodePtr &node) const {
  std::shared_ptr<std::vector<ge::NodePtr>> related_thread_nodes = nullptr;
  related_thread_nodes = node->GetOpDesc()->TryGetExtAttr(kAttrRelatedThreadsNodes, related_thread_nodes);
  if (related_thread_nodes != nullptr && !related_thread_nodes->empty()) {
    return true;
  }

  ge::NodePtr thread1_node = nullptr;
  thread1_node = node->GetOpDesc()->TryGetExtAttr(kAttrThread1Node, thread1_node);
  if (thread1_node != nullptr) {
    return true;
  }
  return false;
}

bool OpCompilerFormatTune::HasTuneFormatSwitch(const ge::NodePtr &node, const OpKernelInfoPtr &op_kernel_info_ptr,
    std::vector<int64_t> &input_tuneformat_index_vec, std::vector<int64_t> &output_tuneformat_index_vec) const {
  const auto &input_infos = op_kernel_info_ptr->GetAllInputInfo();
  const auto &output_infos = op_kernel_info_ptr->GetAllOutputInfo();
  for (size_t input_info_index = 0; input_info_index < input_infos.size(); input_info_index++) {
    auto cur_input_tensor_desc = node->GetOpDesc()->MutableInputDesc(input_info_index);
    if (cur_input_tensor_desc == nullptr) {
      return false;
    }
    if (input_infos.at(input_info_index)->GetTuneFormatSwitch()) {
      FE_LOGD("Node[%s] input[%zu] needs tuning for format.", node->GetName().c_str(), input_info_index);
      input_tuneformat_index_vec.emplace_back(input_info_index);
    }
  }
  for (size_t output_info_index = 0; output_info_index < output_infos.size(); output_info_index++) {
    auto cur_output_tensor_desc = node->GetOpDesc()->MutableOutputDesc(output_info_index);
    if (cur_output_tensor_desc == nullptr) {
      return false;
    }
    if (output_infos.at(output_info_index)->GetTuneFormatSwitch()) {
      FE_LOGD("Node[%s] output[%zu] needs tuning for format.", node->GetName().c_str(), output_info_index);
              output_tuneformat_index_vec.emplace_back(output_info_index);
    }
  }
  if (!input_tuneformat_index_vec.empty() || !output_tuneformat_index_vec.empty()) {
    return true;
  }
  return false;
}

bool OpCompilerFormatTune::IsNeedTuneFormat(const ge::NodePtr &node, const OpKernelInfoPtr &op_kernel_info_ptr,
    std::vector<int64_t> &input_tuneformat_index_vec, std::vector<int64_t> &output_tuneformat_index_vec) const{
  ge::OpDescPtr op_desc = node->GetOpDesc();
  if (UnknownShapeUtils::IsUnknownShapeOp(*op_desc)) {
    return false;
  }

  if (IsFftsPlusThreadReuseOp(node)) {
    return false;
  }
  int64_t scope_id = 0;
  if (GetFusionScopeAttr(op_desc, scope_id) && scope_id >= 0) {
    return false;
  }

  if (HasTuneFormatSwitch(node, op_kernel_info_ptr, input_tuneformat_index_vec, output_tuneformat_index_vec)) {
    return true;
  }
  return false;
}

bool OpCompilerFormatTune::IsLegalFormat(const ge::GeTensorDescPtr &tensor, const ge::Format &origin_format,
                                         const ge::Format &cur_format) const{
  if (tensor->GetDataType() == ge::DT_INT64) {
    return false;
  }
  bool ret = false;
  if ((std::find(FE_ORIGIN_FORMAT_VECTOR.cbegin(), FE_ORIGIN_FORMAT_VECTOR.cend(), cur_format) !=
                 FE_ORIGIN_FORMAT_VECTOR.cend())) {
    ret = IsLegalOriginFormat(origin_format, cur_format);
  } else {
    ret = IsLegalHeavyFormat(tensor, origin_format, cur_format);
  }
  return ret;
}

bool OpCompilerFormatTune::IsLegalOriginFormat(const ge::Format &origin_format,
                                               const ge::Format &cur_format) const{
  if (cur_format == origin_format || cur_format == ge::FORMAT_ND) {
    return true;
  }
  return false;
}

bool OpCompilerFormatTune::IsLegalHeavyFormat(const ge::GeTensorDescPtr &tensor,
                                              const ge::Format &origin_format,
                                              const ge::Format &cur_format) const{
  /* filter format which transnode unsupport */
  if (origin_format == ge::FORMAT_HWCN && cur_format == ge::FORMAT_NC1HWC0) {
    return false;
  }
  if (origin_format == ge::FORMAT_ND) {
    if (cur_format == ge::FORMAT_NC1HWC0 || cur_format == ge::FORMAT_NDC1HWC0) {
      return false;
    }
  }
  if (origin_format == ge::FORMAT_NHWC && cur_format == ge::FORMAT_FRACTAL_Z) {
    return false;
  }
  auto dim_num = tensor->GetOriginShape().GetDimNum();
  if (origin_format == ge::FORMAT_ND) {
    if (cur_format == ge::FORMAT_FRACTAL_Z && dim_num > MINIMUM_NZ_SHAPE_DIM_NUM) {
      return false;
    }
  }
  if (cur_format == ge::FORMAT_NDC1HWC0 &&
      ((origin_format != ge::FORMAT_NDHWC && origin_format != ge::FORMAT_NCDHW) || dim_num != DIMENSION_NUM_FIVE)) {
    return false;
  }
  if (cur_format == ge::FORMAT_FRACTAL_Z_3D &&
      ((origin_format != ge::FORMAT_NCDHW && origin_format != ge::FORMAT_DHWCN) || dim_num != DIMENSION_NUM_FIVE)) {
    return false;
  }
  return true;
}

void OpCompilerFormatTune::FilterFormatIndex(const std::vector<InputOrOutputInfoPtr> &input_or_output_infos,
                                             const std::vector<int64_t> &tuneformat_index_vec,
                                             const ge::OpDesc::Vistor<ge::GeTensorDescPtr> &tensor_descs,
                                             std::map<std::string, std::vector<ge::Format>> &format_map) {
  for (auto &tuneformat_index : tuneformat_index_vec) {
    auto cur_tensor = tensor_descs.at(tuneformat_index);
    auto cur_origin_format = cur_tensor->GetOriginFormat();
    auto matched_format_index_iter = matched_format_index_vec.begin();

    for (; matched_format_index_iter != matched_format_index_vec.end();) {
      auto cur_tensor_name = input_or_output_infos.at(tuneformat_index)->GetUniqueName();
      auto cur_format = format_map[cur_tensor_name].at(*matched_format_index_iter);
      if (!IsLegalFormat(cur_tensor, cur_origin_format, cur_format)) {
        auto matched_format_index_iter_tmp = matched_format_index_iter;
        matched_format_index_iter = matched_format_index_vec.erase(matched_format_index_iter_tmp);
      } else {
        matched_format_index_iter++;
      }
    }
  }
}

Status OpCompilerFormatTune::SetTuneFormatReqAttr(const std::vector<InputOrOutputInfoPtr> &input_or_output_infos,
                                                  const ge::OpDesc::Vistor<ge::GeTensorDescPtr> &tensor_descs,
                                                  std::map<string, std::vector<ge::Format>> &format_map) const {
  std::vector<int64_t> tuneformat_req_vec;
  for (size_t tensor_desc_index = 0; tensor_desc_index < tensor_descs.size(); tensor_desc_index++) {
    auto cur_tensor_desc = tensor_descs.at(tensor_desc_index);
    auto cur_tensor_name = input_or_output_infos.at(tensor_desc_index)->GetUniqueName();
    for (size_t matched_format_index : matched_format_index_vec) {
      tuneformat_req_vec.emplace_back(static_cast<int64_t>(
                                      format_map[cur_tensor_name].at(matched_format_index)));
    }
    if (!ge::AttrUtils::SetListInt(cur_tensor_desc, AOE_TUNEFORMAT_REQ, tuneformat_req_vec)) {
      auto cur_tensor_desc_index = static_cast<int64_t>(tensor_desc_index);
      for (; cur_tensor_desc_index >= 0; cur_tensor_desc_index--) {
        tensor_descs.at(cur_tensor_desc_index)->DelAttr(AOE_TUNEFORMAT_REQ);
      }
      return FAILED;
    }
    tuneformat_req_vec.clear();
  }
  return SUCCESS;
}

void OpCompilerFormatTune::GetMatchedIndexVec(std::map<std::string, std::vector<ge::DataType>> &datatype_map,
                                              const OpKernelInfoPtr &op_kernel_info_ptr,
                                              const ge::OpDescPtr &op_desc_ptr) {
  const auto &input_infos = op_kernel_info_ptr->GetAllInputInfo();
  const auto &output_infos = op_kernel_info_ptr->GetAllOutputInfo();
  auto datatype_map_iter = datatype_map.begin();
  for (size_t datatype_index = 0; datatype_index < datatype_map_iter->second.size(); datatype_index++) {
    bool dtype_matched_flag = true;
    for (size_t tensor_desc_index = 0; tensor_desc_index < input_infos.size(); tensor_desc_index++) {
      auto cur_tensor_name = input_infos.at(tensor_desc_index)->GetUniqueName();
      auto cur_tensor_desc = op_desc_ptr->MutableInputDesc(tensor_desc_index);
      if (cur_tensor_desc == nullptr) {
        continue;
      }
      if (cur_tensor_desc->GetDataType() != datatype_map[cur_tensor_name].at(datatype_index)) {
        dtype_matched_flag = false;
        break;
      }
    }
    if (!dtype_matched_flag) {
      continue;
    }
    for (size_t tensor_desc_index = 0; tensor_desc_index < output_infos.size(); tensor_desc_index++) {
      auto cur_tensor_name = output_infos.at(tensor_desc_index)->GetUniqueName();
      auto cur_tensor_desc = op_desc_ptr->MutableOutputDesc(tensor_desc_index);
      if (cur_tensor_desc == nullptr) {
        continue;
      }
      if (cur_tensor_desc->GetDataType() != datatype_map[cur_tensor_name].at(datatype_index)) {
        dtype_matched_flag = false;
        break;
      }
    }
    if (!dtype_matched_flag) {
      continue;
    }
    matched_format_index_vec.push_back(datatype_index);
  }
}

Status OpCompilerFormatTune::GetFormatSolutionSpace(const ge::NodePtr &node, const OpKernelInfoPtr &op_kernel_info_ptr,
                                                    const FEOpsKernelInfoStorePtr &ops_kernel_info_store_ptr,
                                                    std::vector<int64_t> &input_tuneformat_index_vec,
                                                    std::vector<int64_t> &output_tuneformat_index_vec) {
  auto op_desc_ptr = node->GetOpDesc();
  int64_t op_impl_type = static_cast<int64_t>(EN_RESERVED);
  (void)ge::AttrUtils::GetInt(op_desc_ptr, FE_IMPLY_TYPE, op_impl_type);
  OpImplType impl_type = static_cast<OpImplType>(op_impl_type);
  const auto sub_ops_store_ptr = ops_kernel_info_store_ptr->GetSubOpsStore(impl_type);
  FE_CHECK(sub_ops_store_ptr == nullptr,
           FE_LOGW("sub op store [%s] not found.", GetImplTypeString(impl_type).c_str()),
           return FAILED);
  bool is_dynamic_impl = false;
  (void)ge::AttrUtils::GetBool(op_desc_ptr, ATTR_NAME_IS_OP_DYNAMIC_IMPL, is_dynamic_impl);
  FormatDtypeInfo format_dtype_info;
  Status ret = sub_ops_store_ptr->GetSupportFormatAndDtype(node, op_kernel_info_ptr, is_dynamic_impl,
                                                           format_dtype_info);
  if (ret != SUCCESS) {
    FE_LOGW("Node [%s %s] failed to obtain the supported format map",
            op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str());
    return FAILED;
  }
  GetMatchedIndexVec(format_dtype_info.data_type_map, op_kernel_info_ptr, op_desc_ptr);

  const auto &input_descs = op_desc_ptr->GetAllInputsDescPtr();
  const auto &output_descs = op_desc_ptr->GetAllOutputsDescPtr();
  const auto &input_infos = op_kernel_info_ptr->GetAllInputInfo();
  const auto &output_infos = op_kernel_info_ptr->GetAllOutputInfo();
  FilterFormatIndex(input_infos, input_tuneformat_index_vec, input_descs, format_dtype_info.format_map);
  FilterFormatIndex(output_infos, output_tuneformat_index_vec, output_descs, format_dtype_info.format_map);
  if (SetTuneFormatReqAttr(input_infos, input_descs, format_dtype_info.format_map) != SUCCESS ||
      SetTuneFormatReqAttr(output_infos, output_descs, format_dtype_info.format_map) != SUCCESS) {
    FE_LOGW("Failed to set tuneformat_req attr for node[%s]", node->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status OpCompilerFormatTune::GetFormatTuneKnowledge(ge::NodePtr &node, nlohmann::json &params_json) {
  ge::OpDescPtr op_desc = node->GetOpDesc();
  FE_CHECK_NOTNULL(op_desc);
  OpKernelInfoPtr op_kernel_info_ptr =
      OpsKernelManager::Instance(engine_name_).GetOpKernelInfoByOpDesc(op_desc);
  FE_CHECK_NOTNULL(op_kernel_info_ptr);
  OpStoreAdapterPtr op_store_adapter =
      OpStoreAdapterManager::Instance(engine_name_).GetOpStoreAdapter(op_kernel_info_ptr->GetOpStoreImplType());
  FE_CHECK_NOTNULL(op_store_adapter);

  std::vector<std::string> op_unique_keys;
  if (op_store_adapter->GetOpUniqueKeys(node, op_kernel_info_ptr, op_unique_keys) != SUCCESS) {
    FE_LOGD("Op[name=%s, type=%s]: failed to get op_unique_keys.",
            node->GetName().c_str(), node->GetType().c_str());
    return FAILED;
  }
  FE_LOGD("Op[name=%s, type=%s]: The size of op_unique_keys is [%zu].",
          node->GetName().c_str(), node->GetType().c_str(), op_unique_keys.size());

  std::map<std::string, std::string> config_map;
  config_map.emplace(EM_OP_TYPE, kCannKbOpFormatType);
  std::string kb_result_str;
  if (!GetTuneKnowledgeResult(node, op_unique_keys, config_map, kb_result_str)) {
    FE_LOGD("Op[name=%s, type=%s]: Failed to get tune knowledge.",
            node->GetName().c_str(), node->GetType().c_str());
    return FAILED;
  }
  if (kb_result_str.empty()) {
    FE_LOGW("Op[name=%s, type=%s]: The cann kb result is empty.",
            node->GetName().c_str(), node->GetType().c_str());
    return FAILED;
  }
  FE_LOGD("Op[name=%s, type=%s]: The cann kb result is %s.",node->GetName().c_str(),
          node->GetType().c_str(), kb_result_str.c_str());
  try {
    params_json = nlohmann::json::parse(kb_result_str);
  } catch (nlohmann::json::parse_error& ex) {
    FE_LOGW("[%s] is not in JSON format.", kb_result_str.c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status OpCompilerFormatTune::UpdataFormatAndShapeByFormatTune(FormatTuneInfo &fomat_tune_info) {
  FE_LOGD("Start UpdataFormatAndShapeByFormatTune.");
  ge::OpDescPtr op_desc = fomat_tune_info.node->GetOpDesc();
  FE_CHECK_NOTNULL(op_desc);
  OpKernelInfoPtr op_kernel_info_ptr =
      OpsKernelManager::Instance(engine_name_).GetOpKernelInfoByOpDesc(op_desc);
  FE_CHECK_NOTNULL(op_kernel_info_ptr);
  IndexNameMap index_map;
  if(fomat_tune_info.is_input) {
    (void)GetInputIndexNameMap(*(op_desc.get()), *op_kernel_info_ptr, index_map);
  } else {
    (void)GetOutputIndexNameMap(*(op_desc.get()), *op_kernel_info_ptr, index_map);
  }
  auto tensor_iter = index_map.find(fomat_tune_info.anchor_index);
  if (tensor_iter == index_map.end() || tensor_iter->second.empty()) {
    FE_LOGE("Node[name=%s, type=%s, %s %zu]: Failed to find tensor.",
            fomat_tune_info.node->GetName().c_str(), fomat_tune_info.node->GetType().c_str(),
            IS_INPUT_TO_STRING(fomat_tune_info.is_input), fomat_tune_info.anchor_index);
    return FAILED;
  }
  InputOrOutputInfoPtr tensor_info_ptr = nullptr;
  if (op_kernel_info_ptr->GetTensorInfoByName(fomat_tune_info.is_input,
                                              tensor_iter->second, tensor_info_ptr) != SUCCESS) {
    FE_LOGE("Node[name=%s, type=%s, %s %zu]: Failed to obtain tensor information.",
            fomat_tune_info.node->GetName().c_str(), fomat_tune_info.node->GetType().c_str(),
            IS_INPUT_TO_STRING(fomat_tune_info.is_input), fomat_tune_info.anchor_index);
    return FAILED;
  }
  FE_CHECK_NOTNULL(tensor_info_ptr);
  uint32_t matched_index = 0;
  ge::GeTensorDescPtr tensor_desc = fomat_tune_info.is_input ?
                                    op_desc->MutableInputDesc(fomat_tune_info.anchor_index) :
                                    op_desc->MutableOutputDesc(fomat_tune_info.anchor_index);
  FE_CHECK_NOTNULL(tensor_desc);
  tensor_desc->SetFormat(tensor_desc->GetOriginFormat());
  tensor_desc->SetShape(tensor_desc->GetOriginShape());
  UpdateInfo update_info = {op_kernel_info_ptr, tensor_info_ptr, matched_index, fomat_tune_info.node,
      static_cast<uint32_t>(fomat_tune_info.anchor_index), *tensor_desc, fomat_tune_info.is_input};
  if (CalcNewShapeAndUpdate(update_info, fomat_tune_info.new_format, tensor_desc->GetDataType()) != SUCCESS) {
    FE_LOGE("Node[name=%s, type=%s, %s %zu]: Failed to update format and shape.",
            fomat_tune_info.node->GetName().c_str(), fomat_tune_info.node->GetType().c_str(),
            IS_INPUT_TO_STRING(fomat_tune_info.is_input), fomat_tune_info.anchor_index);
    return FAILED;
  }
  return SUCCESS;
}

Status OpCompilerFormatTune::UpdataGraphByFormatTune(ge::ComputeGraph &graph, ge::NodePtr &node,
                                                     const bool &update_graph_forward_flag,
                                                     const bool &update_graph_backward_flag) {
  FE_LOGD("Start UpdataGraphByFormatTune.");
  FEOpsKernelInfoStorePtr ops_kernel_info_store_ptr = nullptr;
  FE_MAKE_SHARED(ops_kernel_info_store_ptr =
      std::make_shared<FEOpsKernelInfoStore>(engine_name_),
      return fe::OP_COMPILER_MAKE_SHARED_FAILED);
  FE_CHECK_NOTNULL(ops_kernel_info_store_ptr);
  TransNodeManagerPtr trans_node_mgr_ptr = nullptr;
  FE_MAKE_SHARED(trans_node_mgr_ptr = std::make_shared<TransNodeManager>(ops_kernel_info_store_ptr),
                 return fe::OP_COMPILER_MAKE_SHARED_FAILED);
  FE_CHECK_NOTNULL(trans_node_mgr_ptr);
  if (trans_node_mgr_ptr->Initialize() != SUCCESS) {
    FE_LOGE("Failed to initialize transNodeMgrPtr for graph %s.", graph.GetName().c_str());
    return FAILED;
  }

  (void)ge::AttrUtils::SetBool(graph, NEED_RE_PRECOMPILE, true);
  map<std::string, std::string> options = ge::GetThreadLocalContext().GetAllOptions();
  if (ops_kernel_info_store_ptr->Initialize(options) != SUCCESS) {
    REPORT_FE_ERROR("[SubGraphOpt][FormatTune][Init] OpInfoKernelStores init failed!");
    return FAILED;
  }

  if (update_graph_forward_flag) {
    FE_LOGD("Start UpdateGraphByFormatTune forward.");
    // forward
    if (trans_node_mgr_ptr->InsertTransNodes(graph, node) != SUCCESS) {
      FE_LOGW("[SubGraphOpt][FormatTune][ShapeTrans][InsTrans] InsertTransNodes failed!");
      return FAILED;
    }
    if (graph.TopologicalSorting() != ge::GRAPH_SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][FormatTune][ShapeTrans][TopoSort] TopologicalSorting failed!");
      return FAILED;
    }
    if (trans_node_mgr_ptr->MergeAllTransOps(graph) != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][FormatTune][ShapeTrans][MgTrans] MergeAllTransOps failed!");
      return FAILED;
    }
  }

  if (update_graph_backward_flag) {
    FE_LOGD("Start UpdataGraphByFormatTune backward.");
    // backward
    for (auto &node_out : node->GetOutDataNodes()) {
      FE_CHECK_NOTNULL(node_out);
      if (trans_node_mgr_ptr->InsertTransNodes(graph, node_out) != SUCCESS) {
        FE_LOGW("[SubGraphOpt][FormatTune][ShapeTrans][InsTrans] InsertTransNodes failed!");
        return FAILED;
      }
      if (graph.TopologicalSorting() != ge::GRAPH_SUCCESS) {
        REPORT_FE_ERROR("[SubGraphOpt][FormatTune][ShapeTrans][TopoSort] TopologicalSorting failed!");
        return FAILED;
      }
      if (trans_node_mgr_ptr->MergeAllTransOps(graph) == FAILED) {
        REPORT_FE_ERROR("[SubGraphOpt][FormatTune][ShapeTrans][MgTrans] MergeAllTransOps failed!");
        return FAILED;
      }
    }
  }
  (void)ge::AttrUtils::SetBool(node->GetOpDesc(), NEED_RE_PRECOMPILE, true);
  return SUCCESS;
}

Status OpCompilerFormatTune::UpdateTensorByCannKbResult(ge::NodePtr &node, const bool &is_input,
                                                        nlohmann::json &json, bool &update_graph_flag) {
  ge::OpDescPtr op_desc = node->GetOpDesc();
  FE_CHECK_NOTNULL(op_desc);
  size_t tensor_size = is_input ? op_desc->GetInputsSize() : op_desc->GetOutputsSize();
  FE_LOGI("Op[name=%s, type=%s]: The tensor size is %zu.", node->GetName().c_str(), node->GetType().c_str(), tensor_size);
  for (size_t tensor_index = 0U; tensor_index < tensor_size; ++tensor_index) {
    std::string tuneformat_key = IS_INPUT_TO_STRING(is_input) + std::to_string(tensor_index);
    auto iter = json.find(tuneformat_key);
    if (iter == json.end()) {
      FE_LOGW("Op[name=%s, type=%s, %s %zu]: Failed to get tuneformat value.",
              node->GetName().c_str(), node->GetType().c_str(), IS_INPUT_TO_STRING(is_input), tensor_index);
      return FAILED;
    }
    std::string tuneformat_string = iter.value();
    ge::Format tuneformat = ge::TypeUtils::SerialStringToFormat(StringUtils::Trim(tuneformat_string));
    if (!ge::TypeUtilsInner::IsFormatValid(tuneformat)) {
      FE_LOGW("Op[name=%s, type=%s, %s %zu]: Tune format is invalid.", node->GetName().c_str(),
              node->GetType().c_str(), IS_INPUT_TO_STRING(is_input), tensor_index);
      return FAILED;
    }
    ge::GeTensorDescPtr tensor_desc = is_input ?
        op_desc->MutableInputDesc(tensor_index) : op_desc->MutableOutputDesc(tensor_index);
    FE_CHECK_NOTNULL(tensor_desc);
    ge::Format format = static_cast<ge::Format>(ge::GetPrimaryFormat(tensor_desc->GetFormat()));
    if (tuneformat == format) {
      FE_LOGD("Op[name=%s, type=%s, %s %zu]: TuneFormat matches the format.",
              node->GetName().c_str(), node->GetType().c_str(), IS_INPUT_TO_STRING(is_input), tensor_index);
      continue;
    }
    FE_LOGI("Op[name=%s, type=%s, %s %zu]: TuneFormat %s does not match format %s, need to update graph.",
            node->GetName().c_str(), node->GetType().c_str(), IS_INPUT_TO_STRING(is_input), tensor_index,
            ge::TypeUtils::FormatToSerialString(tuneformat).c_str(),
            ge::TypeUtils::FormatToSerialString(format).c_str());
    update_graph_flag = true;
    FormatTuneInfo format_tune_info = {node, tensor_index, tuneformat, is_input};
    if (UpdataFormatAndShapeByFormatTune(format_tune_info) != SUCCESS) {
      FE_LOGE("Op[name=%s, type=%s, %s %zu]: Failed to update format and shape with CANN KB result.",
              node->GetName().c_str(), node->GetType().c_str(), IS_INPUT_TO_STRING(is_input), tensor_index);
      return FAILED;
    }
  }
  return SUCCESS;
}

Status OpCompilerFormatTune::UpdateTensorByNodeAttr(ge::NodePtr &node, const bool &is_input,
                                                    bool &update_graph_flag) {
  ge::OpDescPtr op_desc = node->GetOpDesc();
  FE_CHECK_NOTNULL(op_desc);
  size_t tensor_size = is_input ? op_desc->GetInputsSize() : op_desc->GetOutputsSize();
  FE_LOGI("Op[name=%s, type=%s]: The tensor size is %zu.", node->GetName().c_str(),
          node->GetType().c_str(), tensor_size);
  for (size_t tensor_index = 0U; tensor_index < tensor_size; ++tensor_index) {
    ge::GeTensorDescPtr tensor_desc = is_input ?
        op_desc->MutableInputDesc(tensor_index) : op_desc->MutableOutputDesc(tensor_index);
    FE_CHECK_NOTNULL(tensor_desc);
    uint32_t tuneformat_int = 0;
    if (!ge::AttrUtils::GetInt(tensor_desc, AOE_TUNEFORMAT, tuneformat_int)) {
      return SUCCESS;
    }
    ge::Format tuneformat = static_cast<ge::Format>(tuneformat_int);
    if (!ge::TypeUtilsInner::IsFormatValid(tuneformat)) {
      FE_LOGE("Op[name=%s, type=%s, %s %zu]: Tune format is invalid.", node->GetName().c_str(),
              node->GetType().c_str(), IS_INPUT_TO_STRING(is_input), tensor_index);
      return FAILED;
    }
    ge::Format format = static_cast<ge::Format>(ge::GetPrimaryFormat(tensor_desc->GetFormat()));
    if (tuneformat == format) {
      FE_LOGI("Op[name=%s, type=%s, %s %zu]: TuneFormat matches the format.",
              node->GetName().c_str(), node->GetType().c_str(), IS_INPUT_TO_STRING(is_input), tensor_index);
      continue;
    }
    FE_LOGI("Op[name=%s, type=%s, %s %zu]: TuneFormat %s does not match format %s, need to update graph.",
            node->GetName().c_str(), node->GetType().c_str(), IS_INPUT_TO_STRING(is_input), tensor_index,
            ge::TypeUtils::FormatToSerialString(tuneformat).c_str(),
            ge::TypeUtils::FormatToSerialString(format).c_str());
    update_graph_flag = true;
    FormatTuneInfo fomat_tune_info = {node, tensor_index, tuneformat, is_input};
    if (UpdataFormatAndShapeByFormatTune(fomat_tune_info) != SUCCESS) {
      FE_LOGE("Op[name=%s, type=%s, %s %zu]: Failed to update format and shape by node attribute.",
              node->GetName().c_str(), node->GetType().c_str(), IS_INPUT_TO_STRING(is_input), tensor_index);
      return FAILED;
    }
  }
  return SUCCESS;
}

Status OpCompilerFormatTune::UpdateTuneFormatByCannKbResult(ge::ComputeGraph &graph, bool &need_re_precompile) {
  FE_LOGD("Start UpdateTuneFormatByCannKbResult.");
  bool update_graph_forward_flag = false;
  bool update_graph_backward_flag = false;
  for (ge::NodePtr &node : graph.GetAllNodes()) {
    FE_CHECK_NOTNULL(node);
    if (PLACE_OR_END_SET.count(node->GetType()) > 0) {
      FE_LOGD("Op[name=%s,type=%s]: Node is invalid.", node->GetName().c_str(), node->GetType().c_str());
      continue;
    }

    std::vector<int64_t> input_tuneformat_index_vec;
    std::vector<int64_t> output_tuneformat_index_vec;
    OpKernelInfoPtr op_kernel_info_ptr =
        OpsKernelManager::Instance(engine_name_).GetOpKernelInfoByOpDesc(node->GetOpDesc());
    if (op_kernel_info_ptr == nullptr) {
      return false;
    }
    if (!IsNeedTuneFormat(node, op_kernel_info_ptr, input_tuneformat_index_vec, output_tuneformat_index_vec)) {
      continue;
    }
    FE_LOGD("Op[name=%s, type=%s]: Need to tune format.", node->GetName().c_str(), node->GetType().c_str());
    nlohmann::json params_json;
    if (GetFormatTuneKnowledge(node, params_json) != SUCCESS) {
      continue;
    }
    update_graph_forward_flag = false;
    update_graph_backward_flag = false;
    if (UpdateTensorByCannKbResult(node, true, params_json, update_graph_forward_flag) != SUCCESS) {
      return FAILED;
    }
    if (UpdateTensorByCannKbResult(node, false, params_json, update_graph_backward_flag) != SUCCESS) {
      return FAILED;
    }
    FE_LOGD("Op[name=%s,type=%s]: update_graph_forward_flag is %d, update_graph_backward_flag is %d.",
            node->GetName().c_str(), node->GetType().c_str(),
            update_graph_forward_flag, update_graph_backward_flag);
    if (!update_graph_forward_flag && !update_graph_backward_flag) {
      continue;
    }
    if (UpdataGraphByFormatTune(graph, node, update_graph_forward_flag,
                                update_graph_backward_flag) != SUCCESS) {
      FE_LOGW("Op[name=%s,type=%s]: Failed to Update graph by cann kb result.", node->GetName().c_str(),
              node->GetType().c_str());
      return SUCCESS;
    }
    need_re_precompile = true;
  }
  return SUCCESS;
}

Status OpCompilerFormatTune::UpdateTuneFormatByNodeAttrInner(ge::ComputeGraph &graph) {
  bool update_graph_forward_flag = false;
  bool update_graph_backward_flag = false;
  for (ge::NodePtr &node : graph.GetAllNodes()) {
    FE_CHECK_NOTNULL(node);
    if (PLACE_OR_END_SET.count(node->GetType()) > 0) {
      FE_LOGD("Op[name=%s,type=%s]: Node is invalid.", node->GetName().c_str(), node->GetType().c_str());
      continue;
    }
    update_graph_forward_flag = false;
    update_graph_backward_flag = false;
    if (UpdateTensorByNodeAttr(node, true, update_graph_forward_flag) != SUCCESS) {
      return FAILED;
    }
    if (UpdateTensorByNodeAttr(node, false, update_graph_backward_flag) != SUCCESS) {
      return FAILED;
    }
    FE_LOGD("Op[name=%s,type=%s]: update_graph_forward_flag is %d, update_graph_backward_flag is %d.",
            node->GetName().c_str(), node->GetType().c_str(),
            update_graph_forward_flag, update_graph_backward_flag);
    if (!update_graph_forward_flag && !update_graph_backward_flag) {
      continue;
    }
    if (UpdataGraphByFormatTune(graph, node, update_graph_forward_flag,
                                update_graph_backward_flag) != SUCCESS) {
      FE_LOGW("Op[name=%s, type=%s]: Failed to update graph with node attribute.", node->GetName().c_str(),
              node->GetType().c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

Status OpCompilerFormatTune::ReplacePldAndEnd(std::map<std::string, ge::NodePtr> &ori_node_map,
                                              ge::ComputeGraphPtr &graph_ptr) {
  for (ge::NodePtr &node : graph_ptr->GetAllNodes()) {
    FE_CHECK_NOTNULL(node);
    const auto iter = ori_node_map.find(node->GetName());
    if (iter == ori_node_map.end() || iter->second == nullptr) {
      FE_LOGE("Op[name=%s,type=%s]: Failed to find node from ori_node_map.", node->GetName().c_str(),
              node->GetType().c_str());
      return FAILED;
    }
    // copy member except pld_and_end in graph bac
    if (PLACE_OR_END_SET.count(node->GetType()) <= 0) {
      node->GetOpDesc()->SetOpEngineName(iter->second->GetOpDesc()->GetOpEngineName());
      node->SetOwnerComputeGraph(iter->second->GetOwnerComputeGraph());
      continue;
    }
    // if curr node is original pld_and_end, need skip
    if (node == iter->second) {
      continue;
    }
    // unlink ori pld_and_end in graph bac
    ge::NodeUtils::UnlinkAll(*(iter->second));
    std::vector<int32_t> input_map(node->GetAllInDataAnchorsSize());
    for (uint32_t i = 0; i < node->GetAllInDataAnchorsSize(); ++i) {
      input_map[i] = static_cast<int32_t>(i);
    }
    std::vector<int32_t> output_map(node->GetAllOutDataAnchorsSize());
    for (uint32_t i = 0; i < node->GetAllOutDataAnchorsSize(); ++i) {
      output_map[i] = static_cast<int32_t>(i);
    }
    // replace pld_and_end in in graph bac with original pld_and_end
    auto ret = ge::GraphUtils::ReplaceNodeAnchors(iter->second, node, input_map, output_map);
    if (ret != ge::GRAPH_SUCCESS) {
      FE_LOGE("Op[name=%s,type=%s]: Failed to replace node.", node->GetName().c_str(),
              node->GetType().c_str());
      return FAILED;
    }
    ge::NodeUtils::UnlinkAll(*node);
    ret = ge::GraphUtils::RemoveNodeWithoutRelink(graph_ptr, node);
    if (ret != ge::GRAPH_SUCCESS) {
      FE_LOGE("Op[name=%s,type=%s]: Failed to remove node.", node->GetName().c_str(),
              node->GetType().c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

Status OpCompilerFormatTune::UpdateTuneFormatByNodeAttr(ge::ComputeGraph &graph) {
  FE_LOGD("Start UpdateTuneFormatByNodeAttr.");
  // copy original graph
  auto nodes = graph.GetDirectNode();
  if (nodes.empty()) {
    FE_LOGW("[SubGraphOpt][FormatTune][GetNodes] Failed to retrieve nodes from graph [%s].", graph.GetName().c_str());
    return FAILED;
  }
  GE_CHECK_NOTNULL(nodes.at(0));
  const auto graph_ptr = nodes.at(0)->GetOwnerComputeGraph();
  GE_CHECK_NOTNULL(graph_ptr);
  ge::ComputeGraphPtr graph_bac_ptr = nullptr;
  FE_MAKE_SHARED(graph_bac_ptr = std::make_shared<ge::ComputeGraph>(graph.GetName()), return FAILED);
  GE_CHECK_NOTNULL(graph_bac_ptr);
  if (ge::GraphUtils::CopyComputeGraph(graph_ptr, graph_bac_ptr) != ge::GRAPH_SUCCESS) {
    FE_LOGW("[SubGraphOpt][FormatTune][CopyGraph] Failed to copy graph [%s] by ge.", graph.GetName().c_str());
    return FAILED;
  }
  // generate original node map
  // add original pld_and_end to graph bac
  std::map<std::string, ge::NodePtr> ori_node_map;
  for (ge::NodePtr &node : graph_ptr->GetAllNodes()) {
    FE_CHECK_NOTNULL(node);
    ori_node_map[node->GetName()] = node;
    if (PLACE_OR_END_SET.count(node->GetType()) > 0) {
      (void)graph_bac_ptr->AddNode(node);
    }
  }
  // change original graph by format tune
  if (UpdateTuneFormatByNodeAttrInner(graph) == SUCCESS) {
    return SUCCESS;
  }
  // if inserting transnodes failed, return graph bac with original pld_and_end
  if (ReplacePldAndEnd(ori_node_map, graph_bac_ptr) != SUCCESS) {
    FE_LOGE("[SubGraphOpt][FormatTune][Replace] Failed to replace pld_and_end in graph [%s].",
            graph.GetName().c_str());
    return FAILED;
  }
  graph = *(graph_bac_ptr.get());
  FE_LOGD("[SubGraphOpt][FormatTune][ReturnGraph] Returning to the original graph [%s].",
          graph.GetName().c_str());
  return SUCCESS;
}

bool OpCompilerFormatTune::GetTuneKnowledgeResult(const ge::NodePtr &node,
                                                  const std::vector<std::string> &op_unique_keys,
                                                  const std::map<std::string, std::string> &search_config,
                                                  std::string &kb_result_str) {
  bool kb_hit = false;
  if (!CannKBUtils::Instance().InitCannKb()){
    FE_LOGE("OP[%s][%s] failed to init cannkb.", node->GetNamePtr(), node->GetTypePtr());
    return false;
  }
  for (const auto &op_unique_key : op_unique_keys) {
    std::vector<std::map<std::string, std::string>> result;
    CannKb::CANN_KB_STATUS cann_kb_status = CannKBUtils::Instance().RunCannKbSearch(op_unique_key, search_config, result);
    if (cann_kb_status != CannKb::CANN_KB_STATUS::CANN_KB_SUCC) {
      // kb_hit log can not be deleted, other components are already in use.
      FE_LOGD("[op_kb_hit][%s][%d][%s]", node->GetName().c_str(), kb_hit, node->GetType().c_str());
      return false;
    }
    if (result.empty()) {
      continue;
    }
    auto iter = result.at(0).find(kCannKbKnowledge);
    if (iter == result.at(0).end()) {
      FE_LOGD("The cann kb result does not contain[%s].", kCannKbKnowledge);
      break;
    }
    kb_result_str = iter->second;
  }
  if (!kb_result_str.empty()) {
    kb_hit = true;
  }
  // kb_hit log can not be deleted, other components are already in use.
  FE_LOGD("[op_kb_hit][%s][%d][%s]", node->GetName().c_str(), kb_hit, node->GetType().c_str());
  return true;
}
}

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/op_setter/op_setter.h"
#include <sstream>
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_local_context.h"
#include "common/aicore_util_attr_define.h"
#include "common/ge_common/ge_types.h"
#include "common/string_utils.h"
#include "common/util/op_info_util.h"
#include "common/fe_utils.h"
#include "common/fe_thread_pool.h"
#include "common/configuration.h"
#include "common/op_slice_util.h"
#include "common/fe_inner_attr_define.h"
#include "ops_store/ops_kernel_manager.h"
#include "adapter/common/op_store_adapter_manager.h"
#include "register/graph_optimizer/fusion_common/unknown_shape_utils.h"
#include "mmpa/mmpa_api.h"

namespace fe {
namespace {
const std::string kSetOpInfoThreadPrefix = "judge_2_";

void PrintFallbackCfgInfo(const ge::NodePtr &node_ptr, const std::vector<bool> &fallback_info) {
  if (!IsDebugLogLevel || fallback_info.empty()) {
    return;
  }
  std::stringstream fallback_str;
  fallback_str << "[";
  for (const bool &fallback_cfg : fallback_info) {
    fallback_str << std::boolalpha << fallback_cfg << ", ";
  }
  fallback_str << "]";
  FE_LOGD("[JudgeInsert][SetAclnnAttr] Node[%s, %s] prints fallback configuration: %s.",
          node_ptr->GetNamePtr(), node_ptr->GetTypePtr(), fallback_str.str().c_str());
}

void SetOppKernelPathForAclnn(const ge::OpDescPtr &op_desc) {
  std::string opp_kernel_path = "";
  if (Configuration::Instance(AI_CORE_NAME).GetOppLatestPath(opp_kernel_path) != SUCCESS) {
    FE_LOGD("Node[%s, %s]Unable to retrieve opp latest path", op_desc->GetNamePtr(), op_desc->GetTypePtr());
    return;
  }
  (void)ge::AttrUtils::SetInt(op_desc, ge::ATTR_NAME_BINARY_SOURCE, 1);
  FE_LOGD("Node[%s, %s]Opp latest path: %s, set [%s] to 1", op_desc->GetNamePtr(), op_desc->GetTypePtr(),
          opp_kernel_path.c_str(), ge::ATTR_NAME_BINARY_SOURCE);
}
} // namespace
using OpSliceUtilPtr = std::shared_ptr<OpSliceUtil>;

const std::map<string, string> OpSetter::attr_map_ = {{fe::STR_INPUT_MEM_CONTINUES, fe::ATTR_NAME_CONTINUOUS_INPUT},
                                                      {fe::STR_OUTPUT_MEM_CONTINUES, fe::ATTR_NAME_CONTINUOUS_OUTPUT}};

OpSetter::OpSetter(const std::string &engine_name) : engine_name_(engine_name) {}
OpSetter::~OpSetter() {}

Status OpSetter::InitializeQuerier() {
  FE_MAKE_SHARED(format_dtype_querier_ptr_ = std::make_shared<FormatDtypeQuerier>(engine_name_), return FAILED);
  return SUCCESS;
}

Status OpSetter::MultiThreadSetOpInfo(ge::ComputeGraph &graph, bool only_set_attr) const {
  const auto &nodes = graph.GetAllNodes();
  if (nodes.size() < DEFAULT_THREAD_NUM) {
    return SetOpInfo(graph);
  }
  uint32_t thread_num = 8;
  fe::ThreadPool executor(kSetOpInfoThreadPrefix + fe::GetCurThreadIdStr(), thread_num);
  std::vector<std::future<Status>> vector_future;

  for (auto &node : nodes) {
    std::future<Status> f = executor.commit(OpSetter::SetOpInfoByNode, node, engine_name_, only_set_attr);
    if (!f.valid()) {
      FE_LOGE("[Call][Commit] failed, future is invalid for node: %s", node->GetName().c_str());
      return FAILED;
    }
    vector_future.emplace_back(std::move(f));
  }
  for (size_t i = 0; i < vector_future.size(); ++i) {
    Status ret_status = FAILED;
    try {
      ret_status = vector_future[i].get();
    } catch (const std::exception &exp) {
      FE_LOGE("Exception occurred with error message: %s", exp.what());
      ret_status = FAILED;
    }
    if (ret_status != SUCCESS) {
      REPORT_FE_ERROR("Thread number %zu, SetOpInfoByNode failed", i);
      FE_LOGE("[Check][Param] setting operation info for graph %s failed", graph.GetName().c_str());
      return ret_status;
    }
  }
  return SUCCESS;
}

Status OpSetter::SetOpInfo(const ge::ComputeGraph& graph) const {
  for (auto& node : graph.GetAllNodes()) {
    Status result = SetOpInfoByNode(node, engine_name_);
    if (result != SUCCESS) {
      return result;
    }
  }
  return SUCCESS;
}

Status OpSetter::OnlySetOpDescAttr(const ge::ComputeGraph& graph) const {
  for (auto& node : graph.GetAllNodes()) {
    Status result = SetOpInfoByNode(node, engine_name_, true);
    if (result != SUCCESS) {
      return result;
    }
  }
  return SUCCESS;
}

void OpSetter::GetOpKernelInfoByImplType(const ge::OpDescPtr &op_desc_ptr, const std::string &engine_name,
                                         OpKernelInfoPtr &op_kernel_info_ptr) {
  // 1. check the imply_type
  const string &op_name = op_desc_ptr->GetName();
  const string &op_type = op_desc_ptr->GetType();
  int64_t imply_type = -1;
  if (!ge::AttrUtils::GetInt(op_desc_ptr, FE_IMPLY_TYPE, imply_type)) {
    FE_LOGD("Op[name=%s, type=%s]: Failed to get the attribute FE_IMPLY_TYPE.", op_name.c_str(), op_type.c_str());
    return;
  }

  // 2. get the op_kernel_info_ptr by op_impl_type and op_type
  OpImplType op_impl_type = static_cast<OpImplType>(imply_type);
  op_kernel_info_ptr =
      OpsKernelManager::Instance(engine_name).GetOpKernelInfoByOpType(op_impl_type, op_type);
}

Status OpSetter::SetOpDescAttrByNode(const ge::OpDescPtr &op_desc_ptr,
    const OpKernelInfoPtr &op_kernel_info_ptr) {
  for (auto &it : attr_map_) {
    if (SetOpDescAttr(op_kernel_info_ptr, it.first, it.second, op_desc_ptr) != SUCCESS) {
      REPORT_FE_ERROR(
          "[GraphOptJdgInst][SetOpInfo][SetOpInfo] Op[name=%s,type=%s]: get the attribute [%s] from the op "
          "kernel info, and set the attribute [%s] failed!",
          op_desc_ptr->GetName().c_str(), op_desc_ptr->GetType().c_str(), it.first.c_str(), it.second.c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

Status OpSetter::SetOpSliceInfoByNode(const ge::NodePtr &node_ptr, const std::string &engine_name,
                                      OpKernelInfoPtr &op_kernel_info_ptr) {
  SlicePattern slice_pattern = op_kernel_info_ptr->GetOpSlicePattern();
  OpSliceUtilPtr slice_util_ptr = nullptr;
  FE_MAKE_SHARED(slice_util_ptr = std::make_shared<OpSliceUtil>(), return FAILED);
  FE_CHECK_NOTNULL(slice_util_ptr);
  const string &op_name = node_ptr->GetName();
  const string &op_type = node_ptr->GetType();
  if (slice_util_ptr->SetOpSliceInfo(node_ptr, slice_pattern, true) != SUCCESS) {
    FE_LOGW("Op[name=%s, type=%s]: failed to set slice info for node.", op_name.c_str(), op_type.c_str());
  }

  if (SetTbeOpSliceInfo(node_ptr, engine_name, op_kernel_info_ptr) != SUCCESS) {
    FE_LOGW("Op[name=%s, type=%s]: Failed to set TBE slice info for node!", op_name.c_str(), op_type.c_str());
  }
  return SUCCESS;
}

Status OpSetter::SetOpInfoByNode(const ge::NodePtr &node_ptr, const std::string &engine_name,
                                 const bool only_set_attr) {
  // 1. check the node_ptr and the op_desc_ptr
  FE_CHECK_NOTNULL(node_ptr);
  ge::OpDescPtr op_desc_ptr = node_ptr->GetOpDesc();
  FE_CHECK_NOTNULL(op_desc_ptr);

  OpKernelInfoPtr op_kernel_info_ptr = nullptr;
  GetOpKernelInfoByImplType(op_desc_ptr, engine_name, op_kernel_info_ptr);
  if (op_kernel_info_ptr == nullptr) {
    FE_LOGD("Op kernel information for op [%s, %s] was not found.",
            op_desc_ptr->GetNamePtr(), op_desc_ptr->GetTypePtr());
    return SUCCESS;
  }

  if (SetOpDescAttrByNode(op_desc_ptr, op_kernel_info_ptr) != SUCCESS) {
    return FAILED;
  }

  if (only_set_attr) {
    return SUCCESS;
  }

  if (SetOpSliceInfoByNode(node_ptr, engine_name, op_kernel_info_ptr) != SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

Status OpSetter::SetTbeOpSliceInfo(const ge::NodePtr &node_ptr, const std::string &engine_name,
                                   OpKernelInfoPtr &op_kernel_info_ptr) {
  FE_CHECK_NOTNULL(node_ptr);
  ge::OpDescPtr op_desc_ptr = node_ptr->GetOpDesc();
  FE_CHECK_NOTNULL(op_desc_ptr);
  OpStoreAdapterPtr op_store_adapter = nullptr;
  Status status = OpStoreAdapterManager::Instance(engine_name).GetOpStoreAdapter(EN_IMPL_HW_TBE, op_store_adapter);
  if (status != SUCCESS) {
    REPORT_FE_ERROR("[OrigGraphOpt][OpSetter] Failed to get op store adapter by impl_type [%ld].", EN_IMPL_HW_TBE);
    return OP_COMPILER_CHECK_FALSE_FAILED;
  }
  if (op_kernel_info_ptr->GetOpInfo().opFileName == NULL_OP_FILE) {
    return SUCCESS;
  }

  // set tbe slice info
  Status ret = op_store_adapter->SetTbeOpSliceInfo(node_ptr, op_kernel_info_ptr);
  if (ret != SUCCESS) {
    REPORT_FE_ERROR("[OrigGraphOpt][OpSetter] SetTbeOpSliceInfo failed for graph [%s].", op_desc_ptr->GetName().c_str());
    return ret;
  }
  return SUCCESS;
}

Status OpSetter::SetOpDescAttr(const OpKernelInfoPtr &op_kernel_info_ptr, const std::string &attr_name,
                               const std::string &ge_attr_name, ge::OpDescPtr op_desc_ptr) {
  string op_name = op_desc_ptr->GetName();
  string op_type = op_desc_ptr->GetType();
  bool value = false;

  if (attr_name == STR_INPUT_MEM_CONTINUES) {
    value = op_kernel_info_ptr->IsInputMemContinues();
  } else if (attr_name == STR_OUTPUT_MEM_CONTINUES) {
    value = op_kernel_info_ptr->IsOutputMemContinues();
  } else {
    REPORT_FE_ERROR("[GraphOptJdgInst][SetOpInfo][SetOpDescAttr] the attribute [%s] can not get from op kernel.",
                    attr_name.c_str());
    return FAILED;
  }
  FE_LOGD("Op[name=%s,type=%s]: succeed to get the attribute [%s], value is %d", op_name.c_str(), op_type.c_str(),
          attr_name.c_str(), value);

  if (!ge::AttrUtils::SetBool(op_desc_ptr, ge_attr_name, value)) {
    REPORT_FE_ERROR("[GraphOptJdgInst][SetOpInfo][SetOpDescAttr] Failed to set attribute [%s] for Op[name=%s,type=%s].",
                    ge_attr_name.c_str(), op_name.c_str(), op_type.c_str());
    return FAILED;
  }
  FE_LOGD("Op[name=%s,type=%s]: set the attribute [%s] successfully", op_name.c_str(), op_type.c_str(),
          ge_attr_name.c_str());
  return SUCCESS;
}

void OpSetter::SetOpImplMode(const ge::ComputeGraph& graph) const {
  for (auto& node : graph.GetAllNodes()) {
    SetOpImplModeByNode(node);
  }
  if (!Configuration::Instance(engine_name_).IsEnableCustomImplMode()) {
    return;
  }
  for (auto& node : graph.GetAllNodes()) {
    SetOpCustomImplModeByNode(node);
  }
  return;
}

void OpSetter::SetOpImplModeByNode(const ge::NodePtr &node_ptr) const {
  if (node_ptr == nullptr) {
    return;
  }
  ge::OpDescPtr op_desc_ptr = node_ptr->GetOpDesc();
  std::string op_impl_mode;
  if (!Configuration::Instance(engine_name_)
      .GetOpImplMode(op_desc_ptr->GetName(), op_desc_ptr->GetType(), op_impl_mode)) {
    FE_LOGD("Operation [name=%s, type=%s] cannot obtain op_impl_mode from the configuration.", op_desc_ptr->GetNamePtr(),
            op_desc_ptr->GetTypePtr());
    return;
  }
  auto iter = kOpImplStrToInt.find(op_impl_mode);
  if (iter != kOpImplStrToInt.end()) {
    (void)ge::AttrUtils::SetInt(op_desc_ptr, OP_IMPL_MODE_ENUM, iter->second);
    FE_LOGD("op[name=%s, type=%s] set op attr _op_impl_mode_enum 0x%llx with op_impl_mode %s.",
            op_desc_ptr->GetNamePtr(), op_desc_ptr->GetTypePtr(), iter->second, op_impl_mode.c_str());
  }
}

void OpSetter::SetOpCustomImplModeByNode(const ge::NodePtr &node_ptr) const {
  if (node_ptr == nullptr) {
    return;
  }
  ge::OpDescPtr op_desc_ptr = node_ptr->GetOpDesc();
  std::string op_impl_mode;
  if (!Configuration::Instance(engine_name_).GetOpImplMode(op_desc_ptr->GetName(), "", op_impl_mode)) {
    FE_LOGD("Operation [name=%s, type=%s] cannot obtain op_impl_mode from custom configuration.", op_desc_ptr->GetName().c_str(),
            op_desc_ptr->GetType().c_str());
    return;
  }
  auto iter = kOpImplStrToInt.find(op_impl_mode);
  if (iter != kOpImplStrToInt.end()) {
    (void)ge::AttrUtils::SetInt(op_desc_ptr, OP_CUSTOM_IMPL_MODE_ENUM, iter->second);
    FE_LOGD("op[name=%s, type=%s] set op attr _op_custom_impl_mode_enum 0x%llx with op_impl_mode %s.",
            op_desc_ptr->GetNamePtr(), op_desc_ptr->GetTypePtr(), iter->second, op_impl_mode.c_str());
  }
}

void OpSetter::SetOpDebugAttr(const ge::ComputeGraph& graph) {
  for (auto node : graph.GetDirectNode()) {
    ge::OpDescPtr op_desc = node->GetOpDesc();
    if (fe::Configuration::Instance(fe::AI_CORE_NAME).IsConfigDebugListOp(op_desc)) {
      string op_type = op_desc->GetType().c_str();
      string op_name = op_desc->GetName().c_str();
      FE_LOGD("graph init op[%s, %s] sets op_debug_compile attr",
               op_type.c_str(), op_name.c_str());
      ge::AttrUtils::SetBool(node->GetOpDesc(), kOpDebugCompile, true);
    }
  }
}

void OpSetter::SetQuantDumpableAttr(const ge::ComputeGraph& graph) {
  if (!Configuration::Instance(AI_CORE_NAME).IsQuantDumpable()) {
    return;
  }
  for (const ge::NodePtr &node : graph.GetDirectNode()) {
    if (node == nullptr || (node->GetType() != ASCEND_QUANT && node->GetType() != OP_TYPE_QUANTIZE)) {
      continue;
    }
    (void)ge::AttrUtils::SetBool(node->GetOpDesc(), kAttrDumpAble, true);
    FE_LOGD("Set dumpable flag for node[%s, %s].", node->GetNamePtr(), node->GetTypePtr());
  }
}

void OpSetter::SetAttrForAclnnLowering(const ge::NodePtr &node_ptr, const fe::PrecisionMode &precision_mode) const {
  (void)ge::AttrUtils::SetInt(node_ptr->GetOpDesc(), kPrecisionModeEnum, static_cast<int>(precision_mode));
  std::string deterministic_str = "";
  (void)ge::GetThreadLocalContext().GetOption(ge::DETERMINISTIC, deterministic_str);
  if (!deterministic_str.empty()) {
    (void)ge::AttrUtils::SetStr(node_ptr->GetOpDesc(), ge::DETERMINISTIC, deterministic_str);
  }

  std::string allow_hf32_str = "";
  (void)ge::GetThreadLocalContext().GetOption(ge::ALLOW_HF32, allow_hf32_str);
  if (!allow_hf32_str.empty()) {
    allow_hf32_str = Configuration::Instance(AI_CORE_NAME).EmplaceHf32ModeForAclnn(allow_hf32_str);
    (void)ge::AttrUtils::SetStr(node_ptr->GetOpDesc(), ge::ALLOW_HF32, allow_hf32_str);
  }
}

bool OpSetter::SupportFallback(const ge::NodePtr &node_ptr, const uint32_t &matched_index,
                               const OpKernelInfoPtr &op_kernel_info_ptr) const {
  map<string, vector<ge::Format>> format_map;
  if (format_dtype_querier_ptr_ == nullptr) {
    return false;
  }
  if (format_dtype_querier_ptr_->GetAllSupportFormat(op_kernel_info_ptr, node_ptr, format_map) != SUCCESS) {
    return false;
  }
  for (const auto &item : format_map) {
    if (FE_ORIGIN_FORMAT_SET.count(item.second[matched_index]) == 0) {
      return false;
    }
  }
  return true;
}

bool OpSetter::FallbackConfigured(const ge::NodePtr &node_ptr, const OpKernelInfoPtr &op_kernel_info_ptr,
                                  std::vector<bool> &fallback_flags) const {
  if (format_dtype_querier_ptr_ == nullptr) {
    return false;
  }
  (void)format_dtype_querier_ptr_->GetFallbackFlags(op_kernel_info_ptr, node_ptr, fallback_flags);
  return !fallback_flags.empty();
}

void OpSetter::SetFallbackAttr(const ge::ComputeGraph &graph, const fe::PrecisionMode &precision_mode,
  bool &need_update_stream_core_limit) const {
  for (const auto &node : graph.GetAllNodes()) {
    OpKernelInfoPtr op_kernel_info_ptr = nullptr;
    GetOpKernelInfoByImplType(node->GetOpDesc(), engine_name_, op_kernel_info_ptr);
    if (op_kernel_info_ptr == nullptr) {
      FE_LOGD("Op kernel information for op [%s, %s] was not found",
              node->GetNamePtr(), node->GetTypePtr());
      continue;
    }
    int64_t matched_index = -1;
    (void)ge::AttrUtils::GetInt(node->GetOpDesc(), "judge_match_idx", matched_index);
    FE_LOGD("Got judge_match_idx from node [%s, %s]: [%ld]",
            node->GetNamePtr(), node->GetTypePtr(), matched_index);
    bool is_aclnn_fallback = SetAclnnAttr(node, matched_index, op_kernel_info_ptr, precision_mode);
    if (matched_index != -1 && !is_aclnn_fallback) {
      FE_LOGD("Node[%s, %s] will not do fallback.",
              node->GetNamePtr(), node->GetTypePtr());
    }
    need_update_stream_core_limit = need_update_stream_core_limit || is_aclnn_fallback;
  }
}

bool OpSetter::SetAclnnAttr(const ge::NodePtr &node_ptr, const uint32_t &matched_index,
                            const OpKernelInfoPtr &op_kernel_info_ptr, const fe::PrecisionMode &precision_mode) const {
  FE_LOGD("[JudgeInsert][SetAclnnAttr] Ready to check fallback info for node[%s, %s].",
          node_ptr->GetNamePtr(), node_ptr->GetTypePtr());
  std::vector<bool> fallback_info;
  if (FallbackConfigured(node_ptr, op_kernel_info_ptr, fallback_info)) {
    PrintFallbackCfgInfo(node_ptr, fallback_info);
    if (fallback_info.size() <= matched_index) {
      FE_LOGW("[JudgeInsert][SetAclnnAttr] Node[%s, %s] fallback size [%zu] should not be less than match index [%u].",
              node_ptr->GetNamePtr(), node_ptr->GetTypePtr(), fallback_info.size(), matched_index);
      return false;
    }
    if (fallback_info[matched_index]) {
      FE_LOGD("[JudgeInsert][SetAclnnAttr] Node[%s, %s] set fallback attr",
              node_ptr->GetNamePtr(), node_ptr->GetTypePtr());
      (void)ge::AttrUtils::SetBool(node_ptr->GetOpDesc(), ge::ATTR_NAME_FORCE_UNKNOWN_SHAPE, true);
      (void)ge::AttrUtils::SetBool(node_ptr->GetOpDesc(), ATTR_NAME_FALLBACK_ACLNN, true);
      (void)ge::AttrUtils::SetStr(node_ptr->GetOpDesc(), ge::kAttrLowingFunc, kAclnnLoweringFunc);
      SetOppKernelPathForAclnn(node_ptr->GetOpDesc());
      SetAttrForAclnnLowering(node_ptr, precision_mode);
      return true;
    }
    return false;
  }

  AclnnSupportType aclnn_support_type = op_kernel_info_ptr->GetAclnnSupportType();
  if (aclnn_support_type == AclnnSupportType::ACLNN_ONLY) {
    FE_LOGD("[JudgeInsert][SetAclnnAttr] Node [%s, %s] is an aclnn_only operation, setting fallback attribute",
            node_ptr->GetNamePtr(), node_ptr->GetTypePtr());
    (void)ge::AttrUtils::SetBool(node_ptr->GetOpDesc(), ge::ATTR_NAME_FORCE_UNKNOWN_SHAPE, true);
    (void)ge::AttrUtils::SetBool(node_ptr->GetOpDesc(), ATTR_NAME_FALLBACK_ACLNN, true);
    (void)ge::AttrUtils::SetStr(node_ptr->GetOpDesc(), ge::kAttrLowingFunc, kAclnnLoweringFunc);
    SetOppKernelPathForAclnn(node_ptr->GetOpDesc());
    SetAttrForAclnnLowering(node_ptr, precision_mode);
    return true;
  }
  bool base_cond = Configuration::Instance(AI_CORE_NAME).IsEnableAclnn() || IsDefaultEnableAclnn(node_ptr);
  FE_LOGI("Node [%s, %s] has an ACLNN base condition of [%d], and supports type [%d].",
          node_ptr->GetNamePtr(), node_ptr->GetTypePtr(), base_cond, static_cast<int>(aclnn_support_type));
  if (base_cond && aclnn_support_type == AclnnSupportType::SUPPORT_ACLNN &&
      UnknownShapeUtils::IsUnknownShapeOp(*node_ptr->GetOpDesc()) &&
      SupportFallback(node_ptr, matched_index, op_kernel_info_ptr)) {
    FE_LOGD("[JudgeInsert][SetAclnnAttr] Node [%s, %s] supports ACLNN operations; setting fallback attribute.",
            node_ptr->GetNamePtr(), node_ptr->GetTypePtr());
    (void)ge::AttrUtils::SetBool(node_ptr->GetOpDesc(), ATTR_NAME_FALLBACK_ACLNN, true);
    (void)ge::AttrUtils::SetStr(node_ptr->GetOpDesc(), ge::kAttrLowingFunc, kAclnnLoweringFunc);
    SetOppKernelPathForAclnn(node_ptr->GetOpDesc());
    SetAttrForAclnnLowering(node_ptr, precision_mode);
    return true;
  }
  return false;
}

void OpSetter::DeleteFusionScope(ge::ComputeGraph& graph) const {
  for (auto &node : graph.GetAllNodes()) {
    if (node == nullptr) {
      continue;
    }
    ge::OpDescPtr op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      continue;
    }
    int64_t scope_id = 0;
    if (!ge::AttrUtils::GetInt(op_desc, SCOPE_ID_ATTR, scope_id)) {
      continue;
    }
    if (scope_id >= 0) {
      ge::AttrUtils::SetBool(op_desc, kAttrNameIsFusionOp, true);
    }
    (void)op_desc->DelAttr(SCOPE_ID_ATTR);
  }
}

bool OpSetter::IsDefaultEnableAclnn(const ge::NodePtr &node_ptr) const {
  auto graph = node_ptr->GetOwnerComputeGraph();
  bool is_single_op_scene = false;
  if (!ge::AttrUtils::GetBool(graph, ge::ATTR_SINGLE_OP_SCENE, is_single_op_scene)) {
    FE_LOGI("Node [%s, %s] unable to get single_op_scene[%d] flag from owner graph [%s], attempting to get from root graph.",
            node_ptr->GetNamePtr(), node_ptr->GetTypePtr(), is_single_op_scene, graph->GetName().c_str());
    ge::ComputeGraphPtr root_graph = nullptr;
    root_graph = ge::GraphUtils::FindRootGraph(graph);
    if (root_graph == nullptr) {
      FE_LOGI("Node [%s, %s] owner graph [%s] unable to get root graph, will not default enable aclnn.",
              node_ptr->GetNamePtr(), node_ptr->GetTypePtr(), graph->GetName().c_str());
      return false;
    }
    if (ge::AttrUtils::GetBool(root_graph, ge::ATTR_SINGLE_OP_SCENE, is_single_op_scene)) {
      FE_LOGI("Node [%s, %s] received single_op_scene [%d] flag from the owner root graph [%s].",
              node_ptr->GetNamePtr(), node_ptr->GetTypePtr(), is_single_op_scene, root_graph->GetName().c_str());
    }
  }
  int64_t imply_type = -1;
  (void)ge::AttrUtils::GetInt(node_ptr->GetOpDesc(), FE_IMPLY_TYPE, imply_type);
  OpStoreAdapterPtr op_store_adapter =
        OpStoreAdapterManager::Instance(AI_CORE_NAME).GetOpStoreAdapter(static_cast<OpImplType>(imply_type));
  if (op_store_adapter == nullptr) {
    FE_LOGI("Node[%s, %s] unable to get tbe adapter, ACLNN will not be enabled by default.",
            node_ptr->GetNamePtr(), node_ptr->GetTypePtr());
    return false;
  }
  static bool is_oppkernel_installed = op_store_adapter->JudgeBuiltInOppKernelInstalled();
  static bool is_rt2_mode = Configuration::Instance(AI_CORE_NAME).IsEnableRt2();
  bool res = is_oppkernel_installed && !is_single_op_scene && is_rt2_mode;
  FE_LOGI("Node[%s, %s] enable_aclnn_flag [%d], single_op_scene [%d], is_oppKernel_installed [%d], rt2_mode [%d].",
          node_ptr->GetNamePtr(), node_ptr->GetTypePtr(), res, is_single_op_scene, is_oppkernel_installed, is_rt2_mode);
  return res;
}
}  // namespace fe

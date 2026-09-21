/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/fe_op_info_common.h"
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "common/configuration.h"
#include "common/fe_inner_attr_define.h"
#include "common/fe_type_utils.h"
#include "common/graph/fe_graph_utils.h"
#include "platform/platform_info.h"
#include "common/platform_utils.h"
#include "common/scope_allocator.h"
#include "common/math_util.h"
#include "graph/ge_context.h"
#include "graph/utils/op_desc_utils.h"
#include "ops_store/ops_kernel_manager.h"
#include "graph/utils/op_type_utils.h"
#include "common/tile_fwk_op_info.h"
namespace fe {
std::mutex run_info_map_mutex;
std::unordered_map<std::string, RunInfoPtr> run_info_map;

std::string GetCheckSupportedString(te::CheckSupportedResult &check_supported) {
  auto iter = CHECKSUPPORTED_STR_MAP.find(check_supported);
  if (iter == CHECKSUPPORTED_STR_MAP.end()) {
    return STR_NOT_SUPPORTED;
  } else {
    return iter->second;
  }
}

std::string GetStrByFinComTaskVec(const std::vector<te::FinComTask> &fin_com_task) {
  std::string result;
  size_t size = fin_com_task.size();
  for (size_t i = 0; i < size; ++i) {
    uint64_t task_id = fin_com_task[i].taskId;
    result = result + "[" + std::to_string(task_id) + "]";
    if (i != size - 1) {
      result += ",";
    }
  }
  return result;
}

void GetCoreTypeAndRatio(const ge::OpDescPtr &op_desc, const int64_t cube_ratio, const int64_t vector_ratio,
                         std::string &core_type, int32_t &ratio) {
  string tmp_core_type;
  if (cube_ratio == 0) {
    ratio = 0;
    tmp_core_type = kCoreTypeMixAIV;
  } else if (vector_ratio == 0) {
    ratio = 0;
    tmp_core_type = kCoreTypeMixAIC;
  } else if (cube_ratio > vector_ratio) {
    ratio = static_cast<int32_t>(cube_ratio / vector_ratio);
    tmp_core_type = kCoreTypeMixAIV;
    (void)ge::AttrUtils::SetBool(op_desc, kMixIsAiv, true);
  } else {
    ratio = static_cast<int32_t>(vector_ratio / cube_ratio);
    tmp_core_type = kCoreTypeMixAIC;
  }
  if (core_type != kCoreTypeMixEnhance && core_type != kCoreTypeMixVectorCore &&
      core_type != kCoreTypeMixAICore) {
    core_type = tmp_core_type;
  }
}

Status UpdateMixByTilingKey(const ge::OpDescPtr &op_desc, const RunInfoPtr &tiling_info) {
  if (!op_desc->HasAttr(kMixDynamicRatio)) {
    return SUCCESS;
  }
  FE_LOGD("Update mix ratio attr by tilingKey");
  ge::GeAttrValue::NAMED_ATTRS tiling_with_ratio;
  if (!ge::AttrUtils::GetNamedAttrs(op_desc, kDynRatioAttr, tiling_with_ratio)) {
    FE_LOGE("Mix ratio attr mix_tiling_with_ratio_attr not found.");
    return FAILED;
  }
  std::string tiling_key_str = std::to_string(tiling_info->GetTilingKey());
  std::vector<std::string> tiling_key_vec;
  (void)tiling_with_ratio.GetItem(kDynRatioTiling).GetValue<std::vector<std::string>>(tiling_key_vec);
  size_t all_num = tiling_key_vec.size();
  std::vector<int64_t> c_ratio_vec;
  std::vector<int64_t> v_ratio_vec;
  (void)tiling_with_ratio.GetItem(kDynRatioCRatio).GetValue<std::vector<int64_t>>(c_ratio_vec);
  (void)tiling_with_ratio.GetItem(kDynRatioVRatio).GetValue<std::vector<int64_t>>(v_ratio_vec);
  if (c_ratio_vec.size() != v_ratio_vec.size() || v_ratio_vec.size() != all_num || all_num == 0) {
    FE_LOGE("Mix ratio attr vectors size invalid");
    return FAILED;
  }
  size_t i = 0;
  for (; i < all_num; ++i) {
    if (tiling_key_vec[i] == tiling_key_str) {
      break;
    }
  }
  if (i == all_num) {
    return FAILED;
  }

  FE_LOGD("Matched mix ratio attr with tilingKey[%s].", tiling_key_vec[i].c_str());
  int32_t ratio = 0;
  std::string core_type;
  (void)ge::AttrUtils::GetStr(op_desc, ATTR_NAME_CUBE_VECTOR_CORE_TYPE, core_type);
  GetCoreTypeAndRatio(op_desc, c_ratio_vec[i], v_ratio_vec[i], core_type, ratio);
  ge::AttrUtils::SetStr(op_desc, ATTR_NAME_CUBE_VECTOR_CORE_TYPE, core_type);
  ge::AttrUtils::SetInt(op_desc, kTaskRadio, ratio);
  return SUCCESS;
}

Status TilingResultCheck(const ge::OpDescPtr &op_desc, RunInfoPtr &tiling_info) {
  RunInfoPtr tmp_tiling_info = nullptr;
  tmp_tiling_info = op_desc->TryGetExtAttr(ge::ATTR_NAME_OP_RUN_INFO, nullptr);
  if (tmp_tiling_info != nullptr) {
    FE_LOGD("Node [%s] already has tiling data attr.", op_desc->GetName().c_str());
    tiling_info = tmp_tiling_info;
    return SUCCESS;
  }

  if (IsStcToDynSoftSyncOp(op_desc)) {
    FE_LOGD("Node[%s] is sync op, can not reuse tiling.", op_desc->GetName().c_str());
    return FAILED;
  }

  std::string tiling_key;
  (void)ge::AttrUtils::GetStr(op_desc, kTilingRemoveDuplicates, tiling_key);
  std::lock_guard<std::mutex> lock_guard(run_info_map_mutex);
  auto iter = run_info_map.find(tiling_key);
  if (iter != run_info_map.end() && iter->second != nullptr) {
    if (UpdateMixByTilingKey(op_desc, iter->second) != SUCCESS) {
      FE_LOGE("Node[%s] update mix attr by tilingKey failed.", op_desc->GetName().c_str());
      return FAILED;
    }
    if (op_desc->GetType() != MEMSET_OP_TYPE) {
      op_desc->SetWorkspaceBytes(iter->second->GetAllWorkspaces());
    }
    op_desc->SetExtAttr(ge::ATTR_NAME_OP_RUN_INFO, iter->second);
    (void)ge::AttrUtils::SetInt(op_desc, ge::TVM_ATTR_NAME_BLOCKDIM, iter->second->GetBlockDim());

    FE_LOGD("Node[%s] has cache tiling data with key[%s], tiling_data size:%zu, blockdim:%u.",
        op_desc->GetName().c_str(), tiling_key.c_str(), iter->second->GetAllTilingData().str().size(),
        iter->second->GetBlockDim());
    tiling_info = iter->second;
    return SUCCESS;
  }

  return FAILED;
}

void SetTilingResult(const ge::OpDescPtr &op_desc, const RunInfoPtr &tiling_info) {
  op_desc->SetExtAttr(ge::ATTR_NAME_OP_RUN_INFO, tiling_info);
  if (op_desc->GetType() != MEMSET_OP_TYPE) {
    op_desc->SetWorkspaceBytes(tiling_info->GetAllWorkspaces());
  }
  (void)ge::AttrUtils::SetStr(op_desc, kAttrTilingDataStr, tiling_info->GetAllTilingData().str());
  (void)ge::AttrUtils::SetInt(op_desc, ge::TVM_ATTR_NAME_BLOCKDIM, tiling_info->GetBlockDim());
  (void)ge::AttrUtils::SetInt(op_desc, kAicpuBlockDim, tiling_info->GetAicpuBlockDim());
  (void)ge::AttrUtils::SetInt(op_desc, kAttrScheduleMode, tiling_info->GetScheduleMode());
  (void)ge::AttrUtils::SetInt(op_desc, kLocalMemorySize, tiling_info->GetLocalMemorySize());
  FE_LOGD("Set attr block dim[%u], aicpu blockdim[%u] and schedule mode[%u] for op[%s, %s].",
          tiling_info->GetBlockDim(), tiling_info->GetAicpuBlockDim(), tiling_info->GetScheduleMode(),
          op_desc->GetNamePtr(), op_desc->GetTypePtr());

  if (IsStcToDynSoftSyncOp(op_desc)) {
    FE_LOGD("Node[%s] is sync op, no need to store tiling.", op_desc->GetName().c_str());
    return;
  }

  std::string tiling_key;
  (void)ge::AttrUtils::GetStr(op_desc, kTilingRemoveDuplicates, tiling_key);
  if (tiling_key.empty()) {
    FE_LOGW("Tiling cache key of op[%s] is empty.", op_desc->GetName().c_str());
  } else {
    std::lock_guard<std::mutex> lock_guard(run_info_map_mutex);
    run_info_map.emplace(tiling_key, tiling_info);
  }
}

Status GetRunInfoFromGe(const ge::NodePtr &node, const PlatFormInfos &platform_infos, const RunInfoPtr &tiling_info) {
  ge::Operator op = ge::OpDescUtils::CreateOperatorFromNode(node);
  if (node->GetType() == MEMSET_OP_TYPE) {
    if (optiling::AtomicRtParseAndTiling(op, platform_infos, *tiling_info) != ge::GRAPH_SUCCESS) {
      FE_LOGE("Do tiling failed for node:%s.", node->GetName().c_str());
      return FAILED;
    }
  } else {
    if (optiling::AicoreRtParseAndTiling(op, platform_infos, *tiling_info) != ge::GRAPH_SUCCESS) {
      FE_LOGE("Do tiling failed for node:%s.", node->GetName().c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

Status UpdateTilingResult(const RunInfoPtr &tiling_info, const RunInfoPtr &temp_tiling_info) {
  const std::vector<int64_t> &old_workspace = tiling_info->GetAllWorkspaces();
  if (old_workspace.empty()) {
    tiling_info->SetWorkspaces(temp_tiling_info->GetAllWorkspaces());
    return SUCCESS;
  }

  const std::vector<int64_t> &new_workspace = temp_tiling_info->GetAllWorkspaces();
  if (new_workspace.empty()) {
    return SUCCESS;
  }

  if (old_workspace.size() != new_workspace.size()) {
    FE_LOGE("OldWorkspaceSize[%zu] not equal to newWorkspaceSize[%zu].", old_workspace.size(), new_workspace.size());
    return FAILED;
  }

  std::vector<int64_t> max_workspace;
  for (size_t i = 0U; i < old_workspace.size(); ++i) {
    if (old_workspace[i] < new_workspace[i]) {
      max_workspace.push_back(new_workspace[i]);
    } else {
      max_workspace.push_back(old_workspace[i]);
    }
  }
  tiling_info->SetWorkspaces(max_workspace);

  return SUCCESS;
}

Status TilingForSoftSyncOp(const ge::NodePtr &node, PlatFormInfos &platform_infos,
                           const RunInfoPtr &tiling_info) {
  const std::vector<uint32_t> &vir_type_list = PlatformUtils::Instance().GetVirTypeList();
  for (auto &vir_type : vir_type_list) {
    RunInfoPtr temp_tiling_info;
    FE_MAKE_SHARED(temp_tiling_info = std::make_shared<optiling::utils::OpRunInfo>(0, false, 0), return FAILED);
    FE_CHECK_NOTNULL(temp_tiling_info);
    platform_infos.SetCoreNum(vir_type);
    if (GetRunInfoFromGe(node, platform_infos, temp_tiling_info) != SUCCESS) {
      FE_LOGE("SoftSyncOp[%s] get runinfo from ge with vir_type[%u] failed.", node->GetName().c_str(), vir_type);
      return FAILED;
    }
    if (UpdateTilingResult(tiling_info, temp_tiling_info) != SUCCESS) {
      FE_LOGE("SoftSyncOp[%s] update tiling result with vir_type[%u] failed.", node->GetName().c_str(), vir_type);
      return FAILED;
    }
  }
  return SUCCESS;
}

Status TilingForOneNode(const ge::NodePtr &node) {
  FE_LOGD("Try do Tiling for node:%s.", node->GetName().c_str());
  auto op_desc = node->GetOpDesc();
  RunInfoPtr tiling_info = nullptr;
  if (TilingResultCheck(op_desc, tiling_info) == SUCCESS) {
    SetTilingResult(op_desc, tiling_info);
    return SUCCESS;
  }

  PlatFormInfos platform_infos;
  OptionalInfos opti_compilation_info;
  if (PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_infos,
      opti_compilation_info) != SUCCESS) {
    FE_LOGE("Do tiling failed for node:%s, failed to get platform.", node->GetName().c_str());
    return FAILED;
  }

  FE_MAKE_SHARED(tiling_info = std::make_shared<optiling::utils::OpRunInfo>(0, false, 0), return FAILED);
  FE_CHECK_NOTNULL(tiling_info);
  const std::string *core_type = ge::AttrUtils::GetStr(op_desc, ATTR_NAME_CUBE_VECTOR_CORE_TYPE);
  if (core_type == nullptr) {
    FE_LOGE("Do tiling failed for node:%s, No core type.", node->GetName().c_str());
    return FAILED;
  }
  bool mix_is_aiv = false;
  std::string final_core_type = *core_type;
  (void)ge::AttrUtils::GetBool(op_desc, kMixIsAiv, mix_is_aiv);
  if (final_core_type == kCoreTypeMixEnhance && mix_is_aiv) {
    final_core_type = kCoreTypeMixAIV;
    FE_LOGI("Op[%s,%s] process aicore mix_aiv node", node->GetNamePtr(), node->GetTypePtr());
  }
  platform_infos.SetCoreNumByCoreType(final_core_type);
  if (GetRunInfoFromGe(node, platform_infos, tiling_info) != SUCCESS) {
    return FAILED;
  }

  if (IsStcToDynSoftSyncOp(op_desc)) {
    FE_LOGD("Node:%s is soft sync node.", node->GetName().c_str());
    if (TilingForSoftSyncOp(node, platform_infos, tiling_info) != SUCCESS) {
      FE_LOGE("Soft sync node[%s] tiling failed.", node->GetName().c_str());
      return FAILED;
    }
  }
  if (UpdateMixByTilingKey(op_desc, tiling_info) != SUCCESS) {
    FE_LOGE("Node[%s] update mix attr by tilingKey failed.", node->GetName().c_str());
    return FAILED;
  }

  SetTilingResult(op_desc, tiling_info);
  FE_LOGD("Success do Tiling for node:%s, tiling_data size:%zu, blockdim:%u.", node->GetName().c_str(),
          tiling_info->GetAllTilingData().str().size(), tiling_info->GetBlockDim());
  return SUCCESS;
}

Status UpdateTileFwkKernelInfo(const ge::OpDescPtr &op_desc) {
  RunInfoPtr tiling_info = nullptr;
  tiling_info = op_desc->TryGetExtAttr(ge::ATTR_NAME_OP_RUN_INFO, tiling_info);
  FE_CHECK_NOTNULL(tiling_info);
  bool tile_fwk_op_flag = false;
  (void)ge::AttrUtils::GetBool(op_desc, kAttrTileFwkOpStr, tile_fwk_op_flag);
  if (!tile_fwk_op_flag) {
    return SUCCESS;
  }
  uint64_t config_key = tiling_info->GetTilingKey();
  FE_LOGD("Node[%s, %s]: config_key is %lu.", op_desc->GetNamePtr(), op_desc->GetTypePtr(), config_key);

  SubkernelInfo subkernel_info;
  if (TileFwkOpInfo::Instance().GetFatbinInfo(op_desc->GetType(), config_key, subkernel_info) != SUCCESS) {
    FE_LOGE("Node[%s, %s]: failed to get subkernel info by config key %lu.",
            op_desc->GetNamePtr(), op_desc->GetTypePtr(), config_key);
    return FAILED;
  }
  FE_CHECK_NOTNULL(subkernel_info.subkernel_ptr);

  std::vector<int64_t> workspaces = tiling_info->GetAllWorkspaces();
  if (!workspaces.empty()) {
    subkernel_info.workspace_size += workspaces[0];
  } else {
    FE_LOGD("Node[%s, %s]: workspace list is empty.", op_desc->GetNamePtr(), op_desc->GetTypePtr());
  }
  FE_LOGD("Node[%s, %s]: block dim is %ld, workspace size is %ld.", op_desc->GetNamePtr(), op_desc->GetTypePtr(),
          subkernel_info.block_dim, subkernel_info.workspace_size);

  const uint8_t* subkernel_bin = subkernel_info.subkernel_ptr->GetBinData();
  KernelHeader kernel_header;
  if (memcpy_s(&kernel_header, sizeof(KernelHeader), subkernel_bin, sizeof(KernelHeader)) != EOK) {
    FE_LOGE("Node[%s, %s]: failed to get subkernel header.", op_desc->GetNamePtr(), op_desc->GetTypePtr());
    return FAILED;
  }
  std::vector<char> op_binary_bin(
      subkernel_bin + kernel_header.dataOffset[static_cast<size_t>(KernelContextType::OpBinary)],
      subkernel_bin + kernel_header.dataOffset[static_cast<size_t>(KernelContextType::OpBinary)] +
      kernel_header.dataSize[static_cast<size_t>(KernelContextType::OpBinary)]);
  std::vector<char> kernel_bin(
      subkernel_bin + kernel_header.dataOffset[static_cast<size_t>(KernelContextType::Kernel)],
      subkernel_bin + kernel_header.dataOffset[static_cast<size_t>(KernelContextType::Kernel)] +
      kernel_header.dataSize[static_cast<size_t>(KernelContextType::Kernel)]);
  ge::Buffer op_binary_buffer =
      ge::Buffer::CopyFrom(reinterpret_cast<uint8_t*>(op_binary_bin.data()), op_binary_bin.size());
  (void)ge::AttrUtils::SetBytes(op_desc, kAttrSubkernelOpBinaryStr, op_binary_buffer);
  std::string attr_prefix;
  (void)ge::AttrUtils::GetStr(op_desc, kAttrPrefixStr, attr_prefix);
  std::string kernel_name;
  (void)ge::AttrUtils::GetStr(op_desc, attr_prefix + kKernelName, kernel_name);
  ge::OpKernelBinPtr kernel_bin_ptr = nullptr;
  FE_MAKE_SHARED(kernel_bin_ptr =
      std::make_shared<ge::OpKernelBin>(kernel_name, std::move(kernel_bin)), return FAILED);
  (void)op_desc->SetExtAttr(attr_prefix + ge::OP_EXTATTR_NAME_TBE_KERNEL, kernel_bin_ptr);

  int32_t block_dim = 0;
  (void)ge::AttrUtils::GetInt(op_desc, ge::TVM_ATTR_NAME_BLOCKDIM, block_dim);
  if (subkernel_info.block_dim < block_dim) {
    FE_LOGD("Node[%s, %s]: block dim changes from %ld to %ld.", op_desc->GetNamePtr(), op_desc->GetTypePtr(),
            block_dim, subkernel_info.block_dim);
    (void)ge::AttrUtils::SetInt(op_desc, ge::TVM_ATTR_NAME_BLOCKDIM, subkernel_info.block_dim);
  }
  op_desc->SetWorkspaceBytes({subkernel_info.workspace_size});
  FE_LOGD("Node[%s, %s]: op binary size is %zu, kernel size is %ld.", op_desc->GetNamePtr(),
          op_desc->GetTypePtr(), op_binary_bin.size(), kernel_bin_ptr->GetBinDataSize());
  return SUCCESS;
}
}  // namespace fe

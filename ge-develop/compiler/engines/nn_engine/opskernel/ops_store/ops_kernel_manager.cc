/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ops_store/ops_kernel_manager.h"
#include "common/fe_log.h"
#include "common/fe_error_code.h"
#include "common/configuration.h"
#include "common/fe_type_utils.h"
#include "common/aicore_util_constants.h"
#include "common/aicore_util_attr_define.h"
#include "ops_store/ops_kernel_utils.h"
#include "ops_store/ops_kernel_error_codes.h"

namespace fe {
OpsKernelManager::OpsKernelManager(const std::string &engine_name)
    : is_init_(false), engine_name_(engine_name) {}
OpsKernelManager::~OpsKernelManager() {}

OpsKernelManager &OpsKernelManager::Instance(const std::string &engine_name) {
  static std::map<std::string, OpsKernelManager &> ops_kernel_map;
  if (ops_kernel_map.empty()) {
    static OpsKernelManager ai_ops_kernel_manager(AI_CORE_NAME);
    static OpsKernelManager vec_ops_kernel_manager(VECTOR_CORE_NAME);
    static OpsKernelManager dsa_ops_kernel_manager(kDsaCoreName);
    ops_kernel_map.insert({AI_CORE_NAME, ai_ops_kernel_manager});
    ops_kernel_map.insert({VECTOR_CORE_NAME, vec_ops_kernel_manager});
    ops_kernel_map.insert({kDsaCoreName, dsa_ops_kernel_manager});
  }
  std::map<std::string, OpsKernelManager &>::const_iterator iter = ops_kernel_map.find(engine_name);
  if (iter != ops_kernel_map.end()) {
    return iter->second;
  }
  FE_LOGD("Engine name [%s] is not found, using the default instance.", engine_name.c_str());
  /* If engine_name is invalid, we just return the first element of map
   * config_map */
  return ops_kernel_map.begin()->second;
}

Status OpsKernelManager::Initialize() {
  if (is_init_) {
    return SUCCESS;
  }

  const std::vector<FEOpsStoreInfo> &ops_store_info_vec = Configuration::Instance(engine_name_).GetOpsStoreInfo();
  for (const FEOpsStoreInfo &ops_sub_store_info : ops_store_info_vec) {
    SubOpInfoStorePtr sub_ops_kernel_ptr = nullptr;
    FE_MAKE_SHARED(sub_ops_kernel_ptr = std::make_shared<SubOpInfoStore>(ops_sub_store_info),
                   return OP_STORE_MAKE_SHARED_FAILED);
    FE_CHECK_NOTNULL(sub_ops_kernel_ptr);
    /* If InitializeSubStore fails, it is not necessary to finalize its
        memory because all memory allocated dynamically is used shared pointer
        and STL collections. They have auto-destruct mechanics */
    Status result = sub_ops_kernel_ptr->Initialize(engine_name_);
    if (result == SUCCESS) {
      sub_ops_kernel_map_.emplace(std::make_pair(ops_sub_store_info.fe_ops_store_name, sub_ops_kernel_ptr));
      sub_ops_store_map_.emplace(std::make_pair(ops_sub_store_info.op_impl_type, sub_ops_kernel_ptr));
    }
  }

  if (sub_ops_kernel_map_.empty()) {
    std::string init_fail_log = "FEOpsKernelInfoStore: Initialize custom and builtin sub-information library failed, "
                                "please check the related warning or error log above for detailed information!";
    REPORT_FE_ERROR("[GraphOpt][Init] %s", init_fail_log.c_str());
    return FAILED;
  }

  is_init_ = true;
  return SUCCESS;
}

Status OpsKernelManager::Finalize() {
  if (!is_init_) {
    return SUCCESS;
  }

  sub_ops_store_map_.clear();
  for (auto &elem : sub_ops_kernel_map_) {
    if (elem.second == nullptr) {
      FE_LOGW("FEOpsKernelInfoStore::Finalize: pointer in map_all_sub_store_info_ %s should not be nullptr!",
              elem.first.c_str());
      continue;
    }
    elem.second->Finalize();
  }
  sub_ops_kernel_map_.clear();
  is_init_ = false;
  return SUCCESS;
}

void OpsKernelManager::GetAllOpsKernelInfo(map<string, ge::OpInfo> &infos) const {
  for (auto &sub_kernel_itr : sub_ops_kernel_map_) {
    if (sub_kernel_itr.second == nullptr) {
      FE_LOGW("GetAllOpsKernelInfo: pointer in map_all_sub_store_info_ [%s] should not be nullptr!",
              sub_kernel_itr.first.c_str());
      continue;
    }
    const std::map<std::string, OpKernelInfoPtr> &sub_ops_kernel_info_map = sub_kernel_itr.second->GetAllOpKernels();
    for (auto &op : sub_ops_kernel_info_map) {
      if (op.second == nullptr) {
        FE_LOGW("GetAllOpsKernelInfo: pointer in map_sub_store_op_kernel_infos_ [%s] should not be a nullptr!",
                op.first.c_str());
        continue;
      }
      ge::OpInfo op_info = (op.second)->GetOpInfo();
      op_info.opKernelLib = sub_kernel_itr.first;
      /* At most one OpInfo for One OpType */
      if (infos.count(op.first) == 0) {
        infos.emplace(std::make_pair(op.first, op_info));
      }
    }
  }
}

SubOpInfoStorePtr OpsKernelManager::GetSubOpsKernelByStoreName(const std::string &store_name) {
  std::map<std::string, SubOpInfoStorePtr>::const_iterator iter = sub_ops_kernel_map_.find(store_name);
  if (iter == sub_ops_kernel_map_.end()) {
    return nullptr;
  }
  return iter->second;
}

SubOpInfoStorePtr OpsKernelManager::GetSubOpsKernelByImplType(const OpImplType &op_impl_type) {
  std::map<OpImplType, SubOpInfoStorePtr>::const_iterator iter = sub_ops_store_map_.find(op_impl_type);
  if (iter == sub_ops_store_map_.end()) {
    return nullptr;
  }
  return iter->second;
}

OpKernelInfoPtr OpsKernelManager::GetOpKernelInfoByOpType(const std::string &store_name, const std::string &op_type) {
  std::map<std::string, SubOpInfoStorePtr>::const_iterator iter_store = sub_ops_kernel_map_.find(store_name);
  if (iter_store == sub_ops_kernel_map_.end()) {
    return nullptr;
  }
  return iter_store->second->GetOpKernelByOpType(op_type);
}

OpKernelInfoPtr OpsKernelManager::GetOpKernelInfoByOpType(const OpImplType &op_impl_type, const std::string &op_type) {
  std::map<OpImplType, SubOpInfoStorePtr>::const_iterator iter_store = sub_ops_store_map_.find(op_impl_type);
  if (iter_store == sub_ops_store_map_.end()) {
    return nullptr;
  }
  return iter_store->second->GetOpKernelByOpType(op_type);
}

OpKernelInfoPtr OpsKernelManager::GetOpKernelInfoByOpDesc(const ge::OpDescPtr &op_desc_ptr) {
  if (op_desc_ptr == nullptr) {
    return nullptr;
  }
  std::string op_name = op_desc_ptr->GetName();
  std::string op_type = op_desc_ptr->GetType();

  // 1. get fe_imply_type
  int64_t fe_imply_type = -1;
  if (!ge::AttrUtils::GetInt(op_desc_ptr, FE_IMPLY_TYPE, fe_imply_type)) {
    FE_LOGD("Op[name=%s, type=%s]: attribute %s not found.", op_name.c_str(), op_type.c_str(), FE_IMPLY_TYPE.c_str());
    return nullptr;
  }

  // 2. get the opsKernelInfo by OpImplType
  OpImplType op_impl_type = static_cast<OpImplType>(fe_imply_type);

  return this->GetOpKernelInfoByOpType(op_impl_type, op_type);
}

OpKernelInfoPtr OpsKernelManager::GetHighPrioOpKernelInfo(const std::string &op_type) {
  std::lock_guard<std::mutex> lock_guard(ops_kernel_manager_lock_);

  const std::vector<FEOpsStoreInfo> &ops_store_info_vec = Configuration::Instance(engine_name_).GetOpsStoreInfo();
  for (const FEOpsStoreInfo &ops_store : ops_store_info_vec) {
    OpKernelInfoPtr op_kernel_ptr = GetOpKernelInfoByOpType(ops_store.op_impl_type, op_type);
    if (op_kernel_ptr == nullptr) {
      continue;
    }

    return op_kernel_ptr;
  }
  FE_LOGD("Cannot find operation type [%s] in any of the operational sub-stores.", op_type.c_str());
  return nullptr;
}

Status OpsKernelManager::AddSubOpsKernel(SubOpInfoStorePtr sub_op_info_store_ptr) {
  std::lock_guard<std::mutex> lock_guard(ops_kernel_manager_lock_);

  FE_CHECK_NOTNULL(sub_op_info_store_ptr);
  std::string store_name = sub_op_info_store_ptr->GetOpsStoreName();
  if (store_name.empty()) {
    REPORT_FE_ERROR("[GraphOpt][SetDyncCustomOpStoreInfo][AddSubOpsKernel] The library name of the operation information is empty.");
    return FAILED;
  }
  std::map<std::string, SubOpInfoStorePtr>::const_iterator iter = sub_ops_kernel_map_.find(store_name);
  if (iter != sub_ops_kernel_map_.end()) {
    REPORT_FE_ERROR("[GraphOpt][SetDyncCustomOpStoreInfo][AddSubOpsKernel] Sub op info store [%s] already exists.",
                    store_name.c_str());
    return FAILED;
  }

  OpImplType op_impl_type = sub_op_info_store_ptr->GetOpImplType();
  std::map<OpImplType, SubOpInfoStorePtr>::const_iterator iter_store = sub_ops_store_map_.find(op_impl_type);
  if (iter_store != sub_ops_store_map_.end()) {
    REPORT_FE_ERROR("[GraphOpt][SetDyncCustomOpStoreInfo][AddSubOpsKernel] Sub op info store [%s] already exists.",
                    store_name.c_str());
    return FAILED;
  }

  sub_ops_kernel_map_.emplace(std::make_pair(store_name, sub_op_info_store_ptr));
  sub_ops_store_map_.emplace(std::make_pair(op_impl_type, sub_op_info_store_ptr));
  FE_LOGD("Sub-ops information library [%s] has been added to ops kernel manager.", store_name.c_str());
  return SUCCESS;
}

bool OpsKernelManager::FindOpKernelByType(const std::string &op_type, OpKernelInfoPtr &op_kernel_ptr) {
  const std::vector<FEOpsStoreInfo> &fe_ops_store_info_vec = Configuration::Instance(engine_name_).GetOpsStoreInfo();
  for (const auto &ops_store : fe_ops_store_info_vec) {
    op_kernel_ptr = GetOpKernelInfoByOpType(ops_store.fe_ops_store_name, op_type);
    if (op_kernel_ptr != nullptr) {
      FE_LOGD("[FindOpKernelByType] GetOpKernel successfully, op type[%s], op store name[%s].",
              op_type.c_str(), ops_store.fe_ops_store_name.c_str());
      return true;
    }
  }
  return false;
}

void OpsKernelManager::UpdatePatternForAllKernel(
      const std::vector<std::pair<std::string, std::string>> &op_prebuild_patterns) {
  std::lock_guard<std::mutex> lock_guard(ops_kernel_manager_lock_);
  for (const auto &op_pattern_pair : op_prebuild_patterns) {
    OpKernelInfoPtr op_kernel_ptr;
    if (FindOpKernelByType(op_pattern_pair.first, op_kernel_ptr)) {
      op_kernel_ptr->UpdatePrebuildPattern(op_pattern_pair.second);
    }
  }
}

OpKernelInfoPtr OpsKernelManager::GetHighPrioOpKernelInfoPtr(const string &op_type) {
  const std::vector<FEOpsStoreInfo> &fe_ops_store_info_vec = Configuration::Instance(engine_name_).GetOpsStoreInfo();
  for (auto &ops_store : fe_ops_store_info_vec) {
    OpKernelInfoPtr op_kernel_ptr = GetOpKernelInfoByOpType(ops_store.fe_ops_store_name, op_type);
    if (op_kernel_ptr == nullptr) {
      continue;
    }
    return op_kernel_ptr;
  }
  FE_LOGW("Can not find Op (type [%s]) in all ops information store libs.", op_type.c_str());
  return nullptr;
}
}

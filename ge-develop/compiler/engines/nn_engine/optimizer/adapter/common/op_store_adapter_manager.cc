/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "adapter/common/op_store_adapter_manager.h"
#include <utility>
#include <vector>
#include "common/fe_type_utils.h"
#include "common/configuration.h"
#include "common/fe_inner_error_codes.h"
#include "graph/ge_tensor.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "adapter/dsa_adapter/dsa_op_store_adapter.h"
#include "trace_handle_manager/trace_handle_manager.h"

namespace fe {
static const std::string kStrTbeOpAdapter = "tbe_op_adapter";
static const std::string kStrDsaOpAdapter = "dsa_op_adapter";
static const std::string kStrReserve = "reserve";
static const std::unordered_map<OpImplType, std::string> ADAPTER_TYPE_MAP {
        {EN_IMPL_CUSTOM_TIK, kStrReserve},
        {EN_IMPL_CUSTOM_TBE, kStrTbeOpAdapter},
        {EN_IMPL_HW_TIK, kStrReserve},
        {EN_IMPL_HW_TBE, kStrTbeOpAdapter},
        {EN_IMPL_RL, kStrReserve},
        {EN_IMPL_VECTOR_CORE_HW_TBE, kStrTbeOpAdapter},
        {EN_IMPL_VECTOR_CORE_CUSTOM_TBE, kStrTbeOpAdapter},
        {EN_IMPL_NON_PERSISTENT_CUSTOM_TBE, kStrTbeOpAdapter},
        {EN_IMPL_HW_DSA, kStrDsaOpAdapter}};

OpStoreAdapterManager &OpStoreAdapterManager::Instance(const std::string &engine_name) {
  static std::map<std::string, OpStoreAdapterManager &> ops_store_adapter_map;
  if (ops_store_adapter_map.empty()) {
    static OpStoreAdapterManager ai_ops_adapter_manager(AI_CORE_NAME);
    static OpStoreAdapterManager vec_ops_adapter_manager(VECTOR_CORE_NAME);
    static OpStoreAdapterManager dsa_ops_adapter_manager(kDsaCoreName);
    ops_store_adapter_map.insert({AI_CORE_NAME, ai_ops_adapter_manager});
    ops_store_adapter_map.insert({VECTOR_CORE_NAME, vec_ops_adapter_manager});
    ops_store_adapter_map.insert({kDsaCoreName, dsa_ops_adapter_manager});
  }
  std::map<std::string, OpStoreAdapterManager &>::const_iterator iter = ops_store_adapter_map.find(engine_name);
  if (iter != ops_store_adapter_map.end()) {
    return iter->second;
  }
  FE_LOGD("Engine name [%s] is not found, using the default instance.", engine_name.c_str());
  return ops_store_adapter_map.begin()->second;
}

OpStoreAdapterManager::OpStoreAdapterManager(const std::string &engine_name) : init_flag_(false),
                                                                               engine_name_(engine_name) {}

OpStoreAdapterManager::~OpStoreAdapterManager() {}

Status OpStoreAdapterManager::InitializeAdapter(const std::string &adapter_type,
                                                const std::map<std::string, std::string> &options) {
  Status result = SUCCESS;
  FE_LOGD("The InitializeAdapter is adapter[%s].", adapter_type.c_str());
  std::unordered_map<std::string, OpStoreAdapterPtr>::const_iterator adapter_ptr_iter =
          map_all_op_store_adapter_.find(adapter_type);
  if (adapter_ptr_iter != map_all_op_store_adapter_.end()) {
    FE_LOGD("The [%s] has already been initialized.", adapter_type.c_str());
    return SUCCESS;
  }
  if (adapter_type == kStrTbeOpAdapter) {
    TbeOpStoreAdapterPtr tbe_adapter_ptr = nullptr;
    FE_MAKE_SHARED(tbe_adapter_ptr = std::make_shared<TbeOpStoreAdapter>(engine_name_),
                   return OP_STORE_ADAPTER_MAKE_SHARED_FAILED);
    FE_CHECK(tbe_adapter_ptr == nullptr, FE_LOGE("tbeAdapterPtr is nullptr."), return PARAM_INVALID);
    result = tbe_adapter_ptr->Initialize(options);
    if (result == SUCCESS) {
      map_all_op_store_adapter_.emplace(std::make_pair(kStrTbeOpAdapter, tbe_adapter_ptr));
    } else {
      REPORT_FE_ERROR("[SubGraphOpt][PreCompileOp][InitAdapter] InitializeAdapter adapter [%s] failed! Ret [%u]",
                      kStrTbeOpAdapter.c_str(), result);
      return result;
    }
  } else if (adapter_type == kStrDsaOpAdapter) {
    DSAOpStoreAdapterPtr dsa_adapter_ptr = nullptr;
    FE_MAKE_SHARED(dsa_adapter_ptr = std::make_shared<DSAOpStoreAdapter>(), return OP_STORE_ADAPTER_MAKE_SHARED_FAILED);
    FE_CHECK(dsa_adapter_ptr == nullptr, FE_LOGE("dsaAdapterPtr is nullptr"), return PARAM_INVALID);
    result = dsa_adapter_ptr->Initialize(options);
    if (result == SUCCESS) {
      map_all_op_store_adapter_.emplace(std::make_pair(kStrDsaOpAdapter, dsa_adapter_ptr));
    } else {
      REPORT_FE_ERROR("[SubGraphOpt][PreCompileOp][InitAdapter] InitializeAdapter adapter [%s] failed! Ret [%u]",
                      kStrDsaOpAdapter.c_str(), result);
      return result;
    }
  }
  return result;
}

Status OpStoreAdapterManager::Initialize(const std::map<std::string, std::string> &options) {
  if (init_flag_) {
    FE_LOGD("The OpStoreAdapterManager has already been initialized.");
    return SUCCESS;
  }
  /* Before OpStoreAdapterManager is initialized, Configuration class has
    already loaded ops store info vector */
  init_flag_ = true;
  const std::vector<FEOpsStoreInfo> &fe_ops_store_info_vec = Configuration::Instance(engine_name_).GetOpsStoreInfo();

  for (auto &ops_sub_store_info : fe_ops_store_info_vec) {
    OpImplType impl_type = GetMainImplType<OpImplType>(ops_sub_store_info.op_impl_type);
    auto adapter_str_iter = ADAPTER_TYPE_MAP.find(impl_type);
    if (adapter_str_iter == ADAPTER_TYPE_MAP.end()) {
      REPORT_FE_ERROR("[SubGraphOpt][PreCompileOp][Init] Adapter type is not found, OpsStoreName[%s].",
                      ops_sub_store_info.fe_ops_store_name.c_str());
      return OP_ADAPTER_CHECK_FAILED;
    }
    std::string adapter_type_str = adapter_str_iter->second;
    Status result = InitializeAdapter(adapter_type_str, options);
    if (result != SUCCESS) {
      REPORT_FE_ERROR("[SubGraphOpt][PreCompileOp][Init] Initialize op store adapter failed, OpsStoreName[%s].",
                      ops_sub_store_info.fe_ops_store_name.c_str());
      return result;
    }
  }

  return SUCCESS;
}

Status OpStoreAdapterManager::Finalize() {
  FE_LOGD("Finalizing the OpStoreAdapterManager");
  if (!init_flag_) {
    FE_LOGD(
        "OpStoreAdapterManager has not been initialized, Finalize is not "
        "allowed.");
    return SUCCESS;
  }

  for (auto &elem : map_all_op_store_adapter_) {
    if (elem.second == nullptr) {
      FE_LOGW(
          "OpStoreAdapterManager::Finalize: pointer in "
          "mapAllOpStoreAdapter_ [%s] should not be nullptr!",
          elem.first.c_str());
      continue;
    }
    // submit statistics trace info for each adapter
    std::vector<std::string> statistics_msgs;
    elem.second->GetAllCompileStatisticsMsg(statistics_msgs);
    for (const std::string &msg : statistics_msgs) {
      TraceHandleManager::Instance().SubmitStatisticsTrace(msg);
    }
    elem.second->Finalize();
  }
  map_all_op_store_adapter_.clear();
  init_flag_ = false;
  FE_LOGI("OpStoreAdapterManager finalize successfully.");
  return SUCCESS;
}

Status OpStoreAdapterManager::GetOpStoreAdapter(const OpImplType &op_impl_type,
                                                OpStoreAdapterPtr &adapter_ptr) const {
  OpImplType impl_type = GetMainImplType<OpImplType>(op_impl_type);
  auto adapter_str_iter = ADAPTER_TYPE_MAP.find(impl_type);
  if (adapter_str_iter == ADAPTER_TYPE_MAP.end()) {
    REPORT_FE_ERROR("[SubGraphOpt][PreCompileOp][GetOpStoreAdapter] Impl type %s is not found in op store adapter.",
                    GetImplTypeString(impl_type).c_str());
    return OP_ADAPTER_TYPE_CHECK_FAILED;
  }

  auto adapter_ptr_iter = map_all_op_store_adapter_.find(adapter_str_iter->second);
  if (adapter_ptr_iter == map_all_op_store_adapter_.end()) {
    REPORT_FE_ERROR("[SubGraphOpt][PreCompileOp][GetOpStoreAdapter] op store adapter is not found, adapter name [%s].",
                    adapter_str_iter->second.c_str());
    return OP_ADAPTER_CHECK_FAILED;
  }

  adapter_ptr = adapter_ptr_iter->second;
  return SUCCESS;
}

OpStoreAdapterPtr OpStoreAdapterManager::GetOpStoreAdapter(const OpImplType &op_impl_type) const {
  OpImplType impl_type = GetMainImplType<OpImplType>(op_impl_type);
  auto adapter_str_iter = ADAPTER_TYPE_MAP.find(impl_type);
  if (adapter_str_iter == ADAPTER_TYPE_MAP.end()) {
    REPORT_FE_ERROR("[SubGraphOpt][PreCompileOp][GetOpStoreAdapter] Impl type %s is not found in op store adapter.",
                    GetImplTypeString(impl_type).c_str());
    return nullptr;
  }

  auto adapter_ptr_iter = map_all_op_store_adapter_.find(adapter_str_iter->second);
  if (adapter_ptr_iter == map_all_op_store_adapter_.end()) {
      FE_LOGW("[SubGraphOpt][PreCompileOp][GetOpStoreAdapter] op store adapter is not found, adapter name [%s].",
              adapter_str_iter->second.c_str());
    return nullptr;
  }

  return adapter_ptr_iter->second;
}
}  // namespace fe

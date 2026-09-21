/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fusion_manager/fusion_manager.h"

#include <string>
#include <utility>
#include "common/configuration.h"
#include "common/fe_error_code.h"
#include "common/fe_log.h"
#include "common/fe_utils.h"
#include "common/string_utils.h"
#include "common/util/op_info_util.h"
#include "common/sgt_slice_type.h"
#include "common/cmo_id_gen_strategy.h"
#include "common/scope_allocator.h"
#include "ge/ge_api_types.h"
#include "ops_store/binary_kernel_info.h"
#include "adapter/common/op_store_adapter_manager.h"
#include "register/graph_optimizer/fusion_common/fusion_config_info.h"

using std::map;
using std::string;
std::mutex kFmLock;

namespace fe {
static const string FE_OPTIMIZER_NAME = "FEOptimizer";
static const string SINGLE_OP_FLAG_DEFAULT = "0";

FusionManager::FusionManager()
    : ops_kernel_info_store_(nullptr), graph_opt_(nullptr), inited_(false) {}

FusionManager::~FusionManager() {}

FusionManager &FusionManager::Instance(const std::string &engine_name) {
  static FusionManager ai_core_fm;
  static FusionManager vector_core_fm;
  static FusionManager dsa_core_fm;

  if (engine_name == VECTOR_CORE_NAME) {
    return vector_core_fm;
  } if (engine_name == kDsaCoreName) {
    return dsa_core_fm;
  } else {
    return ai_core_fm;
  }
}

Status FusionManager::InitBinKernelInfoStore(const std::string &engine_name) const {
  const std::string &bin_cfg_file = Configuration::Instance(engine_name).GetBinaryConfigFilePath();
  if (bin_cfg_file.empty()) {
    FE_LOGI("[FusionMngr][InitBinKernelInfoStore] Binary config file path is empty.");
    return SUCCESS;
  }
  BinaryKernelInfo &bin_kernel_info = BinaryKernelInfo::Instance();
  FE_LOGD("[FusionMngr][InitBinKernelInfoStore]Start to init binary kernel info by path %s.", bin_cfg_file.c_str());
  return bin_kernel_info.Initialize(bin_cfg_file);
}

void FusionManager::UpdateOpsKernelInfo(const std::string &engine_name, const FEOpsKernelInfoStorePtr ops_kernel_info_store) {
  FE_TIMECOST_START(UpdateOpsKernelInfo);
  if (engine_name != AI_CORE_NAME) {
    FE_LOGI("[Init][UpdateOpsKernelInfo] No need to update ops kernel info for engine[%s].", engine_name.c_str());
    FE_TIMECOST_END(UpdateOpsKernelInfo, "FusionManager::UpdateOpsKernelInfo");
    return;
  }
  OpStoreAdapterPtr op_store_adapter = nullptr;
  (void)OpStoreAdapterManager::Instance(engine_name).GetOpStoreAdapter(EN_IMPL_HW_TBE, op_store_adapter);
  if (op_store_adapter == nullptr || op_store_adapter->UpdatePrebuildPattern() != SUCCESS) {
    FE_LOGW("[Init][UpdateOpsKernelInfo] Cannot update prebuild pattern from tbe.");
    return;
  }
  op_store_adapter->SetOpsKernelInfoStore(ops_kernel_info_store);
  FE_LOGI("[Init][UpdateOpsKernelInfo] Succeed to update ops kernel info for engine[%s].", engine_name.c_str());
  FE_TIMECOST_END(UpdateOpsKernelInfo, "FusionManager::UpdateOpsKernelInfo");
}

/*
 * to initialize the subparts of fusion manager
 * param[in] the options of init
 * return Status(SUCCESS/FAILED)
 */
Status FusionManager::Initialize(const std::string &engine_name, const map<string, string> &options) {
  // add func lock
  std::lock_guard<std::mutex> lock_guard(kFmLock);

  FE_LOGD("Initialize start, engine_name:[%s].", engine_name.c_str());
  FE_TIMECOST_START(FusionMgrInit);
  if (inited_) {
    FE_LOGW("FusionManager has been initialized; return directly.");
    return SUCCESS;
  }

  inited_ = true;
  (void)FusionConfigInfo::Instance().Initialize();
  if (Configuration::Instance(engine_name).Initialize(options) != SUCCESS) {
    REPORT_FE_ERROR("[FusionMngr][Init] Failed to initialize configuration.");
    return CONFIGURATION_INIT_FAILED;
  }

  if (InitBinKernelInfoStore(engine_name) != SUCCESS) {
    REPORT_FE_ERROR("[FusionMngr][Init] Failed to initialize binary kernel info.");
    return FAILED;
  }

  int64_t scope_id = ScopeAllocator::Instance().GetCurrentScopeId();
  FE_LOGD("Current scope ID is [%ld].", scope_id);

  Status ret = OpStoreAdapterManager::Instance(engine_name).Initialize(options);
  if (ret != fe::SUCCESS) {
    REPORT_FE_ERROR("[FusionMngr][Init] Op store adapter manager init failed.");
    return OP_STORE_ADAPTER_MANAGER_INIT_FAILED;
  }

  FE_MAKE_SHARED(ops_kernel_info_store_ = make_shared<FEOpsKernelInfoStore>(engine_name),
                 return FAILED);
  ret = ops_kernel_info_store_->Initialize(options);
  if (ret != fe::SUCCESS) {
    REPORT_FE_ERROR("[FusionMngr][Init] OpInfoKernelStores init failed! Ret [%u]", ret);
    return OPINFO_STORES_INIT_FAILED;
  }
  if (engine_name == kDsaCoreName) {
    FE_MAKE_SHARED(
        dsa_graph_opt_ = make_shared<DSAGraphOptimizer>(ops_kernel_info_store_, engine_name),
        return FAILED);
    ret = dsa_graph_opt_->Initialize(options, nullptr);
    if (ret != fe::SUCCESS) {
      REPORT_FE_ERROR("[FusionMngr][Init] DSAGraphOptimizer init failed.");
      return GRAPH_OPTIMIZER_INIT_FAILED;
    }
  } else {
    FE_MAKE_SHARED(
        graph_opt_ = make_shared<FEGraphOptimizer>(ops_kernel_info_store_, engine_name),
        return FAILED);
  }
  UpdateOpsKernelInfo(engine_name, ops_kernel_info_store_);

  FE_LOGI("Initialize successfully.");
  FE_TIMECOST_END(FusionMgrInit, "FusionManager::Initialize");
  return SUCCESS;
}

/*
 * to release the source of fusion manager
 * return Status(SUCCESS/FAILED)
 */
Status FusionManager::Finalize() {
  // add func lock
  std::lock_guard<std::mutex> lock_guard(kFmLock);

  if (inited_) {
    FE_TIMECOST_START(FusionMgrFinal);

    FE_LOGD("Finalize begin.");
    (void)FusionConfigInfo::Instance().Finalize();
    Status ret1 = SUCCESS;
    if (graph_opt_ != nullptr) {
      ret1 = graph_opt_->Finalize();
      FE_LOGE_IF(ret1 != SUCCESS, "Finalize GraphOptimizer failed.");
    }

    Status ret2 = SUCCESS;
    std::string eng_type;
    if (ops_kernel_info_store_ != nullptr) {
      eng_type = ops_kernel_info_store_->GetFEOpsKernelInfoStoreName();
      ret2 = ops_kernel_info_store_->Finalize();
      FE_LOGE_IF(ret2 != SUCCESS, "Finalize Ops Kernel Info Store failed.");
    }

    Status ret3 = OpStoreAdapterManager::Instance(eng_type).Finalize();
    FE_LOGE_IF(ret3 != SUCCESS, "Finalize Ops Store Adapter Manager failed.");

    Configuration &config_inst = Configuration::Instance(eng_type);
    Status ret4 = config_inst.Finalize();
    FE_LOGE_IF(ret4 != SUCCESS, "Finalize configuration failed.");

    BinaryKernelInfo &bin_kernel = BinaryKernelInfo::Instance();
    bin_kernel.Finalize();

    Status ret5 = CMOIdGenStrategy::Instance().Finalize();
    FE_LOGE_IF(ret5 != SUCCESS, "Finalize CMOIdGenStrategy failed.");

    bool ret_status = ((ret1 != SUCCESS) || (ret2 != SUCCESS) || (ret3 != SUCCESS) || (ret4 != SUCCESS) ||
                       (ret5 != SUCCESS));
    if (ret_status) {
      FE_LOGW("FusionManager finalization did not succeed.");
      return FAILED;
    }

    inited_ = false;
    FE_LOGD("Finalize successfully.");
    FE_TIMECOST_END(FusionMgrFinal, "FusionManager::Finalize");
    return SUCCESS;
  }

  FE_LOGW("Already Finalized, directly return.");
  return SUCCESS;
}

/*
 * to get the information of OpsKernel InfoStores
 * param[out] the map of OpsKernel InfoStores
 */
void FusionManager::GetOpsKernelInfoStores(map<string, OpsKernelInfoStorePtr> &op_kern_infos,
                                           const std::string &engine_name) {
  FE_LOGD("Get OpsKernelInfoStores start.");
  if (ops_kernel_info_store_ != nullptr) {
    op_kern_infos.emplace(
        std::make_pair(ops_kernel_info_store_->GetFEOpsKernelInfoStoreName(), ops_kernel_info_store_));
  } else {
    FE_LOGW("opsKernelInfoStore_ of engine named %s is nullptr! inited_ = %d", engine_name.c_str(), inited_);
  }

  FE_LOGD("Get OpsKernelInfoStores finished.");
}

/*
 * to get the information of Graph Optimizer
 * param[out] the map of Graph Optimizer
 */
void FusionManager::GetGraphOptimizerObjs(map<string, GraphOptimizerPtr> &graph_optimizers,
                                          const std::string &engine_name) {
  FE_LOGD("Start GetGraphOptimizer.");
  ge::GraphOptimizerAttribute attrs;
  if (engine_name != kDsaCoreName && graph_opt_ == nullptr) {
    FE_LOGD("Engine named %s is not initialized.", engine_name.c_str());
    return;
  }
  if (engine_name == kDsaCoreName && dsa_graph_opt_ == nullptr) {
    FE_LOGD("Engine named %s is not initialized.", engine_name.c_str());
    return;
  }
  if (engine_name == kDsaCoreName) {
    (void)dsa_graph_opt_->GetAttributes(attrs);
    graph_optimizers[attrs.engineName] = dsa_graph_opt_;
  } else {
    (void) graph_opt_->GetAttributes(attrs);
    graph_optimizers[attrs.engineName] = graph_opt_;
  }
  FE_LOGD("Get GraphOptimizer success.");
}

}  // namespace fe

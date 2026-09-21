/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "itf_handler/itf_handler.h"
#include <map>
#include <string>

#include "common/aicore_util_constants.h"
#include "platform/platform_info.h"
#include "common/fe_utils.h"
#include "common/fe_error_code.h"
#include "common/platform_utils.h"
#include "fusion_manager/fusion_manager.h"
#include "trace_handle_manager/trace_handle_manager.h"
#include "ge/ge_api_types.h"

/* List of all engine names */
static std::vector<std::string> g_engine_name_list;

/*
 * to invoke the Initialize function of FusionManager
 * param[in] the options of init
 * return Status(SUCCESS/FAILED)
 */
fe::Status Initialize(const std::map<string, string> &options) {
  g_engine_name_list = {fe::AI_CORE_NAME};

  if (fe::TraceHandleManager::Instance().Initialize() != fe::SUCCESS) {
    FE_LOGE("Failed to initialize trace handle manager.");
    return fe::FAILED;
  }
  if (fe::PlatformUtils::Instance().Initialize(options) != fe::SUCCESS) {
    FE_LOGE("Failed to instantiate platform utils.");
    return fe::FAILED;
  }

  // init VectorCore engine only at MDC/DC
  if (fe::PlatformUtils::Instance().IsSupportVectorEngine()) {
    g_engine_name_list.push_back(fe::VECTOR_CORE_NAME);
  }

  // init DSAEngine engine only when dsa workspace size is greater than 0
  if (fe::PlatformUtils::Instance().GetDsaWorkspaceSize() > 0) {
    g_engine_name_list.push_back(fe::kDsaCoreName);
  }

  for (const std::string &engine_name : g_engine_name_list) {
    FE_LOGD("Start initializing in engine [%s].", engine_name.c_str());
    fe::FusionManager &fm = fe::FusionManager::Instance(engine_name);
    fe::Status ret = fm.Initialize(engine_name, options);
    if (ret != fe::SUCCESS) {
      fm.Finalize();
      return ret;
    }
    FE_LOGD("Finish the initialization of engine[%s].", engine_name.c_str());
  }

  return fe::SUCCESS;
}

/*
 * to invoke the Finalize function to release the source of fusion manager
 * return Status(SUCCESS/FAILED)
 */
fe::Status Finalize() {
  if (fe::PlatformUtils::Instance().Finalize() != fe::SUCCESS) {
    return fe::FAILED;
  }
  bool is_any_engine_failed_to_finalize = false;
  for (auto &engine_name : g_engine_name_list) {
    FE_LOGD("Start finalizing in engine [%s]", engine_name.c_str());
    fe::Status ret = fe::FusionManager::Instance(engine_name).Finalize();
    if (ret != fe::SUCCESS) {
      /* If any of the engine fail to finalize, we will return FAILED.
       * Set this flag to true and keep finalizing the next engine. */
      is_any_engine_failed_to_finalize = true;
    }
  }
  fe::TraceHandleManager::Instance().Finalize();
  if (is_any_engine_failed_to_finalize) {
    return fe::FAILED;
  } else {
    return fe::SUCCESS;
  }
}

/*
 * to invoke the same-name function of FusionManager to get the information of OpsKernel InfoStores
 * param[out] the map of OpsKernel InfoStores
 */
void GetOpsKernelInfoStores(std::map<string, OpsKernelInfoStorePtr> &op_kern_infos) {
  for (auto &engine_name : g_engine_name_list) {
    fe::FusionManager &fm = fe::FusionManager::Instance(engine_name);
    fm.GetOpsKernelInfoStores(op_kern_infos, engine_name);
  }
}

/*
 * to invoke the same-name function of FusionManager to get the information of Graph Optimizer
 * param[out] the map of Graph Optimizer
 */
void GetGraphOptimizerObjs(std::map<string, GraphOptimizerPtr> &graph_optimizers) {
  for (auto &engine_name : g_engine_name_list) {
    fe::FusionManager &fm = fe::FusionManager::Instance(engine_name);
    fm.GetGraphOptimizerObjs(graph_optimizers, engine_name);
  }
}

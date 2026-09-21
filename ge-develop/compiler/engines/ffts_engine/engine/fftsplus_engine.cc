/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fftsplus_engine.h"
#include <map>
#include <string>
#include "inc/ffts_utils.h"
#include "inc/ffts_configuration.h"
#include "ge/ge_api_types.h"
#include "engine_manager.h"

/* List of all engine names */
static std::vector<std::string> g_engine_name_list;

ffts::Status Initialize(const map<string, string> &options) {
  g_engine_name_list = {ffts::kFFTSPlusEngineName};
  std::string soc_version = "";
  auto iter_soc = options.find(ge::SOC_VERSION);
  if (iter_soc != options.end() && !iter_soc->second.empty()) {
    FFTS_LOGI("The value of ge.soc_version is %s.", iter_soc->second.c_str());
    soc_version = iter_soc->second;
  } else {
    std::map<std::string, std::string> error_key_map;
    error_key_map[ffts::EM_PARAM] = ffts::STR_SOC_VERSION;
    ffts::LogErrorMessage(ffts::EM_INPUT_PARAM_EMPTY, error_key_map);
    FFTS_LOGE("The parameter 'ge.soc_version' is either missing from the options or its value is empty.");
    return ffts::FAILED;
  }

  for (auto &engine_name : g_engine_name_list) {
    FFTS_LOGD("Start initializing in engine [%s].", engine_name.c_str());
    ffts::EngineManager &em = ffts::EngineManager::Instance(engine_name);
    ffts::Status ret = em.Initialize(options, engine_name, soc_version);
    if (ret != ffts::SUCCESS) {
      em.Finalize();
      return ret;
    }
  }

  return ffts::SUCCESS;
}

ffts::Status Finalize() {
  bool is_any_engine_failed_to_finalize = false;
  for (auto &engine_name : g_engine_name_list) {
    FFTS_LOGD("Start to finalize in engine [%s].", engine_name.c_str());
    ffts::Status ret = ffts::EngineManager::Instance(engine_name).Finalize();
    if (ret != ffts::SUCCESS) {
      /* If any of the engine fail to finalize, we will return FAILED.
       * Set this flag to true and keep finalizing the next engine. */
      is_any_engine_failed_to_finalize = true;
    }
  }
  if (is_any_engine_failed_to_finalize) {
    return ffts::FAILED;
  } else {
    return ffts::SUCCESS;
  }
}

void GetOpsKernelInfoStores(const map<string, OpsKernelInfoStorePtr> &op_kern_infos) {
  (void)op_kern_infos;
}

void GetGraphOptimizerObjs(map<string, GraphOptimizerPtr> &graph_optimizers) {
  for (auto &engine_name : g_engine_name_list) {
    ffts::EngineManager &em = ffts::EngineManager::Instance(engine_name);
    em.GetGraphOptimizerObjs(graph_optimizers, engine_name);
  }
}

void GetCompositeEngines(std::map<string, std::set<std::string>> &compound_engine_contains,
                         std::map<std::string, std::string> &compound_engine_2_kernel_lib_name) {
  if (ffts::GetPlatformFFTSMode() != ffts::FFTSMode::FFTS_MODE_FFTS_PLUS) {
    return;
  }
  std::set<std::string> engine_list;
  engine_list.emplace(ffts::kAIcoreEngineName);
  engine_list.emplace(ffts::kDsaCoreEngineName);
  engine_list.emplace(ffts::kDnnVmAICpu);
  engine_list.emplace(ffts::kDnnVmRts);
  engine_list.emplace(ffts::kDnnVmAICpuAscend);
  engine_list.emplace(ffts::kDnnVmGeLocal);
  compound_engine_contains[ffts::kFFTSPlusCoreName] = engine_list;
  compound_engine_2_kernel_lib_name[ffts::kFFTSPlusCoreName] = ffts::kFFTSPlusCoreName;
}

void GetFFTSPlusSwitch(bool &fftsplus_switch) {
  if (ffts::GetPlatformFFTSMode() != ffts::FFTSMode::FFTS_MODE_FFTS_PLUS) {
    fftsplus_switch = false;
    return;
  }
  auto ffts_flag = ffts::Configuration::Instance().GetFftsOptimizeEnable();
  if (ffts_flag == ffts::kFftsFlagAllSubgraph || ffts_flag == ffts::kFftsFlagDefault) {
    fftsplus_switch = true;
  } else {
    fftsplus_switch = false;
  }
}

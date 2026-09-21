/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "engine_manager.h"

#include <string>
#include "inc/ffts_log.h"
#include "inc/ffts_utils.h"
#include "inc/ffts_configuration.h"
#include "common/sgt_slice_type.h"
#include "inc/ffts_type.h"
#include "ge/ge_api_types.h"

using std::map;
using std::string;

namespace ffts {
EngineManager::EngineManager() : inited_(false) {}

EngineManager::~EngineManager() {}

EngineManager &EngineManager::Instance(const std::string &engine_name) {
  FFTS_LOGD("Instance get, engine_name:[%s]", engine_name.c_str());
  static EngineManager ffts_plus_em;
  return ffts_plus_em;
}

Status EngineManager::Initialize(const map<string, string> &options, const std::string &engine_name,
                                 std::string &soc_version) {
  (void)options;
  FFTS_LOGD("Initialize start, engine_name:[%s]", engine_name.c_str());
  FFTS_TIMECOST_START(EngineMgrInit);
  if (inited_) {
    FFTS_LOGW("EngineManager has been initialized; returning directly!");
    return SUCCESS;
  }
  soc_version_ = soc_version;
  Configuration &config = Configuration::Instance();
  Status ret = config.Initialize();
  if (ret != SUCCESS) {
    FFTS_LOGW("[EngineMngr][Init] Configuration init failed");
  }
  inited_ = true;
  FFTS_LOGI("Initialize successfully.");
  FFTS_TIMECOST_END(EngineMgrInit, "EngineManager::Initialize");
  return SUCCESS;
}

Status EngineManager::Finalize() {
  if (inited_) {
    FFTS_TIMECOST_START(EngineMgrFinal);
    FFTS_LOGD("Finalize begin.");

    Status ret = SUCCESS;
    Configuration &config = Configuration::Instance();
    ret = config.Finalize();
    if (ret != SUCCESS) {
      FFTS_LOGW("[EngineMngr][Init] Configuration finalize failed");
    }
    inited_ = false;
    FFTS_LOGD("Finalize successfully.");
    FFTS_TIMECOST_END(EngineMgrFinal, "EngineManager::Finalize");
    return SUCCESS;
  }

  FFTS_LOGW("Already Finalized, directly return.");
  return SUCCESS;
}


/*
 * to get the information of Graph Optimizer
 * param[out] the map of Graph Optimizer
 */
void EngineManager::GetGraphOptimizerObjs(const map<string, GraphOptimizerPtr> &graph_optimizers,
                                          const std::string &engine_name) const {
  (void)graph_optimizers;
  (void)engine_name;
  FFTS_LOGD("Get GraphOptimizer success.");
}

std::string EngineManager::GetSocVersion() {
  return soc_version_;
}
}  // namespace ffts

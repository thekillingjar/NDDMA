/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "plugin_manager.h"
#include <sstream>
#include "ge/ge_api_types.h"            // ge对内options
#include "framework/common/ge_types.h"  // ge对外options
#include "hccl/hcom.h"
#include "auto_tuning_plugin.h"
#include "hcom_plugin.h"
#include "graph/tuning_utils.h"
#include "op_hcom_comm.h"
#include "hcom_log.h"

static bool g_autoTune = false;
// 单独提供初始化接口
ge::Status Initialize(const std::map<string, string> &options) {
  ge::Status ret;
  HCCL_INFO("hcom ops plugin init start.");
  auto iter = options.find(ge::BUILD_MODE);
  if (iter != options.end()) {
    if (iter->second == ge::BUILD_MODE_TUNING) {
      g_autoTune = true;
      HCCL_INFO("input BUILD_MODE is [%s], result autoTune is true.", iter->second.c_str());
    } else {
      g_autoTune = false;
      HCCL_INFO("input BUILD_MODE is [%s], result autoTune is false.", iter->second.c_str());
    }
  } else {
    g_autoTune = false;
    HCCL_INFO("input BUILD_MODE is not set, result autoTune is false.");
  }

  if (!g_autoTune) {
    HCCL_RUN_INFO("hcom running normal mode.");
    ret = hccl::HcomPlugin::Instance().Initialize(options);
  } else {
    HCCL_RUN_INFO("hcom running auto tuning mode.");
    ret = hccl::AutoTuningPlugin::Instance().Initialize(options);
  }
  HCCL_INFO("hcom ops plugin init end. ret[%u]", ret);
  return ret;
}

// 单独提供销毁接口
ge::Status Finalize() {
  ge::Status ret;
  HCCL_INFO("hccl ops plugin finalize start.");
  if (!g_autoTune) {
    ret = hccl::HcomPlugin::Instance().Finalize();
  } else {
    ret = hccl::AutoTuningPlugin::Instance().Finalize();
  }
  HCCL_INFO("hccl ops plugin finalize end. ret[%u]", ret);
  return ret;
}

void GetGraphOptimizerObjs(std::map<string, GraphOptimizerPtr> &graphOptimizers) {
  if (!g_autoTune) {
    hccl::HcomPlugin::Instance().GetGraphOptimizerObjs(graphOptimizers);
  } else {
    hccl::AutoTuningPlugin::Instance().GetGraphOptimizerObjs(graphOptimizers);
  }
  return;
}

void GetOpsKernelInfoStores(std::map<string, OpsKernelInfoStorePtr> &opKernInfos) {
  if (!g_autoTune) {
    hccl::HcomPlugin::Instance().GetOpsKernelInfoStores(opKernInfos);
  } else {
    hccl::AutoTuningPlugin::Instance().GetOpsKernelInfoStores(opKernInfos);
  }
  return;
}

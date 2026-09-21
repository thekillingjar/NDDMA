/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOM_PLUGIN_H
#define HCOM_PLUGIN_H

#include "hcom_ops_kernel_info_store.h"
#include "hcom_graph_optimizer.h"
#include "hcom_fusion_optimizer.h"
#include "hcom_ops_stores.h"

using HcomOpsKernelInfoStorePtr = std::shared_ptr<hccl::HcomOpsKernelInfoStore>;
using HcomGraphOptimizerPtr = std::shared_ptr<hccl::HcomGraphOptimizer>;
using HcomFusionOptimizerPtr = std::shared_ptr<hccl::HcomFusionOptimizer>;

namespace hccl {
class HcomPlugin {
 public:
  static HcomPlugin &Instance();
  ge::Status Initialize(const std::map<string, string> &options);
  ge::Status Finalize();
  void GetOpsKernelInfoStores(map<string, OpsKernelInfoStorePtr> &opKernInfos);
  void GetOpsKernelInfoPtr(HcomOpsKernelInfoStorePtr &opsKernelInfoStoreInfoPtr);
  void GetGraphOptimizerObjs(map<string, GraphOptimizerPtr> &graphOptimizers);

 private:
  HcomPlugin();
  ~HcomPlugin();
  ge::Status InitializeHcom(const std::map<string, string> &options, HcomInitConfig *comConfig);
  ge::Status ProfilingModeParser(const std::map<string, string> &options);
  HcclResult ConfigHcclDeterministic(const std::map<string, string> &options, HcomInitConfig *comConfig);
  HcclResult ConfigHcclAlgo(const std::map<string, string> &options, HcomInitConfig *comConfig);
  HcclResult ConfigHcclExecTimeOut(const std::map<string, string> &options, HcomInitConfig *comConfig);
  static HcclResult HcomSetGroupToTopoInfo(const char *group, uint32_t rankSize);
  static void HcomUnsetGroupToTopoInfo(const char *group);
  int initializeCount_;
  HcomOpsKernelInfoStorePtr hcomOpsKernelInfoStoreInfoPtr_;
  HcomGraphOptimizerPtr hcomGraphOptimizerPtr_;
  HcomFusionOptimizerPtr hcomFusionOptimizerPtr_;
};

bool GetMasterInfo(const std::map<string, string> &options, std::string &masterIp, std::string &masterPort,
                   std::string &masterDeviceId);
bool GetRankInfo(const std::map<string, string> &options, std::string &rankIp, std::string &rankSize);
}  // namespace hccl
#endif
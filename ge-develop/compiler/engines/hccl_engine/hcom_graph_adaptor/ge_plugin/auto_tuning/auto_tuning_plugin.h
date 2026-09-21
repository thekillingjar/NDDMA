/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTO_TUNING__PLUGIN_H
#define AUTO_TUNING__PLUGIN_H

#include "auto_tuning_hcom_ops_kernel_info_store.h"
#include "auto_tuning_hcom_graph_optimizer.h"

using AutoTuningHcomOpsKernelInfoStorePtr = std::shared_ptr<hccl::AutoTuningHcomOpsKernelInfoStore>;
using AutoTuningHcomGraphOptimizerPtr = std::shared_ptr<hccl::AutoTuningHcomGraphOptimizer>;

namespace hccl {
class AutoTuningPlugin {
 public:
  static AutoTuningPlugin &Instance();
  ge::Status Initialize(const std::map<string, string> &options);
  ge::Status Finalize();
  void GetOpsKernelInfoStores(map<string, OpsKernelInfoStorePtr> &opKernInfos);
  void GetGraphOptimizerObjs(map<string, GraphOptimizerPtr> &graphOptimizers);

 private:
  AutoTuningPlugin();
  ~AutoTuningPlugin();
  AutoTuningHcomOpsKernelInfoStorePtr opsKernelInfoStorePtr_;
  AutoTuningHcomGraphOptimizerPtr graphOptimizerPtr_;
};
}  // namespace hccl

#endif
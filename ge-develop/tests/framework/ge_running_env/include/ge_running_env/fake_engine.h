/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HAF5E9BF2_752F_4E03_B0A5_E1B912A5FA24
#define HAF5E9BF2_752F_4E03_B0A5_E1B912A5FA24

#include <string>
#include "fake_ns.h"
#include "ge_running_env/env_installer.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "engines/manager/opskernel_manager/ops_kernel_manager.h"
#include "register/ops_kernel_builder_registry.h"
#include "fake_ops_kernel_builder.h"
#include "fake_ops_kernel_info_store.h"
#include "fake_graph_optimizer.h"

FAKE_NS_BEGIN

using FakeOpsKernelBuilderPtr = std::shared_ptr<FakeOpsKernelBuilder>;
using FakeOpsKernelInfoStorePtr = std::shared_ptr<FakeOpsKernelInfoStore>;
using FakeGraphOptimiezrPtr = std::shared_ptr<FakeGraphOptimizer>;

struct FakeEngine : EnvInstaller {
  FakeEngine(const std::string& engine_name);
  FakeEngine& KernelBuilder(FakeOpsKernelBuilderPtr);
  FakeEngine& KernelInfoStore(FakeOpsKernelInfoStorePtr);
  FakeEngine& KernelInfoStore(const std::string&);
  FakeEngine& GraphOptimizer(std::string name, FakeGraphOptimiezrPtr);
  FakeEngine& GraphOptimizer(std::string name);
  FakeEngine& Priority(PriorityEnum compute_cost);

 private:
  void InstallTo(std::map<string, OpsKernelInfoStorePtr>&) const override;
  void InstallTo(std::map<string, OpsKernelBuilderPtr>&) const override;
  void InstallTo(std::map<std::string, DNNEnginePtr>&) const override;
  void InstallTo(map<string, GraphOptimizerPtr> &,
                 vector<std::pair<std::string, GraphOptimizerPtr>> &) const override;

  template <typename BasePtr, typename SubClass>
  void InstallFor(std::map<string, BasePtr>& maps, const std::map<std::string, std::shared_ptr<SubClass>>&) const;

 protected:
  std::string engine_name_;
  std::set<std::string> info_store_names_;
  PriorityEnum compute_cost_;

 private:
  std::map<std::string, FakeOpsKernelBuilderPtr> custom_builders_;
  std::map<std::string, FakeOpsKernelInfoStorePtr> custom_info_stores_;
  std::map<std::string, FakeGraphOptimiezrPtr> graph_optimizers_;
};

FAKE_NS_END

#endif

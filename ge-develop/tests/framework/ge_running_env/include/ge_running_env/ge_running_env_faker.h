/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef H99C11FC4_700E_4D4D_B073_7808FA88BEBC
#define H99C11FC4_700E_4D4D_B073_7808FA88BEBC

#include "ge_running_env/fake_engine.h"
#include "fake_ns.h"
#include "engines/manager/opskernel_manager/ops_kernel_manager.h"
#include "register/ops_kernel_builder_registry.h"

FAKE_NS_BEGIN

struct GeRunningEnvFaker {
  GeRunningEnvFaker();
  GeRunningEnvFaker &Reset();
  GeRunningEnvFaker &Install(const EnvInstaller &);
  GeRunningEnvFaker &InstallDefault();
  static void SetEnvForOfflineSoPack();
  static void ResetEnvForOfflineSoPack();
  static void BackupEnv();

 private:
  void flush();

 private:
  std::map<string, vector<OpInfo>> &op_kernel_info_;
  std::map<string, OpsKernelInfoStorePtr> &ops_kernel_info_stores_;
  std::map<string, GraphOptimizerPtr> &ops_kernel_optimizers_;
  std::vector<std::pair<std::string, GraphOptimizerPtr>> &optimizers_by_priority_;
  std::map<string, OpsKernelBuilderPtr> &ops_kernel_builders_;
  std::map<string, std::set<std::string>> &composite_engines_;
  std::map<string, std::string> &composite_engine_kernel_lib_names_;
  std::map<std::string, DNNEnginePtr> &engine_map_;
};

FAKE_NS_END

#endif /* H99C11FC4_700E_4D4D_B073_7808FA88BEBC */

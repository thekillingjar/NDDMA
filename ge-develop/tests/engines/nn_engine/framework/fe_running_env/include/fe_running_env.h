/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_TEST_ENGINES_NNENG_FRAMEWORK_FE_RUNNING_ENV_INCLUDE_FE_RUNNING_ENV_H_
#define AIR_TEST_ENGINES_NNENG_FRAMEWORK_FE_RUNNING_ENV_INCLUDE_FE_RUNNING_ENV_H_

#include "ge/ge_api.h"
#include "itf_handler/itf_handler.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include <sstream>
#define protected public
#define private public
#include "engines/manager/opskernel_manager/ops_kernel_manager.h"
#include "common/configuration.h"
#include "graph/ge_context.h"
#include "graph/ge_local_context.h"
#include "proto/task.pb.h"
#undef protected
#undef private

namespace fe_env {
using OpsKernelInfoStorePtr = std::shared_ptr<ge::OpsKernelInfoStore>;
using GraphOptimizerPtr = std::shared_ptr<ge::GraphOptimizer>;
class FeRunningEnv {
 public:
  FeRunningEnv(const FeRunningEnv &) = delete;
  FeRunningEnv &operator=(const FeRunningEnv &) = delete;

  static FeRunningEnv &Instance();

  fe::Status InitRuningEnv(const std::map<string, string> &options);

  fe::Status FinalizeEnv();

  fe::Status ResetEnv(const std::map<string, string> &options);

  void Clear();

  fe::Status Run(ge::ComputeGraphPtr &compute_graph, std::map<string, string> &session_options,
                 bool need_update_name = false);

  static string GetNetworkPath(const string &network_name);

  static std::map<std::string, std::vector<domi::TaskDef>> tasks_map;
 private:
  void InjectFe();

  void InjectFFTS();

  void InitBasicOptions(const std::map<string, string> &options);

  void RenameLibfe();
  void ReStoreLibfe();
  void RenameLibFFTS();
  void ReStoreLibFFTS();
  void RenameLibfeCfg();
  void ReStoreLibfeCfg();
  FeRunningEnv();
  ~FeRunningEnv();
  std::map<std::string, std::vector<ge::OpInfo>> &ops_kernel_info_;
  std::map<string, OpsKernelInfoStorePtr> &ops_kernel_info_stores_;
  std::map<string, GraphOptimizerPtr> &ops_kernel_optimizers_;
  std::vector<std::pair<std::string, GraphOptimizerPtr>> &atomic_first_optimizers_by_priority_;
  std::map<std::string, std::set<std::string>> &composite_engines_;
  std::map<std::string, std::string> &composite_engine_kernel_lib_names_;
  std::string user_engine_name_;
  std::map<string, string> options_;
  std::map<ge::AscendString, ge::AscendString> ascend_options_;
  string ascend_path_;
};
}
#endif //AIR_TEST_ENGINES_NNENG_FRAMEWORK_FE_RUNNING_ENV_INCLUDE_FE_RUNNING_ENV_H_

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <map>
#include "ge/ge_api.h"
#include "engines/manager/opskernel_manager/ops_kernel_builder_manager.h"
#include "api/gelib/gelib.h"
#include "engines/manager/engine/engine_manager.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "ge_default_running_env.h"
#include "op/fake_op_repo.h"
#include "graph_metadef/common/plugin/plugin_manager.h"

FAKE_NS_BEGIN

namespace {
const char *const kEnvName = "ASCEND_OPP_PATH";
const string kOpsProto = "libopsproto_rt2.0.so";
const string kOpMaster = "libopmaster_rt2.0.so";
const string kInner = "built-in";
const string kx86OpsProtoPath = "/op_proto/lib/linux/x86_64/";
const string kx86OpMasterPath = "/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/";
const string kaarch64OpsProtoPath = "/op_proto/lib/linux/aarch64/";
const string kaarch64OpMasterPath = "/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/";

struct InitEnv {
  static InitEnv& GetInstance() {
    static InitEnv instance;
    return instance;
  }

  void reset(std::map<string, OpsKernelInfoStorePtr> &ops_kernel_info_stores,
             std::map<string, OpsKernelBuilderPtr> &builders,
             std::map<string, GraphOptimizerPtr> &ops_kernel_optimizers,
             std::vector<std::pair<std::string, GraphOptimizerPtr>> &optimizers_by_priority,
             std::map<string, std::set<std::string>> &composite_engines,
             std::map<string, std::string> &composite_engine_kernel_lib_names,
             std::map<string, DNNEnginePtr> &engines) {
    std::set<string> remove_info_names;
    for (auto iter : builders) {
      if (kernel_info_names.find(iter.first) == kernel_info_names.end()) {
        remove_info_names.insert(iter.first);
      }
    }
    for (auto info_name : remove_info_names) {
      ops_kernel_info_stores.erase(info_name);
      builders.erase(info_name);
      ops_kernel_optimizers.erase(info_name);
      composite_engines.erase(info_name);
      composite_engine_kernel_lib_names.erase(info_name);
      engines.erase(info_name);
    }

    for (auto iter = ops_kernel_optimizers.begin(); iter != ops_kernel_optimizers.end(); ) {
      if (optimizer_names.count(iter->first) == 0) {
        iter = ops_kernel_optimizers.erase(iter);
      } else {
        ++iter;
      }
    }

    for (auto iter = optimizers_by_priority.begin(); iter != optimizers_by_priority.end(); ) {
      if (optimizer_names.count(iter->first) == 0) {
        iter = optimizers_by_priority.erase(iter);
      } else {
        ++iter;
      }
    }
  }

 private:
  InitEnv() {
    for (auto iter : OpsKernelManager::GetInstance().GetAllOpsKernelInfoStores()) {
      kernel_info_names.insert(iter.first);
    }
    for (const auto &iter : OpsKernelManager::GetInstance().GetAllGraphOptimizerObjs()) {
      optimizer_names.insert(iter.first);
    }
  }

 private:
  std::set<std::string> kernel_info_names;
  std::set<std::string> optimizer_names;
};
}  // namespace

GeRunningEnvFaker::GeRunningEnvFaker()
    : op_kernel_info_(const_cast<std::map<std::string, vector<OpInfo>>&>(
          OpsKernelManager::GetInstance().GetAllOpsKernelInfo())),
      ops_kernel_info_stores_(const_cast<std::map<std::string, OpsKernelInfoStorePtr>&>(
          OpsKernelManager::GetInstance().GetAllOpsKernelInfoStores())),
      ops_kernel_optimizers_(const_cast<std::map<std::string, GraphOptimizerPtr>&>(
          OpsKernelManager::GetInstance().GetAllGraphOptimizerObjs())),
      optimizers_by_priority_(const_cast<std::vector<std::pair<std::string, GraphOptimizerPtr>>&>(
          OpsKernelManager::GetInstance().GetAllGraphOptimizerObjsByPriority())),
      ops_kernel_builders_(const_cast<std::map<std::string, OpsKernelBuilderPtr>&>(
        OpsKernelBuilderManager::Instance().GetAllOpsKernelBuilders())),
      composite_engines_(const_cast<std::map<std::string, std::set<std::string>>&>(
          OpsKernelManager::GetInstance().GetCompositeEngines())),
      composite_engine_kernel_lib_names_(const_cast<std::map<std::string, std::string>&>(
          OpsKernelManager::GetInstance().GetCompositeEngineKernelLibNames())),
      engine_map_(const_cast<std::map<std::string, DNNEnginePtr>&>(DNNEngineManager::GetInstance().GetAllEngines())) {
  Reset();
}

GeRunningEnvFaker& GeRunningEnvFaker::Reset() {
  InitEnv& init_env = InitEnv::GetInstance();
  FakeOpRepo::Reset();
  init_env.reset(ops_kernel_info_stores_, ops_kernel_builders_,
                 ops_kernel_optimizers_, optimizers_by_priority_,
                 composite_engines_, composite_engine_kernel_lib_names_, engine_map_);
  flush();
  return *this;
}

void GeRunningEnvFaker::BackupEnv() { InitEnv::GetInstance(); }

GeRunningEnvFaker& GeRunningEnvFaker::Install(const EnvInstaller& installer) {
  installer.Install();
  installer.InstallTo(ops_kernel_info_stores_);
  installer.InstallTo(ops_kernel_optimizers_, optimizers_by_priority_);
  installer.InstallTo(ops_kernel_builders_);
  installer.InstallTo(composite_engines_);
  installer.InstallTo(composite_engine_kernel_lib_names_);
  installer.InstallTo(engine_map_);

  flush();
  return *this;
}

void GeRunningEnvFaker::flush() {
  op_kernel_info_.clear();
  OpsKernelManager::GetInstance().GetOpsKernelInfo("");
}

GeRunningEnvFaker& GeRunningEnvFaker::InstallDefault() {
  Reset();
  GeDefaultRunningEnv::InstallTo(*this);
  return *this;
}

void GeRunningEnvFaker::SetEnvForOfflineSoPack() {
  std::string path = GetModelPath();
  path = path.substr(0, path.rfind('/'));
  path = path.substr(0, path.rfind('/'));
  path = path.substr(0, path.rfind('/') + 1);

  std::string opp_path = path + "opp/";
  mmSetEnv(kEnvName, opp_path.c_str(), 1);
}

void GeRunningEnvFaker::ResetEnvForOfflineSoPack() {
  std::string opp_path = "./";
  std::string path_vendors = opp_path + "vendors";
  system(("rm -rf " + path_vendors).c_str());
  std::string path_so = opp_path + kInner;
  system(("rm -rf " + path_so).c_str());
  unsetenv(kEnvName);
}

FAKE_NS_END

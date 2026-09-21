/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ge_running_env/fake_engine.h"

#include <utility>

FAKE_NS_BEGIN

FakeEngine::FakeEngine(const std::string &engine_name) : engine_name_(engine_name) {}

FakeEngine &FakeEngine::KernelInfoStore(const std::string &info_store) {
  info_store_names_.insert(info_store);
  return *this;
}

FakeEngine &FakeEngine::KernelInfoStore(FakeOpsKernelInfoStorePtr ptr) {
  info_store_names_.insert(ptr->GetLibName());
  custom_info_stores_.insert(std::make_pair(ptr->GetLibName(), ptr));
  return *this;
}

FakeEngine &FakeEngine::KernelBuilder(FakeOpsKernelBuilderPtr builder) {
  info_store_names_.insert(builder->GetLibName());
  custom_builders_.insert(std::make_pair(builder->GetLibName(), builder));
  return *this;
}

FakeEngine &FakeEngine::Priority(PriorityEnum compute_cost) {
  compute_cost_ = compute_cost;
  return *this;
}

namespace {
template <typename BasePtr, typename SubClass>
void InstallDefault(std::map<string, BasePtr> &maps, const std::string &info_store_name,
                    const std::string &engine_name) {
  auto parent_obj = std::make_shared<SubClass>(info_store_name);
  if (parent_obj == nullptr) {
    return;
  }
  parent_obj->EngineName(engine_name);
  maps.insert(std::make_pair(parent_obj->GetLibName(), parent_obj));
}
}  // namespace

template <typename BasePtr, typename SubClass>
void FakeEngine::InstallFor(std::map<string, BasePtr> &maps,
                            const std::map<std::string, std::shared_ptr<SubClass>> &child_maps) const {
  if (info_store_names_.empty()) {
    InstallDefault<BasePtr, SubClass>(maps, engine_name_, engine_name_);
  } else {
    for (auto &info_store_name : info_store_names_) {
      auto iter = child_maps.find(info_store_name);
      if (iter == child_maps.end()) {
        InstallDefault<BasePtr, SubClass>(maps, info_store_name, engine_name_);
      } else {
        maps[iter->second->GetLibName()] = iter->second;
      }
    }
  }
}

void FakeEngine::InstallTo(std::map<string, OpsKernelInfoStorePtr> &ops_kernel_info_stores) const {
  InstallFor<OpsKernelInfoStorePtr, FakeOpsKernelInfoStore>(ops_kernel_info_stores, custom_info_stores_);
}

void FakeEngine::InstallTo(std::map<string, OpsKernelBuilderPtr> &ops_kernel_builders) const {
  InstallFor<OpsKernelBuilderPtr, FakeOpsKernelBuilder>(ops_kernel_builders, custom_builders_);
}

void FakeEngine::InstallTo(std::map<string, DNNEnginePtr> &engines) const {
  DNNEngineAttribute attr;
  attr.engine_name = engine_name_;
  attr.atomic_engine_flag = true;
  attr.compute_cost = compute_cost_;
  engines[engine_name_] = MakeShared<DNNEngine>(attr);
}
FakeEngine &FakeEngine::GraphOptimizer(std::string name, FakeGraphOptimiezrPtr fake_optimizer) {
  graph_optimizers_[std::move(name)] = std::move(fake_optimizer);
  return *this;
}
FakeEngine &FakeEngine::GraphOptimizer(std::string name) {
  return GraphOptimizer(std::move(name), MakeShared<FakeGraphOptimizer>());
}
void FakeEngine::InstallTo(map<string, GraphOptimizerPtr> &optimizers,
                           vector<std::pair<std::string, GraphOptimizerPtr>> &optimizers_by_priority) const {
  optimizers.insert(graph_optimizers_.begin(), graph_optimizers_.end());
  optimizers_by_priority.insert(optimizers_by_priority.end(), graph_optimizers_.begin(), graph_optimizers_.end());
}

FAKE_NS_END

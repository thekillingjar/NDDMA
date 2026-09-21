/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include <algorithm>
#include <map>
#include <mutex>
#include <utility>
#include <vector>
#include "framework/common/debug/ge_log.h"

namespace fe {
class FusionPassRegistry::FusionPassRegistryImpl {
 public:
  void RegisterPass(const GraphFusionPassType pass_type, const std::string &pass_name,
                    FusionPassRegistry::CreateFn const create_fn, PassAttr attr) {
    RegPassCompileLevel(pass_name, attr);
    const std::string pass_module = IsPassAttrTypeOn(attr, PassAttrType::FE_PASS_FLAG) ? "FE" : "TBE";
    const std::lock_guard<std::mutex> my_lock(mu_);
    if (pass_descs_.find(pass_type) != pass_descs_.end()) {
      pass_descs_[pass_type][pass_name].attr = attr;
      pass_descs_[pass_type][pass_name].create_fn = create_fn;
      GELOGD("GraphFusionPass[type=%d, name=%s, attr=%lu, module=%s]: the pass type has already existed.",
             pass_type, pass_name.c_str(), attr, pass_module.c_str());
      return;
    }

    std::map<std::string, PassDesc> pass_desc;
    pass_desc[pass_name] = {attr, create_fn};
    pass_descs_[pass_type] = pass_desc;
    GELOGD("GraphFusionPass[type=%d, name=%s, attr=%lu, module=%s]: the pass type does not exist.",
           pass_type, pass_name.c_str(), attr, pass_module.c_str());
  }

  std::map<std::string, PassDesc> GetPassDesc(const GraphFusionPassType &pass_type) {
    const std::lock_guard<std::mutex> my_lock(mu_);
    std::map<GraphFusionPassType, map<std::string, PassDesc>>::const_iterator iter = pass_descs_.find(pass_type);
    if (iter == pass_descs_.end()) {
      std::map<std::string, PassDesc> ret;
      return ret;
    }

    return iter->second;
  }

  std::map<std::string, FusionPassRegistry::CreateFn> GetCreateFn(const GraphFusionPassType &pass_type) {
    const std::lock_guard<std::mutex> my_lock(mu_);
    const auto iter = pass_descs_.find(pass_type);
    std::map<std::string, FusionPassRegistry::CreateFn> ret;
    if (iter == pass_descs_.end()) {
      return ret;
    }

    for (const auto &ele : iter->second) {
      std::ignore = ret.emplace(std::make_pair(ele.first, ele.second.create_fn));
    }
    return ret;
  }
private:
  std::mutex mu_;
  std::map<GraphFusionPassType, map<std::string, PassDesc>> pass_descs_;
};

FusionPassRegistry::FusionPassRegistry() {
  impl_ = std::unique_ptr<FusionPassRegistryImpl>(new (std::nothrow) FusionPassRegistryImpl);
}

FusionPassRegistry::~FusionPassRegistry() {}

FusionPassRegistry &FusionPassRegistry::GetInstance() {
  static FusionPassRegistry instance;
  return instance;
}

void FusionPassRegistry::RegisterPass(const GraphFusionPassType &pass_type, const std::string &pass_name,
                                      CreateFn create_fn, PassAttr attr) const {
  if (impl_ == nullptr) {
    GELOGE(ge::MEMALLOC_FAILED, "[Check][Param]param impl is nullptr, GraphFusionPass[type=%d,name=%s]: "
           "failed to register the graph fusion pass",
           pass_type, pass_name.c_str());
    return;
  }
  impl_->RegisterPass(pass_type, pass_name, create_fn, attr);
}

std::map<std::string, FusionPassRegistry::PassDesc> FusionPassRegistry::GetPassDesc(
    const GraphFusionPassType &pass_type) {
  if (impl_ == nullptr) {
    GELOGE(ge::MEMALLOC_FAILED, "[Check][Param]param impl is nullptr, GraphFusionPass[type=%d]: "
                                "failed to get pass desc.", pass_type);
    std::map<std::string, PassDesc> ret;
    return ret;
  }
  return impl_->GetPassDesc(pass_type);
}

std::map<std::string, FusionPassRegistry::CreateFn> FusionPassRegistry::GetCreateFnByType(
    const GraphFusionPassType &pass_type) {
  if (impl_ == nullptr) {
    GELOGE(ge::MEMALLOC_FAILED, "[Check][Param]param impl is nullptr, GraphFusionPass[type=%d]: "
           "failed to create the graph fusion pass.", pass_type);
    return std::map<std::string, CreateFn>{};
  }
  return impl_->GetCreateFn(pass_type);
}

FusionPassRegistrar::FusionPassRegistrar(const GraphFusionPassType &pass_type, const std::string &pass_name,
                                         GraphPass *(*create_fn)(), PassAttr attr) {
  if ((pass_type < BUILT_IN_GRAPH_PASS) || (pass_type >= GRAPH_FUSION_PASS_TYPE_RESERVED)) {
    GELOGE(ge::PARAM_INVALID, "[Check][Param:pass_type] value:%d is not supported.", pass_type);
    return;
  }

  if (pass_name.empty()) {
    GELOGE(ge::PARAM_INVALID, "[Check][Param:pass_name]Failed to register the graph fusion pass, "
           "param pass_name is empty.");
    return;
  }
  FusionPassRegistry::GetInstance().RegisterPass(pass_type, pass_name, create_fn, attr);
}
}  // namespace fe

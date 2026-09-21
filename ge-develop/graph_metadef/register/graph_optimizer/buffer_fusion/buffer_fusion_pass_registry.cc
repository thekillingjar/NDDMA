/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include <algorithm>
#include <map>
#include <mutex>
#include <utility>
#include <vector>
#include "framework/common/debug/ge_log.h"

namespace fe {
class BufferFusionPassRegistry::BufferFusionPassRegistryImpl {
 public:
  void RegisterPass(const BufferFusionPassType pass_type, const std::string &pass_name,
                    BufferFusionPassRegistry::CreateFn const create_fn, PassAttr attr) {
    RegPassCompileLevel(pass_name, attr);
    const std::string pass_module = IsPassAttrTypeOn(attr, PassAttrType::FE_PASS_FLAG) ? "FE" : "TBE";
    const std::lock_guard<std::mutex> lock(mu_);
    std::map<BufferFusionPassType, std::map<std::string, PassDesc>>::const_iterator iter = pass_descs_.find(pass_type);
    if (iter != pass_descs_.cend()) {
      pass_descs_[pass_type][pass_name].attr = attr;
      pass_descs_[pass_type][pass_name].create_fn = create_fn;
      GELOGI("type=%d, name=%s, attr=%lu, module=%s.", pass_type, pass_name.c_str(), attr, pass_module.c_str());
      return;
    }

    std::map<std::string, PassDesc> pass_desc;
    pass_desc[pass_name] = {attr, create_fn};
    pass_descs_[pass_type] = pass_desc;
    GELOGI("type=%d, name=%s, attr=%lu, module=%s.", pass_type, pass_name.c_str(), attr, pass_module.c_str());
  }

  std::map<std::string, BufferFusionPassRegistry::CreateFn> GetCreateFn(const BufferFusionPassType &pass_type) {
    const std::lock_guard<std::mutex> lock(mu_);
    std::map<std::string, BufferFusionPassRegistry::CreateFn> ret;
    std::map<BufferFusionPassType, std::map<std::string, PassDesc>>::const_iterator iter = pass_descs_.find(pass_type);
    if (iter == pass_descs_.cend()) {
      return ret;
    }
    for (const auto &ele : iter->second) {
      std::ignore = ret.emplace(std::make_pair(ele.first, ele.second.create_fn));
    }
    return ret;
  }

  std::map<std::string, PassDesc> GetPassDesc(const BufferFusionPassType &pass_type) {
    const std::lock_guard<std::mutex> lock(mu_);
    std::map<BufferFusionPassType, std::map<std::string, PassDesc>>::const_iterator iter = pass_descs_.find(pass_type);
    if (iter == pass_descs_.cend()) {
      std::map<std::string, PassDesc> result;
      return result;
    }
    return iter->second;
  }
 private:
  std::mutex mu_;
  std::map<BufferFusionPassType, std::map<std::string, PassDesc>> pass_descs_;
};

BufferFusionPassRegistry::BufferFusionPassRegistry() {
  impl_ = std::unique_ptr<BufferFusionPassRegistryImpl>(new (std::nothrow) BufferFusionPassRegistryImpl);
}

BufferFusionPassRegistry::~BufferFusionPassRegistry() {}

BufferFusionPassRegistry &BufferFusionPassRegistry::GetInstance() {
  static BufferFusionPassRegistry instance;
  return instance;
}

void BufferFusionPassRegistry::RegisterPass(const BufferFusionPassType pass_type, const std::string &pass_name,
                                            CreateFn create_fn, PassAttr attr) {
  if (impl_ == nullptr) {
    GELOGE(ge::MEMALLOC_FAILED, "[Check][Param]UbFusionPass[type=%d,name=%s]: failed to register the ub fusion pass",
           pass_type, pass_name.c_str());
    return;
  }
  impl_->RegisterPass(pass_type, pass_name, create_fn, attr);
}

std::map<std::string, BufferFusionPassRegistry::PassDesc> BufferFusionPassRegistry::GetPassDesc(
    const BufferFusionPassType &pass_type) {
  if (impl_ == nullptr) {
    GELOGE(ge::MEMALLOC_FAILED, "[Check][Param]UbFusionPass[type=%d]: failed to get pass desc", pass_type);
    std::map<std::string, PassDesc> ret;
    return ret;
  }
  return impl_->GetPassDesc(pass_type);
}

std::map<std::string, BufferFusionPassRegistry::CreateFn> BufferFusionPassRegistry::GetCreateFnByType(
    const BufferFusionPassType &pass_type) {
  if (impl_ == nullptr) {
    GELOGE(ge::MEMALLOC_FAILED, "[Check][Param]UbFusionPass[type=%d]: failed to create the ub fusion pass", pass_type);
    return std::map<std::string, CreateFn>{};
  }
  return impl_->GetCreateFn(pass_type);
}

BufferFusionPassRegistrar::BufferFusionPassRegistrar(const BufferFusionPassType &pass_type,
                                                     const std::string &pass_name,
                                                     BufferFusionPassBase *(*create_fun)(),
                                                     PassAttr attr) {
  if ((pass_type < BUILT_IN_AI_CORE_BUFFER_FUSION_PASS) || (pass_type >= BUFFER_FUSION_PASS_TYPE_RESERVED)) {
    GELOGE(ge::PARAM_INVALID, "[Check][Param:pass_type] value %d is not supported.", pass_type);
    return;
  }

  if (pass_name.empty()) {
    GELOGE(ge::PARAM_INVALID, "[Check][Param:pass_name]Failed to register the ub fusion pass, the pass name is empty.");
    return;
  }

  BufferFusionPassRegistry::GetInstance().RegisterPass(pass_type, pass_name, create_fun, attr);
}
}  // namespace fe

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_REGISTER_GRAPH_OPTIMIZER_GRAPH_FUSION_FUSION_PASS_MANAGER_FUSION_PASS_REGISTRY_H_
#define INC_REGISTER_GRAPH_OPTIMIZER_GRAPH_FUSION_FUSION_PASS_MANAGER_FUSION_PASS_REGISTRY_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "register/graph_optimizer/fusion_common/fusion_pass_desc.h"
#include "register/graph_optimizer/graph_fusion/graph_fusion_pass_base.h"

namespace fe {
class FusionPassRegistry {
 public:
  using CreateFn = GraphPass *(*)();
  struct PassDesc {
    PassAttr attr;
    CreateFn create_fn;
  };
  ~FusionPassRegistry();

  static FusionPassRegistry &GetInstance();

  void RegisterPass(const GraphFusionPassType &pass_type, const std::string &pass_name, CreateFn create_fn,
                    PassAttr attr) const;

  std::map<std::string, PassDesc> GetPassDesc(const GraphFusionPassType &pass_type);

  std::map<std::string, CreateFn> GetCreateFnByType(const GraphFusionPassType &pass_type);

 private:
  FusionPassRegistry();
  class FusionPassRegistryImpl;
  std::unique_ptr<FusionPassRegistryImpl> impl_;
};

class FusionPassRegistrar {
 public:
  FusionPassRegistrar(const GraphFusionPassType &pass_type, const std::string &pass_name,
                      GraphPass *(*create_fn)(), PassAttr attr);

  ~FusionPassRegistrar() {}
};

#define REGISTER_PASS(pass_name, pass_type, pass_class) \
  REG_PASS(pass_name, pass_type, pass_class, 0)

#define REG_PASS(pass_name, pass_type, pass_class, attr) \
  REG_PASS_UNIQ_HELPER(__COUNTER__, pass_name, pass_type, pass_class, attr)

#define REG_PASS_UNIQ_HELPER(ctr, pass_name, pass_type, pass_class, attr) \
  REG_PASS_UNIQ(ctr, pass_name, pass_type, pass_class, attr)

#define REG_PASS_UNIQ(ctr, pass_name, pass_type, pass_class, attr)                                                 \
  static ::fe::FusionPassRegistrar register_fusion_pass##ctr __attribute__((unused)) = ::fe::FusionPassRegistrar( \
      pass_type, pass_name, []() -> ::fe::GraphPass * { return new (std::nothrow) pass_class(); }, attr)
}  // namespace fe
#endif  // INC_REGISTER_GRAPH_OPTIMIZER_GRAPH_FUSION_FUSION_PASS_MANAGER_FUSION_PASS_REGISTRY_H_

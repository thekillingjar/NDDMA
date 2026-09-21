/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_REGISTER_GRAPH_OPTIMIZER_BUFFER_FUSION_BUFFER_FUSION_PASS_REGISTRY_H_
#define INC_REGISTER_GRAPH_OPTIMIZER_BUFFER_FUSION_BUFFER_FUSION_PASS_REGISTRY_H_
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "register/graph_optimizer/fusion_common/fusion_pass_desc.h"
#include "register/graph_optimizer/buffer_fusion/buffer_fusion_pass_base.h"

namespace fe {
class BufferFusionPassRegistry {
 public:
  using CreateFn = BufferFusionPassBase *(*)();
  struct PassDesc {
    PassAttr attr;
    CreateFn create_fn;
  };
  ~BufferFusionPassRegistry();

  static BufferFusionPassRegistry &GetInstance();

  void RegisterPass(const BufferFusionPassType pass_type, const std::string &pass_name,
                    CreateFn create_fn, PassAttr attr);

  std::map<std::string, PassDesc> GetPassDesc(const BufferFusionPassType &pass_type);

  std::map<std::string, CreateFn> GetCreateFnByType(const BufferFusionPassType &pass_type);

 private:
  BufferFusionPassRegistry();
  class BufferFusionPassRegistryImpl;
  std::unique_ptr<BufferFusionPassRegistryImpl> impl_;
};

class BufferFusionPassRegistrar {
 public:
  BufferFusionPassRegistrar(const BufferFusionPassType &pass_type, const std::string &pass_name,
                            BufferFusionPassBase *(*create_fun)(), PassAttr attr);

  ~BufferFusionPassRegistrar() {}
};

#define REGISTER_BUFFER_FUSION_PASS(pass_name, pass_type, pass_class) \
  REG_BUFFER_FUSION_PASS(pass_name, pass_type, pass_class, 0)

#define REG_BUFFER_FUSION_PASS(pass_name, pass_type, pass_class, attr) \
  REG_BUFFER_FUSION_PASS_UNIQ_HELPER(__COUNTER__, pass_name, pass_type, pass_class, attr)

#define REG_BUFFER_FUSION_PASS_UNIQ_HELPER(ctr, pass_name, pass_type, pass_class, attr) \
  REG_BUFFER_FUSION_PASS_UNIQ(ctr, pass_name, pass_type, pass_class, attr)

#define REG_BUFFER_FUSION_PASS_UNIQ(ctr, pass_name, pass_type, pass_class, attr)                     \
  static ::fe::BufferFusionPassRegistrar register_buffer_fusion_##ctr __attribute__((unused)) = \
      ::fe::BufferFusionPassRegistrar(                                                               \
      (pass_type), (pass_name),                                          \
      []() -> ::fe::BufferFusionPassBase * { return new (std::nothrow) pass_class();}, (attr))
}  // namespace fe
#endif  // INC_REGISTER_GRAPH_OPTIMIZER_BUFFER_FUSION_BUFFER_FUSION_PASS_REGISTRY_H_

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_EXECUTOR_HYBRID_COMMON_BIN_CACHE_NODE_BIN_SELECTOR_FACTORY_H_
#define AIR_CXX_EXECUTOR_HYBRID_COMMON_BIN_CACHE_NODE_BIN_SELECTOR_FACTORY_H_
#include <array>
#include <memory>
#include "node_bin_selector.h"
#include "bin_cache_def.h"
#include "common/plugin/ge_make_unique_util.h"

namespace ge {
class NodeBinSelectorFactory {
 public:
  static NodeBinSelectorFactory &GetInstance();
  NodeBinSelector *GetNodeBinSelector(const fuzz_compile::NodeBinMode node_bin_type);

  template <typename T>
  class Register {
   public:
    explicit Register(const fuzz_compile::NodeBinMode node_bin_mode) {
      if (node_bin_mode < fuzz_compile::kNodeBinModeEnd) {
        NodeBinSelectorFactory::GetInstance().selectors_[node_bin_mode] = MakeUnique<T>();
      }
    }

    template<typename... Args>
    Register(const fuzz_compile::NodeBinMode node_bin_mode, Args... args) {
      if (node_bin_mode < fuzz_compile::kNodeBinModeEnd) {
        NodeBinSelectorFactory::GetInstance().selectors_[node_bin_mode] = MakeUnique<T>(args...);
      }
    }
  };

 private:
  NodeBinSelectorFactory();
  std::array<std::unique_ptr<NodeBinSelector>, fuzz_compile::kNodeBinModeEnd> selectors_;
};
}  // namespace ge

#define REGISTER_BIN_SELECTOR(node_bin_mode, T, ...) \
  static ge::NodeBinSelectorFactory::Register<T> Register_##T##_(node_bin_mode, ##__VA_ARGS__)
#endif // AIR_CXX_EXECUTOR_HYBRID_COMMON_BIN_CACHE_NODE_BIN_SELECTOR_FACTORY_H_

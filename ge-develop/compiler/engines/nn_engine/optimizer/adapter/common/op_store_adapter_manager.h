/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_ADAPTER_COMMON_OP_STORE_ADAPTER_MANAGER_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_ADAPTER_COMMON_OP_STORE_ADAPTER_MANAGER_H_

#include <unordered_map>
#include <memory>
#include <string>
#include "adapter/adapter_itf/op_store_adapter.h"

namespace fe {
class OpStoreAdapterManager {
 public:
  static OpStoreAdapterManager &Instance(const std::string &engine_name);

  OpStoreAdapterManager(const OpStoreAdapterManager &) = delete;

  OpStoreAdapterManager &operator=(const OpStoreAdapterManager &) = delete;

  Status Initialize(const std::map<std::string, std::string> &options);

  Status Finalize();

  Status GetOpStoreAdapter(const OpImplType &op_impl_type, OpStoreAdapterPtr &adapter_ptr) const;

  OpStoreAdapterPtr GetOpStoreAdapter(const OpImplType &op_impl_type) const;

 private:
  explicit OpStoreAdapterManager(const std::string &engine_name);
  ~OpStoreAdapterManager();
  bool init_flag_;
  std::string engine_name_;
  std::unordered_map<std::string, OpStoreAdapterPtr> map_all_op_store_adapter_;
  Status InitializeAdapter(const std::string &adapter_type, const std::map<std::string, std::string> &options);
};
}  // namespace fe
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_ADAPTER_COMMON_OP_STORE_ADAPTER_MANAGER_H_

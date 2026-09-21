/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GE_LOCAL_ENGINE_OPS_KERNEL_STORE_OP_OP_FACTORY_H_
#define GE_GE_LOCAL_ENGINE_OPS_KERNEL_STORE_OP_OP_FACTORY_H_

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "common/plugin/ge_make_unique_util.h"
#include "engines/local_engine/ops_kernel_store/op/op.h"

namespace ge {
namespace ge_local {
using OP_CREATOR_FUNC = std::function<std::shared_ptr<Op>(const Node &, RunContext &)>;

/**
 * manage all the op, support create op.
 */
class GE_FUNC_VISIBILITY OpFactory {
 public:
  static OpFactory &Instance();

  /**
   *  @brief create Op.
   *  @param [in] node share ptr of node
   *  @param [in] run_context run context
   *  @return not nullptr success
   *  @return nullptr fail
   */
  std::shared_ptr<Op> CreateOp(const Node &node, RunContext &run_context);

  /**
   *  @brief Register Op create function.
   *  @param [in] type Op type
   *  @param [in] func Op create func
   */
  void RegisterCreator(const std::string &type, const OP_CREATOR_FUNC &func);

  const std::vector<std::string> &GetAllOps() const { return all_ops_; }

  OpFactory(const OpFactory &) = delete;

  OpFactory &operator=(const OpFactory &) = delete;

  OpFactory(OpFactory &&) = delete;

  OpFactory &operator=(OpFactory &&) = delete;

 private:
  OpFactory() = default;

  ~OpFactory() = default;

  // the op creator function map
  std::map<std::string, OP_CREATOR_FUNC> op_creator_map_;
  std::vector<std::string> all_ops_;
};

class GE_FUNC_VISIBILITY OpRegistrar {
 public:
  OpRegistrar(const std::string &type, const OP_CREATOR_FUNC &func) noexcept {
    OpFactory::Instance().RegisterCreator(type, func);
  }

  ~OpRegistrar() = default;

  OpRegistrar(const OpRegistrar &) = delete;

  OpRegistrar &operator=(const OpRegistrar &) = delete;

  OpRegistrar(OpRegistrar &&) = delete;

  OpRegistrar &operator=(OpRegistrar &&) = delete;
};

#define REGISTER_OP_CREATOR(type, clazz)                                              \
  std::shared_ptr<Op> Creator_##type##Op(const Node &node, RunContext &run_context) { \
    return MakeShared<clazz>(node, run_context);                                      \
  }                                                                                   \
  OpRegistrar g_##type##Op_creator(#type, Creator_##type##Op)
}  // namespace ge_local
}  // namespace ge

#endif  // GE_GE_LOCAL_ENGINE_OPS_KERNEL_STORE_OP_OP_FACTORY_H_

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef _RTS_ENGINE_OP_OP_FACTORY_H_
#define _RTS_ENGINE_OP_OP_FACTORY_H_

#include <string>
#include <memory>
#include <map>
#include <functional>

#include "op.h"

namespace cce {
namespace runtime {
using OP_CREATOR_FUNC = std::function<std::shared_ptr<Op>(const ge::Node &, ge::RunContext &)>;

/**
 * manage all the op, support create op.
 */
class OpFactory {
 public:
  static OpFactory &Instance();

  /**
   *  @brief create Op.
   *  @param [in] node share ptr of node
   *  @param [in] runContext run context
   *  @return not nullptr success
   *  @return nullptr fail
   */
  std::shared_ptr<Op> CreateOp(const ge::Node &node, ge::RunContext &runContext);

  /**
   *  @brief Register Op create function.
   *  @param [in] type Op type
   *  @param [in] func Op create func
   */
  void RegisterCreator(const std::string &type, const OP_CREATOR_FUNC &func);

  const std::vector<std::string> &GetAllOps() {
    return all_ops_;
  }

  OpFactory(const OpFactory &) = delete;

  OpFactory &operator=(const OpFactory &) = delete;

  OpFactory(OpFactory &&) = delete;

  OpFactory &operator=(OpFactory &&) = delete;

 private:
  OpFactory() = default;

  ~OpFactory() = default;

 private:
  // the op creator funtion map
  std::map<std::string, OP_CREATOR_FUNC> op_creator_map_;
  std::vector<std::string> all_ops_;
};

class OpRegistrar {
 public:
  OpRegistrar(const std::string &type, const OP_CREATOR_FUNC &func) {
    OpFactory::Instance().RegisterCreator(type, func);
  }

  ~OpRegistrar() = default;

  OpRegistrar(const OpRegistrar &) = delete;

  OpRegistrar &operator=(const OpRegistrar &) = delete;

  OpRegistrar(OpRegistrar &&) = delete;

  OpRegistrar &operator=(OpRegistrar &&) = delete;
};

#define REGISTER_OP_CREATOR(type, clazz)                                                            \
  static std::shared_ptr<Op> Creator_##type##Op(const ge::Node &node, ge::RunContext &runContext) { \
    try {                                                                                           \
      return std::make_shared<clazz>(node, runContext);                                             \
    } catch (...) {                                                                                 \
      return nullptr;                                                                               \
    }                                                                                               \
  }                                                                                                 \
  OpRegistrar g_##type##Op_creator(#type, Creator_##type##Op)
}  // namespace runtime
}  // namespace cce

#endif

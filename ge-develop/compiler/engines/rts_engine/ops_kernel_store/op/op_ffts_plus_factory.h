/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RTS_ENGINE_OPS_KERNEL_STORE_OP_OP_FFTS_PLUS_FACTORY_H_
#define RTS_ENGINE_OPS_KERNEL_STORE_OP_OP_FFTS_PLUS_FACTORY_H_

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
class OpFftsPlusFactory {
 public:
  static OpFftsPlusFactory &Instance();

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
    return allOps_;
  }

  OpFftsPlusFactory(const OpFftsPlusFactory &) = delete;

  OpFftsPlusFactory &operator=(const OpFftsPlusFactory &) = delete;

  OpFftsPlusFactory(OpFftsPlusFactory &&) = delete;

  OpFftsPlusFactory &operator=(OpFftsPlusFactory &&) = delete;

 private:
  OpFftsPlusFactory() = default;

  ~OpFftsPlusFactory() = default;

 private:
  // the op creator funtion map
  std::map<std::string, OP_CREATOR_FUNC> opCreatorMap_;
  std::vector<std::string> allOps_;
};

class OpFftsPlusRegistrar {
 public:
  OpFftsPlusRegistrar(const std::string &type, const OP_CREATOR_FUNC &func) {
    OpFftsPlusFactory::Instance().RegisterCreator(type, func);
  }

  ~OpFftsPlusRegistrar() = default;

  OpFftsPlusRegistrar(const OpFftsPlusRegistrar &) = delete;

  OpFftsPlusRegistrar &operator=(const OpFftsPlusRegistrar &) = delete;

  OpFftsPlusRegistrar(OpFftsPlusRegistrar &&) = delete;

  OpFftsPlusRegistrar &operator=(OpFftsPlusRegistrar &&) = delete;
};

#define REGISTER_OP_FFTS_PLUS_CREATOR(type, clazz)                                                   \
  std::shared_ptr<Op> Creator_##type##OpFftsPlus(const ge::Node &node, ge::RunContext &runContext) { \
    try {                                                                                            \
      return std::make_shared<clazz>(node, runContext);                                              \
    } catch (...) {                                                                                  \
      return nullptr;                                                                                \
    }                                                                                                \
  }                                                                                                  \
  OpFftsPlusRegistrar g_##type##OpFftsPlus_creator(#type, Creator_##type##OpFftsPlus)
}  // namespace runtime
}  // namespace cce

#endif

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_OPTIMIZE_SYMBOLIC_SYMBOLIC_KERNEL_FACTORY_H
#define AIR_CXX_COMPILER_GRAPH_OPTIMIZE_SYMBOLIC_SYMBOLIC_KERNEL_FACTORY_H
#include "graph/optimize/symbolic/symbol_compute_context.h"
#include "exe_graph/runtime/base_type.h"

namespace ge {
using InferSymbolComputeKernelFunc = UINT32 (*)(gert::InferSymbolComputeContext *);
class SymbolicKernelFactory {
 public:
  static SymbolicKernelFactory &GetInstance() {
    static SymbolicKernelFactory instance;
    return instance;
  }

  InferSymbolComputeKernelFunc Create(const std::string &op_type) const;

  class SymbolicKernelRegister {
   public:
    SymbolicKernelRegister(const std::string &op_type, const InferSymbolComputeKernelFunc &creator) noexcept {
      SymbolicKernelFactory::GetInstance().RegisterCreator(op_type, creator);
    }
    ~SymbolicKernelRegister() = default;
  };

 private:
  SymbolicKernelFactory() = default;
  ~SymbolicKernelFactory() = default;

  /**
   * Register a symbolic kernel
   * @param op_type   op type
   * @param builder   build function
   */
  Status RegisterCreator(const std::string &op_type, const InferSymbolComputeKernelFunc &creator);
  std::map<std::string, InferSymbolComputeKernelFunc> creators_;
};
}  // namespace ge

#define REGISTER_SYMBOLIC_KERNEL(op_type, func)                                \
  static SymbolicKernelFactory::SymbolicKernelRegister __attribute__((unused)) \
      g_symbolic_kernel_register_##op_type(#op_type, func)

#endif  // AIR_CXX_COMPILER_GRAPH_OPTIMIZE_SYMBOLIC_SYMBOLIC_KERNEL_FACTORY_H

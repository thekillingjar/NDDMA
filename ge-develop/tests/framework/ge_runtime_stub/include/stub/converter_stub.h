/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_SRC_CONVERTER_STUB_H_
#define AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_SRC_CONVERTER_STUB_H_
#include <string>
#include <vector>
#include "graph/compute_graph.h"
#include "register/node_converter_registry.h"
#include "optional_backup.h"
namespace gert {
class ConverterStub {
 public:
  ConverterStub() = default;
  explicit ConverterStub(ge::ComputeGraphPtr compute_graph) : compute_graph_(std::move(compute_graph)) {}
  ~ConverterStub();
  ConverterStub(const ConverterStub &) = delete;
  ConverterStub &operator=(const ConverterStub &) = delete;
  ConverterStub &Register(const std::string &key, NodeConverterRegistry::NodeConverter func);
  ConverterStub &Register(const std::string &key, NodeConverterRegistry::NodeConverter func, int32_t placement);
  void Clear();

 private:
  ge::ComputeGraphPtr compute_graph_;
  std::unordered_map<std::string, OptionalBackup<NodeConverterRegistry::ConverterRegisterData>> backup_;
};
}  // namespace gert

#endif  // AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_SRC_CONVERTER_STUB_H_

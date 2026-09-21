/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_STUB_TASK_INFO_REGISTRY_STUB_H_
#define AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_STUB_TASK_INFO_REGISTRY_STUB_H_
#include "graph/load/model_manager/task_info/task_info_factory.h"
#include "graph/utils/graph_utils.h"
namespace ge {
struct OpIndexAndIo {
  IOType io_type;
  int32_t io_index;
  int64_t node_index;

  bool operator==(const OpIndexAndIo &other) const {
    return io_type == other.io_type && io_index == other.io_index && node_index == other.node_index;
  }
};
}  // namespace ge
namespace std {
template <>
struct hash<ge::OpIndexAndIo> {
  size_t operator()(const ge::OpIndexAndIo &ins) const {
    return (static_cast<size_t>(ins.io_type) << 48) | (static_cast<size_t>(ins.io_index) << 32) |
           static_cast<size_t>(ins.node_index);
  }
};
}  // namespace std
namespace ge {
class TaskInfoRegistryStub {
 public:
  ~TaskInfoRegistryStub();
  TaskInfoRegistryStub &StubTaskInfo(ModelTaskType type, std::shared_ptr<TaskInfo> task_info_ins);
  template <typename T>
  TaskInfoRegistryStub &StubTaskInfo(ModelTaskType type) {
    return StubTaskInfo(type, [this]() { return std::make_shared<T>(this); });
  }

 public:
  using SymbolsToIOs = std::unordered_map<std::string, std::vector<OpIndexAndIo>>;
  using IOsToSymbol = std::unordered_map<OpIndexAndIo, std::string>;
  void SetSymbol(SymbolsToIOs symbols_to_ios, IOsToSymbol ios_to_symbol) {
    symbols_to_ios_ = std::move(symbols_to_ios);
    ios_to_symbol_ = std::move(ios_to_symbol);
  }
  const IOsToSymbol &GetIOsToSymbol() const {
    return ios_to_symbol_;
  }
  const SymbolsToIOs &GetSymbolsToIOs() const {
    return symbols_to_ios_;
  }

  using SymbolsToLogicalAddr = std::unordered_map<std::string, uint64_t>;
  void SetSymbolsToLogicalAddr(SymbolsToLogicalAddr symbols_to_addr) {
    symbols_to_addr_ = std::move(symbols_to_addr);
  }
  const SymbolsToLogicalAddr &GetSymbolsToAddr() const {
    return symbols_to_addr_;
  }

  using NodeNamesToWsAddrs = std::unordered_map<int64_t, std::vector<uint64_t>>;
  void SetWsAddrs(NodeNamesToWsAddrs node_names_to_ws_addrs) {
    node_names_to_ws_addrs_ = std::move(node_names_to_ws_addrs);
  }
  const NodeNamesToWsAddrs &GetWsAddrs() const {
    return node_names_to_ws_addrs_;
  }

 private:
  TaskInfoRegistryStub &StubTaskInfo(ModelTaskType type, TaskInfoFactory::TaskInfoCreatorFun creator);
  void StubRegistryIfNeed();
  static void TaskInfoFactoryDeleter(TaskInfoFactory *ins);

 private:
  std::shared_ptr<TaskInfoFactory> registry_;
  SymbolsToIOs symbols_to_ios_;
  IOsToSymbol ios_to_symbol_;
  SymbolsToLogicalAddr symbols_to_addr_;
  NodeNamesToWsAddrs node_names_to_ws_addrs_;
};
}  // namespace ge

#endif  // AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_STUB_TASK_INFO_REGISTRY_STUB_H_

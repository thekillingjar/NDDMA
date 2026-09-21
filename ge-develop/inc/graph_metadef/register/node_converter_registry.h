/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_LOWERING_NODE_CONVERTER_REGISTRY_H_
#define AIR_CXX_RUNTIME_V2_LOWERING_NODE_CONVERTER_REGISTRY_H_
#include <string>
#include <functional>
#include <vector>

#include "graph/node.h"
#include "graph/types.h"
#include "exe_graph/lowering/dev_mem_value_holder.h"
#include "exe_graph/lowering/lowering_global_data.h"
#include "common/hyper_status.h"
#include "exe_graph/lowering/value_holder.h"

namespace gert {
struct LowerInput {
  std::vector<bg::ValueHolderPtr> input_shapes;
  std::vector<bg::DevMemValueHolderPtr> input_addrs;
  LoweringGlobalData *global_data;
};
struct LowerResult {
  HyperStatus result;
  std::vector<bg::ValueHolderPtr> order_holders;
  std::vector<bg::ValueHolderPtr> out_shapes;
  std::vector<bg::DevMemValueHolderPtr> out_addrs;
};

struct LowerInputInfo {
  std::vector<bg::ValueHolderPtr> input_shapes;
  std::vector<bg::DevMemValueHolderPtr> input_addrs;
};

class NodeConverterRegistry {
 public:
  using NodeConverter = LowerResult (*)(const ge::NodePtr &node, const LowerInput &lower_input);
  struct ConverterRegisterData {
    NodeConverter converter;
    int32_t require_placement;
  };
  static NodeConverterRegistry &GetInstance();
  NodeConverter FindNodeConverter(const std::string &func_name);
  const ConverterRegisterData *FindRegisterData(const std::string &func_name) const;
  void RegisterNodeConverter(const std::string &func_name, NodeConverter func);
  void Register(const std::string &func_name, const ConverterRegisterData &data);

 private:
  std::unordered_map<std::string, ConverterRegisterData> names_to_register_data_;
};

class NodeConverterRegister {
 public:
  NodeConverterRegister(const ge::char_t *lower_func_name, NodeConverterRegistry::NodeConverter func) noexcept;
  NodeConverterRegister(const ge::char_t *lower_func_name, int32_t require_placement,
                        NodeConverterRegistry::NodeConverter func) noexcept;
};
}  // namespace gert

#ifdef __GNUC__
#define ATTRIBUTE_USED __attribute__((used))
#else
#define ATTRIBUTE_USED
#endif

#define GERT_REGISTER_NODE_CONVERTER_COUNTER2(type, placement, func, counter)                                          \
  static const gert::NodeConverterRegister g_register_node_converter_##counter ATTRIBUTE_USED =                        \
      gert::NodeConverterRegister(type, placement, func)
#define GERT_REGISTER_NODE_CONVERTER_COUNTER(type, placement, func, counter)                                           \
  GERT_REGISTER_NODE_CONVERTER_COUNTER2(type, placement, func, counter)
#define REGISTER_NODE_CONVERTER_PLACEMENT(type, placement, func)                                                       \
  GERT_REGISTER_NODE_CONVERTER_COUNTER(type, placement, func, __COUNTER__)
#define REGISTER_NODE_CONVERTER(type, func) REGISTER_NODE_CONVERTER_PLACEMENT(type, -1, func)

#endif  // AIR_CXX_RUNTIME_V2_LOWERING_NODE_CONVERTER_REGISTRY_H_

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_LOWERING_PLACEMENT_PLACED_LOWERING_REGISTER_H_
#define AIR_CXX_RUNTIME_V2_LOWERING_PLACEMENT_PLACED_LOWERING_REGISTER_H_

#include <functional>
#include <string>
#include <unordered_map>
#include "graph/node.h"
#include "exe_graph/lowering/lowering_global_data.h"
#include "graph/types.h"
#include "lowering/placement/placed_lowering_result.h"
namespace gert {
using PlacedLower = std::function<ge::graphStatus(
    const ge::Node* const_node, LoweringGlobalData &global_data, ge::DataType data_type,
    const OutputLowerResult &src, int64_t dst_logic_stream_id, OutputLowerResult &dst)>;
class PlacedLoweringRegistry {
 public:
  static PlacedLoweringRegistry &GetInstance();
  PlacedLower FindPlacedLower(const std::string &type);
  void Register(const std::string &type, PlacedLower func);

 private:
  std::unordered_map<std::string, PlacedLower> type_to_register_data_;
};
class PlacedLoweringRegister {
 public:
  PlacedLoweringRegister(const char *type, PlacedLower func) noexcept;
};
}  // namespace gert

#define REGISTER_OUTPUT_LOWER_COUNTER(type, func, counter) \
  static auto g_placed_lowering_register_##counter = gert::PlacedLoweringRegister(#type, func)
#define REGISTER_OUTPUT_LOWER(type, func) REGISTER_OUTPUT_LOWER_COUNTER(type, func, __COUNTER__)

#endif // AIR_CXX_RUNTIME_V2_LOWERING_PLACEMENT_PLACED_LOWERING_REGISTER_H_

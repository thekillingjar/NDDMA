/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_ASSIGN_DEVICE_MEM_H_
#define AIR_CXX_RUNTIME_V2_ASSIGN_DEVICE_MEM_H_
#include "exe_graph/lowering/value_holder.h"
#include "exe_graph/lowering/lowering_global_data.h"
#include "graph/ge_attr_value.h"
#include "graph/debug/ge_attr_define.h"
#include "common/debug/ge_log.h"
namespace gert {
constexpr uint32_t kFlattenKeySize = 2U;
inline std::vector<int64_t> GetFlattenOffsetInfo(const ge::AttrHolder* holder) {
  ge::GeAttrValue av;
  std::vector<int64_t> flatten_weight;
  if ((holder->GetAttr(ge::ATTR_NAME_GRAPH_FLATTEN_OFFSET, av) != ge::GRAPH_SUCCESS) ||
     (av.GetValue(flatten_weight) != ge::GRAPH_SUCCESS) ||
     (flatten_weight.size() != kFlattenKeySize)) {
    GELOGW("get flatten weight failed, flatten_weight size[%zu]", flatten_weight.size());
  }
  return flatten_weight;
}
class AssignDeviceMem {
  public:
  static bg::ValueHolderPtr GetBaseWeightAddr(LoweringGlobalData &global_data);
  static bg::ValueHolderPtr GetOrCreateMemAssigner(LoweringGlobalData &global_data);
};
}
#endif // AIR_CXX_RUNTIME_V2_ASSIGN_DEVICE_MEM_H_
 

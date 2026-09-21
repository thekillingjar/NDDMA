/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "slice/data_slice_factory.h"
#include "graph/operator_factory_impl.h"
#include "framework/common/debug/ge_log.h"

namespace ge {
std::shared_ptr<DataSliceInferBase> DataSliceFactory::GetClassByAxisType(ge::AxisType axis_type) {
  DataSliceInferBase *instance = nullptr;
  auto it = class_map_.find(axis_type);
  if (it != class_map_.end()) {
    instance = it->second();
  }
  if (instance != nullptr) {
    return std::shared_ptr<DataSliceInferBase>(instance);
  }
  return nullptr;
}

void DataSliceFactory::RegistClass(ge::AxisType axis_type, std::function<DataSliceInferBase*(void)> infer_util_class) {
  auto ret = class_map_.insert(std::pair<ge::AxisType,
      std::function<DataSliceInferBase*(void)>>(axis_type, infer_util_class));
  if (!ret.second) {
    GELOGW("[DataSlice][Status] Axis type %d has already registed.", static_cast<int8_t>(axis_type));
  }
}
}

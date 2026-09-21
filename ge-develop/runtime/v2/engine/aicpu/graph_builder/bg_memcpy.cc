/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bg_memcpy.h"

namespace gert {
namespace bg {
std::vector<ValueHolderPtr> GetHostSummary(const std::vector<ValueHolderPtr> &device_summary) {
  std::vector<ValueHolderPtr> inputs;
  inputs.insert(inputs.cend(), device_summary.cbegin(), device_summary.cend());
  return ValueHolder::CreateDataOutput("GetHostSummary", inputs, inputs.size());
}

std::vector<ValueHolderPtr> GetSummaryDataSizes(const std::vector<ValueHolderPtr> &summary) {
  std::vector<ValueHolderPtr> inputs;
  inputs.insert(inputs.cend(), summary.cbegin(), summary.cend());
  return ValueHolder::CreateDataOutput("GetSummaryDataSizes", inputs, inputs.size());
}

std::vector<ValueHolderPtr> GetSummaryShapeSizes(const std::vector<ValueHolderPtr> &summary) {
  std::vector<ValueHolderPtr> inputs;
  inputs.insert(inputs.cend(), summary.cbegin(), summary.cend());
  return ValueHolder::CreateDataOutput("GetSummaryShapeSizes", inputs, inputs.size());
}

std::vector<ValueHolderPtr> GetOutputShapeFromHbmBuffer(const std::vector<ValueHolderPtr> &summary,
                                                        const std::vector<DevMemValueHolderPtr> &shape_buffer) {
  std::vector<ValueHolderPtr> inputs;
  inputs.insert(inputs.cend(), summary.cbegin(), summary.cend());
  inputs.insert(inputs.cend(), shape_buffer.cbegin(), shape_buffer.cend());
  return ValueHolder::CreateDataOutput("GetOutputShapeFromHbmBuffer", inputs, shape_buffer.size());
}
} // bg
} // gert

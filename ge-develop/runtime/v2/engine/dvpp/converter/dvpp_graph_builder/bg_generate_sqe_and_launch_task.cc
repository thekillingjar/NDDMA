/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bg_generate_sqe_and_launch_task.h"

namespace gert {
namespace bg {
ValueHolderPtr GenerateSqeAndLaunchTask(
    const std::vector<ValueHolderPtr> &input_shapes,
    const std::vector<ValueHolderPtr> &output_shapes,
    const std::vector<ValueHolderPtr> &addrs_info,
    const ValueHolderPtr stream) {
  std::vector<ValueHolderPtr> inputs;
  inputs.insert(inputs.end(), input_shapes.cbegin(), input_shapes.cend());
  inputs.insert(inputs.end(), output_shapes.cbegin(), output_shapes.cend());
  inputs.insert(inputs.end(), addrs_info.cbegin(), addrs_info.cend());
  inputs.emplace_back(stream);

  return ValueHolder::CreateSingleDataOutput("GenerateSqeAndLaunchTask",
                                             inputs);
}
} // namespace bg
} // namespace gert

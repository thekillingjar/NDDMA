/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_FAKER_KERNEL_OUTPUTS_FAKER_H_
#define AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_FAKER_KERNEL_OUTPUTS_FAKER_H_
#include <vector>
#include "exe_graph/runtime/kernel_context.h"
namespace gert {
struct OutputsHolder {
  std::vector<Chain> outputs;
  std::vector<void *> pointer;

  static OutputsHolder Fake(size_t num) {
    OutputsHolder holder;
    holder.outputs.resize(num);
    holder.pointer.reserve(num);
    for (size_t i = 0; i < num; ++i) {
      holder.outputs[i].Set(nullptr, nullptr);
      holder.pointer.push_back(&holder.outputs[i]);
    }

    return holder;
  }

  ~OutputsHolder() {
    for (auto &output : outputs) {
      output.Set(nullptr, nullptr);
    }
  }
};
}
#endif  // AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_FAKER_KERNEL_OUTPUTS_FAKER_H_

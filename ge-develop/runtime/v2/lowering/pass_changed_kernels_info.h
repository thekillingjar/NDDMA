/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_LOWERING_PASS_CHANGED_KERNELS_INFO_H_
#define AIR_CXX_RUNTIME_V2_LOWERING_PASS_CHANGED_KERNELS_INFO_H_
#include <string>
#include <vector>
#include <cstdint>
namespace gert {
struct KernelNameAndIdx {
  std::string kernel_name;
  int32_t idx;
  std::string launch_name = "";
  bool operator==(const KernelNameAndIdx &rhs) const {
    return ((rhs.idx == idx) && (rhs.kernel_name == kernel_name));
  }
};

struct PassChangedKernels {
  // first in pair is src, second in pair is dst
  std::vector<std::pair<KernelNameAndIdx, KernelNameAndIdx>> pass_changed_kernels;
};
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_LOWERING_PASS_CHANGED_KERNELS_INFO_H_

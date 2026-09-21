/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "node_bin_selector_factory.h"

#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/debug/ge_log.h"
namespace ge {
NodeBinSelectorFactory &NodeBinSelectorFactory::GetInstance() {
  static NodeBinSelectorFactory factory;
  return factory;
}
NodeBinSelector *NodeBinSelectorFactory::GetNodeBinSelector(const fuzz_compile::NodeBinMode node_bin_type) {
  if (node_bin_type < fuzz_compile::kNodeBinModeEnd) {
    return selectors_[node_bin_type].get();
  } else {
    GELOGE(FAILED, "node bin type %u is out of range.", static_cast<uint32_t>(node_bin_type));
    return nullptr;
  }
}
NodeBinSelectorFactory::NodeBinSelectorFactory() = default;
}
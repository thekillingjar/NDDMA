/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DVPP_ENGINE_COMMON_DVPP_CHIP_CAPABILITY_H_
#define DVPP_ENGINE_COMMON_DVPP_CHIP_CAPABILITY_H_

#include "common/dvpp_ops.h"
#include "graph/node.h"

namespace dvpp {
class DvppChipCapability {
 public:
  /**
   * @brief check if dvpp chip capablity support the op
   * @param dvpp_op_info op info in dvpp ops lib
   * @param node node pointer
   * @param unsupported_reason if not support, need to write reason
   * @return whether support the op
   */
  static bool CheckSupported(const DvppOpInfo& dvpp_op_info,
      const ge::NodePtr& node, std::string& unsupported_reason);
}; // class DvppChipCapability
} // namespace dvpp

#endif // DVPP_ENGINE_COMMON_DVPP_CHIP_CAPABILITY_H_

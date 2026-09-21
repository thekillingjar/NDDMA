/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_OP_TILING_TILING_MEMCHECK_H_
#define GE_COMMON_OP_TILING_TILING_MEMCHECK_H_

#include <string>
#include <mutex>
#include <memory>
#include "graph/op_desc.h"
#include "graph/utils/args_format_desc_utils.h"
#include "register/op_tiling_registry.h"

namespace optiling {
class TilingMemCheck {
 public:
  static ge::graphStatus ConstructMemCheckInfo(const ge::OpDescPtr &op_desc,
                                               const OpRunInfoV2 &run_info,
                                               const std::vector<ge::ArgDesc> &arg_descs,
                                               std::string &memcheck_info);
};
}  // namespace optiling
#endif  // GE_COMMON_OP_TILING_TILING_MEMCHECK_H_
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_EXECUTOR_HYBRID_COMMON_BIN_CACHE_BIN_CACHE_DEF_H_
#define AIR_CXX_EXECUTOR_HYBRID_COMMON_BIN_CACHE_BIN_CACHE_DEF_H_
#include <cstdint>
namespace ge {
namespace fuzz_compile {
enum NodeBinMode : std::int32_t
{
  kOneNodeSingleBinMode,
  kOneNodeMultipleBinsMode,
  kNodeBinModeEnd
};
}
}
#endif // AIR_CXX_EXECUTOR_HYBRID_COMMON_BIN_CACHE_BIN_CACHE_DEF_H_

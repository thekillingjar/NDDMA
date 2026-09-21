/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RESOURCE_DEF_H__
#define RESOURCE_DEF_H__

#include "runtime/rt.h"

namespace fe {
/**
 * @ingroup fe
 * @brief function parameter type
 */
enum class FuncParamType {
  FUSION_L2,
  GLOBAL_MEMORY_CLEAR,
  MAX_NUM,
};

/**
 * @ingroup fe
 * @brief set function point state
 */
bool SetFunctionState(FuncParamType type, bool isOpen);

/**
 * @ingroup cce
 * @brief cce get function point state
 */
bool GetFunctionState(FuncParamType type);

}  // namespace fe
#endif  // RESOURCE_DEF_H__

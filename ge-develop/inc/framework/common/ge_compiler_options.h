/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_FRAMEWORK_COMMON_GE_COMPILER_OPTIONS_H_
#define INC_FRAMEWORK_COMMON_GE_COMPILER_OPTIONS_H_

namespace ge {
#ifdef __GNUC__
#define GE_ATTRIBUTE_UNUSED __attribute__((unused))
#define GE_FUNCTION_IDENTIFIER __PRETTY_FUNCTION__
#define GE_BUILTIN_PREFETCH(args_addr) __builtin_prefetch(args_addr)
#else
#define GE_ATTRIBUTE_UNUSED
#define GE_FUNCTION_IDENTIFIER __FUNCSIG__
#define GE_BUILTIN_PREFETCH(args_addr)
#endif
}  // namespace ge

#endif  // INC_FRAMEWORK_COMMON_GE_COMPILER_OPTIONS_H_

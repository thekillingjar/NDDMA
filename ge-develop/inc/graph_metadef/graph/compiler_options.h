/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_GRAPH_COMPILER_OPTIONS_H_
#define INC_GRAPH_COMPILER_OPTIONS_H_

#ifdef __GNUC__
#define METADEF_ATTRIBUTE_UNUSED __attribute__((unused))
#define METADEF_FUNCTION_IDENTIFIER __PRETTY_FUNCTION__
#define METADEF_BUILTIN_PREFETCH(args_addr) __builtin_prefetch(args_addr)

#ifdef HOST_VISIBILITY
#define GE_FUNC_HOST_VISIBILITY __attribute__((visibility("default")))
#else
#define GE_FUNC_HOST_VISIBILITY
#endif

#ifdef DEV_VISIBILITY
#define GE_FUNC_DEV_VISIBILITY __attribute__((visibility("default")))
#else
#define GE_FUNC_DEV_VISIBILITY
#endif

#else // WINDOWS
#define METADEF_ATTRIBUTE_UNUSED
#define METADEF_FUNCTION_IDENTIFIER __FUNCSIG__
#define METADEF_BUILTIN_PREFETCH(args_addr)
#define GE_FUNC_HOST_VISIBILITY
#define GE_FUNC_DEV_VISIBILITY
#endif

#endif  // INC_GRAPH_COMPILER_OPTIONS_H_

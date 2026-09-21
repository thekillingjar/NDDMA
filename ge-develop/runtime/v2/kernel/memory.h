/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_KERNEL_MEMORY_H_
#define AIR_CXX_RUNTIME_V2_KERNEL_MEMORY_H_

#include "graph/ge_error_codes.h"
#include "exe_graph/runtime/kernel_context.h"

namespace gert {
namespace kernel {
/*
 * inputs: size
 * outputs: addr
 */
ge::graphStatus AllocHbmMem(KernelContext *context);

/*
 * inputs: addr
 * outputs: none
 */
ge::graphStatus FreeHbmMem(KernelContext *context);
/*
 * inputs: size,...
 * outputs: address,...
 */
ge::graphStatus BatchAllocHbmMem(KernelContext *context);

/*
 * inputs: adress,...
 * outputs: none
 */
ge::graphStatus BatchFreeHbmMem(KernelContext *context);
}
}
#endif  // AIR_CXX_RUNTIME_V2_KERNEL_MEMORY_H_

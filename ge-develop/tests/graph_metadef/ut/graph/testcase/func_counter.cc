/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "func_counter.h"
namespace ge {
size_t FuncCounter::construct_times = 0;
size_t FuncCounter::copy_construct_times = 0;
size_t FuncCounter::move_construct_times = 0;
size_t FuncCounter::copy_assign_times = 0;
size_t FuncCounter::move_assign_times = 0;
size_t FuncCounter::destruct_times = 0;
}  // namespace ge

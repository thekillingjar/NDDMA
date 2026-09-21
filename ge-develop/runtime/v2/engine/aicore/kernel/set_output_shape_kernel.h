/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_KERNEL_SET_OUTPUT_SHAPE_KERNEL_H_
#define AIR_CXX_RUNTIME_V2_KERNEL_SET_OUTPUT_SHAPE_KERNEL_H_
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tensor.h"
#include "exe_graph/runtime/compute_node_info.h"
#include "runtime/rt_ffts_plus_define.h"
#include "runtime/rt_ffts_plus.h"
#include "runtime/mem.h"
#include "kernel/kernel_log.h"
#include "kernel/memory/mem_block.h"
#include "kernel/memory/ffts_mem_allocator.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "engine/aicore/fe_rt2_common.h"
#include "kernel/common_kernel_impl/infer_shape.h"
namespace gert {
namespace kernel {
ge::graphStatus SetOutputShape(KernelContext *context);
}  // namespace kernel
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_KERNEL_SET_OUTPUT_SHAPE_KERNEL_H_
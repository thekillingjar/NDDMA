/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_KERNEL_FFTS_PLUS_MIXL2_UPDATE_KERNEL_H_
#define AIR_CXX_RUNTIME_V2_KERNEL_FFTS_PLUS_MIXL2_UPDATE_KERNEL_H_
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tensor.h"
#include "exe_graph/runtime/compute_node_info.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "register/ffts_node_calculater_registry.h"
namespace gert {
namespace kernel {

enum class MixL2UpdateKey {
  FLUSH_DATA = 0,
  BLOCK_DIM,
  SCHEDULE_MODE,
  TASK_INFO,
  CTX_ID,
  MIX_RATIO = 5,
  TILING_KEY,
  TILING_VEC,
  CRATIO_VEC,
  VRATIO_VEC,
  RESERVED = 10
};

enum class MixL2ArgsInKey {
  WORKSPACE = 0,
  NEED_MODE_ADDR = 1,
  SINK_RET = 2,
  SHAPE_BUFFER = 3,
  HAS_ASSERT = 4,
  IO_ADDR_NUM = 5,
  IO_START = 6,
  kNUM
};
enum class MixL2ArgsOutKey {
  FLUSH_DATA = 0,
  ARGS_PARA = 1,
  kNUM
};

// data dump
enum class MixL2DataDumpKey {
  CONTEXT_ID = 0,
  IN_NUM,
  OUT_NUM,
  /**
  * IO_START memory layer:
  *     IO_ADDRS: input_addrs and output_addrs tensors
  */
  IO_START,
  RESERVED
};

// Exception dump
enum class MixL2ExceptionDumpKey {
  WORKSPACE = 0,
  ARGS_PARA,
  RESERVED
};

}
}
#endif  // AIR_CXX_RUNTIME_V2_KERNEL_FFTS_PLUS_MIXL2_UPDATE_KERNEL_H_

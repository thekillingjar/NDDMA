/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_OP_TILING_TILING_DFX_H_
#define GE_COMMON_OP_TILING_TILING_DFX_H_

#include <string>
#include <mutex>
#include <memory>
#include <map>
#include "graph/op_desc.h"
#include "graph/utils/args_format_desc_utils.h"
#include "register/op_tiling_registry.h"
#include "exe_graph/runtime/shape.h"

namespace optiling {
enum class ArgsRole : int32_t {
  kInput = 0,
  kOutput = 1,
};

struct ArgsIndexToIoIndex {
  ArgsRole args_role;
  size_t args_index;
  size_t io_index;
};

class TilingDfx {
 public:
  static ge::graphStatus GetArgsSizeWithArgsFormat(const ge::OpDescPtr &op_desc,
                                                   const std::vector<ge::ArgDesc> &arg_descs,
                                                   std::vector<int64_t> &args_size_list,
                                                   std::vector<ArgsIndexToIoIndex> &args_index_to_io_index);

  static ge::graphStatus GetArgsSizeWithoutArgsFormat(size_t input_size,
                                                      size_t output_size,
                                                      std::vector<int64_t> &args_size_list,
                                                      std::vector<ArgsIndexToIoIndex> &args_index_to_io_index);
};
}  // namespace optiling
#endif  // GE_COMMON_OP_TILING_TILING_MEMCHECK_H_
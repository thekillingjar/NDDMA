/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSDATA_OP_IMPL_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSDATA_OP_IMPL_H_
#include <cstdint>
#include "exe_graph/runtime/shape.h"
#include "graph/types.h"
#include "common/table_driven.h"
#include "register/op_compile_info_base.h"
#include "exe_graph/runtime/kernel_context.h"
#include "graph/ge_error_codes.h"

namespace gert {
namespace kernel {
struct TransDataCompileInfo : public optiling::CompileInfoBase {
  int64_t ub_size;
  int64_t block_dim;
  int64_t group;
  int64_t vnc_fp32_flag;
};
ge::graphStatus TilingForTransData(KernelContext *context);
}
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_TRANSDATA_OP_IMPL_H_

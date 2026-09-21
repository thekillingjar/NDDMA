/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_INC_EXE_GRAPH_RUNTIME_GETCDIM_H_
#define METADEF_CXX_INC_EXE_GRAPH_RUNTIME_GETCDIM_H_
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/tiling_context.h"
namespace gert {
  int64_t GetInputCDim(gert::TilingContext *kernel_context, const size_t index);
  int64_t GetOutputCDim(gert::TilingContext *kernel_context, const size_t index);
}  // namespace gert
#endif  // METADEF_CXX_INC_EXE_GRAPH_RUNTIME_TILING_PARSE_CONTEXT_H_

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/ge_error_codes.h"
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/tiling_context.h"

namespace gert {
namespace kernel {
struct StubAtomicCleanTilingData {
  uint64_t tiling_data[8];
};
// todo impl atomic clean
ge::graphStatus TilingForAtomicClean(TilingContext *context) {
  context->GetTilingData<StubAtomicCleanTilingData>();
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus TilingParseForAtomicClean(KernelContext *context) {
  return ge::GRAPH_SUCCESS;
}
IMPL_OP(AtomicClean).Tiling(TilingForAtomicClean).TilingParse<StubAtomicCleanTilingData>(TilingParseForAtomicClean);
}  // namespace kernel
}  // namespace gert

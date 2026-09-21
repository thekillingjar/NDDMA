/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstddef>
#include "graph/ge_error_codes.h"
#include "exe_graph/runtime/kernel_context.h"

struct AfTilingParseData {
  uint32_t aiv_num;
  uint32_t ub_size;
};

class TilingSymbolEvalContext;
class InferShapeSymbolEvalContext;
class SymbolTilingParseContext;
extern "C" uint32_t TilingFunc(TilingSymbolEvalContext *context);
extern "C" size_t GetTilingDataSize();
extern "C" ge::graphStatus InferShape(InferShapeSymbolEvalContext *context);
extern "C" ge::graphStatus GetSymbolTilingCacheKey(TilingSymbolEvalContext *context);
extern "C" ge::graphStatus TilingParse(SymbolTilingParseContext *context);
extern "C" ge::graphStatus DfxInputSymbolInfo(TilingSymbolEvalContext *context, char *out_symbol_info, size_t size);
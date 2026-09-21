/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_BENCHMARK_RUNTIME_ASM_ASM_TEST_H_
#define AIR_CXX_TESTS_BENCHMARK_RUNTIME_ASM_ASM_TEST_H_
#include "exe_graph/runtime/kernel_context.h"
#include "exe_graph/runtime/extended_kernel_context.h"
#include "exe_graph/runtime/tiling_context.h"
namespace gert_asm {
using namespace gert;
const Chain *GetInputFromContext(KernelContext *context, size_t index);
Chain *GetOutputFromContext(KernelContext *context, size_t index);
Chain *GetOutputFromContext2(KernelContext *context, size_t index);
const void *GetExtendInfoFromContext(KernelContext *context);

// todo infershape/tiling context/extended context
void *GetAttrsFromContext(ExtendedKernelContext *context, size_t index);
Chain *GetAttrsObjFromContext(ExtendedKernelContext *context, size_t index);
struct TestTilingData {
  int64_t a;
  int64_t b;
  int64_t c;
};

struct TestData {
  int64_t a;
  int32_t b;
  int16_t c;
  int16_t d;
};
TestData *GetTilingData(TilingContext *context);
TilingData *GetRawTilingData(TilingContext *context);
const void *GetCompileInfo(TilingContext *context);

const uint64_t *GetInlineInputPointerFromContext(KernelContext *context, size_t index);
uint64_t *GetInlineOutputPointerFromContext(KernelContext *context, size_t index);
const TestData *GetAllocInputPointerFromContext(KernelContext *context, size_t index);
TestData *GetAllocOutputPointerFromContext(KernelContext *context, size_t index);
}  // namespace gert

#endif  //AIR_CXX_TESTS_BENCHMARK_RUNTIME_ASM_ASM_TEST_H_

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "asm_test.h"
namespace gert_asm {
const Chain *GetInputFromContext(KernelContext *context, size_t index) {
  return context->GetInput(index);
}
Chain *GetOutputFromContext(KernelContext *context, size_t index) {
  return context->GetOutput(index);
}
Chain *GetOutputFromContext2(KernelContext *context, size_t index) {
  return context->GetOutput2(index);
}
const void *GetExtendInfoFromContext(KernelContext *context) {
  return context->GetComputeNodeExtend();
}
const uint64_t *GetInlineInputPointerFromContext(KernelContext *context, size_t index) {
  return context->GetInputPointer<uint64_t>(index);
}
uint64_t *GetInlineOutputPointerFromContext(KernelContext *context, size_t index) {
  return context->GetOutputPointer<uint64_t>(index);
}
const TestData *GetAllocInputPointerFromContext(KernelContext *context, size_t index) {
  return context->GetInputPointer<TestData>(index);
}
TestData *GetAllocOutputPointerFromContext(KernelContext *context, size_t index) {
  return context->GetOutputPointer<TestData>(index);
}
TestData *GetTilingData(TilingContext *context) {
  return context->GetTilingData<TestData>();
}
TilingData *GetRawTilingData(TilingContext *context) {
  return context->GetRawTilingData();
}
const void *GetCompileInfo(TilingContext *context) {
  return context->GetCompileInfo();
}
}

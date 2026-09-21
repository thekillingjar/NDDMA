/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/load/model_manager/device_memory_ptr.h"
#include <gtest/gtest.h>
#include "stub/gert_runtime_stub.h"
#include "runtime/mem.h"
namespace ge {
namespace {
uint64_t AllocDevice() {
  void *ptr = nullptr;
  rtMalloc(&ptr, 1024, RT_MEMORY_HBM, 0);
  return PtrToValue(ptr);
}
}  // namespace
class DeviceMemoryPtrUT : public testing::Test {};
TEST_F(DeviceMemoryPtrUT, ConstructAndDeconstruct_Ok_EmptyPtr) {
  DeviceMemoryPtr mem;
  ASSERT_EQ(mem.Get(), 0UL);
}
TEST_F(DeviceMemoryPtrUT, ConstructAndDeconstruct_Ok_WithPtr) {
  gert::GertRuntimeStub runtime_stub;
  auto ptr = AllocDevice();
  DeviceMemoryPtr mem(ptr);
  ASSERT_EQ(mem.Get(), ptr);
}
TEST_F(DeviceMemoryPtrUT, Reset_Ok) {
  gert::GertRuntimeStub runtime_stub;
  DeviceMemoryPtr mem;
  ASSERT_EQ(mem.Get(), 0UL);
  auto ptr = AllocDevice();
  mem.Reset(ptr);
  ASSERT_EQ(mem.Get(), ptr);
}
TEST_F(DeviceMemoryPtrUT, Reset_AutoFree) {
  gert::GertRuntimeStub runtime_stub;
  DeviceMemoryPtr mem;
  ASSERT_EQ(mem.Get(), 0UL);

  auto ptr = AllocDevice();
  mem.Reset(ptr);
  ASSERT_EQ(mem.Get(), ptr);

  ptr = AllocDevice();
  mem.Reset(ptr);
  ASSERT_EQ(mem.Get(), ptr);
}
TEST_F(DeviceMemoryPtrUT, MoveConstruct_Ok) {
  gert::GertRuntimeStub runtime_stub;
  auto ptr = AllocDevice();
  DeviceMemoryPtr mem1(ptr);
  DeviceMemoryPtr mem2(std::move(mem1));
  ASSERT_EQ(mem2.Get(), ptr);
}
TEST_F(DeviceMemoryPtrUT, MoveAssignment_Ok) {
  gert::GertRuntimeStub runtime_stub;
  DeviceMemoryPtr mem1;
  auto ptr = AllocDevice();
  DeviceMemoryPtr mem2(ptr);
  mem1 = std::move(mem2);
  ASSERT_EQ(mem1.Get(), ptr);
}
TEST_F(DeviceMemoryPtrUT, MoveAssignment_AutoFree) {
  gert::GertRuntimeStub runtime_stub;
  DeviceMemoryPtr mem(AllocDevice());
  auto ptr = AllocDevice();
  DeviceMemoryPtr mem2(ptr);
  mem = std::move(mem2);
  ASSERT_EQ(mem.Get(), ptr);
}
}  // namespace ge

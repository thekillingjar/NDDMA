/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "kernel/memory/span/span_layer_id.h"

using namespace gert;

struct BaseTypeTest : public testing::Test {
 protected:
  size_t pageIdem{16};
  MemSize pageSize{1 << 16};
};

TEST_F(BaseTypeTest, mem_size) {
  ASSERT_EQ(0,            MemSize_GetAlignedOf(0, pageSize));
  ASSERT_EQ(pageSize,     MemSize_GetAlignedOf(1, pageSize));
  ASSERT_EQ(pageSize,     MemSize_GetAlignedOf(pageSize - 1, pageSize));
  ASSERT_EQ(pageSize * 2, MemSize_GetAlignedOf(pageSize + 1, pageSize));
  ASSERT_EQ(pageSize * 2, MemSize_GetAlignedOf(2 * pageSize - 1, pageSize));
  ASSERT_EQ(pageSize * 3, MemSize_GetAlignedOf(2 * pageSize + 1, pageSize));
}

TEST_F(BaseTypeTest, span_layer_id_from_size) {
  ASSERT_EQ(0,       SpanLayerId_GetIdFromSize(0, pageIdem));
  ASSERT_EQ(0,       SpanLayerId_GetIdFromSize(64_KB - 1, pageIdem));
  ASSERT_EQ(1,       SpanLayerId_GetIdFromSize(64_KB, pageIdem));
  ASSERT_EQ(1,       SpanLayerId_GetIdFromSize(64_KB + 1, pageIdem));
  ASSERT_EQ(2,       SpanLayerId_GetIdFromSize(128_KB, pageIdem));
  ASSERT_EQ(15,      SpanLayerId_GetIdFromSize(1_MB - 1, pageIdem));
  ASSERT_EQ(16,      SpanLayerId_GetIdFromSize(1_MB, pageIdem));
  ASSERT_EQ(16,      SpanLayerId_GetIdFromSize(1_MB + 1, pageIdem));
  ASSERT_EQ(1023,    SpanLayerId_GetIdFromSize(64_MB - 1, pageIdem));
  ASSERT_EQ(1024,    SpanLayerId_GetIdFromSize(64_MB, pageIdem));
  ASSERT_EQ(1024,    SpanLayerId_GetIdFromSize(64_MB + 1, pageIdem));
  ASSERT_EQ(1 << 18, SpanLayerId_GetIdFromSize(16_GB, pageIdem));
}

TEST_F(BaseTypeTest, span_layer_id_from_aligned_size) {
  ASSERT_EQ(0,       SpanLayerId_GetIdFromSize(MemSize_GetAlignedOf(0, pageSize), pageIdem));
  ASSERT_EQ(1,       SpanLayerId_GetIdFromSize(MemSize_GetAlignedOf(1, pageSize), pageIdem));
  ASSERT_EQ(1,       SpanLayerId_GetIdFromSize(MemSize_GetAlignedOf(64_KB-1, pageSize), pageIdem));
  ASSERT_EQ(1,       SpanLayerId_GetIdFromSize(MemSize_GetAlignedOf(64_KB, pageSize), pageIdem));
  ASSERT_EQ(2,       SpanLayerId_GetIdFromSize(MemSize_GetAlignedOf(64_KB+1, pageSize), pageIdem));
  ASSERT_EQ(2,       SpanLayerId_GetIdFromSize(MemSize_GetAlignedOf(128_KB-1, pageSize), pageIdem));
  ASSERT_EQ(2,       SpanLayerId_GetIdFromSize(MemSize_GetAlignedOf(128_KB, pageSize), pageIdem));
  ASSERT_EQ(3,       SpanLayerId_GetIdFromSize(MemSize_GetAlignedOf(128_KB+1, pageSize), pageIdem));
}

TEST_F(BaseTypeTest, page_len_get_mem_size) {
  ASSERT_EQ(0,       PageLen_GetMemSize(0, pageIdem));
  ASSERT_EQ(64_KB,   PageLen_GetMemSize(1, pageIdem));
  ASSERT_EQ(128_KB,  PageLen_GetMemSize(2, pageIdem));
  ASSERT_EQ(64_MB,   PageLen_GetMemSize(1024, pageIdem));
}

TEST_F(BaseTypeTest, page_len_from_size) {
  ASSERT_EQ(0,       PageLen_GetLenFromSize(0, pageIdem));
  ASSERT_EQ(0,       PageLen_GetLenFromSize(1, pageIdem));
  ASSERT_EQ(0,       PageLen_GetLenFromSize(64_KB-1, pageIdem));
  ASSERT_EQ(1,       PageLen_GetLenFromSize(64_KB, pageIdem));
  ASSERT_EQ(1,       PageLen_GetLenFromSize(64_KB + 1, pageIdem));
  ASSERT_EQ(1,       PageLen_GetLenFromSize(128_KB - 1, pageIdem));
}

TEST_F(BaseTypeTest, page_len_forward_addr) {
  MemAddr baseAddr = (MemAddr)0x10001000;

  ASSERT_EQ(baseAddr,                    PageLen_ForwardAddr(0, pageIdem, baseAddr));
  ASSERT_EQ(baseAddr + pageSize,         PageLen_ForwardAddr(1, pageIdem, baseAddr));
  ASSERT_EQ(baseAddr + pageSize * 2,     PageLen_ForwardAddr(2, pageIdem, baseAddr));
  ASSERT_EQ(baseAddr + pageSize * 1024000,  PageLen_ForwardAddr(1024000, pageIdem, baseAddr));
}

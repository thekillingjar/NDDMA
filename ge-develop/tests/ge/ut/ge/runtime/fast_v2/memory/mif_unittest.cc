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
#include "kernel/memory/mif.h"

namespace gert {
namespace memory {
class MifUT : public testing::Test {
 public:
  size_t stream_num_ = 8U;
};
TEST_F(MifUT, Set_Clear_success) {
  Mif mif;
  mif.ReInit(stream_num_);
  int64_t maintained_stream = 0;
  int64_t stream = 1;
  ASSERT_EQ(mif.IsSet(maintained_stream, stream), false);
  mif.Set(maintained_stream, stream);
  ASSERT_EQ(mif.IsSet(maintained_stream, stream), true);
  mif.Clear(maintained_stream, stream);
  ASSERT_EQ(mif.IsSet(maintained_stream, stream), false);
}

TEST_F(MifUT, SetAll_Success) {
  Mif mif;
  mif.ReInit(stream_num_);
  int64_t maintained_stream = 0;
  mif.SetAll(maintained_stream);
  for (size_t i = 0U; i < stream_num_; i++) {
    ASSERT_EQ(mif.IsSet(i, maintained_stream), true);
  }
}

TEST_F(MifUT, SetMaintainedAll_Success) {
  Mif mif;
  mif.ReInit(stream_num_);
  int64_t maintained_stream = 0;
  mif.SetMaintainedAll(maintained_stream);
  for (size_t i = 0U; i < stream_num_; i++) {
    ASSERT_EQ(mif.IsSet(maintained_stream, i), true);
  }
}

TEST_F(MifUT, MergeClearedFrom_Success) {
  Mif mif;
  mif.ReInit(stream_num_);
  int64_t src_stream = 0;
  int64_t dst_stream = 1;
  mif.SetMaintainedAll(src_stream);
  for (size_t i = 0U; i < stream_num_; i++) {
    ASSERT_EQ(mif.IsSet(src_stream, i), true);
  }
  ASSERT_EQ(mif.IsAnySet(dst_stream), false);
  mif.MergeClearedFrom(src_stream, dst_stream);
  for (size_t i = 0U; i < stream_num_; i++) {
    ASSERT_EQ(mif.IsSet(dst_stream, i), false);
  }
}
}  // namespace memory
}  // namespace gert
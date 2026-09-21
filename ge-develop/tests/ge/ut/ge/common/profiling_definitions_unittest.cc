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
#include "common/profiling_definitions.h"
#include "depends/profiler/src/profiling_test_util.h"

namespace ge {
class ProfilingUt : public testing::Test {
 protected:
  void TearDown() override {
    ProfilingTestUtil::Instance().Clear();
  }
};

TEST_F(ProfilingUt, DefaultRegisteredString) {
  profiling::ProfilingContext pc;
  EXPECT_EQ(strcmp(pc.GetProfiler()->GetStringHashes()[profiling::kInferShape].str, "InferShape"), 0);
  EXPECT_EQ(strcmp(pc.GetProfiler()->GetStringHashes()[profiling::kAllocMem].str, "AllocMemory"), 0);
  for (auto i = 0; i < profiling::kProfilingIndexEnd; ++i) {
    EXPECT_NE(strlen(pc.GetProfiler()->GetStringHashes()[i].str), 0);
  }
  EXPECT_EQ(strlen(pc.GetProfiler()->GetStringHashes()[profiling::kProfilingIndexEnd].str), 0);
}

TEST_F(ProfilingUt, RegisteredStringHash) {
  profiling::ProfilingContext pc;
  const std::string &str = __FUNCTION__;
  const uint64_t test_hash_id = 0x5aU;
  const int64_t idx = pc.RegisterStringHash(test_hash_id, str);
  EXPECT_EQ(idx != 0, true);
}

TEST_F(ProfilingUt, RecordOne) {
  profiling::ProfilingContext::GetInstance().SetEnable();
  PROFILING_START(-1, profiling::kInferShape);
  PROFILING_END(-1, profiling::kInferShape);
  EXPECT_EQ(profiling::ProfilingContext::GetInstance().GetProfiler()->GetRecordNum(), 2);
}

TEST_F(ProfilingUt, RegisterStringBeforeRecord) {
  auto index = profiling::ProfilingContext::GetInstance().RegisterString("Data");
  EXPECT_EQ(strcmp(profiling::ProfilingContext::GetInstance().GetProfiler()->GetStringHashes()[index].str, "Data"), 0);
}

TEST_F(ProfilingUt, DumpWhenDisable) {
  profiling::ProfilingContext::GetInstance().Reset();
  profiling::ProfilingContext::GetInstance().SetDisable();
  PROFILING_START(-1, profiling::kInferShape);
  PROFILING_END(-1, profiling::kInferShape);
  EXPECT_EQ(profiling::ProfilingContext::GetInstance().GetProfiler()->GetRecordNum(), 0);
  std::stringstream ss;
  profiling::ProfilingContext::GetInstance().Dump(ss);
  profiling::ProfilingContext::GetInstance().Reset();
  EXPECT_EQ(ss.str(), std::string("Profiling not enable, skip to dump\n"));
}
TEST_F(ProfilingUt, ScopeRecordOk) {
  profiling::ProfilingContext::GetInstance().SetEnable();
  {
    PROFILING_SCOPE(-1, profiling::kInferShape);
  }
  EXPECT_EQ(profiling::ProfilingContext::GetInstance().GetProfiler()->GetRecordNum(), 2);
}

TEST_F(ProfilingUt, UpdateElementHashId) {
  profiling::ProfilingContext pc;
  pc.UpdateElementHashId();
  EXPECT_NE(0, pc.GetProfiler()->GetStringHashes()[profiling::kProfilingIndexEnd-1].hash);
}

/* just for coverage */
TEST_F(ProfilingUt, IsDumpToStdEnabled) {
  EXPECT_TRUE(profiling::ProfilingContext::GetInstance().IsDumpToStdEnabled());
}
TEST_F(ProfilingUt, DumpToStdOut) {
  EXPECT_NO_THROW(profiling::ProfilingContext::GetInstance().SetEnable());
  EXPECT_NO_THROW(profiling::ProfilingContext::GetInstance().DumpToStdOut());
}
}

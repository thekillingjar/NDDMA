/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/utils/dumper/ge_graph_dumper.h"
#include <gtest/gtest.h>

namespace ge {
class GeGraphDumperUt : public testing::Test {};

namespace {
int64_t dump_times = 0;
class TestDumper : public GeGraphDumper {
 public:
  void Dump(const ComputeGraphPtr &graph, const string &suffix) override {
    ++dump_times;
  }
};
}

TEST_F(GeGraphDumperUt, DefaultImpl)  {
  dump_times = 0;
  GraphDumperRegistry::Unregister();
  GraphDumperRegistry::GetDumper().Dump(nullptr, "test");
  EXPECT_EQ(dump_times, 0);
}

TEST_F(GeGraphDumperUt, RegisterOk)  {
  dump_times = 0;
  TestDumper dumper;
  GraphDumperRegistry::Unregister();
  GraphDumperRegistry::Register(dumper);
  GraphDumperRegistry::GetDumper().Dump(nullptr, "test");
  EXPECT_EQ(dump_times, 1);
  GraphDumperRegistry::GetDumper().Dump(nullptr, "test");
  EXPECT_EQ(dump_times, 2);
}

TEST_F(GeGraphDumperUt, UnregisterOk)  {
  dump_times = 0;
  TestDumper dumper;
  GraphDumperRegistry::Register(dumper);
  GraphDumperRegistry::GetDumper().Dump(nullptr, "test");
  EXPECT_EQ(dump_times, 1);
  GraphDumperRegistry::GetDumper().Dump(nullptr, "test");
  EXPECT_EQ(dump_times, 2);

  GraphDumperRegistry::Unregister();
  GraphDumperRegistry::GetDumper().Dump(nullptr, "test");
  EXPECT_EQ(dump_times, 2);
  GraphDumperRegistry::GetDumper().Dump(nullptr, "test");
  EXPECT_EQ(dump_times, 2);
}
}

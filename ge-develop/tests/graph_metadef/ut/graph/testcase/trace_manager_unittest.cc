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
#include <memory>
#include <fstream>
#include <vector>
#include <iostream>
#include "mmpa/mmpa_api.h"
#include "common/util/trace_manager/trace_manager.h"

using namespace std;

namespace {
size_t GetFileLinesNum(const std::string fn) {
  std::ifstream f;
  f.open(fn, std::ios::in);
  size_t num = 0U;
  if (f.fail()) {
    return 0U;
  } else {
    std::string s;
    while (std::getline(f, s)) {
      num++;
    }
    f.close();
  }
  return num;
}
}  // namespace

namespace ge {
class UtestTraceManager : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestTraceManager, add_trace_basic_0) {
  auto &instance = TraceManager::GetInstance();
  instance.ClearTraceOwner();
  instance.SetTraceOwner("a", "b", "c");
  instance.trace_index_ = 0;
  EXPECT_EQ(instance.Initialize("."), SUCCESS);
  instance.enabled_ = true;
  instance.ClearTraceOwner();
  instance.SetTraceOwner("a", "b", "c");
  EXPECT_EQ(instance.trace_header_, "a:b");
  EXPECT_EQ(instance.graph_name_, "c");
  instance.AddTrace("0");
  EXPECT_EQ(instance.trace_index_, 1);
  for (int i = 0; i < 10000; i++) {
    instance.AddTrace(std::to_string(i + 1));
  }
  instance.Finalize();
  EXPECT_EQ(instance.current_file_saved_nums_, 10001);
  EXPECT_EQ(GetFileLinesNum(instance.current_saving_file_name_), 10001);
  std::string pre_file_name = instance.current_saving_file_name_;
  instance.stopped_ = false;
  instance.current_file_saved_nums_ = 2000000U + 1U;
  EXPECT_EQ(instance.Initialize("."), SUCCESS);

  for (int i = 0; i < 100; i++) {
    instance.AddTrace(std::to_string(i + 1));
  }
  instance.Finalize();

  EXPECT_NE(pre_file_name, instance.current_saving_file_name_);
  remove(instance.current_saving_file_name_.c_str());
  remove(pre_file_name.c_str());
}
}  // namespace ge

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_ST_STUBS_UTILS_PROFILING_TEST_DATA_H_
#define AIR_CXX_TESTS_ST_STUBS_UTILS_PROFILING_TEST_DATA_H_
#include <string>
#include <sstream>

#include <gtest/gtest.h>

#include "framework/common/profiling_definitions.h"
#include "framework/common/string_util.h"

namespace ge {
inline std::vector<std::string> SplitLines(const std::string &str) {
  auto lines = StringUtils::Split(str, '\n');
  std::vector<std::string> final_lines;
  for (const auto &line : lines) {
    final_lines.emplace_back(StringUtils::Split(line, '\r')[0]);
  }
  return final_lines;
}
class ProfilingData {
 public:
  ProfilingData() {
    std::stringstream ss;
    profiling::ProfilingContext::GetInstance().Dump(ss);

    lines_ = SplitLines(ss.str());
  }

  ProfilingData &HasRecord(int64_t event) {
    std::string event_name;
    auto index_to_strrings = profiling::ProfilingContext::GetInstance().GetProfiler()->GetStringHashes();
    if (event < 0 || index_to_strrings[event].str == 0) {
      event_name = "UNKNOWN(" + std::to_string(event) + ")";
    } else {
      event_name = index_to_strrings[event].str;
    }

    int32_t found_times = 0;
    for (const auto &line : lines_) {
      if (line.find(event_name) == std::string::npos) {
        continue;
      }
      if (line.find("Start") == std::string::npos && line.find("End") == std::string::npos) {
        continue;
      }
      if (++found_times >= 2) {
        return *this;
      }
    }
    std::cout << "Failed to find evnet " << event_name << std::endl;
    EXPECT_TRUE(false);
    return *this;
  }
 private:

 private:
  std::vector<std::string> lines_;
};
}
#endif //AIR_CXX_TESTS_ST_STUBS_UTILS_PROFILING_TEST_DATA_H_

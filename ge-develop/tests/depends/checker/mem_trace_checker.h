/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_MEM_TRACE_CHECKER_H_
#define AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_MEM_TRACE_CHECKER_H_
#include <utility>
#include <unordered_map>
#include "memory_profiling_log_matcher.h"

namespace gert {
namespace {
constexpr const char *kMemKernelTrace = R"(\[KernelTrace]\[(.*)]\[MEM](.*)address (0x\w+))";
using SingleExpectGroup = std::vector<std::pair<int32_t, std::string>>;
const std::unordered_map<std::string, SingleExpectGroup> kRegexs2CheckerTemplates = {
    {kAllocRe, {{2, "0"}}},
    {kWander, {{2, "0"}, {3, "1"}}},
    {kFreeRe, {{2, "0"}}},
    {kLocalRecycleRe, {{2, "0"}}},
    {kSendEventWithMem, {{3, "0"}}},
    {kWaitEventWithMem, {{3, "1"}}},
    {kBorrowRecycleRe, {{2, "1"}}},
    {kBirthRecycleRe, {{2, "0"}}}
};

struct RegexsAndChecker {
  std::string regex;
  SingleExpectGroup expect_groups;

  bool IsLogMatches(const std::string &target_log) {
    std::smatch matches;
    std::regex res(regex.c_str());
    if (std::regex_search(target_log, matches, res) && IsGroupMatch(matches, expect_groups)) {
      return true;
    }
    return false;
  }

 private:
  bool IsGroupMatch(const std::smatch &matches, const SingleExpectGroup &single_expect_groups) {
    for (const auto &index_and_str : single_expect_groups) {
      if (index_and_str.first >= (int32_t)matches.size()) {
        throw std::range_error("Index out of range");
      }
      if (index_and_str.second != matches[index_and_str.first].str()) {
        return false;
      }
    }
    return true;
  }
};

std::vector<std::string> CollectAddrMemLog(const SlogStubImpl *slog_stub, const std::string &addr) {
  std::vector<std::string> target_logs;
  std::regex reg(kMemKernelTrace);
  std::smatch matches;
  for (const auto &onelog : slog_stub->GetLogs(DLOG_INFO)) {
    if (std::regex_search(onelog.content, matches, reg) && (onelog.content.find(addr.c_str()) != std::string::npos)) {
      target_logs.emplace_back(onelog.content);
      std::cout<<"addr mem log:" << onelog.content << std::endl;
    }
  }
  return target_logs;
}

} //namespace
class MemoryTraceChecker {
 public:
  struct EventWithStream {
    std::string event;
    int64_t logic_stream_id;
  };
  struct EventWithStreams {
    std::string event;
    int64_t src_logic_stream_id;
    int64_t dst_logic_stream_id;
  };
  MemoryTraceChecker() = delete;
  MemoryTraceChecker(const SlogStubImpl &slog_stub, const void *addr) {
    addr_ = ToHex(addr);
    slog_stub_ = const_cast<SlogStubImpl *>(&slog_stub);
  }

  MemoryTraceChecker &AppendExpectEvent(const std::string &event, int64_t logic_stream_id) {
    auto iter = kRegexs2CheckerTemplates.find(event);
    if (iter == kRegexs2CheckerTemplates.cend()) {
      std::cout << "Can not find Checker Template of event: " << event.c_str() << std::endl;
      EXPECT_TRUE(false);
      return *this;
    }
    auto expect_pairs = iter->second;
    EXPECT_TRUE(expect_pairs.size() == 1U);
    expect_pairs[0].second = logic_stream_id;
    checkers_.emplace_back(RegexsAndChecker({event, expect_pairs}));
    return *this;
  }
  MemoryTraceChecker &AppendExpectEventWithSrc(const std::string &event, int64_t src_logic_stream_id,
                                               int64_t dst_logic_stream_id) {
    auto expect_pairs = kRegexs2CheckerTemplates.at(event);
    EXPECT_TRUE(expect_pairs.size() == 2U);

    expect_pairs[0].second = src_logic_stream_id;
    expect_pairs[1].second = dst_logic_stream_id;
    checkers_.emplace_back(RegexsAndChecker({event, expect_pairs}));
    return *this;
  }
  bool AsYouWish() {
    std::stringstream error_msg;
    const auto target_logs = CollectAddrMemLog(slog_stub_, addr_);
    if (target_logs.size() != checkers_.size()) {
      error_msg << "Target logs has " << target_logs.size() << ", while Expected logs size is " << checkers_.size() << std::endl;
      std::cout << error_msg.str() << std::endl;
      return false;
    }

    for (size_t i = 0U; i < target_logs.size(); ++i) {
      std::cout << "index" << i << std::endl;
      if (checkers_[i].IsLogMatches(target_logs[i])) {
        error_msg << "Index: " << i << "Target logs not as expect: " << target_logs[i] << std::endl;
        std::cout << error_msg.str() << std::endl;
        return false;
      }
    }
    return true;
  }

 private:
  std::string addr_;
  SlogStubImpl *slog_stub_;
  std::vector<RegexsAndChecker> checkers_;
};
}  // namespace gert
#endif  // AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_MEM_TRACE_CHECKER_H_

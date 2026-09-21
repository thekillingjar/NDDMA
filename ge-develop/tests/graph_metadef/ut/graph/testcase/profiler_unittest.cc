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
#include "graph/utils/profiler.h"
namespace ge {
namespace profiling {
namespace {

std::string RandomStr(const int len) {
  std::string res(len,' ');
  for(int i = 0; i < len; ++i) {
    res[i] = 'A' + rand() % 26;
  }
  return res;
}


std::string FindNext(const std::string &s, size_t &pos) {
  std::stringstream ss;
  for (; pos < s.size(); ++pos) {
    if (s[pos] == '\r') {
      ++pos;
      if (pos + 1 < s.size() && s[pos + 1] == '\n') {
        ++pos;
      }
      return ss.str();
    }
    if (s[pos] == '\n') {
      ++pos;
      if (pos + 1 < s.size() && s[pos + 1] == '\r') {
        ++pos;
      }
      return ss.str();
    }
    ss << s[pos];
  }
  return ss.str();
}
std::vector<std::string> SplitLines(const std::string &s) {
  std::vector<std::string> strings;
  size_t i = 0;
  while (i < s.size()) {
    strings.emplace_back(FindNext(s, i));
  }
  return strings;
}
std::vector<std::string> Split(const std::string &s, std::string spliter) {
  std::vector<std::string> strings;
  size_t i = 0;
  while (i < s.size()) {
    auto pos = s.find_first_of(spliter, i);
    if (pos == std::string::npos) {
      strings.emplace_back(s, i);
      break;
    } else {
      strings.emplace_back(s, i, pos - i);
      i = pos + spliter.size();
    }
  }

  return strings;
}
}
class ProfilerUt : public testing::Test {};

TEST_F(ProfilerUt, OneRecord) {
  auto p = Profiler::Create();
  p->Record(0, 1, 2, EventType::kEventStart, std::chrono::system_clock::now());
  EXPECT_EQ(p->GetRecordNum(), 1);

  std::stringstream ss;
  p->Dump(ss);
  auto lines = SplitLines(ss.str());
  EXPECT_EQ(lines.size(), 3);
  auto elements = Split(lines[1], " ");
  EXPECT_EQ(elements.size(), 5);
  EXPECT_EQ(elements[1], "1");
  EXPECT_EQ(elements[2], "UNKNOWN(0)");
  EXPECT_EQ(elements[3], "UNKNOWN(2)");
  EXPECT_EQ(elements[4], "Start");
}

TEST_F(ProfilerUt, TimeStampRecord) {
  auto p = Profiler::Create();
  p->Record(0, 1, 2, EventType::kEventTimestamp, std::chrono::system_clock::now());

  std::stringstream ss;
  p->Dump(ss);
  auto lines = SplitLines(ss.str());
  EXPECT_EQ(lines.size(), 3);
  auto elements = Split(lines[1], " ");
  EXPECT_EQ(elements.size(), 4);
  EXPECT_EQ(elements[1], "1");
  EXPECT_EQ(elements[2], "UNKNOWN(0)");
  EXPECT_EQ(elements[3], "UNKNOWN(2)");
}

TEST_F(ProfilerUt, MultipleRecords) {
  auto p = Profiler::Create();
  p->Record(0, 1, 2, EventType::kEventStart, std::chrono::system_clock::now());
  p->Record(0, 1, 2, EventType::kEventEnd, std::chrono::system_clock::now());
  EXPECT_EQ(p->GetRecordNum(), 2);

  std::stringstream ss;
  p->Dump(ss);
  auto lines = SplitLines(ss.str());
  EXPECT_EQ(lines.size(), 4);

  auto elements = Split(lines[1], " ");
  EXPECT_EQ(elements.size(), 5);
  EXPECT_EQ(elements[1], "1");
  EXPECT_EQ(elements[2], "UNKNOWN(0)");
  EXPECT_EQ(elements[3], "UNKNOWN(2)");
  EXPECT_EQ(elements[4], "Start");

  elements = Split(lines[2], " ");
  EXPECT_EQ(elements.size(), 5);
  EXPECT_EQ(elements[1], "1");
  EXPECT_EQ(elements[2], "UNKNOWN(0)");
  EXPECT_EQ(elements[3], "UNKNOWN(2)");
  EXPECT_EQ(elements[4], "End");
}

TEST_F(ProfilerUt, RecordStr) {
  auto p = Profiler::Create();
  p->RegisterString(0, "Node1");
  p->RegisterString(2, "InferShape");
  p->Record(0, 1, 2, EventType::kEventStart, std::chrono::system_clock::now());

  std::stringstream ss;
  p->Dump(ss);
  auto lines = SplitLines(ss.str());
  EXPECT_EQ(lines.size(), 3);
  auto elements = Split(lines[1], " ");
  EXPECT_EQ(elements.size(), 5);
  EXPECT_EQ(elements[1], "1");
  EXPECT_EQ(elements[2], "[Node1]");
  EXPECT_EQ(elements[3], "[InferShape]");
  EXPECT_EQ(elements[4], "Start");
}

TEST_F(ProfilerUt, RecordCurrentThread) {
  auto p = Profiler::Create();
  p->RecordCurrentThread(0, 2, EventType::kEventStart, std::chrono::system_clock::now());
  p->RecordCurrentThread(0, 2, EventType::kEventEnd);

  std::stringstream ss;
  p->Dump(ss);
  auto lines = SplitLines(ss.str());
  EXPECT_EQ(lines.size(), 4);

  auto elements = Split(lines[1], " ");
  EXPECT_EQ(elements.size(), 5);
  EXPECT_EQ(elements[2], "UNKNOWN(0)");
  EXPECT_EQ(elements[3], "UNKNOWN(2)");
  EXPECT_EQ(elements[4], "Start");

  elements = Split(lines[2], " ");
  EXPECT_EQ(elements.size(), 5);
  EXPECT_EQ(elements[2], "UNKNOWN(0)");
  EXPECT_EQ(elements[3], "UNKNOWN(2)");
  EXPECT_EQ(elements[4], "End");
}

TEST_F(ProfilerUt, Reset) {
  auto p = Profiler::Create();
  p->RegisterString(0, "Node1");
  p->RegisterString(2, "InferShape");
  p->Record(0, 1, 2, EventType::kEventStart, std::chrono::system_clock::now());
  p->Reset();
  std::stringstream ss;
  p->Dump(ss);
  auto lines = SplitLines(ss.str());
  EXPECT_EQ(lines.size(), 0);
}

TEST_F(ProfilerUt, ResetRemainsRegisteredString) {
  auto p = Profiler::Create();
  p->RegisterString(0, "Node1");
  p->RegisterString(2, "InferShape");
  p->Record(0, 1, 2, EventType::kEventStart, std::chrono::system_clock::now());
  p->Reset();
  std::stringstream ss;
  p->Dump(ss);
  auto lines = SplitLines(ss.str());
  EXPECT_EQ(lines.size(), 0);


  p->Record(0, 1, 2, EventType::kEventStart, std::chrono::system_clock::now());
  ss = std::stringstream();
  p->Dump(ss);
  lines = SplitLines(ss.str());
  EXPECT_EQ(lines.size(), 3);
  auto elements = Split(lines[1], " ");
  EXPECT_EQ(elements.size(), 5);
  EXPECT_EQ(elements[1], "1");
  EXPECT_EQ(elements[2], "[Node1]");
  EXPECT_EQ(elements[3], "[InferShape]");
  EXPECT_EQ(elements[4], "Start");
}

TEST_F(ProfilerUt, RegisterStringBeyondMaxSize) {
  auto p = Profiler::Create();
  p->RegisterString(2, "InferShape");
  p->RegisterString(kMaxStrIndex, "[Node1]");
  p->Record(kMaxStrIndex, 1, 2, EventType::kEventStart, std::chrono::system_clock::now());

  std::stringstream ss;
  p->Dump(ss);
  auto lines = SplitLines(ss.str());
  EXPECT_EQ(lines.size(), 3);
  auto elements = Split(lines[1], " ");
  EXPECT_EQ(elements.size(), 5);
  EXPECT_EQ(elements[1], "1");
  EXPECT_EQ(elements[2], "UNKNOWN(" + std::to_string(kMaxStrIndex) + ")");
  EXPECT_EQ(elements[3], "[InferShape]");
  EXPECT_EQ(elements[4], "Start");
}

TEST_F(ProfilerUt, EventTypeBeyondRange) {
  auto p = Profiler::Create();
  p->Record(0, 1, 2, EventType::kEventTypeEnd, std::chrono::system_clock::now());

  std::stringstream ss;
  p->Dump(ss);
  auto lines = SplitLines(ss.str());
  EXPECT_EQ(lines.size(), 3);
  auto elements = Split(lines[1], " ");
  EXPECT_EQ(elements.size(), 5);
  EXPECT_EQ(elements[1], "1");
  EXPECT_EQ(elements[2], "UNKNOWN(0)");
  EXPECT_EQ(elements[3], "UNKNOWN(2)");
  EXPECT_EQ(elements[4], "UNKNOWN(3)");
}

TEST_F(ProfilerUt, GetRecords) {
  auto p = Profiler::Create();
  p->Record(0, 1, 2, EventType::kEventTypeEnd, std::chrono::system_clock::now());
  auto rec = p->GetRecords();
  EXPECT_EQ(rec->element, 0);
  EXPECT_EQ(rec->thread, 1);
  EXPECT_EQ(rec->event, 2);
  EXPECT_EQ(rec->et, EventType::kEventTypeEnd);
}

TEST_F(ProfilerUt, GetStringHashes) {
  auto p = Profiler::Create();
  p->RegisterString(0, "Node1");
  p->RegisterString(2, "InferShape");
  auto s = p->GetStringHashes();
  EXPECT_EQ(strcmp(s[0].str, "Node1"), 0);
  EXPECT_EQ(strcmp(s[2].str, "InferShape"), 0);
  p->RegisterStringHash(3, 0x55, "Node2");
  p->RegisterStringHash(4, 0xaa, "Tiling");
  EXPECT_EQ(s[3].hash == 0x55, true);
  EXPECT_EQ(s[4].hash == 0xaa, true);
  p->UpdateHashByIndex(0, 0x5a);
  p->UpdateHashByIndex(2, 0xa5);
  EXPECT_EQ(s[0].hash == 0x5a, true);
  EXPECT_EQ(s[2].hash == 0xa5, true);
}

TEST_F(ProfilerUt, RegisterTooLongString) {
  auto p = Profiler::Create();
  std::string input = RandomStr(300);
  std::string gt_res = input.substr(0,255);
  p->RegisterString(0, input);
  p->RegisterString(2, "InferShape");
  auto s = p->GetStringHashes();
  EXPECT_EQ(strcmp(s[0].str, gt_res.c_str()), 0);
  EXPECT_EQ(strcmp(s[2].str, "InferShape"), 0);
}

TEST_F(ProfilerUt, ModifyStrings) {
  auto p = Profiler::Create();
  p->RegisterString(0, "AbcdefghijklmnopqrstuvwxyzAbcdefghijklmnopqrstuvwxyzAbcdefghijklmnopqrstuvwxyz");
  p->RegisterString(2, "InferShape");
  auto s = p->GetStringHashes();
  strcpy(s[2].str, "Tiling");
  EXPECT_EQ(strcmp(s[0].str, "AbcdefghijklmnopqrstuvwxyzAbcdefghijklmnopqrstuvwxyzAbcdefghijklmnopqrstuvwxyz"), 0);
  EXPECT_EQ(strcmp(p->GetStringHashes()[2].str, "Tiling"), 0);
}

/* takes very long time
TEST_F(ProfilerUt, BeyondMaxRecordsNum) {
  auto p = Profiler::Create();
  for (int64_t i = 0; i < profiling::kMaxRecordNum; ++i) {
    p->Record(0, 1, i, kEventStart);
    p->Record(0, 1, i, kEventEnd);
  }

  std::stringstream ss;
  p->Dump(ss);
  auto lines = SplitLines(ss.str());
  EXPECT_EQ(lines.size(), profiling::kMaxRecordNum + 3);
}
*/
}
}

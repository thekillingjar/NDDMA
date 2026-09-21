/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/screen_printer.h"

#include <gtest/gtest.h>
#include <thread>
#include "graph/ge_context.h"
#include "graph/ge_local_context.h"
#include "mmpa/mmpa_api.h"
#include "depends/mmpa/src/mmpa_stub.h"

namespace ge {
namespace {
constexpr const char *kFormatTime = "[2023-08-08-20:08:00.001.001]";

int32_t system_time_ret = 0;
int32_t time_of_day_ret = 0;
class MockMmpa : public ge::MmpaStubApiGe {
 public:
  INT32 mmGetSystemTime(mmSystemTime_t *sysTime) override{
    if (system_time_ret == -1) {
      return EN_ERR;
    }
    sysTime->wYear = 2023;
    sysTime->wMonth = 8;
    sysTime->wDay = 8;
    sysTime->wHour = 20;
    sysTime->wMinute = 8;
    sysTime->wSecond = 0;
    return EN_OK;
  }
  INT32 mmGetTimeOfDay(mmTimeval *timeVal, mmTimezone *timeZone) override {
    if (time_of_day_ret == -1) {
      return EN_ERR;
    }
    timeVal->tv_usec = 1001;
    timeVal->tv_sec = 1001;
    return EN_OK;
  }
};
}
class UtestScreenPrinter : public testing::Test {
 protected:
  void SetUp() {
    MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  }

  void TearDown() {
    MmpaStub::GetInstance().Reset();
  }
};

TEST_F(UtestScreenPrinter, log_ok) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  std::stringstream ss;
  std::streambuf *coutbuf = std::cout.rdbuf();
  std::cout.rdbuf(ss.rdbuf());

  std::string option = "input_shape_range";
  SCREEN_LOG("Option %s is deprecated", option.c_str());

  std::cout.rdbuf(coutbuf);
  std::string out_log = ss.str();
  std::string expect_log = kFormatTime + std::to_string(mmGetTid()) + " Option input_shape_range is deprecated" + "\n";
  EXPECT_EQ(out_log, expect_log);
  MmpaStub::GetInstance().Reset();
}

TEST_F(UtestScreenPrinter, multi_thread_log_ok) {
  std::stringstream ss;
  std::streambuf *coutbuf = std::cout.rdbuf();
  std::cout.rdbuf(ss.rdbuf());

  auto func = [](char c) -> void {
    std::string option(100, c);
    SCREEN_LOG("%s", option.c_str());
  };

  char a = 'a';
  char b = 'b';
  char c = 'c';
  std::thread t1(func, a);
  std::thread t2(func, b);
  std::thread t3(func, c);
  t1.join();
  t2.join();
  t3.join();
  std::cout.rdbuf(coutbuf);
  std::string out_log = ss.str();

  std::string expect_a(100, a);
  std::string expect_b(100, b);
  std::string expect_c(100, c);
  std::unordered_set<std::string> expect_set;
  std::string tmp;
  while(getline(ss, tmp)) {
    if (tmp.find(expect_a) != std::string::npos) {
      expect_set.emplace(expect_a);
    } else if (tmp.find(expect_b) != std::string::npos) {
      expect_set.emplace(expect_b);
    } else if (tmp.find(expect_c) != std::string::npos) {
      expect_set.emplace(expect_c);
    }
  }
  EXPECT_EQ(expect_set.size(), 3);
}

TEST_F(UtestScreenPrinter, log_len_ok) {
  std::stringstream ss;
  std::streambuf *coutbuf = std::cout.rdbuf();
  std::cout.rdbuf(ss.rdbuf());

  std::string option(1024, 'a');
  SCREEN_LOG("%s", option.c_str());
  std::cout.rdbuf(coutbuf);
  std::string expect_log = kFormatTime + std::to_string(mmGetTid()) + " " + option + "\n";
  EXPECT_EQ(ss.str(), expect_log);
}

TEST_F(UtestScreenPrinter, log_len_over) {
  std::stringstream ss;
  std::streambuf *coutbuf = std::cout.rdbuf();
  std::cout.rdbuf(ss.rdbuf());

  std::string option(1025, 'a');
  SCREEN_LOG("%s", option.c_str());
  std::cout.rdbuf(coutbuf);
  std::string expect_log = "";
  EXPECT_EQ(ss.str(), expect_log);
}

TEST_F(UtestScreenPrinter, fmt_nullptr) {
  std::stringstream ss;
  std::streambuf *coutbuf = std::cout.rdbuf();
  std::cout.rdbuf(ss.rdbuf());

  SCREEN_LOG(nullptr);

  std::cout.rdbuf(coutbuf);
  std::string out_log = ss.str();
  std::string expect_log = "";
  EXPECT_EQ(out_log, expect_log);
}

TEST_F(UtestScreenPrinter, log_time_err) {
  std::stringstream ss;
  std::streambuf *coutbuf = std::cout.rdbuf();
  std::cout.rdbuf(ss.rdbuf());

  system_time_ret = -1;
  std::string option = "input_shape_range";
  SCREEN_LOG("Option %s is deprecated", option.c_str());
  system_time_ret = 0;

  time_of_day_ret = -1;
  SCREEN_LOG("Option %s is deprecated", option.c_str());
  time_of_day_ret = 0;

  std::cout.rdbuf(coutbuf);
  std::string out_log = ss.str();
  std::string expect_log = "";
  expect_log += expect_log;
  EXPECT_EQ(out_log, expect_log);
}

TEST_F(UtestScreenPrinter, log_disable) {
  std::stringstream ss;
  std::streambuf *coutbuf = std::cout.rdbuf();
  std::cout.rdbuf(ss.rdbuf());

  std::map<std::string, std::string> options;
  options.emplace("ge.screen_print_mode", "disable");
  ScreenPrinter::GetInstance().Init(options["ge.screen_print_mode"]);

  std::string option = "input_shape_range";
  SCREEN_LOG("Option %s is deprecated", option.c_str());

  std::cout.rdbuf(coutbuf);
  std::string out_log = ss.str();
  std::string expect_log = "";
  EXPECT_EQ(out_log, expect_log);
  GetThreadLocalContext().SetGlobalOption(std::map<std::string, std::string>{});
}

TEST_F(UtestScreenPrinter, log_ensable) {
  std::stringstream ss;
  std::streambuf *coutbuf = std::cout.rdbuf();
  std::cout.rdbuf(ss.rdbuf());

  std::map<std::string, std::string> options;
  options.emplace("ge.screen_print_mode", "enable");
  ScreenPrinter::GetInstance().Init(options["ge.screen_print_mode"]);

  std::string option = "input_shape_range";
  SCREEN_LOG("Option %s is deprecated", option.c_str());

  std::cout.rdbuf(coutbuf);
  std::string out_log = ss.str();
  std::string expect_log = kFormatTime + std::to_string(mmGetTid()) + " Option input_shape_range is deprecated" + "\n";
  EXPECT_EQ(out_log, expect_log);
  GetThreadLocalContext().SetGlobalOption(std::map<std::string, std::string>{});
}
}

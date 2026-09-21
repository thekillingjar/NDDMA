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
#include <fstream>
#include "macro_utils/dt_public_scope.h"
#include "common/utils/memory_statistic_manager.h"
#include "macro_utils/dt_public_unscope.h"

using namespace std;
namespace ge {
class MemoryStatisticManagerTest : public testing::Test {
 protected:
  void SetUp() override {
    ResetStatistic();
  }
  void TearDown() override {
    ResetStatistic();
  }
  void ResetStatistic() {
    auto &inst = MemoryStatisticManager::Instance();
    inst.Finalize();
    inst.rss_avg_ = 0;
    inst.rss_hwm_ = 0;
    inst.rss_statistic_count_ = 0;
    inst.xsmem_avg_ = 0;
    inst.xsmem_hwm_ = 0;
    inst.xsmem_statistic_count_ = 0;
  }
};

TEST_F(MemoryStatisticManagerTest, InitAndFinalize) {
  auto &inst = MemoryStatisticManager::Instance();
  inst.Initialize("DM_QS_GROUP_12300");
  EXPECT_TRUE(inst.thread_run_flag_);
  inst.Finalize();
  EXPECT_FALSE(inst.thread_run_flag_);
}

TEST_F(MemoryStatisticManagerTest, StatisticRss_file_not_exits) {
  auto &inst = MemoryStatisticManager::Instance();
  inst.process_status_file_name_ = "./not_exist_file.xxx";
  inst.StatisticRss();
  EXPECT_EQ(inst.rss_statistic_count_, 0);
  EXPECT_EQ(static_cast<uint64_t>(inst.rss_avg_), 0);
  EXPECT_EQ(inst.rss_hwm_, 0);
}

TEST_F(MemoryStatisticManagerTest, StatisticRss_avoid_division_by_zero) {
  auto &inst = MemoryStatisticManager::Instance();
  inst.process_status_file_name_ = MemoryStatisticManager::GetProcessStatusFile();
  inst.rss_statistic_count_ = UINT32_MAX;
  inst.StatisticRss();
  EXPECT_EQ(inst.rss_statistic_count_, 1);
  EXPECT_GE(inst.rss_hwm_, inst.rss_avg_);
}

TEST_F(MemoryStatisticManagerTest, StatisticRss_normal) {
  std::string file_name = "./test_process_status";
  {
    std::ofstream of(file_name);
    of << "Name:\ttest" << std::endl;
    of << "VmHWM:	   13568 kB" << std::endl;
    of << "VmRSS:	   13500 kB" << std::endl;
    of << "RssAnon:	    5204 kB" << std::endl;
    of << "RssFile:	    8364 kB" << std::endl;
    of << "RssShmem:	       0 kB" << std::endl;
  }

  auto &inst = MemoryStatisticManager::Instance();
  inst.process_status_file_name_ = file_name;
  inst.StatisticRss();
  EXPECT_EQ(inst.rss_statistic_count_, 1);
  EXPECT_EQ(static_cast<uint64_t>(inst.rss_avg_), 13500);
  EXPECT_EQ(inst.rss_hwm_, 13568);
  remove(file_name.c_str());
}

TEST_F(MemoryStatisticManagerTest, StatisticXsmem_file_not_exits) {
  auto &inst = MemoryStatisticManager::Instance();
  inst.xsmem_summary_file_name_ = "./not_exist_file.xxx";
  inst.xsmem_group_name_ = "DM_QS_GROUP_12300";
  inst.StatisticXsmem();
  EXPECT_EQ(inst.xsmem_statistic_count_, 0);
  EXPECT_EQ(static_cast<uint64_t>(inst.xsmem_avg_), 0);
  EXPECT_EQ(inst.xsmem_hwm_, 0);
}

TEST_F(MemoryStatisticManagerTest, StatisticXsmem_normal) {
  std::string file_name = "./test_xsmem_summary";
  {
    std::ofstream of(file_name);
    of << "task pid 12345 pool cnt 1" << std::endl;
    of << "pool id(name): alloc_size real_size peak_size" << std::endl;
    of << "75(DM_QS_GROUP_12300_18446621857194198400: 7216672 7221248 17221248" << std::endl;
    of << "summary: 7216672 7221248" << std::endl;
  }

  auto &inst = MemoryStatisticManager::Instance();
  inst.xsmem_summary_file_name_ = file_name;
  inst.xsmem_group_name_ = "DM_QS_GROUP_12300";
  inst.StatisticXsmem();
  EXPECT_EQ(inst.xsmem_statistic_count_, 1);
  EXPECT_EQ(static_cast<uint64_t>(inst.xsmem_avg_), 7221248);
  EXPECT_EQ(inst.xsmem_hwm_, 17221248);
  // test avoid division by zero
  inst.xsmem_statistic_count_ = UINT32_MAX;
  inst.StatisticXsmem();
  EXPECT_EQ(inst.xsmem_statistic_count_, 1);
  remove(file_name.c_str());
}

TEST_F(MemoryStatisticManagerTest, StatisticXsmem_avg) {
  std::string file_name = "./test_xsmem_summary";
  {
    std::ofstream of(file_name);
    of << "task pid 12345 pool cnt 1" << std::endl;
    of << "pool id(name): alloc_size real_size peak_size" << std::endl;
    of << "75(DM_QS_GROUP_12300_18446621857194198400: 0 0 0" << std::endl;
    of << "summary: 0 0" << std::endl;
  }

  auto &inst = MemoryStatisticManager::Instance();
  inst.xsmem_summary_file_name_ = file_name;
  inst.xsmem_group_name_ = "DM_QS_GROUP_12300";
  inst.StatisticXsmem();
  EXPECT_EQ(inst.xsmem_statistic_count_, 0);
  EXPECT_EQ(static_cast<uint64_t>(inst.xsmem_avg_), 0);
  EXPECT_EQ(inst.xsmem_hwm_, 0);

  {
    std::ofstream of(file_name);
    of << "task pid 12345 pool cnt 1" << std::endl;
    of << "pool id(name): alloc_size real_size peak_size" << std::endl;
    of << "75(DM_QS_GROUP_12300_18446621857194198400: 100 200 300" << std::endl;
    of << "summary: 100 200" << std::endl;
  }

  inst.StatisticXsmem();
  EXPECT_EQ(inst.xsmem_statistic_count_, 1);
  EXPECT_EQ(static_cast<uint64_t>(inst.xsmem_avg_), 200);
  EXPECT_EQ(inst.xsmem_hwm_, 300);

  {
    std::ofstream of(file_name);
    of << "task pid 12345 pool cnt 1" << std::endl;
    of << "pool id(name): alloc_size real_size peak_size" << std::endl;
    of << "75(DM_QS_GROUP_12300_18446621857194198400: 250 300 400" << std::endl;
    of << "summary: 250 300" << std::endl;
  }

  inst.StatisticXsmem();
  EXPECT_EQ(inst.xsmem_statistic_count_, 2);
  EXPECT_EQ(static_cast<uint64_t>(inst.xsmem_avg_), (200 + 300) / 2);
  EXPECT_EQ(inst.xsmem_hwm_, 400);

  {
    std::ofstream of(file_name);
    of << "task pid 12345 pool cnt 1" << std::endl;
    of << "pool id(name): alloc_size real_size peak_size" << std::endl;
    of << "75(DM_QS_GROUP_12300_18446621857194198400: 100 100 400" << std::endl;
    of << "summary: 100 100" << std::endl;
  }

  inst.StatisticXsmem();
  EXPECT_EQ(inst.xsmem_statistic_count_, 3);
  EXPECT_EQ(static_cast<uint64_t>(inst.xsmem_avg_), (200 + 300 + 100) / 3);
  EXPECT_EQ(inst.xsmem_hwm_, 400);
  remove(file_name.c_str());
}
}  // namespace ge

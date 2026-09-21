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
#include <vector>
#include <string>
#include <map>
#include <stdlib.h>

#include <unistd.h>
#include "stdlib.h"
#include "framework/common/debug/ge_log.h"
#include "graph/ge_context.h"
#include "common/profiling/profiling_properties.h"
#include "common/ge_inner_attrs.h"

#include "macro_utils/dt_public_scope.h"
#include "common/subprocess/subprocess_manager.h"
#include "macro_utils/dt_public_unscope.h"
#include "depends/mmpa/src/mmpa_stub.h"

using namespace std;

namespace {
volatile int32_t g_waitpid_status;
volatile int32_t g_waitpid_ret;
class MockMmpa : public ge::MmpaStubApiGe {
 public:
  int32_t WaitPid(mmProcess pid, INT32 *status, INT32 options) override {
    *status = g_waitpid_status;
    int32_t ret = g_waitpid_ret;
    g_waitpid_ret = 0;
    return ret;
  }
  int32_t RealPath(const CHAR *path, CHAR *realPath, INT32 realPathLen) override {
    INT32 ret = EN_OK;
    char *ptr = realpath(path, realPath);
    if (ptr == nullptr) {
      ret = EN_OK;
    }
    return ret;
  }
  INT32 StatGet(const CHAR *path, mmStat_t *buffer) override {
    buffer->st_mode = S_IFREG;
    return EN_OK;
  }
};

}  // namespace
namespace ge {
class SubprocessManagerTest : public testing::Test {
 protected:
  void SetUp() override {
    g_waitpid_ret = 0;
    g_waitpid_status = 0;
  }
  void TearDown() override {
    MmpaStub::GetInstance().Reset();
  }
};

TEST_F(SubprocessManagerTest, UtSubProcessManager_01) {
    std::function<void(const ProcStatus &)> func = [](const ProcStatus &status) {};
    pid_t pid = getpid();
    MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
    SubprocessManager::GetInstance().RegExcptHandleCallback(pid, func);
    EXPECT_EQ(SubprocessManager::GetInstance().Initialize(), SUCCESS);
    usleep(10);  // 0.01ms, Wait for RunThread.
    SubprocessManager::GetInstance().Finalize();
    SubprocessManager::GetInstance().UnRegExcptHandleCallback(pid);
}
TEST_F(SubprocessManagerTest, UtSubProcessManagerWaitpidRet0) {
  std::function<void(const ProcStatus &)> func = [](const ProcStatus &status) {};
  pid_t pid = getpid();
  g_waitpid_ret = 0;
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  SubprocessManager::GetInstance().RegExcptHandleCallback(pid, func);
  EXPECT_EQ(SubprocessManager::GetInstance().Initialize(), SUCCESS);
  usleep(10);  // 0.01ms, Wait for RunThread.
  SubprocessManager::GetInstance().Finalize();
  SubprocessManager::GetInstance().UnRegExcptHandleCallback(pid);
}

TEST_F(SubprocessManagerTest, UtShutdownSubprocess) {
  std::function<void(const ProcStatus &)> func = [](const ProcStatus &status) {};
  // not exist pid
  pid_t pid = 9999999;
  g_waitpid_ret = 0;
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  SubprocessManager::GetInstance().RegExcptHandleCallback(pid, func);
  SubprocessManager::GetInstance().NotifySubprocessShutdown(pid);
  SubprocessManager::GetInstance().ShutdownSubprocess(pid, 1);
  EXPECT_EQ(SubprocessManager::GetInstance().planned_shutdown_[pid], true);
  SubprocessManager::GetInstance().Finalize();
}

TEST_F(SubprocessManagerTest, UtSubProcessManagerWaitpidStop) {
  std::mutex mt;
  std::condition_variable condition;
  bool callback_finish = false;
  ProcStatus status_result = ProcStatus::NORMAL;
  std::function<void(const ProcStatus &)> func = [&](const ProcStatus &status) {
    status_result = status;
    callback_finish = true;
    condition.notify_all();
  };
  pid_t pid = getpid();
  g_waitpid_ret = 3;
  g_waitpid_status = 0x7f;
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  SubprocessManager::GetInstance().RegExcptHandleCallback(pid, func);
  SubprocessManager::GetInstance().Initialize();
  // 等待回调
  std::unique_lock<std::mutex> lock(mt);
  condition.wait_for(lock, std::chrono::seconds(1),
                     [&callback_finish]() {
                       return callback_finish;
                     });

  SubprocessManager::GetInstance().run_flag_.store(false);
  if (SubprocessManager::GetInstance().watch_sub_proc_thread_.joinable()) {
    SubprocessManager::GetInstance().watch_sub_proc_thread_.join();
  }
  EXPECT_EQ(status_result, ProcStatus::STOPPED);
  SubprocessManager::GetInstance().Finalize();
  SubprocessManager::GetInstance().UnRegExcptHandleCallback(pid);
}

TEST_F(SubprocessManagerTest, UtSubProcessManagerWaitpidExit) {
  std::mutex mt;
  std::condition_variable condition;
  bool callback_finish = false;
  ProcStatus status_result = ProcStatus::NORMAL;
  std::function<void(const ProcStatus &)> func = [&](const ProcStatus &status) {
    status_result = status;
    callback_finish = true;
    condition.notify_all();
  };
  pid_t pid = 1000000;
  g_waitpid_ret = -1;
  g_waitpid_status = 0;
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  SubprocessManager::GetInstance().RegExcptHandleCallback(pid, func);
  SubprocessManager::GetInstance().Initialize();

  // 等待回调
  std::unique_lock<std::mutex> lock(mt);
  condition.wait_for(lock, std::chrono::seconds(1),
                     [&callback_finish]() {
                       return callback_finish;
                     });
  SubprocessManager::GetInstance().run_flag_.store(false);
  if (SubprocessManager::GetInstance().watch_sub_proc_thread_.joinable()) {
    SubprocessManager::GetInstance().watch_sub_proc_thread_.join();
  }
  EXPECT_EQ(status_result, ProcStatus::EXITED);
  SubprocessManager::GetInstance().Finalize();
}

TEST_F(SubprocessManagerTest, UtSubProcessManagerWaitpidContinue) {
  ProcStatus status_result = ProcStatus::NORMAL;
  std::function<void(const ProcStatus &)> func = [&](const ProcStatus &status) {
    status_result = status;
  };
  pid_t pid = getpid();
  g_waitpid_ret = 3;
  g_waitpid_status = 0xffff;
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  SubprocessManager::GetInstance().RegExcptHandleCallback(pid, func);
  SubprocessManager::GetInstance().Initialize();
  usleep(10);  // 0.01ms, Wait for RunThread.
  SubprocessManager::GetInstance().run_flag_.store(false);
  if (SubprocessManager::GetInstance().watch_sub_proc_thread_.joinable()) {
    SubprocessManager::GetInstance().watch_sub_proc_thread_.join();
  }
  EXPECT_EQ(status_result, ProcStatus::NORMAL);
  SubprocessManager::GetInstance().Finalize();
  SubprocessManager::GetInstance().UnRegExcptHandleCallback(pid);
}

TEST_F(SubprocessManagerTest, UtSubProcessManagerWaitpid_WIFSIGNALED) {
  std::function<void(const ProcStatus &)> func = [](const ProcStatus &status) {};
  pid_t pid = getpid();
  g_waitpid_ret = 3;
  g_waitpid_status = 0xd;
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  SubprocessManager::GetInstance().RegExcptHandleCallback(pid, func);
  EXPECT_EQ(SubprocessManager::GetInstance().Initialize(), SUCCESS);
  usleep(10);
  SubprocessManager::GetInstance().run_flag_.store(false);
  if (SubprocessManager::GetInstance().watch_sub_proc_thread_.joinable()) {
    SubprocessManager::GetInstance().watch_sub_proc_thread_.join();
  }
  SubprocessManager::GetInstance().Finalize();
  SubprocessManager::GetInstance().UnRegExcptHandleCallback(pid);
}

TEST_F(SubprocessManagerTest, UtGetFlowGwBinDir) {
  std::string flowgw_bin_dir;
  // /var/queue_schedule not exist
  auto ret = SubprocessManager::GetFlowGwBinDir("/mock_path", flowgw_bin_dir);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(flowgw_bin_dir.empty(), true);
  class MockMmpaInner : public ge::MmpaStubApiGe {
  public:
    int32_t RealPath(const CHAR *path, CHAR *realPath, INT32 realPathLen) override {
      (void)strncpy_s(realPath, realPathLen, path, strlen(path));
      return EN_OK;
    }
    INT32 StatGet(const CHAR *path, mmStat_t *buffer) override {
      if (path == std::string("/var/host_queue_schedule")) {
        return EN_ERROR;
      }
      buffer->st_mode = S_IFREG;
      return EN_OK;
    }
  };
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaInner>());
  ret = SubprocessManager::GetFlowGwBinDir("/mock_path", flowgw_bin_dir);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(flowgw_bin_dir, "/mock_path/host_queue_schedule");
}
}  // namespace ge

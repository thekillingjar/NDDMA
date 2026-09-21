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
#include <gmock/gmock.h>
#include <thread>
#include "common/ge_inner_attrs.h"

#include "macro_utils/dt_public_scope.h"
#include "common/data_flow/queue/heterogeneous_exchange_service.h"
#include "executor/engine_daemon.h"
#include "macro_utils/dt_public_unscope.h"

#include "graph/build/graph_builder.h"
#include "ge/ge_api.h"
#include "depends/runtime/src/runtime_stub.h"
#include "depends/mmpa/src/mmpa_stub.h"
#include "common/util/sanitizer_options.h"

using namespace std;
using namespace ::testing;

namespace ge {
namespace {
class MockRuntime : public RuntimeStub {
 public:
  MOCK_METHOD5(rtEschedWaitEvent, int32_t(int32_t, uint32_t, uint32_t, int32_t, rtEschedEventSummary_t * ));
};

int32_t AicpuLoadModelWithQStub(void *ptr) {
  (void) ptr;
  return 0;
}

int32_t AICPUModelDestroyStub(uint32_t modelId) {
  (void) modelId;
  return 0;
}

int32_t InitCpuSchedulerStub(const CpuSchedInitParam *const initParam) {
  (void) initParam;
  return 0;
}

int32_t StopCPUSchedulerStub(const uint32_t deviceId, const pid_t hostPid) {
  (void) deviceId;
  (void) hostPid;
  return 0;
}

int32_t HcclRegisterGlobalMemoryStub(void *addr, uint64_t size) {
  (void) addr;
  (void) size;
  return 0;
}

int32_t HcclUnregisterGlobalMemoryStub(void *addr) {
  (void) addr;
  return 0;
}

int32_t HcclRegisterGlobalMemoryStub1(void *addr, uint64_t size) {
  (void) addr;
  (void) size;
  return 1;
}

int32_t HcclUnregisterGlobalMemoryStub1(void *addr) {
  (void) addr;
  return 1;
}

class MockMmpa : public MmpaStubApiGe {
 public:
  void *DlOpen(const char *file_name, int32_t mode) {
    if (std::string(file_name) == "libaicpu_scheduler.so" || std::string(file_name) == "libhccd.so") {
      return (void *) 0x12345678;
    }
    return dlopen(file_name, mode);
  }

  void *DlSym(void *handle, const char *func_name) override {
    if (std::string(func_name) == "AicpuLoadModelWithQ") {
      return (void *) &AicpuLoadModelWithQStub;
    } else if (std::string(func_name) == "AICPUModelDestroy") {
      return (void *) &AICPUModelDestroyStub;
    } else if (std::string(func_name) == "InitCpuScheduler") {
      return (void *) &InitCpuSchedulerStub;
    } else if (std::string(func_name) == "HcclRegisterGlobalMemory") {
      return (void *) &HcclRegisterGlobalMemoryStub;
    } else if (std::string(func_name) == "HcclUnregisterGlobalMemory") {
      return (void *) &HcclUnregisterGlobalMemoryStub;
    } else if (std::string(func_name) == "StopCPUScheduler") {
      return (void *) &StopCPUSchedulerStub;
    }
    return dlsym(handle, func_name);
  }

  int32_t DlClose(void *handle) override {
    if (handle == (void *) 0x12345678) {
      return 0;
    }
    return dlclose(handle);
  }
};

class MockMmpa1 : public MmpaStubApiGe {
 public:
  void *DlOpen(const char *file_name, int32_t mode) {
    if (std::string(file_name) == "libhccd.so") {
      return (void *) 0x12345678;
    }
    return dlopen(file_name, mode);
  }

  void *DlSym(void *handle, const char *func_name) override {
    if (std::string(func_name) == "HcclRegisterGlobalMemory") {
      return (void *) &HcclRegisterGlobalMemoryStub1;
    } else if (std::string(func_name) == "HcclUnregisterGlobalMemory") {
      return (void *) &HcclUnregisterGlobalMemoryStub1;
    } else if (std::string(func_name) == "StopCPUScheduler") {
      return (void *) &StopCPUSchedulerStub;
    }
    return dlsym(handle, func_name);
  }

  int32_t DlClose(void *handle) override {
    if (handle == (void *) 0x12345678) {
      return 0;
    }
    return dlclose(handle);
  }
};

class MockMmpa2 : public MmpaStubApiGe {
 public:
  void *DlOpen(const char *file_name, int32_t mode) {
    if (std::string(file_name) == "libhccd.so") {
      return (void *) 0x12345678;
    }
    return dlopen(file_name, mode);
  }

  void *DlSym(void *handle, const char *func_name) override {
    if (std::string(func_name) == "HcclUnregisterGlobalMemory") {
      return (void *) &HcclUnregisterGlobalMemoryStub1;
    } else if (std::string(func_name) == "StopCPUScheduler") {
      return (void *) &StopCPUSchedulerStub;
    }
    return dlsym(handle, func_name);
  }

  int32_t DlClose(void *handle) override {
    if (handle == (void *) 0x12345678) {
      return 1;
    }
    return dlclose(handle);
  }
};

class CallbackManager {
 public:
  static CallbackManager &GetInstance() {
    static CallbackManager callbackManager;
    return callbackManager;
  }

  void Register(const char *moduleName, rtTaskFailCallback callback) {
    const std::string name = moduleName;
    callbacks_.emplace(name, callback);
  }

  void Call(const char *moduleName, rtExceptionInfo *excpt_info) {
    const std::string name = moduleName;
    auto iter = callbacks_.find(name);
    rtTaskFailCallback callback = iter->second;
    callback(excpt_info);
  }

  std::map<std::string, rtTaskFailCallback> callbacks_;
};

class MockRuntime2 : public RuntimeStub {
 public:
  struct MbufStub {
    MbufStub() {
      buffer.resize(1, 0);
      head.resize(512, 0);
      // set finalize msg type
      ExchangeService::MsgInfo *msg_info = reinterpret_cast<ExchangeService::MsgInfo *>(
          static_cast<uint8_t *>(&head[0]) + 512 - sizeof(ExchangeService::MsgInfo));
      msg_info->msg_type = 2;
    }

    std::vector<uint8_t> head;
    std::vector<uint8_t> buffer;
  };

  rtError_t rtRegTaskFailCallbackByModule(const char *moduleName,
                                          rtTaskFailCallback callback) override {
    CallbackManager::GetInstance().Register(moduleName, callback);
    return RT_ERROR_NONE;
  }

  rtError_t rtMemQueueDeQueue(int32_t device, uint32_t qid, void **mbuf) override {
    *mbuf = &mbuf_stub_;
    return 0;
  }

  rtError_t rtMemQueueEnQueue(int32_t dev_id, uint32_t qid, void *mem_buf) override {
    return 0;
  }

  rtError_t rtMbufGetBuffAddr(rtMbufPtr_t mbuf, void **databuf) override {
    *databuf = data_;
    return 0;
  }

  rtError_t rtMbufGetBuffSize(rtMbufPtr_t mbuf, uint64_t *size) override {
    *size = 0;
    return 0;
  }

  rtError_t rtMbufFree(rtMbufPtr_t mbuf) override {
    // 由MockRuntimeNoLeaks统一释放
    return RT_ERROR_NONE;
  }
  rtError_t rtMbufAlloc(rtMbufPtr_t *mbuf, uint64_t size) override {
    // 此处打桩记录所有申请的Mbuf,此UT不会Dequeue和Free而造成泄漏,因此在MockRuntime析构时统一释放
    RuntimeStub::rtMbufAlloc(mbuf, size);
    std::lock_guard<std::mutex> lk(mu_);
    mem_bufs_.emplace_back(*mbuf);
    return 0;
  }

  ~MockRuntime2() {
    for (auto &mbuf : mem_bufs_) {
      RuntimeStub::rtMbufFree(mbuf);
    }
    mem_bufs_.clear();
  }

 private:
  std::mutex mu_;
  uint8_t data_[128] = {};
  std::vector<void *> mem_bufs_;
  MbufStub mbuf_stub_;
};
}  // namespace
class EngineDaemonTest : public testing::Test {
 protected:
  void SetUp() override {
  }
  void TearDown() override {
    HeterogeneousExchangeService::GetInstance().Finalize();
    RuntimeStub::Reset();
  }
};

TEST_F(EngineDaemonTest, TestEngineDaemon) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpa>());
  RuntimeStub::SetInstance(std::make_shared<MockRuntime2>());
  EngineDaemon engine_daemon;
  auto device_id = std::to_string(0);
  auto queue_id = std::to_string(0);
  auto event_group_id = std::to_string(1);
  const std::string process_name = "npu_executor";
  const char_t *argv[] = {
      process_name.c_str(),
      "BufferGroupName",
      queue_id.c_str(),
      device_id.c_str(),
      event_group_id.c_str(),
      "--base_dir=./",
      "--device_id=0",
      "--msg_queue_device_id=0",
  };
  EXPECT_EQ(engine_daemon.InitializeWithArgs(8, (char **)argv), SUCCESS);
  std::thread loop_thread = std::thread([&]() { engine_daemon.LoopEvents(); });
  sleep(1);
  engine_daemon.SignalHandler(9);
  if (loop_thread.joinable()) {
    loop_thread.join();
  }
  engine_daemon.Finalize();
  MmpaStub::GetInstance().Reset();
}

TEST_F(EngineDaemonTest, TestInitProfilingFromOption) {
  EngineDaemon engine_daemon;
  std::map<std::string, std::string> options;
  options[kProfilingDeviceConfigData] = "test1";
  auto ret = engine_daemon.InitProfilingFromOption(options);
  EXPECT_EQ(ret, SUCCESS);
  options[kProfilingIsExecuteOn] = "1";
  ret = engine_daemon.InitProfilingFromOption(options);
  EXPECT_EQ(ret, SUCCESS);

  const char_t * const kEnvValue = "MS_PROF_INIT_FAIL";
  // 设置环境变量
  char_t npu_collect_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &npu_collect_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&npu_collect_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvValue, fail_collect_path.c_str(), 1);
  ret = engine_daemon.InitProfilingFromOption(options);
  EXPECT_NE(ret, SUCCESS);
  unsetenv(kEnvValue);

  // init profiling with ge option
  EXPECT_EQ(engine_daemon.InitProfilingFromOption(options), SUCCESS);
}

TEST_F(EngineDaemonTest, TestInitDumpFromOption) {
  EngineDaemon engine_daemon;
  std::map<std::string, std::string> options;
  options[OPTION_EXEC_ENABLE_DUMP] = "1";
  auto ret = engine_daemon.InitDumpFromOption(options);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(EngineDaemonTest, TestFinalizeMaintenance) {
  EngineDaemon engine_daemon;
  engine_daemon.is_dump_inited_ = true;
  auto ret = engine_daemon.FinalizeMaintenance();
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(EngineDaemonTest, TestHandleEvent) {
  RuntimeStub::SetInstance(std::make_shared<MockRuntime2>());
  EngineDaemon engine_daemon;
  engine_daemon.device_id_ = 0;
  engine_daemon.req_msg_queue_id_ = 1;
  engine_daemon.rsp_msg_queue_id_ = 2;
  engine_daemon.executor_message_server_ = MakeShared<MessageServer>(0, 1, 2);
  ASSERT_NE(engine_daemon.executor_message_server_, nullptr);
  ASSERT_EQ(engine_daemon.executor_message_server_->Initialize(), SUCCESS);

  rtMbufPtr_t mbuf = nullptr;
  rtMbufAlloc(&mbuf, 4);
  rtMemQueueEnQueue(engine_daemon.device_id_, engine_daemon.req_msg_queue_id_, mbuf);
  deployer::ExecutorRequest request;
  deployer::ExecutorResponse response;
  auto ret = engine_daemon.HandleEvent(request, response);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(EngineDaemonTest, TestEngineDaemonRegCallback) {
  auto mock_runtime = std::make_shared<MockRuntime2>();
  RuntimeStub::SetInstance(mock_runtime);
  EngineDaemon engine_daemon;
  auto device_id = std::to_string(0);
  auto queue_id = std::to_string(0);
  auto event_group_id = std::to_string(1);
  const std::string process_name = "npu_executor";
  const std::string without_model_executor = std::to_string(false);
  const char_t *argv[] = {
      process_name.c_str(),
      "BufferGroupName",
      queue_id.c_str(),
      device_id.c_str(),
      event_group_id.c_str(),
      without_model_executor.c_str(),
  };
  EXPECT_EQ(engine_daemon.InitializeWithArgs(6, (char **)argv), SUCCESS);
  rtExceptionInfo exceptionInfo;
  exceptionInfo.retcode = ACL_ERROR_RT_SOCKET_CLOSE;
  CallbackManager::GetInstance().Call("NpuExe", &exceptionInfo);
  exceptionInfo.retcode = 507018U;
  CallbackManager::GetInstance().Call("NpuExe", &exceptionInfo);
  exceptionInfo.retcode = 555U;
  CallbackManager::GetInstance().Call("NpuExe", &exceptionInfo);
  engine_daemon.Finalize();
}

TEST_F(EngineDaemonTest, TestEngineDaemonHost) {
  auto mock_runtime = std::make_shared<MockRuntime2>();
  auto back_up_context = GetThreadLocalContext();
  RuntimeStub::SetInstance(mock_runtime);
  EngineDaemon engine_daemon(true);
  EXPECT_EQ(engine_daemon.InitializeGeExecutor(), SUCCESS);
  engine_daemon.Finalize();
  RuntimeStub::Reset();
  GetThreadLocalContext() = back_up_context;
}
}  // namespace ge



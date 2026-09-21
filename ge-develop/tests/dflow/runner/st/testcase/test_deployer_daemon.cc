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
#include "depends/mmpa/src/mmpa_stub.h"
#include "depends/runtime/src/runtime_stub.h"
#include "graph/ge_local_context.h"

#include "macro_utils/dt_public_scope.h"
#include "common/data_flow/queue/heterogeneous_exchange_service.h"
#include "daemon/model_deployer_daemon.h"
#include "deploy/flowrm/tsd_client.h"
#include "macro_utils/dt_public_unscope.h"
#include "common/env_path.h"

using namespace std;

namespace ge {
namespace {
class MockRuntime : public RuntimeStub {
 public:
  struct MbufStub {
    MbufStub() {
      buffer.resize(1, 0);
      head.resize(1024, 0);
    }

    std::vector<uint8_t> head;
    std::vector<uint8_t> buffer;
  };

  rtError_t rtMemQueueDeQueue(int32_t device, uint32_t qid, void **mbuf) override {
    *mbuf = &mbuf_stub_;
    // for wait event
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
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

  ~MockRuntime() {
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

class MockMmpaDeployer : public MmpaStubApiGe {
 public:
  int32_t RealPath(const CHAR *path, CHAR *realPath, INT32 realPathLen) override {
    (void)strncpy_s(realPath, realPathLen, path, strlen(path));
    return 0;
  }

  virtual int32_t Sleep(UINT32 microSecond) override {
    return 0;
  }

  void *DlSym(void *handle, const char *func_name) override {
    if (std::string(func_name) == "TsdCapabilityGet") {
      return get_tsd_capability_func_;
    }
    std::cout << "func name:" << func_name << " not stub\n";
    return (void *)0xFFFFFFFF;
  }

  void *DlOpen(const char *fileName, int32_t mode) override {
    std::cout << "dlopen stub file name = " << fileName << std::endl;
    if (std::string(fileName) == "libtsdclient.so") {
      return (void *)0xFFFFFFFF;
    }
    return dlopen(fileName, mode);
  }

  int32_t DlClose(void *handle) override {
    if (handle == (void *)0xFFFFFFFF) {
      return 0;
    }
    return dlclose(handle);
  }

  void *get_tsd_capability_func_ = (void *)&TsdCapabilityGet;
};
}  // namespace

class ModelDeployerDaemonTest : public testing::Test {
 protected:
  void SetUp() override {
    std::string real_path =
        PathUtils::Join({EnvPath().GetAirBasePath(), "/tests/dflow/runner/st/st_run_data/json/helper_runtime/host/numa_config.json"});
    setenv("RESOURCE_CONFIG_PATH", real_path.c_str(), 1);
    // clear options
    std::map<std::string, std::string> empty_options;
    ge::GetThreadLocalContext().SetGlobalOption(empty_options);
    ge::GetThreadLocalContext().SetSessionOption(empty_options);
    ge::GetThreadLocalContext().SetGraphOption(empty_options);
    RuntimeStub::SetInstance(std::make_shared<MockRuntime>());
  }
  void TearDown() override {
    HeterogeneousExchangeService::GetInstance().Finalize();
    RuntimeStub::Reset();
    MmpaStub::GetInstance().Reset();
    unsetenv("RESOURCE_CONFIG_PATH");
  }
};

TEST_F(ModelDeployerDaemonTest, TestInitializeAndFinalizeOnCpu) {
  MmpaStub::GetInstance().SetImpl(std::make_shared<MockMmpaDeployer>());
  EXPECT_EQ(Configurations::GetInstance().InitInformation(), SUCCESS);
  Status ret = FAILED;
  ge::ModelDeployerDaemon daemon;
  daemon.is_sub_deployer_ = true;
  const std::string process_name = "sub_deployer";
  const char_t *argv[] = {
      process_name.c_str(),
      "MEM_GROUP_123456",
      "0",
      "1",
  };
  std::thread start = std::thread([&]() { ret = daemon.Start(4, (char **)argv); });
  sleep(1);
  daemon.SignalHandler(9);
  if (start.joinable()) {
    start.join();
  }
  EXPECT_EQ(ret, SUCCESS);
}
} // namespace ge

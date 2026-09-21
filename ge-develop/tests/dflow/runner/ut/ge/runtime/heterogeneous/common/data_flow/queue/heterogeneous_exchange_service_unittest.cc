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
#include <runtime_stub.h>
#include <test_tools_task_info.h>
#include "securec.h"

#include "macro_utils/dt_public_scope.h"
#include "common/data_flow/queue/heterogeneous_exchange_service.h"
#include "common/utils/heterogeneous_profiler.h"
#include "common/profiling/profiling_properties.h"
#include "macro_utils/dt_public_unscope.h"

#include "framework/common/ge_inner_error_codes.h"
#include "depends/runtime/src/runtime_stub.h"
#include "framework/common/runtime_tensor_desc.h"
#include "graph/ge_local_context.h"
#include "common/util/sanitizer_options.h"
#include "graph_metadef/common/ge_common/util.h"

using namespace ::testing;
bool enqueue_dequeue_error_flag = false;
namespace ge {
namespace {
  class MockRuntime : public RuntimeStub {
  public:
    rtError_t rtMemQueueEnQueue(int32_t dev_id, uint32_t qid, void *mem_buf) override {
      return 207014;
    }

    rtError_t rtMemQueueDeQueue(int32_t device, uint32_t qid, void **mbuf) override {
      return 207013;
    }
  };
  class MockRuntime2 : public RuntimeStub {
   public:
    rtError_t rtMemQueueEnQueueBuff(int32_t devId, uint32_t qid, rtMemQueueBuff_t *inBuf, int32_t timeout) {
      if (!enqueue_dequeue_error_flag) {
        return 0;
      }
      return 207014;
    }
  };
}
class HeterogeneousExchangeServiceTest : public testing::Test {
 protected:
  void SetUp() override {
  }

  void TearDown() override {
  }
};

TEST_F(HeterogeneousExchangeServiceTest, TestCreateClientQueue) {
  HeterogeneousExchangeService exchange_service;
  uint32_t queue_id = 0;

  std::string queue_name_max(127, 'a');
  MemQueueAttr attr;
  attr.is_client = true;
  attr.work_mode = RT_MQ_MODE_PULL;
  ASSERT_EQ(exchange_service.CreateQueue(0, queue_name_max, attr, queue_id), SUCCESS);
}

TEST_F(HeterogeneousExchangeServiceTest, TestCreateQueue) {
  HeterogeneousExchangeService exchange_service;
  uint32_t queue_id = 0;

  std::string queue_name_max(127, 'a');
  ASSERT_EQ(exchange_service.CreateQueue(0, queue_name_max, 2, RT_MQ_MODE_PULL, queue_id), SUCCESS);

  std::string queue_name_too_long(128, 'a');
  ASSERT_EQ(exchange_service.CreateQueue(0, queue_name_too_long, 2, RT_MQ_MODE_PULL, queue_id), PARAM_INVALID);
}

TEST_F(HeterogeneousExchangeServiceTest, TestDestroyQueue) {
  HeterogeneousExchangeService exchange_service;
  exchange_service.ResetQueueInfo(0, 2);
  ASSERT_EQ(exchange_service.DestroyQueue(0, 2), SUCCESS);
}

class TestClientQMockRuntime : public RuntimeStub {
public:
  rtError_t rtMemQueuePeek(int32_t device, uint32_t qid, size_t *bufLen, int32_t timeout) override {
    *bufLen = 1280;
    return 0;
  }

  rtError_t rtCtxCreate(rtContext_t *ctx, uint32_t flags, int32_t device) {
    static uintptr_t x = 1;
    *ctx = (rtContext_t *)x;
    return 0;
  }
};

TEST_F(HeterogeneousExchangeServiceTest, TestEnqueueAndDequeueClientQ) {
  TestClientQMockRuntime mock_runtime;
  RuntimeStub::Install(&mock_runtime);
  GE_MAKE_GUARD(recover, [&mock_runtime]() { RuntimeStub::UnInstall(&mock_runtime); });
  HeterogeneousExchangeService exchange_service;
  uint32_t queue_id = 0;
  MemQueueAttr attr;
  attr.is_client = true;
  attr.work_mode = RT_MQ_MODE_PULL;
  ASSERT_EQ(exchange_service.CreateQueue(0, "queue", attr, queue_id), SUCCESS);
  uint8_t buf[128];
  ExchangeService::ControlInfo control_info = {};
  control_info.timeout = 1000;
  {
    // Enqueue时创建Mbuf,rtMemQueueEnQueueBuff打桩没有保存地址,也没有Consumer取出
    DT_ALLOW_LEAKS_GUARD(ExchangeServiceEnqueue);
    ASSERT_EQ(exchange_service.Enqueue(0, queue_id, buf, sizeof(buf), control_info), SUCCESS);
  }
  ASSERT_EQ(exchange_service.Dequeue(0, queue_id, buf, 100, control_info), SUCCESS);
  g_runtime_stub_mock = "rtCtxGetCurrent";
  GE_MAKE_GUARD(mock, []() { g_runtime_stub_mock = ""; });
  GeTensor tensor;
  ASSERT_EQ(exchange_service.DequeueTensor(0, queue_id, tensor, control_info), SUCCESS);
}

TEST_F(HeterogeneousExchangeServiceTest, TestEnqueueAndDequeue) {
  HeterogeneousExchangeService exchange_service;
  uint32_t queue_id = 0;
  ASSERT_EQ(exchange_service.CreateQueue(0, "queue", 2, RT_MQ_MODE_PULL, queue_id), SUCCESS);
  uint8_t buf[128];
  ExchangeService::ControlInfo control_info = {};
  control_info.timeout = 1000;
  ASSERT_EQ(exchange_service.Enqueue(0, queue_id, buf, sizeof(buf), control_info), SUCCESS);
  ASSERT_EQ(exchange_service.Dequeue(0, queue_id, buf, 100, control_info), SUCCESS);
}

TEST_F(HeterogeneousExchangeServiceTest, TestEnqueueAndDequeueTensor) {
  HeterogeneousExchangeService exchange_service;
  exchange_service.subscribed_enqueues_[0] = false;
  exchange_service.subscribed_dequeues_[0] = false;
  uint32_t queue_id = 0;
  uint32_t client_queue_id = 1;
  ASSERT_EQ(exchange_service.CreateQueue(0, "queue", 2, RT_MQ_MODE_PULL, queue_id), SUCCESS);
  ExchangeService::ControlInfo control_info = {};
  control_info.timeout = 1000;
  const size_t buffer_size = 128;
  std::shared_ptr<AlignedPtr> aligned_ptr = nullptr;
  rtMbufPtr_t m_buf = nullptr;
  rtMbufPtr_t dev_m_buf = nullptr;
  rtMbufAlloc(&m_buf, buffer_size);
  void *data = nullptr;
  rtMbufGetBuffAddr(m_buf, &data);

  RuntimeTensorDesc runtime_tensor_desc{};
  runtime_tensor_desc.shape[0] = 16;
  runtime_tensor_desc.shape[1] = 16;
  runtime_tensor_desc.original_shape[0] = 16;
  runtime_tensor_desc.original_shape[1] = 16;
  const std::vector<ExchangeService::BuffInfo> buffs{
      {.addr = &runtime_tensor_desc, .len = sizeof(runtime_tensor_desc)},
      {.addr = ValueToPtr(PtrToValue(data)), .len = buffer_size},
      {.addr = nullptr, .len = 0}};
  ASSERT_EQ(exchange_service.Enqueue(0, queue_id, buffs, control_info), SUCCESS);
  // 避免直接使用私有成员，此处queue_id不应复用
  exchange_service.AddClientQueue(client_queue_id);
  ASSERT_EQ(exchange_service.Enqueue(0, client_queue_id, buffs, control_info), SUCCESS);
  rtMbufFree(m_buf);
  // rtQueue中的Mbuf需要释放
  //  ASSERT_EQ(exchange_service.DequeueMbuf(0, queue_id, &dev_m_buf, control_info.timeout), SUCCESS);
  GeTensor output_tensor;
  ASSERT_EQ(exchange_service.DequeueTensor(0, queue_id, output_tensor, control_info), SUCCESS);
  rtMbufFree(dev_m_buf);
  exchange_service.Finalize();
}

TEST_F(HeterogeneousExchangeServiceTest, TestMbufEnqueueAndDequeue) {
  HeterogeneousExchangeService exchange_service;
  uint32_t queue_id = 0;
  ASSERT_EQ(exchange_service.CreateQueue(0, "queue", 2, RT_MQ_MODE_PULL, queue_id), SUCCESS);
  ExchangeService::ControlInfo control_info = {};
  control_info.timeout = 1000;
  const size_t buffer_size = 128;
  std::shared_ptr<AlignedPtr> aligned_ptr = nullptr;
  rtMbufPtr_t m_buf = nullptr;
  rtMbufAlloc(&m_buf, buffer_size);

  ASSERT_EQ(exchange_service.Enqueue(0, queue_id, buffer_size, m_buf, control_info), SUCCESS);
  ASSERT_EQ(exchange_service.DequeueMbufTensor(0, queue_id, aligned_ptr, buffer_size, control_info), SUCCESS);
  rtMbufFree(m_buf);
}

TEST_F(HeterogeneousExchangeServiceTest, TestDequeueEmpty) {
  HeterogeneousExchangeService exchange_service;
  uint32_t queue_id = 0;
  ASSERT_EQ(exchange_service.CreateQueue(0, "queue", 2, RT_MQ_MODE_PULL, queue_id), SUCCESS);
  uint8_t buf[128];
  ExchangeService::ControlInfo control_info;
  ASSERT_NE(exchange_service.Dequeue(0, queue_id, buf, sizeof(buf), control_info), SUCCESS);
}

TEST_F(HeterogeneousExchangeServiceTest, TestEnqueueAndDequeueFail) {
  MockRuntime mock_runtime;
  RuntimeStub::Install(&mock_runtime);
  HeterogeneousExchangeService exchange_service;
  exchange_service.subscribed_enqueues_[0] = false;
  exchange_service.subscribed_dequeues_[0] = false;
  uint32_t queue_id = 0;
  ASSERT_EQ(exchange_service.CreateQueue(0, "queue", 2, RT_MQ_MODE_PULL, queue_id), SUCCESS);
  uint8_t buf[128];
  ExchangeService::ControlInfo enqueue_control_info = {};
  enqueue_control_info.timeout = 1;
  exchange_service.Enqueue(0, queue_id, buf, sizeof(buf), enqueue_control_info);
  ExchangeService::ControlInfo control_info;
  control_info.timeout = 1;
  exchange_service.Dequeue(0, queue_id, buf, sizeof(buf), control_info);
  exchange_service.Finalize();
  RuntimeStub::UnInstall(&mock_runtime);
}
TEST_F(HeterogeneousExchangeServiceTest, TestHeterogeneousProfiler) {
  setenv("GE_PROFILING_TO_STD_OUT", "2", true);
  uint32_t queue_id = 0;
  int32_t device_id = 1;
  HeterogeneousProfiler profiler;
  profiler.InitHeterogeneousPoriler();
  profiler.RecordHeterogeneousProfilerEvent(ProfilerType::kStartPoint,
                                                       ProfilerEvent::kMbufAlloc, device_id, queue_id);
  profiler.RecordHeterogeneousProfilerEvent(ProfilerType::kEndPoint,
                                                       ProfilerEvent::kMbufAlloc, device_id, queue_id);
  profiler.RecordHeterogeneousProfilerEvent(ProfilerType::kStartPoint,
                                                       ProfilerEvent::kMemCopyToMbuf, device_id, queue_id);
  profiler.RecordHeterogeneousProfilerEvent(ProfilerType::kEndPoint,
                                                       ProfilerEvent::kMemCopyToMbuf, device_id, queue_id);
  profiler.RecordHeterogeneousProfilerEvent(ProfilerType::kStartPoint,
                                                       ProfilerEvent::kMbufEnqueue, device_id, queue_id);
  profiler.RecordHeterogeneousProfilerEvent(ProfilerType::kEndPoint,
                                                       ProfilerEvent::kMbufEnqueue, device_id, queue_id);
  profiler.RecordHeterogeneousProfilerEvent(ProfilerType::kStartPoint,
                                                       ProfilerEvent::kMbufDequeue, device_id, queue_id);
  profiler.RecordHeterogeneousProfilerEvent(ProfilerType::kEndPoint,
                                                       ProfilerEvent::kMbufDequeue, device_id, queue_id);
  profiler.RecordHeterogeneousProfilerEvent(ProfilerType::kStartPoint,
                                                       ProfilerEvent::kMbufCopyToMem, device_id, queue_id);
  profiler.RecordHeterogeneousProfilerEvent(ProfilerType::kEndPoint,
                                                       ProfilerEvent::kMbufCopyToMem, device_id, queue_id);
  profiler.RecordHeterogeneousProfilerEvent(ProfilerType::kStartPoint, ProfilerEvent::kPrepareInputs);
  profiler.RecordHeterogeneousProfilerEvent(ProfilerType::kEndPoint, ProfilerEvent::kPrepareInputs);
  profiler.RecordHeterogeneousProfilerEvent(ProfilerType::kStartPoint, ProfilerEvent::kPrepareOutputs);
  profiler.RecordHeterogeneousProfilerEvent(ProfilerType::kEndPoint, ProfilerEvent::kPrepareOutputs);
  profiler.RecordHeterogeneousProfilerEvent(ProfilerType::kStartPoint, ProfilerEvent::kDynamicExecute);
  profiler.RecordHeterogeneousProfilerEvent(ProfilerType::kEndPoint, ProfilerEvent::kDynamicExecute);
  profiler.RecordHeterogeneousProfilerEvent(ProfilerType::kStartPoint, ProfilerEvent::kUpdateOutputs);
  profiler.RecordHeterogeneousProfilerEvent(ProfilerType::kEndPoint, ProfilerEvent::kUpdateOutputs);
  ASSERT_NE(profiler.profiler_record_.size(), 0);
  profiler.PrintHeterogeneousProfilerData();
  ASSERT_EQ(profiler.profiler_record_.size(), 0);
  unsetenv("GE_PROFILING_TO_STD_OUT");
}

TEST_F(HeterogeneousExchangeServiceTest, TestModelIoProfiling) {
  HeterogeneousExchangeService exchange_service;
  uint32_t queue_id = 0;
  ASSERT_EQ(exchange_service.CreateQueue(0, "queue", 2, RT_MQ_MODE_PULL, queue_id), SUCCESS);
  ExchangeService::ControlInfo control_info = {};
  control_info.timeout = 1000;
  const size_t buffer_size = 128;
  std::shared_ptr<AlignedPtr> aligned_ptr = nullptr;
  rtMbufPtr_t m_buf = nullptr;
  rtMbufAlloc(&m_buf, buffer_size);

  bool old_value = ProfilingProperties::Instance().ProfilingTrainingTraceOn();
  ProfilingProperties::Instance().SetTrainingTrace(true);
  ASSERT_EQ(exchange_service.Enqueue(0, queue_id, buffer_size, m_buf, control_info), SUCCESS);
  ASSERT_EQ(exchange_service.DequeueMbufTensor(0, queue_id, aligned_ptr, buffer_size, control_info), SUCCESS);
  rtMbufFree(m_buf);
  ProfilingProperties::Instance().SetTrainingTrace(old_value);
}

TEST_F(HeterogeneousExchangeServiceTest, TestBuffsEnqueueAndDequeue) {
  RuntimeStub::SetInstance(std::make_shared<MockRuntime2>());
  HeterogeneousExchangeService exchange_service;
  uint32_t queue_id = 0;
  ASSERT_EQ(exchange_service.CreateQueue(0, "queue", 2, RT_MQ_MODE_PULL, queue_id), SUCCESS);
  ExchangeService::ControlInfo control_info = {};
  control_info.timeout = 1000;
  const size_t buffer_size = 128;
  rtMbufPtr_t m_buf = nullptr;
  rtMbufAlloc(&m_buf, buffer_size);
  void *data = nullptr;
  rtMbufGetBuffAddr(m_buf, &data);

  RuntimeTensorDesc runtime_tensor_desc;
  const std::vector<ExchangeService::BuffInfo> buffs{
      {.addr = &runtime_tensor_desc, .len = sizeof(runtime_tensor_desc)},
      {.addr = ValueToPtr(PtrToValue(data)), .len = buffer_size},
      {.addr = nullptr, .len = 0}};
  // 场景1：成功执行,创建Mbuf并入Queue,最后出Queue释放
  ASSERT_EQ(exchange_service.Enqueue(0, queue_id, buffs, control_info), SUCCESS);
  rtMbufPtr_t read_buf = nullptr;
  ASSERT_EQ(exchange_service.DequeueMbuf(0, queue_id, &read_buf, control_info.timeout), SUCCESS);
  rtMbufFree(read_buf);

  // 场景2：client侧queue,直接走ProcessEnqueueBuff
  exchange_service.client_queue_ids_.emplace(queue_id);
  ASSERT_EQ(exchange_service.Enqueue(0, queue_id, buffs, control_info), SUCCESS);

  // 场景3：rtMemQueueEnQueueBuff失败
  enqueue_dequeue_error_flag = true;
  ASSERT_EQ(exchange_service.Enqueue(0, queue_id, buffs, control_info), ACL_ERROR_RT_QUEUE_FULL);
  enqueue_dequeue_error_flag = false;
  rtMbufFree(m_buf);
  exchange_service.Finalize();
  RuntimeStub::Reset();
}

class MockRuntime1 : public RuntimeStub {
 public:
  rtError_t rtMbufGetPrivInfo(rtMbufPtr_t mbuf, void **priv, uint64_t *size) override {
    static int8_t datas[63] = {};
    *priv = datas;
    *size = 63;
    return 0;
  }
};

TEST_F(HeterogeneousExchangeServiceTest, UserData_Failed) {
  HeterogeneousExchangeService exchange_service;
  ExchangeService::ControlInfo control_info = {};
  control_info.timeout = 1000;
  const size_t buffer_size = 128;
  std::shared_ptr<AlignedPtr> aligned_ptr = nullptr;
  rtMbufPtr_t m_buf = nullptr;
  rtMbufAlloc(&m_buf, buffer_size);
  MockRuntime1 mock_runtime;
  RuntimeStub::Install(&mock_runtime);
  ASSERT_NE(exchange_service.InitHeadInfo(control_info, m_buf), SUCCESS);
  ASSERT_NE(exchange_service.CheckResult(m_buf, control_info), SUCCESS);
  RuntimeStub::UnInstall(&mock_runtime);
  rtMbufFree(m_buf);
}

TEST_F(HeterogeneousExchangeServiceTest, check_gen_trans_id) {
  HeterogeneousExchangeService exchange_service;
  ExchangeService::ControlInfo control_info = {};
  ExchangeService::MsgInfo msg_info = {};
  control_info.msg_info = &msg_info;
  msg_info.data_flag |= kCustomTransIdFlagBit;
  control_info.timeout = 1000;

  ExchangeService::ControlInfo control_info_ret = {};
  ExchangeService::MsgInfo msg_info_ret = {};
  control_info_ret.msg_info = &msg_info_ret;

  rtMbufPtr_t m_buf = nullptr;
  rtMbufAlloc(&m_buf, 128);
  uint64_t expect_trans_id = 1;
  msg_info.trans_id = expect_trans_id;
  EXPECT_EQ(exchange_service.InitHeadInfo(control_info, m_buf), SUCCESS);
  EXPECT_EQ(exchange_service.SetTransId(0, 0 , m_buf), SUCCESS);
  EXPECT_EQ(exchange_service.CheckResult(m_buf, control_info_ret), SUCCESS);
  EXPECT_EQ(msg_info_ret.trans_id, expect_trans_id);

  msg_info.trans_id = 0;
  // trans id auto increase by last trans id.
  EXPECT_EQ(exchange_service.InitHeadInfo(control_info, m_buf), SUCCESS);
  EXPECT_EQ(exchange_service.SetTransId(0, 0 , m_buf), SUCCESS);
  EXPECT_EQ(exchange_service.CheckResult(m_buf, control_info_ret), SUCCESS);
  EXPECT_EQ(msg_info_ret.trans_id, expect_trans_id + 1);

  expect_trans_id = UINT32_MAX;
  msg_info.trans_id = expect_trans_id;
  EXPECT_EQ(exchange_service.InitHeadInfo(control_info, m_buf), SUCCESS);
  EXPECT_EQ(exchange_service.SetTransId(0, 0 , m_buf), SUCCESS);
  EXPECT_EQ(exchange_service.CheckResult(m_buf, control_info_ret), SUCCESS);
  EXPECT_EQ(msg_info_ret.trans_id, expect_trans_id);

  msg_info.trans_id = 100;
  // trans id can not change small
  EXPECT_EQ(exchange_service.InitHeadInfo(control_info, m_buf), SUCCESS);
  EXPECT_NE(exchange_service.SetTransId(0, 0 , m_buf), SUCCESS);

  msg_info.trans_id = UINT64_MAX;
  // trans id can not be UINT64_MAX
  EXPECT_EQ(exchange_service.InitHeadInfo(control_info, m_buf), SUCCESS);
  EXPECT_NE(exchange_service.SetTransId(0, 0 , m_buf), SUCCESS);

  expect_trans_id = UINT64_MAX - 1;
  msg_info.trans_id = UINT64_MAX - 1;
  EXPECT_EQ(exchange_service.InitHeadInfo(control_info, m_buf), SUCCESS);
  EXPECT_EQ(exchange_service.SetTransId(0, 0 , m_buf), SUCCESS);
  EXPECT_EQ(exchange_service.CheckResult(m_buf, control_info_ret), SUCCESS);
  EXPECT_EQ(msg_info_ret.trans_id, expect_trans_id);

  msg_info.trans_id = 0;
  EXPECT_EQ(exchange_service.InitHeadInfo(control_info, m_buf), SUCCESS);
  EXPECT_NE(exchange_service.SetTransId(0, 0 , m_buf), SUCCESS);

  HeterogeneousExchangeService exchange_service_new;
  // data_flag with out custom trans id, transId increase auto
  msg_info.data_flag = 0;
  msg_info.trans_id = 100;
  EXPECT_EQ(exchange_service_new.InitHeadInfo(control_info, m_buf), SUCCESS);
  EXPECT_EQ(exchange_service_new.SetTransId(0, 0 , m_buf), SUCCESS);
  EXPECT_EQ(exchange_service_new.CheckResult(m_buf, control_info_ret), SUCCESS);
  EXPECT_EQ(msg_info_ret.trans_id, 1);

  rtMbufFree(m_buf);
}

TEST_F(HeterogeneousExchangeServiceTest, MultiThreadCopy_small_memory) {
  size_t size = 1024;
  std::unique_ptr<uint8_t[]> src = std::make_unique<uint8_t[]>(size);
  std::unique_ptr<uint8_t[]> dst = std::make_unique<uint8_t[]>(size);
  memset_s(src.get(), size, 'A', size);
  HeterogeneousExchangeService exchange_service;
  EXPECT_EQ(exchange_service.MultiThreadCopy(dst.get(), size, src.get(), size), SUCCESS);
  EXPECT_EQ(memcmp(src.get(), dst.get(), size), 0);
}

TEST_F(HeterogeneousExchangeServiceTest, MultiThreadCopy_big_memory) {
  size_t size = 50 * 1024 * 1024;
  std::unique_ptr<uint8_t[]> src = std::make_unique<uint8_t[]>(size);
  std::unique_ptr<uint8_t[]> dst = std::make_unique<uint8_t[]>(size);
  memset_s(src.get(), size, 'A', size);
  HeterogeneousExchangeService exchange_service;
  EXPECT_EQ(exchange_service.MultiThreadCopy(dst.get(), size, src.get(), size), SUCCESS);
  EXPECT_EQ(memcmp(src.get(), dst.get(), size), 0);
}
}  // namesapce ge

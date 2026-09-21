/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <thread>
#include "gtest/gtest.h"
#include "securec.h"
#include "ascend_hal.h"
#define private public

#include "flow_func/flow_func_processor.h"
#include "flow_func/flow_func_config_manager.h"

#undef private
#include "flow_func/meta_flow_func.h"
#include "flow_func/meta_multi_func.h"
#include "flow_func/flow_func_manager.h"
#include "flow_func/flow_func_log.h"
#include "config/global_config.h"
#include "model/attr_value_impl.h"

#include "mmpa/mmpa_api.h"
#include "mockcpp/mockcpp.hpp"
#include "execute/flow_model_impl.h"
#include "common/inner_error_codes.h"
#include "utils/udf_test_helper.h"
#include "common/data_utils.h"
#include "common/scope_guard.h"
#include "toolchain/dump/udf_dump_manager.h"

namespace FlowFunc {
namespace {
class SingleFlowFuncStub : public MetaFlowFunc {
 public:
  SingleFlowFuncStub() = default;

  ~SingleFlowFuncStub() override = default;

  int32_t Init() override {
    int64_t int_value = 0;
    auto ret = context_->GetAttr("not_exits_attr", int_value);
    EXPECT_EQ(ret, FLOW_FUNC_ERR_ATTR_NOT_EXITS);
    FLOW_FUNC_RUN_LOG_INFO("SingleFlowFuncStub init");
    (void)context_->GetAttr("need_re_init_attr", need_re_init_);
    if (need_re_init_ && init_times_ == 0) {
      ++init_times_;
      FLOW_FUNC_RUN_LOG_ERROR("test need reinit");
      return FLOW_FUNC_ERR_INIT_AGAIN;
    }

    (void)context_->GetAttr("need_proc_pend_attr", need_proc_pend_);
    return FLOW_FUNC_SUCCESS;
  }

  int32_t Proc(const std::vector<std::shared_ptr<FlowMsg>> &input_tensors) override {
    if (need_proc_pend_ && (proc_times_ == 0)) {
      ++proc_times_;
      context_->SetOutput(100, nullptr);
      return FLOW_FUNC_ERR_PROC_PENDING;
    }
    auto output = context_->AllocTensorMsgWithAlign({1}, TensorDataType::DT_INT64, 256);
    context_->SetOutput(0, output);
    return FLOW_FUNC_SUCCESS;
  }

 private:
  bool need_re_init_ = false;
  bool need_proc_pend_ = false;
  uint32_t init_times_ = 0;
  uint32_t proc_times_ = 0;
};

class TestProcSlow : public MetaFlowFunc {
 public:
  TestProcSlow() = default;

  ~TestProcSlow() override = default;

  int32_t Init() override {
    return FLOW_FUNC_SUCCESS;
  }

  int32_t Proc(const std::vector<std::shared_ptr<FlowMsg>> &input_tensors) override {
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    return FLOW_FUNC_ERR_PROC_PENDING;
  }
};

class MultiFlowFuncStub : public MetaMultiFunc {
 public:
  MultiFlowFuncStub() = default;
  ~MultiFlowFuncStub() override = default;

  virtual int32_t Init(const std::shared_ptr<MetaParams> &params) {
    (void)params->GetAttr("need_re_init_attr", need_re_init_);
    if (need_re_init_ && init_times_ == 0) {
      ++init_times_;
      return FLOW_FUNC_ERR_INIT_AGAIN;
    }
    (void)params->GetAttr("need_proc_pend_attr", need_proc_pend_);
    return FLOW_FUNC_SUCCESS;
  }

  int32_t Proc1(const std::shared_ptr<MetaRunContext> &run_context,
                const std::vector<std::shared_ptr<FlowMsg>> &input_msgs) {
    if (need_proc_pend_ && (proc_times_ == 0)) {
      ++proc_times_;
      return FLOW_FUNC_ERR_PROC_PENDING;
    }
    auto output = run_context->AllocTensorMsg({1}, TensorDataType::DT_INT64);
    run_context->SetOutput(0, output);
    proc_times_ = 0;
    return FLOW_FUNC_SUCCESS;
  }

  int32_t Proc2(const std::shared_ptr<MetaRunContext> &run_context,
                const std::vector<std::shared_ptr<FlowMsg>> &input_msgs) {
    if (need_proc_pend_ && (proc_times_ == 0)) {
      ++proc_times_;
      return FLOW_FUNC_ERR_PROC_PENDING;
    }
    auto output = run_context->AllocTensorMsg({1}, TensorDataType::DT_INT64);
    run_context->SetOutput(0, output);
    proc_times_ = 0;
    return FLOW_FUNC_SUCCESS;
  }

  int32_t Proc3(const std::shared_ptr<MetaRunContext> &run_context,
                const std::vector<std::shared_ptr<FlowMsgQueue>> &input_msg_queues) {
    if (need_proc_pend_ && (proc_times_ == 0)) {
      ++proc_times_;
      return FLOW_FUNC_ERR_PROC_PENDING;
    }
    auto output = run_context->AllocTensorMsg({1}, TensorDataType::DT_INT64);
    run_context->SetOutput(0, output);
    return FLOW_FUNC_SUCCESS;
  }

 private:
  bool need_re_init_ = false;
  bool need_proc_pend_ = false;
  uint32_t init_times_ = 0;
  uint32_t proc_times_ = 0;
};

REGISTER_FLOW_FUNC("TestProcSlow", TestProcSlow);
REGISTER_FLOW_FUNC("flow_func_for_ut_normal", SingleFlowFuncStub);
FLOW_FUNC_REGISTRAR(MultiFlowFuncStub)
    .RegProcFunc("processor_ut_Proc1", &MultiFlowFuncStub::Proc1)
    .RegProcFunc("processor_ut_Proc2", &MultiFlowFuncStub::Proc2)
    .RegProcFunc("processor_ut_Proc3", &MultiFlowFuncStub::Proc3);

struct MbufImpl {
  uint32_t mbuf_size;
  uint8_t reserve_head[256];
  RuntimeTensorDesc tensor_desc;
  uint8_t data[1024];
};

int halMbufAllocExStub(uint64_t size, unsigned int align, unsigned long flag, int grp_id, Mbuf **mbuf) {
  MbufImpl *mbuf_impl = new (std::nothrow) MbufImpl();
  memset_s(mbuf_impl, sizeof(MbufImpl), 0, sizeof(MbufImpl));
  mbuf_impl->mbuf_size = size;
  *mbuf = reinterpret_cast<Mbuf *>(mbuf_impl);
  return DRV_ERROR_NONE;
}

int halMbufFreeStub(Mbuf *mbuf) {
  MbufImpl *mbuf_impl = reinterpret_cast<MbufImpl *>(mbuf);
  delete mbuf_impl;
  return DRV_ERROR_NONE;
}

int halMbufCopyRefStub(Mbuf *mbuf, Mbuf **new_mbuf) {
  MbufImpl *src_mbuf_impl = reinterpret_cast<MbufImpl *>(mbuf);
  MbufImpl *new_mbuf_impl = new (std::nothrow) MbufImpl();
  memcpy_s(new_mbuf_impl, sizeof(MbufImpl), src_mbuf_impl, sizeof(MbufImpl));
  *new_mbuf = reinterpret_cast<Mbuf *>(new_mbuf_impl);
  return DRV_ERROR_NONE;
}

int halMbufGetBuffAddrStub(Mbuf *mbuf, void **buf) {
  MbufImpl *mbuf_impl = reinterpret_cast<MbufImpl *>(mbuf);
  *buf = &(mbuf_impl->tensor_desc);
  return DRV_ERROR_NONE;
}

int halMbufGetBuffSizeStub(Mbuf *mbuf, uint64_t *total_size) {
  MbufImpl *mbuf_impl = reinterpret_cast<MbufImpl *>(mbuf);
  *total_size = mbuf_impl->mbuf_size;
  return DRV_ERROR_NONE;
}

int halMbufSetDataLenStub(Mbuf *mbuf, uint64_t len) {
  MbufImpl *mbuf_impl = reinterpret_cast<MbufImpl *>(mbuf);
  mbuf_impl->mbuf_size = len;
  return DRV_ERROR_NONE;
}

int halMbufGetPrivInfoStub(Mbuf *mbuf, void **priv, unsigned int *size) {
  MbufImpl *mbuf_impl = reinterpret_cast<MbufImpl *>(mbuf);
  // start from first byte.
  *priv = &mbuf_impl->reserve_head;
  *size = 256;
  return DRV_ERROR_NONE;
}

int halMbufGetDataLenStub(Mbuf *mbuf, uint64_t *len) {
  MbufImpl *mbuf_impl = reinterpret_cast<MbufImpl *>(mbuf);
  *len = mbuf_impl->mbuf_size;
  return DRV_ERROR_NONE;
}

drvError_t halQueueDeQueueStub(unsigned int dev_id, unsigned int qid, void **mbuf) {
  Mbuf *buf = nullptr;
  (void)halMbufAllocExStub(sizeof(RuntimeTensorDesc) + 1024, 0, 0, 0, &buf);
  *mbuf = buf;
  void *data_buf = nullptr;
  halMbufGetBuffAddrStub(buf, &data_buf);
  auto tensor_desc = static_cast<RuntimeTensorDesc *>(data_buf);
  tensor_desc->dtype = static_cast<int64_t>(TensorDataType::DT_INT8);
  tensor_desc->shape[0] = 1;
  tensor_desc->shape[1] = 1024;
  return DRV_ERROR_NONE;
}

drvError_t halQueueEnQueueStub(unsigned int dev_id, unsigned int qid, void *mbuf) {
  halMbufFreeStub(static_cast<Mbuf *>(mbuf));
  return DRV_ERROR_NONE;
}

int halMbufAllocStub(uint64_t size, Mbuf **mbuf) {
  MbufImpl *mbuf_impl = new (std::nothrow) MbufImpl();
  memset_s(mbuf_impl, sizeof(MbufImpl), 0, sizeof(MbufImpl));
  mbuf_impl->mbuf_size = size;
  *mbuf = reinterpret_cast<Mbuf *>(mbuf_impl);
  return DRV_ERROR_NONE;
}

drvError_t halQueueDeQueueBuffStub(unsigned int dev_id, unsigned int qid, struct buff_iovec *vector, int timeout) {
  auto tensor_desc = static_cast<RuntimeTensorDesc *>(vector->ptr[0].iovec_base);
  tensor_desc->dtype = static_cast<int64_t>(TensorDataType::DT_INT8);
  tensor_desc->shape[0] = 1;
  tensor_desc->shape[1] = 1024;
  return DRV_ERROR_NONE;
}

drvError_t halQueuePeekStub(unsigned int dev_id, unsigned int qid, uint64_t *buf_len, int timeout) {
  *buf_len = sizeof(RuntimeTensorDesc) + 1024;
  return DRV_ERROR_NONE;
}
}  // namespace
class FlowFuncProcessorUTest : public testing::Test {
 protected:
  virtual void SetUp() {
    MOCKER(halMbufAllocEx).defaults().will(invoke(halMbufAllocExStub));
    MOCKER(halMbufFree).defaults().will(invoke(halMbufFreeStub));
    MOCKER(halMbufGetBuffAddr).defaults().will(invoke(halMbufGetBuffAddrStub));
    MOCKER(halMbufGetBuffSize).defaults().will(invoke(halMbufGetBuffSizeStub));
    MOCKER(halMbufSetDataLen).defaults().will(invoke(halMbufSetDataLenStub));
    MOCKER(halMbufGetPrivInfo).defaults().will(invoke(halMbufGetPrivInfoStub));
    MOCKER(halMbufGetDataLen).defaults().will(invoke(halMbufGetDataLenStub));
    MOCKER(halMbufCopyRef).defaults().will(invoke(halMbufCopyRefStub));
    MOCKER(halQueueDeQueue).defaults().will(invoke(halQueueDeQueueStub));
    MOCKER(halQueueEnQueue).defaults().will(invoke(halQueueEnQueueStub));
    MOCKER(halMbufAlloc).defaults().will(invoke(halMbufAllocStub));
    MOCKER(halQueuePeek).defaults().will(invoke(halQueuePeekStub));
    MOCKER(halQueueDeQueueBuff).defaults().will(invoke(halQueueDeQueueBuffStub));
  }

  virtual void TearDown() {
    GlobalMockObject::verify();
    FlowFuncManager::Instance().Reset();
  }
};

TEST_F(FlowFuncProcessorUTest, flow_func_processor_lib_load_failed) {
  std::string flow_func_name = "flow_func_processor_lib_load_failed";
  std::string pp_name = "pp0";
  std::string lib_path = "/xxx";
  std::vector<QueueDevInfo> input_queue_infos = UdfTestHelper::CreateQueueDevInfos({1});
  std::vector<uint32_t> output_queue_indexes = {0};
  std::vector<QueueDevInfo> all_output_queue_infos = UdfTestHelper::CreateQueueDevInfos({2, 3});

  std::shared_ptr<FlowFuncParams> params(
      new (std::nothrow) FlowFuncParams(pp_name, input_queue_infos.size(), all_output_queue_infos.size(), 0, 0));
  params->SetInstanceName(pp_name + "@0_1_1@0@0");
  void *hanlde = nullptr;
  MOCKER(mmDlopen).stubs().will(returnValue(hanlde));
  FlowFuncProcessor processor(params, flow_func_name, input_queue_infos, all_output_queue_infos, output_queue_indexes);
  processor.SetInputAlignAttrs(100, 10000, false);

  auto ret = processor.Init(0);
  EXPECT_EQ(ret, FLOW_FUNC_FAILED);
}

TEST_F(FlowFuncProcessorUTest, flow_func_proecssor_not_register) {
  std::string flow_func_name = "flow_func_proecssor_not_register";
  std::string pp_name = "pp0";
  std::string lib_path = "/xxx";
  auto input_queue_infos = UdfTestHelper::CreateQueueDevInfos({1});
  std::vector<uint32_t> output_queue_indexes = {0};
  auto all_output_queue_infos = UdfTestHelper::CreateQueueDevInfos({2, 3});

  std::shared_ptr<FlowFuncParams> params(
      new (std::nothrow) FlowFuncParams(pp_name, input_queue_infos.size(), output_queue_indexes.size(), 0, 0));
  params->SetInstanceName(pp_name + "@0_1_1@0@0");
  FlowFuncProcessor processor(params, flow_func_name, input_queue_infos, all_output_queue_infos, output_queue_indexes);
  auto ret = processor.Init(0);
  EXPECT_EQ(ret, FLOW_FUNC_FAILED);
}

TEST_F(FlowFuncProcessorUTest, normal_test) {
  std::string flow_func_name1 = "processor_ut_Proc1";
  std::string flow_func_name2 = "processor_ut_Proc2";
  std::string pp_name = "pp0";
  std::string lib_path = "/xxx";
  auto input1_queue_infos = UdfTestHelper::CreateQueueDevInfos({1});
  auto input2_queue_infos = UdfTestHelper::CreateQueueDevInfos({2});
  std::vector<uint32_t> output1_queue_indexes = {0};
  std::vector<uint32_t> output2_queue_indexes = {1};
  auto all_output_queue_infos = UdfTestHelper::CreateQueueDevInfos({3, 4});

  std::shared_ptr<FlowFuncParams> params(new (std::nothrow) FlowFuncParams(
      pp_name, input1_queue_infos.size() + input2_queue_infos.size(), all_output_queue_infos.size(), 0, 0));
  params->SetInstanceName(pp_name + "@0_1_1@0@0");
  FlowFuncProcessor processor1(params, flow_func_name1, input1_queue_infos, all_output_queue_infos,
                               output1_queue_indexes);
  processor1.SetInputAlignAttrs(100, 10000, false);
  auto ret = processor1.Init(0);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = processor1.InitFlowFunc();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  FlowFuncProcessor processor2(params, flow_func_name2, input2_queue_infos, all_output_queue_infos,
                               output2_queue_indexes);
  processor1.SetInputAlignAttrs(100, 10000, true);
  ret = processor2.Init(0);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = processor2.InitFlowFunc();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  bool sch_ret = processor1.Schedule(0U);
  EXPECT_TRUE(sch_ret);
  sch_ret = processor2.Schedule(0U);
  EXPECT_TRUE(sch_ret);
}

TEST_F(FlowFuncProcessorUTest, normal_stream_input_test) {
  std::string flow_func_name1 = "processor_ut_Proc1";
  std::string flow_func_name3 = "processor_ut_Proc3";
  std::string pp_name = "pp0";
  std::string lib_path = "/xxx";
  auto input1_queue_infos = UdfTestHelper::CreateQueueDevInfos({1});
  auto input2_queue_infos = UdfTestHelper::CreateQueueDevInfos({2});
  std::vector<uint32_t> output1_queue_indexes = {0};
  std::vector<uint32_t> output2_queue_indexes = {1};
  auto all_output_queue_infos = UdfTestHelper::CreateQueueDevInfos({3, 4});

  std::shared_ptr<FlowFuncParams> params(new (std::nothrow) FlowFuncParams(
      pp_name, input1_queue_infos.size() + input2_queue_infos.size(), all_output_queue_infos.size(), 0, 0));
  params->SetInstanceName(pp_name + "@0_1_1@0@0");
  params->SetStreamInputFuncNames({flow_func_name3});
  FlowFuncProcessor processor1(params, flow_func_name1, input1_queue_infos, all_output_queue_infos,
                               output1_queue_indexes);
  processor1.SetInputAlignAttrs(100, 10000, false);
  auto ret = processor1.Init(0);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = processor1.InitFlowFunc();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  FlowFuncProcessor processor2(params, flow_func_name3, input2_queue_infos, all_output_queue_infos,
                               output2_queue_indexes);
  ret = processor2.Init(0);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = processor2.InitFlowFunc();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  bool sch_ret = processor1.Schedule(0U);
  EXPECT_TRUE(sch_ret);
  sch_ret = processor2.Schedule(0U);
  EXPECT_TRUE(sch_ret);
  FlowFuncConfigManager::config_ = nullptr;
  EXPECT_EQ(FlowFuncConfigManager::GetConfig()->GetWorkerSchedGroupId(), 0);
  EXPECT_EQ(FlowFuncConfigManager::GetConfig()->GetCurrentSchedGroupId(), 0);
  EXPECT_EQ(FlowFuncConfigManager::GetConfig()->GetCurrentSchedThreadIdx(), 0);
  EXPECT_EQ(FlowFuncConfigManager::GetConfig()->GetFlowMsgQueueSchedGroupId(), 0);
  EXPECT_EQ(FlowFuncConfigManager::GetConfig()->GetWorkerNum(), 0);
  GlobalConfig::Instance().SetWorkerNum(2);
  FlowFuncConfigManager::SetConfig(std::shared_ptr<FlowFuncConfig>(&GlobalConfig::Instance(), [](FlowFuncConfig *) {}));
  EXPECT_EQ(FlowFuncConfigManager::GetConfig()->GetWorkerNum(), 2);
  sch_ret = processor1.Schedule(0U);
  EXPECT_TRUE(sch_ret);
  sch_ret = processor2.Schedule(0U);
  EXPECT_TRUE(sch_ret);
}

TEST_F(FlowFuncProcessorUTest, normal_test_dummy_q) {
  std::string flow_func_name1 = "processor_ut_Proc1";
  std::string pp_name = "pp0";
  std::string lib_path = "/xxx";
  auto input1_queue_infos = UdfTestHelper::CreateQueueDevInfos({1});
  std::vector<uint32_t> output1_queue_indexes = {0};
  auto all_output_queue_infos = UdfTestHelper::CreateQueueDevInfos({UINT32_MAX});

  std::shared_ptr<FlowFuncParams> params(
      new (std::nothrow) FlowFuncParams(pp_name, input1_queue_infos.size(), all_output_queue_infos.size(), 0, 0));
  params->SetInstanceName(pp_name + "@0_1_1@0@0");
  FlowFuncProcessor processor1(params, flow_func_name1, input1_queue_infos, all_output_queue_infos,
                               output1_queue_indexes);
  auto ret = processor1.Init(0);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = processor1.InitFlowFunc();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  bool sch_ret = processor1.Schedule(0U);
  EXPECT_TRUE(sch_ret);
}

TEST_F(FlowFuncProcessorUTest, single_func_re_init_and_proc_pend_test) {
  std::string flow_func_name1 = "flow_func_for_ut_normal";
  std::string pp_name = "re_init_and_proc_pend_pp0";
  std::string lib_path = "/xxx";
  auto input1_queue_infos = UdfTestHelper::CreateQueueDevInfos({1});
  std::vector<uint32_t> output_queue_indexes = {0};
  auto output_queue_infos = UdfTestHelper::CreateQueueDevInfos({4});

  std::shared_ptr<FlowFuncParams> params(
      new (std::nothrow) FlowFuncParams(pp_name, input1_queue_infos.size(), output_queue_infos.size(), 0, 0));
  ff::udf::AttrValue proto_attr;
  proto_attr.set_b(true);
  auto attr_impl = std::make_shared<AttrValueImpl>(proto_attr);
  std::map<std::string, std::shared_ptr<const AttrValue>> attr_map;
  attr_map.emplace("need_re_init_attr", attr_impl);
  attr_map.emplace("need_proc_pend_attr", attr_impl);

  params->SetAttrMap(attr_map);
  params->SetInstanceName(pp_name + "@0_1_1@0@0");
  FlowFuncProcessor processor1(params, flow_func_name1, input1_queue_infos, output_queue_infos, output_queue_indexes);
  auto ret = processor1.Init(0);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = processor1.InitFlowFunc();
  EXPECT_EQ(ret, FLOW_FUNC_ERR_INIT_AGAIN);
  ret = processor1.InitFlowFunc();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  bool sch_ret = processor1.Schedule(0U);
  EXPECT_FALSE(sch_ret);
  sch_ret = processor1.Schedule(0U);
  EXPECT_TRUE(sch_ret);
}

TEST_F(FlowFuncProcessorUTest, multi_func_re_init_and_proc_pend_test) {
  std::string flow_func_name1 = "processor_ut_Proc1";
  std::string flow_func_name2 = "processor_ut_Proc2";
  std::string pp_name = "re_init_and_proc_pend_pp0";
  std::string lib_path = "/xxx";
  auto input1_queue_infos = UdfTestHelper::CreateQueueDevInfos({1});
  auto input2_queue_infos = UdfTestHelper::CreateQueueDevInfos({2});
  std::vector<uint32_t> output_queue_indexes = {4};
  auto all_output_queue_infos = UdfTestHelper::CreateQueueDevInfos({3, 4});

  std::shared_ptr<FlowFuncParams> params(new (std::nothrow) FlowFuncParams(
      pp_name, input1_queue_infos.size() + input2_queue_infos.size(), all_output_queue_infos.size(), 0, 0));
  ff::udf::AttrValue proto_attr;
  proto_attr.set_b(true);
  auto attr_impl = std::make_shared<AttrValueImpl>(proto_attr);
  std::map<std::string, std::shared_ptr<const AttrValue>> attr_map;
  attr_map.emplace("need_re_init_attr", attr_impl);
  attr_map.emplace("need_proc_pend_attr", attr_impl);

  params->SetAttrMap(attr_map);
  params->SetInstanceName(pp_name + "@0_1_1@0@0");
  FlowFuncProcessor processor1(params, flow_func_name1, input1_queue_infos, all_output_queue_infos,
                               output_queue_indexes);
  auto ret = processor1.Init(0);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = processor1.InitFlowFunc();
  EXPECT_EQ(ret, FLOW_FUNC_ERR_INIT_AGAIN);
  ret = processor1.InitFlowFunc();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  FlowFuncProcessor processor2(params, flow_func_name2, input2_queue_infos, all_output_queue_infos,
                               output_queue_indexes);
  ret = processor2.Init(0);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = processor2.InitFlowFunc();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  bool sch_ret = processor1.Schedule(0U);
  EXPECT_FALSE(sch_ret);
  sch_ret = processor1.Schedule(0U);
  EXPECT_TRUE(sch_ret);

  sch_ret = processor2.Schedule(0U);
  EXPECT_FALSE(sch_ret);
  sch_ret = processor2.Schedule(0U);
  EXPECT_TRUE(sch_ret);
}

namespace {
int halMbufGetPrivInfoStubInvalid(Mbuf *mbuf, void **priv, unsigned int *size) {
  MbufImpl *mbuf_impl = reinterpret_cast<MbufImpl *>(mbuf);
  // start from first byte.
  *priv = &mbuf_impl->reserve_head;
  *size = 256 + 1;
  return DRV_ERROR_NONE;
}
}  // namespace

TEST_F(FlowFuncProcessorUTest, copy_head_msg_failed) {
  MOCKER(halMbufGetPrivInfo).stubs().will(invoke(halMbufGetPrivInfoStubInvalid));
  std::string flow_func_name = "processor_ut_Proc1";
  std::string pp_name = "pp0";
  std::string lib_path = "/xxx";
  auto input_queue_infos = UdfTestHelper::CreateQueueDevInfos({1});
  std::vector<uint32_t> output_queue_indexes = {0};
  auto all_output_queue_infos = UdfTestHelper::CreateQueueDevInfos({2});

  std::shared_ptr<FlowFuncParams> params(
      new (std::nothrow) FlowFuncParams(pp_name, input_queue_infos.size(), all_output_queue_infos.size(), 0, 0));
  params->SetInstanceName(pp_name + "@0_1_1@0@0");
  FlowFuncProcessor processor(params, flow_func_name, input_queue_infos, all_output_queue_infos, output_queue_indexes);
  auto ret = processor.Init(0);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = processor.InitFlowFunc();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  void *buf = nullptr;
  halQueueDeQueue(0, 0, &buf);
  Mbuf *mbuf = static_cast<Mbuf *>(buf);
  std::vector<Mbuf *> data = {mbuf};
  processor.SetInputData(data);
  EXPECT_NE(processor.current_trans_id_, UINT64_MAX);
  EXPECT_FALSE(processor.status_ == FlowFuncProcessorStatus::kCallFlowFunc);
  if (data[0] != nullptr) {
    halMbufFreeStub(data[0]);
  }
}

TEST_F(FlowFuncProcessorUTest, set_output_null_tensor) {
  std::string flow_func_name = "processor_ut_Proc1";
  std::string pp_name = "pp0";
  std::string lib_path = "/xxx";
  auto input_queue_infos = UdfTestHelper::CreateQueueDevInfos({1});
  std::vector<uint32_t> output_queue_indexes = {0};
  auto all_output_queue_infos = UdfTestHelper::CreateQueueDevInfos({2});

  std::shared_ptr<FlowFuncParams> params(
      new (std::nothrow) FlowFuncParams(pp_name, input_queue_infos.size(), all_output_queue_infos.size(), 0, 0));
  params->SetInstanceName(pp_name + "@0_1_1@0@0");
  FlowFuncProcessor processor(params, flow_func_name, input_queue_infos, all_output_queue_infos, output_queue_indexes);
  auto ret = processor.Init(0);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = processor.InitFlowFunc();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::vector<int64_t> shape = {0};
  TensorDataType data_type = TensorDataType::DT_INT32;
  auto mbuf_flow_msg = MbufFlowMsg::AllocEmptyTensorMsg(0, {});
  uint64_t expect_data_size = CalcDataSize(shape, data_type);
  uint64_t expect_element_cnt = CalcElementCnt(shape);
  EXPECT_NE(mbuf_flow_msg, nullptr);
  auto mbuf_tensor = mbuf_flow_msg->GetTensor();
  EXPECT_EQ(mbuf_tensor, nullptr);
  processor.SetOutput(0, mbuf_flow_msg);
  // for coverage
  processor.is_stream_input_ = true;
  auto mbuf_flow_msg1 = MbufFlowMsg::AllocEmptyTensorMsg(0, {});
  EXPECT_NE(mbuf_flow_msg1, nullptr);
  processor.SetOutput(0, mbuf_flow_msg1);
}

TEST_F(FlowFuncProcessorUTest, set_input_null_tensor) {
  std::string flow_func_name = "processor_ut_Proc1";
  std::string pp_name = "pp0";
  std::string lib_path = "/xxx";
  auto input_queue_infos = UdfTestHelper::CreateQueueDevInfos({1});
  std::vector<uint32_t> output_queue_indexes = {0};
  auto all_output_queue_infos = UdfTestHelper::CreateQueueDevInfos({2});

  std::shared_ptr<FlowFuncParams> params(
      new (std::nothrow) FlowFuncParams(pp_name, input_queue_infos.size(), all_output_queue_infos.size(), 0, 0));
  params->SetInstanceName(pp_name + "@0_1_1@0@0");
  FlowFuncProcessor processor(params, flow_func_name, input_queue_infos, all_output_queue_infos, output_queue_indexes);
  auto ret = processor.Init(0);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = processor.InitFlowFunc();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  std::vector<int64_t> shape = {0};
  TensorDataType data_type = TensorDataType::DT_INT32;
  auto mbuf_flow_msg = MbufFlowMsg::AllocEmptyTensorMsg(0, {});
  uint64_t expect_data_size = CalcDataSize(shape, data_type);
  uint64_t expect_element_cnt = CalcElementCnt(shape);
  EXPECT_NE(mbuf_flow_msg, nullptr);
  auto mbuf_tensor = mbuf_flow_msg->GetTensor();
  EXPECT_EQ(mbuf_tensor, nullptr);
  Mbuf *mbuf = nullptr;
  halMbufCopyRef(mbuf_flow_msg->GetMbuf(), &mbuf);
  std::vector<Mbuf *> data = {mbuf};
  processor.SetInputData(data);
}

TEST_F(FlowFuncProcessorUTest, publish_failed_republish_failed) {
  std::string flow_func_name = "processor_ut_Proc1";
  std::string pp_name = "pp0";
  std::string lib_path = "/xxx";
  auto input_queue_infos = UdfTestHelper::CreateQueueDevInfos({1});
  std::vector<uint32_t> output_queue_indexes = {0};
  auto all_output_queue_infos = UdfTestHelper::CreateQueueDevInfos({2, 3});

  std::shared_ptr<FlowFuncParams> params(
      new (std::nothrow) FlowFuncParams(pp_name, input_queue_infos.size(), all_output_queue_infos.size(), 0, 0));
  params->SetInstanceName(pp_name + "@0_1_1@0@0");
  FlowFuncProcessor processor(params, flow_func_name, input_queue_infos, all_output_queue_infos, output_queue_indexes);

  auto ret = processor.Init(0);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = processor.InitFlowFunc();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  MOCKER(halQueueEnQueue).stubs().will(returnValue(DRV_ERROR_QUEUE_FULL)).then(invoke(halQueueEnQueueStub));
  bool sch_ret = processor.Schedule(0U);
  EXPECT_TRUE(sch_ret);
}

TEST_F(FlowFuncProcessorUTest, dequeue_empty) {
  std::string flow_func_name = "processor_ut_Proc1";
  std::string pp_name = "pp0";
  std::string lib_path = "/xxx";
  auto input_queue_infos = UdfTestHelper::CreateQueueDevInfos({1});
  std::vector<uint32_t> output_queue_indexes = {0};
  auto all_output_queue_infos = UdfTestHelper::CreateQueueDevInfos({2, 3});

  std::shared_ptr<FlowFuncParams> params(
      new (std::nothrow) FlowFuncParams(pp_name, input_queue_infos.size(), all_output_queue_infos.size(), 0, 0));
  params->SetInstanceName(pp_name + "@0_1_1@0@0");
  FlowFuncProcessor processor(params, flow_func_name, input_queue_infos, all_output_queue_infos, output_queue_indexes);
  auto ret = processor.Init(0);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = processor.InitFlowFunc();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  MOCKER(halQueueDeQueue).stubs().will(returnValue(DRV_ERROR_QUEUE_EMPTY));
  bool sch_ret = processor.Schedule(0U);
  EXPECT_FALSE(sch_ret);
  EXPECT_TRUE(processor.IsOk());
}

TEST_F(FlowFuncProcessorUTest, dequeue_failed) {
  std::string flow_func_name = "processor_ut_Proc1";
  std::string pp_name = "pp0";
  std::string lib_path = "/xxx";
  auto input_queue_infos = UdfTestHelper::CreateQueueDevInfos({1});
  std::vector<uint32_t> output_queue_indexes = {0};
  auto all_output_queue_infos = UdfTestHelper::CreateQueueDevInfos({2, 3});

  std::shared_ptr<FlowFuncParams> params(
      new (std::nothrow) FlowFuncParams(pp_name, input_queue_infos.size(), all_output_queue_infos.size(), 0, 0));
  params->SetInstanceName(pp_name + "@0_1_1@0@0");
  FlowFuncProcessor processor(params, flow_func_name, input_queue_infos, all_output_queue_infos, output_queue_indexes);
  auto ret = processor.Init(0);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = processor.InitFlowFunc();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  MOCKER(halQueueDeQueue).stubs().will(returnValue(DRV_ERROR_INNER_ERR));
  bool sch_ret = processor.Schedule(0U);
  EXPECT_FALSE(sch_ret);
  EXPECT_FALSE(processor.IsOk());
}

TEST_F(FlowFuncProcessorUTest, publish_full) {
  std::string flow_func_name = "processor_ut_Proc1";
  std::string pp_name = "pp0";
  std::string lib_path = "/xxx";
  auto input_queue_infos = UdfTestHelper::CreateQueueDevInfos({1});
  std::vector<uint32_t> output_queue_indexes = {0};
  auto all_output_queue_infos = UdfTestHelper::CreateQueueDevInfos({2, 3});

  std::shared_ptr<FlowFuncParams> params(
      new (std::nothrow) FlowFuncParams(pp_name, input_queue_infos.size(), all_output_queue_infos.size(), 0, 0));
  params->SetInstanceName(pp_name + "@0_1_1@0@0");
  FlowFuncProcessor processor(params, flow_func_name, input_queue_infos, all_output_queue_infos, output_queue_indexes);
  auto ret = processor.Init(0);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = processor.InitFlowFunc();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  MOCKER(halQueueEnQueue).stubs().will(returnValue(DRV_ERROR_QUEUE_FULL));
  bool sch_ret = processor.Schedule(0U);
  EXPECT_FALSE(sch_ret);
  EXPECT_TRUE(processor.IsOk());
}

TEST_F(FlowFuncProcessorUTest, publish_failed) {
  std::string flow_func_name = "processor_ut_Proc1";
  std::string pp_name = "pp0";
  std::string lib_path = "/xxx";
  auto input_queue_infos = UdfTestHelper::CreateQueueDevInfos({1});
  std::vector<uint32_t> output_queue_indexes = {0};
  auto all_output_queue_infos = UdfTestHelper::CreateQueueDevInfos({2, 3});

  std::shared_ptr<FlowFuncParams> params(
      new (std::nothrow) FlowFuncParams(pp_name, input_queue_infos.size(), all_output_queue_infos.size(), 0, 0));
  params->SetInstanceName(pp_name + "@0_1_1@0@0");
  FlowFuncProcessor processor(params, flow_func_name, input_queue_infos, all_output_queue_infos, output_queue_indexes);
  auto ret = processor.Init(0);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = processor.InitFlowFunc();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  MOCKER(halQueueEnQueue).stubs().will(returnValue(DRV_ERROR_INNER_ERR));
  bool sch_ret = processor.Schedule(0U);
  EXPECT_FALSE(sch_ret);
  EXPECT_FALSE(processor.IsOk());
}

// first wait not full, not full event come need reschedule
TEST_F(FlowFuncProcessorUTest, full_to_not_full_test_need_reschedule) {
  std::string flow_func_name = "flow_func_for_ut_normal";
  std::string pp_name = "pp0";
  std::string lib_path = "/xxx";
  auto input_queue_infos = UdfTestHelper::CreateQueueDevInfos({1, 2});
  std::vector<uint32_t> output_queue_indexes = {0, 1};
  auto all_output_queue_infos = UdfTestHelper::CreateQueueDevInfos({3, 4});
  std::shared_ptr<FlowFuncParams> params(
      new (std::nothrow) FlowFuncParams(pp_name, input_queue_infos.size(), all_output_queue_infos.size(), 0, 0));
  params->SetInstanceName(pp_name + "@0_1_1@0@0");
  FlowFuncProcessor processor(params, flow_func_name, input_queue_infos, all_output_queue_infos, output_queue_indexes);
  auto ret = processor.Init(0);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = processor.InitFlowFunc();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  // first clear;
  processor.ClearNotFullEventFlag();
  auto need_retry = processor.CheckAndSetWaitNotFull();
  EXPECT_FALSE(need_retry);
  auto need_reschedule = processor.FullToNotFull();
  EXPECT_TRUE(need_reschedule);
}

// after clear not full event, then not full event come, check not full need retry
TEST_F(FlowFuncProcessorUTest, full_to_not_full_test_need_retry) {
  std::string flow_func_name = "flow_func_for_ut_normal";
  std::string pp_name = "pp0";
  std::string lib_path = "/xxx";
  auto input_queue_infos = UdfTestHelper::CreateQueueDevInfos({1, 2});
  std::vector<uint32_t> output_queue_indexes = {0, 1};
  auto all_output_queue_infos = UdfTestHelper::CreateQueueDevInfos({3, 4});
  std::shared_ptr<FlowFuncParams> params(
      new (std::nothrow) FlowFuncParams(pp_name, input_queue_infos.size(), all_output_queue_infos.size(), 0, 0));
  params->SetInstanceName(pp_name + "@0_1_1@0@0");
  FlowFuncProcessor processor(params, flow_func_name, input_queue_infos, all_output_queue_infos, output_queue_indexes);
  auto ret = processor.Init(0);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = processor.InitFlowFunc();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  // first clear;
  processor.ClearNotFullEventFlag();
  auto need_reschedule = processor.FullToNotFull();
  EXPECT_FALSE(need_reschedule);
  auto need_retry = processor.CheckAndSetWaitNotFull();
  EXPECT_TRUE(need_retry);
}

TEST_F(FlowFuncProcessorUTest, full_to_not_full_test_no_need_retry) {
  std::string flow_func_name = "flow_func_for_ut_normal";
  std::string pp_name = "pp0";
  std::string lib_path = "/xxx";
  auto input_queue_infos = UdfTestHelper::CreateQueueDevInfos({1, 2});
  std::vector<uint32_t> output_queue_indexes = {0, 1};
  auto all_output_queue_infos = UdfTestHelper::CreateQueueDevInfos({3, 4});
  std::shared_ptr<FlowFuncParams> params(
      new (std::nothrow) FlowFuncParams(pp_name, input_queue_infos.size(), all_output_queue_infos.size(), 0, 0));
  params->SetInstanceName(pp_name + "@0_1_1@0@0");
  FlowFuncProcessor processor(params, flow_func_name, input_queue_infos, all_output_queue_infos, output_queue_indexes);
  auto ret = processor.Init(0);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  // first clear;
  processor.ClearNotFullEventFlag();
  auto need_retry = processor.CheckAndSetWaitNotFull();
  EXPECT_FALSE(need_retry);
}

TEST_F(FlowFuncProcessorUTest, empty_to_not_empty_test_need_retry) {
  std::string flow_func_name = "flow_func_for_ut_normal";
  std::string pp_name = "pp0";
  std::string lib_path = "/xxx";
  auto input_queue_infos = UdfTestHelper::CreateQueueDevInfos({1, 2});
  std::vector<uint32_t> output_queue_indexes = {0, 1};
  auto all_output_queue_infos = UdfTestHelper::CreateQueueDevInfos({3, 4});
  std::shared_ptr<FlowFuncParams> params(
      new (std::nothrow) FlowFuncParams(pp_name, input_queue_infos.size(), all_output_queue_infos.size(), 0, 0));
  params->SetInstanceName(pp_name + "@0_1_1@0@0");
  FlowFuncProcessor processor(params, flow_func_name, input_queue_infos, all_output_queue_infos, output_queue_indexes);
  auto ret = processor.Init(0);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = processor.InitFlowFunc();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  // first clear;
  processor.ClearNotEmptyEventFlag();
  auto need_retry = processor.CheckAndSetWaitNotEmpty();
  EXPECT_FALSE(need_retry);
}

TEST_F(FlowFuncProcessorUTest, empty_to_not_empty_test_no_need_retry) {
  std::string flow_func_name = "flow_func_for_ut_normal";
  std::string pp_name = "pp0";
  std::string lib_path = "/xxx";
  auto input_queue_infos = UdfTestHelper::CreateQueueDevInfos({1, 2});
  std::vector<uint32_t> output_queue_indexes = {0, 1};
  auto all_output_queue_infos = UdfTestHelper::CreateQueueDevInfos({3, 4});
  std::shared_ptr<FlowFuncParams> params(
      new (std::nothrow) FlowFuncParams(pp_name, input_queue_infos.size(), all_output_queue_infos.size(), 0, 0));
  params->SetInstanceName(pp_name + "@0_1_1@0@0");
  FlowFuncProcessor processor(params, flow_func_name, input_queue_infos, all_output_queue_infos, output_queue_indexes);
  auto ret = processor.Init(0);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  // first clear;
  processor.ClearNotEmptyEventFlag();
  auto need_reschedule = processor.EmptyToNotEmpty();
  EXPECT_FALSE(need_reschedule);
  auto need_retry = processor.CheckAndSetWaitNotEmpty();
  EXPECT_TRUE(need_retry);
}

TEST_F(FlowFuncProcessorUTest, empty_to_not_empty_test_need_reschedule) {
  std::string flow_func_name = "flow_func_for_ut_normal";
  std::string pp_name = "pp0";
  std::string lib_path = "/xxx";
  auto input_queue_infos = UdfTestHelper::CreateQueueDevInfos({1, 2});
  std::vector<uint32_t> output_queue_indexes = {0, 1};
  auto all_output_queue_infos = UdfTestHelper::CreateQueueDevInfos({3, 4});
  std::shared_ptr<FlowFuncParams> params(
      new (std::nothrow) FlowFuncParams(pp_name, input_queue_infos.size(), all_output_queue_infos.size(), 0, 0));
  params->SetInstanceName(pp_name + "@0_1_1@0@0");
  FlowFuncProcessor processor(params, flow_func_name, input_queue_infos, all_output_queue_infos, output_queue_indexes);
  auto ret = processor.Init(0);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = processor.InitFlowFunc();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  // first clear;
  processor.ClearNotEmptyEventFlag();
  auto need_retry = processor.CheckAndSetWaitNotEmpty();
  EXPECT_FALSE(need_retry);
  auto need_reschedule = processor.EmptyToNotEmpty();
  EXPECT_TRUE(need_reschedule);
}

TEST_F(FlowFuncProcessorUTest, with_invoked_model) {
  std::string flow_func_name = "Proc2";
  std::string pp_name = "pp0";
  std::string lib_path = "/xxx";
  auto input_queue_infos = UdfTestHelper::CreateQueueDevInfos({1});
  std::vector<uint32_t> output_queue_indexes = {0};
  auto all_output_queue_infos = UdfTestHelper::CreateQueueDevInfos({2, 3});

  std::shared_ptr<FlowFuncParams> params(
      new (std::nothrow) FlowFuncParams(pp_name, input_queue_infos.size(), all_output_queue_infos.size(), 0, 0));
  std::unique_ptr<FlowModelImpl> impl(new (std::nothrow) FlowModelImpl(UdfTestHelper::CreateQueueDevInfos({3}),
                                                                       UdfTestHelper::CreateQueueDevInfos({4})));
  params->AddFlowModel("model_key", std::move(impl));
  params->SetInstanceName(pp_name + "@0_1_1@0@0");
  FlowFuncProcessor processor(params, flow_func_name, input_queue_infos, all_output_queue_infos, output_queue_indexes);
  auto ret = processor.Init(0);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = processor.InitFlowFunc();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
}

TEST_F(FlowFuncProcessorUTest, schedule_parall_test) {
  std::string flow_func_name1 = "TestProcSlow";
  std::string pp_name = "TestProcSlow";
  std::string lib_path = "/xxx";
  auto input1_queue_infos = UdfTestHelper::CreateQueueDevInfos({1});
  std::vector<uint32_t> output_queue_indexes = {0};
  auto all_output_queue_infos = UdfTestHelper::CreateQueueDevInfos({4});

  std::shared_ptr<FlowFuncParams> params(
      new (std::nothrow) FlowFuncParams(pp_name, input1_queue_infos.size(), all_output_queue_infos.size(), 0, 0));
  params->SetInstanceName(pp_name + "@0_1_1@0@0");
  ff::udf::AttrValue proto_attr;
  proto_attr.set_b(true);
  auto attr_impl = std::make_shared<AttrValueImpl>(proto_attr);
  FlowFuncProcessor processor1(params, flow_func_name1, input1_queue_infos, all_output_queue_infos,
                               output_queue_indexes);
  auto ret = processor1.Init(0);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = processor1.InitFlowFunc();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);

  std::thread t1([&processor1]() {
    bool sch_ret = processor1.Schedule(0U);
    EXPECT_TRUE(sch_ret);
  });

  std::thread t2([&processor1]() {
    usleep(200 * 1000);
    bool sch_ret = processor1.Schedule(0U);
    EXPECT_FALSE(sch_ret);
  });
  t1.join();
  t2.join();
}

TEST_F(FlowFuncProcessorUTest, normal_test_proxy) {
  std::string flow_func_name1 = "processor_ut_Proc1";
  std::string flow_func_name2 = "processor_ut_Proc2";
  std::string pp_name = "pp0";
  std::string lib_path = "/xxx";
  auto input1_queue_infos = UdfTestHelper::CreateQueueDevInfos({1}, Common::kDeviceTypeNpu, 1, true);
  auto input2_queue_infos = UdfTestHelper::CreateQueueDevInfos({2}, Common::kDeviceTypeNpu, 2, true);
  std::vector<uint32_t> output_queue_indexes = {0};
  auto all_output_queue_infos = UdfTestHelper::CreateQueueDevInfos({3}, Common::kDeviceTypeNpu, 3, true);

  std::shared_ptr<FlowFuncParams> params(new (std::nothrow) FlowFuncParams(
      pp_name, input1_queue_infos.size() + input2_queue_infos.size(), all_output_queue_infos.size(), 0, 0));
  params->SetInstanceName(pp_name + "@0_1_1@0@0");
  FlowFuncProcessor processor1(params, flow_func_name1, input1_queue_infos, all_output_queue_infos,
                               output_queue_indexes);
  auto ret = processor1.Init(0);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = processor1.InitFlowFunc();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  bool sch_ret = processor1.Schedule(0U);
  EXPECT_TRUE(sch_ret);
}

TEST_F(FlowFuncProcessorUTest, normal_test_proxy_dequeue_error) {
  MOCKER(halQueuePeek).stubs().will(returnValue(DRV_ERROR_INNER_ERR));
  std::string flow_func_name1 = "processor_ut_Proc1";
  std::string flow_func_name2 = "processor_ut_Proc2";
  std::string pp_name = "pp0";
  std::string lib_path = "/xxx";
  auto input1_queue_infos = UdfTestHelper::CreateQueueDevInfos({1}, Common::kDeviceTypeNpu, 1, true);
  auto input2_queue_infos = UdfTestHelper::CreateQueueDevInfos({2});
  std::vector<uint32_t> output_queue_indexes = {0};
  auto all_output_queue_infos = UdfTestHelper::CreateQueueDevInfos({3});

  std::shared_ptr<FlowFuncParams> params(new (std::nothrow) FlowFuncParams(
      pp_name, input1_queue_infos.size() + input2_queue_infos.size(), all_output_queue_infos.size(), 0, 0));
  params->SetInstanceName(pp_name + "@0_1_1@0@0");
  FlowFuncProcessor processor1(params, flow_func_name1, input1_queue_infos, all_output_queue_infos,
                               output_queue_indexes);
  auto ret = processor1.Init(0);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = processor1.InitFlowFunc();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  bool sch_ret = processor1.Schedule(0U);
  EXPECT_FALSE(processor1.IsOk());
}

TEST_F(FlowFuncProcessorUTest, dump_data_with_null_executor_failed) {
  UdfDumpManager::Instance().SetHostPid(9999);
  UdfDumpManager::Instance().EnableDump();
  UdfDumpManager::Instance().SetDumpPath("./");
  UdfDumpManager::Instance().SetDeviceId(0);
  EXPECT_EQ(UdfDumpManager::Instance().Init(), FLOW_FUNC_SUCCESS);
  std::string flow_func_name1 = "processor_ut_Proc1";
  std::string flow_func_name2 = "processor_ut_Proc2";
  std::string pp_name = "pp0";
  std::string lib_path = "/xxx";
  auto input1_queue_infos = UdfTestHelper::CreateQueueDevInfos({1}, Common::kDeviceTypeNpu, 1, true);
  auto input2_queue_infos = UdfTestHelper::CreateQueueDevInfos({2}, Common::kDeviceTypeNpu, 2, true);
  std::vector<uint32_t> output_queue_indexes = {0};
  auto all_output_queue_infos = UdfTestHelper::CreateQueueDevInfos({3}, Common::kDeviceTypeNpu, 3, true);

  std::shared_ptr<FlowFuncParams> params(new (std::nothrow) FlowFuncParams(
      pp_name, input1_queue_infos.size() + input2_queue_infos.size(), all_output_queue_infos.size(), 0, 0));
  params->SetInstanceName(pp_name + "@0_1_1@0@0");
  FlowFuncProcessor processor1(params, flow_func_name1, input1_queue_infos, all_output_queue_infos,
                               output_queue_indexes);
  auto ret = processor1.Init(0);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = processor1.InitFlowFunc();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  bool sch_ret = processor1.Schedule(0U);
  EXPECT_TRUE(sch_ret);
  UdfDumpManager::Instance().ClearDumpInfo();
}

TEST_F(FlowFuncProcessorUTest, dump_data_with_valid_executor_success) {
  UdfDumpManager::Instance().SetHostPid(9999);
  UdfDumpManager::Instance().EnableDump();
  UdfDumpManager::Instance().SetDumpPath("./");
  UdfDumpManager::Instance().SetDeviceId(0);
  EXPECT_EQ(UdfDumpManager::Instance().Init(), FLOW_FUNC_SUCCESS);
  std::string flow_func_name1 = "processor_ut_Proc1";
  std::string flow_func_name2 = "processor_ut_Proc2";
  std::string pp_name = "pp0";
  std::string lib_path = "/xxx";
  auto input1_queue_infos = UdfTestHelper::CreateQueueDevInfos({1}, Common::kDeviceTypeNpu, 1, true);
  auto input2_queue_infos = UdfTestHelper::CreateQueueDevInfos({2}, Common::kDeviceTypeNpu, 2, true);
  std::vector<uint32_t> output_queue_indexes = {0};
  auto all_output_queue_infos = UdfTestHelper::CreateQueueDevInfos({3}, Common::kDeviceTypeNpu, 3, true);
  auto async_executor = std::make_shared<AsyncExecutor>("test_th_", 1);
  std::shared_ptr<FlowFuncParams> params(new (std::nothrow) FlowFuncParams(
      pp_name, input1_queue_infos.size() + input2_queue_infos.size(), all_output_queue_infos.size(), 0, 0));
  params->SetInstanceName(pp_name + "@0_1_1@0@0");
  FlowFuncProcessor processor1(params, flow_func_name1, input1_queue_infos, all_output_queue_infos,
                               output_queue_indexes, async_executor);
  auto ret = processor1.Init(0);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = processor1.InitFlowFunc();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  bool sch_ret = processor1.Schedule(0U);
  EXPECT_TRUE(sch_ret);
  processor1.DumpOutputData("name", 0, nullptr, 0);
  async_executor.reset();
  UdfDumpManager::Instance().ClearDumpInfo();
}

TEST_F(FlowFuncProcessorUTest, set_input_not_aligned) {
  std::string flow_func_name1 = "processor_ut_Proc1";
  std::string pp_name = "pp0";
  std::string lib_path = "/xxx";
  auto input1_queue_infos = UdfTestHelper::CreateQueueDevInfos({1, 2});
  std::vector<uint32_t> output_queue_indexes = {3};
  auto all_output_queue_infos = UdfTestHelper::CreateQueueDevInfos({3});

  std::shared_ptr<FlowFuncParams> params(
      new (std::nothrow) FlowFuncParams(pp_name, input1_queue_infos.size(), all_output_queue_infos.size(), 0, 0));
  params->SetInstanceName(pp_name + "@0_1_1@0@0");
  FlowFuncProcessor processor1(params, flow_func_name1, input1_queue_infos, all_output_queue_infos,
                               output_queue_indexes);
  processor1.CreateRunContext(0);
  std::vector<Mbuf *> data(2);
  ScopeGuard guard([&data]() {
    for (auto &buf : data) {
      if (buf != nullptr) {
        halMbufFree(buf);
        buf = nullptr;
      }
    }
  });

  halMbufAlloc(1028, &data[0]);
  halMbufSetDataLen(data[0], 1028);
  MbufImpl *mbuf_impl = (MbufImpl *)data[0];
  processor1.SetInputData(data);
  ASSERT_NE(processor1.input_data_[1], nullptr);
  EXPECT_EQ(processor1.input_data_[1]->GetRetCode(), FLOW_FUNC_ERR_DATA_ALIGN_FAILED);
  processor1.SendEventToExecutor(UdfEvent::kEventIdFlowFuncReportStatus, FLOW_FUNC_SUCCESS);
}

TEST_F(FlowFuncProcessorUTest, NeedReplenishSchedule_status_check) {
  std::string flow_func_name1 = "processor_ut_Proc1";
  std::string pp_name = "pp0";
  std::string lib_path = "/xxx";
  auto input1_queue_infos = UdfTestHelper::CreateQueueDevInfos({1, 2});
  std::vector<uint32_t> output_queue_indexes = {3};
  auto all_output_queue_infos = UdfTestHelper::CreateQueueDevInfos({3});

  std::shared_ptr<FlowFuncParams> params(
      new (std::nothrow) FlowFuncParams(pp_name, input1_queue_infos.size(), all_output_queue_infos.size(), 0, 0));
  params->SetInstanceName(pp_name + "@0_1_1@0@0");
  FlowFuncProcessor processor1(params, flow_func_name1, input1_queue_infos, all_output_queue_infos,
                               output_queue_indexes);
  processor1.CreateRunContext(0);
  EXPECT_FALSE(processor1.NeedReplenishSchedule());
  processor1.status_ = FlowFuncProcessorStatus::kPrepareInputData;
  EXPECT_FALSE(processor1.NeedReplenishSchedule());
  auto over_time = std::chrono::milliseconds(100);
  processor1.last_schedule_time_ = std::chrono::steady_clock::now() - over_time;
  EXPECT_FALSE(processor1.NeedReplenishSchedule());
  over_time = std::chrono::milliseconds(3000);
  processor1.last_schedule_time_ = std::chrono::steady_clock::now() - over_time;
  EXPECT_TRUE(processor1.NeedReplenishSchedule());
}

TEST_F(FlowFuncProcessorUTest, schedule_when_suspend) {
  std::string flow_func_name = "processor_ut_Proc1";
  std::string pp_name = "pp0";
  std::string lib_path = "/xxx";
  auto input_queue_infos = UdfTestHelper::CreateQueueDevInfos({1});
  std::vector<uint32_t> output_queue_indexes = {0};
  auto all_output_queue_infos = UdfTestHelper::CreateQueueDevInfos({2, 3});

  std::shared_ptr<FlowFuncParams> params(
      new (std::nothrow) FlowFuncParams(pp_name, input_queue_infos.size(), all_output_queue_infos.size(), 0, 0));
  params->SetInstanceName(pp_name + "@0_1_1@0@0");
  FlowFuncProcessor processor(params, flow_func_name, input_queue_infos, all_output_queue_infos, output_queue_indexes);
  auto ret = processor.Init(0);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = processor.InitFlowFunc();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  processor.SetClearAndSuspend();
  EXPECT_TRUE(processor.try_clear_and_suspend_);

  processor.status_ = FlowFuncProcessorStatus::kReady;
  MOCKER(halQueueDeQueue).stubs().will(returnValue(DRV_ERROR_NONE)).then(returnValue(DRV_ERROR_QUEUE_EMPTY));
  MOCKER(halMbufFree).defaults().will(returnValue(DRV_ERROR_NONE));
  bool sch_ret = processor.Schedule(0U);
  EXPECT_FALSE(sch_ret);
  EXPECT_TRUE(processor.IsOk());
  EXPECT_TRUE(processor.cache_output_data_.empty());
  processor.ReleaseFuncWrapper();
  EXPECT_EQ(processor.func_wrapper_, nullptr);

  EXPECT_TRUE(processor.input_data_.empty());
  EXPECT_EQ(processor.wait_schedule_flag_, false);
  EXPECT_EQ(processor.input_ready_times_, 0);
  EXPECT_EQ(processor.call_flow_func_times_, 0);
  EXPECT_EQ(processor.schedule_finish_times_, 0);
  EXPECT_FALSE(processor.try_clear_and_suspend_);
  auto status = processor.status_;
  EXPECT_EQ(status, FlowFuncProcessorStatus::kSuspend);

  sch_ret = processor.Schedule(0U);
  EXPECT_FALSE(sch_ret);
  EXPECT_TRUE(processor.IsOk());
  status = processor.status_;
  EXPECT_EQ(status, FlowFuncProcessorStatus::kSuspend);
}

TEST_F(FlowFuncProcessorUTest, stream_input_processor_schedule_when_suspend) {
  std::string flow_func_name = "processor_ut_Proc3";
  std::string pp_name = "pp0";
  std::string lib_path = "/xxx";
  auto input_queue_infos = UdfTestHelper::CreateQueueDevInfos({1});
  std::vector<uint32_t> output_queue_indexes = {0};
  auto all_output_queue_infos = UdfTestHelper::CreateQueueDevInfos({2, 3});

  std::shared_ptr<FlowFuncParams> params(
      new (std::nothrow) FlowFuncParams(pp_name, input_queue_infos.size(), all_output_queue_infos.size(), 0, 0));
  params->SetInstanceName(pp_name + "@0_1_1@0@0");
  params->SetStreamInputFuncNames({flow_func_name});
  FlowFuncProcessor processor(params, flow_func_name, input_queue_infos, all_output_queue_infos, output_queue_indexes);
  auto ret = processor.Init(0);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = processor.InitFlowFunc();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  processor.SetClearAndSuspend();
  EXPECT_TRUE(processor.try_clear_and_suspend_);

  processor.status_ = FlowFuncProcessorStatus::kReady;
  MOCKER(halQueueDeQueue).stubs().will(returnValue(DRV_ERROR_NONE)).then(returnValue(DRV_ERROR_QUEUE_EMPTY));
  MOCKER(halMbufFree).defaults().will(returnValue(DRV_ERROR_NONE));
  bool sch_ret = processor.Schedule(0U);
  EXPECT_FALSE(sch_ret);
  EXPECT_TRUE(processor.IsOk());
  auto status = processor.status_;
  EXPECT_EQ(status, FlowFuncProcessorStatus::kSuspend);
}

TEST_F(FlowFuncProcessorUTest, schedule_when_recover_not_support_reset_state) {
  std::string flow_func_name = "processor_ut_Proc1";
  std::string pp_name = "pp0";
  std::string lib_path = "/xxx";
  auto input_queue_infos = UdfTestHelper::CreateQueueDevInfos({1});
  std::vector<uint32_t> output_queue_indexes = {0};
  auto all_output_queue_infos = UdfTestHelper::CreateQueueDevInfos({2, 3});

  std::shared_ptr<FlowFuncParams> params(
      new (std::nothrow) FlowFuncParams(pp_name, input_queue_infos.size(), all_output_queue_infos.size(), 0, 0));
  params->SetInstanceName(pp_name + "@0_1_1@0@0");
  FlowFuncProcessor processor(params, flow_func_name, input_queue_infos, all_output_queue_infos, output_queue_indexes);
  auto ret = processor.Init(0);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = processor.InitFlowFunc();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  // not support reset state, func wrapper will be set null after suspend.
  // status will change to init flow func after recover
  processor.ReleaseFuncWrapper();
  processor.SetClearAndRecover();
  EXPECT_TRUE(processor.try_clear_and_recover_);
  processor.status_ = FlowFuncProcessorStatus::kSuspend;
  MOCKER(halQueueDeQueue).stubs().will(returnValue(DRV_ERROR_QUEUE_EMPTY));
  MOCKER(halMbufFree).defaults().will(returnValue(DRV_ERROR_NONE));
  bool sch_ret = processor.Schedule(0U);
  EXPECT_FALSE(sch_ret);
  EXPECT_TRUE(processor.IsOk());
  auto status = processor.status_;
  EXPECT_EQ(status, FlowFuncProcessorStatus::kInitFlowFunc);
  EXPECT_FALSE(processor.try_clear_and_recover_);
}

TEST_F(FlowFuncProcessorUTest, schedule_when_recover_support_reset_state) {
  std::string flow_func_name = "processor_ut_Proc1";
  std::string pp_name = "pp0";
  std::string lib_path = "/xxx";
  auto input_queue_infos = UdfTestHelper::CreateQueueDevInfos({1});
  std::vector<uint32_t> output_queue_indexes = {0};
  auto all_output_queue_infos = UdfTestHelper::CreateQueueDevInfos({2, 3});

  std::shared_ptr<FlowFuncParams> params(
      new (std::nothrow) FlowFuncParams(pp_name, input_queue_infos.size(), all_output_queue_infos.size(), 0, 0));
  params->SetInstanceName(pp_name + "@0_1_1@0@0");
  FlowFuncProcessor processor(params, flow_func_name, input_queue_infos, all_output_queue_infos, output_queue_indexes);
  auto ret = processor.Init(0);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = processor.InitFlowFunc();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  // support reset state, status will change to ready
  processor.SetClearAndRecover();
  EXPECT_TRUE(processor.try_clear_and_recover_);
  processor.status_ = FlowFuncProcessorStatus::kSuspend;
  MOCKER(halQueueDeQueue).stubs().will(returnValue(DRV_ERROR_QUEUE_EMPTY));
  MOCKER(halMbufFree).defaults().will(returnValue(DRV_ERROR_NONE));
  bool sch_ret = processor.Schedule(0U);
  EXPECT_FALSE(sch_ret);
  EXPECT_TRUE(processor.IsOk());
  auto status = processor.status_;
  EXPECT_EQ(status, FlowFuncProcessorStatus::kReady);
  EXPECT_FALSE(processor.try_clear_and_recover_);
}
TEST_F(FlowFuncProcessorUTest, pre_check_failed) {
  std::string flow_func_name = "processor_ut_Proc1";
  std::string pp_name = "pp0";
  std::string lib_path = "/xxx";
  auto input_queue_infos = UdfTestHelper::CreateQueueDevInfos({1});
  std::vector<uint32_t> output_queue_indexes = {0};
  auto all_output_queue_infos = UdfTestHelper::CreateQueueDevInfos({2, 3});

  std::shared_ptr<FlowFuncParams> params(
      new (std::nothrow) FlowFuncParams(pp_name, input_queue_infos.size(), all_output_queue_infos.size(), 0, 0));
  params->SetInstanceName(pp_name + "@0_1_1@0@0");
  FlowFuncProcessor processor(params, flow_func_name, input_queue_infos, all_output_queue_infos, output_queue_indexes);
  auto ret = processor.Init(0);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = processor.InitFlowFunc();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  processor.SetClearAndRecover();
  EXPECT_TRUE(processor.try_clear_and_recover_);
  processor.status_ = FlowFuncProcessorStatus::kSuspend;
  processor.SetClearAndSuspend();
  EXPECT_TRUE(processor.try_clear_and_recover_);
  MOCKER(halQueueDeQueue).stubs().will(returnValue(DRV_ERROR_QUEUE_EMPTY));
  MOCKER(halMbufFree).defaults().will(returnValue(DRV_ERROR_NONE));
  bool sch_ret = processor.Schedule(0U);
  EXPECT_FALSE(sch_ret);
  EXPECT_FALSE(processor.IsOk());
  auto status = processor.status_;
  EXPECT_EQ(status, FlowFuncProcessorStatus::kProcError);
}

TEST_F(FlowFuncProcessorUTest, proc_call_func_err) {
  std::string flow_func_name = "processor_ut_Proc1";
  std::string pp_name = "pp0";
  std::string lib_path = "/xxx";
  auto input_queue_infos = UdfTestHelper::CreateQueueDevInfos({1});
  std::vector<uint32_t> output_queue_indexes = {0};
  auto all_output_queue_infos = UdfTestHelper::CreateQueueDevInfos({2, 3});
  MOCKER(halQueueDeQueue).stubs().will(returnValue(DRV_ERROR_QUEUE_EMPTY));
  MOCKER(halMbufFree).defaults().will(returnValue(DRV_ERROR_NONE));
  std::shared_ptr<FlowFuncParams> params(
      new (std::nothrow) FlowFuncParams(pp_name, input_queue_infos.size(), all_output_queue_infos.size(), 0, 0));
  params->SetInstanceName(pp_name + "@0_1_1@0@0");
  FlowFuncProcessor processor(params, flow_func_name, input_queue_infos, all_output_queue_infos, output_queue_indexes);
  auto ret = processor.Init(0);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = processor.InitFlowFunc();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  processor.ProcCallFlowFuncFailed();
  auto status = processor.status_;
  EXPECT_EQ(status, FlowFuncProcessorStatus::kProcError);
  processor.SetClearAndSuspend();
  processor.ProcCallFlowFuncFailed();
  status = processor.status_;
  EXPECT_EQ(status, FlowFuncProcessorStatus::kSuspend);
  EXPECT_EQ(processor.try_clear_and_suspend_, false);
}

TEST_F(FlowFuncProcessorUTest, call_flow_func_proc_exp) {
  std::string flow_func_name = "processor_ut_Proc1";
  std::string pp_name = "pp0";
  std::string lib_path = "/xxx";
  auto input_queue_infos = UdfTestHelper::CreateQueueDevInfos({1});
  std::vector<uint32_t> output_queue_indexes = {0};
  auto all_output_queue_infos = UdfTestHelper::CreateQueueDevInfos({2, 3});
  MOCKER(halQueueDeQueue).stubs().will(returnValue(DRV_ERROR_QUEUE_EMPTY));
  MOCKER(halMbufFree).defaults().will(returnValue(DRV_ERROR_NONE));
  std::shared_ptr<FlowFuncParams> params(
      new (std::nothrow) FlowFuncParams(pp_name, input_queue_infos.size(), all_output_queue_infos.size(), 0, 0));
  params->SetInstanceName(pp_name + "@0_1_1@0@0");
  params->SetScope("World");
  params->SetEnableRaiseException(true);
  FlowFuncProcessor processor(params, flow_func_name, input_queue_infos, all_output_queue_infos, output_queue_indexes);
  processor.align_max_cache_num_ = 10;
  auto ret = processor.Init(0);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = processor.InitFlowFunc();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  processor.status_ == FlowFuncProcessorStatus::kCallFlowFunc;
  processor.CallFlowFuncProcExp();

  UdfExceptionInfo exception_info{};
  const uint32_t priv_size = 256;
  MbufHeadMsg *head_msg = reinterpret_cast<MbufHeadMsg *>(exception_info.exp_context + priv_size - sizeof(MbufHeadMsg));
  head_msg->trans_id = 1;
  exception_info.exp_code = -1;
  exception_info.exp_context_size = priv_size;
  exception_info.trans_id = head_msg->trans_id;
  exception_info.user_context_id = 100;
  processor.RecordExceptionInfo(exception_info);

  UdfExceptionInfo exception_info2{};
  MbufHeadMsg *head_msg1 =
      reinterpret_cast<MbufHeadMsg *>(exception_info2.exp_context + priv_size - sizeof(MbufHeadMsg));
  head_msg1->trans_id = 100;
  exception_info2.exp_code = -1;
  exception_info2.exp_context_size = priv_size;
  exception_info2.trans_id = head_msg1->trans_id;
  exception_info2.user_context_id = 100;
  processor.RecordExceptionInfo(exception_info2);

  processor.RecordDeleteException(0);
  processor.RecordDeleteException(1);
  processor.RecordDeleteException(2);
  processor.CallFlowFuncProcExp();
  EXPECT_EQ(processor.trans_id_to_exception_infos_.size(), 0);
}

TEST_F(FlowFuncProcessorUTest, call_flow_func_proc_exp_failed_head_invalid) {
  std::string flow_func_name = "processor_ut_Proc1";
  std::string pp_name = "pp0";
  std::string lib_path = "/xxx";
  auto input_queue_infos = UdfTestHelper::CreateQueueDevInfos({1});
  std::vector<uint32_t> output_queue_indexes = {0};
  auto all_output_queue_infos = UdfTestHelper::CreateQueueDevInfos({2, 3});
  MOCKER(halQueueDeQueue).stubs().will(returnValue(DRV_ERROR_QUEUE_EMPTY));
  MOCKER(halMbufFree).defaults().will(returnValue(DRV_ERROR_NONE));
  std::shared_ptr<FlowFuncParams> params(
      new (std::nothrow) FlowFuncParams(pp_name, input_queue_infos.size(), all_output_queue_infos.size(), 0, 0));
  params->SetInstanceName(pp_name + "@0_1_1@0@0");
  params->SetScope("World");
  params->SetEnableRaiseException(true);
  FlowFuncProcessor processor(params, flow_func_name, input_queue_infos, all_output_queue_infos, output_queue_indexes);
  processor.align_max_cache_num_ = 10;
  auto ret = processor.Init(0);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = processor.InitFlowFunc();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  processor.status_ == FlowFuncProcessorStatus::kCallFlowFunc;

  UdfExceptionInfo exception_info{};
  const uint32_t priv_size = 256;
  MbufHeadMsg *head_msg = reinterpret_cast<MbufHeadMsg *>(exception_info.exp_context + priv_size - sizeof(MbufHeadMsg));
  head_msg->trans_id = 1;
  exception_info.exp_code = -1;
  // invalid length
  exception_info.exp_context_size = 1024;
  exception_info.trans_id = head_msg->trans_id;
  exception_info.user_context_id = 100;
  processor.RecordExceptionInfo(exception_info);

  processor.CallFlowFuncProcExp();
  EXPECT_EQ(processor.trans_id_to_exception_infos_.size(), 1);
  auto status = processor.status_;
  EXPECT_EQ(status, FlowFuncProcessorStatus::kProcError);
}

TEST_F(FlowFuncProcessorUTest, write_status_queue_failed_no_enable) {
  std::string flow_func_name = "processor_ut_Proc1";
  std::string pp_name = "pp0";
  std::string lib_path = "/xxx";
  auto input_queue_infos = UdfTestHelper::CreateQueueDevInfos({1});
  std::vector<uint32_t> output_queue_indexes = {0};
  auto all_output_queue_infos = UdfTestHelper::CreateQueueDevInfos({2, 3});
  MOCKER(halQueueDeQueue).stubs().will(returnValue(DRV_ERROR_QUEUE_EMPTY));
  MOCKER(halMbufFree).defaults().will(returnValue(DRV_ERROR_NONE));
  std::shared_ptr<FlowFuncParams> params(
      new (std::nothrow) FlowFuncParams(pp_name, input_queue_infos.size(), all_output_queue_infos.size(), 0, 0));
  params->SetInstanceName(pp_name + "@0_1_1@0@0");
  params->SetScope("World");
  params->SetEnableRaiseException(false);
  FlowFuncProcessor processor(params, flow_func_name, input_queue_infos, all_output_queue_infos, output_queue_indexes);
  auto ret = processor.Init(0);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = processor.InitFlowFunc();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ReportExceptionMbufGenFunc mbuf_gen_func = [this](const std::string &current_scope,
                                                    const UdfExceptionInfo &exception_info,
                                                    Mbuf *&mbuf_to_generate) -> int32_t { return FLOW_FUNC_SUCCESS; };
  ret = processor.WriteStatusOutputQueue(1, mbuf_gen_func);
  EXPECT_EQ(ret, FLOW_FUNC_FAILED);
}

TEST_F(FlowFuncProcessorUTest, write_status_queue_failed_without_input) {
  std::string flow_func_name = "processor_ut_Proc1";
  std::string pp_name = "pp0";
  std::string lib_path = "/xxx";
  auto input_queue_infos = UdfTestHelper::CreateQueueDevInfos({1});
  std::vector<uint32_t> output_queue_indexes = {0};
  auto all_output_queue_infos = UdfTestHelper::CreateQueueDevInfos({2, 3});
  MOCKER(halQueueDeQueue).stubs().will(returnValue(DRV_ERROR_QUEUE_EMPTY));
  MOCKER(halMbufFree).defaults().will(returnValue(DRV_ERROR_NONE));
  std::shared_ptr<FlowFuncParams> params(new (std::nothrow)
                                             FlowFuncParams(pp_name, 0, all_output_queue_infos.size(), 0, 0));
  params->SetInstanceName(pp_name + "@0_1_1@0@0");
  params->SetScope("World");
  params->SetEnableRaiseException(true);
  FlowFuncProcessor processor(params, flow_func_name, {}, all_output_queue_infos, output_queue_indexes);
  auto ret = processor.Init(0);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = processor.InitFlowFunc();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ReportExceptionMbufGenFunc mbuf_gen_func = [this](const std::string &current_scope,
                                                    const UdfExceptionInfo &exception_info,
                                                    Mbuf *&mbuf_to_generate) -> int32_t { return FLOW_FUNC_SUCCESS; };
  ret = processor.WriteStatusOutputQueue(1, mbuf_gen_func);
  EXPECT_EQ(ret, FLOW_FUNC_FAILED);
}

TEST_F(FlowFuncProcessorUTest, write_status_queue_failed_get_empty_exp) {
  std::string flow_func_name = "processor_ut_Proc1";
  std::string pp_name = "pp0";
  std::string lib_path = "/xxx";
  auto input_queue_infos = UdfTestHelper::CreateQueueDevInfos({1});
  std::vector<uint32_t> output_queue_indexes = {0};
  auto all_output_queue_infos = UdfTestHelper::CreateQueueDevInfos({2, 3});
  MOCKER(halQueueDeQueue).stubs().will(returnValue(DRV_ERROR_QUEUE_EMPTY));
  MOCKER(halMbufFree).defaults().will(returnValue(DRV_ERROR_NONE));
  std::shared_ptr<FlowFuncParams> params(
      new (std::nothrow) FlowFuncParams(pp_name, input_queue_infos.size(), all_output_queue_infos.size(), 0, 0));
  params->SetInstanceName(pp_name + "@0_1_1@0@0");
  params->SetScope("World");
  params->SetEnableRaiseException(true);
  FlowFuncProcessor processor(params, flow_func_name, input_queue_infos, all_output_queue_infos, output_queue_indexes);
  auto ret = processor.Init(0);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ret = processor.InitFlowFunc();
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  ReportExceptionMbufGenFunc mbuf_gen_func = [this](const std::string &current_scope,
                                                    const UdfExceptionInfo &exception_info,
                                                    Mbuf *&mbuf_to_generate) -> int32_t { return FLOW_FUNC_SUCCESS; };
  ret = processor.WriteStatusOutputQueue(1, mbuf_gen_func);
  EXPECT_EQ(ret, FLOW_FUNC_FAILED);
}
}  // namespace FlowFunc

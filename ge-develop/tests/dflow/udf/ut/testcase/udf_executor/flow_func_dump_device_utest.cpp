/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "securec.h"
#include "dlog_pub.h"
#include "flow_func/flow_func_defines.h"
#include "toolchain/dump/udf_dump.h"
#include "toolchain/dump/udf_dump_manager.h"
#include "flow_func/mbuf_flow_msg.h"
#include "mockcpp/mockcpp.hpp"
#define private public
#include "config/global_config.h"
#include "toolchain/dump/udf_dump_task_device.h"
#undef private
#include "mmpa/mmpa_api.h"
#include "common/data_utils.h"
#include "ascend_hal.h"

extern int32_t FlowFuncTestMain(int32_t argc, char *argv[]);
namespace FlowFunc {
namespace {
struct MbufImpl {
  uint32_t mbuf_size;
  uint8_t reserve_head[256 - sizeof(MbufHeadMsg)];
  MbufHeadMsg head_msg;
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
}  // namespace
class DumpDeviceUTest : public testing::Test {
 protected:
  virtual void SetUp() {
    back_is_on_device_ = GlobalConfig::Instance().IsOnDevice();
    GlobalConfig::on_device_ = true;
    MOCKER(halMbufAllocEx).defaults().will(invoke(halMbufAllocExStub));
    MOCKER(halMbufFree).defaults().will(invoke(halMbufFreeStub));
    MOCKER(halMbufGetBuffAddr).defaults().will(invoke(halMbufGetBuffAddrStub));
    MOCKER(halMbufGetBuffSize).defaults().will(invoke(halMbufGetBuffSizeStub));
    MOCKER(halMbufSetDataLen).defaults().will(invoke(halMbufSetDataLenStub));
    MOCKER(halMbufGetPrivInfo).defaults().will(invoke(halMbufGetPrivInfoStub));
    MOCKER(halMbufGetDataLen).defaults().will(invoke(halMbufGetDataLenStub));
  }

  virtual void TearDown() {
    GlobalMockObject::verify();
    GlobalConfig::on_device_ = back_is_on_device_;
    UdfDumpManager::Instance().ClearDumpInfo();
  }

 private:
  bool back_is_on_device_ = false;
};

TEST_F(DumpDeviceUTest, test_from_main_no_hostpid) {
  UdfDumpManager::Instance().ClearDumpInfo();
  char process_name[] = "flow_func_executor";
  char param_device_id_ok[] = "--device_id=1";
  char param_load_path_ok[] = "--load_path=./model/batch_model_test.proto";
  char param_grp_name_ok[] = "--group_name=Grp1";
  char dump_enable[] = "ge.exec.enableDump=1";
  char dump_path[] = "ge.exec.dumpPath=/usr/local";
  char dump_step[] = "ge.exec.dumpStep=0-5";
  char dump_mode[] = "ge.exec.dumpMode=all";
  char *argv[] = {process_name, param_device_id_ok, param_load_path_ok, param_grp_name_ok,
                  dump_enable,  dump_path,          dump_step,          dump_mode};
  int32_t argc = 8;
  int32_t ret = FlowFuncTestMain(argc, argv);
  EXPECT_NE(ret, 0);
  bool enable_dump = UdfDumpManager::Instance().IsEnableDump();
  EXPECT_EQ(enable_dump, true);
}

TEST_F(DumpDeviceUTest, test_from_main) {
  UdfDumpManager::Instance().ClearDumpInfo();
  char process_name[] = "flow_func_executor";
  char param_device_id_ok[] = "--device_id=1";
  char param_load_path_ok[] = "--load_path=./model/batch_model_test.proto";
  char param_grp_name_ok[] = "--group_name=Grp1";
  char dump_enable[] = "ge.exec.enableDump=1";
  char dump_path[] = "ge.exec.dumpPath=/usr/local";
  char dump_step[] = "ge.exec.dumpStep=0-5";
  char dump_mode[] = "ge.exec.dumpMode=all";
  ::setenv("ASCEND_HOSTPID", "123", 1);
  bool is_on_device = GlobalConfig::on_device_;
  GlobalConfig::on_device_ = true;
  char *argv[] = {process_name, param_device_id_ok, param_load_path_ok, param_grp_name_ok,
                  dump_enable,  dump_path,         dump_step,          dump_mode};
  int32_t argc = 8;
  int32_t ret = FlowFuncTestMain(argc, argv);
  EXPECT_NE(ret, 0);
  bool enable_dump = UdfDumpManager::Instance().IsEnableDump();
  EXPECT_EQ(enable_dump, true);
  const auto path = UdfDumpManager::Instance().GetDumpPath();
  EXPECT_EQ(path, "/usr/local");
  const auto mode = UdfDumpManager::Instance().GetDumpMode();
  EXPECT_EQ(mode, "all");
  const auto step = UdfDumpManager::Instance().GetDumpStep();
  EXPECT_EQ(step, "0-5");
  const auto pid = UdfDumpManager::Instance().GetHostPid();
  EXPECT_EQ(pid, 123);
  UdfDumpManager::Instance().InitAicpuDump();
  ::unsetenv("ASCEND_HOSTPID");
  GlobalConfig::on_device_ = is_on_device;
}

TEST_F(DumpDeviceUTest, dump_output_with_dump_input) {
  UdfDumpManager::Instance().EnableDump();
  UdfDumpManager::Instance().SetDumpMode("input");
  UdfDumpManager::Instance().ClearDumpInfo();
  MbufTensor *input_flow_tensor;
  std::string flow_func_info = "test_func1";
  int32_t ret = UdfDump::DumpOutput(flow_func_info, 0, input_flow_tensor, 0);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
}

TEST_F(DumpDeviceUTest, dump_input_success) {
  std::vector<MbufTensor *> input_flow_tensors;
  std::vector<int64_t> shape = {1, 2, 3};
  TensorDataType data_type = TensorDataType::DT_INT32;
  std::shared_ptr<MbufFlowMsg> mbuf_flow_msg = MbufFlowMsg::AllocTensorMsg(shape, data_type, 0, {});
  uint64_t expect_data_size = CalcDataSize(shape, data_type);
  uint64_t expect_element_cnt = CalcElementCnt(shape);
  EXPECT_NE(mbuf_flow_msg, nullptr);
  auto mbuf_tensor = mbuf_flow_msg->GetTensor();
  EXPECT_NE(mbuf_tensor, nullptr);
  input_flow_tensors.emplace_back(dynamic_cast<MbufTensor *>(mbuf_tensor));
  const std::string flow_func_info = "test_func1";
  UdfDumpTaskDevice udf_dump_task_ptr(flow_func_info, 1, 1);
  auto ret = udf_dump_task_ptr.PreProcessInput(input_flow_tensors);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(udf_dump_task_ptr.base_dump_data_.input_size(), 1);
  ret = udf_dump_task_ptr.DumpOpInfo("/var/log/", 0);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
}

TEST_F(DumpDeviceUTest, dump_input_with_null_tensor_success) {
  std::vector<MbufTensor *> input_flow_tensors(2);
  std::vector<int64_t> shape = {1, 2, 3};
  TensorDataType data_type = TensorDataType::DT_INT32;
  std::shared_ptr<MbufFlowMsg> mbuf_flow_msg = MbufFlowMsg::AllocTensorMsg(shape, data_type, 0, {});
  uint64_t expect_data_size = CalcDataSize(shape, data_type);
  uint64_t expect_element_cnt = CalcElementCnt(shape);
  EXPECT_NE(mbuf_flow_msg, nullptr);
  auto mbuf_tensor = mbuf_flow_msg->GetTensor();
  EXPECT_NE(mbuf_tensor, nullptr);
  input_flow_tensors[0] = nullptr;
  input_flow_tensors[1] = dynamic_cast<MbufTensor *>(mbuf_tensor);
  const std::string flow_func_info = "test_func1";
  UdfDumpTaskDevice udf_dump_task_ptr(flow_func_info, 1, 1);
  auto ret = udf_dump_task_ptr.PreProcessInput(input_flow_tensors);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(udf_dump_task_ptr.base_dump_data_.input_size(), 2);
  auto &input = udf_dump_task_ptr.base_dump_data_.input(0);
  EXPECT_EQ(input.size(), 0);
  ret = udf_dump_task_ptr.DumpOpInfo("/var/log/", 0);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
}

TEST_F(DumpDeviceUTest, dump_output_success) {
  std::vector<int64_t> shape = {1, 2, 3};
  TensorDataType data_type = TensorDataType::DT_INT32;
  std::shared_ptr<MbufFlowMsg> mbuf_flow_msg = MbufFlowMsg::AllocTensorMsg(shape, data_type, 0, {});
  uint64_t expect_data_size = CalcDataSize(shape, data_type);
  uint64_t expect_element_cnt = CalcElementCnt(shape);
  auto mbuf_tensor = dynamic_cast<MbufTensor *>(mbuf_flow_msg->GetTensor());
  const std::string flow_func_info = "test_func1";
  UdfDumpTaskDevice udf_dump_task_ptr(flow_func_info, 1, 1);
  auto ret = udf_dump_task_ptr.PreProcessOutput(0, mbuf_tensor);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(udf_dump_task_ptr.base_dump_data_.output_size(), 1);
  ret = udf_dump_task_ptr.DumpOpInfo("/var/log/", 0);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
}

TEST_F(DumpDeviceUTest, dump_output_with_index_1_success) {
  std::vector<int64_t> shape = {1, 2, 3};
  TensorDataType data_type = TensorDataType::DT_INT32;
  std::shared_ptr<MbufFlowMsg> mbuf_flow_msg = MbufFlowMsg::AllocTensorMsg(shape, data_type, 0, {});
  uint64_t expect_data_size = CalcDataSize(shape, data_type);
  uint64_t expect_element_cnt = CalcElementCnt(shape);
  auto mbuf_tensor = dynamic_cast<MbufTensor *>(mbuf_flow_msg->GetTensor());
  const std::string flow_func_info = "test_func1";
  UdfDumpTaskDevice udf_dump_task_ptr(flow_func_info, 1, 1);
  auto ret = udf_dump_task_ptr.PreProcessOutput(1, mbuf_tensor);
  EXPECT_EQ(ret, FLOW_FUNC_SUCCESS);
  EXPECT_EQ(udf_dump_task_ptr.base_dump_data_.output_size(), 2);
  auto &output = udf_dump_task_ptr.base_dump_data_.output(0);
  EXPECT_EQ(output.size(), 0);
}
}  // namespace FlowFunc
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
#include <gmock/gmock.h>
#include <mockcpp/mockcpp.hpp>
#include "hccl_stub.h"
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <runtime/rt.h>
#include <iostream>
#include <fstream>

#include <nlohmann/json.hpp>
#define private public
#define protected public
#include "hcom_executor.h"
#include "hcom_executor_internel.h"
#undef private
#undef protected
#include "llt_hccl_stub_ge.h"
#include "common/ge_common/ge_types.h"
#include "v80_rank_table.h"
#include "llt_hccl_stub_hccl.h"
#include "dlhccl_function.h"
#include "adapter_dlhcclfunc.h"

using namespace std;
using namespace hccl;

class HcomExecutorTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "HcomExecutorTest SetUP" << std::endl;
  }
  static void TearDownTestCase() {
    std::cout << "HcomExecutorTest TearDown" << std::endl;
  }
  virtual void SetUp() {
    std::cout << "A Test SetUP" << std::endl;
  }
  virtual void TearDown() {
    std::cout << "A Test TearDown" << std::endl;
  }
};

static HcclResult g_excutorStatus;
void setExecutorStatus(HcclResult status) {
  HCCL_INFO("this is setExecutorStatus fuction");
  g_excutorStatus = status;
}

void getExecutorStatus(HcclResult &status) {
  status = g_excutorStatus;
}

static std::vector<HcclResult> g_mutiexcutorStatus;
void setMutiExecutorStatus(HcclResult status) {
  HCCL_INFO("this is setExecutorStatus fuction");
  g_mutiexcutorStatus.push_back(status);
}

void getMutiExecutorStatus(HcclResult &status, int count) {
  if (g_mutiexcutorStatus.size() != count) {
    status = HCCL_E_INTERNAL;
    return;
  }
  for (int i = 0; i < g_mutiexcutorStatus.size(); i++) {
    if (g_mutiexcutorStatus[i] != HCCL_SUCCESS) {
      status = HCCL_E_INTERNAL;
      return;
    }
  }
  status = HCCL_SUCCESS;
  g_mutiexcutorStatus.clear();
  return;
}

ge::graphStatus FakeGetOption1(ge::GEThreadLocalContext *that, const std::string &optionExec,
                               std::string &dumpDebugValue) {
  nlohmann::json group_list = {{{"group_name", "aa"}, {"group_rank_list", {0, 1}}}};
  if (optionExec == ge::OPTION_EXEC_HCOM_GROUPLIST) {
    dumpDebugValue = group_list.dump();
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FakeGetOptionGrouplist(ge::GEThreadLocalContext *that, const std::string &optionExec,
                                       std::string &dumpDebugValue) {
  nlohmann::json rank_table = {{"status", "completed"},
                               {"deploy_mode", "lab"},
                               {"group_count", "1"},
                               {"chip_info", "910"},
                               {"board_id", "0x0000"},
                               {"para_plane_nic_location", "device"},
                               {"para_plane_nic_num", "1"},
                               {"para_plane_nic_name", {"eth0"}},
                               {"group_list",
                                {{
                                    {"group_name", ""},
                                    {"device_num", "1"},
                                    {"server_num", "1"},
                                    {"instance_count", "1"},
                                    {"instance_list",
                                     {{{"rank_id", "0"},
                                       {"server_id", "172.17.1.120"},
                                       {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.1.120"}}}}}}},
                                }}}};

  nlohmann::json group_list = {{{"group_name", "bbb"}, {"group_rank_list", {0, 1}}}};
  if (optionExec == ge::OPTION_EXEC_RANK_TABLE) {
    dumpDebugValue = rank_table.dump();
  } else if (optionExec == ge::OPTION_EXEC_HCOM_GROUPLIST) {
    dumpDebugValue = group_list.dump();
  } else if (optionExec == ge::OPTION_EXEC_RANK_ID) {
    dumpDebugValue.push_back('1');
  } else if (optionExec == ge::OPTION_EXEC_ROLE_TABLE_ADDR) {
    return !ge::GRAPH_SUCCESS;
  } else if (optionExec == ge::OPTION_EXEC_RANK_TABLE_ADDR) {
    dumpDebugValue = std::to_string(ge::PtrToValue(rank_table.dump().data()));
    return ge::GRAPH_SUCCESS;
  } else {
    dumpDebugValue.push_back('1');
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FakeGetOptionError1(ge::GEThreadLocalContext *that, const std::string &optionExec,
                                    std::string &dumpDebugValue) {
  if (optionExec == ge::OPTION_EXEC_HCOM_GROUPLIST) {
    dumpDebugValue = R"({"rank_map":[{"logic_rank_id":1,"model_rank_id":0},{"logic_rank_id":1,"model_rank_id":0]})";
  }
  return ge::GRAPH_SUCCESS;
}
#define HCCL_COM_DATA_SIZE 1024

TEST_F(HcomExecutorTest, ut_executor_initgroup) {
  set_board_id(0x0000);
  nlohmann::json rank_table = rank_table_910_1server_1rank;

  char file_name_t[] = "./ut_executor_initgroup.json";
  std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

  if (outfile.is_open()) {
    outfile << std::setw(1) << rank_table << std::endl;
    HCCL_INFO("open %s success", file_name_t);
  } else {
    HCCL_ERROR("open %s failed", file_name_t);
  }

  outfile.close();

  int ret = HCCL_SUCCESS;

  ret = hrtSetDevice(0);
  EXPECT_EQ(ret, HCCL_SUCCESS);

  // 走1910 4pring
  char *rank_table_file = "./ut_executor_initgroup.json";
  char *rank_ID = "0";

  MOCKER_CPP(&ge::GEThreadLocalContext::GetOption).stubs().will(invoke(FakeGetOption1));

  ret = hccl::HcomExecutor::GetInstance().InitGroup();
  EXPECT_EQ(ret, HCCL_SUCCESS);

  ret = HcomDestroy();
  EXPECT_EQ(ret, HCCL_SUCCESS);
  remove(file_name_t);

  GlobalMockObject::verify();
}

TEST_F(HcomExecutorTest, ut_executor_initgroup1) {
  set_board_id(0x0000);
  nlohmann::json rank_table = rank_table_910_1server_1rank;

  char file_name_t[] = "./ut_executor_initgroup1.json";
  std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

  if (outfile.is_open()) {
    outfile << std::setw(1) << rank_table << std::endl;
    HCCL_INFO("open %s success", file_name_t);
  } else {
    HCCL_ERROR("open %s failed", file_name_t);
  }

  outfile.close();

  int ret = HCCL_SUCCESS;

  ret = hrtSetDevice(0);
  EXPECT_EQ(ret, HCCL_SUCCESS);

  MOCKER_CPP(&ge::GEThreadLocalContext::GetOption).stubs().will(invoke(FakeGetOptionGrouplist));

  u32 rankId = 0;
  MOCKER(HcomGetRankId).stubs().with(mockcpp::any(), outBound(&rankId)).will(returnValue(HCCL_SUCCESS));

  MOCKER(HcomCreateGroup).stubs().will(returnValue(HCCL_SUCCESS));

  ret = hccl::HcomExecutor::GetInstance().InitGroup();
  EXPECT_EQ(ret, HCCL_SUCCESS);

  ret = HcomDestroy();
  EXPECT_EQ(ret, HCCL_SUCCESS);
  remove(file_name_t);

  GlobalMockObject::verify();
}

TEST_F(HcomExecutorTest, ut_executor_initgroup1_error) {
  set_board_id(0x0000);
  nlohmann::json rank_table = rank_table_910_1server_1rank;

  char file_name_t[] = "./ut_executor_initgroup1_error.json";
  std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

  if (outfile.is_open()) {
    outfile << std::setw(1) << rank_table << std::endl;
    HCCL_INFO("open %s success", file_name_t);
  } else {
    HCCL_ERROR("open %s failed", file_name_t);
  }

  outfile.close();

  int ret = HCCL_SUCCESS;

  ret = hrtSetDevice(0);
  EXPECT_EQ(ret, HCCL_SUCCESS);

  MOCKER_CPP(&ge::GEThreadLocalContext::GetOption).stubs().will(invoke(FakeGetOptionError1));

  u32 rankId = 0;
  MOCKER(HcomGetRankId).stubs().with(mockcpp::any(), outBound(&rankId)).will(returnValue(HCCL_SUCCESS));

  MOCKER(HcomCreateGroup).stubs().will(returnValue(HCCL_SUCCESS));

  ret = hccl::HcomExecutor::GetInstance().InitGroup();
  EXPECT_EQ(ret, HCCL_E_PARA);

  ret = HcomDestroy();
  EXPECT_EQ(ret, HCCL_SUCCESS);
  remove(file_name_t);

  GlobalMockObject::verify();
}

#if 1
TEST_F(HcomExecutorTest, ut_executor_broadcast) {
  set_board_id(0x0000);
  nlohmann::json rank_table = rank_table_910_1server_1rank;

  char file_name_t[] = "./ut_executor_broadcast.json";
  std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

  if (outfile.is_open()) {
    outfile << std::setw(1) << rank_table << std::endl;
    HCCL_INFO("open %s success", file_name_t);
  } else {
    HCCL_ERROR("open %s failed", file_name_t);
  }

  outfile.close();

  int ret = HCCL_SUCCESS;
  rtError_t rt_ret = RT_ERROR_NONE;
  rtStream_t stream;
  s8 *sendbuf;
  s32 rank = 0;
  s32 errors = 0;
  s32 count = HCCL_COM_DATA_SIZE;
  ret = hrtSetDevice(0);
  EXPECT_EQ(ret, HCCL_SUCCESS);

  MOCKER_CPP(&ge::GEThreadLocalContext::GetOption).stubs().will(invoke(FakeGetOptionGrouplist));

  ret = HcomExecInitialize();
  EXPECT_EQ(ret, HCCL_SUCCESS);

  HcclComm hcclComm;
  ret = HcomGetCommHandleByGroup(NULL, &hcclComm);
  EXPECT_EQ(ret, HCCL_SUCCESS);
  ret = hccl::HcomExecutor::GetInstance().InitHcclComm(nullptr, hcclComm);
  EXPECT_EQ(ret, HCCL_SUCCESS);

  rt_ret = rtStreamCreate(&stream, 5);
  EXPECT_EQ(rt_ret, RT_ERROR_NONE);
  sendbuf = (s8 *)sal_malloc(count * sizeof(s8));
  sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));

  for (int j = 0; j < count; j++) {
    sendbuf[j] = 2;
  }

  ret = HcomBroadcast("testbcast", sendbuf, count, HCCL_DATA_TYPE_INT8, 0, NULL, stream);
  EXPECT_EQ(ret, HCCL_SUCCESS);

  rt_ret = rtStreamSynchronize(stream);
  EXPECT_EQ(rt_ret, RT_ERROR_NONE);

  for (int j = 0; j < count; j++) {
    if (sendbuf[j] != 2) {
      HCCL_ERROR("\n rank:%d sendbuf[%d]:%f", rank, j, sendbuf[j]);
      errors++;
      break;
    }
  }

  for (int j = 0; j < count; j++) {
    sendbuf[j] = 2;
  }

  (void)HcomSetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
  ret = HcclBroadcast(sendbuf, count, HCCL_DATA_TYPE_INT8, 0, hcclComm, stream);
  EXPECT_EQ(ret, HCCL_SUCCESS);

  rt_ret = rtStreamSynchronize(stream);
  EXPECT_EQ(rt_ret, RT_ERROR_NONE);

  for (int j = 0; j < count; j++) {
    if (sendbuf[j] != 2) {
      HCCL_ERROR("\n rank:%d sendbuf[%d]:%f", rank, j, sendbuf[j]);
      errors++;
      break;
    }
  }

  ret = HcomExecFinalize();
  EXPECT_EQ(ret, HCCL_SUCCESS);

  sal_free(sendbuf);
  rt_ret = rtStreamDestroy(stream);

  ret = HcomDestroy();
  EXPECT_EQ(ret, HCCL_SUCCESS);
  remove(file_name_t);
  EXPECT_EQ(rt_ret, RT_ERROR_NONE);
  EXPECT_EQ(errors, 0);
}

TEST_F(HcomExecutorTest, ut_executor_allreduce) {
  nlohmann::json rank_table = rank_table_910_1server_1rank;

  char file_name_t[] = "./ut_executor_allreduce.json";
  std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

  if (outfile.is_open()) {
    outfile << std::setw(1) << rank_table << std::endl;
    HCCL_INFO("open %s success", file_name_t);
  } else {
    HCCL_ERROR("open %s failed", file_name_t);
  }

  outfile.close();

  int ret = HCCL_SUCCESS;
  rtError_t rt_ret = RT_ERROR_NONE;
  rtStream_t stream;
  s8 *sendbuf;
  s8 *recvbuf;
  s32 rank = 0;
  s32 errors = 0;
  s32 count = HCCL_COM_DATA_SIZE;
  ret = hrtSetDevice(0);
  EXPECT_EQ(ret, HCCL_SUCCESS);

  MOCKER_CPP(&ge::GEThreadLocalContext::GetOption).stubs().will(invoke(FakeGetOptionGrouplist));

  ret = HcomExecInitialize();
  EXPECT_EQ(ret, HCCL_SUCCESS);
  HcclComm hcclComm;
  ret = HcomGetCommHandleByGroup(NULL, &hcclComm);
  EXPECT_EQ(ret, HCCL_SUCCESS);
  ret = hccl::HcomExecutor::GetInstance().InitHcclComm(nullptr, hcclComm);
  EXPECT_EQ(ret, HCCL_SUCCESS);

  rt_ret = rtStreamCreate(&stream, 5);
  EXPECT_EQ(rt_ret, RT_ERROR_NONE);
  sendbuf = (s8 *)sal_malloc(count * sizeof(s8));
  sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));
  recvbuf = (s8 *)sal_malloc(count * sizeof(s8));
  sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));

  for (int j = 0; j < count; j++) {
    sendbuf[j] = 2;
  }

  ret = HcomAllReduce("testallreduce", sendbuf, recvbuf, count, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, NULL, stream);
  EXPECT_EQ(ret, HCCL_SUCCESS);

  rt_ret = rtStreamSynchronize(stream);
  EXPECT_EQ(rt_ret, RT_ERROR_NONE);

  for (int j = 0; j < count; j++) {
    if (recvbuf[j] != 2) {
      errors++;
      break;
    }
  }

  for (int j = 0; j < count; j++) {
    sendbuf[j] = 2;
  }

  (void)HcomSetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
  ret = HcclAllReduce(sendbuf, recvbuf, count, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, hcclComm, stream);
  EXPECT_EQ(ret, HCCL_SUCCESS);
  rt_ret = rtStreamSynchronize(stream);
  EXPECT_EQ(rt_ret, RT_ERROR_NONE);

  for (int j = 0; j < count; j++) {
    if (recvbuf[j] != 2) {
      errors++;
      break;
    }
  }

  ret = HcomExecFinalize();
  EXPECT_EQ(ret, HCCL_SUCCESS);
  sal_free(sendbuf);
  sal_free(recvbuf);
  rt_ret = rtStreamDestroy(stream);

  ret = HcomDestroy();
  EXPECT_EQ(ret, HCCL_SUCCESS);
  remove(file_name_t);
  EXPECT_EQ(rt_ret, RT_ERROR_NONE);
}
#endif
#if 1
TEST_F(HcomExecutorTest, ut_executor_equeue_allreduce_mix) {
  nlohmann::json rank_table = rank_table_910_1server_1rank;

  char file_name_t[] = "./ut_executor_equeue_allreduce_mix.json";
  std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

  if (outfile.is_open()) {
    outfile << std::setw(1) << rank_table << std::endl;
    HCCL_INFO("open %s success", file_name_t);
  } else {
    HCCL_ERROR("open %s failed", file_name_t);
  }

  outfile.close();

  int ret = HCCL_SUCCESS;
  rtError_t rt_ret = RT_ERROR_NONE;
  rtStream_t stream;
  s8 *sendbuf;
  s8 *recvbuf;
  s32 rank = 0;
  s32 errors = 0;
  s32 count = HCCL_COM_DATA_SIZE;
  ret = hrtSetDevice(0);
  EXPECT_EQ(ret, HCCL_SUCCESS);

  ret = HcomExecInitialize();
  EXPECT_EQ(ret, HCCL_SUCCESS);

  HcclComm hcclComm;
  ret = HcomGetCommHandleByGroup(NULL, &hcclComm);
  EXPECT_EQ(ret, HCCL_SUCCESS);
  ret = hccl::HcomExecutor::GetInstance().InitHcclComm(nullptr, hcclComm);
  EXPECT_EQ(ret, HCCL_SUCCESS);

  rt_ret = rtStreamCreate(&stream, 5);
  EXPECT_EQ(rt_ret, RT_ERROR_NONE);
  sendbuf = (s8 *)sal_malloc(count * sizeof(s8));
  sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));
  recvbuf = (s8 *)sal_malloc(count * sizeof(s8));
  sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));

  for (int j = 0; j < count; j++) {
    sendbuf[j] = 2;
  }

  ret = HcomAllReduce("testallreduce", sendbuf, recvbuf, count, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, NULL, stream);
  EXPECT_EQ(ret, HCCL_SUCCESS);

  rt_ret = rtStreamSynchronize(stream);
  EXPECT_EQ(rt_ret, RT_ERROR_NONE);

  for (int j = 0; j < count; j++) {
    if (recvbuf[j] != 2) {
      errors++;
      break;
    }
  }

  for (int j = 0; j < count; j++) {
    sendbuf[j] = 2;
  }
  setExecutorStatus(HCCL_E_RESERVED);

  HCCL_INFO("executor start");
  HcomOperation opInfo;
  opInfo.hcclType = HCCL_TYPE_ALLREDUCE;
  opInfo.inputPtr = sendbuf;
  opInfo.outputPtr = recvbuf;
  opInfo.dataType = HCCL_DATA_TYPE_INT8;
  opInfo.count = count;
  opInfo.opType = HCCL_REDUCE_SUM;
  ret = HcomExecEnqueueOperation(opInfo, setExecutorStatus);
  if (ret != HCCL_SUCCESS) {
    HcomExecFinalize();
  }
  EXPECT_EQ(ret, HCCL_SUCCESS);

  HcclResult excutorStatus = HCCL_E_RESERVED;
  const std::chrono::seconds TIMEOUT(1);
  const auto start = std::chrono::steady_clock::now();
  while (excutorStatus != HCCL_SUCCESS) {
    getExecutorStatus(excutorStatus);
    std::this_thread::sleep_for(std::chrono::milliseconds(LLT_SOCKET_SLEEP_MILLISECONDS));
    /* ?????????????timeout?????? */
    const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start);
    if (elapsed > TIMEOUT) {
      HcomExecFinalize();
      HCCL_ERROR("Wait timeout for getExecutor status timeout[%lld]", TIMEOUT);
      return;
    }
  }
  EXPECT_EQ(excutorStatus, HCCL_SUCCESS);
  for (int j = 0; j < count; j++) {
    if (recvbuf[j] != 2) {
      errors++;
      break;
    }
  }

  ret = HcomExecFinalize();
  EXPECT_EQ(ret, HCCL_SUCCESS);
  sal_free(sendbuf);
  sal_free(recvbuf);
  rt_ret = rtStreamDestroy(stream);

  ret = HcomDestroy();
  EXPECT_EQ(ret, HCCL_SUCCESS);
  setExecutorStatus(HCCL_E_RESERVED);
  remove(file_name_t);
  EXPECT_EQ(rt_ret, RT_ERROR_NONE);
  EXPECT_EQ(errors, 0);
}
#endif

#if 1
TEST_F(HcomExecutorTest, ut_executor_equeue_allreduce) {
  nlohmann::json rank_table = rank_table_910_1server_1rank;

  char file_name_t[] = "./ut_executor_equeue_allreduce.json";
  std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

  if (outfile.is_open()) {
    outfile << std::setw(1) << rank_table << std::endl;
    HCCL_INFO("open %s success", file_name_t);
  } else {
    HCCL_ERROR("open %s failed", file_name_t);
  }

  outfile.close();

  int ret = HCCL_SUCCESS;
  rtError_t rt_ret = RT_ERROR_NONE;
  rtStream_t stream;
  s8 *sendbuf;
  s8 *recvbuf;
  s32 rank = 0;
  s32 errors = 0;
  s32 count = HCCL_COM_DATA_SIZE;
  ret = hrtSetDevice(0);
  EXPECT_EQ(ret, HCCL_SUCCESS);

  ret = HcomExecInitialize();
  EXPECT_EQ(ret, HCCL_SUCCESS);

  HcclComm hcclComm;
  ret = HcomGetCommHandleByGroup(NULL, &hcclComm);
  EXPECT_EQ(ret, HCCL_SUCCESS);
  ret = hccl::HcomExecutor::GetInstance().InitHcclComm(nullptr, hcclComm);
  EXPECT_EQ(ret, HCCL_SUCCESS);

  rt_ret = rtStreamCreate(&stream, 5);
  EXPECT_EQ(rt_ret, RT_ERROR_NONE);
  sendbuf = (s8 *)sal_malloc(count * sizeof(s8));
  sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));
  recvbuf = (s8 *)sal_malloc(count * sizeof(s8));
  sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));

  for (int j = 0; j < count; j++) {
    sendbuf[j] = 2;
  }
  setExecutorStatus(HCCL_E_RESERVED);

  HCCL_INFO("executor start");
  HcomOperation opInfo;
  opInfo.hcclType = HCCL_TYPE_ALLREDUCE;
  opInfo.inputPtr = sendbuf;
  opInfo.outputPtr = recvbuf;
  opInfo.dataType = HCCL_DATA_TYPE_INT8;
  opInfo.count = count;
  opInfo.opType = HCCL_REDUCE_SUM;
  ret = HcomExecEnqueueOperation(opInfo, setExecutorStatus);
  if (ret != HCCL_SUCCESS) {
    HCCL_ERROR("HcomExecEnqueueOperation error");
    HcomExecFinalize();
  }
  EXPECT_EQ(ret, HCCL_SUCCESS);
  HcclResult excutorStatus = HCCL_E_RESERVED;
  const std::chrono::seconds TIMEOUT(1);
  const auto start = std::chrono::steady_clock::now();
  while (excutorStatus != HCCL_SUCCESS) {
    getExecutorStatus(excutorStatus);
    // HCCL_INFO("getExecutorStatus start excutorStatus[%d]",excutorStatus);
    std::this_thread::sleep_for(std::chrono::milliseconds(LLT_SOCKET_SLEEP_MILLISECONDS));
    /* ?????????????timeout?????? */
    const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start);
    if (elapsed > TIMEOUT) {
      HCCL_ERROR("Wait timeout for getExecutor status timeout[%lld]", TIMEOUT);
      HcomExecFinalize();
      return;
    }
  }
  std::this_thread::sleep_for(std::chrono::seconds(2));
  HCCL_INFO("HcomExcutor excutorStatus[%d]", excutorStatus);
  EXPECT_EQ(excutorStatus, HCCL_SUCCESS);
  for (int j = 0; j < count; j++) {
    if (recvbuf[j] != 2) {
      errors++;
      break;
    }
  }

  HCCL_INFO("HcomExecFinalize start");
  ret = HcomExecFinalize();
  EXPECT_EQ(ret, HCCL_SUCCESS);
  sal_free(sendbuf);
  sal_free(recvbuf);
  rt_ret = rtStreamDestroy(stream);

  ret = HcomDestroy();
  EXPECT_EQ(ret, HCCL_SUCCESS);
  setExecutorStatus(HCCL_E_RESERVED);
  remove(file_name_t);
  EXPECT_EQ(rt_ret, RT_ERROR_NONE);
  EXPECT_EQ(errors, 0);
}
#endif

#if 1
TEST_F(HcomExecutorTest, ut_executor_equeue_reduce_scatter) {
  nlohmann::json rank_table = rank_table_910_1server_1rank;

  char file_name_t[] = "./ut_executor_equeue_reduce_scatter.json";
  std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

  if (outfile.is_open()) {
    outfile << std::setw(1) << rank_table << std::endl;
    HCCL_INFO("open %s success", file_name_t);
  } else {
    HCCL_ERROR("open %s failed", file_name_t);
  }

  outfile.close();

  int ret = HCCL_SUCCESS;
  rtError_t rt_ret = RT_ERROR_NONE;
  rtStream_t stream;
  s8 *sendbuf;
  s8 *recvbuf;
  s32 rank = 0;
  s32 errors = 0;
  s32 count = HCCL_COM_DATA_SIZE;
  ret = hrtSetDevice(0);
  EXPECT_EQ(ret, HCCL_SUCCESS);

  ret = HcomExecInitialize();
  EXPECT_EQ(ret, HCCL_SUCCESS);

  HcclComm hcclComm;
  ret = HcomGetCommHandleByGroup(NULL, &hcclComm);
  EXPECT_EQ(ret, HCCL_SUCCESS);
  ret = hccl::HcomExecutor::GetInstance().InitHcclComm(nullptr, hcclComm);
  EXPECT_EQ(ret, HCCL_SUCCESS);

  rt_ret = rtStreamCreate(&stream, 5);
  EXPECT_EQ(rt_ret, RT_ERROR_NONE);
  sendbuf = (s8 *)sal_malloc(count * sizeof(s8));
  sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));
  recvbuf = (s8 *)sal_malloc(count * sizeof(s8));
  sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));

  for (int j = 0; j < count; j++) {
    sendbuf[j] = 2;
  }
  setExecutorStatus(HCCL_E_RESERVED);

  HCCL_INFO("executor start");
  HcomOperation opInfo;
  opInfo.hcclType = HCCL_TYPE_REDUCESCATTER;
  opInfo.inputPtr = sendbuf;
  opInfo.outputPtr = recvbuf;
  opInfo.dataType = HCCL_DATA_TYPE_INT8;
  opInfo.count = count;
  opInfo.opType = HCCL_REDUCE_SUM;
  ret = HcomExecEnqueueOperation(opInfo, setExecutorStatus);
  if (ret != HCCL_SUCCESS) {
    HCCL_ERROR("HcomExecEnqueueOperation error");
    HcomExecFinalize();
  }
  EXPECT_EQ(ret, HCCL_SUCCESS);
  HcclResult excutorStatus = HCCL_E_RESERVED;
  const std::chrono::seconds TIMEOUT(1);
  const auto start = std::chrono::steady_clock::now();
  while (excutorStatus != HCCL_SUCCESS) {
    getExecutorStatus(excutorStatus);
    std::this_thread::sleep_for(std::chrono::milliseconds(LLT_SOCKET_SLEEP_MILLISECONDS));
    const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start);
    if (elapsed > TIMEOUT) {
      HCCL_ERROR("Wait timeout for getExecutor status timeout[%lld]", TIMEOUT);
      HcomExecFinalize();
      return;
    }
  }
  std::this_thread::sleep_for(std::chrono::seconds(2));
  HCCL_INFO("HcomExcutor excutorStatus[%d]", excutorStatus);
  EXPECT_EQ(excutorStatus, HCCL_SUCCESS);
  for (int j = 0; j < count; j++) {
    if (recvbuf[j] != 2) {
      errors++;
      break;
    }
  }

  HCCL_INFO("HcomExecFinalize start");
  ret = HcomExecFinalize();
  EXPECT_EQ(ret, HCCL_SUCCESS);
  sal_free(sendbuf);
  sal_free(recvbuf);
  rt_ret = rtStreamDestroy(stream);

  ret = HcomDestroy();
  EXPECT_EQ(ret, HCCL_SUCCESS);
  setExecutorStatus(HCCL_E_RESERVED);
  remove(file_name_t);
  EXPECT_EQ(rt_ret, RT_ERROR_NONE);
  EXPECT_EQ(errors, 0);
}
#endif

#if 1
TEST_F(HcomExecutorTest, ut_executor_equeue_broardcast) {
  nlohmann::json rank_table = rank_table_910_1server_1rank;

  char file_name_t[] = "./ut_executor_equeue_broardcast.json";
  std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

  if (outfile.is_open()) {
    outfile << std::setw(1) << rank_table << std::endl;
    HCCL_INFO("open %s success", file_name_t);
  } else {
    HCCL_ERROR("open %s failed", file_name_t);
  }

  outfile.close();

  int ret = HCCL_SUCCESS;
  rtError_t rt_ret = RT_ERROR_NONE;
  s8 *sendbuf;
  s32 rank = 0;
  s32 errors = 0;
  s32 count = HCCL_COM_DATA_SIZE;
  ret = hrtSetDevice(0);
  EXPECT_EQ(ret, HCCL_SUCCESS);

  ret = HcomExecInitialize();
  EXPECT_EQ(ret, HCCL_SUCCESS);
  EXPECT_EQ(rt_ret, RT_ERROR_NONE);

  HcclComm hcclComm;
  ret = HcomGetCommHandleByGroup(NULL, &hcclComm);
  EXPECT_EQ(ret, HCCL_SUCCESS);
  ret = hccl::HcomExecutor::GetInstance().InitHcclComm(nullptr, hcclComm);
  EXPECT_EQ(ret, HCCL_SUCCESS);

  sendbuf = (s8 *)sal_malloc(count * sizeof(s8) + 512);
  sal_memset(sendbuf, count * sizeof(s8) + 512, 0, count * sizeof(s8) + 512);

  for (int j = 0; j < count; j++) {
    sendbuf[j + 128] = 2;
  }

  HCCL_INFO("executor start");
  HcomOperation opInfo;
  opInfo.hcclType = HCCL_TYPE_BROADCAST;
  opInfo.inputPtr = sendbuf + 128;
  opInfo.dataType = HCCL_DATA_TYPE_INT8;
  opInfo.count = count;
  opInfo.opType = HCCL_REDUCE_SUM;
  opInfo.root = 0;
  for (int i = 0; i < 5; i++) {
    ret = HcomExecEnqueueOperation(opInfo, setMutiExecutorStatus);
    if (ret != HCCL_SUCCESS) {
      HCCL_ERROR("HcomExecEnqueueOperation error");
      HcomExecFinalize();
    }
    EXPECT_EQ(ret, HCCL_SUCCESS);
  }

  HcclResult excutorStatus = HCCL_E_RESERVED;
  const std::chrono::seconds TIMEOUT(1);
  const auto start = std::chrono::steady_clock::now();
  while (excutorStatus != HCCL_SUCCESS) {
    getMutiExecutorStatus(excutorStatus, 5);
    std::this_thread::sleep_for(std::chrono::milliseconds(LLT_SOCKET_SLEEP_MILLISECONDS));
    const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start);
    if (elapsed > TIMEOUT) {
      HCCL_ERROR("Wait timeout for getExecutor status timeout[%lld]", TIMEOUT);
      HcomExecFinalize();
      return;
    }
  }
  HCCL_INFO("HcomExcutor excutorStatus[%d]", excutorStatus);
  EXPECT_EQ(excutorStatus, HCCL_SUCCESS);
  for (int j = 0; j < count; j++) {
    if (sendbuf[j + 128] != 2) {
      HCCL_ERROR("\n rank:%d sendbuf[%d]:%f", rank, j + 128, sendbuf[j + 128]);
      errors++;
      break;
    }
  }

  HCCL_INFO("HcomExecFinalize start");
  ret = HcomExecFinalize();
  EXPECT_EQ(ret, HCCL_SUCCESS);
  sal_free(sendbuf);

  ret = HcomDestroy();
  EXPECT_EQ(ret, HCCL_SUCCESS);
  setExecutorStatus(HCCL_E_RESERVED);
  remove(file_name_t);
  EXPECT_EQ(rt_ret, RT_ERROR_NONE);
  EXPECT_EQ(errors, 0);
}
#endif

#if 1
TEST_F(HcomExecutorTest, ut_executor_equeue_allgather) {
  nlohmann::json rank_table = rank_table_910_1server_1rank;

  char file_name_t[] = "./ut_executor_equeue_allgather.json";
  std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

  if (outfile.is_open()) {
    outfile << std::setw(1) << rank_table << std::endl;
    HCCL_INFO("open %s success", file_name_t);
  } else {
    HCCL_ERROR("open %s failed", file_name_t);
  }

  outfile.close();

  int ret = HCCL_SUCCESS;
  rtError_t rt_ret = RT_ERROR_NONE;
  s8 *sendbuf;
  s8 *recvbuf;
  s32 rank = 0;
  s32 errors = 0;
  s32 count = HCCL_COM_DATA_SIZE;
  ret = hrtSetDevice(0);
  EXPECT_EQ(ret, HCCL_SUCCESS);

  ret = HcomExecInitialize();
  EXPECT_EQ(ret, HCCL_SUCCESS);
  EXPECT_EQ(rt_ret, RT_ERROR_NONE);

  HcclComm hcclComm;
  ret = HcomGetCommHandleByGroup(NULL, &hcclComm);
  EXPECT_EQ(ret, HCCL_SUCCESS);
  ret = hccl::HcomExecutor::GetInstance().InitHcclComm(nullptr, hcclComm);
  EXPECT_EQ(ret, HCCL_SUCCESS);

  sendbuf = (s8 *)sal_malloc(count * sizeof(s8) + 512);
  sal_memset(sendbuf, count * sizeof(s8) + 512, 0, count * sizeof(s8) + 512);
  recvbuf = (s8 *)sal_malloc(count * sizeof(s8) + 512);
  sal_memset(recvbuf, count * sizeof(s8) + 512, 0, count * sizeof(s8) + 512);

  for (int j = 0; j < count; j++) {
    sendbuf[j + 128] = 2;
  }
  setExecutorStatus(HCCL_E_RESERVED);

  HCCL_INFO("executor start");
  HcomOperation opInfo;
  opInfo.hcclType = HCCL_TYPE_ALLGATHER;
  opInfo.inputPtr = sendbuf + 128;
  opInfo.outputPtr = recvbuf + 128;
  opInfo.dataType = HCCL_DATA_TYPE_INT8;
  opInfo.count = count;
  opInfo.opType = HCCL_REDUCE_SUM;
  ret = HcomExecEnqueueOperation(opInfo, setExecutorStatus);
  if (ret != HCCL_SUCCESS) {
    HCCL_ERROR("HcomExecEnqueueOperation error");
    HcomExecFinalize();
  }
  EXPECT_EQ(ret, HCCL_SUCCESS);

  HcclResult excutorStatus = HCCL_E_RESERVED;
  const std::chrono::seconds TIMEOUT(1);
  const auto start = std::chrono::steady_clock::now();
  while (excutorStatus != HCCL_SUCCESS) {
    getExecutorStatus(excutorStatus);
    std::this_thread::sleep_for(std::chrono::milliseconds(LLT_SOCKET_SLEEP_MILLISECONDS));
    const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start);
    if (elapsed > TIMEOUT) {
      HCCL_ERROR("Wait timeout for getExecutor status timeout[%lld]", TIMEOUT);
      HcomExecFinalize();
      return;
    }
  }
  HCCL_INFO("HcomExcutor excutorStatus[%d]", excutorStatus);
  EXPECT_EQ(excutorStatus, HCCL_SUCCESS);
  for (int j = 0; j < count; j++) {
    if (recvbuf[j + 128] != 2) {
      errors++;
      break;
    }
  }

  HCCL_INFO("HcomExecFinalize start");
  ret = HcomExecFinalize();
  EXPECT_EQ(ret, HCCL_SUCCESS);
  sal_free(sendbuf);
  sal_free(recvbuf);

  ret = HcomDestroy();
  EXPECT_EQ(ret, HCCL_SUCCESS);
  setExecutorStatus(HCCL_E_RESERVED);
  remove(file_name_t);
  EXPECT_EQ(rt_ret, RT_ERROR_NONE);
  EXPECT_EQ(errors, 0);
}
#endif

#if 1
TEST_F(HcomExecutorTest, ut_executor_equeue_allgather_mutiInit) {
  nlohmann::json rank_table = rank_table_910_1server_1rank;

  char file_name_t[] = "./ut_executor_equeue_allgather_mutiInit.json";
  std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

  if (outfile.is_open()) {
    outfile << std::setw(1) << rank_table << std::endl;
    HCCL_INFO("open %s success", file_name_t);
  } else {
    HCCL_ERROR("open %s failed", file_name_t);
  }

  outfile.close();

  int ret = HCCL_SUCCESS;
  rtError_t rt_ret = RT_ERROR_NONE;
  s8 *sendbuf;
  s8 *recvbuf;
  s32 rank = 0;
  s32 errors = 0;
  s32 count = HCCL_COM_DATA_SIZE;
  ret = hrtSetDevice(0);
  EXPECT_EQ(ret, HCCL_SUCCESS);

  ret = HcomExecInitialize();
  EXPECT_EQ(ret, HCCL_SUCCESS);
  EXPECT_EQ(rt_ret, RT_ERROR_NONE);

  HcclComm hcclComm;
  ret = HcomGetCommHandleByGroup(NULL, &hcclComm);
  EXPECT_EQ(ret, HCCL_SUCCESS);
  ret = hccl::HcomExecutor::GetInstance().InitHcclComm(nullptr, hcclComm);
  EXPECT_EQ(ret, HCCL_SUCCESS);

  sendbuf = (s8 *)sal_malloc(count * sizeof(s8) + 512);
  sal_memset(sendbuf, count * sizeof(s8) + 512, 0, count * sizeof(s8) + 512);
  recvbuf = (s8 *)sal_malloc(count * sizeof(s8) + 512);
  sal_memset(recvbuf, count * sizeof(s8) + 512, 0, count * sizeof(s8) + 512);

  for (int j = 0; j < count; j++) {
    sendbuf[j + 128] = 2;
  }
  setExecutorStatus(HCCL_E_RESERVED);

  HCCL_INFO("executor start");
  HcomOperation opInfo;
  opInfo.hcclType = HCCL_TYPE_ALLGATHER;
  opInfo.inputPtr = sendbuf + 128;
  opInfo.outputPtr = recvbuf + 128;
  opInfo.dataType = HCCL_DATA_TYPE_INT8;
  opInfo.count = count;
  opInfo.opType = HCCL_REDUCE_SUM;
  ret = HcomExecEnqueueOperation(opInfo, setExecutorStatus);
  if (ret != HCCL_SUCCESS) {
    HCCL_ERROR("HcomExecEnqueueOperation error");
    HcomExecFinalize();
  }
  EXPECT_EQ(ret, HCCL_SUCCESS);

  HcclResult excutorStatus = HCCL_E_RESERVED;
  const std::chrono::seconds TIMEOUT(1);
  const auto start = std::chrono::steady_clock::now();
  while (excutorStatus != HCCL_SUCCESS) {
    getExecutorStatus(excutorStatus);
    std::this_thread::sleep_for(std::chrono::milliseconds(LLT_SOCKET_SLEEP_MILLISECONDS));
    const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start);
    if (elapsed > TIMEOUT) {
      HCCL_ERROR("Wait timeout for getExecutor status timeout[%lld]", TIMEOUT);
      HcomExecFinalize();
      return;
    }
  }
  HCCL_INFO("HcomExcutor excutorStatus[%d]", excutorStatus);
  EXPECT_EQ(excutorStatus, HCCL_SUCCESS);
  for (int j = 0; j < count; j++) {
    if (recvbuf[j + 128] != 2) {
      errors++;
      break;
    }
  }

  HCCL_INFO("HcomExecFinalize start");
  ret = HcomExecFinalize();
  EXPECT_EQ(ret, HCCL_SUCCESS);

  ret = HcomExecInitialize();
  EXPECT_EQ(ret, HCCL_SUCCESS);

  ret = HcomExecFinalize();
  EXPECT_EQ(ret, HCCL_SUCCESS);

  sal_free(sendbuf);
  sal_free(recvbuf);

  ret = HcomDestroy();
  EXPECT_EQ(ret, HCCL_SUCCESS);
  setExecutorStatus(HCCL_E_RESERVED);
  remove(file_name_t);
  EXPECT_EQ(rt_ret, RT_ERROR_NONE);
  EXPECT_EQ(errors, 0);
}
#endif

#if 1
TEST_F(HcomExecutorTest, ut_executor_equeue_mutiInit) {
  nlohmann::json rank_table = rank_table_910_1server_1rank;

  char file_name_t[] = "./ut_executor_equeue_mutiInit.json";
  std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

  if (outfile.is_open()) {
    outfile << std::setw(1) << rank_table << std::endl;
    HCCL_INFO("open %s success", file_name_t);
  } else {
    HCCL_ERROR("open %s failed", file_name_t);
  }

  outfile.close();

  int ret = HCCL_SUCCESS;
  rtError_t rt_ret = RT_ERROR_NONE;
  s32 rank = 0;
  s32 errors = 0;
  s32 count = HCCL_COM_DATA_SIZE;
  ret = hrtSetDevice(0);
  EXPECT_EQ(ret, HCCL_SUCCESS);

  ret = HcomExecInitialize();
  EXPECT_EQ(ret, HCCL_SUCCESS);
  HCCL_INFO("HcomExecFinalize start");
  ret = HcomExecFinalize();
  EXPECT_EQ(ret, HCCL_SUCCESS);

  ret = HcomExecInitialize();
  EXPECT_EQ(ret, HCCL_SUCCESS);

  ret = HcomExecFinalize();
  EXPECT_EQ(ret, HCCL_SUCCESS);

  ret = HcomDestroy();
  EXPECT_EQ(ret, HCCL_SUCCESS);
  setExecutorStatus(HCCL_E_RESERVED);
  remove(file_name_t);
  EXPECT_EQ(rt_ret, RT_ERROR_NONE);
  EXPECT_EQ(errors, 0);
}
#endif

TEST_F(HcomExecutorTest, ut_executor_equeue_alltoallv) {
  nlohmann::json rank_table = rank_table_910_1server_1rank;
  char file_name_t[] = "./ut_executor_equeue_alltoallv.json";
  std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

  if (outfile.is_open()) {
    outfile << std::setw(1) << rank_table << std::endl;
    HCCL_INFO("open %s success", file_name_t);
  } else {
    HCCL_ERROR("open %s failed", file_name_t);
  }

  outfile.close();
  char *identify = "0";
  s32 rankSize = 1;
  s32 rank = atoi(identify);
  u64 count = 2;
  HCCL_INFO("alltoall_int32 : identify[%d], count[%llu]", rank, count);
  u32 devLogicId = 0xFFFF;
  HcclResult ret = hrtSetDevice(devLogicId);
  EXPECT_EQ(ret, HCCL_SUCCESS);

  ret = HcomExecInitialize();
  EXPECT_EQ(ret, HCCL_SUCCESS);

  HcclComm hcclComm;
  ret = HcomGetCommHandleByGroup(NULL, &hcclComm);
  EXPECT_EQ(ret, HCCL_SUCCESS);
  ret = hccl::HcomExecutor::GetInstance().InitHcclComm(nullptr, hcclComm);
  EXPECT_EQ(ret, HCCL_SUCCESS);

  const int COUNT_PER_RANK = count;
  u64 memSize = COUNT_PER_RANK * rankSize * sizeof(s32);
  HostMem hostSendMem = HostMem::alloc(memSize);
  memset_s(hostSendMem.ptr(), memSize, 0, COUNT_PER_RANK * rankSize);
  for (u32 i = 0; i < COUNT_PER_RANK * rankSize; i++) {
    *((s32 *)hostSendMem.ptr() + i) = rank + 1;
  }

  // 构造入参
  vector<u64> sendCounts(rankSize, COUNT_PER_RANK);
  vector<u64> recvCounts(rankSize, COUNT_PER_RANK);
  vector<u64> sdispls(rankSize, 0);
  vector<u64> rdispls(rankSize, 0);
  for (int i = 0; i < rankSize; i++) {
    sdispls[i] = COUNT_PER_RANK * i;
    rdispls[i] = COUNT_PER_RANK * i;
    HCCL_INFO("num[%d] displs[%d]", i, COUNT_PER_RANK * i);
  }
  DeviceMem devSendCounts = DeviceMem::alloc(rankSize * sizeof(u64));
  ret = hrtMemSyncCopy(devSendCounts.ptr(), rankSize * sizeof(u64), sendCounts.data(), rankSize * sizeof(u64),
                       HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE);
  EXPECT_EQ(ret, HCCL_SUCCESS);
  DeviceMem devSdispls = DeviceMem::alloc(rankSize * sizeof(u64));
  ret = hrtMemSyncCopy(devSdispls.ptr(), rankSize * sizeof(u64), sdispls.data(), rankSize * sizeof(u64),
                       HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE);
  EXPECT_EQ(ret, HCCL_SUCCESS);
  DeviceMem devRecvCounts = DeviceMem::alloc(rankSize * sizeof(u64));
  ret = hrtMemSyncCopy(devRecvCounts.ptr(), rankSize * sizeof(u64), recvCounts.data(), rankSize * sizeof(u64),
                       HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE);
  EXPECT_EQ(ret, HCCL_SUCCESS);
  DeviceMem devRdispls = DeviceMem::alloc(rankSize * sizeof(u64));
  ret = hrtMemSyncCopy(devRdispls.ptr(), rankSize * sizeof(u64), rdispls.data(), rankSize * sizeof(u64),
                       HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE);
  EXPECT_EQ(ret, HCCL_SUCCESS);

  DeviceMem sendMem = DeviceMem::alloc(memSize);
  ret = hrtMemSyncCopy(sendMem.ptr(), memSize, hostSendMem.ptr(), memSize,
                       HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE);
  EXPECT_EQ(ret, HCCL_SUCCESS);
  DeviceMem recvMem = DeviceMem::alloc(memSize);

  HcomAllToAllVParams opInfo;
  opInfo.group = nullptr;
  opInfo.sendbuf = sendMem.ptr();
  opInfo.sendcounts = devSendCounts.ptr();
  opInfo.sdispls = devSdispls.ptr();
  opInfo.sendtype = HCCL_DATA_TYPE_INT32;
  opInfo.recvbuf = recvMem.ptr();
  opInfo.recvcounts = devRecvCounts.ptr();
  opInfo.rdispls = devRdispls.ptr();
  opInfo.recvtype = HCCL_DATA_TYPE_INT32;

  setExecutorStatus(HCCL_E_RESERVED);

  ret = HcomExecEnqueueAllToAllV(opInfo, setExecutorStatus);
  EXPECT_EQ(ret, HCCL_SUCCESS);

  HCCL_INFO("HcomExecFinalize start");
  ret = HcomExecFinalize();
  EXPECT_EQ(ret, HCCL_SUCCESS);

  ret = HcomDestroy();
  EXPECT_EQ(ret, HCCL_SUCCESS);
  setExecutorStatus(HCCL_E_RESERVED);
  remove(file_name_t);
  (void)aclrtResetDevice(0);
}

TEST_F(HcomExecutorTest, ut_executor_equeue_alltoallvc) {
  nlohmann::json rank_table = rank_table_910_1server_1rank;
  char file_name_t[] = "./ut_executor_equeue_alltoallvc.json";
  std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

  if (outfile.is_open()) {
    outfile << std::setw(1) << rank_table << std::endl;
    HCCL_INFO("open %s success", file_name_t);
  } else {
    HCCL_ERROR("open %s failed", file_name_t);
  }

  outfile.close();
  char *identify = "0";
  s32 rankSize = 1;
  s32 rank = atoi(identify);
  u64 count = 2;
  HCCL_INFO("alltoallvc_int32 : identify[%d], count[%llu]", rank, count);
  u32 devLogicId = 0xFFFF;
  HcclResult ret = hrtSetDevice(devLogicId);
  EXPECT_EQ(ret, HCCL_SUCCESS);

  ret = HcomExecInitialize();
  EXPECT_EQ(ret, HCCL_SUCCESS);

  HcclComm hcclComm;
  ret = HcomGetCommHandleByGroup(NULL, &hcclComm);
  EXPECT_EQ(ret, HCCL_SUCCESS);
  ret = hccl::HcomExecutor::GetInstance().InitHcclComm(nullptr, hcclComm);
  EXPECT_EQ(ret, HCCL_SUCCESS);

  const int COUNT_PER_RANK = count;
  u64 memSize = COUNT_PER_RANK * rankSize * sizeof(s32);
  HostMem hostSendMem = HostMem::alloc(memSize);
  memset_s(hostSendMem.ptr(), memSize, 0, COUNT_PER_RANK * rankSize);
  for (u32 i = 0; i < COUNT_PER_RANK * rankSize; i++) {
    *((s32 *)hostSendMem.ptr() + i) = rank + 1;
  }

  // 构造入参
  vector<u64> sendCounts(rankSize, COUNT_PER_RANK);
  vector<u64> recvCounts(rankSize, COUNT_PER_RANK);
  vector<u64> sendCountMatrix(rankSize * rankSize, COUNT_PER_RANK);

  DeviceMem devSendCountMatrix = DeviceMem::alloc(rankSize * rankSize * sizeof(u64));
  ret = hrtMemSyncCopy(devSendCountMatrix.ptr(), rankSize * rankSize * sizeof(u64), sendCountMatrix.data(),
                       rankSize * rankSize * sizeof(u64), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE);
  EXPECT_EQ(ret, HCCL_SUCCESS);

  DeviceMem sendMem = DeviceMem::alloc(memSize);
  ret = hrtMemSyncCopy(sendMem.ptr(), memSize, hostSendMem.ptr(), memSize,
                       HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE);
  EXPECT_EQ(ret, HCCL_SUCCESS);
  DeviceMem recvMem = DeviceMem::alloc(memSize);

  HcomAllToAllVCParams opInfo;
  opInfo.group = nullptr;
  opInfo.sendbuf = sendMem.ptr();
  opInfo.sendcountmatrix = devSendCountMatrix.ptr();
  opInfo.sendtype = HCCL_DATA_TYPE_INT32;
  opInfo.recvbuf = recvMem.ptr();
  opInfo.recvtype = HCCL_DATA_TYPE_INT32;

  setExecutorStatus(HCCL_E_RESERVED);

  ret = HcomExecEnqueueAllToAllVC(opInfo, setExecutorStatus);
  EXPECT_EQ(ret, HCCL_SUCCESS);

  HCCL_INFO("HcomExecFinalize start");
  ret = HcomExecFinalize();
  EXPECT_EQ(ret, HCCL_SUCCESS);

  ret = HcomDestroy();
  EXPECT_EQ(ret, HCCL_SUCCESS);
  setExecutorStatus(HCCL_E_RESERVED);
  remove(file_name_t);
  (void)aclrtResetDevice(0);
}

TEST_F(HcomExecutorTest, ut_executor_equeue_gather_alltoallv) {
  nlohmann::json rank_table = rank_table_910_1server_1rank;
  char file_name_t[] = "./ut_executor_equeue_gather_alltoallv.json";
  std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

  if (outfile.is_open()) {
    outfile << std::setw(1) << rank_table << std::endl;
    HCCL_INFO("open %s success", file_name_t);
  } else {
    HCCL_ERROR("open %s failed", file_name_t);
  }

  outfile.close();
  char *identify = "0";
  s32 rankSize = 1;
  s32 rank = atoi(identify);
  u64 count = 200;
  HCCL_INFO("alltoall_int32 : identify[%d], count[%llu]", rank, count);
  u32 devLogicId = 0xFFFF;
  HcclResult ret = hrtSetDevice(devLogicId);
  EXPECT_EQ(ret, HCCL_SUCCESS);

  ret = HcomExecInitialize();
  EXPECT_EQ(ret, HCCL_SUCCESS);

  HcclComm hcclComm;
  ret = HcomGetCommHandleByGroup(NULL, &hcclComm);
  EXPECT_EQ(ret, HCCL_SUCCESS);
  ret = hccl::HcomExecutor::GetInstance().InitHcclComm(nullptr, hcclComm);
  EXPECT_EQ(ret, HCCL_SUCCESS);

  u32 countsNum = 200;
  vector<u64> addrInfo;
  vector<u64> addrInfoCountPerRank(rankSize, countsNum);

  const int COUNT_PER_RANK = count;
  u64 memSize = COUNT_PER_RANK * rankSize * sizeof(s32) * countsNum;
  HostMem hostSendMem = HostMem::alloc(memSize);
  memset_s(hostSendMem.ptr(), memSize, 0, COUNT_PER_RANK * rankSize);
  for (u32 i = 0; i < COUNT_PER_RANK * rankSize * countsNum; i++) {
    *((s32 *)hostSendMem.ptr() + i) = rank + 1;
    if (i % COUNT_PER_RANK == 0) {
      addrInfo.push_back((uintptr_t)((s32 *)hostSendMem.ptr() + i));
      addrInfo.push_back(COUNT_PER_RANK * sizeof(s32));
    }
  }

  // 构造入参
  DeviceMem devAddrInfo = DeviceMem::alloc(addrInfo.size() * sizeof(u64));
  ret = hrtMemSyncCopy(devAddrInfo.ptr(), addrInfo.size() * sizeof(u64), addrInfo.data(), addrInfo.size() * sizeof(u64),
                       HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE);
  EXPECT_EQ(ret, HCCL_SUCCESS);
  DeviceMem devCountPerRank = DeviceMem::alloc(addrInfoCountPerRank.size() * sizeof(u64));
  ret = hrtMemSyncCopy(devCountPerRank.ptr(), addrInfoCountPerRank.size() * sizeof(u64), addrInfoCountPerRank.data(),
                       addrInfoCountPerRank.size() * sizeof(u64), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE);
  EXPECT_EQ(ret, HCCL_SUCCESS);

  vector<u64> recvCounts(rankSize, COUNT_PER_RANK);
  vector<u64> rdispls(rankSize, 0);
  for (int i = 0; i < rankSize; i++) {
    rdispls[i] = COUNT_PER_RANK * i;
    HCCL_INFO("num[%d] displs[%d]", i, COUNT_PER_RANK * i);
  }
  DeviceMem devRecvCounts = DeviceMem::alloc(rankSize * sizeof(u64));
  ret = hrtMemSyncCopy(devRecvCounts.ptr(), rankSize * sizeof(u64), recvCounts.data(), rankSize * sizeof(u64),
                       HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE);
  EXPECT_EQ(ret, HCCL_SUCCESS);
  DeviceMem devRdispls = DeviceMem::alloc(rankSize * sizeof(u64));
  ret = hrtMemSyncCopy(devRdispls.ptr(), rankSize * sizeof(u64), rdispls.data(), rankSize * sizeof(u64),
                       HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE);
  EXPECT_EQ(ret, HCCL_SUCCESS);

  EXPECT_EQ(ret, HCCL_SUCCESS);
  DeviceMem recvMem = DeviceMem::alloc(memSize);
  DeviceMem gatherMem = DeviceMem::alloc(memSize);
  HCCL_ERROR("memsize[%llu]", memSize);

  HcomGatherAllToAllVParams opInfo;
  opInfo.addrInfo = devAddrInfo.ptr();
  opInfo.addrInfoCountPerRank = devCountPerRank.ptr();
  opInfo.recvbuf = recvMem.ptr();
  opInfo.recvcounts = devRecvCounts.ptr();
  opInfo.rdispls = devRdispls.ptr();
  opInfo.recvtype = HCCL_DATA_TYPE_INT32;
  opInfo.gatheredbuf = gatherMem.ptr();
  constexpr char HCCL_WORLD_GROUP[] = "hccl_world_group";
  opInfo.group = HCCL_WORLD_GROUP;
  opInfo.addrLength = -1;

  setExecutorStatus(HCCL_E_RESERVED);

  ret = HcomExecEnqueueGatherAllToAllV(opInfo, setExecutorStatus);
  EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);

  HcclResult exeRet = HCCL_E_RESERVED;
  const std::chrono::seconds TIMEOUT(1);
  const auto start = std::chrono::steady_clock::now();
  while (exeRet != HCCL_SUCCESS) {
    getExecutorStatus(exeRet);
    std::this_thread::sleep_for(std::chrono::milliseconds(LLT_SOCKET_SLEEP_MILLISECONDS));
    const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start);
    if (elapsed > TIMEOUT) {
      HCCL_ERROR("Wait timeout for getExecutor status timeout[%lld]", TIMEOUT);
      HcomExecFinalize();
      HcomDestroy();
      return;
    }
  }

  HCCL_INFO("HcomExecFinalize start");
  ret = HcomExecFinalize();
  EXPECT_EQ(ret, HCCL_SUCCESS);

  ret = HcomDestroy();
  EXPECT_EQ(ret, HCCL_SUCCESS);
  setExecutorStatus(HCCL_E_RESERVED);
  remove(file_name_t);
  (void)aclrtResetDevice(0);
}

#if 1
TEST_F(HcomExecutorTest, ut_executor_reduce) {
  nlohmann::json rank_table = rank_table_910_1server_1rank;

  char file_name_t[] = "./ut_executor_reduce.json";
  std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

  if (outfile.is_open()) {
    outfile << std::setw(1) << rank_table << std::endl;
    HCCL_INFO("open %s success", file_name_t);
  } else {
    HCCL_ERROR("open %s failed", file_name_t);
  }

  outfile.close();

  int ret = HCCL_SUCCESS;
  rtError_t rt_ret = RT_ERROR_NONE;
  rtStream_t stream;
  s8 *sendbuf;
  s8 *recvbuf;
  s32 rank = 0;
  s32 errors = 0;
  s32 count = HCCL_COM_DATA_SIZE;
  ret = hrtSetDevice(0);
  EXPECT_EQ(ret, HCCL_SUCCESS);

  ret = HcomExecInitialize();
  EXPECT_EQ(ret, HCCL_SUCCESS);

  HcclComm hcclComm;
  ret = HcomGetCommHandleByGroup(NULL, &hcclComm);
  EXPECT_EQ(ret, HCCL_SUCCESS);
  ret = hccl::HcomExecutor::GetInstance().InitHcclComm(nullptr, hcclComm);
  EXPECT_EQ(ret, HCCL_SUCCESS);

  rt_ret = rtStreamCreate(&stream, 5);
  EXPECT_EQ(rt_ret, RT_ERROR_NONE);
  sendbuf = (s8 *)sal_malloc(count * sizeof(s8));
  sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));
  recvbuf = (s8 *)sal_malloc(count * sizeof(s8));
  sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));

  for (int j = 0; j < count; j++) {
    sendbuf[j] = 2;
  }

  ret = HcomReduce("testreduce", sendbuf, recvbuf, count, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, 0, NULL, stream);
  EXPECT_EQ(ret, HCCL_SUCCESS);

  rt_ret = rtStreamSynchronize(stream);
  EXPECT_EQ(rt_ret, RT_ERROR_NONE);

  for (int j = 0; j < count; j++) {
    if (recvbuf[j] != 2) {
      HCCL_ERROR("\n rank:%d recvbuf[%d]:%f", rank, j, recvbuf[j]);
      errors++;
      break;
    }
  }

  for (int j = 0; j < count; j++) {
    sendbuf[j] = 2;
  }

  (void)HcomSetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
  ret = HcclReduce(sendbuf, recvbuf, count, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, 0, hcclComm, stream);
  EXPECT_EQ(ret, HCCL_SUCCESS);

  rt_ret = rtStreamSynchronize(stream);
  EXPECT_EQ(rt_ret, RT_ERROR_NONE);

  for (int j = 0; j < count; j++) {
    if (recvbuf[j] != 2) {
      HCCL_ERROR("\n rank:%d recvbuf[%d]:%f", rank, j, recvbuf[j]);
      errors++;
      break;
    }
  }

  ret = HcomExecFinalize();
  EXPECT_EQ(ret, HCCL_SUCCESS);
}
#endif

TEST_F(HcomExecutorTest, ut_executor_runGather) {
  MOCKER_CPP(&HcomExecutor::GatherMemCopyThread).stubs().with().will(returnValue(HCCL_SUCCESS));

  MOCKER(hrtMemSyncCopy).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

  std::vector<u64> addrInfo = {10, 20, 30, 40, 50, 60};
  std::vector<u64> addrInfoCountPerRank = {1, 2};
  u32 rankSize = 2;
  u64 sendCounts[rankSize];
  u64 sdispls[rankSize];
  void *sendDevBuf = nullptr;
  s32 addrLength = -1;
  HcclResult result = hccl::HcomExecutor::GetInstance().RunGather(addrInfo, addrInfoCountPerRank, rankSize, sendCounts,
                                                                  sdispls, sendDevBuf, addrLength);
  EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(HcomExecutorTest, Ut_HcomExec) {
  struct HcomRemoteOperation opInfo;
  auto callback = [](HcclResult status) {};
  HcclResult result = HcomExecEnqueueRemoteOperation(opInfo, callback);
  EXPECT_EQ(result, HCCL_E_NOT_SUPPORT);

  std::string remoteAccessType = "testType";
  std::vector<HcomRemoteAccessAddrInfo> addrInfos;
  result = HcomExecEnqueueRemoteAccess(remoteAccessType, addrInfos, callback);
  EXPECT_EQ(result, HCCL_E_NOT_SUPPORT);

  HcomOperation_t opInfo1;
  opInfo1.group = "testGroup";
  opInfo1.inputPtr = nullptr;
  opInfo1.count = 10;
  opInfo1.dataType = HCCL_DATA_TYPE_INT32;
  opInfo1.root = 0;
  MOCKER_CPP(HcceBroadcast).stubs().will(returnValue(HCCL_SUCCESS));
  result = hccl::HcomExecutor::GetInstance().ExecuteBroadcast(opInfo1);
  EXPECT_EQ(result, HCCL_SUCCESS);

  MOCKER_CPP(HcceAllReduce).stubs().will(returnValue(HCCL_SUCCESS));
  result = hccl::HcomExecutor::GetInstance().ExecuteAllreduce(opInfo1);
  EXPECT_EQ(result, HCCL_SUCCESS);

  MOCKER_CPP(HcceAllGather).stubs().will(returnValue(HCCL_SUCCESS));
  result = hccl::HcomExecutor::GetInstance().ExecuteAllGather(opInfo1);
  EXPECT_EQ(result, HCCL_SUCCESS);

  MOCKER_CPP(HcceReduceScatter).stubs().will(returnValue(HCCL_SUCCESS));
  result = hccl::HcomExecutor::GetInstance().ExecuteReduceScatter(opInfo1);
  EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(HcomExecutorTest, Ut_ExecuteAlltoAll) {
  void *baseAddr = malloc(100);
  u64 offset = 0;
  std::vector<u64> addrInfo = {10, 20, 30, 40};
  u64 beginIndex = 0;
  u64 count = 2;
  u64 tmpMemSize = 0;

  hccl::HcomExecutor::GetInstance().GatherMemCopyThread(baseAddr, offset, addrInfo, beginIndex, count, tmpMemSize);
  free(baseAddr);

  HcomAllToAllVParams opInfo;
  opInfo.group = nullptr;
  opInfo.sendbuf = (void *)0x1000;
  opInfo.sendcounts = (u64 *)0x2000;
  opInfo.sdispls = (u64 *)0x3000;
  opInfo.sendtype = HCCL_DATA_TYPE_INT8;
  opInfo.recvbuf = (void *)0x4000;
  opInfo.recvcounts = (u64 *)0x5000;
  opInfo.rdispls = (u64 *)0x6000;
  opInfo.recvtype = HCCL_DATA_TYPE_INT8;

  MOCKER_CPP(HcceAlltoAll).stubs().will(returnValue(HCCL_SUCCESS));

  MOCKER_CPP(HcceAlltoAllV).stubs().will(returnValue(HCCL_SUCCESS));

  HcclResult result = hccl::HcomExecutor::GetInstance().ExecuteAlltoAll(opInfo, false);
  EXPECT_EQ(result, HCCL_SUCCESS);
  result = hccl::HcomExecutor::GetInstance().ExecuteAlltoAll(opInfo, true);
  EXPECT_EQ(result, HCCL_SUCCESS);

  HcomAllToAllVCParams opInfo1;
  opInfo1.group = nullptr;
  opInfo1.sendbuf = (void *)0x1000;
  opInfo1.sendtype = HCCL_DATA_TYPE_INT8;
  opInfo1.recvbuf = (void *)0x2000;
  opInfo1.recvtype = HCCL_DATA_TYPE_INT8;
  opInfo1.sendcountmatrix = (void *)0x3000;

  MOCKER_CPP(HcceAlltoAllVC).stubs().will(returnValue(HCCL_SUCCESS));

  result = hccl::HcomExecutor::GetInstance().ExecuteAlltoAllVC(opInfo1);
  EXPECT_EQ(result, HCCL_SUCCESS);
}

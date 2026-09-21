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
#include <memory>
#include <fstream>
#include <string>
#include "ge_executor.h"
#include "exe_model_builder.h"
#include "types.h"
#include "model_desc.h"
#include "model_loader.h"
#include "maintain_manager.h"
#include "ge/ge_error_codes.h"
#include "gmock_stub/dbg_stub.h"
#include "gmock_stub/ascend_rt_stub.h"
#include "vector.h"
#include "dump_thread_manager.h"
#include "profiling.h"
#include "toolchain/prof_api.h"
#include <securec.h>
using namespace testing;
using ::testing::Return;
using ::testing::Eq;
using ::testing::NotNull;
using ::testing::Invoke;
using ::testing::_;
using namespace ge;

class DbgLiteOSTest : public testing::Test {
 protected:
  void SetUp() {
    ON_CALL(MmpaStubMock::GetInstance(), mmMalloc(_)).WillByDefault(Invoke(mmMalloc_Normal_Invoke));
    ON_CALL(MmpaStubMock::GetInstance(), mmTellFile(_)).WillByDefault(Invoke(mmTellFile_Normal_Invoke));
    ON_CALL(MmpaStubMock::GetInstance(), mmReadFile(_,_,_,_)).WillByDefault(Invoke(mmReadFile_Normal_Invoke));
  }
  void TearDown() {
    Mock::VerifyAndClearExpectations(&MmpaStubMock::GetInstance());
    Mock::VerifyAndClearExpectations(&DbgStubMock::GetInstance());
  }
};
#ifdef __cplusplus
extern "C" {
#endif
bool GetProfIsConfig();
#ifdef __cplusplus
}
#endif
TEST_F(DbgLiteOSTest, DumpThreadNormal) {
  EXPECT_CALL(DbgStubMock::GetInstance(), rtDumpInit()).Times(1).WillOnce(Return(SUCCESS));
  Status ret = InitDumpThread();
  ASSERT_EQ(ret, SUCCESS);
  DeInitDumpThread();
  StopDumpThread();
}

TEST_F(DbgLiteOSTest, DumpThreadUnnormal) {
  EXPECT_CALL(DbgStubMock::GetInstance(), rtDumpInit()).Times(1).WillOnce(Return(-1));
  Status ret = InitDumpThread();
  ASSERT_NE(ret, SUCCESS);
  DeInitDumpThread();
  StopDumpThread();
}

TEST_F(DbgLiteOSTest, profiling_static_callback) {
  MsprofCommandHandle data;
  (void)memset_s(&data, sizeof(MsprofCommandHandle), 0, sizeof(MsprofCommandHandle));
  MsprofCtrlHandleFunc(0, &data, sizeof(MsprofCommandHandle));

  int32_t ret = MsprofCtrlHandleFunc(1, &data, sizeof(MsprofCommandHandle) - 1);
  EXPECT_NE(ret, SUCCESS);

  ret = MsprofCtrlHandleFunc(1, NULL, sizeof(MsprofCommandHandle));
  EXPECT_NE(ret, SUCCESS);

  data.profSwitch = MSPROF_TASK_TIME_L0;
  MsprofCtrlHandleFunc(1, &data, sizeof(MsprofCommandHandle));
  bool flag = DbgGetprofEnable();
  ASSERT_EQ(flag, true);

  data.profSwitch = 0x00000000ULL;
  MsprofCtrlHandleFunc(1, &data, sizeof(MsprofCommandHandle));
  flag = DbgGetprofEnable();
  ASSERT_EQ(flag, false);
}

TEST_F(DbgLiteOSTest, profiling_static_unnormal) {
  const char *filePath = "../tests/test_c/ut/testcase/executor/data/acl_profiling1.json";
  Status ret = DbgProfInit(filePath);
  ASSERT_EQ(ret, SUCCESS);

  bool flag = DbgGetprofEnable();
  ASSERT_EQ(flag, false);

  bool cfgFlag = GetProfIsConfig();
  ASSERT_EQ(cfgFlag, false);

  uint32_t modelId = 0;
  char *om = (char *)"concat_nano";
  ret = DbgProfReportDataProcess(modelId, om);
  ASSERT_EQ(ret, SUCCESS);

  ret = DbgNotifySetDevice(0, 0);
  ASSERT_EQ(ret, SUCCESS);

  ret = DbgProfDeInit();
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(DbgLiteOSTest, profiling_static_normal) {
  char buf[80];
  getcwd(buf,sizeof(buf));
  printf("current working directory: %s\n", buf);
  fflush(stdout);

  std::string str = buf;

	std::cout << str << std::endl;
  std::cerr << str << std::endl;

  EXPECT_CALL(DbgStubMock::GetInstance(), MsprofRegisterCallback(_,_)).Times(1).WillOnce(Return(SUCCESS));
  EXPECT_CALL(DbgStubMock::GetInstance(), MsprofInit(_,_,_)).Times(1).WillOnce(Return(SUCCESS));
  EXPECT_CALL(DbgStubMock::GetInstance(), MsprofReportAdditionalInfo(_,_,_)).Times(1).WillOnce(Return(SUCCESS));
  EXPECT_CALL(DbgStubMock::GetInstance(), MsprofFinalize()).Times(1).WillOnce(Return(SUCCESS));
  EXPECT_CALL(DbgStubMock::GetInstance(), MsprofNotifySetDevice(_,_,_)).Times(1).WillOnce(Return(SUCCESS));
  const char *filePath = "../tests/test_c/ut/testcase/executor/data/acl_profiling.json";
  Status ret = DbgProfInit(filePath);
  ASSERT_EQ(ret, SUCCESS);

  SetMsprofEnable(true);
  bool profEnable = DbgGetprofEnable();
  ASSERT_EQ(profEnable, true);

  bool cfgFlag = GetProfIsConfig();
  ASSERT_EQ(cfgFlag, true);

  uint32_t modelId = 0;
  char *om = (char *)"concat_nano";
  ret = DbgProfReportDataProcess(modelId, om);
  ASSERT_EQ(ret, SUCCESS);

  ret = DbgNotifySetDevice(0, 0);
  ASSERT_EQ(ret, SUCCESS);

  ret = DbgProfDeInit();
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(DbgLiteOSTest, profiling_static_callback_unnormal) {
  EXPECT_CALL(DbgStubMock::GetInstance(), MsprofRegisterCallback(_,_)).Times(1).WillOnce(Return(-1));
  EXPECT_CALL(DbgStubMock::GetInstance(), MsprofReportAdditionalInfo(_,_,_)).Times(1).WillOnce(Return(SUCCESS));
  EXPECT_CALL(DbgStubMock::GetInstance(), MsprofFinalize()).Times(1).WillOnce(Return(SUCCESS));
  EXPECT_CALL(DbgStubMock::GetInstance(), MsprofNotifySetDevice(_,_,_)).Times(1).WillOnce(Return(SUCCESS));
  const char *filePath = "../tests/test_c/ut/testcase/executor/data/acl_profiling.json";
  Status ret = DbgProfInit(filePath);
  ASSERT_NE(ret, SUCCESS);

  SetMsprofEnable(true);
  bool profEnable = DbgGetprofEnable();
  ASSERT_EQ(profEnable, true);

  bool cfgFlag = GetProfIsConfig();
  ASSERT_EQ(cfgFlag, true);

  uint32_t modelId = 0;
  char *om = (char *)"concat_nano";
  ret = DbgProfReportDataProcess(modelId, om);
  ASSERT_EQ(ret, SUCCESS);

  ret = DbgNotifySetDevice(0, 0);
  ASSERT_EQ(ret, SUCCESS);

  ret = DbgProfDeInit();
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(DbgLiteOSTest, profiling_static_init_unnormal) {
  EXPECT_CALL(DbgStubMock::GetInstance(), MsprofRegisterCallback(_,_)).Times(1).WillOnce(Return(SUCCESS));
  EXPECT_CALL(DbgStubMock::GetInstance(), MsprofInit(_,_,_)).Times(1).WillOnce(Return(-1));
  EXPECT_CALL(DbgStubMock::GetInstance(), MsprofReportAdditionalInfo(_,_,_)).Times(1).WillOnce(Return(SUCCESS));
  EXPECT_CALL(DbgStubMock::GetInstance(), MsprofFinalize()).Times(1).WillOnce(Return(SUCCESS));
  EXPECT_CALL(DbgStubMock::GetInstance(), MsprofNotifySetDevice(_,_,_)).Times(1).WillOnce(Return(SUCCESS));
  const char *filePath = "../tests/test_c/ut/testcase/executor/data/acl_profiling.json";
  Status ret = DbgProfInit(filePath);
  ASSERT_NE(ret, SUCCESS);

  SetMsprofEnable(true);
  bool profEnable = DbgGetprofEnable();
  ASSERT_EQ(profEnable, true);

  bool cfgFlag = GetProfIsConfig();
  ASSERT_EQ(cfgFlag, true);

  uint32_t modelId = 0;
  char *om = (char *)"concat_nano";
  ret = DbgProfReportDataProcess(modelId, om);
  ASSERT_EQ(ret, SUCCESS);

  ret = DbgNotifySetDevice(0, 0);
  ASSERT_EQ(ret, SUCCESS);

  ret = DbgProfDeInit();
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(DbgLiteOSTest, profiling_static_report_unnormal) {
  EXPECT_CALL(DbgStubMock::GetInstance(), MsprofRegisterCallback(_,_)).Times(1).WillOnce(Return(SUCCESS));
  EXPECT_CALL(DbgStubMock::GetInstance(), MsprofInit(_,_,_)).Times(1).WillOnce(Return(SUCCESS));
  EXPECT_CALL(DbgStubMock::GetInstance(), MsprofReportAdditionalInfo(_,_,_)).Times(1).WillOnce(Return(-1));
  EXPECT_CALL(DbgStubMock::GetInstance(), MsprofFinalize()).Times(1).WillOnce(Return(SUCCESS));
  EXPECT_CALL(DbgStubMock::GetInstance(), MsprofNotifySetDevice(_,_,_)).Times(1).WillOnce(Return(SUCCESS));
  const char *filePath = "../tests/test_c/ut/testcase/executor/data/acl_profiling.json";
  Status ret = DbgProfInit(filePath);
  ASSERT_EQ(ret, SUCCESS);

  SetMsprofEnable(true);
  bool profEnable = DbgGetprofEnable();
  ASSERT_EQ(profEnable, true);

  bool cfgFlag = GetProfIsConfig();
  ASSERT_EQ(cfgFlag, true);

  uint32_t modelId = 0;
  char *om = (char *)"concat_nano";
  ret = DbgProfReportDataProcess(modelId, om);
  ASSERT_NE(ret, SUCCESS);

  ret = DbgNotifySetDevice(0, 0);
  ASSERT_EQ(ret, SUCCESS);

  ret = DbgProfDeInit();
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(DbgLiteOSTest, profiling_static_setdevice_unnormal) {
  EXPECT_CALL(DbgStubMock::GetInstance(), MsprofRegisterCallback(_,_)).Times(1).WillOnce(Return(SUCCESS));
  EXPECT_CALL(DbgStubMock::GetInstance(), MsprofInit(_,_,_)).Times(1).WillOnce(Return(SUCCESS));
  EXPECT_CALL(DbgStubMock::GetInstance(), MsprofReportAdditionalInfo(_,_,_)).Times(1).WillOnce(Return(SUCCESS));
  EXPECT_CALL(DbgStubMock::GetInstance(), MsprofFinalize()).Times(1).WillOnce(Return(SUCCESS));
  EXPECT_CALL(DbgStubMock::GetInstance(), MsprofNotifySetDevice(_,_,_)).Times(1).WillOnce(Return(-1));
  const char *filePath = "../tests/test_c/ut/testcase/executor/data/acl_profiling.json";
  Status ret = DbgProfInit(filePath);
  ASSERT_EQ(ret, SUCCESS);

  SetMsprofEnable(true);
  bool profEnable = DbgGetprofEnable();
  ASSERT_EQ(profEnable, true);

  bool cfgFlag = GetProfIsConfig();
  ASSERT_EQ(cfgFlag, true);

  uint32_t modelId = 0;
  char *om = (char *)"concat_nano";
  ret = DbgProfReportDataProcess(modelId, om);
  ASSERT_EQ(ret, SUCCESS);

  ret = DbgNotifySetDevice(0, 0);
  ASSERT_NE(ret, SUCCESS);

  ret = DbgProfDeInit();
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(DbgLiteOSTest, profiling_static_finalize_unnormal) {
  EXPECT_CALL(DbgStubMock::GetInstance(), MsprofRegisterCallback(_,_)).Times(1).WillOnce(Return(SUCCESS));
  EXPECT_CALL(DbgStubMock::GetInstance(), MsprofInit(_,_,_)).Times(1).WillOnce(Return(SUCCESS));
  EXPECT_CALL(DbgStubMock::GetInstance(), MsprofReportAdditionalInfo(_,_,_)).Times(1).WillOnce(Return(SUCCESS));
  EXPECT_CALL(DbgStubMock::GetInstance(), MsprofFinalize()).Times(1).WillOnce(Return(-1));
  EXPECT_CALL(DbgStubMock::GetInstance(), MsprofNotifySetDevice(_,_,_)).Times(1).WillOnce(Return(SUCCESS));
  const char *filePath = "../tests/test_c/ut/testcase/executor/data/acl_profiling.json";
  Status ret = DbgProfInit(filePath);
  ASSERT_EQ(ret, SUCCESS);

  SetMsprofEnable(true);
  bool profEnable = DbgGetprofEnable();
  ASSERT_EQ(profEnable, true);

  bool cfgFlag = GetProfIsConfig();
  ASSERT_EQ(cfgFlag, true);

  uint32_t modelId = 0;
  char *om = (char *)"concat_nano";
  ret = DbgProfReportDataProcess(modelId, om);
  ASSERT_EQ(ret, SUCCESS);

  ret = DbgNotifySetDevice(0, 0);
  ASSERT_EQ(ret, SUCCESS);

  ret = DbgProfDeInit();
  ASSERT_NE(ret, SUCCESS);
}
